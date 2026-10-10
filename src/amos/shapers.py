r"""Techniques that reshape data, usually before it is studied or analyzed.

These are techniques of the "wrangler" stage of a project. Cleaners (see the
`cleaners` module) remove rows and columns, and mungers (see the `mungers`
module) change what is in the columns. Shapers change what a row is: they
rearrange the same data so that each row stands for something else. A table
with a row for each case becomes one with a row for each judge on each case
(`lists_to_rows`), and back again (`group_rows`). A table with a column for
each year becomes one with a row for each year (`wide_to_long`), and back
again (`long_to_wide`).

The rows of reshaped data are not the rows that it began with, so:

* The rows get new index labels (0, 1, 2, and so on). When rows are repeated
  (by `lists_to_rows` and `wide_to_long`), an index with a name is kept as a
  column of that name, so each new row still says which row it came from.
* A shaper must come before the data is split into training and test rows.
* The label and groups of the dataset must still be columns afterward, or
  the shaper's "label" and "groups" parameters must name those of the
  reshaped data.

Like cleaners and mungers, shapers learn nothing from the data that could
leak from the test rows into the training rows.

Contents:
    Shaper: base class for techniques that reshape data.
    GroupRows: makes one row for each group of rows, with summaries of its
        columns.
    ListsToRows: makes a row for each item of a list.
    LongToWide: spreads the rows of each group into columns.
    WideToLong: stacks columns into rows.

"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from . import base, utilities

# The summaries that `group_rows` (and `merge_summary`) can take of the rows
# of a group, with what `pandas` calls each one.
_SUMMARIES: dict[str, Any] = {
    'all': 'all',
    'any': 'any',
    'count': 'count',
    'first': 'first',
    'last': 'last',
    'list': list,
    'max': 'max',
    'mean': 'mean',
    'median': 'median',
    'min': 'min',
    'nunique': 'nunique',
    'std': 'std',
    'sum': 'sum'}


@dataclasses.dataclass
class Shaper(base.Operation, abc.ABC):
    """Base class for techniques that reshape data.

    A subclass writes a `shape` method, which takes the data and returns it in
    its new shape. The dataset's `history` records how many rows and columns
    there were before and after.

    Every shaper takes "label", "task", and "groups" parameters, which name
    the label and groups of the reshaped data. Without them, the dataset
    keeps the label and groups that it has.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: not used by most shapers. Defaults to `None`.
        parameters: keyword arguments for `shape`, and "label", "task", and
            "groups". Defaults to an empty `dict`.

    """

    """ Required Methods """

    @abc.abstractmethod
    def shape(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Returns `data` in its new shape.

        Args:
            data: the data to reshape. It may be changed in place.
            **kwargs: parameters for reshaping.

        Returns:
            The reshaped data.

        """

    """ Public Methods """

    def implement(
        self,
        item: base.Dataset,
        *,
        label: str | None = None,
        task: str | None = None,
        groups: Sequence[str] | str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Reshapes the data of `item`.

        Args:
            item: the dataset to reshape.
            label: name of the label of the reshaped data. Defaults to `None`,
                which keeps the label of `item`.
            task: "classify" or "regress". Defaults to `None`, which keeps
                the task of `item`, unless `label` names a new label, whose
                task is inferred (see `Dataset.infer_task`).
            groups: names of the groups of the reshaped data. Defaults to
                `None`, which keeps the groups of `item`.
            **kwargs: parameters for `shape`.

        Raises:
            KeyError: if the label or a group is not a column of the reshaped
                data.
            ValueError: if the data has been split, or `task` is not a task.

        Returns:
            The reshaped dataset.

        """
        if item.is_split:
            message = (
                f'{self.name!r} changes the rows of the data, so it must come '
                f'before the data is split'
            )
            raise ValueError(message)
        if task is not None and task not in base._TASKS:
            message = f'task must be one of {base._TASKS}, not {task!r}'
            raise ValueError(message)
        label = label or item.label
        named = item.groups if groups is None else base._listify(groups)
        before = item.data.shape
        shaped = self.shape(item.data.copy(), **kwargs)
        lost = [
            c for c in [label, *named]
            if c is not None and c not in shaped.columns]
        if lost:
            message = (
                f'the reshaped data has no columns {lost}, which are the '
                f'label or groups: set "label" or "groups" in the parameters '
                f'of {self.name!r} to name those of the reshaped data'
            )
            raise KeyError(message)
        named = base._validate_groups(named, shaped, label)
        relabeled = label != item.label
        item.label, item.groups = label, named
        item.replace(shaped)
        if task is not None:
            item.task = task
        elif relabeled:
            item.task = item.infer_task()
        item.record(
            self.name,
            rows = [before[0], shaped.shape[0]],
            columns = [before[1], shaped.shape[1]])
        return item


@dataclasses.dataclass
class GroupRows(Shaper):
    """Makes one row for each group of rows, with summaries of its columns.

    For example, a table with a row for each judge on each case becomes a
    table with a row for each case, with the share of its judges who are
    women and their mean age. It is the opposite of `lists_to_rows`.

    Unless "how" says otherwise, a column that has the same value in every
    row of each group keeps that value (so a case's court stays its court).
    Of the columns that vary within a group, the mean is taken of numbers and
    booleans (the mean of a boolean is the share that are true), and the
    first value is taken of anything else.

    Wraps:
        [`pandas.DataFrame.groupby`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html)
        from pandas.

    """

    def shape(
        self,
        data: pd.DataFrame,
        *,
        by: Sequence[str] | str | None = None,
        how: Mapping[str, Any] | Sequence[str] | str | None = None,
        count: str | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Makes one row for each group of the rows with the same `by`.

        Args:
            data: the data to reshape.
            by: the columns that say which group each row belongs to (such as
                "case_id"). Defaults to `None`, which changes nothing.
            how: the summary to take of the other columns: "mean", "median",
                "sum", "min", "max", "std", "count" (the number of values
                that are not missing), "nunique" (the number of different
                values), "first", "last", "any", "all", or "list" (a list of
                the values). A list of summaries takes each of them. A `dict`
                maps columns to their own summaries, and the columns that it
                does not name are summarized as usual. Defaults to `None`,
                which summarizes every column as usual (see the class).
            count: name of a column to make with the number of rows in each
                group. Defaults to `None`, which makes no such column.
            **kwargs: not used.

        Raises:
            KeyError: if a column of `by` or `how` is not in the data.
            TypeError: if a summary cannot be taken of a column (such as the
                mean of text).
            ValueError: if a summary is not one of those above, or two new
                columns would have the same name.

        Returns:
            The data with one row for each group, in the order in which the
                groups first appear. The columns of `by` come first. A column
                with one summary keeps its name, and a column with several
                becomes a column for each, named "{column}_{summary}". Rows
                with a missing value in `by` are a group of their own.

        """
        keys = [] if by is None else utilities.select_columns(data, by)
        if not keys:
            return data
        grouped = data.groupby(keys, sort = False, dropna = False)
        # The column and summary that make each new column, by its name.
        made: dict[str, tuple[str, Any]] = {}
        for column, summaries in _plan(data, keys, how).items():
            for summary in summaries:
                name = (
                    column if len(summaries) == 1 else f'{column}_{summary}')
                _claim(name, [*keys, *made], self.name)
                made[name] = (column, summary)
        if count is not None:
            _claim(str(count), [*keys, *made], self.name)
            made[str(count)] = (keys[0], 'size')
        if not made:
            return data[keys].drop_duplicates().reset_index(drop = True)
        try:
            table = grouped.agg(**{
                name: (column, _SUMMARIES.get(summary, summary))
                for name, (column, summary) in made.items()})
        except TypeError as error:
            # Finds the column that `pandas` could not summarize, which its
            # message does not name.
            for column, summary in made.values():
                try:
                    grouped[column].agg(_SUMMARIES.get(summary, summary))
                except TypeError:
                    message = (
                        f'{self.name!r} cannot take the {summary} of '
                        f'{column!r} ({data[column].dtype}): choose another '
                        f'summary for it (such as "first" or "list") in "how"'
                    )
                    raise TypeError(message) from error
            raise
        return table.reset_index()


@dataclasses.dataclass
class ListsToRows(Shaper):
    """Makes a row for each item of a list, such as each judge of a case.

    A column that lists several things in one cell (as a Python list, or as
    text such as "Lynch; Thompson; Barron") is split into a row for each of
    them. The other columns are repeated in each of the rows. It is the
    opposite of `group_rows`.

    Wraps:
        [`pandas.DataFrame.explode`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.explode.html)
        from pandas.

    """

    def shape(
        self,
        data: pd.DataFrame,
        *,
        column: str | None = None,
        separator: str | None = None,
        name: str | None = None,
        position: str | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Makes a row for each item in `column`.

        Args:
            data: the data to reshape.
            column: the column of lists. Defaults to `None`, which changes
                nothing.
            separator: the text between the items of a list that is written
                as text (such as ";"). The items are trimmed of spaces, and
                empty ones are left out. Defaults to `None`, which suits a
                column of Python lists.
            name: name of the column of items to make. Defaults to `None`,
                which replaces `column`.
            position: name of a column to make with the place of each item in
                its list (1 for the first). Defaults to `None`, which makes
                no such column.
            **kwargs: not used.

        Raises:
            KeyError: if `column` is not in the data.
            ValueError: if `column` is text and there is no `separator`, or a
                new column would have the name of another column.

        Returns:
            The data with a row for each item. A row with no items is kept
                as one row, with a missing item.

        """
        if column is None:
            return data
        utilities.select_columns(data, [column])
        data = _keep_index(data)
        target = column if name is None else name
        if target != column:
            _claim(target, list(data.columns), self.name)
        if position is not None:
            _claim(position, [*data.columns, target], self.name)
        items = _items(data[column], separator, self.name)
        rows = np.asarray(items.index)
        absent = np.asarray(items.isna())
        # A row with no items keeps one row (with a missing item), but an
        # empty item of a row that has others (as in "a;;b") is left out.
        filled = np.zeros(len(data), dtype = bool)
        filled[rows[~absent]] = True
        first = ~np.asarray(pd.Series(rows).duplicated())
        kept = ~absent | (~filled[rows] & first)
        rows, absent = rows[kept], absent[kept]
        shaped: pd.DataFrame = data.iloc[rows].reset_index(drop = True)
        shaped[target] = items[kept].reset_index(drop = True).infer_objects()
        if position is not None:
            places = pd.Series(rows).groupby(rows).cumcount() + 1
            shaped[position] = places.astype('Int64').mask(absent)
        return shaped


@dataclasses.dataclass
class LongToWide(Shaper):
    """Spreads the rows of each group into columns, such as one for each year.

    For example, a table with a row for each state in each year (with
    "state", "year", and "income" columns) becomes a table with a row for
    each state and a column of income for each year ("income_2019",
    "income_2020", and so on). It is the opposite of `wide_to_long`.

    Wraps:
        [`pandas.DataFrame.unstack`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.unstack.html)
        from pandas.

    """

    def shape(
        self,
        data: pd.DataFrame,
        *,
        names: str | None = None,
        values: str | None = None,
        stubs: Sequence[str] | str | None = None,
        ids: Sequence[str] | str | None = None,
        separator: str = '_',
        **kwargs: Any) -> pd.DataFrame:
        """Makes a column for each value of `names`.

        Use `values` for one column of values, whose new columns are named
        for the values of `names` alone, or `stubs` for one or more, whose
        new columns begin with their own names.

        Args:
            data: the data to reshape.
            names: the column whose values name the new columns (such as
                "year"). Defaults to `None`, which changes nothing.
            values: the column whose values fill the new columns, which are
                named for the values of `names` (such as "2019" and "2020").
                Defaults to `None`.
            stubs: the columns whose values fill the new columns, which are
                named "{stub}{separator}{name}" (such as "income_2019").
                Defaults to `None`.
            ids: the columns that say which new row each row belongs to
                (such as "state"). Defaults to `None`, in which case they
                are every column other than `names` and those of `values`
                or `stubs`. Columns that are none of these are left out.
            separator: the text between a stub and a name. Defaults to "_".
            **kwargs: not used.

        Raises:
            KeyError: if a column is not in the data.
            ValueError: if neither or both of `values` and `stubs` are
                given, a row has no value in `names`, two rows would fill
                the same cell, or a new column would have the name of
                another column.

        Returns:
            The data with one row for each group of `ids`, in the order in
                which the groups first appear, and the new columns in the
                order of their names. A cell that no row fills is missing.

        """
        if names is None:
            return data
        if (values is None) == (stubs is None):
            message = (
                f'{self.name!r} needs the columns that fill the new columns: '
                f'set either "values" (for one column) or "stubs"'
            )
            raise ValueError(message)
        spread = utilities.select_columns(
            data, base._listify(values if stubs is None else stubs))
        utilities.select_columns(data, [names])
        used = [names, *spread]
        keys = (
            [c for c in data.columns if c not in used] if ids is None
            else utilities.select_columns(data, ids))
        named: pd.Series = data[names]
        if isinstance(named.dtype, pd.CategoricalDtype):
            # Only the categories that the rows have become columns.
            named = named.astype(named.cat.categories.dtype)
        blank = int(named.isna().sum())
        if blank:
            message = (
                f'{blank} rows have no value in {names!r}, so there is no '
                f'column for them: remove them first (with drop_missing)'
            )
            raise ValueError(message)
        # The number of the new row that each row belongs to, in the order
        # in which the groups first appear.
        rows = (
            data.groupby(keys, sort = False, dropna = False).ngroup()
            if keys else pd.Series(0, index = data.index))
        cells = pd.MultiIndex.from_arrays([rows, named])
        if cells.has_duplicates:
            repeated = data.loc[cells.duplicated(), [*keys, names]].iloc[0]
            message = (
                f'{int(cells.duplicated().sum())} rows have the same ids and '
                f'{names!r} as an earlier row (such as '
                f'{repeated.to_dict()}), and two rows cannot fill one cell: '
                f'remove them (with drop_duplicates) or summarize them (with '
                f'group_rows) first'
            )
            raise ValueError(message)
        wide = data[spread].set_axis(cells).unstack()  # noqa: PD010
        # The column that fills each new column, and the name that it is for.
        made = cast('list[tuple[Any, Any]]', wide.columns.tolist())
        labels = [
            _written(label) if stubs is None
            else f'{stub}{separator}{_written(label)}'
            for stub, label in made]
        for position, label in enumerate(labels):
            _claim(label, [*keys, *labels[:position]], self.name)
        wide = wide.set_axis(labels, axis = 1).reset_index(drop = True)
        first = data.loc[~np.asarray(rows.duplicated()), keys]
        return pd.concat([first.reset_index(drop = True), wide], axis = 1)


@dataclasses.dataclass
class WideToLong(Shaper):
    """Stacks columns into rows, such as a row for each year.

    For example, a table with a row for each state and a column of income for
    each year ("income_2019", "income_2020", and so on) becomes a table with
    a row for each state in each year. The other columns are repeated in
    each of the rows. It is the opposite of `long_to_wide`.

    Wraps:
        [`pandas.concat`](https://pandas.pydata.org/docs/reference/api/pandas.concat.html)
        from pandas.

    """

    def shape(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | str | None = None,
        stubs: Sequence[str] | str | None = None,
        name: str = 'variable',
        value: str = 'value',
        separator: str = '_',
        **kwargs: Any) -> pd.DataFrame:
        """Makes a row for each of `columns`, or for each column of `stubs`.

        Use `columns` to stack columns into one column of values, or `stubs`
        to stack sets of columns that begin with the same names.

        Args:
            data: the data to reshape.
            columns: the columns to stack into the `value` column. The
                `name` column says which of them each value came from.
                Defaults to `None`.
            stubs: the beginnings of the names of the columns to stack. Each
                stub becomes a column of the values of the columns named
                "{stub}{separator}{name}", and the `name` column has the rest
                of their names. For example, the stubs "income" and "tax"
                stack "income_2019", "income_2020", "tax_2019", and
                "tax_2020" into "income" and "tax" columns, with 2019 and
                2020 in the `name` column. Defaults to `None`. Without
                `columns` or `stubs`, nothing changes.
            name: name of the column to make that says where each value came
                from. If every one is a number (such as a year), the column
                is numbers. Defaults to "variable".
            value: name of the column of values that `columns` makes. It is
                not used with `stubs`. Defaults to "value".
            separator: the text between a stub and the rest of a name.
                Defaults to "_".
            **kwargs: not used.

        Raises:
            KeyError: if one of `columns` is not in the data, or no column
                begins with a stub.
            ValueError: if both `columns` and `stubs` are given, or a new
                column would have the name of another column.

        Returns:
            The data with a row for each stacked column of each row, in the
                order of the rows and then of the columns. A value that a
                row lacks is missing.

        """
        if columns is None and stubs is None:
            return data
        if columns is not None and stubs is not None:
            message = (
                f'{self.name!r} stacks either "columns" or the columns of '
                f'"stubs", not both'
            )
            raise ValueError(message)
        data = _keep_index(data)
        # The columns to stack into each new column, by what `name` says of
        # them.
        sources: dict[str, dict[str, str]] = {}
        if columns is not None:
            stacked = utilities.select_columns(data, columns)
            sources[value] = {str(c): c for c in stacked}
        else:
            for stub in base._listify(stubs):
                start = f'{stub}{separator}'
                found = {
                    str(c)[len(start):]: c for c in data.columns
                    if str(c).startswith(start) and len(str(c)) > len(start)}
                if not found:
                    message = f'no column of the data begins with {start!r}'
                    raise KeyError(message)
                sources[stub] = found
        used = {c for found in sources.values() for c in found.values()}
        kept = [c for c in data.columns if c not in used]
        for position, column in enumerate([name, *sources]):
            _claim(column, [*kept, *[name, *sources][:position]], self.name)
        labels = list(dict.fromkeys(
            label for found in sources.values() for label in found))
        if not labels:
            return data
        blocks = []
        for label in labels:
            block = data[kept].copy()
            block[name] = label
            for column, found in sources.items():
                block[column] = (
                    data[found[label]] if label in found
                    # A set of columns that lacks this one is missing here.
                    else data[next(iter(found.values()))].iloc[:0].reindex(
                        data.index))
            blocks.append(block)
        long = pd.concat(blocks, ignore_index = True)
        order = np.argsort(
            np.tile(np.arange(len(data)), len(labels)), kind = 'stable')
        long = long.iloc[order].reset_index(drop = True)
        if stubs is not None:
            numbers = pd.to_numeric(long[name], errors = 'coerce')
            if not numbers.isna().any():
                long[name] = numbers
        return long


""" Private Functions """


def _claim(name: str, taken: Sequence[str], technique: str) -> None:
    """Checks that a new column can have a name.

    Args:
        name: the name of a column to make.
        taken: the names of the other columns of the reshaped data.
        technique: name of the technique that makes the column, for messages.

    Raises:
        ValueError: if `name` is one of `taken`.

    """
    if name in taken:
        message = (
            f'{technique!r} would make two columns named {name!r}: rename '
            f'one of them first (with rename_columns), or give the new '
            f'column another name'
        )
        raise ValueError(message)


def _items(values: pd.Series, separator: str | None, technique: str) -> (
    pd.Series):
    """Returns every item of the lists in a column.

    Args:
        values: a column of lists, or of text with a separator between the
            items.
        separator: the text between the items of text, or `None`.
        technique: name of the technique that uses the items, for messages.

    Raises:
        ValueError: if `values` is text and there is no `separator`.

    Returns:
        The items, labeled by the position of the row of each. A row with no
            items has one missing item. Items that were split from text are
            trimmed of spaces, and the empty ones are missing.

    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype(values.cat.categories.dtype)
    values = values.reset_index(drop = True)
    if pd.api.types.is_object_dtype(values.dtype):
        # A column of objects may have lists, text, or both.
        if separator is None:
            return values.explode()
        cut = str(separator)
        pieces = values.map(
            lambda cell: cell.split(cut) if isinstance(cell, str) else cell)
        items = pieces.explode().map(
            lambda item: item.strip() if isinstance(item, str) else item)
        return items.mask(items == '')
    if not pd.api.types.is_string_dtype(values.dtype):
        return values
    if separator is None:
        message = (
            f'{technique!r} needs a "separator" (such as ";"), the text '
            f'between the items of the lists in {values.name!r}'
        )
        raise ValueError(message)
    items = values.str.split(str(separator), regex = False).explode()
    items = items.astype('str').str.strip()
    return items.mask(items == '')


def _keep_index(data: pd.DataFrame) -> pd.DataFrame:
    """Returns `data` with an index that has a name as a column.

    Rows that are repeated cannot keep their index labels, which must differ.
    An index with a name (such as "case_id") says something about each row,
    so it becomes a column, which is repeated with the rows.

    Args:
        data: the data to reshape.

    Returns:
        The data with its index as columns, if every level of the index has
            a name that no column has, and otherwise as it is.

    """
    names = list(data.index.names)
    if any(n is None or n in data.columns for n in names):
        return data
    return data.reset_index()


def _plan(
    data: pd.DataFrame,
    keys: Sequence[str],
    how: Mapping[str, Any] | Sequence[str] | str | None) -> (
        dict[str, list[str]]):
    """Returns the summaries to take of each column of the groups of rows.

    Args:
        data: the data to group.
        keys: the columns that say which group each row belongs to.
        how: a summary or a list of them (for every column), a `dict` of the
            summaries of columns, or `None`.

    Raises:
        KeyError: if `how` names a column that is not in `data`.
        ValueError: if a summary is not in `_SUMMARIES`, or `how` names one
            of `keys`.

    Returns:
        The names of the summaries of each column that is not one of `keys`,
            in order. A column that `how` does not cover is summarized as
            usual (see `_usual`).

    """
    chosen: dict[str, list[str]] = {}
    everywhere: list[str] | None = None
    if isinstance(how, Mapping):
        utilities.select_columns(data, list(how))
        chosen = {c: _listed(summaries) for c, summaries in how.items()}
    elif how is not None:
        everywhere = _listed(how)
    grouped = [c for c in chosen if c in keys]
    if grouped:
        message = (
            f'the columns {grouped} say which group each row belongs to, so '
            f'they cannot be summarized'
        )
        raise ValueError(message)
    plan = {}
    for column in data.columns:
        if column in keys:
            continue
        plan[column] = chosen.get(column) or everywhere or [
            _usual(data, keys, column)]
        unknown = [s for s in plan[column] if s not in _SUMMARIES]
        if unknown:
            message = (
                f'a summary must be one of {sorted(_SUMMARIES)}, not '
                f'{unknown[0]!r}'
            )
            raise ValueError(message)
    return plan


def _listed(item: Sequence[str] | str) -> list[str]:
    """Returns a summary, or a sequence of them, as a `list`.

    Args:
        item: the name of a summary or a sequence of names.

    Returns:
        The names.

    """
    return [item] if isinstance(item, str) else [str(i) for i in item]


def _usual(data: pd.DataFrame, keys: Sequence[str], column: str) -> str:
    """Returns the summary of a column that no summary was chosen for.

    Args:
        data: the data to group.
        keys: the columns that say which group each row belongs to.
        column: name of the column to summarize.

    Returns:
        "first" if the column has one value in every group (which is then
            kept as it is) or is not numbers or booleans, and otherwise
            "mean".

    """
    values = data[column]
    if not pd.api.types.is_numeric_dtype(values.dtype):
        return 'first'
    varied = data.groupby(list(keys), sort = False, dropna = False)[
        column].nunique(dropna = False)
    return 'mean' if bool((varied > 1).any()) else 'first'


def _written(label: Any) -> str:
    """Returns a value as it is written in the name of a column.

    Args:
        label: a value of the column that names new columns.

    Returns:
        The value as text. A whole number that is stored as a float (such as
            2019.0) is written without its decimals.

    """
    if isinstance(label, float) and label.is_integer():
        return str(int(label))
    return str(label)
