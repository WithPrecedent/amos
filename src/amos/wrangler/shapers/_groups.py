"""Shapers that group rows or spread lists into rows.

Contents:
    _SUMMARIES: the summaries that `group_rows` (and `merge_summary`) can take
        of the rows of a group, with what `pandas` calls each one.
    GroupRows: makes one row for each group of rows, with summaries of its
        columns.
    ListsToRows: makes a row for each item of a list.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import camina
import numpy as np
import pandas as pd

from ...core import utilities
from ._base import Shaper, _claim, _keep_index

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
class GroupRows(Shaper):
    """Makes one row for each group of rows, with summaries of its columns.

    For example, a table with a row for each student in each class becomes a
    table with a row for each class, with the share of its students who are
    women and their mean age. It is the opposite of `lists_to_rows`.

    Unless "how" says otherwise, a column that has the same value in every
    row of each group keeps that value (so a class keeps its teacher).
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
    """Makes a row for each item of a list, such as each author of a book.

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


""" Private Functions """


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
        chosen = {
            c: [str(s) for s in camina.listify(summaries)]
            for c, summaries in how.items()}
    elif how is not None:
        everywhere = [str(s) for s in camina.listify(how)]
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
