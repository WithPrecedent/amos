r"""Techniques that add the columns of another table to the data.

These are techniques of the "wrangler" stage of a project. Research often
needs data from sources that are not organized the same way: one table has a
row for each case, and another has a row for each judge, or for each court
that a judge has served on. A merger matches each row of the data to a row
of another table and adds the columns of that row. The mergers differ in how
they match rows:

* `merge_keys` matches the row with the same keys (such as the same court).
* `merge_nearest` matches the row with the nearest number or date (such as
  the last rating of a court before a case was decided).
* `merge_ranges` matches the row whose range has a number or date in it (such
  as the judge with a name who was serving in the year of a case).
* `merge_summary` matches every row with the same keys, and adds summaries of
  them (such as the number of judges of a court and their mean age).

A merger never adds, removes, or reorders the rows of the data, so the data
keeps the unit that it studies, and its label, groups, and index. A row
without a match gets missing values. Start from the table whose rows are what
you study, and merge the others into it. To change what a row is first (to
make a row for each judge on each case, for example), use a shaper (see the
`shapers` module).

The other table is the "source" parameter of every merger: the path of a data
file, which is loaded with the project's clerk as loaders load theirs, or a
table (in Python). Mergers learn nothing from the data that could leak from
the test rows into the training rows.

Contents:
    Merger: base class for techniques that add columns from another table.
    MergeKeys: adds columns from another table's row that has the same keys.
    MergeNearest: adds columns from another table's row with the nearest date
        or number.
    MergeRanges: adds columns from another table's row whose range has a date
        or number in it.
    MergeSummary: adds summaries of the rows of another table that have the
        same keys.

"""

from __future__ import annotations

import abc
import dataclasses
import pathlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import nagata
import numpy as np
import pandas as pd

from . import base, loaders, shapers

if TYPE_CHECKING:
    import chrisjen

# What `merge_nearest` calls each direction to look in, with what `pandas`
# calls it.
_DIRECTIONS: dict[str, str] = {
    'after': 'forward', 'before': 'backward', 'nearest': 'nearest'}
# What a merger can do with a row of the data that matches more than one row
# of the other table.
_DUPLICATES: frozenset[str] = frozenset({'error', 'first', 'last'})
# The most pairs of rows that `merge_ranges` compares at once, which limits
# the memory that it uses.
_PAIRS: int = 2_000_000
# Names of the columns that mergers add to the tables that they compare: the
# position of a row in the data and in the other table, the number or date
# of a row of the data, and the start and end of a range.
_ROW: str = '__row__'
_POSITION: str = '__position__'
_VALUE: str = '__value__'
_START: str = '__start__'
_END: str = '__end__'


@dataclasses.dataclass
class Merger(base.Operation, abc.ABC):
    """Base class for techniques that add columns from another table.

    A subclass writes a `match` method, which finds the row of the other
    table that matches each row of the data. The merger adds the columns of
    the matching rows. A subclass can also write a `prepare` method, which
    changes the other table before its rows are matched.

    Every merger takes these parameters, usually set in the
    "{name}_parameters" section of the settings:

    | Parameter | Meaning |
    | --- | --- |
    | `source` | The other table: the path of a data file (looked for in the current folder and then in the clerk's input folder), or a `DataFrame`, a `Dataset`, or anything else that `Dataset.create` accepts. |
    | `reader` | A `dict` of parameters for loading the file, as for `load_file` (such as "file_format", "member", or "sep"). |
    | `columns` | The columns of the other table to add. By default, they are every column whose name the data does not already have, other than the keys of the other table ("other_on"), which repeat those of the data. |
    | `prefix` | Text to put before the names of the added columns (such as "judge_"). |
    | `indicator` | The name of a column to make that says whether each row was matched. |

    The dataset's `history` records the source, the number of rows of the
    data, how many of them were matched, and the columns that were added. A
    merger built from a project's settings uses the project's clerk.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: not used by most mergers. Defaults to `None`.
        parameters: keyword arguments for `match` and `prepare`, and
            "source", "reader", "columns", "prefix", and "indicator".
            Defaults to an empty `dict`.
        clerk: the file manager that finds and loads files. Defaults to
            `None`, in which case one for the current folder is made when it
            is needed.

    """

    clerk: nagata.FileManager | None = dataclasses.field(
        default = None, repr = False)

    """ Required Methods """

    @abc.abstractmethod
    def match(
        self,
        data: pd.DataFrame,
        other: pd.DataFrame,
        **kwargs: Any) -> Any:
        """Returns the row of `other` that matches each row of `data`.

        Args:
            data: the data.
            other: the other table (as `prepare` returns it).
            **kwargs: parameters for matching.

        Returns:
            For each row of `data`, in order, the position (from 0) of the
                row of `other` that matches it, or -1 if none does: an array
                or a sequence of whole numbers.

        """

    """ Class Methods """

    @classmethod
    def build(
        cls,
        name: str,
        project: chrisjen.Project,
        parameters: base.GenericDict | None = None,
        **kwargs: Any) -> Merger:
        """Builds a merger with the clerk of `project`.

        Args:
            name: name of the merger.
            project: the project the merger belongs to.
            parameters: more parameters, which take precedence over those in
                the settings. Defaults to `None`.
            **kwargs: other arguments for the constructor.

        Returns:
            The built merger.

        """
        kwargs.setdefault('clerk', project.clerk)
        return cast('Merger', super().build(
            name, project, parameters, **kwargs))

    """ Public Methods """

    def implement(
        self,
        item: base.Dataset,
        *,
        source: Any = None,
        reader: Mapping[str, Any] | None = None,
        columns: Sequence[str] | str | None = None,
        prefix: str | None = None,
        indicator: str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Adds the columns of the matching rows of another table to `item`.

        Args:
            item: the dataset to add columns to.
            source: the other table: the path of a data file, or a
                `DataFrame`, a `Dataset`, or anything else that
                `Dataset.create` accepts.
            reader: parameters for loading the file at `source` (see
                `Loader.read`). Defaults to `None`.
            columns: the columns of the other table to add. Defaults to
                `None`, in which case they are every column whose name the
                data does not already have, other than the keys of the other
                table (the "other_on" parameter of `match`).
            prefix: text to put before the names of the added columns.
                Defaults to `None`.
            indicator: name of a boolean column to make that says whether
                each row was matched. Defaults to `None`, which makes no such
                column.
            **kwargs: parameters for `prepare` and `match`.

        Raises:
            KeyError: if one of `columns` is not in the other table.
            ValueError: if there is no `source`, `match` does not return a
                position for each row, or an added column would have the
                name of a column of the data.

        Returns:
            The dataset with the added columns. Its rows and index are not
                changed.

        """
        if source is None:
            message = (
                f'{self.name!r} has nothing to merge: set its "source" in the '
                f'"{self.name}_parameters" section of the settings'
            )
            raise ValueError(message)
        other = self.prepare(
            self.read(source, **dict(reader or {})), **kwargs)
        data = item.data
        positions = _positions(
            self.match(data, other, **kwargs), data, other, self.name)
        if columns is None:
            # The keys of the other table repeat those of the data.
            keys = base._listify(kwargs.get('other_on'))
            columns = [
                c for c in other.columns
                if c not in data.columns and c not in keys]
        added = _added(
            data,
            other,
            positions,
            columns = columns,
            prefix = prefix,
            indicator = indicator,
            technique = self.name)
        item.replace(pd.concat([data, added], axis = 1))
        item.record(
            self.name,
            source = (
                str(source) if isinstance(source, str | pathlib.Path)
                else type(source).__name__),
            rows = len(data),
            matched = int((positions >= 0).sum()),
            created = list(added.columns))
        return item

    def prepare(self, other: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Returns the other table, ready for its rows to be matched.

        Subclasses override this to change the other table first (to
        summarize its rows, for example).

        Args:
            other: the other table. It must not be changed in place.
            **kwargs: parameters for preparing it.

        Returns:
            The table whose rows `match` finds and whose columns are added.

        """
        return other

    def read(self, source: Any, **kwargs: Any) -> pd.DataFrame:
        """Returns the other table.

        Args:
            source: the path of a data file, which is loaded with the clerk
                (see `Loader.read`), or a `DataFrame`, a `Dataset`, or
                anything else that `Dataset.create` accepts.
            **kwargs: parameters for loading a file.

        Returns:
            The table.

        """
        if isinstance(source, base.Dataset):
            return source.data
        if isinstance(source, str | pathlib.Path):
            source = loaders.LoadFile(clerk = self.clerk).read(
                source, **kwargs)
        return base.Dataset.create(source).data


@dataclasses.dataclass
class MergeKeys(Merger):
    """Adds columns from another table's row that has the same keys.

    For example, each case gets the columns of its court from a table with a
    row for each court. The keys are the columns that say which row of the
    other table a row of the data belongs with (here, the name of the court).
    Many rows of the data can match the same row of the other table. A row
    with a missing key matches nothing.

    Wraps:
        [`pandas.merge`](https://pandas.pydata.org/docs/reference/api/pandas.merge.html)
        from pandas.

    """

    def match(
        self,
        data: pd.DataFrame,
        other: pd.DataFrame,
        *,
        on: Sequence[str] | str | None = None,
        other_on: Sequence[str] | str | None = None,
        ignorecase: bool = False,
        duplicates: str = 'error',
        **kwargs: Any) -> Any:
        """Returns the row of `other` with the same keys as each row.

        Args:
            data: the data.
            other: the other table.
            on: the key columns of the data.
            other_on: the key columns of the other table, in the same order.
                Defaults to `None`, in which case they have the same names
                as `on`.
            ignorecase: whether text keys match without regard to capital
                and lower-case letters or the spaces around them. Defaults to
                `False`.
            duplicates: what to do if several rows of the other table have
                the same keys: "error" (stop, since a row of the data can
                only match one), "first" (use the first of them), or "last".
                Defaults to "error".
            **kwargs: not used.

        Raises:
            KeyError: if a key is not a column of its table.
            TypeError: if a key is not the same kind of data (numbers, dates,
                or text) in both tables.
            ValueError: if there is no `on`, the numbers of `on` and
                `other_on` differ, `duplicates` is not one of those above, or
                rows of the other table have the same keys and `duplicates`
                is "error".

        Returns:
            The position of the matching row of `other` for each row of
                `data`, or -1.

        """
        left, right = _keys(
            data,
            other,
            on,
            other_on,
            ignorecase = ignorecase,
            technique = self.name)
        return _exact(left, right, duplicates, self.name)


@dataclasses.dataclass
class MergeNearest(Merger):
    """Adds columns from another table's row with the nearest date or number.

    For example, each case gets the most recent rating of its court before
    the day it was decided, from a table with a row for each rating. The row
    can also be limited to those with the same keys (here, the court).

    Wraps:
        [`pandas.merge_asof`](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html)
        from pandas.

    """

    def match(
        self,
        data: pd.DataFrame,
        other: pd.DataFrame,
        *,
        column: str | None = None,
        other_column: str | None = None,
        on: Sequence[str] | str | None = None,
        other_on: Sequence[str] | str | None = None,
        direction: str = 'before',
        tolerance: Any = None,
        ignorecase: bool = False,
        **kwargs: Any) -> Any:
        """Returns the row of `other` with the nearest value to each row's.

        Args:
            data: the data.
            other: the other table.
            column: the column of numbers or dates of the data.
            other_column: the column of numbers or dates of the other table.
                Defaults to `None`, in which case it has the same name as
                `column`.
            on: key columns of the data, which the matching row must also
                have the same values of. Defaults to `None`, for no keys.
            other_on: the key columns of the other table. Defaults to `None`,
                in which case they have the same names as `on`.
            direction: where to look for the nearest row: "before" (the last
                row at or before the value, so nothing is used that was not
                yet known), "after" (the first row at or after it), or
                "nearest" (either). Defaults to "before".
            tolerance: the farthest that the row can be from the value: a
                number, or, for dates, a number of days or text such as
                "12h". Defaults to `None`, for any distance.
            ignorecase: whether text keys match without regard to capital
                and lower-case letters or the spaces around them. Defaults to
                `False`.
            **kwargs: not used.

        Raises:
            KeyError: if a column is not in its table.
            TypeError: if the columns are not both numbers or both dates.
            ValueError: if there is no `column`, or `direction` is not one of
                those above.

        Returns:
            The position of the matching row of `other` for each row of
                `data`, or -1. A row with a missing value or key matches
                nothing.

        """
        if column is None:
            message = (
                f'{self.name!r} needs a "column" of numbers or dates to find '
                f'the nearest row with'
            )
            raise ValueError(message)
        if direction not in _DIRECTIONS:
            message = (
                f'direction must be one of {sorted(_DIRECTIONS)}, not '
                f'{direction!r}'
            )
            raise ValueError(message)
        left, right = _keys(
            data,
            other,
            on,
            other_on,
            ignorecase = ignorecase,
            technique = self.name,
            needed = False)
        keys = list(left.columns)
        near, far = _measures(
            _column(data, column, 'the data'),
            _column(other, other_column or column, 'the other table'),
            self.name)
        left = left.assign(**{
            _VALUE: near.to_numpy(), _ROW: np.arange(len(left))})
        right = right.assign(**{
            _VALUE: far.to_numpy(), _POSITION: np.arange(len(right))})
        # `pandas` needs both tables in the order of their values, without
        # missing ones.
        left = left.dropna().sort_values(_VALUE, kind = 'stable')
        right = right.dropna().sort_values(_VALUE, kind = 'stable')
        positions = np.full(len(data), -1, dtype = 'int64')
        if len(left) > 0 and len(right) > 0:
            found = pd.merge_asof(
                left,
                right,
                on = _VALUE,
                by = keys or None,
                direction = cast('Any', _DIRECTIONS[direction]),
                tolerance = _span(tolerance, near))
            positions[found[_ROW].to_numpy()] = found[_POSITION].fillna(
                -1).to_numpy(dtype = 'int64')
        return positions


@dataclasses.dataclass
class MergeRanges(Merger):
    """Adds columns from another table's row whose range has a date or number.

    For example, each judge on a case gets the columns of the judge's service
    on a court from a table with a row for each court that each judge has
    served on, with the years that the service began and ended. The row must
    have the same keys (here, the judge's name) and a range that has the year
    of the case in it. The start and the end are both in the range.

    Without keys, every row of the data is compared with every row of the
    other table, which suits a small table of ranges (such as the terms of
    presidents).

    Wraps:
        [`pandas.merge`](https://pandas.pydata.org/docs/reference/api/pandas.merge.html)
        from pandas.

    """

    def match(
        self,
        data: pd.DataFrame,
        other: pd.DataFrame,
        *,
        column: str | None = None,
        start: str | None = None,
        end: str | None = None,
        on: Sequence[str] | str | None = None,
        other_on: Sequence[str] | str | None = None,
        duplicates: str = 'error',
        ignorecase: bool = False,
        **kwargs: Any) -> Any:
        """Returns the row of `other` whose range has each row's value in it.

        Args:
            data: the data.
            other: the other table.
            column: the column of numbers or dates of the data.
            start: the column of the other table with the start of each
                range. A range with a missing start has no start. Defaults to
                `None`, for ranges without starts.
            end: the column of the other table with the end of each range. A
                range with a missing end has no end (so it is still going
                on). Defaults to `None`, for ranges without ends.
            on: key columns of the data, which the matching row must also
                have the same values of. Defaults to `None`, for no keys.
            other_on: the key columns of the other table. Defaults to `None`,
                in which case they have the same names as `on`.
            duplicates: what to do if the value of a row is in several
                ranges: "error" (stop, since a row of the data can only match
                one), "first" (use the first of them in the other table), or
                "last". Defaults to "error".
            ignorecase: whether text keys match without regard to capital
                and lower-case letters or the spaces around them. Defaults to
                `False`.
            **kwargs: not used.

        Raises:
            KeyError: if a column is not in its table.
            TypeError: if the columns are not all numbers or all dates.
            ValueError: if there is no `column` or neither `start` nor `end`,
                `duplicates` is not one of those above, or a value is in
                several ranges and `duplicates` is "error".

        Returns:
            The position of the matching row of `other` for each row of
                `data`, or -1. A row with a missing value or key matches
                nothing, and neither does a range with neither a start nor
                an end.

        """
        if column is None or (start is None and end is None):
            message = (
                f'{self.name!r} needs a "column" of numbers or dates, and the '
                f'"start" or "end" (or both) of the ranges of the other table'
            )
            raise ValueError(message)
        _check_duplicates(duplicates)
        left, right = _keys(
            data,
            other,
            on,
            other_on,
            ignorecase = ignorecase,
            technique = self.name,
            needed = False)
        keys = list(left.columns)
        value = _column(data, column, 'the data')
        right = right.assign(**{_POSITION: np.arange(len(right))})
        bounds = [(n, b) for n, b in ((_START, start), (_END, end)) if b]
        for name, bound in bounds:
            value, limit = _measures(
                value, _column(other, bound, 'the other table'), self.name)
            right[name] = limit.to_numpy()
        left = left.assign(**{
            _VALUE: value.to_numpy(), _ROW: np.arange(len(left))}).dropna()
        # A range with neither a start nor an end says nothing.
        right = right.dropna(subset = [n for n, _ in bounds], how = 'all')
        if keys:
            right = right.dropna(subset = keys)
        # Rows are paired a part of the data at a time, so that the pairs fit
        # in memory: no row of the data has more pairs than the most rows of
        # the other table that share keys.
        most = len(right)
        if keys and len(right) > 0:
            most = int(right.groupby(keys).size().max())
        size = max(1, _PAIRS // max(most, 1))
        found = [pd.DataFrame({
            _ROW: pd.Series(dtype = 'int64'),
            _POSITION: pd.Series(dtype = 'int64')})]
        for begin in range(0, len(left), size):
            part = left.iloc[begin:begin + size]
            pairs = (
                part.merge(right, on = keys) if keys
                else part.merge(right, how = 'cross'))
            within = np.ones(len(pairs), dtype = bool)
            if start is not None:
                within &= np.asarray(
                    pairs[_START].isna() | (pairs[_START] <= pairs[_VALUE]))
            if end is not None:
                within &= np.asarray(
                    pairs[_END].isna() | (pairs[_VALUE] <= pairs[_END]))
            found.append(pairs.loc[within, [_ROW, _POSITION]])
        matches = pd.concat(found, ignore_index = True).sort_values(
            [_ROW, _POSITION], kind = 'stable')
        repeated = matches.loc[
            np.asarray(matches[_ROW].duplicated(keep = False)), _ROW]
        if len(repeated) > 0:
            if duplicates == 'error':
                example = data[column].iloc[int(repeated.iloc[0])]
                message = (
                    f'{repeated.nunique()} rows of the data are in more than '
                    f'one range of the other table (such as a row with '
                    f'{column!r} of {example!r}): set "duplicates" to '
                    f'"first" or "last" to use the first or last of them in '
                    f'the other table'
                )
                raise ValueError(message)
            matches = matches.drop_duplicates(
                _ROW, keep = 'first' if duplicates == 'first' else 'last')
        positions = np.full(len(data), -1, dtype = 'int64')
        positions[matches[_ROW].to_numpy(dtype = 'int64')] = matches[
            _POSITION].to_numpy(dtype = 'int64')
        return positions


@dataclasses.dataclass
class MergeSummary(MergeKeys):
    """Adds summaries of the rows of another table that have the same keys.

    For example, each court gets the number of its judges and their mean age
    from a table with a row for each judge. The rows of the other table are
    summarized as `group_rows` summarizes them (see the `shapers` module),
    and each row of the data gets the summaries of the rows with its keys. A
    row that matches no rows gets missing summaries.

    Wraps:
        - [`pandas.DataFrame.groupby`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html)
          from pandas, to summarize the rows.
        - [`pandas.merge`](https://pandas.pydata.org/docs/reference/api/pandas.merge.html)
          from pandas, to match the summaries.

    """

    def prepare(
        self,
        other: pd.DataFrame,
        *,
        on: Sequence[str] | str | None = None,
        other_on: Sequence[str] | str | None = None,
        how: Mapping[str, Any] | Sequence[str] | str | None = None,
        count: str | None = None,
        ignorecase: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the other table with one row of summaries for each key.

        Args:
            other: the other table.
            on: the key columns of the data.
            other_on: the key columns of the other table, in the same order.
                Defaults to `None`, in which case they have the same names
                as `on`.
            how: the summary to take of each column of numbers and booleans
                of the other table: "mean", "median", "sum", "min", "max",
                "std", "count", "nunique", "first", "last", "any", "all", or
                "list". A list of summaries takes each of them. A `dict`
                maps the columns to summarize (of any kind) to their
                summaries. Defaults to `None`, which takes the mean (for a
                boolean, the share that are true).
            count: name of a column to make with the number of rows that
                have each key. Defaults to `None`, which makes no such
                column.
            ignorecase: whether text keys match without regard to capital
                and lower-case letters or the spaces around them. Defaults to
                `False`.
            **kwargs: not used.

        Returns:
            The summaries: a column with one summary keeps its name, and a
                column with several becomes a column for each, named
                "{column}_{summary}".

        """
        names = _names(on, other_on, self.name)[1]
        _require(other, names, 'the other table')
        if ignorecase:
            other = other.assign(**{
                name: _folded(other[name]) for name in names
                if _kind(_plain(other[name])) == 'text'})
        summarized = list(how) if isinstance(how, Mapping) else [
            c for c in other.columns
            if c not in names
            and pd.api.types.is_numeric_dtype(other[c].dtype)]
        _require(other, summarized, 'the other table')
        return shapers.GroupRows().shape(
            other[[*names, *(c for c in summarized if c not in names)]],
            by = names,
            how = how,
            count = count)


""" Private Functions """


def _added(
    data: pd.DataFrame,
    other: pd.DataFrame,
    positions: np.ndarray,
    *,
    columns: Sequence[str] | str,
    prefix: str | None,
    indicator: str | None,
    technique: str) -> pd.DataFrame:
    """Returns the columns of the matching rows of the other table.

    Args:
        data: the data.
        other: the other table.
        positions: for each row of `data`, the position of the row of `other`
            that matches it, or -1.
        columns: the columns of `other` to add.
        prefix: text to put before the names of the columns, or `None`.
        indicator: name of a column that says whether each row was matched,
            or `None`.
        technique: name of the technique that adds the columns, for messages.

    Raises:
        KeyError: if one of `columns` is not in `other`.
        ValueError: if a new column would have the name of a column of
            `data`, or of another new column.

    Returns:
        The new columns, with the index of `data`. A row without a match has
            missing values, so a column of whole numbers becomes one of
            floats, and a column of booleans gets the "boolean" type of
            `pandas`, which allows missing values.

    """
    chosen = [columns] if isinstance(columns, str) else list(columns)
    _require(other, chosen, 'the other table')
    names = [f'{prefix}{c}' if prefix else c for c in chosen]
    made = names if indicator is None else [*names, indicator]
    clashes = [
        n for position, n in enumerate(made)
        if n in data.columns or n in made[:position]]
    if clashes:
        message = (
            f'{technique!r} would add columns named {clashes}, which the '
            f'data already has: set "prefix" to tell them apart, or name the '
            f'columns to add in "columns"'
        )
        raise ValueError(message)
    matched = positions >= 0
    table = other[chosen].reset_index(drop = True)
    if not matched.all():
        table = table.astype({
            c: 'boolean' for c in dict.fromkeys(chosen)
            if pd.api.types.is_bool_dtype(table[c].dtype)})
    # A position of -1 is no row of the table, so its row is missing.
    added = table.reindex(positions).set_axis(names, axis = 1)
    added.index = data.index
    if indicator is not None:
        added[indicator] = matched
    return added


def _check_duplicates(duplicates: str) -> None:
    """Checks what to do with rows that match more than one row.

    Args:
        duplicates: "error", "first", or "last".

    Raises:
        ValueError: if `duplicates` is anything else.

    """
    if duplicates not in _DUPLICATES:
        message = (
            f'duplicates must be one of {sorted(_DUPLICATES)}, not '
            f'{duplicates!r}'
        )
        raise ValueError(message)


def _column(data: pd.DataFrame, column: str, table: str) -> pd.Series:
    """Returns a column of a table.

    Args:
        data: the table.
        column: name of the column.
        table: what the table is ("the data" or "the other table"), for
            messages.

    Raises:
        KeyError: if `column` is not in `data`.

    Returns:
        The column.

    """
    _require(data, [column], table)
    values: pd.Series = data[column]
    return values


def _exact(
    left: pd.DataFrame,
    right: pd.DataFrame,
    duplicates: str,
    technique: str) -> np.ndarray:
    """Returns the row of a table with the same keys as each row of another.

    Args:
        left: the keys of the data (as `_keys` returns them).
        right: the keys of the other table, with the same names.
        duplicates: what to do if several rows of `right` have the same
            keys: "error", "first", or "last".
        technique: name of the technique that matches the rows, for messages.

    Raises:
        ValueError: if rows of `right` have the same keys and `duplicates`
            is "error", or `duplicates` is not one of those above.

    Returns:
        The position of the matching row of `right` for each row of `left`,
            or -1. Missing keys match nothing.

    """
    _check_duplicates(duplicates)
    keys = list(left.columns)
    right = right.assign(**{_POSITION: np.arange(len(right))}).dropna(
        subset = keys)
    repeated = right.duplicated(keys, keep = False)
    if repeated.any():
        if duplicates == 'error':
            example = right.loc[repeated, keys].iloc[0].to_dict()
            message = (
                f'{int(repeated.sum())} rows of the other table have the '
                f'same {keys} as another of its rows (such as {example}), '
                f'and a row of the data can only match one: set '
                f'"duplicates" to "first" or "last" to use one of them, or '
                f'summarize them with merge_summary instead of {technique}'
            )
            raise ValueError(message)
        right = right.drop_duplicates(
            keys, keep = 'first' if duplicates == 'first' else 'last')
    found = left.merge(right, on = keys, how = 'left')[_POSITION]
    return found.fillna(-1).to_numpy(dtype = 'int64')


def _folded(values: pd.Series) -> pd.Series:
    """Returns text the way it is compared when its case is ignored.

    Args:
        values: a column of text.

    Returns:
        The text in lower case, without spaces at its ends.

    """
    return values.astype('str').str.strip().str.lower()


def _keys(
    data: pd.DataFrame,
    other: pd.DataFrame,
    on: Sequence[str] | str | None,
    other_on: Sequence[str] | str | None,
    *,
    ignorecase: bool,
    technique: str,
    needed: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns the keys of two tables, so that they can be compared.

    Args:
        data: the data.
        other: the other table.
        on: the key columns of `data`, or `None`.
        other_on: the key columns of `other`, or `None` if they have the
            same names as `on`.
        ignorecase: whether text keys match without regard to capital and
            lower-case letters or the spaces around them.
        technique: name of the technique that compares the keys, for
            messages.
        needed: whether there must be keys. Defaults to `True`.

    Raises:
        KeyError: if a key is not a column of its table.
        TypeError: if a key is not the same kind of data in both tables.
        ValueError: if there are no keys and they are `needed`, or the
            numbers of `on` and `other_on` differ.

    Returns:
        The keys of `data` and of `other`, with the index of each, both
            named as they are in `other`. Each key has the same type in both
            (see `_same`). They have no columns if there are no keys.

    """
    if on is None and other_on is None and not needed:
        return pd.DataFrame(index = data.index), pd.DataFrame(
            index = other.index)
    mine, theirs = _names(on, other_on, technique)
    _require(data, mine, 'the data')
    _require(other, theirs, 'the other table')
    left: dict[str, pd.Series] = {}
    right: dict[str, pd.Series] = {}
    for near, far in zip(mine, theirs, strict = True):
        first, second = _plain(data[near]), _plain(other[far])
        kinds = _kind(first), _kind(second)
        if kinds[0] != kinds[1]:
            message = (
                f'{technique!r} cannot match {near!r}, which is {kinds[0]} in '
                f'the data, with {far!r}, which is {kinds[1]} in the other '
                f'table: make the column of the data the same kind with a '
                f'munger (such as parse_numbers, parse_dates, or '
                f'convert_types)'
            )
            raise TypeError(message)
        left[far], right[far] = _same(
            first, second, kinds[0], ignorecase = ignorecase)
    return pd.DataFrame(left), pd.DataFrame(right)


def _kind(values: pd.Series) -> str:
    """Returns the kind of data in a column, as mergers compare it.

    Args:
        values: a column that is not categorical (see `_plain`).

    Returns:
        "numbers" (which booleans are compared as), "dates", "text", or
            "other values".

    """
    dtype = values.dtype
    if pd.api.types.is_numeric_dtype(dtype):
        return 'numbers'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'dates'
    # A column of objects is text only if every value in it is.
    if pd.api.types.is_string_dtype(dtype) and (
        not pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(values.dropna())):
        return 'text'
    return 'other values'


def _measures(
    first: pd.Series,
    second: pd.Series,
    technique: str) -> tuple[pd.Series, pd.Series]:
    """Returns two columns of numbers or dates, so that they can be compared.

    Args:
        first: a column of the data.
        second: a column of the other table.
        technique: name of the technique that compares them, for messages.

    Raises:
        TypeError: if the columns are not both numbers or both dates.

    Returns:
        The columns, with the same type (see `_same`).

    """
    first, second = _plain(first), _plain(second)
    kinds = _kind(first), _kind(second)
    if kinds[0] != kinds[1] or kinds[0] not in {'numbers', 'dates'}:
        message = (
            f'{technique!r} compares numbers with numbers or dates with '
            f'dates, but {first.name!r} is {kinds[0]} in the data and '
            f'{second.name!r} is {kinds[1]} in the other table: make the '
            f'column of the data the same kind with a munger (such as '
            f'parse_numbers or parse_dates)'
        )
        raise TypeError(message)
    return _same(first, second, kinds[0], exact = False)


def _names(
    on: Sequence[str] | str | None,
    other_on: Sequence[str] | str | None,
    technique: str) -> tuple[list[str], list[str]]:
    """Returns the names of the key columns of the data and the other table.

    Args:
        on: the key columns of the data, or `None`.
        other_on: the key columns of the other table, or `None` if they
            have the same names as `on`.
        technique: name of the technique that uses the keys, for messages.

    Raises:
        ValueError: if there is no `on`, or the numbers of `on` and
            `other_on` differ.

    Returns:
        The names of the keys of the data and of the other table.

    """
    mine = base._listify(on)
    theirs = base._listify(other_on) or mine
    if not mine:
        message = (
            f'{technique!r} needs the key columns that say which rows '
            f'belong together: set "on" (and "other_on", if they have other '
            f'names in the other table)'
        )
        raise ValueError(message)
    if len(mine) != len(theirs):
        message = (
            f'{technique!r} needs as many keys in "other_on" ({theirs}) as '
            f'in "on" ({mine})'
        )
        raise ValueError(message)
    return mine, theirs


def _plain(values: pd.Series) -> pd.Series:
    """Returns a column with categories as the values that they stand for.

    Args:
        values: a column.

    Returns:
        A categorical column with the type of its categories, and any other
            column as it is.

    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        return values.astype(values.cat.categories.dtype)
    return values


def _positions(
    found: Any,
    data: pd.DataFrame,
    other: pd.DataFrame,
    technique: str) -> np.ndarray:
    """Returns what a `match` method found, checking it.

    Args:
        found: what `match` returned.
        data: the data.
        other: the other table.
        technique: name of the technique that matched the rows, for messages.

    Raises:
        ValueError: if `found` is not the position of a row of `other` (or
            -1) for each row of `data`.

    Returns:
        The positions, as an array of whole numbers.

    """
    try:
        positions = np.asarray(found, dtype = 'int64')
    except (TypeError, ValueError):
        positions = np.empty(0, dtype = 'int64')
    if positions.shape != (len(data),) or (
        len(positions) > 0
        and (positions.min() < -1 or positions.max() >= len(other))):
        message = (
            f'the match method of {technique!r} must return the position of '
            f'a row of the other table (or -1) for each of the {len(data)} '
            f'rows of the data'
        )
        raise ValueError(message)
    return positions


def _require(data: pd.DataFrame, columns: Sequence[str], table: str) -> None:
    """Checks that a table has columns.

    Args:
        data: the table.
        columns: names of the columns that it needs.
        table: what the table is ("the data" or "the other table"), for
            messages.

    Raises:
        KeyError: if a column is not in `data`.

    """
    missing = [c for c in columns if c not in data.columns]
    if missing:
        message = f'the columns {missing} are not in {table}'
        raise KeyError(message)


def _same(
    first: pd.Series,
    second: pd.Series,
    kind: str,
    *,
    exact: bool = True,
    ignorecase: bool = False) -> tuple[pd.Series, pd.Series]:
    """Returns two columns of one kind of data with the same type.

    `pandas` only compares columns of the same type, and a table that was
    loaded from a file often has whole numbers where another has floats, or
    dates of another precision.

    Args:
        first: a column of the data.
        second: a column of the other table, of the same kind (see `_kind`).
        kind: the kind of data in both columns.
        exact: whether numbers of the same type keep it, so that whole
            numbers too large for a float (such as long ids) still match
            exactly. Defaults to `True`.
        ignorecase: whether to compare text without regard to capital and
            lower-case letters or the spaces around it. Defaults to `False`.

    Returns:
        The columns: numbers as floats (unless they have the same type and
            `exact` is true), dates without time zones in nanoseconds, and
            text as the text type of `pandas`. Anything else is not changed.

    """
    if kind == 'numbers' and not (exact and first.dtype == second.dtype):
        return first.astype('float64'), second.astype('float64')
    if kind == 'dates' and first.dt.tz is None and second.dt.tz is None:
        return (
            first.astype('datetime64[ns]'), second.astype('datetime64[ns]'))
    if kind == 'text':
        if ignorecase:
            return _folded(first), _folded(second)
        return first.astype('str'), second.astype('str')
    return first, second


def _span(tolerance: Any, values: pd.Series) -> Any:
    """Returns the farthest that a row can be, as `pandas` takes it.

    Args:
        tolerance: a number, text such as "12h" (for dates), or `None`.
        values: the numbers or dates that rows are near to.

    Returns:
        `None` for any distance, a length of time for dates (a number is a
            number of days), or else a float.

    """
    if tolerance is None:
        return None
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        if isinstance(tolerance, int | float):
            return pd.Timedelta(days = tolerance)
        return pd.Timedelta(tolerance)
    return float(tolerance)
