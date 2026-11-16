"""The base class of mergers, and what they share.

Contents:
    _DUPLICATES: what a merger can do with a row of the data that matches more
        than one row of the other table.
    _ROW, _POSITION, _VALUE: names of the columns that mergers add to the
        tables that they compare: the position of a row in the data and in the
        other table, and the number or date of a row of the data.
    Merger: base class for techniques that add columns from another table.

"""

from __future__ import annotations

import abc
import dataclasses
import pathlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import camina
import nagata
import numpy as np
import pandas as pd

from ...core import base
from .. import loaders

if TYPE_CHECKING:
    import chrisjen


# What a merger can do with a row of the data that matches more than one row
# of the other table.
_DUPLICATES: frozenset[str] = frozenset({'error', 'first', 'last'})
# Names of the columns that mergers add to the tables that they compare: the
# position of a row in the data and in the other table, the number or date
# of a row of the data, and the start and end of a range.
_ROW: str = '__row__'
_POSITION: str = '__position__'
_VALUE: str = '__value__'


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
    | `prefix` | Text to put before the names of the added columns (such as "store_"). |
    | `indicator` | The name of a column to make that says whether each row was matched. |

    The dataset's `history` records the source, the number of rows of the
    data, how many of them were matched, and the columns that were added. A
    merger built from a project's settings uses the project's clerk.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
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
        return super().build(name, project, parameters, **kwargs)

    """ Public Methods """

    def apply(
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
            keys = camina.listify(kwargs.get('other_on'))
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
                f'cleaner (such as clean_numbers or clean_dates) or a munger '
                f'(such as convert_types)'
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
            f'column of the data the same kind with a cleaner (such as '
            f'clean_numbers or clean_dates)'
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
    mine = camina.listify(on)
    theirs = camina.listify(other_on) or mine
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
