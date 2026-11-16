"""A merger that matches rows to ranges.

Contents:
    _PAIRS: the most pairs of rows that `merge_ranges` compares at once, which
        limits the memory that it uses.
    _START, _END: names of the columns of the start and end of a range in
        the tables that `merge_ranges` compares.
    MergeRanges: adds columns from another table's row whose range has a date or
        number in it.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass
from ._base import (
    _POSITION,
    _ROW,
    _VALUE,
    Merger,
    _check_duplicates,
    _column,
    _keys,
    _measures,
)

# The most pairs of rows that `merge_ranges` compares at once, which limits
# the memory that it uses.
_PAIRS: int = 2_000_000
_START: str = '__start__'
_END: str = '__end__'


@dataclasses.dataclass
class MergeRanges(Merger):
    """Adds columns from another table's row whose range has a date or number.

    For example, each answer to a survey gets the columns of the job that the
    employee held when they answered, from a table with a row for each job
    that each employee has held, with the dates that the job began and ended.
    The row must have the same keys (here, the employee's ID) and a range
    that has the date of the answer in it. The start and the end are both in
    the range.

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
