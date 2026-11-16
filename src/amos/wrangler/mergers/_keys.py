"""Mergers that match rows by their keys.

Contents:
    MergeKeys: adds columns from another table's row that has the same keys.
    MergeSummary: adds summaries of the rows of another table that have the same
        keys.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .. import shapers

if TYPE_CHECKING:
    pass
from ._base import (
    _POSITION,
    Merger,
    _check_duplicates,
    _folded,
    _keys,
    _kind,
    _names,
    _plain,
    _require,
)


@dataclasses.dataclass
class MergeKeys(Merger):
    """Adds columns from another table's row that has the same keys.

    For example, each sale gets the columns of its store from a table with a
    row for each store. The keys are the columns that say which row of the
    other table a row of the data belongs with (here, the store's ID).
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
class MergeSummary(MergeKeys):
    """Adds summaries of the rows of another table that have the same keys.

    For example, each school gets the number of its students and their mean age
    from a table with a row for each student. The rows of the other table are
    summarized as `group_rows` summarizes them (see the `shapers` module), and
    each row of the data gets the summaries of the rows with its keys. A row
    that matches no rows gets missing summaries.

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
