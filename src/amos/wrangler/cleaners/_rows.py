"""Cleaners that remove or keep rows.

Contents:
    _COMPARISON: the start of a condition of `filter_rows` that compares a
        column to a value: an operator, or "in" or "not in" as words.
    DropDuplicates: removes duplicate rows.
    DropMissing: removes rows with missing values.
    FilterRows: keeps the rows that match a query, or a condition for each
        column.

"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ...core import utilities
from ._base import Cleaner

# The start of a condition of `filter_rows` that compares a column to a value:
# an operator, or "in" or "not in" as words.
_COMPARISON: re.Pattern[str] = re.compile(r'(==|!=|<=|>=|<|>|in\b|not\s+in\b)')


@dataclasses.dataclass
class DropDuplicates(Cleaner):
    """Removes rows that duplicate an earlier row.

    Wraps:
        [`pandas.DataFrame.drop_duplicates`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Removes duplicate rows, keeping the first.

        Args:
            data: the data to clean.
            columns: columns to compare. Defaults to `None`, which compares
                every column.
            **kwargs: not used.

        Returns:
            The data, without duplicate rows.

        """
        subset = (
            None if columns is None
            else utilities.select_columns(data, columns))
        return data.drop_duplicates(subset = subset)


@dataclasses.dataclass
class DropMissing(Cleaner):
    """Removes rows with missing values.

    Wraps:
        [`pandas.DataFrame.dropna`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Removes rows with a missing value in any of `columns`.

        Args:
            data: the data to clean.
            columns: columns to check. Defaults to `None`, which checks every
                column.
            **kwargs: not used.

        Returns:
            The data, without rows that have missing values.

        """
        subset = (
            None if columns is None
            else utilities.select_columns(data, columns))
        return data.dropna(subset = subset)


@dataclasses.dataclass
class FilterRows(Cleaner):
    """Keeps the rows that match a query, or a condition for each column.

    Each parameter other than "query" is a column, and its value is a
    condition for that column, so "state = not in ['GU', 'PR']" keeps the
    rows whose state is not "GU" or "PR". A condition can start with a
    comparison ("==", "!=", "<", "<=", ">", ">=", "in", or "not in"); "not"
    and a value (as in "not 1") means "!="; a list means "in"; and a value
    alone means "==". Words that are not columns are values, so they need no
    quotes (see `utilities.quote_bare_values`). The rows kept match the query
    and every condition.

    Wraps:
        [`pandas.DataFrame.query`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        query: str | None = None,
        **conditions: Any) -> pd.DataFrame:
        """Keeps the rows that match `query` and every condition.

        Args:
            data: the data to clean.
            query: a `pandas` query, such as "age >= 18". Column names with
                spaces go inside backticks. Defaults to `None`.
            **conditions: a condition for each column, by its name (see the
                class docstring).

        Raises:
            KeyError: if a condition is for a column that is not in the data.

        Returns:
            The rows that match `query` and every condition. If there are
                neither, every row.

        """
        missing = [c for c in conditions if c not in data.columns]
        if missing:
            message = (
                f'the columns {missing} that filter_rows has conditions for '
                f'are not in the data'
            )
            raise KeyError(message)
        parts = [] if query is None else [str(query)]
        parts.extend(_condition_query(c, v) for c, v in conditions.items())
        if not parts:
            return data
        joined = ' and '.join(f'({p})' for p in parts)
        return data.query(utilities.quote_bare_values(joined, data))


""" Private Functions """


def _condition_query(column: str, condition: Any) -> str:
    """Returns a `pandas` query of `condition` for `column`.

    Args:
        column: name of the column.
        condition: a comparison and a value (such as "> 5" or "not in [1,
            2]"), "not" and a value (which means "!="), a list (which means
            "in"), or a value alone (which means "==").

    Returns:
        The query, with the column's name in backticks.

    """
    name = f'`{column}`'
    if isinstance(condition, list | tuple | set):
        return f'{name} in {list(condition)!r}'
    if not isinstance(condition, str):
        return f'{name} == {condition!r}'
    text = condition.strip()
    if _COMPARISON.match(text):
        return f'{name} {text}'
    if text.startswith('not '):
        return f'{name} != {text[4:].strip()}'
    return f'{name} == {text}'
