"""A merger that matches rows by the nearest value.

Contents:
    _DIRECTIONS: what `merge_nearest` calls each direction to look in, with what
        `pandas` calls it.
    MergeNearest: adds columns from another table's row with the nearest date or
        number.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass
from ._base import _POSITION, _ROW, _VALUE, Merger, _column, _keys, _measures

# What `merge_nearest` calls each direction to look in, with what `pandas`
# calls it.
_DIRECTIONS: dict[str, str] = {
    'after': 'forward', 'before': 'backward', 'nearest': 'nearest'}


@dataclasses.dataclass
class MergeNearest(Merger):
    """Adds columns from another table's row with the nearest date or number.

    For example, each trade gets the most recent price of its stock before
    the time of the trade, from a table with a row for each price. The row
    can also be limited to those with the same keys (here, the stock).

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


""" Private Functions """


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
