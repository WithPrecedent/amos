"""Cleaners that turn text into numbers, dates, or booleans.

Contents:
    _FALSE, _TRUE: words (in lower case) that `CleanBooleans` reads as false
        and as true.
    CleanBooleans: turns text such as "yes" and "no" into booleans.
    CleanDates: turns text into dates.
    CleanNumbers: turns text such as "$1,234.50" into numbers.

"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from ...core import utilities
from ._base import Cleaner

# Words (in lower case) that `CleanBooleans` reads as false and as true.
_FALSE: tuple[str, ...] = ('false', 'f', 'no', 'n', '0')
_TRUE: tuple[str, ...] = ('true', 't', 'yes', 'y', '1')


@dataclasses.dataclass
class CleanBooleans(Cleaner):
    """Turns text such as "yes" and "no" into booleans.

    Text is compared without regard to case or the spaces around it. Numbers
    are true if they are 1 and false if they are 0. Anything else is missing,
    so the columns have the `pandas` "boolean" type, which allows missing
    values.

    """

    def clean(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        true_values: Sequence[Any] = _TRUE,
        false_values: Sequence[Any] = _FALSE,
        **kwargs: Any) -> pd.DataFrame:
        """Turns the values in `columns` into booleans.

        Args:
            data: the data to clean.
            columns: columns to change. Defaults to `None`, which changes
                nothing.
            true_values: words that mean true. Defaults to "true", "t", "yes",
                "y", and "1".
            false_values: words that mean false. Defaults to "false", "f",
                "no", "n", and "0".
            **kwargs: not used.

        Returns:
            The data, with booleans. Columns that are already boolean are not
                changed.

        """
        trues = _words(true_values)
        falses = _words(false_values)
        for column in utilities.select_columns(data, columns or []):
            values = data[column]
            if pd.api.types.is_bool_dtype(values.dtype):
                continue
            if pd.api.types.is_numeric_dtype(values.dtype):
                true = (values == 1).to_numpy(dtype = bool)
                false = (values == 0).to_numpy(dtype = bool)
            else:
                words = values.astype('str').str.strip().str.lower()
                true = words.isin(trues).to_numpy(dtype = bool)
                false = words.isin(falses).to_numpy(dtype = bool)
            flags = pd.array(true, dtype = 'boolean')
            flags[~(true | false)] = pd.NA
            data[column] = pd.Series(flags, index = values.index)
        return data


@dataclasses.dataclass
class CleanDates(Cleaner):
    """Turns text into dates.

    Text that is not a date becomes a missing value, rather than an error (as
    it would with `convert_types`).

    Wraps:
        [`pandas.to_datetime`](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        date_format: str | None = None,
        dayfirst: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        """Turns the text in `columns` into dates.

        Args:
            data: the data to clean.
            columns: columns to change. Defaults to `None`, which changes
                nothing.
            date_format: the format of the dates, with the codes of Python's
                `strftime` (such as "%m/%d/%Y"), or "mixed" to find the
                format of each date separately (which is slower). Defaults to
                `None`, in which case `pandas` finds the format from the first
                date and uses it for all of them.
            dayfirst: whether the day comes before the month in dates such as
                "01/02/2020". Defaults to `False`.
            **kwargs: not used.

        Returns:
            The data, with dates.

        """
        for column in utilities.select_columns(data, columns or []):
            values = data[column]
            if isinstance(values.dtype, pd.CategoricalDtype):
                values = values.astype('str')
            data[column] = pd.to_datetime(
                values,
                errors = 'coerce',
                format = date_format,
                dayfirst = dayfirst)
        return data


@dataclasses.dataclass
class CleanNumbers(Cleaner):
    """Turns text such as "$1,234.50" into numbers.

    The first number in the text is kept, so "$1,234.50" becomes 1234.5,
    "12%" becomes 12, and "about 3 years" becomes 3. Text without a number
    becomes a missing value.

    Wraps:
        [`pandas.to_numeric`](https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html)
        from pandas.

    """

    def clean(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        thousands: str = ',',
        decimal: str = '.',
        **kwargs: Any) -> pd.DataFrame:
        """Turns the text in `columns` into numbers.

        Args:
            data: the data to clean.
            columns: columns to change. Defaults to `None`, which changes
                nothing.
            thousands: the character that separates thousands. Defaults to
                ",". Set it to "." (and `decimal` to ",") for numbers such as
                "1.234,50".
            decimal: the character before the decimals. Defaults to ".".
            **kwargs: not used.

        Raises:
            ValueError: if `thousands` and `decimal` are the same.

        Returns:
            The data, with numbers. Columns that are already numbers are not
                changed.

        """
        if thousands == decimal:
            message = 'the thousands and decimal separators must differ'
            raise ValueError(message)
        digits = r'\d+'
        if thousands:
            digits += rf'(?:{re.escape(thousands)}\d{{3}})*'
        point = re.escape(decimal)
        pattern = (
            rf'([-+]?(?:{digits}(?:{point}\d*)?|{point}\d+)'
            rf'(?:[eE][-+]?\d+)?)')
        for column in utilities.select_columns(data, columns or []):
            values = data[column]
            if pd.api.types.is_numeric_dtype(values.dtype):
                continue
            found = values.astype('str').str.extract(pattern, expand = False)
            if thousands:
                found = found.str.replace(thousands, '', regex = False)
            if decimal != '.':
                found = found.str.replace(decimal, '.', regex = False)
            data[column] = pd.to_numeric(found, errors = 'coerce')
        return data


""" Private Functions """


def _words(values: Sequence[Any] | Any) -> list[str]:
    """Returns words that mean true or false, in lower case.

    Args:
        values: a word or a sequence of them. Numbers and booleans (as a
            settings file may give them) are made into text.

    Returns:
        The words, trimmed and in lower case.

    """
    items = [values] if isinstance(values, str) or not isinstance(
        values, Iterable) else list(values)
    return [str(v).strip().lower() for v in items]
