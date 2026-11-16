"""Mungers that tidy text.

Contents:
    _ACCENTS: combining accents (such as the accent of "é" once it is split into
        "e" and the accent), which `NormalizeText` removes. The string is not
        raw, so that Python makes the escapes into characters: the regular
        expressions of pyarrow, which `pandas` uses for text, do not know the
        escapes.
    _CASES: the cases that `NormalizeText` can change text to, which are the
        names of methods of `Series.str`.
    _PUNCTUATION: anything that is not a letter, digit, underscore, or space.
    _WHITESPACE: one or more spaces, tabs, or new lines.
    NormalizeText: collapses spaces in text, and optionally changes its case and
        removes accents and punctuation.
    StripText: trims spaces from (and optionally lowercases) text.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ...core import utilities
from .._patterns import text_columns as _text_columns
from .._patterns import text_of as _text
from .._patterns import tidy as _tidy
from ._base import Munger

# Combining accents (such as the accent of "é" once it is split into "e" and
# the accent), which `NormalizeText` removes. The string is not raw, so that
# Python makes the escapes into characters: the regular expressions of
# pyarrow, which `pandas` uses for text, do not know the escapes.
_ACCENTS: str = '[\u0300-\u036f]'
# The cases that `NormalizeText` can change text to, which are the names of
# methods of `Series.str`.
_CASES: frozenset[str] = frozenset({'lower', 'title', 'upper'})
# Anything that is not a letter, digit, underscore, or space.
_PUNCTUATION: str = r'[^\w\s]'
_WHITESPACE: str = r'\s+'


@dataclasses.dataclass
class NormalizeText(Munger):
    """Normalizes the spaces, case, accents, and punctuation of text.

    Every run of spaces, tabs, and line breaks becomes one space, and spaces
    are trimmed from both ends. It can also change the case of the text,
    remove accents (so "é" becomes "e"), and remove punctuation (so "U.S."
    becomes "US" and "O'Brien" becomes "OBrien"). This makes text that was
    written differently the same, so it can be counted, compared, or encoded
    as one value.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        case: str | None = None,
        remove_accents: bool = False,
        remove_punctuation: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        """Normalizes the text in `columns`.

        Args:
            data: the data to change.
            columns: columns to change. Defaults to `None`, which changes every
                text column.
            case: "lower", "upper", or "title" (which capitalizes each word).
                Defaults to `None`, which does not change the case.
            remove_accents: whether to remove accents from letters. Defaults to
                `False`.
            remove_punctuation: whether to remove punctuation. Defaults to
                `False`.
            **kwargs: not used.

        Raises:
            ValueError: if `case` is not one of the cases above.

        Returns:
            The data, with normalized text. Empty text becomes a missing value.

        """
        if case is not None and case not in _CASES:
            message = f'case must be one of {sorted(_CASES)}, not {case!r}'
            raise ValueError(message)
        names = _text_columns(data) if columns is None else (
            utilities.select_columns(data, columns))
        for column in names:
            text = _text(data, column, self.name)
            if remove_accents:
                text = text.str.normalize('NFKD').str.replace(
                    _ACCENTS, '', regex = True)
            if remove_punctuation:
                text = text.str.replace(_PUNCTUATION, '', regex = True)
            text = text.str.replace(_WHITESPACE, ' ', regex = True)
            if case is not None:
                text = getattr(text.str, case)()
            data[column] = _tidy(text)
        return data


@dataclasses.dataclass
class StripText(Munger):
    """Trims spaces from text, and optionally makes it lowercase.

    Wraps:
        [`pandas.Series.str.strip`](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.strip.html)
        from pandas.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        lowercase: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        """Trims spaces from the text in `columns`.

        Args:
            data: the data to change.
            columns: columns to change. Defaults to `None`, which changes every
                text column.
            lowercase: whether to make the text lowercase too. Defaults to
                `False`.
            **kwargs: not used.

        Returns:
            The data, with trimmed text. Empty text becomes a missing value.

        """
        if columns is None:
            columns = _text_columns(data)
        for column in utilities.select_columns(data, columns):
            text = data[column].str.strip()
            if lowercase:
                text = text.str.lower()
            data[column] = text.mask(text == '')
        return data
