"""Helpers for the techniques that search text for patterns.

Mungers (see the `mungers` module) and parsers (see the `parsers` module)
both search columns of text with regular expressions, through the vectorized
methods of `Series.str`. These helpers check the patterns and the columns
they search.

Contents:
    compile_pattern: returns a pattern compiled, to check it and count its
        groups.
    contains: returns whether each value of a column of text matches a
        pattern.
    text_columns: returns the names of the columns of text that are not
        categorical.
    text_of: returns a column of text, to search or change.
    tidy: returns text trimmed of spaces, with empty text missing.
    GROUPS_WARNING: the message that `pandas` gives when a pattern with groups
        is only used to find whether text matches.
    LEADING_FLAGS: inline flags (such as "(?i)") at the start of a pattern.

"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..core import utilities

# Inline flags (such as "(?i)") at the start of a pattern, which must stay at
# the start when the rest of the pattern is put in a group.
LEADING_FLAGS: re.Pattern[str] = re.compile(r'(?:\(\?[aiLmsux]+\))*')
# The message that `pandas` gives when a pattern with groups is only used to
# find whether text matches, which is all that `contains` needs.
GROUPS_WARNING: str = 'This pattern is interpreted as a regular expression'


""" Public Functions """


def compile_pattern(pattern: Any, flags: int) -> re.Pattern[str]:
    """Returns `pattern` compiled, to check it and count its groups.

    Args:
        pattern: a regular expression. Anything else (such as a number read
            from a settings file) is made into text.
        flags: flags of the `re` module (such as `re.IGNORECASE`).

    Raises:
        ValueError: if `pattern` is not a valid regular expression.

    Returns:
        The compiled pattern. Its `pattern` attribute is the text of the
            pattern, which `pandas` takes with the same `flags`.

    """
    text = str(pattern)
    try:
        return re.compile(text, flags)
    except re.error as error:
        message = f'the pattern {text!r} is not valid: {error}'
        raise ValueError(message) from error


def contains(text: pd.Series, pattern: str, flags: int) -> np.ndarray:
    """Returns whether each value of `text` matches `pattern`.

    Args:
        text: the text to search.
        pattern: a regular expression.
        flags: flags of the `re` module.

    Returns:
        A boolean array, which is false for missing text.

    """
    with warnings.catch_warnings():
        # Patterns often have groups (such as "(F|U\.S)\."), which `pandas`
        # warns are not kept. Only whether they match matters here.
        warnings.filterwarnings('ignore', GROUPS_WARNING, UserWarning)
        found = text.str.contains(pattern, flags = flags, na = False)
    return found.to_numpy(dtype = bool)


def text_columns(data: pd.DataFrame) -> list[str]:
    """Returns the names of the columns of text that are not categorical.

    Args:
        data: the data.

    Returns:
        The names of the columns.

    """
    return [
        c for c in data.columns
        if pd.api.types.is_string_dtype(data[c].dtype)
        and not isinstance(data[c].dtype, pd.CategoricalDtype)]


def text_of(data: pd.DataFrame, column: Any, technique: str) -> pd.Series:
    """Returns the text in `column`, to search or change.

    Args:
        data: the data.
        column: name of a column of text.
        technique: name of the technique that uses the text, for messages.

    Raises:
        ValueError: if `column` is `None`.
        KeyError: if `column` is not in `data`.
        TypeError: if `column` does not hold text.

    Returns:
        The column. A categorical column of text is returned as text.

    """
    if column is None:
        message = f'{technique!r} needs the name of a column of text'
        raise ValueError(message)
    utilities.select_columns(data, [column])
    values: pd.Series = data[column]
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype(values.cat.categories.dtype)
    if not pd.api.types.is_string_dtype(values.dtype):
        message = f'{technique!r} uses text, but {column!r} is not text'
        raise TypeError(message)
    return values


def tidy(text: pd.Series) -> pd.Series:
    """Returns `text` trimmed of spaces, with empty text missing.

    Args:
        text: a column of text.

    Returns:
        The trimmed text.

    """
    text = text.str.strip()
    return text.mask(text == '')
