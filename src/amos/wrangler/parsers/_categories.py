"""Parsers that turn matches into categories.

Contents:
    ParseCategories: makes the values of every pattern that text matches.
    ParseCategory: makes the value of the first pattern that text matches.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pandas as pd

from ._base import (
    PatternParser,
    Rule,
    _add_to,
    _cell_names,
    _is_blank,
    _keep_first,
    _matches,
    _missing,
    _settle,
)


@dataclasses.dataclass
class ParseCategories(PatternParser):
    """Lists the value of every pattern that text matches, in order.

    Each value is added once, after the values of earlier patterns (or
    rules), separated by the rule's separator (a comma and a space unless it
    is set), so a value cannot have the separator in it. For example, rules
    that write "refund" and "complaint" make "refund, complaint" for an email
    that asks for a refund and complains.

    """

    """ Class Methods """

    @classmethod
    def check_value(cls, rule: Rule) -> str:
        """Returns the value of a rule, which cannot have its separator.

        Args:
            rule: the rule, with its value as it was given.

        Raises:
            ValueError: if the value is blank or has the separator in it
                (without the spaces around it, such as a comma).

        Returns:
            The value, as text trimmed of spaces.

        """
        if _is_blank(rule.value):
            message = (
                f'a {cls.kind()} rule needs a value to write when it matches')
            raise ValueError(message)
        text = str(rule.value).strip()
        mark = rule.separator.strip() or rule.separator
        if mark and mark in text:
            message = (
                f'the value {text!r} has {mark!r}, which separates the values '
                f'of {cls.kind()} rules'
            )
            raise ValueError(message)
        return text

    """ Public Methods """

    def find(
        self,
        rule: Rule,
        texts: Mapping[str, pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Adds the rule's value to each row where the pattern appears.

        Args:
            rule: the search.
            texts: the columns of text to search, by name.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        name = rule.names[0]
        found = _missing(len(index))
        found[_matches(rule, texts)] = str(rule.value)
        if name in made:
            before = pd.Series(
                made[name].to_numpy(dtype = object), dtype = 'str').fillna('')
            mark = rule.separator
            item = f'{mark}{rule.value}{mark}'
            listed = (mark + before + mark).str.contains(
                item, regex = False).to_numpy(dtype = bool)
            found[listed] = None
        _add_to(made, name, found, index, rule.separator)


@dataclasses.dataclass
class ParseCategory(PatternParser):
    """Gives each text the value of the first pattern that it matches.

    With several patterns (or rules of the same name), the first that matches
    wins, so they can turn text into categories, such as the name of a state
    into its abbreviation. Text that matches none is missing, unless the parser
    is given a `default`. The type of the column is found from its values, as
    `pandas` does.

    """

    """ Class Methods """

    @classmethod
    def check_value(cls, rule: Rule) -> Any:
        """Returns the value of a rule, as a number if it is one.

        Args:
            rule: the rule, with its value as it was given.

        Raises:
            ValueError: if the value is blank.

        Returns:
            The value: an `int` or `float` if it is text of a number, and
                otherwise the value itself.

        """
        if _is_blank(rule.value):
            message = (
                f'a {cls.kind()} rule needs a value to write when it matches')
            raise ValueError(message)
        return _convert(rule.value)

    """ Public Methods """

    def find(
        self,
        rule: Rule,
        texts: Mapping[str, pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes the rule's value in each row where the pattern appears.

        Args:
            rule: the search.
            texts: the columns of text to search, by name.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        found = _missing(len(index))
        found[_matches(rule, texts)] = rule.value
        _keep_first(made, rule.names[0], found, index)

    def parse(
        self,
        data: pd.DataFrame,
        *,
        default: Any = None,
        **kwargs: Any) -> pd.DataFrame:
        """Makes the value of the first pattern that the text matches.

        Args:
            data: the data to parse.
            default: the value of text that matches no pattern (including
                missing text). Defaults to `None`, which is missing.
            **kwargs: the parameters of `PatternParser.parse`.

        Returns:
            The data, with the values. Their type is found from the values
                (and `default`), as `pandas` does.

        """
        data = super().parse(data, **kwargs)
        if default is None or kwargs.get('pattern') is None:
            return data
        name = _cell_names(kwargs.get('name'))[0]
        values = data[name].to_numpy(dtype = object).copy()
        values[pd.isna(values)] = default
        data[name] = _settle(values, data.index)
        return data


""" Private Functions """


def _convert(value: Any) -> Any:
    """Returns the text of a cell as a number if it is one.

    Args:
        value: the cell.

    Returns:
        An `int` or `float` if the cell is text of a number, the text
            otherwise, or the cell itself if it is not text.

    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text
