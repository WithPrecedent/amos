"""Parsers that flag, count, keep, replace, or split matches.

Contents:
    ParseCounts: counts the matches of a pattern in text.
    ParseFlags: flags whether text matches a pattern.
    ParseMatch: keeps the first match of a pattern in text.
    ParseMatches: keeps every match of a pattern in text.
    ParseReplace: replaces the matches of a pattern in text.
    ParseSplit: splits text into columns where a pattern matches.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import pandas as pd

from .. import _patterns
from ._base import (
    PatternParser,
    Rule,
    _add_to,
    _fill,
    _join,
    _keep_first,
    _matches,
    _missing,
    _wrap,
)


@dataclasses.dataclass
class ParseCounts(PatternParser):
    """Counts the matches of a pattern in text.

    The counts of several columns, patterns, or rules of the same name are
    added together. Missing text has no matches.

    Wraps:
        [`pandas.Series.str.count`](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.count.html)
        from pandas.

    """

    def find(
        self,
        rule: Rule,
        texts: Mapping[str, pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes the number of matches in each row (whole numbers).

        Args:
            rule: the search.
            texts: the columns of text to search, by name.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        name = rule.names[0]
        counts = np.zeros(len(index), dtype = 'int64')
        for text in texts.values():
            found = text.str.count(rule.pattern.pattern, flags = rule.flags)
            counts += found.fillna(0).to_numpy(dtype = 'int64')
        if name in made:
            counts += made[name].to_numpy(dtype = 'int64')
        made[name] = pd.Series(counts, index = index)


@dataclasses.dataclass
class ParseFlags(PatternParser):
    """Flags whether text matches a pattern.

    For example, the pattern "revers|vacat" makes a column that is true for
    each row whose text mentions a reversal or vacatur. With several columns,
    patterns, or rules of the same name, it is true if any matches. Missing
    text is false.

    Wraps:
        [`pandas.Series.str.contains`](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.contains.html)
        from pandas.

    """

    def find(
        self,
        rule: Rule,
        texts: Mapping[str, pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes whether the pattern appears in each row (booleans).

        Args:
            rule: the search.
            texts: the columns of text to search, by name.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        name = rule.names[0]
        found = _matches(rule, texts)
        if name in made:
            found |= made[name].to_numpy(dtype = bool)
        made[name] = pd.Series(found, index = index)


@dataclasses.dataclass
class ParseMatch(PatternParser):
    r"""Keeps the first match of a pattern in text (or the groups of a match).

    The whole match is kept, even if the pattern has groups (which often just
    group alternatives, as in "(F|U\.S)\."). A pattern with named groups
    makes a column for each named group instead, named for the group after
    the `name` and an underscore if there is one. For example,
    "(?P<volume>\d+) F\.3d (?P<page>\d+)" makes "volume" and "page" columns
    from citations such as "512 F.3d 1093". With several columns, the first
    that matches in a row is used, and with several patterns (or rules of the
    same name), the first that matches wins. Text without a match is missing.

    Wraps:
        [`pandas.Series.str.extract`](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.extract.html)
        from pandas.

    """

    """ Class Methods """

    @classmethod
    def check_names(cls, rule: Rule) -> None:
        """Checks the names of a rule, which are optional with named groups.

        Args:
            rule: the rule, whose value is not yet checked.

        Raises:
            ValueError: if the rule has several names, or none and a pattern
                without named groups.

        """
        _check_group_names(cls, rule)

    """ Public Methods """

    def find(
        self,
        rule: Rule,
        texts: Mapping[str, pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes the first match (or its named groups) in each row (text).

        Args:
            rule: the search.
            texts: the columns of text to search, by name, in order.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        source, targets = _extract_targets(rule)
        found = {key: _missing(len(index)) for key in targets}
        for text in texts.values():
            matches = text.str.extract(source, flags = rule.flags)
            # Rows that an earlier column matched keep all of its groups.
            empty = np.ones(len(index), dtype = bool)
            for values in found.values():
                empty &= pd.isna(values)
            for key, values in found.items():
                values[empty] = _strings(matches[key])[empty]
        for key, column in targets.items():
            _keep_first(made, column, found[key], index)


@dataclasses.dataclass
class ParseMatches(PatternParser):
    """Keeps every match of a pattern in text (or the groups of each match).

    The whole matches are kept, even if the pattern has groups, and they are
    separated by the rule's separator (a comma and a space unless it is set).
    A pattern with named groups makes a column for each named group instead,
    as `parse_match` does. The matches of several columns, patterns, or rules
    of the same name are joined in order. Text without a match is missing.

    Wraps:
        [`pandas.Series.str.extractall`](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.extractall.html)
        from pandas.

    """

    """ Class Methods """

    @classmethod
    def check_names(cls, rule: Rule) -> None:
        """Checks the names of a rule, which are optional with named groups.

        Args:
            rule: the rule, whose value is not yet checked.

        Raises:
            ValueError: if the rule has several names, or none and a pattern
                without named groups.

        """
        _check_group_names(cls, rule)

    """ Public Methods """

    def find(
        self,
        rule: Rule,
        texts: Mapping[str, pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes every match (or its named groups) in each row (text).

        Args:
            rule: the search.
            texts: the columns of text to search, by name, in order.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        source, targets = _extract_targets(rule)
        found = {key: _missing(len(index)) for key in targets}
        for text in texts.values():
            plain = text.reset_index(drop = True)
            matches = plain.str.extractall(source, flags = rule.flags)
            if matches.empty:
                continue
            for key in targets:
                kept = _patterns.tidy(matches[key].astype('str')).dropna()
                joined = kept.groupby(level = 0).agg(rule.separator.join)
                found[key] = _join(
                    found[key],
                    joined.reindex(range(len(plain))).to_numpy(dtype = object),
                    rule.separator)
        for key, column in targets.items():
            _add_to(made, column, found[key], index, rule.separator)


@dataclasses.dataclass
class ParseReplace(PatternParser):
    r"""Replaces the matches of a pattern in text.

    The value replaces every match, and it can include the groups of the
    match ("\1" is the first). A blank value removes the matches. With a
    name, the changed text is a new column (so only one column can be
    searched); without one, the text that is searched is changed.

    Wraps:
        [`pandas.Series.str.replace`](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.replace.html)
        from pandas.

    """

    nameless: ClassVar[bool] = True

    """ Class Methods """

    @classmethod
    def check_names(cls, rule: Rule) -> None:
        """Checks that a rule with a name changes only one column.

        Args:
            rule: the rule, whose value is not yet checked.

        Raises:
            ValueError: if the rule has several names, or a name and several
                columns.

        """
        super().check_names(rule)
        if rule.names and len(rule.columns) > 1:
            message = (
                f'a {cls.kind()} rule with a name changes one column: leave '
                f'the name blank to change each column in place'
            )
            raise ValueError(message)

    @classmethod
    def check_value(cls, rule: Rule) -> str:
        """Returns the text that replaces the matches.

        Args:
            rule: the rule, with its value as it was given.

        Returns:
            The value as text, which is empty if it is `None`.

        """
        return '' if rule.value is None else str(rule.value)

    """ Public Methods """

    def find(
        self,
        rule: Rule,
        texts: Mapping[str, pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:  # noqa: ARG002
        """Replaces every match in the text with the rule's value.

        The changed text goes in the rule's name, if it has one, and
        otherwise replaces the text it searched.

        Args:
            rule: the search.
            texts: the columns of text to change, by name.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows (not used, since the changed text
                keeps the index of the text).

        """
        for column, text in texts.items():
            changed = text.str.replace(
                rule.pattern.pattern,
                str(rule.value),
                regex = True,
                flags = rule.flags)
            made[rule.names[0] if rule.names else column] = changed


@dataclasses.dataclass
class ParseSplit(PatternParser):
    r"""Splits text into columns where a pattern matches.

    For example, the pattern "\sv\.\s" and the names "party1, party2" split
    a caption such as "Smith v. Jones" into "Smith" and "Jones". With more
    names, the text is split at the first matches, so the last column has the
    rest of the text. Text without a match is all in the first column. With
    several columns, the first with text in a row is split. Each piece is
    trimmed of spaces, and an empty piece is missing.

    """

    """ Class Methods """

    @classmethod
    def check_names(cls, rule: Rule) -> None:
        """Checks that a rule names at least two columns.

        Args:
            rule: the rule, whose value is not yet checked.

        Raises:
            ValueError: if the rule has fewer than two names.

        """
        if len(rule.names) < 2:  # noqa: PLR2004
            message = (
                f'a {cls.kind()} rule needs two or more names, separated by '
                f'commas (such as "party1, party2")'
            )
            raise ValueError(message)

    """ Public Methods """

    def find(
        self,
        rule: Rule,
        texts: Mapping[str, pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes the pieces of the text between matches (text).

        Args:
            rule: the search.
            texts: the columns of text, by name, in order.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        rest = _missing(len(index))
        for text in texts.values():
            _fill(rest, text.to_numpy(dtype = object))
        source = _wrap(rule.pattern, r'(?s:\A(.*?))(?:', r')(?s:(.*)\Z)')
        pieces = []
        # Each pass splits the rest of the text at its first match.
        for _ in rule.names[:-1]:
            parts = pd.Series(rest, dtype = 'str').str.extract(
                source, flags = rule.flags)
            before = parts.iloc[:, 0].to_numpy(dtype = object)
            after = parts.iloc[:, -1].to_numpy(dtype = object)
            unmatched = pd.isna(before) & ~pd.isna(rest)
            before[unmatched] = rest[unmatched]
            pieces.append(before)
            rest = after
        pieces.append(rest)
        for name, piece in zip(rule.names, pieces, strict = True):
            tidied = _patterns.tidy(pd.Series(piece, dtype = 'str'))
            _keep_first(made, name, tidied.to_numpy(dtype = object), index)


""" Private Functions """


def _check_group_names(kind: type[PatternParser], rule: Rule) -> None:
    """Checks the names of a rule that keeps matches or their named groups.

    A pattern with named groups makes a column for each group, so the rule
    needs no name (but can have one, which starts the names of the columns).

    Args:
        kind: the parser of the rule, for messages.
        rule: the rule.

    Raises:
        ValueError: if the rule has several names, or none and a pattern
            without named groups.

    """
    if len(rule.names) > 1:
        message = (
            f'a {kind.kind()} rule makes one column (or one for each named '
            f'group), not {list(rule.names)}'
        )
        raise ValueError(message)
    if not rule.names and not rule.pattern.groupindex:
        message = (
            'the name is blank (only a pattern with named groups, such as '
            '"(?P<year>\\d{4})", needs none)'
        )
        raise ValueError(message)


def _extract_targets(rule: Rule) -> tuple[str, dict[str | int, str]]:
    """Returns the pattern to extract for a rule and the columns it makes.

    Args:
        rule: a rule that keeps matches, or the named groups of matches.

    Returns:
        The pattern for `Series.str.extract`, and the column that each of
            its groups makes, by the key of the group in what `pandas`
            extracts: the whole match (0) as the rule's name, or each named
            group as its name, after the rule's name and an underscore if it
            has one.

    """
    groups = rule.pattern.groupindex
    if not groups:
        return _wrap(rule.pattern, '(', ')'), {0: rule.names[0]}
    prefix = f'{rule.names[0]}_' if rule.names else ''
    targets: dict[str | int, str] = {g: f'{prefix}{g}' for g in groups}
    return _wrap(rule.pattern, '(?:', ')'), targets


def _strings(values: pd.Series) -> np.ndarray:
    """Returns text trimmed of spaces, as an array with `None` for missing.

    Args:
        values: a column of text.

    Returns:
        The trimmed text, with empty text missing.

    """
    tidied = _patterns.tidy(values.astype('str'))
    return tidied.to_numpy(dtype = object)
