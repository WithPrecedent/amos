"""Parsers that find labeled sections and names in text.

Contents:
    _NAME_PUNCTUATION: the periods and apostrophes (straight and curly) that
        `names` rules remove from names.
    ParseFields: makes a column for each labeled section of text.
    ParseNames: finds the names in text.

"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import pandas as pd

from .. import _patterns
from ._base import PatternParser, Rule, _body, _keep_first, _text_column, _wrap

# The curly apostrophe (U+2019) is added with `chr` so that the source file
# has only plain characters.
_NAME_PUNCTUATION: re.Pattern[str] = re.compile("[.'" + chr(0x2019) + ']')


@dataclasses.dataclass
class ParseFields(PatternParser):
    """Makes a column for each labeled section of text.

    The pattern matches the labels (such as "MEDICAL HISTORY:"), and each
    column has the text after a label, up to the next label. The columns are
    named for the labels in snake case ("medical_history"), after the `name`
    and an underscore if there is one. A label of only punctuation is left
    out. The first text of each label in each row is kept.

    """

    nameless: ClassVar[bool] = True

    """ Class Methods """

    @classmethod
    def check_pattern(cls, rule: Rule) -> None:
        """Checks that the pattern has no named groups, since it is used twice.

        Args:
            rule: the rule, whose value is not yet checked.

        Raises:
            ValueError: if the pattern has named groups.

        """
        if rule.pattern.groupindex:
            message = (
                f'the pattern of a {cls.kind()} rule cannot have named '
                f'groups, since it is used twice'
            )
            raise ValueError(message)

    """ Public Methods """

    def find(
        self,
        rule: Rule,
        texts: Mapping[str, pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes a column for each label, with the text after it (text).

        Args:
            rule: the search.
            texts: the columns of text to search, by name, in order.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        label = _body(rule.pattern)
        source = _wrap(
            rule.pattern,
            '(?P<_label>',
            f')(?P<_text>(?s:.*?))(?=(?:{label})|\\Z)')
        prefix = f'{rule.names[0]}_' if rule.names else ''
        for text in texts.values():
            plain = text.reset_index(drop = True)
            matches = plain.str.extractall(source, flags = rule.flags)
            if matches.empty:
                continue
            keys = [_field_name(str(k)) for k in matches['_label']]
            table = pd.DataFrame({
                'row': matches.index.get_level_values(0),
                'key': [f'{prefix}{k}' for k in keys],
                'text': _patterns.tidy(
                    matches['_text'].astype('str')).to_numpy(dtype = object)})
            # A label of only punctuation has no name to give a column.
            table = table[[bool(k) for k in keys]]
            # The first text of each label in each row is kept.
            wide = table.pivot_table(
                index = 'row',
                columns = 'key',
                values = 'text',
                aggfunc = 'first',
                dropna = False)
            fields = list(dict.fromkeys(table['key']))
            wide = wide.reindex(index = range(len(plain)), columns = fields)
            for field in fields:
                _keep_first(
                    made, field, wide[field].to_numpy(dtype = object), index)


@dataclasses.dataclass
class ParseNames(PatternParser):
    """Finds the names in text.

    The text is split wherever the pattern matches, so the pattern should
    match the words and punctuation between names (such as "AUTHORS", "BY",
    "AND", and commas). The names are made upper-case without periods or
    apostrophes (so "O'Brien" is "OBRIEN"), line breaks always
    separate them, and pieces without two letters are left out. Each name is
    kept once, and the names are separated by the rule's separator (a comma
    and a space unless it is set).

    """

    def find(
        self,
        rule: Rule,
        texts: Mapping[str, pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes the names in each row, once each, with the separator.

        The names are found row by row, since each piece of text is
        cleaned on its own.

        Args:
            rule: the search.
            texts: the columns of text to search, by name, in order.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        name = rule.names[0]
        lists: list[list[str]] = [[] for _ in range(len(index))]
        if name in made:
            earlier = made[name].to_numpy(dtype = object)
            for position, before in enumerate(earlier):
                if not pd.isna(before):
                    lists[position] = str(before).split(rule.separator)
        for text in texts.values():
            for position, value in enumerate(text.to_numpy(dtype = object)):
                if pd.isna(value):
                    continue
                listed = lists[position]
                listed.extend(
                    n for n in _split_names(str(value), rule.pattern)
                    if n not in listed)
        joined = np.array(
            [rule.separator.join(n) if n else None for n in lists],
            dtype = object)
        made[name] = _text_column(joined, index)


""" Private Functions """


def _field_name(label: str) -> str:
    """Returns the name of the column of a label that a `fields` rule found.

    Args:
        label: the label, such as "MEDICAL HISTORY:".

    Returns:
        The label in snake case, such as "medical_history", or an empty `str`
            if it has no letters or digits.

    """
    return re.sub(r'\W+', '_', label.strip().lower()).strip('_')


def _split_names(text: str, separators: re.Pattern[str]) -> list[str]:
    """Splits `text` into names wherever `separators` matches.

    The text is made upper-case and periods and apostrophes are removed first
    (so "O'Brien" becomes "OBRIEN"), and line breaks always separate
    names. Pieces without at least two letters are dropped.

    Args:
        text: text with names, such as "Written by SMITH, JONES, and
            BROWN."
        separators: a pattern that matches the words and punctuation between
            the names. It is matched against the upper-case text.

    Returns:
        The names, in order, without repeats.

    """
    cleaned = _NAME_PUNCTUATION.sub('', text.upper())
    names: list[str] = []
    for line in cleaned.splitlines():
        # The text between matches is used (rather than `re.split`) so that
        # groups in the pattern are not mistaken for names.
        pieces, start = [], 0
        for match in separators.finditer(line):
            pieces.append(line[start:match.start()])
            start = match.end()
        pieces.append(line[start:])
        for piece in pieces:
            name = ' '.join(piece.split())
            if sum(c.isalpha() for c in name) >= 2 and name not in names:  # noqa: PLR2004
                names.append(name)
    return names
