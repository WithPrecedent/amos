r"""Techniques that turn the text in a dataset into data.

These are techniques of the "wrangler" stage of a project. A loader brings
documents into a dataset as they are (`load_files` makes a row for each file,
with its text in a "text" column), and parsers then find information in that
text and make columns of it, such as whether a contract can be renewed, the
city in an address, or the authors of a paper. Like a munger, a parser keeps
the rows and every column. Each search covers every row at once with the
vectorized methods of `pandas` (such as those of `Series.str`), so parsing is
fast even with many documents.

Each kind of search is a parser of its own, which a workflow can name in its
settings with a pattern and the columns to search and make (see
`PatternParser.parse`). Several values in one cell are separated by a comma
and a space unless the `separator` is set:

| Parser | Kind | Makes |
| --- | --- | --- |
| `parse_flags` | `flags` | True if the pattern appears, and otherwise false. |
| `parse_counts` | `counts` | The number of times the pattern appears. |
| `parse_category` | `category` | The `value` if the pattern appears. With several patterns (or rules), the first that matches wins, so they can turn text into categories (such as the name of a state into its abbreviation). |
| `parse_categories` | `categories` | The `value` of every pattern (or rule) that appears, once each, in order. A value cannot have the separator in it. |
| `parse_match` | `match` | The first match, or a column for each named group of the pattern (such as "(?P<year>\d{4})"), named for the group after the `name` if there is one. |
| `parse_matches` | `matches` | Every match, or a column for each named group as `parse_match` makes. |
| `parse_split` | `split` | A column for each name: the text before the first match, between the first and second, and so on, with the rest of the text in the last. Text without a match is all in the first. |
| `parse_names` | `names` | The names in the text, found by splitting the text wherever the pattern matches. The pattern should match the words and punctuation between names (such as "AUTHORS", "BY", "AND", and commas). The names are made upper-case without periods or apostrophes (so "O'Brien" is "OBRIEN"), line breaks always separate them, and pieces without two letters are left out. |
| `parse_fields` | `fields` | A column for each label that the pattern matches (such as "MEDICAL HISTORY:"), with the text after the label, up to the next label. The columns are named for the labels in snake case ("medical_history"), after the `name` if there is one. |
| `parse_replace` | `replace` | The text with every match replaced by the `value`, which can include the groups of the match ("\1" is the first). A blank `value` removes the matches. Without a `name`, the text it searches is changed. |

`parse_all` applies a rulebook: an ordered list of rules, each of which is
one of these searches, named by its kind. Rules are usually rows of CSV files
(spreadsheets), so they can be read and changed without knowing Python. They
can also be `dict`s in settings, with the same keys. The columns are:

| Column | Meaning |
| --- | --- |
| `column` | The columns of text to search, separated by commas. Defaults to the parser's "column" (such as "text") if blank. |
| `name` | The column that the rule makes (or adds to). A `split` rule names two or more, separated by commas. It is blank for a `replace` rule that changes the text it searches, and optional for a `fields` rule (whose columns are named for the labels in the text) and for a `match` or `matches` rule whose pattern has named groups. |
| `kind` | What the rule does: one of the kinds above, or one registered with `register_rule_kind`. |
| `pattern` | A regular expression to search for. |
| `value` | What a `category`, `categories`, or `replace` rule writes. Otherwise blank. |
| `flags` | Any of "ignorecase", "dotall" (so "." also matches line breaks), "multiline" (so "^" and "$" match at each line), and "verbose", separated by commas. Optional. |
| `separator` | What separates several values in one cell, for `categories`, `matches`, and `names` rules. Optional: a comma and a space if blank. |
| `note` | Anything you want to say about the rule. It is not used. Optional. |

Rules are applied from top to bottom, and every column that a rule makes can
be searched by the rules after it, so a rule can find a section of a
document (with `match`, `split`, or `fields`) for later rules to search.
Rules that make the same column are combined: flags are true if any rule
matches, counts are added, the first `category` or `match` that matches is
kept, and the results of `categories`, `matches`, and `names` rules are
joined. A text that is missing matches nothing. A parser given several
patterns combines them in the same way, as rules of the same name.

The kinds that this module first used (`flag`, `count`, `map`, `map_all`,
`extract`, and `extract_all`) are read as the kinds they became. CSV files in
an older layout of rules (with "target" for `column`, "variable" for `name`,
TRUE or FALSE in "ignorecase" and "dotall" columns, and the kinds "label",
"labels", "section", "matches", "excerpts", and "remove") are read too, as
the rules they stand for.

To write a parser, subclass `Parser` and write `parse`. To write a parser
that can also be a kind of rule, subclass `PatternParser`, write `find`, and
name the kind with `register_rule_kind`.

Contents:
    Parser: base class for techniques that turn text into data.
    PatternParser: base class for parsers that search text for a pattern,
        each of which can be a kind of rule.
    ParseAll: finds information in text with a rulebook.
    ParseCategories: makes the values of every pattern that text matches.
    ParseCategory: makes the value of the first pattern that text matches.
    ParseCounts: counts the matches of a pattern in text.
    ParseFields: makes a column for each labeled section of text.
    ParseFlags: flags whether text matches a pattern.
    ParseMatch: keeps the first match of a pattern in text.
    ParseMatches: keeps every match of a pattern in text.
    ParseNames: finds the names in text.
    ParseReplace: replaces the matches of a pattern in text.
    ParseSplit: splits text into columns where a pattern matches.
    Rule: one search of text, as a row of a rulebook.
    Rulebook: an ordered collection of rules.
    register_rule_kind: names a pattern parser as a kind of rule.
    register_rulebook: registers the rules of a rulebook under a name.
    _COLUMN_ALIASES: names of the columns of rules in the older layout (and
        of keys that settings may use), mapped to the names of this layout.
    _FLAGS: the flags that a rule can have, by their names.
    _KIND_ALIASES: kinds of rules in the older layout and the first layout
        of this module, mapped to the kinds they stand for.
    _NAME_PUNCTUATION: the periods and apostrophes (straight and curly) that
        `names` rules remove from names.
    _OLD_FLAGS: columns of flags in the older layout, which hold TRUE or
        FALSE.
    _REQUIRED: columns that every file of rules must have.
    _RULEBOOKS: the sources of each registered rulebook, by its name.
    _RULE_KINDS: the pattern parser of each kind of rule, by the kind.
    _SEPARATOR: what separates several values in one cell of the data.
    _TRUE: words (in lower case) that mean true in the older flag columns.

"""

from __future__ import annotations

import abc
import csv
import dataclasses
import pathlib
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, ClassVar

import camina
import numpy as np
import pandas as pd

from . import _patterns, base

_COLUMN_ALIASES: dict[str, str] = {
    'columns': 'column',
    'names': 'name',
    'target': 'column',
    'variable': 'name'}
_FLAGS: dict[str, re.RegexFlag] = {
    'dotall': re.DOTALL,
    'ignorecase': re.IGNORECASE,
    'multiline': re.MULTILINE,
    'verbose': re.VERBOSE}
_KIND_ALIASES: dict[str, str] = {
    'count': 'counts',
    'excerpts': 'matches',
    'extract': 'match',
    'extract_all': 'matches',
    'flag': 'flags',
    'label': 'category',
    'labels': 'categories',
    'map': 'category',
    'map_all': 'categories',
    'remove': 'replace',
    'section': 'match'}
# The curly apostrophe (U+2019) is added with `chr` so that the source file
# has only plain characters.
_NAME_PUNCTUATION: re.Pattern[str] = re.compile("[.'" + chr(0x2019) + ']')
_OLD_FLAGS: tuple[str, ...] = ('dotall', 'ignorecase')
_REQUIRED: tuple[str, ...] = ('kind', 'pattern')
_RULEBOOKS: dict[str, tuple[Any, ...]] = {}
# The built-in kinds are added after their parsers are defined.
_RULE_KINDS: dict[str, type[PatternParser]] = {}
_SEPARATOR: str = ', '
_TRUE: frozenset[str] = frozenset({'true', 't', 'yes', 'y', '1'})


@dataclasses.dataclass
class Parser(base.Operation, abc.ABC):
    """Base class for techniques that turn the text in a dataset into data.

    A subclass writes a `parse` method, which takes the data and returns it
    with changed or new columns, and the same rows. The dataset's `history`
    records the columns whose values or types changed ("changed") and the new
    columns ("created").

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used by most parsers. Defaults to `None`.
        parameters: keyword arguments for `parse`. Defaults to an empty
            `dict`.

    """

    """ Required Methods """

    @abc.abstractmethod
    def parse(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Returns `data` with the information found in its text.

        Args:
            data: the data to parse. It may be changed in place.
            **kwargs: parameters for parsing.

        Returns:
            The data, with changed or new columns, and the same rows.

        """

    """ Public Methods """

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Parses the text of `item`.

        Args:
            item: the dataset to parse.
            **kwargs: parameters for `parse`.

        Raises:
            ValueError: if `parse` changed the rows or removed columns.

        Returns:
            The changed dataset.

        """
        after = self.parse(item.data.copy(), **kwargs)
        return base._change_columns(item, after, self.name, 'parser')


@dataclasses.dataclass
class PatternParser(Parser, abc.ABC):
    """Base class for parsers that search text for a pattern.

    Each pattern parser is one kind of search, which a workflow can apply on
    its own (with the parameters of `parse`) or name as the kind of a rule in
    a rulebook (see `ParseAll` and `register_rule_kind`). Either way, the
    search is described by a `Rule`, which `find` applies.

    A subclass writes `find`. It can also change what a rule needs: set
    `nameless` if the search can make columns without a name, and override
    `check_names`, `check_pattern`, or `check_value` to check the names or
    pattern of a rule, or to check (and convert) its value.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used. Defaults to `None`.
        parameters: keyword arguments for `parse`. Defaults to an empty
            `dict`.

    """

    # Whether a rule of this kind can leave its name blank.
    nameless: ClassVar[bool] = False

    """ Required Methods """

    @abc.abstractmethod
    def find(
        self,
        rule: Rule,
        texts: Mapping[str, pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Applies `rule` to every row of `texts`.

        The columns that the search makes are put in `made`, rather than in
        the data, so that every column can be added at once. A column of the
        same name that an earlier rule made is in `made` too, and the search
        can add to it (as flags and counts do) or replace it.

        Args:
            rule: the search: its `pattern`, the `names` of the columns to
                make, its `value`, and the `separator` of several values.
            texts: the columns of text to search, by name, in order. Their
                index is that of the data.
            made: the columns that earlier rules made (or changed), by name.
                It is changed in place.
            index: the labels of the rows, for the columns that are made.

        """

    """ Class Methods """

    @classmethod
    def check_names(cls, rule: Rule) -> None:
        """Checks that a rule has as many names as the search needs.

        By default, a rule makes one column, so it needs one name (or none,
        if the class is `nameless`).

        Args:
            rule: the rule, whose value is not yet checked.

        Raises:
            ValueError: if the rule has too many or too few names.

        """
        if not rule.names and not cls.nameless:
            message = 'the name is blank'
            raise ValueError(message)
        if len(rule.names) > 1:
            message = (
                f'a {cls.kind()} rule makes one column, not '
                f'{list(rule.names)}')
            raise ValueError(message)

    @classmethod
    def check_pattern(cls, rule: Rule) -> None:
        """Checks that the search can use the pattern of `rule`.

        By default, any valid pattern can be used.

        Args:
            rule: the rule, whose value is not yet checked.

        """

    @classmethod
    def check_value(cls, rule: Rule) -> Any:
        """Returns the value of a rule, checked for the search.

        By default, the search uses no value.

        Args:
            rule: the rule, with its value as it was given (from a cell of a
                CSV file, settings, or a parameter), or `None`.

        Raises:
            ValueError: if the rule has a value.

        Returns:
            `None`.

        """
        if not _is_blank(rule.value):
            message = (
                f'a {cls.kind()} rule does not use a value, but it has '
                f'{rule.value!r} (is its kind right?)'
            )
            raise ValueError(message)
        return None

    @classmethod
    def kind(cls) -> str:
        """Returns the kind of rule that this parser is.

        Returns:
            The kind that the class is registered as (see
                `register_rule_kind`), or the snake case name of the class
                without "parse_" if it is not registered.

        """
        for kind, parser in _RULE_KINDS.items():
            if parser is cls:
                return kind
        return str(camina.snakify(cls.__name__)).removeprefix('parse_')

    @classmethod
    def make_rule(
        cls,
        pattern: Any,
        *,
        columns: Any = None,
        names: Any = None,
        value: Any = None,
        flags: Any = None,
        separator: str | None = None,
        note: str = '',
        origin: str = '',
        kind: str = '') -> Rule:
        """Returns a rule that this parser applies, checked for it.

        Args:
            pattern: a regular expression, as text or compiled (in which case
                its flags are kept).
            columns: the columns of text to search: a name, names separated
                by commas, or a sequence of names. Defaults to `None`, in
                which case the rule searches the column it is given.
            names: the columns to make, in the same forms as `columns`.
                Defaults to `None`.
            value: what the rule writes, if it uses a value. Defaults to
                `None`.
            flags: names of flags (such as "ignorecase, dotall"), as text or
                a sequence, or flags of the `re` module. Defaults to `None`.
            separator: what separates several values in one cell. Defaults
                to `None`, which is a comma and a space.
            note: a description of the rule. Defaults to an empty `str`.
            origin: where the rule came from, for messages. Defaults to an
                empty `str`.
            kind: the kind of the rule. Defaults to an empty `str`, which
                uses `kind`.

        Raises:
            ValueError: if the pattern is blank or not a valid regular
                expression, a flag is not known, or the names, pattern, or
                value do not suit the search.

        Returns:
            The rule.

        """
        rule = Rule(
            kind = kind or cls.kind(),
            pattern = _compile(pattern, _flags_of(flags)),
            columns = _cell_names(columns),
            names = _cell_names(names),
            value = value,
            separator = _SEPARATOR if separator is None else str(separator),
            note = note,
            origin = origin,
            parser = cls)
        cls.check_names(rule)
        cls.check_pattern(rule)
        return dataclasses.replace(rule, value = cls.check_value(rule))

    """ Public Methods """

    def parse(
        self,
        data: pd.DataFrame,
        *,
        pattern: Any = None,
        column: str | Sequence[str] | None = 'text',
        name: str | Sequence[str] | None = None,
        value: Any = None,
        flags: str | Sequence[str] | int | None = None,
        separator: str | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Searches the text of `data` for `pattern`.

        Args:
            data: the data to parse.
            pattern: a regular expression, a sequence of them, or a mapping
                of them to their values (for the kinds that use a value).
                Several patterns are applied in order as rules of the same
                name, and so combined as rules are. Defaults to `None`, which
                changes nothing.
            column: the columns of text to search, as a name, names separated
                by commas, or a sequence of names, or `None` (or "*") to
                search every column of text that is not categorical. Defaults
                to "text", the column of text that `load_files` makes.
            name: the columns to make, in the same forms as `column`.
                Defaults to `None`, which suits only the kinds that need no
                name.
            value: what the search writes, for the kinds that use a value,
                if `pattern` does not map patterns to values. Defaults to
                `None`.
            flags: names of flags (such as "ignorecase, dotall"), as text or
                a sequence, or flags of the `re` module. Defaults to `None`.
            separator: what separates several values in one cell, for the
                kinds that make them. Defaults to `None`, which is a comma
                and a space.
            **kwargs: not used.

        Raises:
            KeyError: if a column to search is not in the data.
            TypeError: if a column to search is not text.
            ValueError: if a pattern is not valid, `pattern` maps patterns
                to values and `value` is given too, or the names or value do
                not suit the search.

        Returns:
            The data, with the columns that the search makes.

        """
        if pattern is None:
            return data
        if column is None or column == '*':
            column = _patterns.text_columns(data)
            if not column:
                return data
        rules = [
            self.make_rule(
                source, columns = column, names = name, value = written,
                flags = flags, separator = separator)
            for source, written in _pattern_values(pattern, value)]
        return _apply_rules(data, rules, parser = self)


@dataclasses.dataclass
class ParseAll(Parser):
    """Finds information in text with a rulebook.

    The rules can be the name of a registered rulebook (see
    `register_rulebook`), the path of a CSV file of rules or of a folder of
    them, a list of rules (as `dict`s, with the columns of a CSV file as
    keys), or a list of any of these, which are applied in order. Each rule
    is applied by the pattern parser of its kind (see `register_rule_kind`).

    """

    def parse(
        self,
        data: pd.DataFrame,
        *,
        rules: Any = None,
        column: str = 'text',
        **kwargs: Any) -> pd.DataFrame:
        """Applies the rules to the text of `data`.

        Args:
            data: the data to parse.
            rules: the rules to apply (see `Rulebook.create`). Defaults to
                `None`, which changes nothing.
            column: the column of text that rules with a blank `column`
                search. Defaults to "text", the column of text that
                `load_files` makes.
            **kwargs: not used.

        Returns:
            The data, with the columns that the rules make.

        """
        if rules is None or (
            isinstance(rules, Sequence) and not isinstance(rules, str)
            and not rules):
            return data
        return Rulebook.create(rules).apply(data, column = column)


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


_RULE_KINDS.update({
    'categories': ParseCategories,
    'category': ParseCategory,
    'counts': ParseCounts,
    'fields': ParseFields,
    'flags': ParseFlags,
    'match': ParseMatch,
    'matches': ParseMatches,
    'names': ParseNames,
    'replace': ParseReplace,
    'split': ParseSplit})


@dataclasses.dataclass(frozen = True)
class Rule:
    """One search of text, as a row of a rulebook.

    Args:
        kind: the kind of the rule, such as "flags".
        pattern: the compiled regular expression, with its flags.
        columns: names of the columns of text to search, in order. Defaults
            to an empty `tuple`, in which case the parser's "column" is
            searched.
        names: names of the columns that the rule makes. Defaults to an
            empty `tuple` (for a `replace` rule that changes the text it
            searches, or a `fields` rule without a prefix).
        value: for a kind that uses a value (such as `category`), what it
            writes. Defaults to `None`.
        separator: what separates several values in one cell, for the kinds
            that make them (such as `matches`). Defaults to a comma and a
            space.
        note: a description of the rule. Defaults to an empty `str`.
        origin: where the rule came from (such as a file and row), for
            messages. Defaults to an empty `str`.
        parser: the pattern parser that applies the rule. Defaults to `None`,
            in which case it is the parser registered for `kind`.

    """

    kind: str
    pattern: re.Pattern[str]
    columns: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    value: Any = None
    separator: str = _SEPARATOR
    note: str = ''
    origin: str = ''
    parser: type[PatternParser] | None = None

    """ Class Methods """

    @classmethod
    def create(cls, row: Mapping[str, Any], origin: str = '') -> Rule:
        """Returns a rule made from a row of a CSV file (or a `dict`).

        Args:
            row: the cells of the row, by the names of the columns, in this
                module's layout or the older one.
            origin: where the row came from, for messages. Defaults to an
                empty `str`.

        Raises:
            ValueError: if the kind or a flag is not known, the pattern is
                blank or not a valid regular expression, the rule has too
                many or too few names, or it has a value that it does not
                use (or lacks one that it needs).

        Returns:
            The rule.

        """
        cells = _translate(row)
        kind = str(cells.get('kind') or '').strip().lower()
        if kind not in _RULE_KINDS:
            message = f'the kind {kind!r} is not one of {sorted(_RULE_KINDS)}'
            raise ValueError(message)
        return _RULE_KINDS[kind].make_rule(
            cells.get('pattern'),
            columns = cells.get('column'),
            names = cells.get('name'),
            value = cells.get('value'),
            flags = cells.get('flags'),
            # A blank cell is the usual separator.
            separator = cells.get('separator') or None,
            note = str(cells.get('note') or ''),
            origin = origin,
            kind = kind)

    """ Properties """

    @property
    def description(self) -> str:
        """Returns a short description of the rule, for messages.

        Returns:
            The kind of the rule, the columns it makes, and where it came
                from.

        """
        text = f'the {self.kind} rule'
        if self.names:
            text += f' for {_SEPARATOR.join(self.names)}'
        if self.origin:
            text += f' ({self.origin})'
        return text

    @property
    def flags(self) -> int:
        """Returns the flags of the pattern, for the methods of `pandas`.

        Returns:
            The flags, without those that every pattern of text has.

        """
        return self.pattern.flags & ~re.UNICODE

    """ Public Methods """

    def apply(
        self,
        data: pd.DataFrame,
        made: dict[str, pd.Series],
        *,
        column: str = 'text',
        parser: PatternParser | None = None) -> None:
        """Applies the rule to every row of `data`.

        The columns that the rule makes are put in `made`, rather than in
        `data`, so that they can all be added at once (see `Rulebook.apply`).

        Args:
            data: the data, which is not changed.
            made: the columns that earlier rules made (or changed), by name,
                which this rule searches, adds to, or replaces. It is changed
                in place.
            column: the column of text to search if the rule has no
                `columns`. Defaults to "text".
            parser: the parser that applies the rule. Defaults to `None`, in
                which case it is an instance of `parser` or of the parser
                registered for `kind`.

        Raises:
            ValueError: if no parser applies the rule.

        """
        if parser is None:
            kind = self.parser or _RULE_KINDS.get(self.kind)
            if kind is None:
                message = (
                    f'{self.description}: no parser is registered for the '
                    f'kind {self.kind!r}'
                )
                raise ValueError(message)
            parser = kind()
        columns = self.columns or (column,)
        texts = {c: _text(data, made, c, self.description) for c in columns}
        parser.find(self, texts, made, data.index)


@dataclasses.dataclass
class Rulebook:
    """An ordered collection of rules.

    Args:
        rules: the rules, in the order they are applied. Defaults to an empty
            `list`.
        name: a name for the rulebook, such as the files it came from.
            Defaults to an empty `str`.

    """

    rules: list[Rule] = dataclasses.field(default_factory = list)
    name: str = ''

    """ Class Methods """

    @classmethod
    def create(cls, *sources: Any) -> Rulebook:
        """Returns a rulebook with the rules of every source, in order.

        Args:
            *sources: names of registered rulebooks (see
                `register_rulebook`), paths of CSV files of rules or of
                folders of them (whose files are read in alphabetical order),
                rules as `dict`s (with the columns of a CSV file as keys),
                `Rule`s, `Rulebook`s, or sequences of any of these. A text
                with commas that is not a file or registered name is a list
                of them. A file that two sources name is read only the first
                time.

        Raises:
            FileNotFoundError: if a source is not a registered name, a file,
                or a folder with CSV files.
            ValueError: if a file is missing a column or a rule is not valid.
                The message says which file and row (or which rule).

        Returns:
            The rulebook.

        """
        rules: list[Rule] = []
        names: list[str] = []
        read: set[pathlib.Path] = set()
        for source in sources:
            found, labels = _gather(source, read, ())
            rules.extend(found)
            names.extend(labels)
        return cls(rules = rules, name = _SEPARATOR.join(dict.fromkeys(names)))

    """ Properties """

    @property
    def columns(self) -> list[str]:
        """Returns the columns of text that the rules name to search.

        Returns:
            The `columns` of every rule, in order, without repeats.

        """
        return list(dict.fromkeys(c for r in self.rules for c in r.columns))

    @property
    def names(self) -> list[str]:
        """Returns the columns that the rules name to make.

        Returns:
            The `names` of every rule, in order, without repeats. The
                columns of `fields` rules are named for the labels in the
                text, so only their prefixes are listed.

        """
        return list(dict.fromkeys(n for r in self.rules for n in r.names))

    """ Public Methods """

    def apply(self, data: pd.DataFrame, *, column: str = 'text') -> pd.DataFrame:
        """Applies every rule, in order, to every row of `data`.

        Args:
            data: the data, which is not changed.
            column: the column of text that rules with no `columns` search.
                Defaults to "text".

        Raises:
            KeyError: if a rule searches a column that is not in the data.
            TypeError: if a rule searches a column that is not text.

        Returns:
            The data, with the columns that the rules make (in the order
                they were first made) and the text that `replace` rules
                changed.

        """
        return _apply_rules(data, self.rules, column = column)

    """ Dunder Methods """

    def __add__(self, other: Rulebook) -> Rulebook:
        """Returns a rulebook with the rules of both rulebooks.

        Args:
            other: the rulebook whose rules come second.

        Returns:
            A new rulebook with this rulebook's rules followed by those of
                `other`, named for both.

        """
        name = _SEPARATOR.join(n for n in (self.name, other.name) if n)
        return Rulebook(rules = [*self.rules, *other.rules], name = name)

    def __iter__(self) -> Iterator[Rule]:
        """Returns an iterator of the rules.

        Returns:
            An iterator of `rules`, in the order they are applied.

        """
        return iter(self.rules)

    def __len__(self) -> int:
        """Returns the number of rules.

        Returns:
            The length of `rules`.

        """
        return len(self.rules)


""" Public Functions """


def register_rule_kind(kind: str, parser: type[PatternParser]) -> None:
    """Names a pattern parser as a kind of rule.

    Rulebooks (see `parse_all`) can then name `kind` in the "kind" column of
    a rule, and the rule is applied by `parser`. A kind that is registered
    again gets the new parser, and a kind that has the name of an old kind
    (such as "map") takes its place.

    Args:
        kind: the name of the kind, which is made lower-case.
        parser: a subclass of `PatternParser`.

    Raises:
        TypeError: if `parser` is not a subclass of `PatternParser`.
        ValueError: if `kind` is blank.

    """
    name = str(kind).strip().lower()
    if not name:
        message = 'a kind of rule needs a name'
        raise ValueError(message)
    if not (isinstance(parser, type) and issubclass(parser, PatternParser)):
        message = (
            f'a kind of rule needs a subclass of PatternParser, not '
            f'{parser!r}'
        )
        raise TypeError(message)
    _RULE_KINDS[name] = parser


def register_rulebook(name: str, sources: Any) -> None:
    """Registers the rules of a rulebook under `name`.

    `parse_all` (and `Rulebook.create`) then accept `name` in place of its
    sources, so a package built on `amos` can keep its rules in its own CSV
    files and every project can name them in its settings (as in "rules =
    federal"). The sources are read each time the rulebook is used, so
    changes to its files take effect at once. A name that is registered
    again gets the new sources.

    Args:
        name: the name of the rulebook.
        sources: what `Rulebook.create` accepts: paths of CSV files or
            folders, names of other registered rulebooks, rules, or a
            sequence of them, in the order they are applied.

    Raises:
        ValueError: if `name` is blank.

    """
    if not str(name).strip():
        message = 'a rulebook needs a name'
        raise ValueError(message)
    if isinstance(sources, str | pathlib.Path | Mapping | Rule | Rulebook):
        sources = (sources,)
    _RULEBOOKS[str(name).strip()] = tuple(sources)


""" Private Functions """


def _add_to(
    made: dict[str, pd.Series],
    name: str,
    found: np.ndarray,
    index: pd.Index,
    separator: str = _SEPARATOR) -> None:
    """Puts text in a column, after any that earlier rules put there.

    Args:
        made: the columns that rules made, by name. It is changed in place.
        name: the column.
        found: the text of each row (or `None`).
        index: the labels of the rows.
        separator: what goes between the earlier text and the new. Defaults
            to a comma and a space.

    """
    if name in made:
        found = _join(made[name].to_numpy(dtype = object), found, separator)
    made[name] = _text_column(found, index)


def _apply_rules(
    data: pd.DataFrame,
    rules: Iterable[Rule],
    *,
    column: str = 'text',
    parser: PatternParser | None = None) -> pd.DataFrame:
    """Applies `rules`, in order, to every row of `data`.

    Args:
        data: the data, which is not changed.
        rules: the rules.
        column: the column of text that rules with no `columns` search.
            Defaults to "text".
        parser: the parser that applies every rule. Defaults to `None`, in
            which case each rule is applied by its own parser.

    Raises:
        KeyError: if a rule searches a column that is not in the data.
        TypeError: if a rule searches a column that is not text.

    Returns:
        The data, with the columns that the rules make (in the order they
            were first made) and the text that they changed.

    """
    made: dict[str, pd.Series] = {}
    for rule in rules:
        try:
            rule.apply(data, made, column = column, parser = parser)
        except (KeyError, TypeError) as error:
            detail = error.args[0] if error.args else str(error)
            message = f'{rule.description}: {detail}'
            raise type(error)(message) from error
    return _assemble(data, made)


def _assemble(
    data: pd.DataFrame,
    made: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Returns `data` with the columns that rules made or changed.

    The new columns are added all at once, since adding hundreds of columns
    one at a time (as a rulebook of many rules would) is slow.

    Args:
        data: the data.
        made: the columns that rules made or changed, by name, in the order
            they were made.

    Returns:
        The data, with its changed columns replaced and the new columns
            after them.

    """
    if not made:
        return data
    plain = data.reset_index(drop = True)
    new: dict[str, pd.Series] = {}
    for name, values in made.items():
        column = values.reset_index(drop = True)
        if name in plain.columns:
            plain[name] = column
        else:
            new[name] = column
    if new:
        plain = pd.concat([plain, pd.DataFrame(new)], axis = 1)
    plain.index = data.index
    return plain


def _body(pattern: re.Pattern[str]) -> str:
    """Returns the text of `pattern` without the inline flags at its start.

    A verbose pattern ends with a line break, so that a comment at its end
    does not hide what follows it.

    Args:
        pattern: the compiled pattern.

    Returns:
        The text of the pattern.

    """
    text = pattern.pattern
    match = _patterns.LEADING_FLAGS.match(text)
    start = match.end() if match else 0
    end = '\n' if pattern.flags & re.VERBOSE else ''
    return f'{text[start:]}{end}'


def _cell_names(cell: Any) -> tuple[str, ...]:
    """Returns the names in a cell, which separates them with commas.

    Args:
        cell: the cell: text, a sequence of names (as settings may give
            them), or blank.

    Returns:
        The names, trimmed of spaces.

    """
    if cell is None:
        return ()
    items = [cell] if isinstance(cell, str) or not isinstance(
        cell, Iterable) else list(cell)
    names = [
        part.strip() for item in items for part in str(item).split(',')]
    return tuple(n for n in names if n)


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


def _compile(pattern: Any, flags: int) -> re.Pattern[str]:
    """Returns a pattern of a rule compiled with `flags`.

    Args:
        pattern: a regular expression, as text or compiled (in which case
            its own flags are added to `flags`).
        flags: flags of the `re` module.

    Raises:
        ValueError: if the pattern is blank or not a valid regular
            expression.

    Returns:
        The compiled pattern.

    """
    if isinstance(pattern, re.Pattern):
        flags |= pattern.flags & ~re.UNICODE
        pattern = pattern.pattern
    if pattern is None or not str(pattern):
        message = 'the pattern is blank'
        raise ValueError(message)
    return _patterns.compile_pattern(pattern, flags)


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


def _field_name(label: str) -> str:
    """Returns the name of the column of a label that a `fields` rule found.

    Args:
        label: the label, such as "MEDICAL HISTORY:".

    Returns:
        The label in snake case, such as "medical_history", or an empty `str`
            if it has no letters or digits.

    """
    return re.sub(r'\W+', '_', label.strip().lower()).strip('_')


def _fill(found: np.ndarray, other: np.ndarray) -> None:
    """Fills the missing values of `found` with those of `other`.

    Args:
        found: values (or `None`). It is changed in place.
        other: values (or `None`) for the same rows.

    """
    gaps = pd.isna(found) & ~pd.isna(other)
    found[gaps] = other[gaps]


def _flags_of(cell: Any) -> int:
    """Returns the flags of the `re` module named in a cell.

    Args:
        cell: names of flags (such as "ignorecase, dotall"), separated by
            commas or spaces, a sequence of names, flags of the `re` module,
            or blank.

    Raises:
        ValueError: if a name is not one of `_FLAGS`.

    Returns:
        The flags, combined.

    """
    if isinstance(cell, int) and not isinstance(cell, bool):
        return cell
    flags = 0
    for item in _cell_names(cell):
        for word in item.lower().split():
            if word not in _FLAGS:
                message = (
                    f'the flag {word!r} is not one of {sorted(_FLAGS)}')
                raise ValueError(message)
            flags |= _FLAGS[word]
    return flags


def _gather(
    source: Any,
    read: set[pathlib.Path],
    registered: tuple[str, ...]) -> tuple[list[Rule], list[str]]:
    """Returns the rules of a source of `Rulebook.create`, and its names.

    Args:
        source: a source (see `Rulebook.create`).
        read: the files read so far, which are not read again. It is changed
            in place.
        registered: the names of the registered rulebooks that led to
            `source`, so that a rulebook that names itself is caught.

    Raises:
        FileNotFoundError: if a source is not a registered name, a file, or a
            folder with CSV files.
        ValueError: if a registered rulebook names itself, a file is missing
            a column, or a rule is not valid.

    Returns:
        The rules, in order, and names for them (such as the stems of their
            files).

    """
    if isinstance(source, Rulebook):
        return list(source.rules), [source.name] if source.name else []
    if isinstance(source, Rule):
        return [source], []
    if isinstance(source, Mapping):
        return [_rule(source, 'a rule of the settings')], []
    if isinstance(source, str) and source.strip() in _RULEBOOKS:
        name = source.strip()
        if name in registered:
            message = f'the rulebook {name!r} names itself'
            raise ValueError(message)
        rules: list[Rule] = []
        for part in _RULEBOOKS[name]:
            found, _ = _gather(part, read, (*registered, name))
            rules.extend(found)
        return rules, [name]
    if isinstance(source, str | pathlib.Path):
        path = pathlib.Path(source)
        if isinstance(source, str) and ',' in source and not path.exists():
            return _gather(_cell_names(source), read, registered)
        return _read_rules(path, read)
    rules, names = [], []
    for position, part in enumerate(source):
        if isinstance(part, Mapping):
            rules.append(_rule(part, f'rule {position + 1}'))
            continue
        found, labels = _gather(part, read, registered)
        rules.extend(found)
        names.extend(labels)
    return rules, names


def _is_blank(cell: Any) -> bool:
    """Returns whether a cell is blank.

    Args:
        cell: the cell.

    Returns:
        Whether the cell is `None` or text of only spaces.

    """
    return cell is None or (isinstance(cell, str) and not cell.strip())


def _join(
    first: np.ndarray,
    second: np.ndarray,
    separator: str = _SEPARATOR) -> np.ndarray:
    """Returns two columns of text joined in each row, with `separator`.

    Args:
        first: text (or `None`) for each row.
        second: text (or `None`) for the same rows.
        separator: what goes between them. Defaults to a comma and a space.

    Returns:
        The text of both where both have text, and otherwise the text of
            whichever has it (or `None`).

    """
    result = first.copy()
    has_first = ~pd.isna(first)
    has_second = ~pd.isna(second)
    result[~has_first] = second[~has_first]
    both = has_first & has_second
    if both.any():
        joined = (
            pd.Series(first[both], dtype = 'str') + separator
            + pd.Series(second[both], dtype = 'str'))
        result[both] = joined.to_numpy(dtype = object)
    return result


def _keep_first(
    made: dict[str, pd.Series],
    name: str,
    found: np.ndarray,
    index: pd.Index) -> None:
    """Puts values in a column, where earlier rules put none.

    Text is stored as text, and other values in the type that `pandas` finds
    for them.

    Args:
        made: the columns that rules made, by name. It is changed in place.
        name: the column.
        found: the value of each row (or `None`).
        index: the labels of the rows.

    """
    if name in made:
        before = made[name].to_numpy(dtype = object).copy()
        _fill(before, found)
        found = before
    made[name] = _settle(found, index)


def _matches(rule: Rule, texts: Mapping[str, pd.Series]) -> np.ndarray:
    """Returns whether the rule's pattern appears in any of `texts`.

    Args:
        rule: the rule.
        texts: the columns of text to search, by name.

    Returns:
        A boolean array, with a value for each row, which is false where
            every text is missing.

    """
    columns = list(texts.values())
    found = np.zeros(len(columns[0]), dtype = bool)
    for text in columns:
        found |= _patterns.contains(text, rule.pattern.pattern, rule.flags)
    return found


def _missing(count: int) -> np.ndarray:
    """Returns an array of `count` missing values.

    Args:
        count: the number of rows.

    Returns:
        An array of `None`, of objects.

    """
    return np.full(count, None, dtype = object)


def _pattern_values(pattern: Any, value: Any) -> list[tuple[Any, Any]]:
    """Returns the patterns that a parser is given, with their values.

    Args:
        pattern: a pattern, a sequence of them, or a mapping of them to their
            values.
        value: the value of every pattern, unless `pattern` is a mapping.

    Raises:
        ValueError: if `pattern` is a mapping and `value` is given too.

    Returns:
        Each pattern and its value, in order.

    """
    if isinstance(pattern, Mapping):
        if value is not None:
            message = (
                'the values are given twice: in the mapping of patterns and '
                'in "value"'
            )
            raise ValueError(message)
        return list(pattern.items())
    if isinstance(pattern, str | re.Pattern):
        return [(pattern, value)]
    return [(p, value) for p in pattern]


def _read_rules(
    path: pathlib.Path,
    read: set[pathlib.Path]) -> tuple[list[Rule], list[str]]:
    """Returns the rules in a CSV file or a folder of them.

    Args:
        path: path of a CSV file or a folder of them.
        read: the files read so far, which are not read again. It is changed
            in place.

    Raises:
        FileNotFoundError: if `path` is not a file or a folder with CSV
            files.
        ValueError: if a file is missing a column or a row is not a valid
            rule.

    Returns:
        The rules, in order, and the stem of each file.

    """
    if path.is_dir():
        files = sorted(path.glob('*.csv'))
        if not files:
            message = f'there are no CSV files of rules in {path}'
            raise FileNotFoundError(message)
    elif path.is_file():
        files = [path]
    else:
        message = (
            f'{str(path)!r} is not a file of rules, a folder of them, or a '
            f'registered rulebook ({sorted(_RULEBOOKS)})'
        )
        raise FileNotFoundError(message)
    rules: list[Rule] = []
    names: list[str] = []
    for file in files:
        if file.resolve() in read:
            continue
        read.add(file.resolve())
        rules.extend(_read_file(file))
        names.append(file.stem)
    return rules, names


def _read_file(path: pathlib.Path) -> list[Rule]:
    """Returns the rules in the CSV file at `path`.

    Args:
        path: path to a CSV file of rules, in UTF-8 (with or without a byte
            order mark, as Excel saves them).

    Raises:
        ValueError: if the file is missing a column or a row is not a valid
            rule.

    Returns:
        The rules, in order.

    """
    with path.open(encoding = 'utf-8-sig', newline = '') as file:
        reader = csv.DictReader(file)
        header = [
            _COLUMN_ALIASES.get(c.strip().lower(), c.strip().lower())
            for c in reader.fieldnames or []]
        missing = [c for c in _REQUIRED if c not in header]
        if missing:
            message = f'{path} is missing the columns {missing}'
            raise ValueError(message)
        rules = []
        for row in reader:
            cells = {k: v for k, v in row.items() if k is not None}
            if not any(str(v or '').strip() for v in cells.values()):
                continue
            # The reader skips blank lines, so the row is the number of the
            # last line it read (as a spreadsheet numbers it).
            rules.append(_rule(cells, f'{path.name}, row {reader.line_num}'))
    return rules


def _rule(cells: Mapping[str, Any], origin: str) -> Rule:
    """Returns a rule, with where it came from in any error.

    Args:
        cells: the cells of the rule.
        origin: where the rule came from, such as a file and row.

    Raises:
        ValueError: if the rule is not valid. The message starts with
            `origin`.

    Returns:
        The rule.

    """
    try:
        return Rule.create(cells, origin = origin)
    except ValueError as error:
        message = f'{origin}: {error}'
        raise ValueError(message) from error


def _settle(values: np.ndarray, index: pd.Index) -> pd.Series:
    """Returns values as a column, in the type that suits them.

    Args:
        values: the value of each row (or `None`).
        index: the labels of the rows.

    Returns:
        A column of text if every value that is not missing is text (or if
            every value is missing), and otherwise one of the type that
            `pandas` finds for them.

    """
    present = values[~pd.isna(values)]
    if all(isinstance(v, str) for v in present):
        return _text_column(values, index)
    return pd.Series(values, index = index).infer_objects()


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


def _strings(values: pd.Series) -> np.ndarray:
    """Returns text trimmed of spaces, as an array with `None` for missing.

    Args:
        values: a column of text.

    Returns:
        The trimmed text, with empty text missing.

    """
    tidied = _patterns.tidy(values.astype('str'))
    return tidied.to_numpy(dtype = object)


def _text_column(values: np.ndarray, index: pd.Index) -> pd.Series:
    """Returns text as a column of the `pandas` text type.

    Args:
        values: the text of each row (or `None`).
        index: the labels of the rows.

    Returns:
        The column, whose missing values are missing text.

    """
    return pd.Series(values, index = index, dtype = 'str')


def _text(
    data: pd.DataFrame,
    made: Mapping[str, pd.Series],
    column: str,
    rule: str) -> pd.Series:
    """Returns a column of text for a rule to search.

    Args:
        data: the data.
        made: the columns that earlier rules made or changed, by name, which
            take the place of those of `data`.
        column: the name of the column.
        rule: a description of the rule, for messages.

    Raises:
        KeyError: if `column` is neither made nor in `data`.
        TypeError: if `column` does not hold text.

    Returns:
        The column. A categorical column of text is returned as text.

    """
    source = made[column].to_frame(column) if column in made else data
    return _patterns.text_of(source, column, rule)


def _translate(row: Mapping[str, Any]) -> dict[str, Any]:
    """Returns the cells of a rule in this module's layout.

    The columns of the older layout ("target", "variable", and TRUE or FALSE
    in "ignorecase" and "dotall") become those they stand for, and old kinds
    (see `_KIND_ALIASES`) become the kinds they stand for, unless a kind of
    the same name is registered. A "remove" rule replaces its matches with a
    space, as the older layout did.

    Args:
        row: the cells of a rule, by the names of their columns.

    Returns:
        The cells, by the names of the columns of this layout.

    """
    cells: dict[str, Any] = {}
    for key, value in row.items():
        name = str(key).strip().lower()
        cells.setdefault(_COLUMN_ALIASES.get(name, name), value)
    flags = list(_cell_names(cells.get('flags')))
    for name in _OLD_FLAGS:
        if str(cells.pop(name, '') or '').strip().lower() in _TRUE:
            flags.append(name)
    cells['flags'] = flags
    kind = str(cells.get('kind') or '').strip().lower()
    if kind in _KIND_ALIASES and kind not in _RULE_KINDS:
        cells['kind'] = _KIND_ALIASES[kind]
        if kind == 'remove' and not str(cells.get('value') or ''):
            cells['value'] = ' '
    return cells


def _wrap(pattern: re.Pattern[str], before: str, after: str) -> str:
    """Returns the text of `pattern` between `before` and `after`.

    Inline flags at the start of the pattern (such as "(?i)") stay at the
    start, and a verbose pattern ends with a line break, so that a comment at
    its end does not hide `after`.

    Args:
        pattern: the compiled pattern.
        before: text to put before the pattern (after its inline flags).
        after: text to put after the pattern.

    Returns:
        The new pattern, as text.

    """
    text = pattern.pattern
    match = _patterns.LEADING_FLAGS.match(text)
    start = match.end() if match else 0
    return f'{text[:start]}{before}{_body(pattern)}{after}'
