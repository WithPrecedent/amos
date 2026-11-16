r"""Techniques that turn the text in a dataset into data.

These are techniques of the "wrangler" stage of a project. A loader brings
documents into a dataset as they are (`load_files` makes a row for each file,
with its text in a "text" column), and parsers then find information in that
text and make columns of it, such as whether an opinion reversed, the court
that decided it, or the judges on its panel. Like a munger, a parser keeps
the rows and every column. Where a munger does one thing to a column, a
parser applies a rulebook: an ordered list of rules, each of which searches
text for a pattern and records what it finds. Each rule searches every row
at once with the vectorized methods of `pandas` (such as those of
`Series.str`), so a rulebook is fast even with many documents.

Rules are usually rows of CSV files (spreadsheets), so they can be read and
changed without knowing Python. They can also be `dict`s in settings, with
the same keys. The columns are:

| Column | Meaning |
| --- | --- |
| `column` | The columns of text to search, separated by commas. Defaults to the parser's "column" (such as "text") if blank. |
| `name` | The column that the rule makes (or adds to). A `split` rule names two, separated by a comma. It is blank for a `replace` rule that changes the text it searches, and optional for a `fields` rule, whose columns are named for the labels in the text. |
| `kind` | What the rule does (see below). |
| `pattern` | A regular expression to search for. |
| `value` | What a `map`, `map_all`, or `replace` rule writes. Otherwise blank. |
| `flags` | Any of "ignorecase", "dotall" (so "." also matches line breaks), "multiline" (so "^" and "$" match at each line), and "verbose", separated by commas. Optional. |
| `note` | Anything you want to say about the rule. It is not used. Optional. |

The kinds of rules:

| Kind | Makes |
| --- | --- |
| `flag` | True if the pattern appears, and otherwise false. |
| `count` | The number of times the pattern appears. |
| `map` | The rule's `value` if the pattern appears. The first rule for a column that matches wins, so rules can turn text into categories (such as the name of a court into its number). |
| `map_all` | The `value` of every rule for the column whose pattern appears, once each, in the order of the rules and separated by commas. A value cannot have a comma. |
| `extract` | The first match. |
| `extract_all` | Every match, separated by commas. |
| `split` | Two columns: the text before the first match and the text after it. Text without a match is all before. |
| `names` | The names in the text, separated by commas, found by splitting the text wherever the pattern matches. The pattern should match the words and punctuation between names (such as "JUDGES", "BEFORE", and commas). The names are made upper-case without periods or apostrophes (so "O'Scannlain" is "OSCANNLAIN"), line breaks always separate them, and pieces without two letters are left out. |
| `fields` | A column for each label that the pattern matches (such as "PRIOR HISTORY:"), with the text after the label, up to the next label. The columns are named for the labels in snake case ("prior_history"), after the rule's `name` if it has one. |
| `replace` | The text with every match replaced by the `value`, which can include the groups of the match ("\1" is the first). A blank `value` removes the matches. |

Rules are applied from top to bottom, and every column that a rule makes can
be searched by the rules after it, so a rule can find a section of a
document (with `extract`, `split`, or `fields`) for later rules to search.
Rules that make the same column are combined: flags are true if any rule
matches, counts are added, the first `map` or `extract` that matches is
kept, and the results of `map_all`, `extract_all`, and `names` rules are
joined. A text that is missing matches nothing.

CSV files in the layout of courtpy (with "target" for `column`, "variable"
for `name`, TRUE or FALSE in "ignorecase" and "dotall" columns, and the kinds
"label", "labels", "section", "matches", "excerpts", and "remove") are read
too, as the rules they stand for.

Contents:
    KINDS: the kinds of rules.
    Parser: base class for techniques that turn text into data.
    ParseDocuments: finds information in text with a rulebook.
    Rule: one rule.
    Rulebook: an ordered collection of rules.
    register_rulebook: registers the rules of a rulebook under a name.
    _COLUMN_ALIASES: names of the columns of rules in courtpy's layout (and
        of keys that settings may use), mapped to the names of this layout.
    _FLAGS: the flags that a rule can have, by their names.
    _KIND_ALIASES: kinds of rules in courtpy's layout, mapped to the kinds
        they stand for.
    _NAMELESS_KINDS: kinds of rules that need no name.
    _NAME_PUNCTUATION: the periods and apostrophes (straight and curly) that
        `names` rules remove from names.
    _OLD_FLAGS: columns of flags in courtpy's layout, which hold TRUE or
        FALSE.
    _REQUIRED: columns that every file of rules must have.
    _RULEBOOKS: the sources of each registered rulebook, by its name.
    _SEPARATOR: what separates several values in one cell of the data.
    _TRUE: words (in lower case) that mean true in courtpy's flag columns.
    _VALUE_KINDS: kinds of rules that use a value.

"""

from __future__ import annotations

import abc
import csv
import dataclasses
import pathlib
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from . import _patterns, base

KINDS: tuple[str, ...] = (
    'count', 'extract', 'extract_all', 'fields', 'flag', 'map', 'map_all',
    'names', 'replace', 'split')
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
    'excerpts': 'extract_all',
    'label': 'map',
    'labels': 'map_all',
    'matches': 'extract_all',
    'remove': 'replace',
    'section': 'extract'}
_NAMELESS_KINDS: frozenset[str] = frozenset({'fields', 'replace'})
# The curly apostrophe (U+2019) is added with `chr` so that the source file
# has only plain characters.
_NAME_PUNCTUATION: re.Pattern[str] = re.compile("[.'" + chr(0x2019) + ']')
_OLD_FLAGS: tuple[str, ...] = ('dotall', 'ignorecase')
_REQUIRED: tuple[str, ...] = ('kind', 'pattern')
_RULEBOOKS: dict[str, tuple[Any, ...]] = {}
_SEPARATOR: str = ', '
_TRUE: frozenset[str] = frozenset({'true', 't', 'yes', 'y', '1'})
_VALUE_KINDS: frozenset[str] = frozenset({'map', 'map_all', 'replace'})


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
class ParseDocuments(Parser):
    """Finds information in text with a rulebook.

    The rules can be the name of a registered rulebook (see
    `register_rulebook`), the path of a CSV file of rules or of a folder of
    them, a list of rules (as `dict`s, with the columns of a CSV file as
    keys), or a list of any of these, which are applied in order.

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


@dataclasses.dataclass(frozen = True)
class Rule:
    """One rule.

    Args:
        kind: one of `KINDS`.
        pattern: the compiled regular expression, with its flags.
        columns: names of the columns of text to search, in order. Defaults
            to an empty `tuple`, in which case the parser's "column" is
            searched.
        names: names of the columns that the rule makes. Only a `split` rule
            has two. Defaults to an empty `tuple` (for a `replace` rule that
            changes the text it searches, or a `fields` rule without a
            prefix).
        value: for a `map`, `map_all`, or `replace` rule, what it writes.
            Defaults to `None`.
        note: a description of the rule. Defaults to an empty `str`.
        origin: where the rule came from (such as a file and row), for
            messages. Defaults to an empty `str`.

    """

    kind: str
    pattern: re.Pattern[str]
    columns: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    value: Any = None
    note: str = ''
    origin: str = ''

    """ Class Methods """

    @classmethod
    def create(cls, row: Mapping[str, Any], origin: str = '') -> Rule:
        """Returns a rule made from a row of a CSV file (or a `dict`).

        Args:
            row: the cells of the row, by the names of the columns, in this
                module's layout or courtpy's.
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
        if kind not in KINDS:
            message = f'the kind {kind!r} is not one of {list(KINDS)}'
            raise ValueError(message)
        columns = _cell_names(cells.get('column'))
        names = _cell_names(cells.get('name'))
        _check_names(kind, columns, names)
        source = cells.get('pattern')
        if source is None or not str(source):
            message = 'the pattern is blank'
            raise ValueError(message)
        pattern = _patterns.compile_pattern(
            source, _flags_of(cells.get('flags')))
        if kind == 'fields' and pattern.groupindex:
            message = (
                'the pattern of a fields rule cannot have named groups, since '
                'it is used twice'
            )
            raise ValueError(message)
        return cls(
            kind = kind,
            pattern = pattern,
            columns = columns,
            names = names,
            value = _value_of(kind, cells.get('value')),
            note = str(cells.get('note') or ''),
            origin = origin)

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

    """ Public Methods """

    def apply(
        self,
        data: pd.DataFrame,
        made: dict[str, pd.Series],
        *,
        column: str = 'text') -> None:
        """Applies the rule to every row of `data`.

        The columns that the rule makes are put in `made`, rather than in
        `data`, so that a rulebook can add them all at once (see
        `Rulebook.apply`).

        Args:
            data: the data, which is not changed.
            made: the columns that earlier rules made (or changed), by name,
                which this rule searches, adds to, or replaces. It is changed
                in place.
            column: the column of text to search if the rule has no
                `columns`. Defaults to "text".

        """
        columns = self.columns or (column,)
        texts = [_text(data, made, c, self.description) for c in columns]
        if self.kind == 'replace':
            self._replace(texts, columns, made)
            return
        method = getattr(self, f'_{self.kind}')
        method(texts, made, data.index)

    """ Private Methods """

    def _count(
        self,
        texts: list[pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes the number of matches in each row (whole numbers).

        Args:
            texts: the columns of text to search.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        name = self.names[0]
        counts = np.zeros(len(index), dtype = 'int64')
        for text in texts:
            found = text.str.count(self.pattern.pattern, flags = self._flags)
            counts += found.fillna(0).to_numpy(dtype = 'int64')
        if name in made:
            counts += made[name].to_numpy(dtype = 'int64')
        made[name] = pd.Series(counts, index = index)

    def _extract(
        self,
        texts: list[pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes the first match in each row (text).

        Args:
            texts: the columns of text to search, in order.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        source = _wrap(self.pattern, '(', ')')
        found = _missing(len(index))
        for text in texts:
            matches = text.str.extract(source, flags = self._flags)
            _fill(found, _strings(matches.iloc[:, 0]))
        _keep_first(made, self.names[0], found, index)

    def _extract_all(
        self,
        texts: list[pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes every match in each row, separated by commas (text).

        Args:
            texts: the columns of text to search, in order.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        source = _wrap(self.pattern, '(', ')')
        found = _missing(len(index))
        for text in texts:
            plain = text.reset_index(drop = True)
            matches = plain.str.extractall(source, flags = self._flags)
            if matches.empty:
                continue
            kept = _patterns.tidy(matches.iloc[:, 0].astype('str')).dropna()
            joined = kept.groupby(level = 0).agg(_SEPARATOR.join)
            found = _join(found, joined.reindex(range(len(plain))).to_numpy(
                dtype = object))
        _add_to(made, self.names[0], found, index)

    def _fields(
        self,
        texts: list[pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes a column for each label, with the text after it (text).

        Args:
            texts: the columns of text to search, in order.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        label = _body(self.pattern)
        source = _wrap(
            self.pattern,
            '(?P<_label>',
            f')(?P<_text>(?s:.*?))(?=(?:{label})|\\Z)')
        prefix = f'{self.names[0]}_' if self.names else ''
        for text in texts:
            plain = text.reset_index(drop = True)
            matches = plain.str.extractall(source, flags = self._flags)
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

    def _flag(
        self,
        texts: list[pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes whether the pattern appears in each row (booleans).

        Args:
            texts: the columns of text to search.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        name = self.names[0]
        found = self._matches(texts)
        if name in made:
            found |= made[name].to_numpy(dtype = bool)
        made[name] = pd.Series(found, index = index)

    def _map(
        self,
        texts: list[pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes the rule's value in each row where the pattern appears.

        The type of the column is found from its values, as `pandas` does.

        Args:
            texts: the columns of text to search.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        found = _missing(len(index))
        found[self._matches(texts)] = self.value
        _keep_first(made, self.names[0], found, index)

    def _map_all(
        self,
        texts: list[pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Adds the rule's value to each row where the pattern appears.

        Each value is added once, after the values of earlier rules,
        separated by commas (text).

        Args:
            texts: the columns of text to search.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        name = self.names[0]
        found = _missing(len(index))
        found[self._matches(texts)] = str(self.value)
        if name in made:
            before = pd.Series(
                made[name].to_numpy(dtype = object), dtype = 'str').fillna('')
            item = f'{_SEPARATOR}{self.value}{_SEPARATOR}'
            listed = (_SEPARATOR + before + _SEPARATOR).str.contains(
                item, regex = False).to_numpy(dtype = bool)
            found[listed] = None
        _add_to(made, name, found, index)

    def _matches(self, texts: list[pd.Series]) -> np.ndarray:
        """Returns whether the pattern appears in any of `texts` in each row.

        Args:
            texts: the columns of text to search.

        Returns:
            A boolean array, which is false where every text is missing.

        """
        found = np.zeros(len(texts[0]), dtype = bool)
        for text in texts:
            found |= _patterns.contains(text, self.pattern.pattern, self._flags)
        return found

    def _names(
        self,
        texts: list[pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes the names in each row, once each, separated by commas.

        The names are found row by row, since `_split_names` cleans each
        piece of text.

        Args:
            texts: the columns of text to search, in order.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        name = self.names[0]
        lists: list[list[str]] = [[] for _ in range(len(index))]
        if name in made:
            earlier = made[name].to_numpy(dtype = object)
            for position, before in enumerate(earlier):
                if not pd.isna(before):
                    lists[position] = str(before).split(_SEPARATOR)
        for text in texts:
            for position, value in enumerate(text.to_numpy(dtype = object)):
                if pd.isna(value):
                    continue
                listed = lists[position]
                listed.extend(
                    n for n in _split_names(str(value), self.pattern)
                    if n not in listed)
        joined = np.array(
            [_SEPARATOR.join(n) if n else None for n in lists], dtype = object)
        made[name] = _text_column(joined, index)

    def _replace(
        self,
        texts: list[pd.Series],
        columns: Sequence[str],
        made: dict[str, pd.Series]) -> None:
        """Replaces every match in the text with the rule's value.

        The changed text goes in the rule's name, if it has one, and
        otherwise replaces the text it searched.

        Args:
            texts: the columns of text to change.
            columns: their names.
            made: the columns that earlier rules made, by name.

        """
        for column, text in zip(columns, texts, strict = True):
            changed = text.str.replace(
                self.pattern.pattern,
                str(self.value),
                regex = True,
                flags = self._flags)
            made[self.names[0] if self.names else column] = changed

    def _split(
        self,
        texts: list[pd.Series],
        made: dict[str, pd.Series],
        index: pd.Index) -> None:
        """Makes the text before and after the first match (text).

        The first column of `texts` with text in a row is split. Text without
        a match is all in the first part.

        Args:
            texts: the columns of text, in order.
            made: the columns that earlier rules made, by name.
            index: the labels of the rows.

        """
        whole = _missing(len(index))
        for text in texts:
            _fill(whole, text.to_numpy(dtype = object))
        plain = pd.Series(whole, dtype = 'str')
        source = _wrap(self.pattern, r'(?s:\A(.*?))(?:', r')(?s:(.*)\Z)')
        parts = plain.str.extract(source, flags = self._flags)
        before = parts.iloc[:, 0].to_numpy(dtype = object)
        after = parts.iloc[:, -1].to_numpy(dtype = object)
        unmatched = pd.isna(before) & ~pd.isna(whole)
        before[unmatched] = whole[unmatched]
        for name, part in zip(self.names, (before, after), strict = True):
            tidied = _patterns.tidy(pd.Series(part, dtype = 'str'))
            _keep_first(made, name, tidied.to_numpy(dtype = object), index)

    @property
    def _flags(self) -> int:
        """Returns the flags of the pattern, for the methods of `pandas`.

        Returns:
            The flags, without those that every pattern of text has.

        """
        return self.pattern.flags & ~re.UNICODE


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
        made: dict[str, pd.Series] = {}
        for rule in self.rules:
            try:
                rule.apply(data, made, column = column)
            except (KeyError, TypeError) as error:
                detail = error.args[0] if error.args else str(error)
                message = f'{rule.description}: {detail}'
                raise type(error)(message) from error
        return _assemble(data, made)

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


def register_rulebook(name: str, sources: Any) -> None:
    """Registers the rules of a rulebook under `name`.

    `parse_documents` (and `Rulebook.create`) then accept `name` in place of
    its sources, so a package built on `amos` can keep its rules in its own
    CSV files and every project can name them in its settings (as in "rules
    = federal"). The sources are read each time the rulebook is used, so
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
    index: pd.Index) -> None:
    """Puts text in a column, after any that earlier rules put there.

    Args:
        made: the columns that rules made, by name. It is changed in place.
        name: the column.
        found: the text of each row (or `None`).
        index: the labels of the rows.

    """
    if name in made:
        found = _join(made[name].to_numpy(dtype = object), found)
    made[name] = _text_column(found, index)


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


def _check_names(
    kind: str,
    columns: tuple[str, ...],
    names: tuple[str, ...]) -> None:
    """Checks that a rule has as many names as its kind needs.

    Args:
        kind: the kind of the rule.
        columns: the columns it searches.
        names: the columns it makes.

    Raises:
        ValueError: if the rule has too many or too few names.

    """
    if kind == 'split':
        if len(names) != 2:  # noqa: PLR2004
            message = (
                'a split rule needs two names, separated by a comma (such as '
                '"party1, party2")'
            )
            raise ValueError(message)
        return
    if not names and kind not in _NAMELESS_KINDS:
        message = 'the name is blank'
        raise ValueError(message)
    if len(names) > 1:
        message = f'a {kind} rule makes one column, not {list(names)}'
        raise ValueError(message)
    if kind == 'replace' and names and len(columns) > 1:
        message = (
            'a replace rule with a name changes one column: leave the name '
            'blank to change each column in place'
        )
        raise ValueError(message)


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


def _field_name(label: str) -> str:
    """Returns the name of the column of a label that a `fields` rule found.

    Args:
        label: the label, such as "PRIOR HISTORY:".

    Returns:
        The label in snake case, such as "prior_history", or an empty `str`
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
            commas or spaces, or blank.

    Raises:
        ValueError: if a name is not one of `_FLAGS`.

    Returns:
        The flags, combined.

    """
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


def _join(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Returns two columns of text joined in each row, with commas.

    Args:
        first: text (or `None`) for each row.
        second: text (or `None`) for the same rows.

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
            pd.Series(first[both], dtype = 'str') + _SEPARATOR
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


def _missing(count: int) -> np.ndarray:
    """Returns an array of `count` missing values.

    Args:
        count: the number of rows.

    Returns:
        An array of `None`, of objects.

    """
    return np.full(count, None, dtype = object)


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
    (so "O'Scannlain" becomes "OSCANNLAIN"), and line breaks always separate
    names. Pieces without at least two letters are dropped.

    Args:
        text: text with names, such as "Before SMITH, JONES, and BROWN,
            Circuit Judges."
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

    The columns of courtpy's layout ("target", "variable", and TRUE or FALSE
    in "ignorecase" and "dotall") and its kinds ("label", "labels",
    "section", "matches", "excerpts", and "remove") become those they stand
    for. A "remove" rule replaces its matches with a space, as courtpy's
    did.

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
    if kind in _KIND_ALIASES:
        cells['kind'] = _KIND_ALIASES[kind]
        if kind == 'remove' and not str(cells.get('value') or ''):
            cells['value'] = ' '
    return cells


def _value_of(kind: str, cell: Any) -> Any:
    """Returns the value of a rule, checked for its kind.

    Args:
        kind: the kind of the rule.
        cell: the cell of its value.

    Raises:
        ValueError: if a `map` or `map_all` rule has no value, a `map_all`
            value has a comma, or a rule of another kind (other than
            `replace`) has a value.

    Returns:
        The value: a number or text for a `map` rule, text for a `map_all`
            or `replace` rule (which is empty if blank), or `None`.

    """
    blank = cell is None or (isinstance(cell, str) and not cell.strip())
    if kind == 'replace':
        return '' if cell is None else str(cell)
    if kind not in _VALUE_KINDS:
        if not blank:
            message = (
                f'a {kind} rule does not use a value, but it has {cell!r} '
                f'(is its kind right?)'
            )
            raise ValueError(message)
        return None
    if blank:
        message = f'a {kind} rule needs a value to write when it matches'
        raise ValueError(message)
    if kind == 'map_all':
        text = str(cell).strip()
        if ',' in text:
            message = (
                f'the value {text!r} has a comma, which separates the values '
                f'of map_all rules'
            )
            raise ValueError(message)
        return text
    return _convert(cell)


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
