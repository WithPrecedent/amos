"""Rulebooks of parsers, and the parser that applies them.

Contents:
    _REQUIRED: columns that every file of rules must have.
    _RULEBOOKS: the sources of each registered rulebook, by its name.
    ParseAll: finds information in text with a rulebook.
    Rulebook: an ordered collection of rules.
    register_rule_kind: names a pattern parser as a kind of rule.
    register_rulebook: registers the rules of a rulebook under a name.

"""

from __future__ import annotations

import csv
import dataclasses
import pathlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pandas as pd

from ._base import (
    _COLUMN_ALIASES,
    _RULE_KINDS,
    _SEPARATOR,
    Parser,
    PatternParser,
    Rule,
    _apply_rules,
    _cell_names,
)
from ._categories import ParseCategories, ParseCategory
from ._fields import ParseFields, ParseNames
from ._matches import (
    ParseCounts,
    ParseFlags,
    ParseMatch,
    ParseMatches,
    ParseReplace,
    ParseSplit,
)

_REQUIRED: tuple[str, ...] = ('kind', 'pattern')
_RULEBOOKS: dict[str, tuple[Any, ...]] = {}


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


# Each pattern parser of `amos` is a kind of rule.
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
