"""Rulebooks of parsers, and the parser that applies them.

Contents:
    _REQUIRED: columns that every file of rules must have.
    ParseAll: finds information in text with a rulebook.
    Rulebook: an ordered collection of rules.

"""

from __future__ import annotations

import csv
import dataclasses
import pathlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar

import camina
import chrisjen
import pandas as pd

from ._base import (
    _COLUMN_ALIASES,
    _SEPARATOR,
    Parser,
    Rule,
    _apply_rules,
    _cell_names,
)

_REQUIRED: tuple[str, ...] = ('kind', 'pattern')


@dataclasses.dataclass
class ParseAll(Parser):
    """Finds information in text with a rulebook.

    The rules can be the path of a CSV file of rules or of a folder of them,
    a list of rules (as `dict`s, with the columns of a CSV file as keys), the
    name of a rulebook, or a list of any of these, which are applied in order
    (see `Rulebook.create`). Each rule is applied by the pattern parser of its
    kind (see `PatternParser`).

    A subclass that sets `rules` is a rulebook of its own. It is in the
    library under its name, like any technique, so settings can name it as a
    technique ("parse_federal" for `ParseFederal`) that needs no parameters.
    Its name, with or without "parse_", is also the name of its rulebook,
    which other rulebooks and the "rules" of `parse_all` can name ("rules =
    federal"). For example, a package built on `amos` can keep its rules in
    its own CSV files:

    ```py
    @dataclasses.dataclass
    class ParseFederal(amos.parsers.ParseAll):
        rules: ClassVar[Any] = RULES_FOLDER / 'federal'
    ```

    The rules are read each time they are used, so changes to their files
    take effect at once.

    """

    # The rules that `parse` applies when it is given none, in any form that
    # `Rulebook.create` accepts.
    rules: ClassVar[Any] = None

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
                `None`, which applies the class's `rules` (and changes
                nothing if it has none).
            column: the column of text that rules with a blank `column`
                search. Defaults to "text", the column of text that
                `load_files` makes.
            **kwargs: not used.

        Returns:
            The data, with the columns that the rules make.

        """
        if rules is None:
            rules = self.rules
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
            *sources: names of rulebooks (subclasses of `ParseAll` that set
                `rules`, named with or without "parse_"), those subclasses
                themselves, paths of CSV files of rules or of folders of them
                (whose files are read in alphabetical order), rules as
                `dict`s (with the columns of a CSV file as keys), `Rule`s,
                `Rulebook`s, or sequences of any of these. A text with commas
                that is not a file or the name of a rulebook is a list of
                them. A file that two sources name is read only the first
                time.

        Raises:
            FileNotFoundError: if a source is not the name of a rulebook, a
                file, or a folder with CSV files.
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


""" Private Functions """


def _gather(
    source: Any,
    read: set[pathlib.Path],
    named: tuple[str, ...]) -> tuple[list[Rule], list[str]]:
    """Returns the rules of a source of `Rulebook.create`, and its names.

    Args:
        source: a source (see `Rulebook.create`).
        read: the files read so far, which are not read again. It is changed
            in place.
        named: the names of the rulebooks that led to `source`, so that a
            rulebook that names itself is caught.

    Raises:
        FileNotFoundError: if a source is not the name of a rulebook, a file,
            or a folder with CSV files.
        ValueError: if a rulebook names itself, a file is missing a column,
            or a rule is not valid.

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
    book = _rulebook(source)
    if book is not None:
        name = _rulebook_name(book)
        if name in named:
            message = f'the rulebook {name!r} names itself'
            raise ValueError(message)
        found, _ = _gather(book.rules, read, (*named, name))
        return found, [name]
    if isinstance(source, str | pathlib.Path):
        path = pathlib.Path(source)
        if isinstance(source, str) and ',' in source and not path.exists():
            return _gather(_cell_names(source), read, named)
        return _read_rules(path, read)
    rules: list[Rule] = []
    names: list[str] = []
    for position, part in enumerate(source):
        if isinstance(part, Mapping):
            rules.append(_rule(part, f'rule {position + 1}'))
            continue
        found, labels = _gather(part, read, named)
        rules.extend(found)
        names.extend(labels)
    return rules, names


def _rulebook(source: Any) -> type[ParseAll] | None:
    """Returns the rulebook that `source` is or names, if it is one.

    Args:
        source: a source of `Rulebook.create`.

    Raises:
        ValueError: if `source` is a subclass of `ParseAll` without `rules`.

    Returns:
        `source`, if it is a subclass of `ParseAll`, the subclass of
            `ParseAll` in the library that `source` names (with or without
            "parse_"), or `None` if it is neither.

    """
    if isinstance(source, type) and issubclass(source, ParseAll):
        if source.rules is None:
            message = f'{source.__name__} has no rules'
            raise ValueError(message)
        return source
    if isinstance(source, str):
        name = source.strip().lower().removeprefix('parse_')
        return _rulebooks().get(name)
    return None


def _rulebook_name(book: type[ParseAll]) -> str:
    """Returns the name of a rulebook.

    Args:
        book: a subclass of `ParseAll`.

    Returns:
        The snake case name of the class without "parse_".

    """
    return str(camina.snakify(book.__name__)).removeprefix('parse_')


def _rulebooks() -> dict[str, type[ParseAll]]:
    """Returns the rulebooks in the library, by their names.

    Returns:
        Every subclass of `ParseAll` in the library that sets `rules`, by
            its name (see `_rulebook_name`).

    """
    layer = chrisjen.library.get_genre(Parser)
    return {
        _rulebook_name(p): p
        for p in chrisjen.utilities.flatten_dict(layer).values()
        if isinstance(p, type) and issubclass(p, ParseAll)
        and p.rules is not None}


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
            f'rulebook ({sorted(_rulebooks())})'
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
