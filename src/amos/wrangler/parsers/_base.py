"""The base classes of parsers, and what they share.

Contents:
    _COLUMN_ALIASES: names of the columns of rules in the older layout (and of
        keys that settings may use), mapped to the names of this layout.
    _FLAGS: the flags that a rule can have, by their names.
    _KIND_ALIASES: kinds of rules in the older layout and the first layout of
        this module, mapped to the kinds they stand for.
    _OLD_FLAGS: columns of flags in the older layout, which hold TRUE or FALSE.
    _SEPARATOR: what separates several values in one cell of the data.
    _TRUE: words (in lower case) that mean true in the older flag columns.
    Parser: base class for techniques that turn text into data.
    PatternParser: base class for parsers that search text for a pattern, each
        of which can be a kind of rule.
    Rule: one search of text, as a row of a rulebook.

"""

from __future__ import annotations

import abc
import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar

import camina
import chrisjen
import numpy as np
import pandas as pd

from ...core import base
from .. import _patterns

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
_OLD_FLAGS: tuple[str, ...] = ('dotall', 'ignorecase')
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
    a rulebook (see `ParseAll`). Either way, the search is described by a
    `Rule`, which `find` applies.

    Every subclass is a kind of rule, named by `kind`: the snake case name of
    the class without "parse_" (`ParseFlags` is "flags"). Defining the
    subclass is all it takes for rulebooks to name it. Set `rule_aliases` to
    give the kind other names, such as an old name or the name of a kind that
    the subclass replaces. Aliases are not inherited, and they take the
    place of a kind of the same name.

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
    # Other names of the kind, which rulebooks can use in its place. Only the
    # class that sets them has them.
    rule_aliases: ClassVar[tuple[str, ...]] = ()

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
            The snake case name of the class without "parse_".

        """
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
            in which case it is the pattern parser of `kind`.

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
        kinds = _rule_kinds()
        if kind not in kinds:
            message = f'the kind {kind!r} is not one of {sorted(kinds)}'
            raise ValueError(message)
        return kinds[kind].make_rule(
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
                which case it is an instance of `parser` or of the pattern
                parser of `kind`.

        Raises:
            ValueError: if no parser applies the rule.

        """
        if parser is None:
            kind = self.parser or _rule_kinds().get(self.kind)
            if kind is None:
                message = (
                    f'{self.description}: no pattern parser is the kind '
                    f'{self.kind!r}'
                )
                raise ValueError(message)
            parser = kind()
        columns = self.columns or (column,)
        texts = {c: _text(data, made, c, self.description) for c in columns}
        parser.find(self, texts, made, data.index)


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


def _rule_kinds() -> dict[str, type[PatternParser]]:
    """Returns the pattern parser of each kind of rule, by the kind.

    Every pattern parser in the library (every subclass of `PatternParser`)
    is the kind that `PatternParser.kind` names, and the kind of each of its
    own `rule_aliases`. An alias takes the place of a kind of the same name.

    Returns:
        The pattern parsers, by their kinds and aliases.

    """
    layer = chrisjen.library.get_genre(PatternParser)
    parsers = [
        p for p in chrisjen.utilities.flatten_dict(layer).values()
        if isinstance(p, type) and issubclass(p, PatternParser)]
    kinds: dict[str, type[PatternParser]] = {p.kind(): p for p in parsers}
    for parser in parsers:
        for alias in vars(parser).get('rule_aliases', ()):
            kinds[str(alias).strip().lower()] = parser
    return kinds


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
    (see `_KIND_ALIASES`) become the kinds they stand for, unless a pattern
    parser is (or has an alias of) the same name. A "remove" rule replaces
    its matches with a space, as the older layout did.

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
    if kind in _KIND_ALIASES and kind not in _rule_kinds():
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
