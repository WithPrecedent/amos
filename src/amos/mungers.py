r"""Techniques that change what is in columns, usually before data is studied.

These are techniques of the "wrangler" stage of a project. Cleaners (see the
`cleaners` module) change which rows and columns there are. Mungers change
what is in the columns, or make new columns from them, without adding or
removing rows. Each one changes a whole column at once with the vectorized
methods of `pandas` (such as those of `Series.str`), so they are fast even
with a lot of data. Like cleaners, mungers learn nothing from the data that
could leak from the test rows into the training rows, so they are safe to use
before the data is split.

Many mungers search text for patterns. A pattern is a regular expression (see
Python's `re` module), and it is found anywhere in the text, so "revers" is
found in "Reversed and remanded". Characters with special meanings (such as
"." and "(") need a backslash to stand for themselves, as in "F\.3d". Set the
"ignorecase" parameter to ignore the difference between capital and
lower-case letters, and start a pattern with an inline flag for other options
(such as "(?s)", which lets "." match line breaks). Missing text matches
nothing.

Contents:
    Munger: base class for techniques that change the values in columns.
    AutoCategorize: makes columns with few unique values categorical.
    Coalesce: takes the first value that is not missing from several
        columns.
    CombineFlags: combines flags into one.
    ConvertTypes: changes the data types of columns.
    CountPatterns: counts the matches of patterns in text.
    DeriveColumns: makes columns from expressions of other columns.
    ExtractAll: keeps every match of a pattern in text.
    ExtractPattern: keeps the first match of a pattern in text.
    FlagPatterns: makes columns that say whether text matches patterns.
    MapPatterns: turns text into values by the first pattern it matches.
    MapValues: replaces values with others.
    NormalizeText: collapses spaces in text, and optionally changes its case
        and removes accents and punctuation.
    ParseBooleans: turns text such as "yes" and "no" into booleans.
    ParseDates: turns text into dates.
    ParseNumbers: turns text such as "$1,234.50" into numbers.
    ReplaceText: replaces the matches of patterns in text.
    SplitText: splits text into columns where a pattern matches.
    StripText: trims spaces from (and optionally lowercases) text.

"""

from __future__ import annotations

import abc
import dataclasses
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from . import base, options, utilities

# Combining accents (such as the accent of "é" once it is split into "e" and
# the accent), which `NormalizeText` removes. The string is not raw, so that
# Python makes the escapes into characters: the regular expressions of
# pyarrow, which `pandas` uses for text, do not know the escapes.
_ACCENTS: str = '[\u0300-\u036f]'
# The cases that `NormalizeText` can change text to, which are the names of
# methods of `Series.str`.
_CASES: frozenset[str] = frozenset({'lower', 'title', 'upper'})
# Words (in lower case) that `ParseBooleans` reads as false and as true.
_FALSE: tuple[str, ...] = ('false', 'f', 'no', 'n', '0')
_TRUE: tuple[str, ...] = ('true', 't', 'yes', 'y', '1')
# The ways that `CombineFlags` can combine flags.
_HOWS: frozenset[str] = frozenset({'all', 'any', 'count'})
# Inline flags (such as "(?i)") at the start of a pattern, which must stay at
# the start when the rest of the pattern is put in a group.
_LEADING_FLAGS: re.Pattern[str] = re.compile(r'(?:\(\?[aiLmsux]+\))*')
# Anything that is not a letter, digit, underscore, or space.
_PUNCTUATION: str = r'[^\w\s]'
# The message that `pandas` gives when a pattern with groups is only used to
# find whether text matches, which is all that `_contains` needs.
_GROUPS_WARNING: str = 'This pattern is interpreted as a regular expression'
_WHITESPACE: str = r'\s+'


@dataclasses.dataclass
class Munger(base.Operation, abc.ABC):
    """Base class for techniques that change the values in columns.

    A subclass writes a `munge` method, which takes the data and returns it
    with changed or new columns, and the same rows. The dataset's `history`
    records the columns whose values or types changed ("changed") and the new
    columns ("created").

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: not used by most mungers. Defaults to `None`.
        parameters: keyword arguments for `munge`. Defaults to an empty
            `dict`.

    """

    """ Required Methods """

    @abc.abstractmethod
    def munge(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Returns `data` with changed or new columns.

        Args:
            data: the data to change. It may be changed in place.
            **kwargs: parameters for munging.

        Returns:
            The changed data, with the same rows.

        """

    """ Public Methods """

    def implement(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Changes the columns of `item`.

        Args:
            item: the dataset to change.
            **kwargs: parameters for `munge`.

        Raises:
            ValueError: if `munge` changed the rows or removed columns.

        Returns:
            The changed dataset.

        """
        before = item.data
        after = self.munge(before.copy(), **kwargs)
        removed = [c for c in before.columns if c not in after.columns]
        if removed or not after.index.equals(before.index):
            message = (
                f'a munger only changes and adds columns, but {self.name!r} '
                f'changed the rows or removed columns of the data (use a '
                f'cleaner to remove them)'
            )
            raise ValueError(message)
        item.replace(after)
        item.record(
            self.name,
            changed = [
                c for c in before.columns if not after[c].equals(before[c])],
            created = [c for c in after.columns if c not in before.columns])
        return item


@dataclasses.dataclass
class AutoCategorize(Munger):
    """Makes columns with few unique values categorical.

    Text and number columns with `threshold` unique values or fewer become the
    `pandas` "category" type, so they are encoded (rather than scaled) by the
    analyst. Boolean columns are not changed.

    """

    def munge(
        self,
        data: pd.DataFrame,
        columns: Sequence[str] | None = None,
        threshold: int = options._CATEGORY_THRESHOLD,
        **kwargs: Any) -> pd.DataFrame:
        """Makes columns with `threshold` unique values or fewer categorical.

        Args:
            data: the data to change.
            columns: columns to check. Defaults to `None`, which checks every
                column.
            threshold: most unique values that a categorical column can have.
                Defaults to `options._CATEGORY_THRESHOLD`.
            **kwargs: not used.

        Returns:
            The data, with categorical columns.

        """
        for column in utilities.select_columns(data, columns):
            values = data[column]
            if pd.api.types.is_bool_dtype(values.dtype):
                continue
            if values.nunique(dropna = True) <= threshold:
                data[column] = values.astype('category')
        return data


@dataclasses.dataclass
class Coalesce(Munger):
    """Takes the first value that is not missing from several columns.

    This combines columns that hold the same information from different
    sources, such as a date from a case's own data and a date found in its
    text: the first column is used where it has a value, and the next column
    fills in where it does not, and so on.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        name: str | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Makes a column of the first value in `columns` that is not missing.

        Args:
            data: the data to change.
            columns: the columns to take values from, in order of preference.
                Defaults to `None`, which changes nothing.
            name: name of the column to make. Defaults to `None`, in which
                case it is the first of `columns`, whose missing values are
                filled in from the others.
            **kwargs: not used.

        Returns:
            The data, with the combined column. It is missing only where every
                one of `columns` is.

        """
        names = utilities.select_columns(data, columns or [])
        if not names:
            return data
        # Values of several types are combined as objects, and `infer_objects`
        # then finds the type that suits them all.
        first = data[names].astype(object).bfill(axis = 1).iloc[:, 0]
        data[name or names[0]] = first.infer_objects()
        return data


@dataclasses.dataclass
class CombineFlags(Munger):
    """Combines flags into one: whether any or all are true, or how many are.

    For example, a case is about any criminal issue if any of its flags of
    criminal issues ("criminal_search", "criminal_sentence", and so on) are
    true. Flags are boolean columns or numbers (where 0 is false), such as
    those that `flag_patterns` makes. A missing value is false.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        name: str | None = None,
        how: str = 'any',
        **kwargs: Any) -> pd.DataFrame:
        """Makes a column that combines the flags in `columns`.

        Args:
            data: the data to change.
            columns: the flags to combine. Defaults to `None`, which changes
                nothing.
            name: name of the column to make. It is needed if there are
                `columns`. Defaults to `None`.
            how: "any" (true if any flag is true), "all" (true if every flag
                is true), or "count" (the number of flags that are true).
                Defaults to "any".
            **kwargs: not used.

        Raises:
            TypeError: if a column is not a flag.
            ValueError: if `how` is not one of the ways above, or there is no
                `name`.

        Returns:
            The data, with the combined column: a boolean column, or whole
                numbers for "count".

        """
        names = utilities.select_columns(data, columns or [])
        if not names:
            return data
        if how not in _HOWS:
            message = f'how must be one of {sorted(_HOWS)}, not {how!r}'
            raise ValueError(message)
        if name is None:
            message = f'{self.name!r} needs a name for the column it makes'
            raise ValueError(message)
        flags = np.column_stack([_truth(data[c], c) for c in names])
        if how == 'count':
            data[name] = flags.sum(axis = 1).astype('int64')
        else:
            data[name] = np.asarray(
                flags.any(axis = 1) if how == 'any' else flags.all(axis = 1))
        return data


@dataclasses.dataclass
class ConvertTypes(Munger):
    """Changes the data types of columns."""

    def munge(
        self,
        data: pd.DataFrame,
        types: Mapping[str, Any] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Changes the type of each column in `types`.

        Args:
            data: the data to change.
            types: `dict` mapping column names to `pandas` data types (such as
                "category", "float", or "boolean"). Defaults to `None`, which
                changes nothing.
            **kwargs: not used.

        Returns:
            The data, with the new types.

        """
        return data.astype(dict(types or {}))


@dataclasses.dataclass
class CountPatterns(Munger):
    """Counts the matches of patterns in text."""

    def munge(
        self,
        data: pd.DataFrame,
        *,
        column: str | None = None,
        patterns: Mapping[str, str | Sequence[str]] | None = None,
        ignorecase: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        r"""Makes a column with the number of matches of each pattern.

        Args:
            data: the data to change.
            column: the column of text to search.
            patterns: `dict` mapping the names of the columns to make to
                patterns, or to lists of patterns whose matches are added
                together. For example, {"words": "\S+"} counts the words.
                Defaults to `None`, which changes nothing.
            ignorecase: whether to ignore the difference between capital and
                lower-case letters. Defaults to `False`.
            **kwargs: not used.

        Returns:
            The data, with a column of counts (whole numbers) for each name in
                `patterns`. Missing text has no matches.

        """
        if not patterns:
            return data
        text = _text(data, column, self.name)
        flags = re.IGNORECASE if ignorecase else 0
        for name, sources in dict(patterns).items():
            counts = np.zeros(len(text), dtype = 'int64')
            for source in _sequence(sources):
                pattern = _compile(source, flags).pattern
                found = text.str.count(pattern, flags = flags)
                counts += found.fillna(0).to_numpy(dtype = 'int64')
            data[name] = counts
        return data


@dataclasses.dataclass
class DeriveColumns(Munger):
    """Makes columns from expressions of other columns, such as "a / b"."""

    def munge(
        self,
        data: pd.DataFrame,
        *,
        expressions: Mapping[str, str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Makes a column from each expression, in order.

        Args:
            data: the data to change.
            expressions: `dict` mapping the names of the columns to make to
                `pandas` expressions of other columns, such as "damages /
                judges" or "age >= 18" (see `pandas.DataFrame.eval`). Column
                names with spaces go inside backticks. An expression can use
                the columns made before it. Defaults to `None`, which changes
                nothing.
            **kwargs: not used.

        Raises:
            ValueError: if an expression is an assignment (such as "a = b"),
                rather than an expression.

        Returns:
            The data, with the new columns.

        """
        for name, expression in dict(expressions or {}).items():
            values = data.eval(str(expression))
            if isinstance(values, pd.DataFrame):
                message = (
                    f'{expression!r} is an assignment: name the column in the '
                    f'keys of the expressions instead'
                )
                raise ValueError(message)  # noqa: TRY004
            data[name] = values
        return data


@dataclasses.dataclass
class ExtractAll(Munger):
    """Keeps every match of a pattern in text."""

    def munge(
        self,
        data: pd.DataFrame,
        *,
        column: str | None = None,
        pattern: str | None = None,
        name: str | None = None,
        separator: str = ', ',
        ignorecase: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        """Keeps every match of `pattern` in `column`, joined together.

        Args:
            data: the data to change.
            column: the column of text to search.
            pattern: the pattern to find. If it has a group, the text of the
                group is kept rather than the whole match. Defaults to `None`,
                which changes nothing.
            name: name of the column to make. Defaults to `None`, which
                replaces `column`.
            separator: text put between the matches. Defaults to ", ".
            ignorecase: whether to ignore the difference between capital and
                lower-case letters. Defaults to `False`.
            **kwargs: not used.

        Raises:
            ValueError: if `pattern` has more than one group.

        Returns:
            The data, with the matches. Text without a match is missing.

        """
        if pattern is None:
            return data
        text = _text(data, column, self.name)
        flags = re.IGNORECASE if ignorecase else 0
        compiled = _compile(pattern, flags)
        if compiled.groups > 1:
            message = (
                f'the pattern {compiled.pattern!r} has {compiled.groups} '
                f'groups: keep one, and make the others "(?:...)", which is '
                f'not kept'
            )
            raise ValueError(message)
        found = text.str.findall(compiled.pattern, flags = flags)
        data[name or column] = _tidy(found.str.join(separator).astype('str'))
        return data


@dataclasses.dataclass
class ExtractPattern(Munger):
    r"""Keeps the first match of a pattern in text (or the groups of a match).

    A pattern with named groups makes a column for each group. For example,
    "(?P<volume>\d+) F\.3d (?P<page>\d+)" makes "volume" and "page"
    columns from citations such as "512 F.3d 1093".

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        column: str | None = None,
        pattern: str | None = None,
        name: str | None = None,
        ignorecase: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        """Keeps the first match of `pattern` in `column`.

        Args:
            data: the data to change.
            column: the column of text to search.
            pattern: the pattern to find. If it has one unnamed group, the
                text of the group is kept rather than the whole match. If its
                groups are named, each group makes a column of its name.
                Defaults to `None`, which changes nothing.
            name: name of the column to make (unless the groups are named).
                Defaults to `None`, which replaces `column`.
            ignorecase: whether to ignore the difference between capital and
                lower-case letters. Defaults to `False`.
            **kwargs: not used.

        Raises:
            ValueError: if `pattern` has several groups that are not all
                named.

        Returns:
            The data, with the matches. Text without a match is missing.

        """
        if pattern is None:
            return data
        text = _text(data, column, self.name)
        flags = re.IGNORECASE if ignorecase else 0
        compiled = _compile(pattern, flags)
        if compiled.groups > 1 or compiled.groupindex:
            if len(compiled.groupindex) < compiled.groups:
                message = (
                    f'the groups of the pattern {compiled.pattern!r} need '
                    f'names (as in "(?P<year>\\d{{4}})"), so that each makes '
                    f'a column of its name'
                )
                raise ValueError(message)
            found = text.str.extract(
                compiled.pattern, flags = flags, expand = True)
            for group in found.columns:
                data[group] = _tidy(found[group])
            return data
        source = compiled.pattern if compiled.groups else _capture(
            compiled.pattern)
        match = text.str.extract(source, flags = flags, expand = False)
        data[name or column] = _tidy(match)
        return data


@dataclasses.dataclass
class FlagPatterns(Munger):
    """Makes columns that say whether text matches patterns.

    For example, {"reversed": "revers|vacat"} makes a "reversed" column that
    is true for each row whose text mentions a reversal or vacatur.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        column: str | None = None,
        patterns: Mapping[str, str | Sequence[str]] | None = None,
        ignorecase: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        """Makes a boolean column for each pattern.

        Args:
            data: the data to change.
            column: the column of text to search.
            patterns: `dict` mapping the names of the columns to make to
                patterns, or to lists of patterns (any of which can match).
                Defaults to `None`, which changes nothing.
            ignorecase: whether to ignore the difference between capital and
                lower-case letters. Defaults to `False`.
            **kwargs: not used.

        Returns:
            The data, with a boolean column for each name in `patterns`.
                Missing text is false.

        """
        if not patterns:
            return data
        text = _text(data, column, self.name)
        flags = re.IGNORECASE if ignorecase else 0
        for name, sources in dict(patterns).items():
            found = np.zeros(len(text), dtype = bool)
            for source in _sequence(sources):
                found |= _contains(text, _compile(source, flags).pattern, flags)
            data[name] = found
        return data


@dataclasses.dataclass
class MapPatterns(Munger):
    """Turns text into values by the first pattern that it matches.

    For example, {"first circuit": 1, "second circuit": 2} turns the names of
    courts into their numbers. The patterns are tried in order, so put the
    most specific first.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        column: str | None = None,
        patterns: Mapping[str, Any] | None = None,
        name: str | None = None,
        default: Any = None,
        ignorecase: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        """Makes a column with the value of the first pattern that matches.

        Args:
            data: the data to change.
            column: the column of text to search.
            patterns: `dict` mapping patterns to values, in the order in which
                they are tried. Defaults to `None`, which changes nothing.
            name: name of the column to make. Defaults to `None`, which
                replaces `column`.
            default: the value of text that matches no pattern. Defaults to
                `None`, which is missing.
            ignorecase: whether to ignore the difference between capital and
                lower-case letters. Defaults to `False`.
            **kwargs: not used.

        Returns:
            The data, with the values. Their type is found from the values
                (and `default`), as `pandas` does.

        """
        if not patterns:
            return data
        text = _text(data, column, self.name)
        flags = re.IGNORECASE if ignorecase else 0
        rules = list(dict(patterns).items())
        # The position of the value of each row, which starts as the default
        # (after the values). The patterns are tried from last to first, so
        # that the first pattern that matches has the final word.
        choices = np.full(len(text), len(rules))
        for position in reversed(range(len(rules))):
            pattern = _compile(rules[position][0], flags).pattern
            choices[_contains(text, pattern, flags)] = position
        values = pd.Series([value for _, value in rules] + [default])
        data[name or column] = values.take(choices).set_axis(data.index)
        return data


@dataclasses.dataclass
class MapValues(Munger):
    """Replaces values with others, such as "N/A" with a missing value.

    Only whole values that are the same as one in `values` are replaced (use
    `replace_text` to change part of the text). Other values are kept. A
    categorical column stays categorical.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        values: Mapping[Any, Any] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Replaces the values in `columns` that are keys of `values`.

        Args:
            data: the data to change.
            columns: columns to change. Defaults to `None`, which changes
                every column.
            values: `dict` mapping values to the values that replace them.
                `None` makes a value missing. Defaults to `None`, which
                changes nothing.
            **kwargs: not used.

        Returns:
            The data, with the new values. The type of a changed column is
                found from its new values, as `pandas` does.

        """
        if not values:
            return data
        mapping = dict(values)
        for column in utilities.select_columns(data, columns):
            data[column] = _recode(data[column], mapping)
        return data


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
class ParseBooleans(Munger):
    """Turns text such as "yes" and "no" into booleans.

    Text is compared without regard to case or the spaces around it. Numbers
    are true if they are 1 and false if they are 0. Anything else is missing,
    so the columns have the `pandas` "boolean" type, which allows missing
    values.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        true_values: Sequence[Any] = _TRUE,
        false_values: Sequence[Any] = _FALSE,
        **kwargs: Any) -> pd.DataFrame:
        """Turns the values in `columns` into booleans.

        Args:
            data: the data to change.
            columns: columns to change. Defaults to `None`, which changes
                nothing.
            true_values: words that mean true. Defaults to "true", "t", "yes",
                "y", and "1".
            false_values: words that mean false. Defaults to "false", "f",
                "no", "n", and "0".
            **kwargs: not used.

        Returns:
            The data, with booleans. Columns that are already boolean are not
                changed.

        """
        trues = _words(true_values)
        falses = _words(false_values)
        for column in utilities.select_columns(data, columns or []):
            values = data[column]
            if pd.api.types.is_bool_dtype(values.dtype):
                continue
            if pd.api.types.is_numeric_dtype(values.dtype):
                true = (values == 1).to_numpy(dtype = bool)
                false = (values == 0).to_numpy(dtype = bool)
            else:
                words = values.astype('str').str.strip().str.lower()
                true = words.isin(trues).to_numpy(dtype = bool)
                false = words.isin(falses).to_numpy(dtype = bool)
            flags = pd.array(true, dtype = 'boolean')
            flags[~(true | false)] = pd.NA
            data[column] = pd.Series(flags, index = values.index)
        return data


@dataclasses.dataclass
class ParseDates(Munger):
    """Turns text into dates.

    Text that is not a date becomes a missing value, rather than an error (as
    it would with `convert_types`).

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        date_format: str | None = None,
        dayfirst: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        """Turns the text in `columns` into dates.

        Args:
            data: the data to change.
            columns: columns to change. Defaults to `None`, which changes
                nothing.
            date_format: the format of the dates, with the codes of Python's
                `strftime` (such as "%m/%d/%Y"), or "mixed" to find the
                format of each date separately (which is slower). Defaults to
                `None`, in which case `pandas` finds the format from the first
                date and uses it for all of them.
            dayfirst: whether the day comes before the month in dates such as
                "01/02/2020". Defaults to `False`.
            **kwargs: not used.

        Returns:
            The data, with dates.

        """
        for column in utilities.select_columns(data, columns or []):
            values = data[column]
            if isinstance(values.dtype, pd.CategoricalDtype):
                values = values.astype('str')
            data[column] = pd.to_datetime(
                values,
                errors = 'coerce',
                format = date_format,
                dayfirst = dayfirst)
        return data


@dataclasses.dataclass
class ParseNumbers(Munger):
    """Turns text such as "$1,234.50" into numbers.

    The first number in the text is kept, so "$1,234.50" becomes 1234.5,
    "12%" becomes 12, and "about 3 years" becomes 3. Text without a number
    becomes a missing value.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        thousands: str = ',',
        decimal: str = '.',
        **kwargs: Any) -> pd.DataFrame:
        """Turns the text in `columns` into numbers.

        Args:
            data: the data to change.
            columns: columns to change. Defaults to `None`, which changes
                nothing.
            thousands: the character that separates thousands. Defaults to
                ",". Set it to "." (and `decimal` to ",") for numbers such as
                "1.234,50".
            decimal: the character before the decimals. Defaults to ".".
            **kwargs: not used.

        Raises:
            ValueError: if `thousands` and `decimal` are the same.

        Returns:
            The data, with numbers. Columns that are already numbers are not
                changed.

        """
        if thousands == decimal:
            message = 'the thousands and decimal separators must differ'
            raise ValueError(message)
        digits = r'\d+'
        if thousands:
            digits += rf'(?:{re.escape(thousands)}\d{{3}})*'
        point = re.escape(decimal)
        pattern = (
            rf'([-+]?(?:{digits}(?:{point}\d*)?|{point}\d+)'
            rf'(?:[eE][-+]?\d+)?)')
        for column in utilities.select_columns(data, columns or []):
            values = data[column]
            if pd.api.types.is_numeric_dtype(values.dtype):
                continue
            found = values.astype('str').str.extract(pattern, expand = False)
            if thousands:
                found = found.str.replace(thousands, '', regex = False)
            if decimal != '.':
                found = found.str.replace(decimal, '.', regex = False)
            data[column] = pd.to_numeric(found, errors = 'coerce')
        return data


@dataclasses.dataclass
class ReplaceText(Munger):
    r"""Replaces the matches of patterns in text.

    For example, {"\s*\(.*?\)": ""} removes everything in parentheses.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        patterns: Mapping[str, str | None] | None = None,
        ignorecase: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        r"""Replaces every match of each pattern, in order.

        Args:
            data: the data to change.
            columns: columns to change. Defaults to `None`, which changes every
                text column.
            patterns: `dict` mapping patterns to the text that replaces them.
                The text can include the groups of a match ("\1" is the
                first). An empty text (or `None`) removes the matches.
                Defaults to `None`, which changes nothing.
            ignorecase: whether to ignore the difference between capital and
                lower-case letters. Defaults to `False`.
            **kwargs: not used.

        Returns:
            The data, with the text replaced.

        """
        if not patterns:
            return data
        flags = re.IGNORECASE if ignorecase else 0
        rules = [
            (_compile(p, flags).pattern, '' if r is None else str(r))
            for p, r in dict(patterns).items()]
        names = _text_columns(data) if columns is None else (
            utilities.select_columns(data, columns))
        for column in names:
            text = _text(data, column, self.name)
            for pattern, replacement in rules:
                text = text.str.replace(
                    pattern, replacement, regex = True, flags = flags)
            data[column] = text
        return data


@dataclasses.dataclass
class SplitText(Munger):
    r"""Splits text into columns where a pattern matches.

    For example, the pattern " v\. " and the names "party1, party2" split a
    caption such as "Smith v. Jones" into "Smith" and "Jones". The text is
    split at the first matches, so the last column has the rest of the text.
    The original column is kept.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        column: str | None = None,
        pattern: str | None = None,
        names: Sequence[str] | None = None,
        ignorecase: bool = False,
        **kwargs: Any) -> pd.DataFrame:
        """Splits the text in `column` into the columns in `names`.

        Args:
            data: the data to change.
            column: the column of text to split.
            pattern: the pattern to split at. It cannot have groups (use
                "(?:...)" to group without one). Defaults to `None`, which
                changes nothing.
            names: names of the columns to make, one for each piece of the
                text. There must be at least two.
            ignorecase: whether to ignore the difference between capital and
                lower-case letters. Defaults to `False`.
            **kwargs: not used.

        Raises:
            ValueError: if there are fewer than two `names` or `pattern` has
                groups.

        Returns:
            The data, with a column for each piece, trimmed of spaces. A piece
                that is empty (or that the text does not have) is missing.

        """
        if pattern is None:
            return data
        labels = [names] if isinstance(names, str) else list(names or [])
        if len(labels) < 2:  # noqa: PLR2004
            message = (
                f'{self.name!r} needs the names of two or more columns to '
                f'make, not {labels}')
            raise ValueError(message)
        text = _text(data, column, self.name)
        compiled = _compile(pattern, re.IGNORECASE if ignorecase else 0)
        if compiled.groups:
            message = (
                f'the pattern {compiled.pattern!r} has groups, which would be '
                f'kept as pieces: use "(?:...)" to group without them'
            )
            raise ValueError(message)
        pieces = text.str.split(
            compiled, n = len(labels) - 1, expand = True, regex = True)
        for position, label in enumerate(labels):
            if position in pieces.columns:
                piece = pieces[position].astype('str')
            else:
                # No text had this many pieces.
                piece = pd.Series(np.nan, index = text.index, dtype = 'str')
            data[label] = _tidy(piece)
        return data


@dataclasses.dataclass
class StripText(Munger):
    """Trims spaces from text, and optionally makes it lowercase."""

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


""" Private Functions """


def _capture(pattern: str) -> str:
    """Returns `pattern` as one group, so that `pandas` can extract it.

    Args:
        pattern: a regular expression without groups.

    Returns:
        The pattern inside a group, after any inline flags at its start (which
            must stay at the start).

    """
    match = _LEADING_FLAGS.match(pattern)
    start = match.end() if match else 0
    return f'{pattern[:start]}({pattern[start:]})'


def _compile(pattern: Any, flags: int) -> re.Pattern[str]:
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


def _contains(text: pd.Series, pattern: str, flags: int) -> np.ndarray:
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
        warnings.filterwarnings('ignore', _GROUPS_WARNING, UserWarning)
        found = text.str.contains(pattern, flags = flags, na = False)
    return found.to_numpy(dtype = bool)


def _recode(values: pd.Series, mapping: Mapping[Any, Any]) -> pd.Series:
    """Returns `values` with those that are keys of `mapping` replaced.

    Args:
        values: a column of data.
        mapping: values, mapped to the values that replace them.

    Returns:
        The column with the new values, whose type is found from its values,
            or the column unchanged if it has none of the keys. A categorical
            column stays categorical.

    """
    categorical = isinstance(values.dtype, pd.CategoricalDtype)
    plain = values.astype(values.cat.categories.dtype) if categorical else (
        values)
    matched = plain.isin(list(mapping)).to_numpy(dtype = bool)
    if not matched.any():
        return values
    replaced = plain.map(mapping).astype(object).to_numpy()
    kept = plain.astype(object).to_numpy()
    result = pd.Series(
        np.where(matched, replaced, kept),
        index = values.index,
        name = values.name).infer_objects()
    return result.astype('category') if categorical else result


def _sequence(item: str | Sequence[str]) -> list[str]:
    """Returns a pattern, or a sequence of them, as a `list`.

    Args:
        item: a pattern or a sequence of patterns.

    Returns:
        The patterns.

    """
    return [item] if isinstance(item, str) else list(item)


def _text(data: pd.DataFrame, column: Any, technique: str) -> pd.Series:
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


def _text_columns(data: pd.DataFrame) -> list[str]:
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


def _tidy(text: pd.Series) -> pd.Series:
    """Returns `text` trimmed of spaces, with empty text missing.

    Args:
        text: a column of text.

    Returns:
        The trimmed text.

    """
    text = text.str.strip()
    return text.mask(text == '')


def _truth(values: pd.Series, column: Any) -> np.ndarray:
    """Returns a column of flags as plain booleans.

    Args:
        values: a boolean or numeric column.
        column: the name of the column, for messages.

    Raises:
        TypeError: if `values` is not boolean or numeric.

    Returns:
        A boolean array, which is true where `values` is not 0, false, or
            missing.

    """
    if not pd.api.types.is_numeric_dtype(values.dtype):
        message = (
            f'{column!r} is not a flag (true or false): parse_booleans can '
            f'make it one'
        )
        raise TypeError(message)
    return np.asarray(values.astype(float).fillna(0).to_numpy() != 0)


def _words(values: Sequence[Any] | Any) -> list[str]:
    """Returns words that mean true or false, in lower case.

    Args:
        values: a word or a sequence of them. Numbers and booleans (as a
            settings file may give them) are made into text.

    Returns:
        The words, trimmed and in lower case.

    """
    items = [values] if isinstance(values, str) or not isinstance(
        values, Iterable) else list(values)
    return [str(v).strip().lower() for v in items]
