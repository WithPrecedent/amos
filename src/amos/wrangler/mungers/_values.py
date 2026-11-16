"""Mungers that make or replace values.

Contents:
    _HOWS: the ways that `CombineFlags` can combine flags.
    Coalesce: takes the first value that is not missing from several columns.
    CombineFlags: combines flags into one.
    DeriveColumns: makes columns from expressions of other columns.
    MapValues: replaces values with others.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ...core import utilities
from ._base import Munger

# The ways that `CombineFlags` can combine flags.
_HOWS: frozenset[str] = frozenset({'all', 'any', 'count'})


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

    For example, a review mentions a problem if any of its flags of problems
    ("problem_price", "problem_shipping", and so on) are true. Flags are boolean
    columns or numbers (where 0 is false), such as those that `parse_flags`
    makes. A missing value is false.

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
class DeriveColumns(Munger):
    """Makes columns from expressions of other columns, such as "a / b".

    The expressions are "expressions", a `dict` of the names of the new
    columns and their expressions, and any other parameter, whose name is the
    name of a new column and whose value is its expression (as in
    "victim_under_eighteen = victim_age < 18"). Words that are not columns are
    values, so they need no quotes, as in "offender_race == B" (see
    `utilities.quote_bare_values`).

    Wraps:
        [`pandas.DataFrame.eval`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.eval.html)
        from pandas.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        expressions: Mapping[str, str] | None = None,
        **named: str) -> pd.DataFrame:
        """Makes a column from each expression, in order.

        Args:
            data: the data to change.
            expressions: `dict` mapping the names of the columns to make to
                `pandas` expressions of other columns, such as "sales /
                employees" or "age >= 18" (see `pandas.DataFrame.eval`). Column
                names with spaces go inside backticks. An expression can use
                the columns made before it. Defaults to `None`.
            **named: more expressions, by the names of the columns to make,
                which are made after those in `expressions`.

        Raises:
            ValueError: if an expression is an assignment (such as "a = b"),
                rather than an expression.

        Returns:
            The data, with the new columns.

        """
        for name, expression in {**dict(expressions or {}), **named}.items():
            values = data.eval(
                utilities.quote_bare_values(str(expression), data))
            if isinstance(values, pd.DataFrame):
                message = (
                    f'{expression!r} is an assignment: name the column in the '
                    f'keys of the expressions instead'
                )
                raise ValueError(message)  # noqa: TRY004
            data[name] = values
        return data


@dataclasses.dataclass
class MapValues(Munger):
    """Replaces values with others, such as "N/A" with a missing value.

    Only whole values that are the same as one in `values` are replaced (use
    `parse_replace` to change part of the text). Other values are kept. A
    categorical column stays categorical.

    Without "values", each column is changed by its map or function in
    `value_maps`, which a subclass sets. A package built on `amos` can give
    the meanings of its codes once (such as "W" for "white"), and every
    project can then map them by naming the subclass, with or without
    "columns". A function is applied to each value that is not missing. For
    example:

    ```py
    @dataclasses.dataclass
    class MapRaceCodes(amos.mungers.MapValues):
        value_maps: ClassVar[dict[str, Any]] = {
            'offender_race': {'W': 'white', 'B': 'black'},
            'victim_race': {'W': 'white', 'B': 'black'},
            'state': str.upper}
    ```

    """

    # The map (a `dict` of values to the values that replace them) or the
    # function of each column, which `munge` uses when it is not given
    # "values".
    value_maps: ClassVar[
        Mapping[str, Mapping[Any, Any] | Callable[[Any], Any]]] = {}

    def munge(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | str | None = None,
        values: Mapping[Any, Any] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Replaces the values in `columns` that are keys of `values`.

        Args:
            data: the data to change.
            columns: columns to change. Defaults to `None`, which changes
                every column if `values` is given, and otherwise every column
                of `value_maps` that is in the data.
            values: `dict` mapping values to the values that replace them.
                `None` makes a value missing. Defaults to `None`, in which
                case each column is changed by its map or function in
                `value_maps`.
            **kwargs: not used.

        Raises:
            KeyError: if `values` is not given and a column in `columns` has
                no map or function in `value_maps`.
            TypeError: if a map in `value_maps` is neither a mapping nor
                callable.

        Returns:
            The data, with the new values. The type of a changed column is
                found from its new values, as `pandas` does.

        """
        if values:
            mapping = dict(values)
            for column in utilities.select_columns(data, columns):
                data[column] = _recode(data[column], mapping)
            return data
        if columns is None:
            names = [c for c in self.value_maps if c in data.columns]
        else:
            names = utilities.select_columns(data, columns)
        unknown = [c for c in names if c not in self.value_maps]
        if unknown:
            message = (
                f'{self.name} has no values for the columns {unknown}: pass '
                f'"values" or subclass MapValues with "value_maps" for them'
            )
            raise KeyError(message)
        for column in names:
            mapped = self.value_maps[column]
            if isinstance(mapped, Mapping):
                data[column] = _recode(data[column], dict(mapped))
            elif callable(mapped):
                data[column] = data[column].map(mapped, na_action = 'ignore')
            else:
                message = (
                    f'the values for {column!r} must be a dict or a '
                    f'function, not {type(mapped).__name__}'
                )
                raise TypeError(message)
        return data


""" Private Functions """


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
            f'{column!r} is not a flag (true or false): clean_booleans can '
            f'make it one'
        )
        raise TypeError(message)
    return np.asarray(values.astype(float).fillna(0).to_numpy() != 0)
