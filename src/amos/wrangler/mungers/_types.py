"""Mungers that change the types of columns.

Contents:
    _KINDS: the kinds that `SetKinds` sets, and the kind of `Dataset` column
        that each is (see `Dataset.kinds`).
    _FLOAT32_TOLERANCE: relative difference allowed between a number and the
        same number as a `float32`, which `SetKinds` uses if every number of a
        column is within it.
    _LARGEST_INTEGER: whole numbers stored as floats at or above this size are
        not made integers by `SetKinds`, since `int64` cannot hold them.
    AutoCategorize: makes columns with few unique values categorical.
    ConvertTypes: changes the data types of columns.
    SetKinds: sets the kind of each column (dummy, categorical, or numerical),
        with the smallest type that holds it.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import camina
import numpy as np
import pandas as pd

from ...core import base, options, utilities
from ._base import Munger

# The kinds that `SetKinds` sets, and the kind of `Dataset` column that each
# is (see `Dataset.kinds`).
_KINDS: dict[str, str] = {
    'dummy': 'booleans',
    'categorical': 'categoricals',
    'numerical': 'numerics'}
# Relative difference allowed between a number and the same number as a
# `float32`, which `SetKinds` uses if every number of a column is within it.
_FLOAT32_TOLERANCE: float = 1e-6
# Whole numbers stored as floats at or above this size are not made integers
# by `SetKinds`, since `int64` cannot hold them.
_LARGEST_INTEGER: float = 2.0 ** 63


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
        threshold: int | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Makes columns with `threshold` unique values or fewer categorical.

        Args:
            data: the data to change.
            columns: columns to check. Defaults to `None`, which checks every
                column.
            threshold: most unique values that a categorical column can have.
                Defaults to `None`, which uses `options._CATEGORY_THRESHOLD`
                (read when the munger is applied).
            **kwargs: not used.

        Returns:
            The data, with categorical columns.

        """
        if threshold is None:
            threshold = options._CATEGORY_THRESHOLD
        for column in utilities.select_columns(data, columns):
            values = data[column]
            if pd.api.types.is_bool_dtype(values.dtype):
                continue
            if values.nunique(dropna = True) <= threshold:
                data[column] = values.astype('category')
        return data


@dataclasses.dataclass
class ConvertTypes(Munger):
    """Changes the data types of columns.

    Wraps:
        [`pandas.DataFrame.astype`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html)
        from pandas.

    """

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
class SetKinds(Munger):
    """Sets the kind of each column, with the smallest type that holds it.

    The kinds are "dummy" (0 and 1, or false and true), "categorical", and
    "numerical". Each is a parameter that lists its columns, as in "dummy =
    rape, murder". A column that is not listed gets the kind of its values:
    a boolean column, or a column of numbers that are all 0 or 1 (with none
    missing), is a dummy; any other column of numbers is numerical; and a
    categorical column, or text with `options._CATEGORY_THRESHOLD` unique
    values or fewer, is categorical. Other columns (such as dates and longer
    text) and the groups are not changed unless they are listed. If the label
    is not listed, it is a dummy when it is classified and its values are
    booleans (or 0 and 1), and is otherwise not changed.

    Each kind is stored in the smallest type that holds its values exactly:

    * dummy: `int8`, with 0 for false and 1 for true.
    * categorical: the `pandas` "category" type, whose codes are the smallest
      integer type that holds them.
    * numerical: the smallest signed integer type (such as `int8`) if every
      value is a whole number and none is missing. Otherwise `float32` if it
      holds every value to about 7 significant digits, and `float64` if it
      does not.

    The kinds are kept in `Dataset.kinds`, so a dummy is one of the dataset's
    `booleans` (not its `numerics`). Arithmetic on small integer types can
    overflow (`int8` only holds -128 to 127), so make new columns from
    numbers (as `derive_columns` does) before setting their kinds.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        kinds: Mapping[str, str] | None = None,
        **listed: Any) -> pd.DataFrame:
        """Gives each column the smallest type of its kind.

        Args:
            data: the data to change.
            kinds: the kind of each column to change ("booleans",
                "categoricals", or "numerics"), as `apply` finds them.
                Defaults to `None`, in which case they are found from
                `listed`, without a label.
            **listed: the columns of each kind ("dummy", "categorical", and
                "numerical").

        Returns:
            The data, with the new types.

        """
        if kinds is None:
            kinds = _find_kinds(data, listed)
        for column, kind in kinds.items():
            data[column] = _shrink(data[column], column, kind)
        return data

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Sets the kinds of the columns of `item`, and keeps them in `kinds`.

        Args:
            item: the dataset to change.
            **kwargs: the columns of each kind ("dummy", "categorical", and
                "numerical").

        Returns:
            The changed dataset.

        """
        kinds = _find_kinds(
            item.data,
            kwargs,
            label = item.label if item.has_label else None,
            task = item.task,
            groups = item.groups)
        item = super().apply(item, kinds = kinds)
        item.kinds.update(kinds)
        return item


""" Private Functions """


def _find_kinds(
    data: pd.DataFrame,
    listed: Mapping[str, Any],
    *,
    label: str | None = None,
    task: str | None = None,
    groups: Sequence[str] = ()) -> dict[str, str]:
    """Returns the kind of each column that `SetKinds` changes.

    Args:
        data: the data.
        listed: the columns of each kind ("dummy", "categorical", and
            "numerical"), as a column name or a sequence of them.
        label: name of the label, if it is a column. Defaults to `None`.
        task: the task of the dataset. Defaults to `None`.
        groups: names of the group columns, which are only changed if they
            are listed. Defaults to an empty tuple.

    Raises:
        KeyError: if a listed column is not in `data`.
        ValueError: if a kind is not known, or a column is listed as two
            kinds.

    Returns:
        "booleans", "categoricals", or "numerics" for each column to change,
            by its name.

    """
    unknown = [k for k in listed if k not in _KINDS]
    if unknown:
        message = (
            f'set_kinds does not know the kinds {unknown}: they must be '
            f'{sorted(_KINDS)}'
        )
        raise ValueError(message)
    kinds: dict[str, str] = {}
    for name, columns in listed.items():
        if columns is None:
            continue
        for column in utilities.select_columns(data, camina.listify(columns)):
            if kinds.get(column, _KINDS[name]) != _KINDS[name]:
                message = f'set_kinds lists {column!r} as two kinds'
                raise ValueError(message)
            kinds[column] = _KINDS[name]
    for column in data.columns:
        if column in kinds or column in groups:
            continue
        values = data[column]
        if column == label:
            if task == 'classify' and _is_dummy(values):
                kinds[column] = 'booleans'
            continue
        found = _infer_kind(values)
        if found is not None:
            kinds[column] = found
    return kinds


def _infer_kind(values: pd.Series) -> str | None:
    """Returns the kind that `SetKinds` gives a column that is not listed.

    Args:
        values: a column of data.

    Returns:
        "booleans" for a dummy, "numerics" for other numbers, "categoricals"
            for a categorical column or text with few unique values, or
            `None` for a column that is not changed (such as dates or text
            with many unique values).

    """
    dtype = values.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return 'categoricals'
    if _is_dummy(values):
        return 'booleans'
    if pd.api.types.is_bool_dtype(dtype):
        # A boolean column with missing values cannot be a dummy.
        return None
    if pd.api.types.is_numeric_dtype(dtype):
        return 'numerics'
    if base._kind_of(values) == 'dates':
        return None
    if values.nunique(dropna = True) <= options._CATEGORY_THRESHOLD:
        return 'categoricals'
    return None


def _is_dummy(values: pd.Series) -> bool:
    """Returns whether `values` are all 0 or 1 (or false or true).

    Args:
        values: a column of data.

    Returns:
        Whether no value is missing and every value is 0, 1, `False`, or
            `True`.

    """
    if isinstance(values.dtype, pd.CategoricalDtype) or values.isna().any():
        return False
    unique = pd.unique(values.to_numpy())
    # `True` and `False` equal 1 and 0, but text such as "1" does not.
    return all(
        isinstance(v, bool | int | float | np.bool_ | np.number)
        and v in {0, 1}
        for v in unique)


def _shrink(values: pd.Series, column: Any, kind: str) -> pd.Series:
    """Returns `values` in the smallest type of `kind` that holds them.

    Args:
        values: a column of data.
        column: name of the column, for error messages.
        kind: "booleans" (a dummy), "categoricals", or "numerics".

    Raises:
        ValueError: if a dummy has missing values or values other than 0 and
            1 (or false and true), or a numerical column has values that are
            not numbers.

    Returns:
        The column, with its new type.

    """
    if kind == 'categoricals':
        if isinstance(values.dtype, pd.CategoricalDtype):
            return values
        return values.astype('category')
    if kind == 'booleans':
        if values.isna().any():
            message = f'the dummy {column!r} has missing values'
            raise ValueError(message)
        if not _is_dummy(values):
            examples = [
                v for v in pd.unique(values.to_numpy()) if v not in {0, 1}][:5]
            message = (
                f'the dummy {column!r} has values other than 0 and 1 (or '
                f'false and true), such as {examples}'
            )
            raise ValueError(message)
        return values.astype('int8')
    return _shrink_numbers(values, column)


def _shrink_numbers(values: pd.Series, column: Any) -> pd.Series:
    """Returns numbers in the smallest type that holds them exactly.

    Whole numbers with none missing become the smallest signed integer type.
    Other numbers become `float32` if each is within `_FLOAT32_TOLERANCE` of
    itself as a `float32`, and `float64` otherwise.

    Args:
        values: a column of numbers (or of text or categories that are
            numbers).
        column: name of the column, for error messages.

    Raises:
        ValueError: if a value is not a number.

    Returns:
        The column, with its new type.

    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype(object)
    if pd.api.types.is_bool_dtype(values.dtype):
        values = values.astype('Int8' if values.isna().any() else 'int8')
    try:
        numbers = pd.to_numeric(values)
    except (TypeError, ValueError) as error:
        message = (
            f'the numerical column {column!r} has values that are not numbers')
        raise ValueError(message) from error
    missing = bool(numbers.isna().any())
    if pd.api.types.is_integer_dtype(numbers.dtype) and not missing:
        return pd.to_numeric(numbers.astype('int64'), downcast = 'integer')
    floats = numbers.astype('float64')
    whole = (
        not missing
        and bool(np.isfinite(floats).all())
        and bool((floats == floats.round()).all())
        and (floats.empty or float(floats.abs().max()) < _LARGEST_INTEGER))
    if whole:
        return pd.to_numeric(floats.astype('int64'), downcast = 'integer')
    single = floats.astype('float32')
    if np.allclose(
        single.to_numpy(dtype = 'float64'),
        floats.to_numpy(),
        rtol = _FLOAT32_TOLERANCE,
        atol = 0,
        equal_nan = True):
        return single
    return floats
