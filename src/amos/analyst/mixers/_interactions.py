"""A mixer that multiplies features together.

Contents:
    Interactions: mixers.

"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ...core import base, utilities
from ._base import Mixer

""" Mixers """


@dataclasses.dataclass
class Interactions(Mixer):
    """Adds the product of each pair of features, or of the terms of formulas.

    Without formulas, it replaces the numeric features with the features and
    the product of each pair of them, as `PolynomialFeatures` does.

    A formula names the columns to interact, joined by "*", as in "race =
    offender_race * victim_race". Each parameter whose value is a formula
    (and each item of a "formulas" `dict`) adds an interaction named for it,
    and keeps the columns it is made from:

    * Categorical columns make one categorical column, whose categories are
      the combinations of their values, each followed by the part of its
      column's name that differs from the others' (so "W" and "B" make
      "w_offender_b_victim"). An encoder can then encode it like any other.
    * Numbers (including booleans and dummies) make their product.
    * Numbers and categorical columns make the product of the numbers for
      each combination of categories, as columns named for the interaction
      and the combination, which are 0 for the other rows.

    A row with a missing value in any column has a missing interaction. It
    learns nothing from the data, so it can be used before the data is split.

    Wraps:
        [`sklearn.preprocessing.PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
        from scikit-learn, when there are no formulas.

    """

    contents: str = 'sklearn.preprocessing.PolynomialFeatures'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {
            'interaction_only': True,
            'include_bias': False})

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        columns: Sequence[str] | None = None,
        formulas: Mapping[str, str] | None = None,
        **kwargs: Any) -> base.Dataset:
        """Adds the interactions of the formulas, or of each pair of features.

        Args:
            item: the dataset to change.
            columns: names of the columns to interact when there are no
                formulas. Defaults to `None`, which uses `select`.
            formulas: formulas (such as "a * b"), by the names of the
                interactions. Defaults to `None`.
            **kwargs: more formulas (any `str` with "*"), by the names of the
                interactions, and parameters for `PolynomialFeatures` (which
                are only used when there are no formulas).

        Returns:
            The changed dataset.

        """
        named = {
            k: v for k, v in kwargs.items()
            if isinstance(v, str) and '*' in v}
        formulas = {**dict(formulas or {}), **named}
        if not formulas:
            return super().apply(item, columns = columns, **kwargs)
        made = [
            _interact(item, name, formula)
            for name, formula in formulas.items()]
        values = pd.concat(made, axis = 1)
        item.update_features([], values)
        item.record(
            self.name,
            formulas = formulas,
            created = list(values.columns))
        return item


""" Private Functions """


def _interact(item: base.Dataset, name: str, formula: str) -> pd.DataFrame:
    """Returns the interaction of the columns in `formula` (see `Interactions`).

    Args:
        item: the dataset.
        name: name of the interaction.
        formula: names of two or more columns, joined by "*". A name can be
            in backticks.

    Raises:
        KeyError: if a column of `formula` is not in the data.
        ValueError: if `formula` has fewer than two columns, or a column is
            dates or times.

    Returns:
        The interaction: one categorical column or column of numbers named
            `name`, or a column of numbers for each combination of categories.

    """
    terms = [t.strip().strip('`') for t in formula.split('*')]
    if len(terms) < 2 or not all(terms):  # noqa: PLR2004
        message = (
            f'the interaction {name!r} must join two or more columns with '
            f'"*", not {formula!r}'
        )
        raise ValueError(message)
    utilities.select_columns(item.data, terms)
    categories = [
        t for t in terms
        if base._kind_of(item.data[t]) == 'categoricals'
        and item.kinds.get(t) != 'booleans']
    numbers = [t for t in terms if t not in categories]
    dates = [t for t in numbers if base._kind_of(item.data[t]) == 'dates']
    if dates:
        message = f'the interaction {name!r} cannot use dates: {dates}'
        raise ValueError(message)
    index = item.data.index
    product = None
    for term in numbers:
        values = item.data[term].astype('float64')
        product = values if product is None else product * values
    if not categories:
        return pd.DataFrame({name: product}, index = index)
    parts = []
    for term, role in zip(categories, _roles(categories), strict = True):
        column = item.data[term]
        slugs = {v: _slug(v) for v in pd.unique(column.dropna().to_numpy())}
        parts.append(
            column.astype(object).map(slugs).astype('string') + f'_{role}')
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + '_' + part
    if product is None:
        return pd.DataFrame(
            {name: combined.astype('category')}, index = index)
    columns = {}
    for level in sorted(combined.dropna().unique()):
        matched = (combined == level).astype('float64')
        columns[f'{name}_{level}'] = product * matched.mask(combined.isna())
    return pd.DataFrame(columns, index = index)


def _roles(columns: Sequence[str]) -> list[str]:
    """Returns the part of each column's name that differs from the others'.

    The words (separated by "_") that every name starts or ends with are
    removed, so "offender_race" and "victim_race" are "offender" and
    "victim". A name that would be empty, or a single column, keeps its
    whole name.

    Args:
        columns: names of the columns.

    Returns:
        The part of each name that tells it apart, in order.

    """
    words = [str(c).split('_') for c in columns]
    if len(words) < 2:  # noqa: PLR2004
        return [str(c) for c in columns]
    shortest = min(len(w) for w in words)
    start = 0
    while start < shortest and len({w[start] for w in words}) == 1:
        start += 1
    end = 0
    while end < shortest - start and len({w[-1 - end] for w in words}) == 1:
        end += 1
    roles = ['_'.join(w[start:len(w) - end]) for w in words]
    if not all(roles):
        return [str(c) for c in columns]
    return roles


def _slug(value: Any) -> str:
    """Returns `value` as lowercase text with "_" in place of punctuation.

    Args:
        value: a value of a categorical column.

    Returns:
        The value as text that can be part of a name, such as "white" for
            "White".

    """
    return re.sub(r'\W+', '_', str(value).strip().lower()).strip('_')
