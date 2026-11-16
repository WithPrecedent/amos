"""Describers of how columns go together.

Contents:
    CorrelationMethod: methods of correlation that `pandas` supports.
    AssociationRules: rules of which values tend to appear in the same rows.
    Correlations: correlations between the numeric columns.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np
import pandas as pd

from ...core import base, utilities
from ._base import Describer

# Methods of correlation that `pandas` supports.
CorrelationMethod: TypeAlias = Literal['pearson', 'kendall', 'spearman']


@dataclasses.dataclass
class AssociationRules(Describer):
    """Finds rules of which values tend to appear in the same rows.

    mlxtend's association rules, as in an analysis of shopping baskets: each
    row is a set of items, which are the values of the categorical columns
    (as "{column} = {value}") and the boolean columns that are true, and of
    the label if it is classified. The sets of up to `max_len` items in at
    least `min_support` of the rows are found with FP-growth, and each rule
    "antecedents, then consequents" is kept if its `metric` is at least
    `min_threshold` (by default, a confidence of 0.8: 80% of the rows with
    the antecedents also have the consequents). The table has a row for each
    rule, with mlxtend's measures of it, largest lift first (how many times
    more often the consequents appear with the antecedents than they do in
    every row).

    Wraps:
        - [`mlxtend.frequent_patterns.fpgrowth`](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/)
          from mlxtend, for the sets of items.
        - [`mlxtend.frequent_patterns.association_rules`](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)
          from mlxtend, for the rules.

    """

    visual: ClassVar[str] = 'association_rules_plot'

    def describe(
        self,
        item: base.Dataset,
        *,
        columns: Sequence[str] | None = None,
        min_support: float = 0.1,
        metric: str = 'confidence',
        min_threshold: float = 0.8,
        max_len: int | None = 3,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the association rules of the values of `columns`.

        Args:
            item: the dataset to describe.
            columns: the columns whose values are items. Defaults to `None`,
                which uses the categorical and boolean features, and the
                label if it is classified.
            min_support: smallest share of the rows that a set of items must
                be in. Defaults to 0.1.
            metric: the measure that a rule must reach, such as "confidence",
                "lift", "leverage", or "conviction". Defaults to
                "confidence".
            min_threshold: smallest value of `metric` for a rule. Defaults to
                0.8.
            max_len: most items in a set (of the antecedents and consequents
                together), or `None` for any number. Defaults to 3.
            **kwargs: not used.

        Raises:
            ValueError: if no set of items is in `min_support` of the rows.

        Returns:
            One row for each rule, with its "antecedents" and "consequents"
                (each joined by commas) and mlxtend's measures of it.

        """
        frequent = utilities.import_tool('mlxtend.frequent_patterns')
        if columns is None:
            columns = [*item.categoricals, *item.booleans]
            if item.label is not None and item.task == 'classify':
                columns.append(item.label)
        items = _items(item.data, utilities.select_columns(item.data, columns))
        found = frequent.fpgrowth(
            items, min_support = min_support, use_colnames = True,
            max_len = max_len)
        if found.empty:
            message = (
                f'no set of items is in {min_support:.0%} of the rows: lower '
                f'"min_support"')
            raise ValueError(message)
        rules = frequent.association_rules(
            found, len(items), metric = metric, min_threshold = min_threshold)
        for side in ('antecedents', 'consequents'):
            rules[side] = [', '.join(sorted(s)) for s in rules[side]]
        table: pd.DataFrame = rules.sort_values(
            ['lift', 'confidence'], ascending = False, kind = 'stable',
            ignore_index = True)
        return table


@dataclasses.dataclass
class Correlations(Describer):
    """Tabulates the correlations between the numeric columns and the label.

    Wraps:
        [`pandas.DataFrame.corr`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)
        from pandas.

    """

    visual: ClassVar[str] = 'correlation_heatmap'

    def describe(
        self,
        item: base.Dataset,
        method: CorrelationMethod = 'pearson',
        **kwargs: Any) -> pd.DataFrame:
        """Returns the correlations between the numeric and boolean columns.

        Args:
            item: the dataset to describe.
            method: "pearson", "spearman", or "kendall". Defaults to
                "pearson".
            **kwargs: not used.

        Returns:
            A square table of correlations.

        """
        columns = item.numerics + item.booleans
        if item.label is not None and pd.api.types.is_numeric_dtype(
            item.y.dtype):
            columns.append(item.label)
        return item.data[columns].astype(float).corr(method = method)


""" Private Functions """


def _items(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Returns the items of each row, as mlxtend's frequent patterns need them.

    Args:
        data: the data.
        columns: the columns whose values are items.

    Returns:
        A boolean column for each item: a boolean column of `data` (true
            where it is true), or a value of another column (named
            "{column} = {value}"). A missing value is no item.

    """
    items: dict[str, np.ndarray] = {}
    for column in columns:
        values = data[column]
        if pd.api.types.is_bool_dtype(values.dtype):
            items[str(column)] = values.fillna(value = False).to_numpy(
                dtype = bool)
            continue
        values = values.astype(object)
        for value in sorted(values.dropna().unique(), key = str):
            items[f'{column} = {value}'] = (values == value).to_numpy(
                dtype = bool)
    return pd.DataFrame(items, index = data.index)
