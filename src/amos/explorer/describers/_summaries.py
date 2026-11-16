"""Describers that summarize columns.

Contents:
    Describe: the `pandas` description of every column.
    Frequencies: the count and share of each value of the categorical columns.
    LabelBalance: the count and share of each value of the label.
    MissingValues: the count and share of missing values in each column.
    Summarize: summary statistics of the numeric columns.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pandas as pd

from ...core import base
from ._base import Describer


@dataclasses.dataclass
class Describe(Describer):
    """Tabulates `pandas`' description of every column, one row per column.

    Wraps:
        [`pandas.DataFrame.describe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)
        from pandas.

    """

    visual: ClassVar[str] = 'distribution_plots'

    def describe(self, item: base.Dataset, **kwargs: Any) -> pd.DataFrame:
        """Returns `DataFrame.describe` for every column, transposed.

        Args:
            item: the dataset to describe.
            **kwargs: not used.

        Returns:
            One row for each column, with counts, unique values, means,
                quartiles, and so on (as each applies).

        """
        return item.data.describe(include = 'all').T

    """ Private Methods """

    def _visual_parameters(
        self,
        item: base.Dataset,
        parameters: Mapping[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        """Returns the parameters of `distribution_plots`.

        Args:
            item: the dataset, with the technique's table.
            parameters: the parameters that the technique was applied with.

        Returns:
            The columns that were described (every column of the data).

        """
        return {'columns': list(item.tables[self.name].index)}


@dataclasses.dataclass
class Frequencies(Describer):
    """Counts each value of the categorical columns, with its share."""

    visual: ClassVar[str] = 'frequency_plots'

    def describe(
        self,
        item: base.Dataset,
        columns: Sequence[str] | str | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the count and share of each value in `columns`.

        Args:
            item: the dataset to describe.
            columns: columns to count, or "all". Defaults to `None`, which
                (like "all") counts the categorical and boolean features,
                including the dummies of `set_kinds`.
            **kwargs: not used.

        Returns:
            One row for each value of each column, with "column", "value",
                "count", and "share" columns.

        """
        if columns is None or columns == 'all':
            columns = item.categoricals + item.booleans
        elif isinstance(columns, str):
            columns = [columns]
        tables = []
        for column in columns:
            counts = item.data[column].value_counts(dropna = False)
            tables.append(pd.DataFrame({
                'column': column,
                'value': counts.index.astype(str),
                'count': counts.to_numpy(),
                'share': (counts / counts.sum()).to_numpy()}))
        if not tables:
            return pd.DataFrame(columns = ['column', 'value', 'count', 'share'])
        return pd.concat(tables, ignore_index = True)


@dataclasses.dataclass
class LabelBalance(Describer):
    """Counts each value of the label, with its share."""

    visual: ClassVar[str] = 'label_plot'

    def describe(self, item: base.Dataset, **kwargs: Any) -> pd.DataFrame:
        """Returns the count and share of each value of the label.

        Args:
            item: the dataset to describe. It must have a label.
            **kwargs: not used.

        Returns:
            One row for each value of the label, with "count" and "share"
                columns.

        """
        counts = item.y.value_counts(dropna = False).sort_index()
        return pd.DataFrame({
            'count': counts,
            'share': counts / counts.sum()})

    """ Private Methods """

    def _visual_parameters(
        self,
        item: base.Dataset,  # noqa: ARG002
        parameters: Mapping[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        """Returns the parameters of `label_plot`, which draws the label.

        Args:
            item: the dataset, with the technique's table.
            parameters: the parameters that the technique was applied with.

        Returns:
            No parameters.

        """
        return {}


@dataclasses.dataclass
class MissingValues(Describer):
    """Counts the missing values in each column, with their share."""

    visual: ClassVar[str] = 'missing_bars'

    def describe(self, item: base.Dataset, **kwargs: Any) -> pd.DataFrame:
        """Returns the count and share of missing values in each column.

        Args:
            item: the dataset to describe.
            **kwargs: not used.

        Returns:
            One row for each column, with "missing" and "share" columns.

        """
        missing = item.data.isna().sum()
        return pd.DataFrame({
            'missing': missing,
            'share': missing / max(len(item.data), 1)})


@dataclasses.dataclass
class Summarize(Describer):
    """Tabulates the summary statistics of numeric columns that papers report.

    Beyond `pandas.DataFrame.describe`, this includes the number of missing
    values, the median, the skew, and the kurtosis of each column.

    """

    visual: ClassVar[str] = 'box_plots'

    def describe(self, item: base.Dataset, **kwargs: Any) -> pd.DataFrame:
        """Returns summary statistics for the numeric and boolean columns.

        Args:
            item: the dataset to describe.
            **kwargs: not used.

        Returns:
            One row for each numeric or boolean column (including a numeric
                label).

        """
        columns = item.numerics + item.booleans
        if item.label is not None and pd.api.types.is_numeric_dtype(
            item.y.dtype):
            columns.append(item.label)
        data = item.data[columns].astype(float)
        return pd.DataFrame({
            'count': data.count(),
            'missing': data.isna().sum(),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'q1': data.quantile(0.25),
            'median': data.median(),
            'q3': data.quantile(0.75),
            'max': data.max(),
            'skew': data.skew(),
            'kurtosis': data.kurt()})

    """ Private Methods """

    def _visual_parameters(
        self,
        item: base.Dataset,
        parameters: Mapping[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        """Returns the parameters of `box_plots`.

        Args:
            item: the dataset, with the technique's table.
            parameters: the parameters that the technique was applied with.

        Returns:
            The numeric columns that were summarized (booleans are left out,
                since a box of 0s and 1s says little), in boxes that are not
                split.

        """
        columns = [
            c for c in item.tables[self.name].index if c not in item.booleans]
        return {'columns': columns, 'split': False}
