"""Techniques that describe data with tables.

These are the techniques of the "explorer" stage of a project. Each one adds a
table to the dataset's `tables` (under the technique's name) and leaves the
data unchanged. They describe every row, so use them before the data is split
or to describe the data as it was collected.

Contents:
    Describer: base class for techniques that describe data with a table.
    Correlations: correlations between the numeric columns.
    Describe: the `pandas` description of every column.
    Frequencies: the count and share of each value of the categorical columns.
    KaplanMeier: the share of rows without an event over time (survival).
    LabelBalance: the count and share of each value of the label.
    MissingValues: the count and share of missing values in each column.
    Summarize: summary statistics of the numeric columns.
    survival_curves: fits a Kaplan-Meier estimator for each group.

"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import pandas as pd

from . import base, utilities

# Methods of correlation that `pandas` supports.
CorrelationMethod: TypeAlias = Literal['pearson', 'kendall', 'spearman']


@dataclasses.dataclass
class Describer(base.Operation, abc.ABC):
    """Base class for techniques that describe data with a table.

    A subclass writes a `describe` method, which returns a `DataFrame`. It is
    stored in the dataset's `tables` under the technique's name.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: not used by most describers. Defaults to `None`.
        parameters: keyword arguments for `describe`. Defaults to an empty
            `dict`.

    """

    """ Required Methods """

    @abc.abstractmethod
    def describe(self, item: base.Dataset, **kwargs: Any) -> pd.DataFrame:
        """Returns a table that describes `item`.

        Args:
            item: the dataset to describe.
            **kwargs: parameters for the description.

        Returns:
            The table.

        """

    """ Public Methods """

    def implement(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Adds the table from `describe` to the tables of `item`.

        Args:
            item: the dataset to describe.
            **kwargs: parameters for `describe`.

        Returns:
            The dataset, with the new table.

        """
        item.tables[self.name] = self.describe(item, **kwargs)
        item.record(self.name, table = self.name)
        return item


@dataclasses.dataclass
class Correlations(Describer):
    """Correlations between the numeric columns (including the label)."""

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


@dataclasses.dataclass
class Describe(Describer):
    """The `pandas` description of every column, one row per column."""

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


@dataclasses.dataclass
class Frequencies(Describer):
    """The count and share of each value of the categorical columns."""

    def describe(
        self,
        item: base.Dataset,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the count and share of each value in `columns`.

        Args:
            item: the dataset to describe.
            columns: columns to count. Defaults to `None`, which counts the
                categorical and boolean features.
            **kwargs: not used.

        Returns:
            One row for each value of each column, with "column", "value",
                "count", and "share" columns.

        """
        if columns is None:
            columns = item.categoricals + item.booleans
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
class KaplanMeier(Describer):
    """The share of rows without an event over time (a survival curve).

    The label is the time until the event (such as rearrest) or until the row
    stopped being observed. Set "event" to the column that is 1 if the event
    happened and 0 if the row was censored (by default, every event was
    observed), and "group" to a column to make a curve for each group. It
    wraps lifelines' `KaplanMeierFitter`.

    """

    def describe(
        self,
        item: base.Dataset,
        event: str | None = None,
        group: str | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the estimated survival at each time.

        Args:
            item: the dataset to describe. Its label is the time.
            event: name of the column that is 1 if the event happened.
                Defaults to `None`, which means every event was observed.
            group: name of a column of groups. Defaults to `None`, which
                makes one curve for every row.
            **kwargs: not used.

        Returns:
            One row for each time in each group, with "group", "time",
                "survival" (the share without the event), and "at_risk".

        """
        tables = []
        for name, fitter in survival_curves(item, event, group).items():
            survival = fitter.survival_function_
            tables.append(pd.DataFrame({
                'group': name,
                'time': survival.index.to_numpy(),
                'survival': survival.iloc[:, 0].to_numpy(),
                'at_risk': fitter.event_table['at_risk'].to_numpy()}))
        return pd.concat(tables, ignore_index = True)


@dataclasses.dataclass
class LabelBalance(Describer):
    """The count and share of each value of the label."""

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


@dataclasses.dataclass
class MissingValues(Describer):
    """The count and share of missing values in each column."""

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
    """Summary statistics of the numeric columns, as reported in papers.

    Beyond `pandas.DataFrame.describe`, this includes the number of missing
    values, the median, the skew, and the kurtosis of each column.

    """

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


""" Public Functions """


def survival_curves(
    item: base.Dataset,
    event: str | None = None,
    group: str | None = None) -> dict[str, Any]:
    """Returns a fitted Kaplan-Meier estimator for each group.

    `kaplan_meier` and `survival_curves` (in the plots) both use this.

    Args:
        item: the dataset. Its label is the time until the event.
        event: name of the column that is 1 if the event happened. Defaults
            to `None`, which means every event was observed.
        group: name of a column of groups. Defaults to `None`, which makes
            one estimate, called "all", for every row.

    Returns:
        The fitted `lifelines.KaplanMeierFitter` of each group, by its name.

    """
    lifelines = utilities.import_tool('lifelines')
    times = item.y
    observed = None if event is None else item.data[event].astype(int)
    if group is None:
        groups = {'all': item.data.index}
    else:
        groups = {
            str(name): rows
            for name, rows in item.data.groupby(group).groups.items()}
    fitters = {}
    for name, rows in groups.items():
        fitters[name] = lifelines.KaplanMeierFitter(label = name).fit(
            times.loc[rows], None if observed is None else observed.loc[rows])
    return fitters
