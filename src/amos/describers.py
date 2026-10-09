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
    survival_steps: the steps of a Kaplan-Meier curve, with its confidence
        interval.

"""

from __future__ import annotations

import abc
import dataclasses
import statistics
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
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
    """Correlations between the numeric columns (including the label).

    Wraps:
        [`pandas.DataFrame.corr`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)
        from pandas.

    """

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
    """The `pandas` description of every column, one row per column.

    Wraps:
        [`pandas.DataFrame.describe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)
        from pandas.

    """

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
    wraps statsmodels' `SurvfuncRight`.

    Wraps:
        [`statsmodels.duration.survfunc.SurvfuncRight`](https://www.statsmodels.org/stable/generated/statsmodels.duration.survfunc.SurvfuncRight.html)
        from statsmodels.

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
            One row for each time in each group (0, each time with an event,
                and the last time), with "group", "time", "survival" (the
                share without the event), and "at_risk".

        """
        tables = []
        for name, curve in survival_curves(item, event, group).items():
            steps = survival_steps(curve)
            steps.insert(0, 'group', name)
            tables.append(steps[['group', 'time', 'survival', 'at_risk']])
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

    Wraps:
        [`statsmodels.duration.survfunc.SurvfuncRight`](https://www.statsmodels.org/stable/generated/statsmodels.duration.survfunc.SurvfuncRight.html)
        from statsmodels.

    Args:
        item: the dataset. Its label is the time until the event.
        event: name of the column that is 1 if the event happened. Defaults
            to `None`, which means every event was observed.
        group: name of a column of groups. Defaults to `None`, which makes
            one estimate, called "all", for every row.

    Returns:
        The fitted statsmodels `SurvfuncRight` of each group, by its name.

    """
    survfunc = utilities.import_tool('statsmodels.api.SurvfuncRight')
    times = item.y.astype(float)
    if event is None:
        observed = pd.Series(1, index = item.data.index)
    else:
        observed = item.data[event].astype(int)
    if group is None:
        groups = {'all': item.data.index}
    else:
        groups = {
            str(name): rows
            for name, rows in item.data.groupby(group).groups.items()}
    return {
        name: survfunc(
            times.loc[rows].to_numpy(), observed.loc[rows].to_numpy(),
            title = name)
        for name, rows in groups.items()}


def survival_steps(curve: Any) -> pd.DataFrame:
    """Returns the steps of a Kaplan-Meier curve from `survival_curves`.

    The curve starts at a time of 0 and ends at the last time that a row was
    observed. The confidence interval is the usual 95% interval of the log of
    minus the log of the survival, with Greenwood's variance.

    Args:
        curve: a fitted statsmodels `SurvfuncRight`.

    Returns:
        One row for each time (0, each time with an event, and the last time),
            with "time", "survival", "ci_lower", "ci_upper", and "at_risk"
            (the number of rows whose time is at least that time).

    """
    observed = np.sort(np.asarray(curve.time, dtype = float))
    events = np.asarray(curve.surv_times, dtype = float)
    survival = np.asarray(curve.surv_prob, dtype = float)
    at_risk = np.asarray(curve.n_risk, dtype = float)
    happened = np.asarray(curve.n_events, dtype = float)
    z = statistics.NormalDist().inv_cdf(0.975)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        greenwood = np.cumsum(happened / (at_risk * (at_risk - happened)))
        logged = np.log(survival)
        spread = z * np.sqrt(greenwood) / -logged
        lower = np.exp(-np.exp(np.log(-logged) + spread))
        upper = np.exp(-np.exp(np.log(-logged) - spread))
    # Once every row at risk has had the event, the survival is surely 0.
    lower[survival == 0] = 0.0
    upper[survival == 0] = 0.0
    times = np.unique(np.concatenate([[0.0], events, observed[-1:]]))
    # Before the first event, nobody has had it. Afterward, each time takes
    # the step of the last event at or before it.
    steps = np.searchsorted(events, times, side = 'right')
    return pd.DataFrame({
        'time': times,
        'survival': np.concatenate([[1.0], survival])[steps],
        'ci_lower': np.concatenate([[1.0], lower])[steps],
        'ci_upper': np.concatenate([[1.0], upper])[steps],
        'at_risk': len(observed) - np.searchsorted(observed, times)})
