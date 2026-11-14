"""Techniques that describe data with tables.

These are the techniques of the "explorer" stage of a project. Each one adds a
table to the dataset's `tables` (under the technique's name) and leaves the
data unchanged. They describe every row, so use them before the data is split
or to describe the data as it was collected. Pass `visualize = True` (or set
"visualize_explorers" in the "general" section of the settings) and each one
also draws its table as a figure in the dataset's `figures`, under its name
(see `base.Visualizer`).

The explorers whose names start with "is_" test whether each column fits a
distribution (or is skewed), with the usual test of each, and
`distributions` names the distribution that fits each column best.

Contents:
    Describer: base class for techniques that describe data with a table.
    DistributionTest: base class for explorers that test whether columns fit
        a distribution.
    AssociationRules: rules of which values tend to appear in the same rows.
    Correlations: correlations between the numeric columns.
    Describe: the `pandas` description of every column.
    Distributions: the distribution that fits each column best, and its
        skew.
    Frequencies: the count and share of each value of the categorical columns.
    IsBernoulli: whether each column is a Bernoulli distribution (0s and 1s).
    IsBeta: whether each column fits a beta distribution.
    IsBimodal: whether each column has more than one peak (Hartigan's dip
        test).
    IsBinomial: whether each column fits a binomial distribution.
    IsExponential: whether each column fits an exponential distribution.
    IsLeftSkewed: whether each column is skewed to the left.
    IsLogNormal: whether each column fits a log-normal distribution.
    IsNormal: whether each column fits a normal distribution.
    IsPoisson: whether each column fits a Poisson distribution.
    IsRightSkewed: whether each column is skewed to the right.
    IsStudent: whether each column fits a Student's t distribution.
    IsUniform: whether each column fits a uniform distribution.
    KaplanMeier: the share of rows without an event over time (survival).
    LabelBalance: the count and share of each value of the label.
    MissingValues: the count and share of missing values in each column.
    Summarize: summary statistics of the numeric columns.
    best_distribution: the distribution whose fit to some values has the
        lowest AIC.
    fit_distribution: fits a distribution to some values.
    fits_distribution: whether some values fit a distribution, by its usual
        test.
    is_skewed: whether some values are skewed to the left or right.
    survival_curves: fits a Kaplan-Meier estimator for each group.
    survival_steps: the steps of a Kaplan-Meier curve, with its confidence
        interval.
    _FAMILIES: the distributions that are tested and fitted, in the order
        that breaks ties of their AIC.
    _FEWEST_VALUES: fewest values that a distribution is tested on.
    _MOST_CELLS: most cells of a chi-square test.
    _PARAMETERS: the number of parameters that each distribution fits.
    _SHAPIRO_LIMIT: most values that the Shapiro-Wilk test is used on.

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
import importlib.util
import math
import statistics
import typing
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np
import pandas as pd

from . import base, utilities

# Methods of correlation that `pandas` supports.
CorrelationMethod: TypeAlias = Literal['pearson', 'kendall', 'spearman']
# The distributions that the "is_" explorers test and `distributions` fits.
Family: TypeAlias = Literal[
    'bernoulli', 'binomial', 'poisson', 'uniform', 'exponential', 'normal',
    'log_normal', 'beta', 'student', 'bimodal']
# The directions in which values can be skewed.
Direction: TypeAlias = Literal['left', 'right']

# The distributions in the order that breaks ties of their AIC (the simpler
# ones first).
_FAMILIES: tuple[str, ...] = typing.get_args(Family)
# Fewest values that a distribution is tested on. D'Agostino's test of skew
# needs at least 8.
_FEWEST_VALUES: int = 8
# Most cells of a chi-square test. Columns with more distinct values are
# grouped into cells with about the same number of values.
_MOST_CELLS: int = 200
# The number of parameters that each distribution fits (for its AIC). A
# binomial distribution fits one more if its number of trials is not given.
_PARAMETERS: dict[str, int] = {
    'bernoulli': 1, 'binomial': 1, 'poisson': 1, 'uniform': 2,
    'exponential': 1, 'normal': 2, 'log_normal': 2, 'beta': 2, 'student': 3,
    'bimodal': 5}
# Most values that the Shapiro-Wilk test is used on. Its p-value is not
# accurate beyond 5,000, so D'Agostino and Pearson's test is used instead.
_SHAPIRO_LIMIT: int = 5000


@dataclasses.dataclass
class Describer(base.Operation, base.Visualizer, abc.ABC):
    """Base class for techniques that describe data with a table.

    A subclass writes a `describe` method, which returns a `DataFrame`. It is
    stored in the dataset's `tables` under the technique's name. A subclass
    names the plot that draws its table as `visual`, which `apply` uses if
    it is given `visualize` (see `base.Visualizer`).

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used by most describers. Defaults to `None`.
        parameters: keyword arguments for `describe`. Defaults to an empty
            `dict`.

    """

    stage_setting: ClassVar[str] = 'visualize_explorers'

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

    def apply(
        self,
        item: base.Dataset,
        *,
        visualize: bool = False,
        **kwargs: Any) -> base.Dataset:
        """Adds the table from `describe` to the tables of `item`.

        Args:
            item: the dataset to describe.
            visualize: whether to draw the table as a figure (with the plot
                named by `visual`), which is added to the figures of `item`
                under the technique's name. The dataset's
                `visualize_explorers` (the setting of the "general" section)
                replaces it, if it is set. Defaults to `False`.
            **kwargs: parameters for `describe`.

        Returns:
            The dataset, with the new table (and figure).

        """
        item.tables[self.name] = self.describe(item, **kwargs)
        item.record(self.name, table = self.name)
        self._visualize(item, visualize, kwargs)
        return item


@dataclasses.dataclass
class DistributionTest(Describer, abc.ABC):
    """Base class for explorers that test whether columns fit a distribution.

    A subclass names its distribution as `family` (see `fits_distribution`),
    or writes a `test` method that says whether the values of one column
    fit. The table has a row for each column that was tested, with a
    boolean column named after the technique. A column that is not numeric
    or boolean, or that has too few values to test, does not fit. Missing
    values are left out, and every other row is tested, so a test of a very
    large column rejects even a distribution that fits it closely.

    """

    visual: ClassVar[str] = 'distribution_fits'
    # The distribution that `test` tests, and whose fit `distribution_fits`
    # draws over each column.
    family: ClassVar[str] = 'normal'

    """ Public Methods """

    def test(
        self,
        values: pd.Series,
        *,
        alpha: float,
        trials: int | None) -> bool:
        """Returns whether `values` fit the distribution named by `family`.

        Subclasses that test something other than a distribution (such as
        skew) override this.

        Args:
            values: the values of one column.
            alpha: the significance level of the test.
            trials: the number of trials of a binomial distribution, or
                `None` to fit it.

        Returns:
            Whether the values fit.

        """
        return fits_distribution(
            values, self.family, alpha = alpha, trials = trials)

    def describe(
        self,
        item: base.Dataset,
        columns: Sequence[str] | str | None = None,
        alpha: float = 0.05,
        trials: int | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Returns whether each of `columns` fits the distribution.

        Args:
            item: the dataset to describe.
            columns: columns to test, or "all". Defaults to `None`, which
                (like "all") tests every column.
            alpha: the significance level of the test. Defaults to 0.05.
            trials: the number of trials of a binomial distribution.
                Defaults to `None`, which fits it to each column.
            **kwargs: not used.

        Returns:
            One row for each column, with a boolean column named after the
                technique.

        """
        names = _columns_to_test(item, columns)
        fits = [
            self.test(item.data[name], alpha = alpha, trials = trials)
            for name in names]
        return pd.DataFrame(
            {self.name: fits}, index = pd.Index(names, dtype = object),
            dtype = bool)

    """ Private Methods """

    def _visual_parameters(
        self,
        item: base.Dataset,
        parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Returns the parameters of `distribution_fits`.

        Args:
            item: the dataset, with the technique's table.
            parameters: the parameters that the technique was applied with.

        Returns:
            The columns that were tested, the distribution to draw over each,
                and the number of trials of a binomial distribution.

        """
        return {
            'columns': list(item.tables[self.name].index),
            'families': self.family,
            'trials': parameters.get('trials')}


@dataclasses.dataclass
class AssociationRules(Describer):
    """Rules of which values tend to appear in the same rows.

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
    """Correlations between the numeric columns (including the label).

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


@dataclasses.dataclass
class Describe(Describer):
    """The `pandas` description of every column, one row per column.

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
class Distributions(Describer):
    """The distribution that fits each column best, and its skew.

    Each distribution that suits a column's values (for example, a Poisson
    distribution only for counts, and a beta distribution only for values
    between 0 and 1) is fitted to the column by maximum likelihood, and
    tested with the test of its "is_" explorer. The column's distribution
    is the one with the lowest AIC of those that pass their tests, or "none"
    if none does. Its "closest" is the one with the lowest AIC, whether or
    not it passes. For a column of whole numbers, the continuous
    distributions are compared by the chance of each whole number (the
    probability within half of it), so that they can be compared with the
    discrete ones. Every row is used, so a test of a very large column
    rejects even a distribution that fits it closely. A bimodal
    distribution is considered only if Hartigan's dip test finds more than
    one peak (since a mixture of two normal distributions also fits one
    peak with long tails), and only if diptest is installed.

    Wraps:
        - [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html)
          from SciPy, for the distributions and their tests.
        - [`diptest.diptest`](https://github.com/RUrlus/diptest) from
          diptest, for the test of a bimodal distribution.
        - [`sklearn.mixture.GaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
          from scikit-learn, for the fit of a bimodal distribution.

    """

    visual: ClassVar[str] = 'distribution_fits'

    def describe(
        self,
        item: base.Dataset,
        columns: Sequence[str] | str | None = None,
        alpha: float = 0.05,
        trials: int | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the distribution of each of `columns`.

        Args:
            item: the dataset to describe.
            columns: columns to describe, or "all". Defaults to `None`, which
                (like "all") describes every column.
            alpha: the significance level of the tests. Defaults to 0.05.
            trials: the number of trials of a binomial distribution.
                Defaults to `None`, which fits it to each column.
            **kwargs: not used.

        Returns:
            One row for each column, with "distribution" (the distribution
                with the lowest AIC that passes its test, or "none"),
                "closest" (the distribution with the lowest AIC, or "none"
                if no distribution suits the column), and "skew" ("left",
                "right", or "symmetric", or "none" for a column that is not
                numeric or has too few values).

        """
        names = _columns_to_test(item, columns)
        bimodal = _dip_test_installed()
        rows = [
            _distribution_of(
                item.data[name], alpha, trials, item.seed, bimodal)
            for name in names]
        return pd.DataFrame(
            rows, index = pd.Index(names, dtype = object),
            columns = ['distribution', 'closest', 'skew'])

    """ Private Methods """

    def _visual_parameters(
        self,
        item: base.Dataset,
        parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Returns the parameters of `distribution_fits`.

        Args:
            item: the dataset, with the technique's table.
            parameters: the parameters that the technique was applied with.

        Returns:
            The columns that were described, the closest distribution of each
                (to draw over it), and the number of trials of a binomial
                distribution.

        """
        table = item.tables[self.name]
        return {
            'columns': list(table.index),
            'families': {
                name: None if closest == 'none' else closest
                for name, closest in table['closest'].items()},
            'trials': parameters.get('trials')}


@dataclasses.dataclass
class Frequencies(Describer):
    """The count and share of each value of the categorical columns."""

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
class IsBernoulli(DistributionTest):
    """Whether each column is a Bernoulli distribution (0s and 1s).

    A column fits if every value is 0 or 1 (or true or false). Any such
    column is a Bernoulli distribution, so there is nothing to test.

    """

    family: ClassVar[str] = 'bernoulli'


@dataclasses.dataclass
class IsBeta(DistributionTest):
    """Whether each column fits a beta distribution.

    A column fits if every value is between 0 and 1 (not either) and the
    Kolmogorov-Smirnov test does not reject the beta distribution fitted to
    it by maximum likelihood.

    Wraps:
        [`scipy.stats.kstest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html)
        from SciPy.

    """

    family: ClassVar[str] = 'beta'


@dataclasses.dataclass
class IsBimodal(DistributionTest):
    """Whether each column has more than one peak (Hartigan's dip test).

    A column fits if Hartigan's dip test rejects that it has a single peak.
    `distribution_fits` draws a mixture of two normal distributions over it.

    Wraps:
        [`diptest.diptest`](https://github.com/RUrlus/diptest) from diptest.

    """

    family: ClassVar[str] = 'bimodal'


@dataclasses.dataclass
class IsBinomial(DistributionTest):
    """Whether each column fits a binomial distribution.

    A column fits if its values are whole numbers from 0 to the number of
    trials and the chi-square test does not reject the binomial distribution
    with its mean. The number of trials is "trials", or by default the one
    that makes the values most likely (at least the largest value).

    Wraps:
        [`scipy.stats.chisquare`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html)
        from SciPy.

    """

    family: ClassVar[str] = 'binomial'


@dataclasses.dataclass
class IsExponential(DistributionTest):
    """Whether each column fits an exponential distribution.

    A column fits if none of its values is negative and the Anderson-Darling
    test does not reject the exponential distribution with its mean.

    Wraps:
        [`scipy.stats.anderson`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html)
        from SciPy.

    """

    family: ClassVar[str] = 'exponential'


@dataclasses.dataclass
class IsLeftSkewed(DistributionTest):
    """Whether each column is skewed to the left.

    A column is skewed to the left (its longer tail is of small values) if
    D'Agostino's test finds that its skew is below 0. `distribution_fits`
    draws a normal distribution over it, to show the asymmetry.

    Wraps:
        [`scipy.stats.skewtest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html)
        from SciPy.

    """

    def test(
        self,
        values: pd.Series,
        *,
        alpha: float,
        trials: int | None) -> bool:  # noqa: ARG002
        """Returns whether `values` are skewed.

        Args:
            values: the values of one column.
            alpha: the significance level of the test.
            trials: not used.

        Returns:
            Whether the values are skewed to the left.

        """
        return is_skewed(values, 'left', alpha = alpha)


@dataclasses.dataclass
class IsLogNormal(DistributionTest):
    """Whether each column fits a log-normal distribution.

    A column fits if all of its values are above 0 and their logarithms fit
    a normal distribution (by the test of `is_normal`).

    Wraps:
        - [`scipy.stats.shapiro`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
          from SciPy, for up to 5,000 values.
        - [`scipy.stats.normaltest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)
          from SciPy, for more values.

    """

    family: ClassVar[str] = 'log_normal'


@dataclasses.dataclass
class IsNormal(DistributionTest):
    """Whether each column fits a normal distribution.

    A column fits if the Shapiro-Wilk test (for up to 5,000 values) or
    D'Agostino and Pearson's test (for more, since the p-value of the
    Shapiro-Wilk test is not accurate beyond them) does not reject it.

    Wraps:
        - [`scipy.stats.shapiro`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
          from SciPy, for up to 5,000 values.
        - [`scipy.stats.normaltest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)
          from SciPy, for more values.

    """

    family: ClassVar[str] = 'normal'


@dataclasses.dataclass
class IsPoisson(DistributionTest):
    """Whether each column fits a Poisson distribution.

    A column fits if its values are whole numbers of at least 0 and the
    chi-square test does not reject the Poisson distribution with its mean.

    Wraps:
        [`scipy.stats.chisquare`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html)
        from SciPy.

    """

    family: ClassVar[str] = 'poisson'


@dataclasses.dataclass
class IsRightSkewed(DistributionTest):
    """Whether each column is skewed to the right.

    A column is skewed to the right (its longer tail is of large values) if
    D'Agostino's test finds that its skew is above 0. `distribution_fits`
    draws a normal distribution over it, to show the asymmetry.

    Wraps:
        [`scipy.stats.skewtest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html)
        from SciPy.

    """

    def test(
        self,
        values: pd.Series,
        *,
        alpha: float,
        trials: int | None) -> bool:  # noqa: ARG002
        """Returns whether `values` are skewed.

        Args:
            values: the values of one column.
            alpha: the significance level of the test.
            trials: not used.

        Returns:
            Whether the values are skewed to the right.

        """
        return is_skewed(values, 'right', alpha = alpha)


@dataclasses.dataclass
class IsStudent(DistributionTest):
    """Whether each column fits a Student's t distribution.

    A column fits if the Kolmogorov-Smirnov test does not reject the
    Student's t distribution (with a location and scale) fitted to it by
    maximum likelihood.

    Wraps:
        [`scipy.stats.kstest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html)
        from SciPy.

    """

    family: ClassVar[str] = 'student'


@dataclasses.dataclass
class IsUniform(DistributionTest):
    """Whether each column fits a uniform distribution.

    A column fits if the uniform distribution from its smallest to its
    largest value is not rejected: by the chi-square test for whole numbers
    (each whole number equally likely), and by the Kolmogorov-Smirnov test
    for other numbers.

    Wraps:
        - [`scipy.stats.chisquare`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html)
          from SciPy, for whole numbers.
        - [`scipy.stats.kstest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html)
          from SciPy, for other numbers.

    """

    family: ClassVar[str] = 'uniform'


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

    visual: ClassVar[str] = 'survival_curves'

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

    """ Private Methods """

    def _visual_parameters(
        self,
        item: base.Dataset,  # noqa: ARG002
        parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Returns the parameters of `survival_curves`.

        Args:
            item: the dataset, with the technique's table.
            parameters: the parameters that the technique was applied with.

        Returns:
            The "event" and "group" that the curves were estimated with.

        """
        return {
            'event': parameters.get('event'),
            'group': parameters.get('group')}


@dataclasses.dataclass
class LabelBalance(Describer):
    """The count and share of each value of the label."""

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
    """The count and share of missing values in each column."""

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
    """Summary statistics of the numeric columns, as reported in papers.

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


""" Public Functions """


def best_distribution(
    values: pd.Series | np.ndarray,
    *,
    alpha: float = 0.05,
    trials: int | None = None,
    seed: int | None = None) -> str | None:
    """Returns the distribution whose fit to `values` has the lowest AIC.

    Every distribution that suits the values is fitted by maximum
    likelihood (see `fit_distribution`). A bimodal distribution (a mixture
    of two normal ones) is fitted only if Hartigan's dip test finds more
    than one peak, since a mixture also fits one peak with long tails (and
    only if diptest is installed). For
    whole numbers, the continuous distributions are compared by the chance
    of each whole number (the probability within half of it), so that they
    can be compared with the discrete ones. Ties go to the simpler
    distribution.

    Args:
        values: the values, such as a column. Missing values are left out.
        alpha: the significance level of the dip test. Defaults to 0.05.
        trials: the number of trials of a binomial distribution. Defaults to
            `None`, which fits it too.
        seed: seed of the fit of a bimodal distribution. Defaults to `None`.

    Returns:
        The name of the distribution (one of `Family`), or `None` if the
            values are not numbers or no distribution suits them.

    """
    numbers = _numbers(values)
    if numbers is None:
        return None
    fits = _candidates(
        numbers, alpha, trials, seed, bimodal = _dip_test_installed())
    scores = _criteria(numbers, fits, trials)
    return min(scores, key = scores.__getitem__) if scores else None


def fit_distribution(
    values: pd.Series | np.ndarray,
    family: str,
    *,
    trials: int | None = None,
    seed: int | None = None) -> Any:
    """Returns the distribution named `family`, fitted to `values`.

    Each is fitted by maximum likelihood: a Bernoulli, Poisson,
    exponential, or normal distribution (and the logarithms of a log-normal
    one) from the mean (and standard deviation) of the values; a binomial
    distribution from their mean and, unless it is given, the number of
    trials that makes them most likely (at least their largest value); a
    uniform
    distribution from their smallest to their largest value (each whole
    number equally likely, if they are whole numbers); a beta distribution
    (between 0 and 1) and a Student's t distribution (with a location and a
    scale) by SciPy's `fit`; and a bimodal distribution as a mixture of two
    normal distributions.

    Wraps:
        - [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html)
          from SciPy.
        - [`sklearn.mixture.GaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
          from scikit-learn, for a bimodal distribution.

    Args:
        values: the values, such as a column. Missing values are left out.
        family: the name of the distribution (one of `Family`).
        trials: the number of trials of a binomial distribution. Defaults to
            `None`, which fits it too.
        seed: seed of the fit of a bimodal distribution. Defaults to `None`.

    Raises:
        ValueError: if `family` is not the name of a distribution, `trials`
            is below 1, or the values are not numbers that suit the
            distribution (such as negative values for a Poisson one).

    Returns:
        A frozen SciPy distribution (or, for a bimodal distribution, an
            object with the same `pdf`, `logpdf`, and `cdf` methods).

    """
    _check_family(family, trials)
    numbers = _numbers(values)
    if numbers is None or not _suits(numbers, family, trials):
        message = f'the values do not suit a {family} distribution'
        raise ValueError(message)
    stats = importlib.import_module('scipy.stats')
    mean = float(numbers.mean())
    low, high = float(numbers.min()), float(numbers.max())
    if family == 'bernoulli':
        return stats.bernoulli(mean)
    if family == 'binomial':
        count = _binomial_trials(numbers) if trials is None else trials
        return stats.binom(count, mean / count)
    if family == 'poisson':
        return stats.poisson(mean)
    if family == 'uniform':
        if _is_whole(numbers):
            return stats.randint(int(low), int(high) + 1)
        return stats.uniform(low, high - low)
    if family == 'exponential':
        return stats.expon(scale = mean)
    if family == 'normal':
        return stats.norm(mean, float(numbers.std()))
    if family == 'log_normal':
        logs = np.log(numbers)
        return stats.lognorm(float(logs.std()), scale = np.exp(logs.mean()))
    with warnings.catch_warnings():
        # The optimizers warn of steps that they recover from.
        warnings.simplefilter('ignore', RuntimeWarning)
        if family == 'beta':
            shapes = stats.beta.fit(numbers, floc = 0, fscale = 1)
            return stats.beta(*shapes)
        if family == 'student':
            return _fit_student(numbers)
    mixture = importlib.import_module('sklearn.mixture').GaussianMixture(
        2, random_state = seed).fit(numbers.reshape(-1, 1))
    return _Mixture(
        weights = mixture.weights_,
        means = mixture.means_.ravel(),
        deviations = np.sqrt(mixture.covariances_.ravel()))


def fits_distribution(
    values: pd.Series | np.ndarray,
    family: str,
    *,
    alpha: float = 0.05,
    trials: int | None = None) -> bool:
    """Returns whether `values` fit the distribution named `family`.

    Each distribution is tested with its usual test, against its fit by
    `fit_distribution`: Shapiro-Wilk (for up to 5,000 values, and
    D'Agostino and Pearson's test for more) for a normal distribution and
    the logarithms of a log-normal one; chi-square for a binomial or Poisson
    distribution, and a uniform one of whole numbers; Anderson-Darling for
    an exponential distribution; Kolmogorov-Smirnov for a beta, Student's t,
    or other uniform distribution; and Hartigan's dip test for a bimodal
    one (which fits if the test rejects a single peak; whole numbers are
    first spread evenly within half of each, since the test needs values
    without ties). Values fit a
    Bernoulli distribution if each is 0 or 1. Values that do not suit the
    distribution (such as negative values for a Poisson one), or that are
    fewer than 8 (for every distribution but the Bernoulli), do not fit.

    Wraps:
        - [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html)
          from SciPy, for most tests.
        - [`diptest.diptest`](https://github.com/RUrlus/diptest) from
          diptest, for the test of a bimodal distribution.

    Args:
        values: the values, such as a column. Missing values are left out.
        family: the name of the distribution (one of `Family`).
        alpha: the significance level of the test. Defaults to 0.05.
        trials: the number of trials of a binomial distribution. Defaults to
            `None`, which fits it too.

    Raises:
        ValueError: if `family` is not the name of a distribution or
            `trials` is below 1.

    Returns:
        Whether the values fit.

    """
    _check_family(family, trials)
    numbers = _numbers(values)
    if numbers is None or not _suits(numbers, family, trials):
        return False
    distribution = None
    if family in ('binomial', 'poisson', 'uniform', 'beta', 'student'):
        distribution = fit_distribution(numbers, family, trials = trials)
    return _passes(numbers, family, alpha, distribution, trials)


def is_skewed(
    values: pd.Series | np.ndarray,
    direction: Direction,
    *,
    alpha: float = 0.05) -> bool:
    """Returns whether `values` are skewed to the left or the right.

    Values are skewed to the left (a longer tail of small values) if
    D'Agostino's test finds that their skew is below 0, and to the right if
    it finds that it is above 0.

    Wraps:
        [`scipy.stats.skewtest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html)
        from SciPy.

    Args:
        values: the values, such as a column. Missing values are left out.
        direction: "left" or "right".
        alpha: the significance level of the test. Defaults to 0.05.

    Raises:
        ValueError: if `direction` is not "left" or "right".

    Returns:
        Whether the values are skewed in `direction`. Values that are not
            numbers, that are fewer than 8, or that are all the same are not.

    """
    if direction not in ('left', 'right'):
        message = f'"direction" must be "left" or "right", not {direction!r}'
        raise ValueError(message)
    numbers = _numbers(values)
    if (
        numbers is None
        or numbers.size < _FEWEST_VALUES
        or numbers.min() == numbers.max()):
        return False
    stats = importlib.import_module('scipy.stats')
    alternative = 'less' if direction == 'left' else 'greater'
    p_value = stats.skewtest(numbers, alternative = alternative).pvalue
    return bool(p_value < alpha)


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


""" Private Functions """


def _binomial_trials(numbers: np.ndarray) -> int:
    """Returns the number of trials that makes values most likely binomial.

    The number is at least the largest value. If the values vary less than
    their mean (as binomial values do), each number up to twice the
    estimate of the method of moments (and at most 1,000 more than the
    largest value) is tried, with the share of successes that matches the
    mean. Otherwise, the likelihood grows with the number of trials, and the
    largest value is used.

    Wraps:
        [`scipy.stats.binom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)
        from SciPy.

    Args:
        numbers: whole numbers of at least 0, not all the same.

    Returns:
        The number of trials.

    """
    high = int(numbers.max())
    mean, variance = float(numbers.mean()), float(numbers.var())
    if variance >= mean:
        return high
    moment = mean ** 2 / (mean - variance)
    upper = min(max(high, math.ceil(2 * moment)), high + 1000)
    tried = np.arange(high, upper + 1)
    values, counts = np.unique(numbers, return_counts = True)
    binom = importlib.import_module('scipy.stats').binom
    likelihoods = counts @ binom.logpmf(
        values[:, np.newaxis], tried, mean / tried)
    return int(tried[np.argmax(likelihoods)])


def _candidates(
    numbers: np.ndarray,
    alpha: float,
    trials: int | None,
    seed: int | None,
    bimodal: bool) -> dict[str, Any]:
    """Returns the fit of each distribution that `distributions` compares.

    Args:
        numbers: the values, without missing ones.
        alpha: the significance level of the dip test.
        trials: the number of trials of a binomial distribution, or `None`
            to fit it.
        seed: seed of the fit of a bimodal distribution.
        bimodal: whether to consider a bimodal distribution (which needs
            diptest).

    Returns:
        The fitted distribution of each name that suits the values, in the
            order that breaks ties. A bimodal distribution is left out
            unless Hartigan's dip test finds more than one peak, since a
            mixture of two normal distributions also fits one peak with
            long tails.

    """
    return {
        family: fit_distribution(
            numbers, family, trials = trials, seed = seed)
        for family in _FAMILIES
        if _suits(numbers, family, trials) and (
            family != 'bimodal'
            or (bimodal and _passes(numbers, family, alpha, None, trials)))}


def _dip_test_installed() -> bool:
    """Returns whether diptest is installed, with a warning if it is not.

    Returns:
        Whether diptest, which tests for a bimodal distribution, can be
            imported.

    """
    if importlib.util.find_spec('diptest') is not None:
        return True
    base.logger.warning(
        'a bimodal distribution is left out, since diptest is not '
        'installed: install it with "pip install amos[statistics]"')
    return False


def _check_family(family: str, trials: int | None) -> None:
    """Checks the name of a distribution and its number of trials.

    Args:
        family: the name of the distribution.
        trials: the number of trials of a binomial distribution, or `None`.

    Raises:
        ValueError: if `family` is not the name of a distribution or
            `trials` is below 1.

    """
    if family not in _FAMILIES:
        message = (
            f'{family!r} is not a distribution: choose from '
            f'{", ".join(_FAMILIES)}')
        raise ValueError(message)
    if trials is not None and trials < 1:
        message = f'"trials" must be at least 1, not {trials}'
        raise ValueError(message)


def _chi_square_p(
    numbers: np.ndarray,
    distribution: Any,
    ddof: int) -> float:
    """Returns the p-value of a chi-square test of a discrete distribution.

    Each cell holds the values above the last cell's and up to a value that
    was seen (or, if more than `_MOST_CELLS` values were seen, a quantile of
    them), with the first and last cells reaching to the ends of the
    distribution. Neighboring cells are pooled until each is expected to
    hold at least 5 values.

    Wraps:
        [`scipy.stats.chisquare`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html)
        from SciPy.

    Args:
        numbers: the values, without missing ones.
        distribution: the fitted, frozen SciPy distribution.
        ddof: the number of the distribution's parameters that were fitted.

    Returns:
        The p-value, or NaN if too few cells are left to test.

    """
    stats = importlib.import_module('scipy.stats')
    edges = np.unique(numbers)
    if edges.size > _MOST_CELLS:
        edges = np.unique(np.quantile(
            numbers, np.linspace(0, 1, _MOST_CELLS + 1)[1:],
            method = 'inverted_cdf'))
    observed = np.bincount(
        np.searchsorted(edges, numbers), minlength = edges.size)
    below = np.concatenate([[0.0], distribution.cdf(edges[:-1]), [1.0]])
    expected = np.diff(below) * numbers.size
    pooled_observed: list[float] = []
    pooled_expected: list[float] = []
    held_observed = held_expected = 0.0
    for count, expect in zip(observed, expected, strict = True):
        held_observed += count
        held_expected += expect
        if held_expected >= 5:  # noqa: PLR2004
            pooled_observed.append(held_observed)
            pooled_expected.append(held_expected)
            held_observed = held_expected = 0.0
    if pooled_observed:
        pooled_observed[-1] += held_observed
        pooled_expected[-1] += held_expected
    if len(pooled_observed) - 1 - ddof < 1:
        return float('nan')
    result = stats.chisquare(
        pooled_observed, pooled_expected, ddof = ddof)
    return float(result.pvalue)


def _columns_to_test(
    item: base.Dataset,
    columns: Sequence[str] | str | None) -> list[str]:
    """Returns the columns that the distribution explorers describe.

    Args:
        item: the dataset to describe.
        columns: columns, a column, "all", or `None` (every column).

    Returns:
        The names of the columns.

    """
    if columns is None or columns == 'all':
        return list(item.data.columns)
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def _criteria(
    numbers: np.ndarray,
    fits: Mapping[str, Any],
    trials: int | None) -> dict[str, float]:
    """Returns the AIC of each fitted distribution.

    Args:
        numbers: the values, without missing ones.
        fits: the fitted distribution of each name, in the order that breaks
            ties.
        trials: the number of trials of a binomial distribution, or `None`
            if it was fitted too.

    Returns:
        The AIC of each distribution whose likelihood is not NaN, in the
            order of `fits`.

    """
    whole = _is_whole(numbers)
    scores: dict[str, float] = {}
    for family, distribution in fits.items():
        likelihood = _log_likelihood(numbers, distribution, whole)
        if np.isnan(likelihood):
            continue
        parameters = _PARAMETERS[family] + int(
            family == 'binomial' and trials is None)
        scores[family] = 2 * parameters - 2 * likelihood
    return scores


def _distribution_of(
    values: pd.Series,
    alpha: float,
    trials: int | None,
    seed: int | None,
    bimodal: bool) -> tuple[str, str, str]:
    """Returns the row of a column in the table of `distributions`.

    Args:
        values: the values of the column.
        alpha: the significance level of the tests.
        trials: the number of trials of a binomial distribution, or `None`
            to fit it.
        seed: seed of the fit of a bimodal distribution.
        bimodal: whether to consider a bimodal distribution (which needs
            diptest).

    Returns:
        The distribution with the lowest AIC that passes its test, the
            distribution with the lowest AIC, and the direction of the skew
            ("left", "right", or "symmetric"). Each is "none" if there is
            none, and all are "none" for values that are not numbers or are
            too few.

    """
    numbers = _numbers(values)
    if numbers is None or numbers.size < _FEWEST_VALUES:
        return ('none', 'none', 'none')
    fits = _candidates(numbers, alpha, trials, seed, bimodal)
    scores = _criteria(numbers, fits, trials)
    passing = [
        family for family in scores
        if _passes(numbers, family, alpha, fits[family], trials)]
    best = min(passing, key = scores.__getitem__) if passing else 'none'
    closest = min(scores, key = scores.__getitem__) if scores else 'none'
    if is_skewed(numbers, 'left', alpha = alpha):
        skew = 'left'
    elif is_skewed(numbers, 'right', alpha = alpha):
        skew = 'right'
    else:
        skew = 'symmetric'
    return (best, closest, skew)


def _fit_student(numbers: np.ndarray) -> Any:
    """Returns the Student's t distribution fitted to values.

    The degrees of freedom, location, and scale are fitted by maximum
    likelihood with L-BFGS-B and the gradient of the likelihood, which is
    far faster than SciPy's own `fit` for many values, and starts from the
    median, half of the interquartile range, and the degrees of freedom
    whose kurtosis matches the values' (SciPy's first guess can leave its
    optimizer far from the best fit). The degrees of freedom are kept
    between 0.2 and 10,000.

    Wraps:
        - [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
          from SciPy.
        - [`scipy.stats.t`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)
          from SciPy.

    Args:
        numbers: the values, without missing ones, not all the same.

    Returns:
        The frozen SciPy distribution.

    """
    stats = importlib.import_module('scipy.stats')
    optimize = importlib.import_module('scipy.optimize')
    special = importlib.import_module('scipy.special')
    kurtosis = float(stats.kurtosis(numbers))
    # The excess kurtosis of a t distribution is 6 / (df - 4).
    degrees = min(4 + 6 / kurtosis, 100.0) if kurtosis > 0 else 30.0
    spread = float(stats.iqr(numbers)) / 2 or float(numbers.std())

    def negative(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        """Returns minus the log likelihood and its gradient.

        Args:
            parameters: the log of the degrees of freedom, the location,
                and the log of the scale.

        Returns:
            Minus the log likelihood, and its gradient by `parameters`.

        """
        free, location, scale = (
            np.exp(parameters[0]), parameters[1], np.exp(parameters[2]))
        z = (numbers - location) / scale
        squared = z * z
        ratio = (free + 1) / (free + squared)
        likelihood = numbers.size * (
            special.gammaln((free + 1) / 2) - special.gammaln(free / 2)
            - 0.5 * np.log(free * np.pi) - np.log(scale)) - (
            (free + 1) / 2 * np.sum(np.log1p(squared / free)))
        by_free = numbers.size * (
            0.5 * special.digamma((free + 1) / 2)
            - 0.5 * special.digamma(free / 2) - 0.5 / free) + np.sum(
            0.5 * squared * ratio / free - 0.5 * np.log1p(squared / free))
        by_location = np.sum(ratio * z) / scale
        by_scale = np.sum(ratio * squared) - numbers.size
        gradient = np.array([free * by_free, by_location, by_scale])
        return -float(likelihood), -gradient

    result = optimize.minimize(
        negative,
        np.array([np.log(degrees), float(np.median(numbers)), np.log(spread)]),
        jac = True,
        method = 'L-BFGS-B',
        bounds = [(np.log(0.2), np.log(1e4)), (None, None), (None, None)])
    free, location, scale = result.x
    return stats.t(np.exp(free), location, np.exp(scale))


def _is_whole(numbers: np.ndarray) -> bool:
    """Returns whether every value is a whole number.

    Args:
        numbers: the values, without missing ones.

    Returns:
        Whether every value is a whole number.

    """
    return bool(np.all(numbers == np.round(numbers)))


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


def _log_likelihood(
    numbers: np.ndarray,
    distribution: Any,
    whole: bool) -> float:
    """Returns the log likelihood of `numbers` under a distribution.

    Args:
        numbers: the values, without missing ones.
        distribution: the fitted distribution (see `fit_distribution`).
        whole: whether every value is a whole number, in which case a
            continuous distribution gives the probability within half of
            each value, so that it can be compared with discrete ones.

    Returns:
        The log likelihood, which is minus infinity if a value is
            impossible.

    """
    values, counts = np.unique(numbers, return_counts = True)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        if hasattr(distribution, 'logpmf'):
            logged = distribution.logpmf(values)
        elif whole:
            logged = np.log(
                distribution.cdf(values + 0.5)
                - distribution.cdf(values - 0.5))
        else:
            logged = distribution.logpdf(values)
    return float(np.sum(counts * logged))


def _numbers(values: pd.Series | np.ndarray) -> np.ndarray | None:
    """Returns the finite numbers of a column, as floats.

    Args:
        values: the values, such as a column.

    Returns:
        The values that are not missing or infinite (with booleans as 0 and
            1), or `None` if the values are not numbers or booleans.

    """
    column = values if isinstance(values, pd.Series) else pd.Series(values)
    if base._kind_of(column) not in ('booleans', 'numerics'):
        return None
    numbers: np.ndarray = column.dropna().to_numpy(dtype = float)
    finite: np.ndarray = numbers[np.isfinite(numbers)]
    return finite


def _passes(
    numbers: np.ndarray,
    family: str,
    alpha: float,
    distribution: Any,
    trials: int | None) -> bool:
    """Returns whether values that suit a distribution pass its test.

    Args:
        numbers: the values, without missing ones, which suit the
            distribution (see `_suits`).
        family: the name of the distribution.
        alpha: the significance level of the test.
        distribution: the fitted distribution (see `fit_distribution`), which
            the tests of binomial, Poisson, uniform, beta, and Student's t
            distributions use, or `None` for the others.
        trials: the number of trials of a binomial distribution, or `None`
            if it was fitted (and so is a parameter of the chi-square test).

    Returns:
        Whether the values pass.

    """
    stats = importlib.import_module('scipy.stats')
    with warnings.catch_warnings():
        # The tests warn of ties and of large samples, which do not change
        # what they find.
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        if family == 'bernoulli':
            return True
        if family in ('normal', 'log_normal'):
            tested = np.log(numbers) if family == 'log_normal' else numbers
            if tested.size <= _SHAPIRO_LIMIT:
                p_value = stats.shapiro(tested).pvalue
            else:
                p_value = stats.normaltest(tested).pvalue
        elif family == 'binomial':
            p_value = _chi_square_p(
                numbers, distribution, ddof = 1 + int(trials is None))
        elif family == 'poisson':
            p_value = _chi_square_p(numbers, distribution, ddof = 1)
        elif family == 'uniform' and _is_whole(numbers):
            p_value = _chi_square_p(numbers, distribution, ddof = 0)
        elif family == 'exponential':
            p_value = stats.anderson(
                numbers, 'expon', method = 'interpolate').pvalue
        elif family == 'bimodal':
            dip = utilities.import_tool('diptest').diptest
            tested = _spread(numbers) if _is_whole(numbers) else numbers
            return bool(dip(tested)[1] < alpha)
        else:
            p_value = stats.kstest(numbers, distribution.cdf).pvalue
    return bool(p_value >= alpha)


def _spread(numbers: np.ndarray) -> np.ndarray:
    """Returns whole numbers spread evenly within half of each.

    The values equal to each number are spread evenly between half below it
    and half above it, so that they have no ties (which the dip test needs)
    and are spread the same way each time.

    Args:
        numbers: whole numbers.

    Returns:
        The spread values, in order.

    """
    ordered = np.sort(numbers)
    _, starts, counts = np.unique(
        ordered, return_index = True, return_counts = True)
    ranks = np.arange(ordered.size) - np.repeat(starts, counts)
    sizes = np.repeat(counts, counts)
    result: np.ndarray = ordered + (ranks + 0.5) / sizes - 0.5
    return result


def _suits(
    numbers: np.ndarray,
    family: str,
    trials: int | None) -> bool:
    """Returns whether values can be tested against a distribution.

    Args:
        numbers: the values, without missing ones.
        family: the name of the distribution.
        trials: the number of trials of a binomial distribution, or `None`
            to fit it.

    Returns:
        Whether the values suit the distribution: 0s and 1s for a Bernoulli
            one; and for the others, at least `_FEWEST_VALUES` values that
            are not all the same, which are whole numbers of at least 0 (and
            at most `trials`) for a binomial distribution, whole numbers of
            at least 0 for a Poisson one, at least 0 for an exponential one,
            above 0 for a log-normal one, and between 0 and 1 for a beta one.

    """
    if numbers.size == 0:
        return False
    if family == 'bernoulli':
        return bool(np.isin(numbers, (0.0, 1.0)).all())
    low, high = numbers.min(), numbers.max()
    if numbers.size < _FEWEST_VALUES or low == high:
        return False
    if family == 'binomial':
        return _is_whole(numbers) and low >= 0 and (
            trials is None or high <= trials)
    if family == 'poisson':
        return _is_whole(numbers) and low >= 0
    if family == 'exponential':
        return bool(low >= 0)
    if family == 'log_normal':
        return bool(low > 0)
    if family == 'beta':
        return bool(low > 0 and high < 1)
    return True


""" Private Classes """


@dataclasses.dataclass(frozen = True)
class _Mixture:
    """A mixture of two normal distributions, the fit of a bimodal one.

    It has the methods of a frozen SciPy distribution that the explorers and
    `distribution_fits` use.

    Args:
        weights: the share of the values in each normal distribution.
        means: the mean of each normal distribution.
        deviations: the standard deviation of each normal distribution.

    """

    weights: np.ndarray
    means: np.ndarray
    deviations: np.ndarray

    """ Public Methods """

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Returns the cumulative probability at each of `x`.

        Args:
            x: the values.

        Returns:
            The cumulative probability of each value.

        """
        norm = importlib.import_module('scipy.stats').norm
        points = np.asarray(x, dtype = float)[..., np.newaxis]
        result: np.ndarray = np.sum(
            self.weights * norm.cdf(points, self.means, self.deviations),
            axis = -1)
        return result

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Returns the log of the density at each of `x`.

        Args:
            x: the values.

        Returns:
            The log of the density of each value, computed so that it does
                not underflow in the tails.

        """
        scipy_stats = importlib.import_module('scipy.stats')
        special = importlib.import_module('scipy.special')
        points = np.asarray(x, dtype = float)[..., np.newaxis]
        logged = scipy_stats.norm.logpdf(
            points, self.means, self.deviations) + np.log(self.weights)
        result: np.ndarray = special.logsumexp(logged, axis = -1)
        return result

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Returns the density at each of `x`.

        Args:
            x: the values.

        Returns:
            The density of each value.

        """
        density: np.ndarray = np.exp(self.logpdf(x))
        return density
