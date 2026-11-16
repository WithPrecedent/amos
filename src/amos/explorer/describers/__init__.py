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
    CorrelationMethod: methods of correlation that `pandas` supports.
    AssociationRules: finds rules of which values tend to appear in the same
        rows.
    Correlations: tabulates the correlations between the numeric columns and the
        label.
    Family: the distributions that the "is_" explorers test and
        `distributions` fits.
    Direction: the directions in which values can be skewed.
    DistributionTest: base class for explorers that test whether columns fit a
        distribution.
    Distributions: the distribution that fits each column best, and its skew.
    IsBernoulli: whether each column is a Bernoulli distribution (0s and 1s).
    IsBeta: whether each column fits a beta distribution.
    IsBimodal: whether each column has more than one peak (Hartigan's dip test).
    IsBinomial: whether each column fits a binomial distribution.
    IsExponential: whether each column fits an exponential distribution.
    IsLeftSkewed: whether each column is skewed to the left.
    IsLogNormal: whether each column fits a log-normal distribution.
    IsNormal: whether each column fits a normal distribution.
    IsPoisson: whether each column fits a Poisson distribution.
    IsRightSkewed: whether each column is skewed to the right.
    IsStudent: whether each column fits a Student's t distribution.
    IsUniform: whether each column fits a uniform distribution.
    best_distribution: the distribution whose fit to some values has the lowest
        AIC.
    fit_distribution: returns the distribution named `family`, fitted to
        `values`.
    fits_distribution: whether some values fit a distribution, by its usual
        test.
    is_skewed: whether some values are skewed to the left or right.
    Describe: the `pandas` description of every column.
    Frequencies: counts each value of the categorical columns, with its share.
    LabelBalance: the count and share of each value of the label.
    MissingValues: the count and share of missing values in each column.
    Summarize: tabulates the summary statistics of numeric columns that papers
        report.
    KaplanMeier: the share of rows without an event over time (survival).
    survival_curves: fits a Kaplan-Meier estimator for each group.
    survival_steps: the steps of a Kaplan-Meier curve, with its confidence
        interval.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'AssociationRules',
    'CorrelationMethod',
    'Correlations',
    'Describe',
    'Describer',
    'Direction',
    'DistributionTest',
    'Distributions',
    'Family',
    'Frequencies',
    'IsBernoulli',
    'IsBeta',
    'IsBimodal',
    'IsBinomial',
    'IsExponential',
    'IsLeftSkewed',
    'IsLogNormal',
    'IsNormal',
    'IsPoisson',
    'IsRightSkewed',
    'IsStudent',
    'IsUniform',
    'KaplanMeier',
    'LabelBalance',
    'MissingValues',
    'Summarize',
    'best_distribution',
    'fit_distribution',
    'fits_distribution',
    'is_skewed',
    'survival_curves',
    'survival_steps']

from ._associations import AssociationRules, CorrelationMethod, Correlations
from ._base import Describer
from ._distributions import (
    Direction,
    DistributionTest,
    Distributions,
    Family,
    IsBernoulli,
    IsBeta,
    IsBimodal,
    IsBinomial,
    IsExponential,
    IsLeftSkewed,
    IsLogNormal,
    IsNormal,
    IsPoisson,
    IsRightSkewed,
    IsStudent,
    IsUniform,
    best_distribution,
    fit_distribution,
    fits_distribution,
    is_skewed,
)
from ._distributions import _is_whole as _is_whole
from ._distributions import _numbers as _numbers
from ._distributions import _suits as _suits
from ._summaries import (
    Describe,
    Frequencies,
    LabelBalance,
    MissingValues,
    Summarize,
)
from ._survival import KaplanMeier, survival_curves, survival_steps

# The public names say that they are in `amos.describers` (rather than in the
# private module that defines them, or in `amos.explorer.describers`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.describers.Name`. `amos` makes `amos.describers` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.describers'
