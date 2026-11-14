"""Techniques that evaluate a fitted model with tables.

These are techniques of the "critic" stage. Each one adds a table to the
dataset's `tables` (under the technique's name). Most evaluate the model on
the test rows (or on every row, if the data has not been split).
`factor_analysis` and `pca` instead describe the features that the model
learned from, so they work in any stage. Pass `visualize = True` (or set
"visualize_critics" in the "general" section of the settings) and each one
also draws its table as a figure in the dataset's `figures`, under its name
(see `base.Visualizer`).

Contents:
    Evaluator: base class for techniques that evaluate a model with a table.
    BiasVariance: the model's expected error, as its bias and variance.
    ClassificationReport: precision, recall, and f1 for each class.
    Confusion: how many rows of each class were predicted to be each class.
    Conformal: prediction intervals (or sets) with a known rate of coverage.
    ExplainWeights: eli5's explanation of the weights of the features.
    FactorAnalysis: the hidden factors that explain the correlations of the
        features.
    Fairness: how the model does for each group, and the gaps between them.
    FeatureImportance: the importance that the model gives each feature.
    PCA: how much of the variance of the features each principal component
        has.
    PermutationImportance: how much the score drops when each feature is
        shuffled.
    RegressionTable: the coefficients (or average marginal effects) of every
        model side by side, with their standard errors and cross-validated
        metrics, ready to publish like a scorecard.
    Scorecard: every standard metric for every branch of an analysis, ready
        to publish as csv, Markdown, LaTeX, HTML, Word, or an image.
    ShapImportance: the mean absolute SHAP value of each feature.

"""

from __future__ import annotations

__all__: list[str] = [
    'PCA',
    'BiasVariance',
    'ClassificationReport',
    'Conformal',
    'Confusion',
    'Evaluator',
    'ExplainWeights',
    'FactorAnalysis',
    'Fairness',
    'FeatureImportance',
    'PermutationImportance',
    'RegressionTable',
    'Scorecard',
    'ShapImportance',
]

from ._base import Evaluator
from ._diagnostics import (
    PCA,
    BiasVariance,
    ClassificationReport,
    Conformal,
    Confusion,
    ExplainWeights,
    FactorAnalysis,
    Fairness,
    FeatureImportance,
    PermutationImportance,
    ShapImportance,
)
from ._diagnostics import _other_columns as _other_columns
from ._diagnostics import _with_given as _with_given
from ._regression_table import RegressionTable
from ._regression_table import _stata_number as _stata_number
from ._scorecard import Scorecard

# The public names say that they are in this package (rather than in
# the private module that defines them), where they are imported
# from, so that records, pickles, and the documentation name them as
# `amos.evaluators.Name`.
for _name in __all__:
    globals()[_name].__module__ = __name__
