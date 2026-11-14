"""Techniques that infer causal effects: how much a treatment changes things.

A model predicts the label. An inferer measures how much a treatment (such
as a program, a policy, or a type of ruling) changes the label, after
accounting for the other features, or, for time series, which variables cause
which others and when. These are techniques of the "critic" stage, since they
judge what the data can say about causes (any technique can be used in any
stage). They come from four packages:

| Package | Techniques | Python |
| --- | --- | --- |
| DoubleML | `partially_linear`, `partially_linear_iv`, `partially_logistic`, `partially_linear_panel`, `interactive_regression`, `interactive_iv`, `potential_outcomes`, `quantile_effects`, `difference_in_differences`, `regression_discontinuity`, `sample_selection` | every version |
| DoWhy | `regression_adjustment`, `glm_adjustment`, `doubly_robust`, `propensity_matching`, `propensity_stratification`, `propensity_weighting`, `distance_matching`, `instrumental_variable` | 3.13 and later |
| causalml | `s_learner`, `t_learner`, `x_learner`, `dr_learner`, `tmle` | 3.11 and 3.12 |
| tigramite | `pcmci`, `pcmci_plus`, `lpcmci`, `time_series_effect` | every version |

causalml has no version for Python 3.13 and later, and on Python 3.11 and
3.12 the versions of DoWhy that work with pandas 3 cannot be installed with
causalml, so the `causal` extra installs causalml before Python 3.13 and
DoWhy after.

Unlike a model, an inferer uses every row (it does not need a split) and
makes no predictions. It uses only real rows, though: the rows that a
sampler (such as `random_over` or `smote`) added to the training rows, which
are copies of rows or made up, are left out, since they would count some
rows more than once and change the estimates and their standard errors. A
row that a sampler removed (such as with `random_under`) is not brought
back. The effect of a treatment is a table in the dataset's
`tables` under the technique's name, with a row for each effect (of each
treatment, group, quantile, or level) and its estimate ("coefficient"),
standard error, test statistic, p-value, and 95% confidence interval.
tigramite's methods of causal discovery make a table of the causal links that
they find instead. These estimates are causal only if their assumptions hold:
most need every feature that affects both the treatment and the label to be
among the features.

Contents:
    Inferer: base class for techniques that infer causal effects.
    Inference: what an inferer found, which `Inferer.apply` adds to the
        dataset.
    Causalml, Doubleml, Dowhy, Tigramite: genres of the techniques of each
        package.
    DRLearner, SLearner, TLearner, XLearner: causalml's meta-learners.
    TMLE: causalml's targeted maximum likelihood estimation.
    DifferenceInDifferences: DoubleML's difference-in-differences, over two
        or more periods.
    InteractiveIV: DoubleML's effect of a treatment with an instrument.
    InteractiveRegression: DoubleML's average effect of a treatment with two
        values.
    PartiallyLinear: DoubleML's effect of a treatment that adds to the label.
    PartiallyLinearIV: DoubleML's partially linear model with an
        instrument.
    PartiallyLinearPanel: DoubleML's partially linear model of a panel.
    PartiallyLogistic: DoubleML's partially linear model of a label with two
        classes.
    PotentialOutcomes: DoubleML's effects of each level of a treatment.
    QuantileEffects: DoubleML's effects on quantiles of the label.
    RegressionDiscontinuity: DoubleML's effect at a cutoff of a running
        variable.
    SampleSelection: DoubleML's effect when the label is not always seen.
    DistanceMatching, PropensityMatching, PropensityStratification,
        PropensityWeighting: DoWhy's matching, stratification, and
        weighting.
    DoublyRobust: DoWhy's doubly robust estimator.
    GLMAdjustment, RegressionAdjustment: DoWhy's regressions that adjust for
        the features.
    InstrumentalVariable: DoWhy's estimator with an instrument.
    LPCMCI, PCMCI, PCMCIPlus: tigramite's causal discovery in time series.
    TimeSeriesEffect: tigramite's effect of one series on the label.

"""

from __future__ import annotations

__all__: list[str] = [
    'LPCMCI',
    'PCMCI',
    'TMLE',
    'Causalml',
    'DRLearner',
    'DifferenceInDifferences',
    'DistanceMatching',
    'Doubleml',
    'DoublyRobust',
    'Dowhy',
    'GLMAdjustment',
    'Inference',
    'Inferer',
    'InstrumentalVariable',
    'InteractiveIV',
    'InteractiveRegression',
    'PCMCIPlus',
    'PartiallyLinear',
    'PartiallyLinearIV',
    'PartiallyLinearPanel',
    'PartiallyLogistic',
    'PotentialOutcomes',
    'PropensityMatching',
    'PropensityStratification',
    'PropensityWeighting',
    'QuantileEffects',
    'RegressionAdjustment',
    'RegressionDiscontinuity',
    'SLearner',
    'SampleSelection',
    'TLearner',
    'Tigramite',
    'TimeSeriesEffect',
    'XLearner',
]

from ._base import Inference, Inferer
from ._base import _fits_on_gpu as _fits_on_gpu
from ._base import _learner as _learner
from ._causalml import TMLE, Causalml, DRLearner, SLearner, TLearner, XLearner
from ._causalml import _import_causalml as _import_causalml
from ._doubleml import (
    DifferenceInDifferences,
    Doubleml,
    InteractiveIV,
    InteractiveRegression,
    PartiallyLinear,
    PartiallyLinearIV,
    PartiallyLinearPanel,
    PartiallyLogistic,
    PotentialOutcomes,
    QuantileEffects,
    RegressionDiscontinuity,
    SampleSelection,
)
from ._dowhy import (
    DistanceMatching,
    DoublyRobust,
    Dowhy,
    GLMAdjustment,
    InstrumentalVariable,
    PropensityMatching,
    PropensityStratification,
    PropensityWeighting,
    RegressionAdjustment,
)
from ._tigramite import LPCMCI, PCMCI, PCMCIPlus, Tigramite, TimeSeriesEffect

# The public names say that they are in this package (rather than in
# the private module that defines them), where they are imported
# from, so that records, pickles, and the documentation name them as
# `amos.inferers.Name`.
for _name in __all__:
    globals()[_name].__module__ = __name__
