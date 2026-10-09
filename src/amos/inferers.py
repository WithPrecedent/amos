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
makes no predictions. The effect of a treatment is a table in the dataset's
`tables` under the technique's name, with a row for each effect (of each
treatment, group, quantile, or level) and its estimate ("coefficient"),
standard error, test statistic, p-value, and 95% confidence interval.
tigramite's methods of causal discovery make a table of the causal links that
they find instead. These estimates are causal only if their assumptions hold:
most need every feature that affects both the treatment and the label to be
among the features.

Contents:
    Inferer: base class for techniques that infer causal effects.
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

import abc
import contextlib
import dataclasses
import importlib
import io
import logging
import sys
import warnings
from collections.abc import Hashable, Iterator, Sequence
from typing import Any, ClassVar

import chrisjen
import numpy as np
import pandas as pd

from . import base, models, utilities

# The columns of a table of estimates.
_COLUMNS: tuple[str, ...] = (
    'coefficient', 'standard_error', 'statistic', 'p_value', 'ci_lower',
    'ci_upper')
# DoubleML's groups of units that difference-in-differences compares the
# treated with.
_COMPARISONS: frozenset[str] = frozenset({'never_treated', 'not_yet_treated'})
# Packages whose messages (printed or logged) the inferers keep quiet.
_LOGGERS: tuple[str, ...] = ('causalml', 'dowhy', 'tigramite')
# DoubleML's scores of `quantile_effects`, by the names that it takes.
_QUANTILE_SCORES: dict[str, str] = {
    'cvar': 'CVaR', 'local_quantile': 'LPQ', 'quantile': 'PQ'}
# DoWhy's refuters that its inferers can run.
_REFUTERS: frozenset[str] = frozenset({
    'data_subset_refuter', 'placebo_treatment_refuter',
    'random_common_cause'})
# The column that numbers the rows, for DoubleML's data of units.
_ROW: str = '_amos_row'
# tigramite's tests of conditional independence, by name.
_TESTS: dict[str, str] = {
    'cmiknn': 'tigramite.independence_tests.cmiknn.CMIknn',
    'parcorr': 'tigramite.independence_tests.parcorr.ParCorr',
    'robust_parcorr':
        'tigramite.independence_tests.robust_parcorr.RobustParCorr'}
# The value of the standard normal distribution below which 97.5% of it
# falls, which turns a 95% confidence interval into a standard error.
_Z_95: float = 1.959963984540054


@dataclasses.dataclass
class Inferer(base.Operation, abc.ABC):
    """Base class for techniques that infer causal effects.

    Most inferers share these parameters (set in the settings or passed to
    `apply`); each technique lists the others that it takes:

    | Parameter | Meaning |
    | --- | --- |
    | `treatment` | Name of the column with the treatment. Required, except by tigramite's methods of causal discovery. |
    | `outcome_model` | Name of an `amos` model for the label (by default, `random_forest`). DoubleML and causalml. |
    | `treatment_model` | Name of an `amos` model for the treatment, and for any instrument or selection (by default, the outcome model). DoubleML and causalml. |
    | `n_folds` | Number of folds for cross-fitting (5 by default). DoubleML and causalml. |

    The techniques of each package are a genre within this one: `Doubleml`,
    `Dowhy`, `Causalml`, and `Tigramite`, which write `implement`. To add a
    technique of one of those packages, subclass its genre; to add one of
    another kind, subclass `Inferer` and write `implement`.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, in which case it is the snake case name of the
            class.
        contents: the estimator that the technique wraps, as an import path.
            Defaults to `None`.
        parameters: keyword arguments for `implement`. Defaults to an empty
            `dict`.

    """


@dataclasses.dataclass(frozen = True)
class _Learners:
    """Builds the models that an inferer fits, named as `amos` models.

    Args:
        outcome: name of the model of the label.
        treatment: name of the model of the treatment (and of any instrument
            or selection).
        seed: seed for the models.

    """

    outcome: str
    treatment: str
    seed: int | None

    def of_outcome(self, task: str) -> Any:
        """Returns an unfitted model of the label.

        Args:
            task: "classify" or "regress".

        Returns:
            The model.

        """
        return _learner(self.outcome, task, self.seed)

    def of_treatment(self, task: str) -> Any:
        """Returns an unfitted model of the treatment.

        Args:
            task: "classify" or "regress".

        Returns:
            The model.

        """
        return _learner(self.treatment, task, self.seed)


@dataclasses.dataclass(frozen = True)
class _Setup:
    """What the estimator of an inferer is built from.

    Args:
        item: the dataset.
        frame: the label, the treatment, any other columns with roles in the
            model, and the controls, as numbers.
        label: name of the label.
        treatment: name of the treatment.
        controls: names of the controls (the other features).
        learners: builds the models of the label and treatment.
        n_folds: number of folds for cross-fitting.
        options: the technique's other parameters.

    """

    item: base.Dataset
    frame: pd.DataFrame
    label: str
    treatment: str
    controls: list[str]
    learners: _Learners
    n_folds: int
    options: dict[str, Any]


""" causalml """


@dataclasses.dataclass(frozen = True)
class _Arrays:
    """The data that causalml's estimators take.

    Args:
        x: the controls.
        treatment: the treatment of each row (its value in the data).
        y: the label.
        p: the propensity score of each row, or `None` to let causalml find
            them.
        seed: seed for any randomness.

    """

    x: np.ndarray
    treatment: np.ndarray
    y: np.ndarray
    p: np.ndarray | None
    seed: int | None


@dataclasses.dataclass
class Causalml(Inferer, abc.ABC):
    """Genre of causalml's estimators of effects.

    causalml's meta-learners combine models of the label (and of the
    probability of treatment, the propensity score) to estimate the effect of
    a treatment for each row, and their average over every row. The models
    are `amos` models, named by "outcome_model" and "treatment_model", and the
    propensity scores are cross-fitted with "n_folds" folds. The treatment
    can have more than two values: "control" names the one that the others
    are compared to (by default, the first in sorted order), and the table
    has a row for each of the others. The effect of each row is stored in
    `tables` as "{name}_effects".

    causalml has no version for Python 3.13 and later. Three of its
    estimators are left out because, in causalml 0.17, they do not work: the
    R-learner divides its standard errors by the number of rows rather than
    its square root (so its confidence intervals are far too narrow), the
    causal forest misestimates even the effect of a treatment assigned at
    random, and the doubly robust learner with an instrument passes its
    arguments out of order and cannot be fitted.

    Wraps:
        [causalml](https://causalml.readthedocs.io/en/latest/causalml.html#module-causalml.inference.meta),
        whose meta-learners each technique names.

    """

    contents: Any = None
    # Whether the estimator needs the propensity score.
    propensity: ClassVar[bool] = False
    # Whether the estimator finds the effect of each row.
    individual: ClassVar[bool] = True

    def implement(
        self,
        item: base.Dataset,
        *,
        treatment: str | None = None,
        outcome_model: str = 'random_forest',
        treatment_model: str | None = None,
        control: Any = None,
        n_folds: int = 5,
        **kwargs: Any) -> base.Dataset:
        """Estimates the effect of `treatment` on the label with causalml.

        Args:
            item: the dataset. It must have a label.
            treatment: name of the column with the treatment. Defaults to
                `None`, but it is required.
            outcome_model: name of an `amos` model for the label. Defaults to
                "random_forest".
            treatment_model: name of an `amos` model for the treatment.
                Defaults to `None`, which uses `outcome_model`.
            control: the value of the treatment that the others are compared
                to. Defaults to `None`, which uses the first in sorted order.
            n_folds: number of folds for cross-fitting the propensity scores.
                Defaults to 5.
            **kwargs: not used.

        Raises:
            ValueError: if there is no treatment, the treatment has fewer than
                two values (or more than two, for `tmle`), or a feature is not
                a number.

        Returns:
            The dataset, with the estimates in `tables`.

        """
        treatment = _require_treatment(treatment, self.name)
        _import_causalml()
        frame, label, controls = _prepare(item, [treatment], self.name)
        values = item.data[treatment]
        levels = _levels(values)
        control = levels[0] if control is None else control
        if len(levels) < 2 or control not in levels:  # noqa: PLR2004
            message = (
                f'{self.name!r} needs a treatment with two or more values, '
                f'one of them the control, {control!r}'
            )
            raise ValueError(message)
        groups = [level for level in levels if level != control]
        learners = _Learners(
            outcome_model, treatment_model or outcome_model, item.seed)
        setup = _Setup(
            item, frame, label, treatment, controls, learners, n_folds,
            {**kwargs, 'control': control})
        x = frame[controls].to_numpy(dtype = float)
        p = None
        if self.propensity and len(groups) == 1:
            p = _propensity(
                x, (values == groups[0]).to_numpy(dtype = int),
                learners.of_treatment('classify'), n_folds, item.seed)
        elif self.propensity and not self.individual:
            message = f'{self.name!r} needs a treatment with two values'
            raise ValueError(message)
        arrays = _Arrays(
            x = x,
            treatment = values.to_numpy(),
            y = frame[label].to_numpy(dtype = float),
            p = p,
            seed = item.seed)
        with _quiet(), _seeded(item.seed):
            estimator = self._estimator(setup)
            ate, lower, upper = self._ate(estimator, arrays)
            effects = (
                self._individual(estimator, arrays) if self.individual
                else None)
        lower, upper = np.ravel(lower), np.ravel(upper)
        names = [str(g) for g in groups]
        table = _estimates(
            names, np.ravel(ate), (upper - lower) / (2 * _Z_95),
            lower = lower, upper = upper, name = treatment)
        item.tables[self.name] = table
        if effects is not None:
            item.tables[f'{self.name}_effects'] = pd.DataFrame(
                np.asarray(effects).reshape(len(item.data), -1),
                index = item.data.index,
                columns = names)
        item.fitted[self.name] = estimator
        _record(
            item, self.name, estimator, table, treatment = treatment,
            control = control, outcome_model = outcome_model,
            treatment_model = treatment_model or outcome_model)
        return item

    """ Private Methods """

    def _ate(self, estimator: Any, arrays: _Arrays) -> Any:
        """Returns the average effects and their 95% confidence intervals.

        Args:
            estimator: the causalml estimator.
            arrays: the data.

        Returns:
            The average effect for each group, and the lower and upper ends of
                their confidence intervals.

        """
        return estimator.estimate_ate(
            arrays.x, arrays.treatment, arrays.y, p = arrays.p)

    def _estimator(self, setup: _Setup) -> Any:
        """Returns the causalml estimator, with the model of the label.

        Args:
            setup: the data and settings of the inferer.

        Returns:
            The estimator.

        """
        tool = utilities.import_tool(self.contents)
        return tool(
            learner = setup.learners.of_outcome('regress'),
            control_name = setup.options['control'])

    def _individual(self, estimator: Any, arrays: _Arrays) -> Any:
        """Returns the effect for each row, from the fitted estimator.

        Args:
            estimator: the fitted causalml estimator.
            arrays: the data.

        Returns:
            The effect of each group for each row.

        """
        return estimator.predict(arrays.x)


@dataclasses.dataclass
class DRLearner(Causalml):
    """causalml's doubly robust learner of the effect of a treatment.

    It combines models of the label for the treated and the untreated with
    weighting by the propensity score, so that its estimate is right if
    either the models of the label or the propensity scores are.

    Wraps:
        [`causalml.inference.meta.BaseDRRegressor`](https://causalml.readthedocs.io/en/latest/causalml.html#causalml.inference.meta.BaseDRRegressor)
        from causalml.

    """

    contents: Any = 'causalml.inference.meta.BaseDRRegressor'
    propensity: ClassVar[bool] = True

    def _ate(self, estimator: Any, arrays: _Arrays) -> Any:
        """Returns the average effects and their 95% confidence intervals.

        Args:
            estimator: the causalml estimator.
            arrays: the data.

        Returns:
            The average effect for each group, and the lower and upper ends of
                their confidence intervals.

        """
        return estimator.estimate_ate(
            arrays.x, arrays.treatment, arrays.y, p = arrays.p,
            seed = arrays.seed)


@dataclasses.dataclass
class SLearner(Causalml):
    """causalml's S-learner: one model of the label, with the treatment in it.

    The effect of each row is the difference between the model's predictions
    with and without the treatment.

    Wraps:
        [`causalml.inference.meta.BaseSRegressor`](https://causalml.readthedocs.io/en/latest/causalml.html#causalml.inference.meta.BaseSRegressor)
        from causalml.

    """

    contents: Any = 'causalml.inference.meta.BaseSRegressor'

    def _ate(self, estimator: Any, arrays: _Arrays) -> Any:
        """Returns the average effects and their 95% confidence intervals.

        Args:
            estimator: the causalml estimator.
            arrays: the data.

        Returns:
            The average effect for each group, and the lower and upper ends of
                their confidence intervals.

        """
        return estimator.estimate_ate(
            arrays.x, arrays.treatment, arrays.y, return_ci = True)


@dataclasses.dataclass
class TLearner(Causalml):
    """causalml's T-learner: one model of the label for each group.

    The effect of each row is the difference between the predictions of the
    model of its treatment group and of the model of the control group.

    Wraps:
        [`causalml.inference.meta.BaseTRegressor`](https://causalml.readthedocs.io/en/latest/causalml.html#causalml.inference.meta.BaseTRegressor)
        from causalml.

    """

    contents: Any = 'causalml.inference.meta.BaseTRegressor'

    def _ate(self, estimator: Any, arrays: _Arrays) -> Any:
        """Returns the average effects and their 95% confidence intervals.

        Args:
            estimator: the causalml estimator.
            arrays: the data.

        Returns:
            The average effect for each group, and the lower and upper ends of
                their confidence intervals.

        """
        return estimator.estimate_ate(arrays.x, arrays.treatment, arrays.y)


@dataclasses.dataclass
class TMLE(Causalml):
    """causalml's targeted maximum likelihood estimation of an average effect.

    It updates a model of the label with the propensity scores so that the
    average effect has the least bias. The treatment must have two values,
    and there is no effect for each row.

    Wraps:
        [`causalml.inference.meta.TMLELearner`](https://causalml.readthedocs.io/en/latest/causalml.html#causalml.inference.meta.TMLELearner)
        from causalml.

    """

    contents: Any = 'causalml.inference.meta.TMLELearner'
    propensity: ClassVar[bool] = True
    individual: ClassVar[bool] = False

    def _estimator(self, setup: _Setup) -> Any:
        """Returns causalml's `TMLELearner`, cross-fitted with `n_folds` folds.

        Args:
            setup: the data and settings of the inferer.

        Returns:
            The estimator.

        """
        tool = utilities.import_tool(self.contents)
        model_selection = importlib.import_module('sklearn.model_selection')
        return tool(
            learner = setup.learners.of_outcome('regress'),
            control_name = setup.options['control'],
            cv = model_selection.KFold(
                n_splits = setup.n_folds, shuffle = True,
                random_state = setup.item.seed))


@dataclasses.dataclass
class XLearner(Causalml):
    """causalml's X-learner, which suits a treatment that few rows have.

    It models the label in each group, imputes the effect for each row from
    the other group's model, and models those effects, weighting the two
    groups' estimates by the propensity score.

    Wraps:
        [`causalml.inference.meta.BaseXRegressor`](https://causalml.readthedocs.io/en/latest/causalml.html#causalml.inference.meta.BaseXRegressor)
        from causalml.

    """

    contents: Any = 'causalml.inference.meta.BaseXRegressor'
    propensity: ClassVar[bool] = True

    def _individual(self, estimator: Any, arrays: _Arrays) -> Any:
        """Returns the effect for each row, weighted by the propensity scores.

        Args:
            estimator: the fitted causalml estimator.
            arrays: the data.

        Returns:
            The effect of each group for each row.

        """
        if arrays.p is None:
            return estimator.predict(arrays.x)
        return estimator.predict(arrays.x, p = arrays.p)


""" DoubleML """


@dataclasses.dataclass
class Doubleml(Inferer, abc.ABC):
    """Genre of DoubleML's models (double machine learning).

    Double (or debiased) machine learning fits a model of the label and one
    of the treatment (and of any instrument), each to some folds of the rows
    and applied to the others, so that flexible models can control for the
    features without biasing the estimate. The models are `amos` models,
    named by "outcome_model" and "treatment_model", and the folds are drawn
    with the dataset's seed. A subclass writes `_estimator`.

    Wraps:
        [DoubleML](https://docs.doubleml.org/stable/api/dml_models.html),
        whose models each technique names.

    """

    # DoubleML's class of the data of the model.
    data: ClassVar[str] = 'doubleml.DoubleMLData'
    # The arguments of `data` that the columns named by parameters fill, by
    # the names of the parameters.
    roles: ClassVar[dict[str, str]] = {}
    # Parameters in `roles` that must be given.
    required: ClassVar[tuple[str, ...]] = ()
    # Other arguments for `data`.
    data_options: ClassVar[dict[str, Any]] = {}
    # Whether amos draws the folds from the rows (rather than DoubleML drawing
    # them, from the units of a panel or with folds within each fold).
    row_folds: ClassVar[bool] = True

    def implement(
        self,
        item: base.Dataset,
        *,
        treatment: str | None = None,
        outcome_model: str = 'random_forest',
        treatment_model: str | None = None,
        n_folds: int = 5,
        **kwargs: Any) -> base.Dataset:
        """Estimates the effect of `treatment` on the label with DoubleML.

        Args:
            item: the dataset. It must have a label.
            treatment: name of the column with the treatment. Defaults to
                `None`, but it is required.
            outcome_model: name of an `amos` model for the label. Defaults to
                "random_forest".
            treatment_model: name of an `amos` model for the treatment.
                Defaults to `None`, which uses `outcome_model`.
            n_folds: number of folds for cross-fitting. Defaults to 5.
            **kwargs: the parameters of the model (see each technique).

        Raises:
            ValueError: if there is no treatment (or another column that the
                model needs), or a feature is not a number.

        Returns:
            The dataset, with the estimates in `tables`.

        """
        treatment = _require_treatment(treatment, self.name)
        utilities.import_tool('doubleml')
        named: dict[str, str] = {
            p: kwargs[p] for p in self.roles if kwargs.get(p) is not None}
        missing = [p for p in self.required if p not in named]
        if missing:
            message = f'{self.name!r} needs the name of a {missing[0]!r} column'
            raise ValueError(message)
        frame, label, controls = _prepare(
            item, [treatment, *named.values()], self.name)
        learners = _Learners(
            outcome_model, treatment_model or outcome_model, item.seed)
        setup = _Setup(
            item, frame, label, treatment, controls, learners, n_folds, kwargs)
        setup = dataclasses.replace(setup, frame = self._adjust(setup))
        builder = utilities.import_tool(self.data)
        data = builder(
            setup.frame,
            y_col = label,
            d_cols = treatment,
            x_cols = controls,
            **{self.roles[p]: c for p, c in named.items()},
            **self._data_options(setup))
        with _seeded(item.seed):
            estimator = self._estimator(setup, data)
            if self.row_folds:
                estimator.set_sample_splitting(
                    _folds(len(frame), n_folds, item.seed))
            estimator.fit()
        table = self._table(setup, estimator)
        item.tables[self.name] = table
        item.fitted[self.name] = estimator
        _record(
            item, self.name, estimator, table, treatment = treatment,
            outcome_model = outcome_model,
            treatment_model = treatment_model or outcome_model,
            n_folds = n_folds, **named)
        return item

    """ Private Methods """

    def _adjust(self, setup: _Setup) -> pd.DataFrame:
        """Returns the data for the model, changed if the model needs it.

        Args:
            setup: the data and settings of the inferer.

        Returns:
            The data.

        """
        return setup.frame

    def _data_options(
        self,
        setup: _Setup) -> dict[str, Any]:  # noqa: ARG002
        """Returns the other arguments for DoubleML's data.

        Args:
            setup: the data and settings of the inferer.

        Returns:
            The arguments (by default, `data_options`).

        """
        return dict(self.data_options)

    @abc.abstractmethod
    def _estimator(self, setup: _Setup, data: Any) -> Any:
        """Returns the DoubleML estimator, ready to fit.

        Args:
            setup: the data and settings of the inferer.
            data: DoubleML's data of the label, treatment, and features.

        Returns:
            The estimator, ready to fit.

        """

    def _table(
        self,
        setup: _Setup,  # noqa: ARG002
        estimator: Any) -> pd.DataFrame:
        """Returns the table of estimates of the fitted estimator.

        Args:
            setup: the data and settings of the inferer.
            estimator: the fitted DoubleML estimator.

        Returns:
            The estimates.

        """
        return _from_summary(estimator.summary)


@dataclasses.dataclass
class DifferenceInDifferences(Doubleml):
    """DoubleML's difference-in-differences: a treatment's effect over time.

    It compares how the label changed, from before a treatment to after it,
    in the units that were treated and in those that were not, adjusting for
    the features, to find the average effect on the treated. Each row is a
    unit in a period ("time", a column of numbers), and the treatment is the
    period in which the row's unit was first treated (missing for units that
    never were). With two periods, the treatment may instead be 1 for the
    treated group and 0 for the rest. With "unit" (the column that names each
    unit), the rows are a panel: the same units in each period. Without it,
    the units can differ between the periods (repeated cross-sections).

    Units may be first treated in different periods (staggered adoption):
    each group of units first treated in the same period is compared, in
    each period, with the units never treated (or, with "comparison" set to
    "not_yet_treated", with those not yet treated). The table under its name
    has the average effect over the groups and the periods after their
    treatment, and "{name}_periods" has the effect by the number of periods
    since the treatment (an event study, whose periods before the treatment
    should show no effect if the groups' labels would otherwise have changed
    alike). Set "score" to "experimental" if the treatment was assigned at
    random.

    Wraps:
        [`doubleml.did.DoubleMLDIDMulti`](https://docs.doubleml.org/stable/api/generated/doubleml.did.DoubleMLDIDMulti.html)
        from DoubleML.

    """

    contents: Any = 'doubleml.did.DoubleMLDIDMulti'
    data: ClassVar[str] = 'doubleml.DoubleMLPanelData'
    roles: ClassVar[dict[str, str]] = {'time': 't_col', 'unit': 'id_col'}
    required: ClassVar[tuple[str, ...]] = ('time',)
    row_folds: ClassVar[bool] = False

    def _adjust(self, setup: _Setup) -> pd.DataFrame:
        """Returns the data, with the period in which each unit was treated.

        Args:
            setup: the data and settings of the inferer, with "time" and
                "unit".

        Returns:
            The data, in which the treatment is the first treated period
                (infinite for units never treated) and the units are numbered
                (or, without "unit", each row is a unit).

        """
        frame = setup.frame.copy()
        periods = sorted(frame[setup.options['time']].dropna().unique())
        first = frame[setup.treatment]
        if len(periods) == 2 and first.dropna().isin([0, 1]).all():  # noqa: PLR2004
            first = first.where(first != 1, periods[1]).where(first != 0)
        frame[setup.treatment] = first.fillna(np.inf)
        unit = setup.options.get('unit')
        if unit:
            frame[unit] = pd.factorize(setup.item.data[unit])[0]
        else:
            frame[_ROW] = np.arange(len(frame))
        return frame

    def _data_options(self, setup: _Setup) -> dict[str, Any]:
        """Returns the column of the units, if each row is one.

        Args:
            setup: the data and settings of the inferer, with "unit".

        Returns:
            The arguments for DoubleML's data.

        """
        return {} if setup.options.get('unit') else {'id_col': _ROW}

    def _estimator(self, setup: _Setup, data: Any) -> Any:
        """Returns DoubleML's `DoubleMLDIDMulti`.

        Args:
            setup: the data and settings of the inferer, with "unit",
                "comparison" ("never_treated" by default, or
                "not_yet_treated"), and "score" ("observational" by default,
                or "experimental").
            data: DoubleML's data of the label, treatment, and features.

        Raises:
            ValueError: if "comparison" is not one of those.

        Returns:
            The estimator, ready to fit.

        """
        comparison = setup.options.get('comparison') or 'never_treated'
        if comparison not in _COMPARISONS:
            message = (
                f'comparison must be one of {sorted(_COMPARISONS)}, not '
                f'{comparison!r}'
            )
            raise ValueError(message)
        score = setup.options.get('score') or 'observational'
        treatment = None if score == 'experimental' else (
            setup.learners.of_treatment('classify'))
        tool = utilities.import_tool(self.contents)
        return tool(
            data,
            setup.learners.of_outcome(_task(setup.frame[setup.label])),
            treatment,
            control_group = comparison,
            n_folds = setup.n_folds,
            score = score,
            panel = bool(setup.options.get('unit')))

    def _table(self, setup: _Setup, estimator: Any) -> pd.DataFrame:
        """Returns the average effect, and stores the effect by period.

        Args:
            setup: the data and settings of the inferer.
            estimator: the fitted DoubleML estimator.

        Returns:
            The average effect on the treated, over the groups and periods.

        """
        periods = _from_summary(
            estimator.aggregate('eventstudy').aggregated_summary)
        periods.index = pd.Index(
            [float(p) for p in periods.index], name = 'periods_since')
        setup.item.tables[f'{self.name}_periods'] = periods
        table = _from_summary(estimator.aggregate('group').overall_summary)
        table.index = pd.Index([setup.treatment])
        return table


@dataclasses.dataclass
class InteractiveIV(Doubleml):
    """DoubleML's effect for those whom an instrument moves to take a treatment.

    It estimates the local average treatment effect with DoubleML's
    interactive IV model. The treatment and the instrument ("instrument")
    must each have two values. The instrument must change the treatment but
    affect the label only through it.

    Wraps:
        [`doubleml.DoubleMLIIVM`](https://docs.doubleml.org/stable/api/generated/doubleml.irm.DoubleMLIIVM.html)
        from DoubleML.

    """

    contents: Any = 'doubleml.DoubleMLIIVM'
    roles: ClassVar[dict[str, str]] = {'instrument': 'z_cols'}
    required: ClassVar[tuple[str, ...]] = ('instrument',)

    def _estimator(self, setup: _Setup, data: Any) -> Any:
        """Returns DoubleML's `DoubleMLIIVM`.

        Args:
            setup: the data and settings of the inferer, with "instrument".
            data: DoubleML's data of the label, treatment, and features.

        Returns:
            The estimator, ready to fit.

        """
        _require_values(setup.frame[setup.treatment], 2, self.name)
        _require_values(setup.frame[setup.options['instrument']], 2, self.name)
        tool = utilities.import_tool(self.contents)
        return tool(
            data,
            setup.learners.of_outcome(_task(setup.frame[setup.label])),
            setup.learners.of_treatment('classify'),
            setup.learners.of_treatment('classify'),
            n_folds = setup.n_folds)


@dataclasses.dataclass
class InteractiveRegression(Doubleml):
    """DoubleML's average effect of a treatment with two values.

    It allows the effect to differ from row to row, and estimates its average
    over every row (the average treatment effect) with DoubleML's interactive
    regression model, or over the treated rows with "score" set to "ATTE".
    The model of the label is a classifier if the label has two classes.

    Wraps:
        [`doubleml.DoubleMLIRM`](https://docs.doubleml.org/stable/api/generated/doubleml.irm.DoubleMLIRM.html)
        from DoubleML.

    """

    contents: Any = 'doubleml.DoubleMLIRM'

    def _estimator(self, setup: _Setup, data: Any) -> Any:
        """Returns DoubleML's interactive regression model (`DoubleMLIRM`).

        Args:
            setup: the data and settings of the inferer, with "score"
                ("ATE" by default, or "ATTE").
            data: DoubleML's data of the label, treatment, and features.

        Returns:
            The estimator, ready to fit.

        """
        _require_values(setup.frame[setup.treatment], 2, self.name)
        tool = utilities.import_tool(self.contents)
        return tool(
            data,
            setup.learners.of_outcome(_task(setup.frame[setup.label])),
            setup.learners.of_treatment('classify'),
            n_folds = setup.n_folds,
            score = setup.options.get('score') or 'ATE')


@dataclasses.dataclass
class PartiallyLinear(Doubleml):
    """DoubleML's effect of a treatment that adds the same amount for every row.

    It assumes that the treatment changes the label by the same amount for
    every row, while the features may affect both in any way, with DoubleML's
    partially linear regression model. The treatment can be a number or have
    two values.

    Wraps:
        [`doubleml.DoubleMLPLR`](https://docs.doubleml.org/stable/api/generated/doubleml.plm.DoubleMLPLR.html)
        from DoubleML.

    """

    contents: Any = 'doubleml.DoubleMLPLR'

    def _estimator(self, setup: _Setup, data: Any) -> Any:
        """Returns DoubleML's partially linear regression model (`DoubleMLPLR`).

        Args:
            setup: the data and settings of the inferer.
            data: DoubleML's data of the label, treatment, and features.

        Returns:
            The estimator, ready to fit.

        """
        tool = utilities.import_tool(self.contents)
        return tool(
            data,
            setup.learners.of_outcome('regress'),
            setup.learners.of_treatment(_task(setup.frame[setup.treatment])),
            n_folds = setup.n_folds)


@dataclasses.dataclass
class PartiallyLinearIV(Doubleml):
    """DoubleML's partially linear model of a treatment with an instrument.

    Like `partially_linear`, but the treatment may share causes with the
    label that are not among the features, so its effect is found through an
    instrument ("instrument"): a column that changes the treatment but
    affects the label only through it. Its treatment model fits the
    treatment and the instrument as numbers, so it must regress (as `linear`
    and `random_forest` do, but `logit` does not).

    Wraps:
        [`doubleml.DoubleMLPLIV`](https://docs.doubleml.org/stable/api/generated/doubleml.plm.DoubleMLPLIV.html)
        from DoubleML.

    """

    contents: Any = 'doubleml.DoubleMLPLIV'
    roles: ClassVar[dict[str, str]] = {'instrument': 'z_cols'}
    required: ClassVar[tuple[str, ...]] = ('instrument',)

    def _estimator(self, setup: _Setup, data: Any) -> Any:
        """Returns DoubleML's `DoubleMLPLIV`.

        Args:
            setup: the data and settings of the inferer, with "instrument".
            data: DoubleML's data of the label, treatment, and features.

        Returns:
            The estimator, ready to fit.

        """
        tool = utilities.import_tool(self.contents)
        return tool(
            data,
            setup.learners.of_outcome('regress'),
            setup.learners.of_treatment('regress'),
            setup.learners.of_treatment('regress'),
            n_folds = setup.n_folds)


@dataclasses.dataclass
class PartiallyLinearPanel(Doubleml):
    """DoubleML's partially linear model of a panel: units seen over time.

    Each row is one unit ("unit", a column that identifies it) in one period
    ("time", a column of numbered periods). Like `partially_linear`, it
    assumes that the treatment changes the label by the same amount for
    every row, and it also removes whatever about each unit does not change
    over time, by the "approach" of Clarke and Polselli (2025):
    "fd_exact" (first differences, by default), "wg_approx" (within
    groups), "cre_general", or "cre_normal" (correlated random effects).
    The folds keep each unit's rows together.

    Wraps:
        [`doubleml.DoubleMLPLPR`](https://docs.doubleml.org/stable/api/generated/doubleml.plm.DoubleMLPLPR.html)
        from DoubleML.

    """

    contents: Any = 'doubleml.DoubleMLPLPR'
    data: ClassVar[str] = 'doubleml.DoubleMLPanelData'
    roles: ClassVar[dict[str, str]] = {'unit': 'id_col', 'time': 't_col'}
    required: ClassVar[tuple[str, ...]] = ('unit', 'time')
    data_options: ClassVar[dict[str, Any]] = {'static_panel': True}
    row_folds: ClassVar[bool] = False

    def _adjust(self, setup: _Setup) -> pd.DataFrame:
        """Returns the data, with each unit numbered.

        Args:
            setup: the data and settings of the inferer, with "unit".

        Returns:
            The data.

        """
        frame = setup.frame.copy()
        unit = setup.options['unit']
        frame[unit] = pd.factorize(setup.item.data[unit])[0]
        return frame

    def _estimator(self, setup: _Setup, data: Any) -> Any:
        """Returns DoubleML's `DoubleMLPLPR`.

        Args:
            setup: the data and settings of the inferer, with "approach"
                ("fd_exact" by default).
            data: DoubleML's data of the label, treatment, and features.

        Returns:
            The estimator, ready to fit.

        """
        tool = utilities.import_tool(self.contents)
        return tool(
            data,
            setup.learners.of_outcome('regress'),
            setup.learners.of_treatment(_task(setup.frame[setup.treatment])),
            n_folds = setup.n_folds,
            approach = setup.options.get('approach') or 'fd_exact')

    def _table(self, setup: _Setup, estimator: Any) -> pd.DataFrame:
        """Returns the estimate, labeled with the name of the treatment.

        Args:
            setup: the data and settings of the inferer.
            estimator: the fitted DoubleML estimator.

        Returns:
            The estimates (DoubleML names the treatment for its
                transformation, such as "{treatment}_diff").

        """
        table = _from_summary(estimator.summary)
        table.index = pd.Index([setup.treatment] * len(table))
        return table


@dataclasses.dataclass
class PartiallyLogistic(Doubleml):
    """DoubleML's partially linear model of a label with two classes.

    Like `partially_linear`, but for a label with two classes: the treatment
    changes the log-odds of the second class by the same amount for every
    row, and the coefficient is that change. The outcome model fits both the
    chance of the second class and its log-odds, so it must classify and
    regress (as `random_forest` does, but `logit` does not).

    Wraps:
        [`doubleml.DoubleMLLPLR`](https://docs.doubleml.org/stable/api/generated/doubleml.plm.DoubleMLLPLR.html)
        from DoubleML.

    """

    contents: Any = 'doubleml.DoubleMLLPLR'
    row_folds: ClassVar[bool] = False

    def _estimator(self, setup: _Setup, data: Any) -> Any:
        """Returns DoubleML's `DoubleMLLPLR`.

        Args:
            setup: the data and settings of the inferer.
            data: DoubleML's data of the label, treatment, and features.

        Raises:
            ValueError: if the label does not have two classes.

        Returns:
            The estimator, ready to fit.

        """
        if _task(setup.frame[setup.label]) != 'classify':
            message = f'{self.name!r} needs a label with two classes'
            raise ValueError(message)
        tool = utilities.import_tool(self.contents)
        return tool(
            data,
            setup.learners.of_outcome('classify'),
            setup.learners.of_outcome('regress'),
            setup.learners.of_treatment(_task(setup.frame[setup.treatment])),
            n_folds = setup.n_folds)


@dataclasses.dataclass
class PotentialOutcomes(Doubleml):
    """DoubleML's effects of each level of a treatment, compared to one of them.

    For a treatment with several levels (such as kinds of programs), it
    estimates the average label if every row had each level (the average
    potential outcomes) and the differences between them. "levels" chooses
    the levels (by default, every one), and the table has a row comparing
    each with "reference" (by default, the first in sorted order).

    Wraps:
        [`doubleml.DoubleMLAPOS`](https://docs.doubleml.org/stable/api/generated/doubleml.irm.DoubleMLAPOS.html)
        from DoubleML.

    """

    contents: Any = 'doubleml.DoubleMLAPOS'

    def _adjust(self, setup: _Setup) -> pd.DataFrame:
        """Returns the data, with levels that are not numbers numbered.

        Args:
            setup: the data and settings of the inferer.

        Returns:
            The data.

        """
        frame = setup.frame.copy()
        values = setup.item.data[setup.treatment]
        frame[setup.treatment] = [_code(values, value) for value in values]
        return frame

    def _estimator(self, setup: _Setup, data: Any) -> Any:
        """Returns DoubleML's `DoubleMLAPOS`.

        Args:
            setup: the data and settings of the inferer, with "levels".
            data: DoubleML's data of the label, treatment, and features.

        Returns:
            The estimator, ready to fit.

        """
        values = setup.item.data[setup.treatment]
        levels = _listed(setup.options.get('levels')) or _levels(values)
        tool = utilities.import_tool(self.contents)
        return tool(
            data,
            setup.learners.of_outcome(_task(setup.frame[setup.label])),
            setup.learners.of_treatment('classify'),
            treatment_levels = [_code(values, v) for v in levels],
            n_folds = setup.n_folds)

    def _table(self, setup: _Setup, estimator: Any) -> pd.DataFrame:
        """Returns the differences between each level and the reference.

        Args:
            setup: the data and settings of the inferer, with "levels" and
                "reference".
            estimator: the fitted DoubleML estimator.

        Returns:
            The estimates, with a row for each level other than the
                reference ("{level} vs {reference}").

        """
        values = setup.item.data[setup.treatment]
        levels = _listed(setup.options.get('levels')) or _levels(values)
        reference = setup.options.get('reference')
        reference = levels[0] if reference is None else reference
        contrast = estimator.causal_contrast(
            reference_levels = _code(values, reference))
        table = _from_summary(contrast.summary)
        table.index = pd.Index(
            [f'{level} vs {reference}' for level in levels
             if level != reference], name = setup.treatment)
        return table


@dataclasses.dataclass
class QuantileEffects(Doubleml):
    """DoubleML's effects of a treatment on quantiles of the label.

    For a treatment with two values, it estimates how the treatment changes
    each of "quantiles" of the label (by default, the quartiles 0.25, 0.5,
    and 0.75), with "score" set to "quantile" (by default). "cvar" estimates
    the change in the conditional value at risk (the mean of the label above
    each quantile), and "local_quantile" the change for those whom an
    instrument ("instrument", with two values) moves to take the treatment.

    Wraps:
        [`doubleml.DoubleMLQTE`](https://docs.doubleml.org/stable/api/generated/doubleml.irm.DoubleMLQTE.html)
        from DoubleML.

    """

    contents: Any = 'doubleml.DoubleMLQTE'
    roles: ClassVar[dict[str, str]] = {'instrument': 'z_cols'}

    def _estimator(self, setup: _Setup, data: Any) -> Any:
        """Returns DoubleML's `DoubleMLQTE`.

        Args:
            setup: the data and settings of the inferer, with "quantiles",
                "score", and "instrument".
            data: DoubleML's data of the label, treatment, and features.

        Raises:
            ValueError: if "score" is not one of those above, or is
                "local_quantile" without an instrument.

        Returns:
            The estimator, ready to fit.

        """
        _require_values(setup.frame[setup.treatment], 2, self.name)
        score = setup.options.get('score') or 'quantile'
        if score not in _QUANTILE_SCORES:
            message = (
                f'score must be one of {sorted(_QUANTILE_SCORES)}, not '
                f'{score!r}'
            )
            raise ValueError(message)
        if score == 'local_quantile' and not setup.options.get('instrument'):
            message = (
                '"local_quantile" needs the name of an "instrument" column')
            raise ValueError(message)
        quantiles = [
            float(q) for q in _listed(setup.options.get('quantiles'))
            or (0.25, 0.5, 0.75)]
        task = 'regress' if score == 'cvar' else 'classify'
        tool = utilities.import_tool(self.contents)
        return tool(
            data,
            setup.learners.of_outcome(task),
            setup.learners.of_treatment('classify'),
            quantiles = quantiles,
            score = _QUANTILE_SCORES[score],
            n_folds = setup.n_folds)

    def _table(
        self,
        setup: _Setup,  # noqa: ARG002
        estimator: Any) -> pd.DataFrame:
        """Returns the estimates, with a row for each quantile.

        Args:
            setup: the data and settings of the inferer.
            estimator: the fitted DoubleML estimator.

        Returns:
            The estimates.

        """
        table = _from_summary(estimator.summary)
        table.index.name = 'quantile'
        return table


@dataclasses.dataclass
class RegressionDiscontinuity(Doubleml):
    """DoubleML's regression discontinuity: the effect at a cutoff.

    A treatment given to the rows whose "running" column (such as a score or
    an age) is at or above "cutoff" (0 by default) is estimated from the rows
    near the cutoff, since those just above and just below it are alike but
    for the treatment. The features make the estimate more precise, through
    flexible models. If the treatment is 1 exactly for the rows at or above
    the cutoff, the design is sharp. Otherwise, it is fuzzy: crossing the
    cutoff only makes the treatment more likely, and the effect is for those
    whom crossing it moves to take the treatment. The estimate is rdrobust's
    conventional one, and its standard error, statistic, p-value, and
    confidence interval are robust to the estimate's bias (as rdrobust
    recommends). It needs the rdrobust package.

    Wraps:
        [`doubleml.rdd.RDFlex`](https://docs.doubleml.org/stable/api/generated/doubleml.rdd.RDFlex.html)
        from DoubleML.

    """

    contents: Any = 'doubleml.rdd.RDFlex'
    data: ClassVar[str] = 'doubleml.DoubleMLRDDData'
    roles: ClassVar[dict[str, str]] = {'running': 'score_col'}
    required: ClassVar[tuple[str, ...]] = ('running',)
    row_folds: ClassVar[bool] = False

    def _estimator(self, setup: _Setup, data: Any) -> Any:
        """Returns DoubleML's `RDFlex`.

        Args:
            setup: the data and settings of the inferer, with "running" and
                "cutoff" (0 by default).
            data: DoubleML's data of the label, treatment, and features.

        Returns:
            The estimator, ready to fit.

        """
        frame = setup.frame
        _require_values(frame[setup.treatment], 2, self.name)
        cutoff = float(setup.options.get('cutoff') or 0)
        above = (frame[setup.options['running']] >= cutoff).astype(float)
        sharp = bool((above == frame[setup.treatment]).all())
        utilities.import_tool('rdrobust')
        tool = utilities.import_tool(self.contents)
        return tool(
            data,
            setup.learners.of_outcome(_task(frame[setup.label])),
            None if sharp else setup.learners.of_treatment('classify'),
            fuzzy = not sharp,
            cutoff = cutoff,
            n_folds = setup.n_folds)

    def _table(self, setup: _Setup, estimator: Any) -> pd.DataFrame:
        """Returns the conventional estimate with its robust uncertainty.

        Args:
            setup: the data and settings of the inferer.
            estimator: the fitted `RDFlex`.

        Returns:
            The estimate.

        """
        # rdrobust's rows are the conventional, bias-corrected, and robust
        # estimates.
        robust = 2
        values = [
            float(estimator.coef[0]),
            float(estimator.se[robust]),
            float(estimator.t_stat[robust]),
            float(estimator.pval[robust]),
            float(estimator.ci[robust, 0]),
            float(estimator.ci[robust, 1])]
        return pd.DataFrame(
            [values], columns = list(_COLUMNS),
            index = pd.Index([setup.treatment]))


@dataclasses.dataclass
class SampleSelection(Doubleml):
    """DoubleML's effect of a treatment when the label is only sometimes seen.

    "selected" names a column that is 1 for the rows whose label was seen
    and 0 for the others (whose labels are not used and can be missing). The
    treatment must have two values. The label is assumed to be missing at
    random given the features and treatment, unless "instrument" names a
    column that changes whether the label is seen but not the label itself.

    Wraps:
        [`doubleml.DoubleMLSSM`](https://docs.doubleml.org/stable/api/generated/doubleml.irm.DoubleMLSSM.html)
        from DoubleML.

    """

    contents: Any = 'doubleml.DoubleMLSSM'
    data: ClassVar[str] = 'doubleml.DoubleMLSSMData'
    roles: ClassVar[dict[str, str]] = {
        'selected': 's_col', 'instrument': 'z_cols'}
    required: ClassVar[tuple[str, ...]] = ('selected',)

    def _adjust(self, setup: _Setup) -> pd.DataFrame:
        """Returns the data, with 0 for the labels that were not seen.

        DoubleML needs a number in every row, although it does not use the
        labels of the rows that were not selected.

        Args:
            setup: the data and settings of the inferer, with "selected".

        Returns:
            The data.

        """
        frame = setup.frame.copy()
        selected = frame[setup.options['selected']] == 1
        frame[setup.label] = frame[setup.label].where(selected, 0.0)
        return frame

    def _estimator(self, setup: _Setup, data: Any) -> Any:
        """Returns DoubleML's `DoubleMLSSM`.

        Args:
            setup: the data and settings of the inferer, with "selected" and
                "instrument".
            data: DoubleML's data of the label, treatment, and features.

        Returns:
            The estimator, ready to fit.

        """
        _require_values(setup.frame[setup.treatment], 2, self.name)
        score = 'nonignorable' if setup.options.get('instrument') else (
            'missing-at-random')
        tool = utilities.import_tool(self.contents)
        return tool(
            data,
            setup.learners.of_outcome('regress'),
            setup.learners.of_treatment('classify'),
            setup.learners.of_treatment('classify'),
            n_folds = setup.n_folds,
            score = score)


""" DoWhy """


@dataclasses.dataclass
class Dowhy(Inferer, abc.ABC):
    """Genre of DoWhy's methods of estimation.

    DoWhy models the features as causes of both the treatment and the label
    (and any instrument as a cause of the treatment alone), identifies the
    effect from that model, and estimates it with the technique's method.
    Methods without a formula for their standard errors find them by
    bootstrapping, with "simulations" samples (100 by default). "refuters"
    names DoWhy's refuters to run: "placebo_treatment_refuter" (a random
    treatment should have no effect), "random_common_cause" (an added random
    cause should not change the effect), and "data_subset_refuter" (the
    effect should be about the same in subsets of the rows). Their results
    are stored in `tables` as "{name}_refutations".

    DoWhy is installed with the `causal` extra on Python 3.13 and later.

    Wraps:
        [`dowhy.CausalModel`](https://www.pywhy.org/dowhy/v0.14/dowhy.html#dowhy.causal_model.CausalModel)
        from DoWhy, which identifies, estimates, and refutes each effect.

    """

    # The name of DoWhy's method of estimation.
    method: ClassVar[str] = 'backdoor.linear_regression'
    # Whether DoWhy finds the method's standard error by bootstrapping.
    bootstrapped: ClassVar[bool] = True
    # Whether the method needs a treatment with two values.
    binary: ClassVar[bool] = True
    # Whether the method needs an instrument.
    instrumented: ClassVar[bool] = False

    def implement(
        self,
        item: base.Dataset,
        *,
        treatment: str | None = None,
        simulations: int = 100,
        refuters: Sequence[str] | str | None = None,
        instrument: str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Estimates the effect of `treatment` on the label with DoWhy.

        Args:
            item: the dataset. It must have a label.
            treatment: name of the column with the treatment. Defaults to
                `None`, but it is required.
            simulations: number of samples for bootstrapping (and for the
                refuters). Defaults to 100.
            refuters: names of DoWhy's refuters to run. Defaults to `None`.
            instrument: name of the column with an instrument, for
                `instrumental_variable`. Defaults to `None`.
            **kwargs: not used.

        Raises:
            ValueError: if there is no treatment (or a needed instrument), the
                treatment does not have two values for a method that needs
                them, a feature is not a number, a refuter is not one of
                those above, or DoWhy cannot identify the effect.

        Returns:
            The dataset, with the estimate in `tables`.

        """
        treatment = _require_treatment(treatment, self.name)
        if self.instrumented and instrument is None:
            message = f'{self.name!r} needs the name of an "instrument" column'
            raise ValueError(message)
        names = _listed(refuters)
        unknown = [r for r in names if r not in _REFUTERS]
        if unknown:
            message = (
                f'the refuters {unknown} are not among {sorted(_REFUTERS)}')
            raise ValueError(message)
        dowhy = _import_dowhy()
        columns = [treatment, *([instrument] if instrument else [])]
        frame, label, controls = _prepare(item, columns, self.name)
        if self.binary:
            _require_values(frame[treatment], 2, self.name)
        parameters = self._method_parameters(_task(frame[label]) == 'classify')
        with _quiet(), _seeded(item.seed):
            model = dowhy.CausalModel(
                data = frame,
                treatment = treatment,
                outcome = label,
                common_causes = controls or None,
                instruments = [instrument] if instrument else None)
            estimand = model.identify_effect(proceed_when_unidentifiable = True)
            estimate = model.estimate_effect(
                estimand,
                method_name = self.method,
                method_params = parameters or None)
            if estimate.value is None:
                message = (
                    f'DoWhy could not identify the effect for {self.name!r}')
                raise ValueError(message)
            lower, upper, error = self._uncertainty(estimate, simulations)
            refutations = _refute(
                model, estimand, estimate, names, simulations = simulations,
                seed = item.seed)
        table = _estimates(
            [treatment], [float(estimate.value)], [error], lower = [lower],
            upper = [upper])
        item.tables[self.name] = table
        if refutations is not None:
            item.tables[f'{self.name}_refutations'] = refutations
        item.fitted[self.name] = estimate
        _record(
            item, self.name, estimate, table, treatment = treatment,
            method = self.method, simulations = simulations,
            **({'instrument': instrument} if instrument else {}),
            **({'refuters': names} if names else {}))
        return item

    """ Private Methods """

    def _method_parameters(
        self,
        binary: bool) -> dict[str, Any]:  # noqa: ARG002
        """Returns DoWhy's parameters for the method.

        Args:
            binary: whether the label has two classes.

        Returns:
            The parameters.

        """
        return {}

    def _uncertainty(
        self,
        estimate: Any,
        simulations: int) -> tuple[float, float, float]:
        """Returns the 95% confidence interval and standard error of `estimate`.

        Args:
            estimate: DoWhy's estimate.
            simulations: number of samples for bootstrapping.

        Returns:
            The lower and upper ends of the interval and the standard error.

        """
        if self.bootstrapped:
            interval = estimate.get_confidence_intervals(
                num_simulations = simulations)
            error = estimate.get_standard_error(num_simulations = simulations)
        else:
            interval = estimate.get_confidence_intervals()
            error = estimate.get_standard_error()
        lower, upper = (float(v) for v in np.ravel(interval)[:2])
        return lower, upper, float(np.ravel(error)[0])


@dataclasses.dataclass
class DistanceMatching(Dowhy):
    """DoWhy's matching of treated rows with the most similar untreated rows.

    It matches rows by the distance between their features (rather than by
    their propensity scores) and compares the labels of matched rows. The
    treatment must have two values.

    Wraps:
        [`dowhy.causal_estimators.distance_matching_estimator.DistanceMatchingEstimator`](https://www.pywhy.org/dowhy/v0.14/dowhy.causal_estimators.html#dowhy.causal_estimators.distance_matching_estimator.DistanceMatchingEstimator)
        from DoWhy.

    """

    contents: Any = (
        'dowhy.causal_estimators.distance_matching_estimator.'
        'DistanceMatchingEstimator')
    method: ClassVar[str] = 'backdoor.distance_matching'


@dataclasses.dataclass
class DoublyRobust(Dowhy):
    """DoWhy's doubly robust estimator (augmented inverse propensity weighting).

    It combines a regression of the label with weighting by the propensity
    score, so that its estimate is right if either one is. The treatment
    must have two values.

    Wraps:
        [`dowhy.causal_estimators.doubly_robust_estimator.DoublyRobustEstimator`](https://www.pywhy.org/dowhy/v0.14/dowhy.causal_estimators.html#dowhy.causal_estimators.doubly_robust_estimator.DoublyRobustEstimator)
        from DoWhy.

    """

    contents: Any = (
        'dowhy.causal_estimators.doubly_robust_estimator.DoublyRobustEstimator')
    method: ClassVar[str] = 'backdoor.doubly_robust'


@dataclasses.dataclass
class GLMAdjustment(Dowhy):
    """DoWhy's generalized linear model of the label, adjusted for the features.

    For a label with two classes, it is a logistic regression, and the effect
    is the change in the chance of the second class. Otherwise, it is a
    linear regression. The treatment must have two values.

    Wraps:
        [`dowhy.causal_estimators.generalized_linear_model_estimator.GeneralizedLinearModelEstimator`](https://www.pywhy.org/dowhy/v0.14/dowhy.causal_estimators.html#dowhy.causal_estimators.generalized_linear_model_estimator.GeneralizedLinearModelEstimator)
        from DoWhy.

    """

    contents: Any = (
        'dowhy.causal_estimators.generalized_linear_model_estimator.'
        'GeneralizedLinearModelEstimator')
    method: ClassVar[str] = 'backdoor.generalized_linear_model'

    def _method_parameters(self, binary: bool) -> dict[str, Any]:
        """Returns the family of the model: binomial or Gaussian.

        Args:
            binary: whether the label has two classes.

        Returns:
            The parameters.

        """
        families = utilities.import_tool('statsmodels.genmod.families')
        family = families.Binomial() if binary else families.Gaussian()
        return {'glm_family': family}


@dataclasses.dataclass
class InstrumentalVariable(Dowhy):
    """DoWhy's estimate of an effect through an instrument.

    The instrument ("instrument") must change the treatment but affect the
    label only through it, so that the treatment may share causes with the
    label that are not among the features. With an instrument of two values,
    it is the Wald estimator, and otherwise two-stage least squares.

    Wraps:
        [`dowhy.causal_estimators.instrumental_variable_estimator.InstrumentalVariableEstimator`](https://www.pywhy.org/dowhy/v0.14/dowhy.causal_estimators.html#dowhy.causal_estimators.instrumental_variable_estimator.InstrumentalVariableEstimator)
        from DoWhy.

    """

    contents: Any = (
        'dowhy.causal_estimators.instrumental_variable_estimator.'
        'InstrumentalVariableEstimator')
    method: ClassVar[str] = 'iv.instrumental_variable'
    binary: ClassVar[bool] = False
    instrumented: ClassVar[bool] = True


@dataclasses.dataclass
class PropensityMatching(Dowhy):
    """DoWhy's matching of treated and untreated rows by propensity score.

    Each row is matched with the row of the other group whose chance of
    treatment (given the features) is closest, and the labels of matched
    rows are compared. The treatment must have two values.

    Wraps:
        [`dowhy.causal_estimators.propensity_score_matching_estimator.PropensityScoreMatchingEstimator`](https://www.pywhy.org/dowhy/v0.14/dowhy.causal_estimators.html#dowhy.causal_estimators.propensity_score_matching_estimator.PropensityScoreMatchingEstimator)
        from DoWhy.

    """

    contents: Any = (
        'dowhy.causal_estimators.propensity_score_matching_estimator.'
        'PropensityScoreMatchingEstimator')
    method: ClassVar[str] = 'backdoor.propensity_score_matching'


@dataclasses.dataclass
class PropensityStratification(Dowhy):
    """DoWhy's comparison of the treated and untreated in strata of propensity.

    The rows are divided into groups (strata) of similar propensity scores,
    and the effect is the average of the differences within them. The
    treatment must have two values.

    Wraps:
        [`dowhy.causal_estimators.propensity_score_stratification_estimator.PropensityScoreStratificationEstimator`](https://www.pywhy.org/dowhy/v0.14/dowhy.causal_estimators.html#dowhy.causal_estimators.propensity_score_stratification_estimator.PropensityScoreStratificationEstimator)
        from DoWhy.

    """

    contents: Any = (
        'dowhy.causal_estimators.propensity_score_stratification_estimator.'
        'PropensityScoreStratificationEstimator')
    method: ClassVar[str] = 'backdoor.propensity_score_stratification'


@dataclasses.dataclass
class PropensityWeighting(Dowhy):
    """DoWhy's inverse propensity weighting.

    Each row is weighted by the inverse of its chance of the treatment that
    it had, so that the treated and untreated rows resemble each other. The
    treatment must have two values.

    Wraps:
        [`dowhy.causal_estimators.propensity_score_weighting_estimator.PropensityScoreWeightingEstimator`](https://www.pywhy.org/dowhy/v0.14/dowhy.causal_estimators.html#dowhy.causal_estimators.propensity_score_weighting_estimator.PropensityScoreWeightingEstimator)
        from DoWhy.

    """

    contents: Any = (
        'dowhy.causal_estimators.propensity_score_weighting_estimator.'
        'PropensityScoreWeightingEstimator')
    method: ClassVar[str] = 'backdoor.propensity_score_weighting'


@dataclasses.dataclass
class RegressionAdjustment(Dowhy):
    """DoWhy's linear regression of the label, which adjusts for the features.

    The effect is the treatment's coefficient in a linear regression of the
    label on the treatment and features, with its standard error from the
    regression. The treatment can be a number or have two values.

    Wraps:
        [`dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator`](https://www.pywhy.org/dowhy/v0.14/dowhy.causal_estimators.html#dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator)
        from DoWhy.

    """

    contents: Any = (
        'dowhy.causal_estimators.linear_regression_estimator.'
        'LinearRegressionEstimator')
    method: ClassVar[str] = 'backdoor.linear_regression'
    bootstrapped: ClassVar[bool] = False
    binary: ClassVar[bool] = False


""" tigramite """


@dataclasses.dataclass
class Tigramite(Inferer, abc.ABC):
    """Genre of tigramite's methods of causal discovery in time series.

    The rows are a time series, in order, and the variables are the label
    and the features (which must be numbers, without missing values). The
    method tests which variables, at which lags (up to "max_lag", 5 by
    default), cause which others, with tigramite's test of conditional
    independence "test": "parcorr" (partial correlation, by default),
    "robust_parcorr" (for variables that are not normally distributed), or
    "cmiknn" (conditional mutual information, for relationships that are not
    linear). The table has a row for each link that it finds at the level
    "alpha" (0.05 by default): the cause, the effect, the lag, the kind of
    link ("-->" for a cause, and "o-o" or "x-x" for links whose direction is
    not known or is in conflict), the strength of the link (for "parcorr",
    the partial correlation), and its p-value.

    Wraps:
        - [`tigramite.independence_tests.parcorr.ParCorr`](https://jakobrunge.github.io/tigramite/#tigramite.independence_tests.parcorr.ParCorr)
          from tigramite, for the "parcorr" test.
        - [`tigramite.independence_tests.robust_parcorr.RobustParCorr`](https://jakobrunge.github.io/tigramite/#tigramite.independence_tests.robust_parcorr.RobustParCorr)
          from tigramite, for the "robust_parcorr" test.
        - [`tigramite.independence_tests.cmiknn.CMIknn`](https://jakobrunge.github.io/tigramite/#tigramite.independence_tests.cmiknn.CMIknn)
          from tigramite, for the "cmiknn" test.

    """

    def implement(
        self,
        item: base.Dataset,
        *,
        max_lag: int = 5,
        alpha: float = 0.05,
        test: str = 'parcorr',
        **kwargs: Any) -> base.Dataset:
        """Finds the causal links among the label and features.

        Args:
            item: the dataset, whose rows are a time series in order.
            max_lag: the longest lag to test. Defaults to 5.
            alpha: level at which links are significant. Defaults to 0.05.
            test: name of the test of conditional independence. Defaults to
                "parcorr".
            **kwargs: not used.

        Returns:
            The dataset, with the links in `tables`.

        """
        series, names = _series(item, self.name)
        with _quiet():
            results = self._discover(
                series, _independence_test(test, item.seed), max_lag, alpha)
        table = _links(results, names)
        item.tables[self.name] = table
        item.fitted[self.name] = results
        item.record(
            self.name, tool = utilities.describe_tool(self.contents),
            max_lag = max_lag, alpha = alpha, test = test, links = len(table))
        return item

    """ Private Methods """

    @abc.abstractmethod
    def _discover(
        self,
        series: Any,
        test: Any,
        max_lag: int,
        alpha: float) -> dict[str, Any]:
        """Returns tigramite's results: its graph and the links' statistics.

        Args:
            series: tigramite's data frame of the time series.
            test: tigramite's test of conditional independence.
            max_lag: the longest lag to test.
            alpha: level at which links are significant.

        Returns:
            The graph and the links' statistics.

        """


@dataclasses.dataclass
class LPCMCI(Tigramite):
    """tigramite's LPCMCI: causal discovery in time series with hidden causes.

    Unlike `pcmci_plus`, it allows for causes that are not among the
    variables (latent confounders), so a link may be "o->" (one end not
    known) or "<->" (caused by a hidden variable) as well as "-->".

    Wraps:
        [`tigramite.lpcmci.LPCMCI.run_lpcmci`](https://jakobrunge.github.io/tigramite/#tigramite.lpcmci.LPCMCI.run_lpcmci)
        from tigramite.

    """

    contents: Any = 'tigramite.lpcmci.LPCMCI'

    def _discover(
        self,
        series: Any,
        test: Any,
        max_lag: int,
        alpha: float) -> dict[str, Any]:
        """Returns the results of `LPCMCI.run_lpcmci`.

        Args:
            series: tigramite's data frame of the time series.
            test: tigramite's test of conditional independence.
            max_lag: the longest lag to test.
            alpha: level at which links are significant.

        Returns:
            The graph and the links' statistics.

        """
        tool = utilities.import_tool(self.contents)
        method = tool(series, cond_ind_test = test, verbosity = 0)
        results: dict[str, Any] = method.run_lpcmci(
            tau_min = 0, tau_max = max_lag, pc_alpha = alpha)
        return results


@dataclasses.dataclass
class PCMCI(Tigramite):
    """tigramite's PCMCI: which variables cause which others at later times.

    It finds the lagged causes (at lags of 1 to "max_lag") of each variable,
    with the PC algorithm and then the momentary conditional independence
    test, which keeps false links rare even when the series are
    autocorrelated.

    Wraps:
        [`tigramite.pcmci.PCMCI.run_pcmci`](https://jakobrunge.github.io/tigramite/#tigramite.pcmci.PCMCI.run_pcmci)
        from tigramite.

    """

    contents: Any = 'tigramite.pcmci.PCMCI'

    def _discover(
        self,
        series: Any,
        test: Any,
        max_lag: int,
        alpha: float) -> dict[str, Any]:
        """Returns the results of `PCMCI.run_pcmci`.

        Args:
            series: tigramite's data frame of the time series.
            test: tigramite's test of conditional independence.
            max_lag: the longest lag to test.
            alpha: level at which links are significant.

        Returns:
            The graph and the links' statistics.

        """
        tool = utilities.import_tool(self.contents)
        method = tool(series, cond_ind_test = test, verbosity = 0)
        results: dict[str, Any] = method.run_pcmci(
            tau_min = 1, tau_max = max_lag, pc_alpha = None,
            alpha_level = alpha)
        return results


@dataclasses.dataclass
class PCMCIPlus(Tigramite):
    """tigramite's PCMCI+: causes at later times and at the same time.

    Like `pcmci`, but it also finds links between variables at the same time
    (lag 0), whose direction may not be known ("o-o").

    Wraps:
        [`tigramite.pcmci.PCMCI.run_pcmciplus`](https://jakobrunge.github.io/tigramite/#tigramite.pcmci.PCMCI.run_pcmciplus)
        from tigramite.

    """

    contents: Any = 'tigramite.pcmci.PCMCI'

    def _discover(
        self,
        series: Any,
        test: Any,
        max_lag: int,
        alpha: float) -> dict[str, Any]:
        """Returns the results of `PCMCI.run_pcmciplus`.

        Args:
            series: tigramite's data frame of the time series.
            test: tigramite's test of conditional independence.
            max_lag: the longest lag to test.
            alpha: level at which links are significant.

        Returns:
            The graph and the links' statistics.

        """
        tool = utilities.import_tool(self.contents)
        method = tool(series, cond_ind_test = test, verbosity = 0)
        results: dict[str, Any] = method.run_pcmciplus(
            tau_min = 0, tau_max = max_lag, pc_alpha = alpha)
        return results


@dataclasses.dataclass
class TimeSeriesEffect(Tigramite):
    """tigramite's causal effect of one series (the treatment) on the label.

    It finds the causes of each variable with `pcmci`, fits a linear model of
    each variable on its causes, and multiplies the coefficients along every
    causal path from "treatment" at "lag" (1 by default) earlier to the
    label (tigramite's `LinearMediation`). The effect is in the units of the
    data, and it is 0 if no causal path joins them. Its confidence interval
    comes from "simulations" bootstrap samples (100 by default). The links
    that `pcmci` found are stored in `tables` as "{name}_links".

    Wraps:
        - [`tigramite.pcmci.PCMCI.run_pcmci`](https://jakobrunge.github.io/tigramite/#tigramite.pcmci.PCMCI.run_pcmci)
          from tigramite, to find the causes of each series.
        - [`tigramite.models.LinearMediation`](https://jakobrunge.github.io/tigramite/#tigramite.models.LinearMediation)
          from tigramite, to estimate the effect.

    """

    contents: Any = 'tigramite.models.LinearMediation'

    def implement(
        self,
        item: base.Dataset,
        *,
        treatment: str | None = None,
        lag: int = 1,
        max_lag: int = 5,
        alpha: float = 0.05,
        test: str = 'parcorr',
        simulations: int = 100,
        **kwargs: Any) -> base.Dataset:
        """Estimates the effect of `treatment` on the label `lag` periods later.

        Args:
            item: the dataset, whose rows are a time series in order. It must
                have a label.
            treatment: name of the column with the treatment. Defaults to
                `None`, but it is required.
            lag: number of periods between the treatment and the label.
                Defaults to 1.
            max_lag: the longest lag to test. Defaults to 5.
            alpha: level at which links are significant. Defaults to 0.05.
            test: name of the test of conditional independence. Defaults to
                "parcorr".
            simulations: number of bootstrap samples. Defaults to 100.
            **kwargs: not used.

        Raises:
            ValueError: if there is no treatment, or `lag` is less than 1 or
                more than `max_lag`.

        Returns:
            The dataset, with the estimate and the links in `tables`.

        """
        treatment = _require_treatment(treatment, self.name)
        label = item._require_label()
        if not 1 <= lag <= max_lag:
            message = f'lag must be from 1 to max_lag ({max_lag}), not {lag}'
            raise ValueError(message)
        series, names = _series(item, self.name)
        tool = utilities.import_tool(self.contents)
        cause, effect = names.index(treatment), names.index(label)
        arguments = {'i': cause, 'tau': lag, 'j': effect}
        with _quiet():
            results = self._discover(
                series, _independence_test(test, item.seed), max_lag, alpha)
            mediation = tool(series, data_transform = None)
            mediation.fit_model(
                all_parents = _parents(results['graph']), tau_max = max_lag)
            estimate = float(mediation.get_ce(**arguments))
            mediation.fit_model_bootstrap(
                boot_blocklength = 1, seed = item.seed,
                boot_samples = simulations)
            lower, upper = (
                float(v) for v in mediation.get_bootstrap_of(
                    function = 'get_ce', function_args = arguments,
                    conf_lev = 0.95))
        table = _estimates(
            [f'{treatment} (lag {lag})'], [estimate],
            [(upper - lower) / (2 * _Z_95)], lower = [lower], upper = [upper])
        item.tables[self.name] = table
        item.tables[f'{self.name}_links'] = _links(results, names)
        item.fitted[self.name] = mediation
        _record(
            item, self.name, mediation, table, treatment = treatment,
            lag = lag, max_lag = max_lag, alpha = alpha, test = test,
            simulations = simulations)
        return item

    """ Private Methods """

    def _discover(
        self,
        series: Any,
        test: Any,
        max_lag: int,
        alpha: float) -> dict[str, Any]:
        """Returns the results of `PCMCI.run_pcmci` (lagged causes only).

        Args:
            series: tigramite's data frame of the time series.
            test: tigramite's test of conditional independence.
            max_lag: the longest lag to test.
            alpha: level at which links are significant.

        Returns:
            The graph and the links' statistics.

        """
        return PCMCI()._discover(series, test, max_lag, alpha)


""" Private Functions """


def _code(values: pd.Series, value: Any) -> float:
    """Returns the number that stands for a level of a treatment.

    Args:
        values: the treatment.
        value: one of its levels.

    Raises:
        ValueError: if `value` is not one of the levels.

    Returns:
        The level itself if the treatment is numbers, and otherwise its
            position among the levels in sorted order.

    """
    if pd.api.types.is_numeric_dtype(values.dtype):
        return float(value)
    levels = _levels(values)
    if value not in levels:
        message = f'{value!r} is not one of the levels {levels}'
        raise ValueError(message)
    return float(levels.index(value))


def _estimates(
    index: Sequence[Hashable],
    coefficients: Sequence[float] | np.ndarray,
    errors: Sequence[float] | np.ndarray,
    *,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    name: str | None = None) -> pd.DataFrame:
    """Returns a table of estimates, with their test statistics and p-values.

    Wraps:
        [`scipy.stats.norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
        from SciPy, for the p-values.

    Args:
        index: the label of each estimate.
        coefficients: the estimates.
        errors: their standard errors.
        lower: the lower ends of their 95% confidence intervals.
        upper: the upper ends of their 95% confidence intervals.
        name: name of the index. Defaults to `None`.

    Returns:
        The table, with the columns of `_COLUMNS`. The statistic is the
            estimate divided by its standard error, and the p-value is
            two-sided, from the normal distribution. Both are missing if the
            standard error is 0 or missing.

    """
    stats = importlib.import_module('scipy.stats')
    coefficients = np.asarray(coefficients, dtype = float)
    errors = np.asarray(errors, dtype = float)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        statistic = np.where(errors > 0, coefficients / errors, np.nan)
    return pd.DataFrame(
        {
            'coefficient': coefficients,
            'standard_error': errors,
            'statistic': statistic,
            'p_value': 2 * stats.norm.sf(np.abs(statistic)),
            'ci_lower': np.asarray(lower, dtype = float),
            'ci_upper': np.asarray(upper, dtype = float)},
        index = pd.Index(list(index), name = name))


def _folds(
    rows: int,
    n_folds: int,
    seed: int | None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns folds of the rows for cross-fitting.

    Wraps:
        [`sklearn.model_selection.KFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
        from scikit-learn.

    Args:
        rows: number of rows.
        n_folds: number of folds.
        seed: seed for shuffling the rows.

    Returns:
        The positions of the rows used to fit and to apply the models, for
            each fold.

    """
    model_selection = importlib.import_module('sklearn.model_selection')
    splitter = model_selection.KFold(
        n_splits = n_folds, shuffle = True, random_state = seed)
    return list(splitter.split(np.arange(rows)))


def _from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Returns DoubleML's table of estimates with the columns of `_COLUMNS`.

    Args:
        summary: DoubleML's summary, whose first six columns are the
            estimate, standard error, statistic, p-value, and confidence
            interval.

    Returns:
        The table.

    """
    table = summary.iloc[:, :6].copy()
    table.columns = list(_COLUMNS)
    return table


def _import_causalml() -> Any:
    """Returns the `causalml` module, with its meta-learners imported.

    causalml sets matplotlib's style (to "fivethirtyeight") when its metrics
    are imported, which the meta-learners import, so matplotlib's settings
    are restored after importing them.

    Wraps:
        [causalml](https://causalml.readthedocs.io/en/latest/),
        the package that it imports.

    Raises:
        ImportError: if causalml cannot be imported, with why (it has no
            version for Python 3.13 and later).

    Returns:
        The module.

    """
    if sys.version_info >= (3, 13):
        message = (
            'causalml has no version for Python 3.13 and later, so its '
            'techniques need Python 3.11 or 3.12'
        )
        raise ImportError(message)
    with _quiet():
        module = utilities.import_tool('causalml')
        # causalml needs matplotlib, so it is installed.
        matplotlib = importlib.import_module('matplotlib')
        with matplotlib.rc_context():
            importlib.import_module('causalml.inference.meta')
    return module


def _import_dowhy() -> Any:
    """Returns the `dowhy` module.

    Wraps:
        [DoWhy](https://www.pywhy.org/dowhy/v0.14/),
        the package that it imports.

    Raises:
        ImportError: if DoWhy cannot be imported, with how to install it.

    Returns:
        The module.

    """
    try:
        with _quiet():
            return importlib.import_module('dowhy')
    except ImportError as error:
        message = (
            f'{error}. The causal extra ("pip install amos[causal]") installs '
            f'DoWhy on Python 3.13 and later. On Python 3.11 and 3.12, the '
            f'versions of DoWhy that work with pandas 3 cannot be installed '
            f'with causalml (which the extra installs there), but "pip '
            f'install dowhy" installs it without causalml'
        )
        raise ImportError(message) from error


def _independence_test(name: str, seed: int | None) -> Any:
    """Returns one of tigramite's tests of conditional independence.

    Wraps:
        - [`tigramite.independence_tests.parcorr.ParCorr`](https://jakobrunge.github.io/tigramite/#tigramite.independence_tests.parcorr.ParCorr)
          from tigramite, for "parcorr".
        - [`tigramite.independence_tests.robust_parcorr.RobustParCorr`](https://jakobrunge.github.io/tigramite/#tigramite.independence_tests.robust_parcorr.RobustParCorr)
          from tigramite, for "robust_parcorr".
        - [`tigramite.independence_tests.cmiknn.CMIknn`](https://jakobrunge.github.io/tigramite/#tigramite.independence_tests.cmiknn.CMIknn)
          from tigramite, for "cmiknn".

    Args:
        name: "parcorr", "robust_parcorr", or "cmiknn".
        seed: seed for the shuffles of "cmiknn".

    Raises:
        ValueError: if `name` is not one of those.

    Returns:
        The test, with analytic p-values (or, for "cmiknn", p-values from
            shuffles).

    """
    if name not in _TESTS:
        message = f'test must be one of {sorted(_TESTS)}, not {name!r}'
        raise ValueError(message)
    tool = utilities.import_tool(_TESTS[name])
    if name == 'cmiknn':
        return tool(significance = 'shuffle_test', seed = seed)
    return tool(significance = 'analytic')


def _learner(name: str, task: str, seed: int | None) -> Any:
    """Returns an unfitted model, chosen by the name of an `amos` model.

    Args:
        name: name of an `amos` model (such as "random_forest").
        task: "classify" or "regress".
        seed: seed for the model, if it takes one.

    Raises:
        TypeError: if `name` is not the name of a model.
        ValueError: if the model cannot do `task`.

    Returns:
        The model, built with the `amos` model's default parameters.

    """
    technique = chrisjen.library.borrow(name, genre = 'model')()
    if not isinstance(technique, models.Model):
        message = f'{name!r} is not a model'
        raise TypeError(message)
    tool = technique.contents or technique.tools.get(task)
    if tool is None:
        message = f'the model {name!r} cannot {task}'
        raise ValueError(message)
    built = utilities.import_tool(tool)
    # A model's defaults (such as the kind of a statsmodels model) can depend
    # on the task, so they are chosen for an empty dataset of the task.
    parameters = technique._prepare(
        base.Dataset(task = task),
        {**technique._keywords(), 'random_state': seed})
    return built(**utilities.accepted_parameters(built, parameters))


def _levels(values: pd.Series) -> list[Any]:
    """Returns the distinct values of a column, sorted if they can be.

    Args:
        values: a column.

    Returns:
        Its distinct values that are not missing.

    """
    levels = list(values.dropna().unique().tolist())
    try:
        return sorted(levels)
    except TypeError:
        return levels


def _links(results: dict[str, Any], names: Sequence[str]) -> pd.DataFrame:
    """Returns the causal links in tigramite's results, one for each row.

    Args:
        results: tigramite's results, with its "graph", "val_matrix", and
            "p_matrix".
        names: the names of the variables, in the order of the results.

    Returns:
        The links, with their "cause", "effect", "lag", kind of "link",
            "strength", and "p_value". A link at lag 0 is listed once.

    """
    graph = results['graph']
    rows = [
        {
            'cause': names[i],
            'effect': names[j],
            'lag': tau,
            'link': str(graph[i, j, tau]),
            'strength': float(results['val_matrix'][i, j, tau]),
            'p_value': float(results['p_matrix'][i, j, tau])}
        for i in range(len(names)) for j in range(len(names))
        for tau in range(graph.shape[2])
        if graph[i, j, tau] != '' and (tau > 0 or i < j)]
    columns = ['cause', 'effect', 'lag', 'link', 'strength', 'p_value']
    return pd.DataFrame(rows, columns = columns)


def _listed(item: Any) -> list[Any]:
    """Returns a setting as a `list`.

    Args:
        item: `None`, a `str` (which may list items separated by commas), or
            a sequence.

    Returns:
        The items (an empty `list` for `None`).

    """
    if item is None:
        return []
    if isinstance(item, str):
        return [i.strip() for i in item.split(',') if i.strip()]
    if isinstance(item, Sequence):
        return list(item)
    return [item]


def _numbers(values: pd.Series) -> pd.Series:
    """Returns `values` as numbers.

    Args:
        values: a column.

    Returns:
        The column as floats. A column with two values that are not numbers
            becomes 1 for the last value (in sorted order) and 0 for the
            other.

    """
    if pd.api.types.is_numeric_dtype(values.dtype):
        return values.astype(float)
    levels = sorted(values.dropna().unique().tolist())
    numbers: pd.Series = (values == levels[-1]).astype(float)
    return numbers


def _parents(graph: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """Returns the lagged causes of each variable in tigramite's graph.

    Args:
        graph: tigramite's graph, whose entry for cause `i`, effect `j`, and
            lag `tau` is "-->" if `i` at `tau` periods before causes `j`.

    Returns:
        For each variable, its causes, as (variable, negative lag) pairs.

    """
    variables, _, lags = graph.shape
    return {
        j: [
            (i, -tau) for i in range(variables) for tau in range(1, lags)
            if graph[i, j, tau] == '-->']
        for j in range(variables)}


def _prepare(
    item: base.Dataset,
    columns: Sequence[str],
    technique: str) -> tuple[pd.DataFrame, str, list[str]]:
    """Returns the data of an inferer: the label, `columns`, and the controls.

    The controls are the features that are not among `columns`.

    Args:
        item: the dataset. It must have a label.
        columns: names of the treatment and of any other columns with roles
            in the model (such as an instrument).
        technique: name of the technique, for messages.

    Raises:
        KeyError: if a column is not in the data.
        ValueError: if a control is not a number or boolean.

    Returns:
        The data (as numbers, see `_numbers`), the name of the label, and the
            names of the controls.

    """
    label = item._require_label()
    missing = [c for c in columns if c not in item.data.columns]
    if missing:
        message = f'the columns {missing} are not in the data'
        raise KeyError(message)
    controls = [c for c in item.features if c not in columns]
    unusable = [
        c for c in controls
        if base._kind_of(item.data[c]) not in {'numerics', 'booleans'}]
    if unusable:
        message = (
            f'{technique!r} needs numbers, but {unusable} are not: encode or '
            f'remove them first'
        )
        raise ValueError(message)
    frame = item.data[controls].astype(float)
    frame[label] = _numbers(item.y)
    for column in columns:
        frame[column] = _numbers(item.data[column])
    return frame, label, controls


def _propensity(
    x: np.ndarray,
    treated: np.ndarray,
    model: Any,
    n_folds: int,
    seed: int | None) -> np.ndarray:
    """Returns cross-fitted probabilities of treatment (propensity scores).

    Wraps:
        - [`sklearn.model_selection.StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
          from scikit-learn, for the folds.
        - [`sklearn.model_selection.cross_val_predict`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html)
          from scikit-learn, to predict each fold.

    Args:
        x: the features.
        treated: 1 for each treated row and 0 for the others.
        model: an unfitted classifier.
        n_folds: number of folds.
        seed: seed for the folds.

    Returns:
        The probability of treatment of each row, from a model fitted to the
            other folds, kept away from 0 and 1.

    """
    model_selection = importlib.import_module('sklearn.model_selection')
    folds = model_selection.StratifiedKFold(
        n_splits = n_folds, shuffle = True, random_state = seed)
    probabilities = model_selection.cross_val_predict(
        model, x, treated, cv = folds, method = 'predict_proba')[:, 1]
    return np.asarray(np.clip(probabilities, 0.01, 0.99), dtype = float)


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Keeps the inferers' packages from printing, logging, and warning.

    causalml prints a message when it is imported, and the packages log
    their progress and warn about the data.

    Yields:
        None: control, to the code in the context.

    """
    loggers = [logging.getLogger(n) for n in _LOGGERS]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            warnings.catch_warnings()):
            warnings.simplefilter('ignore')
            yield
    finally:
        for logger, level in zip(loggers, levels, strict = True):
            logger.setLevel(level)


def _record(
    item: base.Dataset,
    technique: str,
    tool: Any,
    table: pd.DataFrame,
    **details: Any) -> None:
    """Records what an inferer found in the dataset's history.

    Args:
        item: the dataset.
        technique: name of the technique.
        tool: the estimator that found the effect.
        table: the table of estimates.
        **details: the inferer's parameters to record.

    """
    effect = float(table['coefficient'].iloc[0]) if len(table) else None
    item.record(
        technique,
        tool = utilities.describe_tool(type(tool)),
        **details,
        effect = effect)


def _refute(
    model: Any,
    estimand: Any,
    estimate: Any,
    refuters: Sequence[str],
    *,
    simulations: int,
    seed: int | None) -> pd.DataFrame | None:
    """Returns the results of DoWhy's refuters of an estimate.

    Wraps:
        [`dowhy.CausalModel.refute_estimate`](https://www.pywhy.org/dowhy/v0.14/dowhy.html#dowhy.causal_model.CausalModel.refute_estimate)
        from DoWhy.

    Args:
        model: DoWhy's causal model.
        estimand: the estimand that DoWhy identified.
        estimate: DoWhy's estimate.
        refuters: names of the refuters.
        simulations: number of simulations of each refuter.
        seed: seed for the refuters.

    Returns:
        A row for each refuter, with the "effect", the "new_effect" that the
            refuter found, and its "p_value", or `None` if there are no
            refuters.

    """
    if not refuters:
        return None
    rows = {}
    for name in refuters:
        refutation = model.refute_estimate(
            estimand, estimate, method_name = name,
            num_simulations = simulations, random_seed = seed)
        result = refutation.refutation_result or {}
        rows[name] = {
            'effect': float(refutation.estimated_effect),
            'new_effect': float(refutation.new_effect),
            'p_value': float(result.get('p_value', np.nan))}
    table = pd.DataFrame.from_dict(rows, orient = 'index')
    table.index.name = 'refuter'
    return table


def _require_treatment(treatment: str | None, technique: str) -> str:
    """Returns the name of the treatment, which is required.

    Args:
        treatment: name of the treatment, or `None`.
        technique: name of the technique, for the message.

    Raises:
        ValueError: if `treatment` is `None`.

    Returns:
        The name of the treatment.

    """
    if treatment is None:
        message = f'{technique!r} needs the name of a "treatment" column'
        raise ValueError(message)
    return treatment


def _require_values(values: pd.Series, count: int, technique: str) -> None:
    """Raises an error if a column does not have `count` values.

    Args:
        values: a column.
        count: the number of values that it must have.
        technique: name of the technique, for the message.

    Raises:
        ValueError: if `values` does not have `count` distinct values.

    """
    found = values.nunique(dropna = True)
    if found != count:
        message = (
            f'{technique!r} needs {values.name!r} to have {count} values, '
            f'but it has {found}'
        )
        raise ValueError(message)


@contextlib.contextmanager
def _seeded(seed: int | None) -> Iterator[None]:
    """Seeds `numpy`'s global random numbers in the context.

    DoubleML (for inner folds), DoWhy (for bootstrapping), and causalml draw
    from `numpy`'s global random numbers, which are put back as they were
    afterwards.

    Args:
        seed: the seed.

    Yields:
        None: control, to the code in the context.

    """
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(seed)  # noqa: NPY002
    try:
        yield
    finally:
        np.random.set_state(state)  # noqa: NPY002


def _series(item: base.Dataset, technique: str) -> tuple[Any, list[str]]:
    """Returns tigramite's data frame of the label and features.

    Wraps:
        [`tigramite.data_processing.DataFrame`](https://jakobrunge.github.io/tigramite/#tigramite.data_processing.DataFrame)
        from tigramite.

    Args:
        item: the dataset, whose rows are a time series in order.
        technique: name of the technique, for messages.

    Raises:
        ValueError: if a variable is not a number or has missing values.

    Returns:
        tigramite's data frame, and the names of its variables (the features
            and the label, in the order of the columns).

    """
    names = [
        c for c in item.data.columns
        if c in item.features or c == item.label]
    unusable = [
        c for c in names
        if base._kind_of(item.data[c]) not in {'numerics', 'booleans'}]
    if unusable:
        message = (
            f'{technique!r} needs numbers, but {unusable} are not: encode or '
            f'remove them first'
        )
        raise ValueError(message)
    values = item.data[names].astype(float)
    if values.isna().to_numpy().any():
        message = f'{technique!r} needs a time series without missing values'
        raise ValueError(message)
    processing = utilities.import_tool('tigramite.data_processing')
    return processing.DataFrame(values.to_numpy(), var_names = names), names


def _task(values: pd.Series) -> str:
    """Returns the task of a model of a column.

    Args:
        values: a column.

    Returns:
        "classify" if the column has two values, and otherwise "regress".

    """
    two = values.nunique(dropna = True) == 2  # noqa: PLR2004
    return 'classify' if two else 'regress'
