"""DoubleML's estimators of the effects of a treatment.

Contents:
    Doubleml: genres of the techniques of each package.
    DifferenceInDifferences: DoubleML's difference-in-differences, over two or
        more periods.
    InteractiveIV: DoubleML's effect of a treatment with an instrument.
    InteractiveRegression: DoubleML's average effect of a treatment with two
        values.
    PartiallyLinear: DoubleML's effect of a treatment that adds to the label.
    PartiallyLinearIV: DoubleML's partially linear model with an instrument.
    PartiallyLinearPanel: DoubleML's partially linear model of a panel.
    PartiallyLogistic: DoubleML's partially linear model of a label with two
        classes.
    PotentialOutcomes: DoubleML's effects of each level of a treatment.
    QuantileEffects: DoubleML's effects on quantiles of the label.
    RegressionDiscontinuity: DoubleML's effect at a cutoff of a running
        variable.
    SampleSelection: DoubleML's effect when the label is not always seen.

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .. import base, utilities
from ._base import (
    Inference,
    Inferer,
    _Learners,
    _Setup,
    _levels,
    _listed,
    _prepare,
    _require_treatment,
    _require_values,
    _seeded,
    _task,
)

# The columns of a table of estimates.
_COLUMNS: tuple[str, ...] = (
    'coefficient', 'standard_error', 'statistic', 'p_value', 'ci_lower',
    'ci_upper')


# DoubleML's groups of units that difference-in-differences compares the
# treated with.
_COMPARISONS: frozenset[str] = frozenset({'never_treated', 'not_yet_treated'})


# DoubleML's scores of `quantile_effects`, by the names that it takes.
_QUANTILE_SCORES: dict[str, str] = {
    'cvar': 'CVaR', 'local_quantile': 'LPQ', 'quantile': 'PQ'}


# The column that numbers the rows, for DoubleML's data of units.
_ROW: str = '_amos_row'


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

    def infer(
        self,
        item: base.Dataset,
        *,
        treatment: str | None = None,
        outcome_model: str = 'random_forest',
        treatment_model: str | None = None,
        n_folds: int = 5,
        **kwargs: Any) -> Inference:
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
            The estimates.

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
        learners = _Learners.create(
            item, outcome_model, treatment_model or outcome_model, self.name)
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
        return Inference(
            table = self._table(setup, estimator),
            fitted = estimator,
            details = {
                'treatment': treatment,
                'outcome_model': outcome_model,
                'treatment_model': treatment_model or outcome_model,
                'n_folds': n_folds,
                **named})

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
