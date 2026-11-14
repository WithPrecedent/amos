"""causalml's estimators of the effects of a treatment.

Contents:
    Causalml: genres of the techniques of each package.
    DRLearner, SLearner, TLearner, XLearner: causalml's meta-learners.
    TMLE: causalml's targeted maximum likelihood estimation.

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
import sys
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .. import base, utilities
from ._base import (
    _Z_95,
    Inference,
    Inferer,
    _Learners,
    _Setup,
    _estimates,
    _levels,
    _prepare,
    _quiet,
    _require_treatment,
    _seeded,
)


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

    def infer(
        self,
        item: base.Dataset,
        *,
        treatment: str | None = None,
        outcome_model: str = 'random_forest',
        treatment_model: str | None = None,
        control: Any = None,
        n_folds: int = 5,
        **kwargs: Any) -> Inference:
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
            The estimates.

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
        learners = _Learners.create(
            item, outcome_model, treatment_model or outcome_model, self.name)
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
        tables = {} if effects is None else {
            'effects': pd.DataFrame(
                np.asarray(effects).reshape(len(item.data), -1),
                index = item.data.index,
                columns = names)}
        return Inference(
            table = table,
            fitted = estimator,
            tables = tables,
            details = {
                'treatment': treatment,
                'control': control,
                'outcome_model': outcome_model,
                'treatment_model': treatment_model or outcome_model})

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


""" Private Functions """


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
