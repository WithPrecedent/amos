"""DoWhy's estimators of the effects of a treatment.

Contents:
    Dowhy: genres of the techniques of each package.
    DistanceMatching, PropensityMatching, PropensityStratification,
        PropensityWeighting: DoWhy's matching, stratification, and weighting.
    DoublyRobust: DoWhy's doubly robust estimator.
    GLMAdjustment, RegressionAdjustment: DoWhy's regressions that adjust for the
        features.
    InstrumentalVariable: DoWhy's estimator with an instrument.

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .. import base, utilities
from ._base import (
    Inference,
    Inferer,
    _estimates,
    _listed,
    _prepare,
    _quiet,
    _require_treatment,
    _require_values,
    _seeded,
    _task,
)

# DoWhy's refuters that its inferers can run.
_REFUTERS: frozenset[str] = frozenset({
    'data_subset_refuter', 'placebo_treatment_refuter',
    'random_common_cause'})


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

    def infer(
        self,
        item: base.Dataset,
        *,
        treatment: str | None = None,
        simulations: int = 100,
        refuters: Sequence[str] | str | None = None,
        instrument: str | None = None,
        **kwargs: Any) -> Inference:
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
            The estimate, and the refutations of it if any were asked for.

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
        return Inference(
            table = table,
            fitted = estimate,
            tables = (
                {} if refutations is None else {'refutations': refutations}),
            details = {
                'treatment': treatment,
                'method': self.method,
                'simulations': simulations,
                **({'instrument': instrument} if instrument else {}),
                **({'refuters': names} if names else {})})

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


""" Private Functions """


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
