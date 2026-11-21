"""Adapters that give statistical packages the scikit-learn interface.

Contents:
    Statsmodel: adapts statsmodels regressions to the scikit-learn interface.
    FixedEffects: adapts pyfixest regressions to the scikit-learn interface.
    ProportionalHazards: adapts statsmodels' Cox model to the scikit-learn
        interface.
    GaussianProcessBoosting: adapts GPBoost's boosting with random effects to
        the scikit-learn interface.
    BayesianMixedModel: adapts Bambi's Bayesian (mixed) regressions to the
        scikit-learn interface.
    BayesianAdditiveTrees: adapts PyMC's Bayesian additive regression trees
        to the scikit-learn interface.

"""

from __future__ import annotations

import dataclasses
import importlib
import warnings
from collections.abc import Callable, Sequence
from typing import Any, cast

import camina
import numpy as np
import pandas as pd

from ...core import utilities

# What each kind of `Statsmodel` predicts: "values" (numbers), "counts" (whole
# numbers of 0 or more), "binary" (one of two classes), or "classes" (one of
# any number of classes). A binomial "glm" or "gee" is "binary".
_STATSMODELS: dict[str, str] = {
    'binomial_bayes_mixedglm': 'binary',
    'gee': 'values',
    'generalized_poisson': 'counts',
    'glm': 'values',
    'logit': 'binary',
    'mixedlm': 'values',
    'mnlogit': 'classes',
    'negative_binomial': 'counts',
    'ols': 'values',
    'ordinal_regression': 'classes',
    'poisson': 'counts',
    'probit': 'binary',
    'quantile_regression': 'values',
    'robust_regression': 'values',
    'wls': 'values',
    'zero_inflated_poisson': 'counts'}
# Kinds of `Statsmodel` whose fitting methods need features that are not
# combinations of each other (collinear), so collinear features are left out.
# The others (least squares and generalized linear models) find coefficients
# for collinear features themselves.
_INDEPENDENT: frozenset[str] = frozenset({
    'binomial_bayes_mixedglm',
    'gee',
    'generalized_poisson',
    'logit',
    'mixedlm',
    'mnlogit',
    'negative_binomial',
    'ordinal_regression',
    'poisson',
    'probit',
    'zero_inflated_poisson'})


@dataclasses.dataclass
class Statsmodel:
    """Adapts a statsmodels regression to the scikit-learn interface.

    statsmodels reports the standard errors, test statistics, p-values, and
    confidence intervals that academic research usually needs. The
    `coefficients` method returns them as a table. `kind` chooses the model:

    | Kind | statsmodels model | Predicts |
    | --- | --- | --- |
    | "ols" | [`OLS`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) | Values. |
    | "wls" | [`WLS`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html) | Values, from rows with `weights`. |
    | "glm" | [`GLM`](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html) | Values, or a class (for the "binomial" `family`). |
    | "gee" | [`GEE`](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_estimating_equations.GEE.html) | The same as "glm", for rows correlated within `groups`. |
    | "mixedlm" | [`MixedLM`](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html) | Values, with a random intercept for each of `groups`. |
    | "quantile_regression" | [`QuantReg`](https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html) | A `quantile` of the label. |
    | "robust_regression" | [`RLM`](https://www.statsmodels.org/stable/generated/statsmodels.robust.robust_linear_model.RLM.html) | Values, with less weight on outliers. |
    | "poisson" | [`Poisson`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Poisson.html) | Counts. |
    | "negative_binomial" | [`NegativeBinomial`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html) | Counts that vary more than a Poisson's. |
    | "generalized_poisson" | [`GeneralizedPoisson`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.GeneralizedPoisson.html) | Counts that vary more or less than a Poisson's. |
    | "zero_inflated_poisson" | [`ZeroInflatedPoisson`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.count_model.ZeroInflatedPoisson.html) | Counts with more zeros than a Poisson's. |
    | "logit" | [`Logit`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html) | One of two classes. |
    | "probit" | [`Probit`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Probit.html) | One of two classes. |
    | "binomial_bayes_mixedglm" | [`BinomialBayesMixedGLM`](https://www.statsmodels.org/stable/generated/statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.html) | One of two classes, with a random intercept for each of `groups`. |
    | "mnlogit" | [`MNLogit`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.MNLogit.html) | One of any number of classes. |
    | "ordinal_regression" | [`OrderedModel`](https://www.statsmodels.org/stable/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html) | One of ordered classes. |

    Wraps:
        [statsmodels](https://www.statsmodels.org/stable/),
        whose model for each kind is linked in the table.

    Args:
        kind: the kind of model, from the table. Defaults to "ols".
        family: family of a "glm" or "gee", by the name of a class in
            `statsmodels.api.families` in snake case ("binomial", "gaussian",
            "poisson", "negative_binomial", "gamma", and so on). A "binomial"
            model is a classifier of two classes. Defaults to "gaussian".
        constant: whether to add a constant (intercept). An
            "ordinal_regression" has thresholds between its classes instead,
            so it never has one. Defaults to `True`.
        groups: name of the column that identifies each row's group, for a
            "gee", "mixedlm", or "binomial_bayes_mixedglm" model. Defaults to
            `None`.
        weights: name of the column with the weight of each row, for "wls".
            Defaults to `None`.
        quantile: the quantile that a "quantile_regression" predicts.
            Defaults to 0.5 (the median).
        norm: the norm of a "robust_regression", by the name of a class in
            `statsmodels.robust.norms` in snake case ("huber_t",
            "tukey_biweight", "hampel", and so on). Defaults to "huber_t".
        distribution: "logit" or "probit", for an "ordinal_regression".
            Defaults to "logit".
        covariance: how the rows of a group are correlated in a "gee", by
            the name of a class in `statsmodels.genmod.cov_struct` in snake
            case ("independence" or "exchangeable"). Defaults to
            "independence".
        reference: the class that the coefficients of an "mnlogit" compare
            the other classes with. A class matches if it equals `reference`
            or has the same text (so "2" matches the class 2). Defaults to
            `None`, which uses the first class (sorted).

    Attributes:
        results: the fitted statsmodels results.
        classes_: the classes of a classifier, sorted.
        levels_: the classes of a model of any number of classes, in the
            order that statsmodels numbers them: for an "ordinal_regression",
            the order of the categories of a label that is an ordered
            categorical; for an "mnlogit", the `reference` class first; and
            otherwise sorted.
        feature_names_in_: the features (not the groups or weights).
        dropped_: features that were left out because they are combinations
            of the features before them (collinear), which most models
            fitted by maximum likelihood cannot estimate. Their rows in
            `coefficients` are empty.

    """

    kind: str = 'ols'
    family: str = 'gaussian'
    constant: bool = True
    groups: str | None = None
    weights: str | None = None
    quantile: float = 0.5
    norm: str = 'huber_t'
    distribution: str = 'logit'
    covariance: str = 'independence'
    reference: Any = None
    results: Any = dataclasses.field(default = None, init = False, repr = False)
    classes_: Any = dataclasses.field(
        default = None, init = False, repr = False)
    levels_: Any = dataclasses.field(default = None, init = False, repr = False)
    feature_names_in_: Any = dataclasses.field(
        default = None, init = False, repr = False)
    dropped_: list[str] = dataclasses.field(
        default_factory = list, init = False, repr = False)

    """ Properties """

    @property
    def coef_(self) -> np.ndarray:
        """Returns the coefficients of the features.

        An "mnlogit" has a row of coefficients for each class other than
        the reference, in the order of `levels_`. A feature that was left out (see `dropped_`) has a missing
        coefficient.

        """
        params = self._params()
        names = [str(n) for n in self.feature_names_in_]
        if isinstance(params, pd.DataFrame):
            return np.asarray(params.reindex(names)).T
        return np.asarray(params.reindex(names))

    @property
    def output(self) -> str:
        """Returns what the model predicts.

        Raises:
            ValueError: if `kind` is not a kind of model.

        Returns:
            "values" (numbers), "counts" (whole numbers of 0 or more),
                "binary" (one of two classes), or "classes" (one of any
                number of classes).

        """
        if self.kind in {'gee', 'glm'} and self.family == 'binomial':
            return 'binary'
        if self.kind not in _STATSMODELS:
            message = (
                f'{self.kind!r} is not a kind of statsmodels model: use one '
                f'of {sorted(_STATSMODELS)}'
            )
            raise ValueError(message)
        return _STATSMODELS[self.kind]

    @property
    def predict_proba(self) -> Callable[[pd.DataFrame], np.ndarray]:
        """Returns the function that predicts probabilities, for a classifier.

        Raises:
            AttributeError: if the model is not a classifier, so that a model
                that is not does not seem to predict probabilities.

        Returns:
            A function that takes features and returns the predicted
                probability of each class, in the order of `classes_`.

        """
        if self.output not in {'binary', 'classes'}:
            message = (
                'only a classifier predicts probabilities: a binomial glm or '
                'gee, logit_sm, probit, binomial_bayes_mixedglm, mnlogit, or '
                'ordinal_regression'
            )
            raise AttributeError(message)
        return self._probabilities

    """ Public Methods """

    def coefficients(self) -> pd.DataFrame:
        """Returns the coefficients and their statistics.

        Returns:
            One row for each coefficient, with its "coefficient",
                "standard_error", "statistic", "p_value", "ci_lower", and
                "ci_upper". The coefficients of an "mnlogit" are named
                "{feature} ({class})", for each class other than the
                reference. A "binomial_bayes_mixedglm" reports the
                mean and standard deviation of each coefficient's posterior
                and its 95% credible interval, and no p-values. A feature
                that was left out (see `dropped_`) has an empty row, and the
                statistics are missing if statsmodels could not find the
                covariance of the coefficients.

        """
        results = self.results
        labels = None
        if self.kind == 'binomial_bayes_mixedglm':
            means = self._params()
            deviations = pd.Series(
                np.asarray(results.fe_sd), index = means.index)
            # Variational Bayes gives each coefficient a normal posterior.
            table = pd.DataFrame({
                'coefficient': means,
                'standard_error': deviations,
                'statistic': means / deviations,
                'p_value': np.nan,
                'ci_lower': means - 1.96 * deviations,
                'ci_upper': means + 1.96 * deviations})
        else:
            params = results.params
            columns = [params] + [
                _statistic(results, name, params)
                for name in ('bse', 'tvalues', 'pvalues')]
            try:
                intervals = np.asarray(results.conf_int())
            except ValueError:
                intervals = np.full((params.size, 2), np.nan)
            if isinstance(params, pd.DataFrame):
                labels = [str(c) for c in self.levels_[1:]]
                columns = [_by_class(frame, labels) for frame in columns]
            # The intervals are in the same order as the coefficients (class
            # by class, for an mnlogit).
            table = pd.DataFrame({
                'coefficient': columns[0],
                'standard_error': columns[1],
                'statistic': columns[2],
                'p_value': columns[3],
                'ci_lower': intervals[:, 0],
                'ci_upper': intervals[:, 1]})
        empty = list(self.dropped_) if labels is None else [
            f'{name} ({label})' for label in labels for name in self.dropped_]
        return table.reindex([*table.index, *empty])

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Statsmodel:
        """Fits the model.

        Args:
            x: features, and any `groups` and `weights` columns.
            y: labels.

        Raises:
            ValueError: if the label does not suit the model.

        Returns:
            This adapter.

        """
        extras = {self.groups, self.weights} - {None}
        self.feature_names_in_ = np.asarray(
            [c for c in x.columns if c not in extras])
        self.dropped_ = []
        exog = self._exog(x)
        if self.kind in _INDEPENDENT:
            self.dropped_ = _collinear(exog)
            if self.dropped_:
                exog = exog.drop(columns = self.dropped_)
        self.results = self._estimate(self._endog(y), exog, x)
        return self

    def get_params(
        self,
        deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Returns the parameters, as scikit-learn expects.

        Args:
            deep: not used.

        Returns:
            The parameters.

        """
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self) if field.init}

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Returns predictions (the predicted class of a classifier).

        Args:
            x: features.

        Returns:
            The predictions.

        """
        output = self.output
        if output == 'binary':
            positive = (self._mean(x) >= 0.5).astype(int)  # noqa: PLR2004
            return np.asarray(self.classes_[positive])
        if output == 'classes':
            chosen = np.argmax(self._probabilities(x), axis = 1)
            return np.asarray(self.classes_[chosen])
        return self._mean(x)

    def set_params(self, **parameters: Any) -> Statsmodel:
        """Sets parameters, as scikit-learn expects.

        Args:
            **parameters: any of the parameters of the adapter.

        Returns:
            This adapter.

        """
        for key, value in parameters.items():
            setattr(self, key, value)
        return self

    """ Private Methods """

    def _column(self, x: pd.DataFrame, parameter: str) -> pd.Series:
        """Returns the column named by `parameter` (`groups` or `weights`).

        Args:
            x: features and other columns.
            parameter: "groups" or "weights".

        Raises:
            ValueError: if the parameter does not name a column.

        Returns:
            The column.

        """
        name = getattr(self, parameter)
        if name is None:
            message = (
                f'a {self.kind} model needs the name of a "{parameter}" column')
            raise ValueError(message)
        column: pd.Series = x[name]
        return column

    def _endog(self, y: pd.Series) -> np.ndarray:
        """Returns the label as statsmodels needs it.

        Args:
            y: labels.

        Raises:
            ValueError: if a model of two classes is given another number of
                classes, a model of counts is given other values, or the
                `reference` of an "mnlogit" is not one of its classes.

        Returns:
            The label as floats, or as 0 and 1 for a model of two classes, or
                as the position of each class in `levels_` for a model of any
                number of classes.

        """
        output = self.output
        target = np.asarray(y)
        if output in {'binary', 'classes'}:
            self.classes_ = np.unique(target)
        if output == 'binary':
            if len(self.classes_) != 2:  # noqa: PLR2004
                message = (
                    f'a {self._title} model needs a label with two classes')
                raise ValueError(message)
            return np.asarray(target == self.classes_[1], dtype = float)
        if output == 'classes':
            levels = list(self.classes_)
            dtype = getattr(y, 'dtype', None)
            if (
                self.kind == 'ordinal_regression'
                and isinstance(dtype, pd.CategoricalDtype)
                and dtype.ordered):
                levels = [c for c in dtype.categories if c in set(levels)]
            if self.kind == 'mnlogit' and self.reference is not None:
                # statsmodels compares the other classes with class 0.
                levels.insert(0, levels.pop(self._reference_position(levels)))
            self.levels_ = np.asarray(levels, dtype = object)
            positions = {c: i for i, c in enumerate(levels)}
            return np.asarray([positions[v] for v in target])
        values = target.astype(float)
        if output == 'counts' and not (
            np.isfinite(values).all()
            and (values >= 0).all()
            and (values == np.round(values)).all()):
            message = (
                f'a {self.kind} model needs counts: a label of whole numbers '
                f'of 0 or more'
            )
            raise ValueError(message)
        return values

    def _estimate(
        self,
        endog: np.ndarray,
        exog: pd.DataFrame,
        x: pd.DataFrame) -> Any:
        """Fits the statsmodels model of `kind`.

        Args:
            endog: the label, from `_endog`.
            exog: the features, from `_exog`.
            x: features and other columns (such as `groups`).

        Returns:
            The fitted statsmodels results.

        """
        api = importlib.import_module('statsmodels.api')
        match self.kind:
            case 'ols':
                return api.OLS(endog, exog).fit()
            case 'wls':
                weights = self._column(x, 'weights').astype(float)
                return api.WLS(endog, exog, weights = weights).fit()
            case 'glm':
                family = _statsmodels_class(api.families, self.family)()
                return api.GLM(endog, exog, family = family).fit()
            case 'gee':
                family = _statsmodels_class(api.families, self.family)()
                covariance = _statsmodels_class(
                    api.cov_struct, self.covariance)()
                return api.GEE(
                    endog,
                    exog,
                    groups = self._column(x, 'groups'),
                    family = family,
                    cov_struct = covariance).fit()
            case 'mixedlm':
                return api.MixedLM(
                    endog, exog, groups = self._column(x, 'groups')).fit()
            case 'binomial_bayes_mixedglm':
                # Each group has a random intercept: one variance component,
                # with a column of 0s and 1s for each group.
                groups = pd.get_dummies(
                    self._column(x, 'groups'),
                    dtype = float)
                return api.BinomialBayesMixedGLM(
                    endog,
                    exog,
                    exog_vc = groups.to_numpy(),
                    ident = np.zeros(groups.shape[1], dtype = int)).fit_vb()
            case 'quantile_regression':
                return api.QuantReg(endog, exog).fit(q = self.quantile)
            case 'robust_regression':
                norm = _statsmodels_class(api.robust.norms, self.norm)()
                return api.RLM(endog, exog, M = norm).fit()
            case 'ordinal_regression':
                ordinal = importlib.import_module(
                    'statsmodels.miscmodels.ordinal_model')
                return ordinal.OrderedModel(
                    endog, exog, distr = self.distribution).fit(
                        method = 'bfgs', disp = 0)
        # The discrete models of statsmodels are fitted by maximum
        # likelihood, which prints its progress unless `disp` is 0. `fit` has
        # already left out their collinear features, so they do not check the
        # rank of the features again (with a QR decomposition of a copy of
        # every row).
        model = {
            'generalized_poisson': api.GeneralizedPoisson,
            'logit': api.Logit,
            'mnlogit': api.MNLogit,
            'negative_binomial': api.NegativeBinomial,
            'poisson': api.Poisson,
            'probit': api.Probit,
            'zero_inflated_poisson': api.ZeroInflatedPoisson}[self.kind](
                endog, exog, check_rank = False)
        try:
            return model.fit(disp = 0)
        except np.linalg.LinAlgError:
            # Newton's method fails if a feature is a combination of others.
            # BFGS still finds coefficients, though statsmodels then warns
            # that it cannot find their standard errors.
            return model.fit(method = 'bfgs', maxiter = 1000, disp = 0)

    def _exog(self, x: pd.DataFrame) -> pd.DataFrame:
        """Returns the features as floats, with a constant if one is added.

        statsmodels needs one array of numbers, but the features can be of
        several types (such as small integers, nullable integers, and
        floats), so each is made a float, and a missing value a NaN. The
        floats are made in one array, by columns (as LAPACK uses them), which
        the returned frame and statsmodels share rather than copy, since a
        copy of every row of the features can be several gigabytes.

        Args:
            x: features, and any other columns, which are left out.

        Raises:
            ValueError: if a constant is added and a feature is named "const".

        Returns:
            The exogenous variables for statsmodels, without the features in
                `dropped_`.

        """
        sources = {str(c): c for c in self.feature_names_in_}
        names = list(sources)
        if self.constant and self.kind != 'ordinal_regression':
            if 'const' in sources:
                message = (
                    'a feature is named "const", which is the name of the '
                    'constant that the model adds')
                raise ValueError(message)
            names.insert(0, 'const')
        names = [n for n in names if n not in self.dropped_]
        values = np.empty((len(x), len(names)), order = 'F')
        for position, name in enumerate(names):
            if name in sources:
                values[:, position] = x[sources[name]].to_numpy(
                    dtype = float, na_value = np.nan)
            else:
                values[:, position] = 1.0
        return pd.DataFrame(
            values, index = x.index, columns = names, copy = False)

    def _mean(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the model's prediction of the mean of the label.

        Args:
            x: features.

        Returns:
            The predictions: the probability of the second class, for a model
                of two classes, or of each class in `levels_`, for a model of
                any number of classes.

        """
        return np.asarray(self.results.predict(self._exog(x)))

    def _params(self) -> pd.Series | pd.DataFrame:
        """Returns the estimated coefficients, named as strings.

        Returns:
            The coefficients, or the means of their posteriors for a
                "binomial_bayes_mixedglm" (whose `params` are not named and
                include its variance).

        """
        results = self.results
        if self.kind == 'binomial_bayes_mixedglm':
            names = [str(n) for n in results.model.exog_names]
            return pd.Series(np.asarray(results.fe_mean), index = names)
        params: pd.Series | pd.DataFrame = results.params.copy()
        params.index = params.index.map(str)
        return params

    def _probabilities(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the predicted probability of each class.

        Args:
            x: features.

        Returns:
            One column for each class, in the order of `classes_`.

        """
        values = self._mean(x)
        if self.output == 'binary':
            return np.column_stack([1 - values, values])
        levels = list(self.levels_)
        return values[:, [levels.index(c) for c in self.classes_]]

    def _reference_position(self, levels: list[Any]) -> int:
        """Returns the position of `reference` among the classes.

        A class matches if it equals `reference`, or else if it has the same
        text, so that a reference from a settings file (which may be text)
        matches a class that is a number.

        Args:
            levels: the classes of the label.

        Raises:
            ValueError: if no class matches `reference`.

        Returns:
            The position of the first class that matches.

        """
        for matches in (
            lambda c: c == self.reference,
            lambda c: str(c) == str(self.reference)):
            for position, level in enumerate(levels):
                if matches(level):
                    return position
        message = (
            f'the reference {self.reference!r} of a {self.kind} model is not '
            f'one of its classes: {levels}')
        raise ValueError(message)

    @property
    def _title(self) -> str:
        """Returns the name of the model for messages."""
        if self.kind in {'gee', 'glm'}:
            return f'{self.family} {self.kind}'
        return self.kind


@dataclasses.dataclass
class FixedEffects:
    """Adapts a pyfixest regression to the scikit-learn interface.

    pyfixest estimates regressions with many fixed effects (such as one for
    each firm or year) quickly, with standard errors clustered by any
    column, as R's fixest does. The `coefficients` method returns the
    coefficients of the features and their statistics as a table.

    pyfixest cannot predict a row whose level of a fixed effect has no
    estimate: a level that was not in the training rows, or one whose rows
    it dropped (a singleton, or a level whose rows all have the same class,
    which separates it in a logit or probit model). Such a row is predicted
    with that fixed effect at the average of its estimated levels (the
    unweighted mean), so every row with its features has a prediction and
    metrics such as the area under the ROC curve can score every row.

    Wraps:
        - [`pyfixest.feols`](https://py-econometrics.github.io/pyfixest/reference/estimation.api.feols.feols.html)
          from pyfixest, for the "gaussian" family.
        - [`pyfixest.fepois`](https://py-econometrics.github.io/pyfixest/reference/estimation.api.fepois.fepois.html)
          from pyfixest, for the "poisson" family.
        - [`pyfixest.feglm`](https://py-econometrics.github.io/pyfixest/reference/estimation.api.feglm.feglm.html)
          from pyfixest, for the "logit" and "probit" families.

    Args:
        fixed_effects: names of the columns whose levels get fixed effects.
            Defaults to `None`.
        cluster: name of the column to cluster the standard errors by.
            Defaults to `None`, which uses heteroskedasticity-robust standard
            errors.
        family: "gaussian" (least squares), "poisson" (counts), or "logit" or
            "probit" (two classes). Defaults to "gaussian".

    Attributes:
        results: the fitted pyfixest results.
        classes_: the two classes of a "logit" or "probit" model.
        feature_names_in_: the features (not the fixed effects or cluster).
        effect_levels_: the estimate of each level of each fixed effect, in
            the order of `fixed_effects`, once a row has needed them.

    """

    fixed_effects: Sequence[str] | str | None = None
    cluster: str | None = None
    family: str = 'gaussian'
    results: Any = dataclasses.field(default = None, init = False, repr = False)
    classes_: Any = dataclasses.field(
        default = None, init = False, repr = False)
    feature_names_in_: Any = dataclasses.field(
        default = None, init = False, repr = False)
    effect_levels_: list[dict[str, float]] | None = dataclasses.field(
        default = None, init = False, repr = False)

    """ Properties """

    @property
    def coef_(self) -> np.ndarray:
        """Returns the coefficients of the features."""
        return np.asarray(self.coefficients()['coefficient'])

    @property
    def is_binary(self) -> bool:
        """Returns whether the model classifies two classes."""
        return self.family in {'logit', 'probit'}

    """ Public Methods """

    def coefficients(self) -> pd.DataFrame:
        """Returns the coefficients and their statistics.

        Returns:
            One row for each feature, with its "coefficient",
                "standard_error", "statistic", "p_value", "ci_lower", and
                "ci_upper". The fixed effects themselves are not listed.

        """
        with utilities.preserved_logging():
            table: pd.DataFrame = self.results.tidy().iloc[:, :6].copy()
        table.columns = [
            'coefficient', 'standard_error', 'statistic', 'p_value',
            'ci_lower', 'ci_upper']
        names = dict(
            zip(self._safe_names(), self.feature_names_in_, strict = True))
        table.index = [names.get(str(n), str(n)) for n in table.index]
        return table

    def fit(self, x: pd.DataFrame, y: pd.Series) -> FixedEffects:
        """Fits the model.

        Args:
            x: features, fixed effect columns, and the cluster column.
            y: labels.

        Raises:
            ValueError: if a two-class model's label does not have two
                classes.

        Returns:
            This adapter.

        """
        pyfixest = utilities.import_tool('pyfixest')
        extras = [*self._effects(), *camina.listify(self.cluster)]
        self.feature_names_in_ = np.asarray(
            [c for c in x.columns if c not in extras])
        target = np.asarray(y)
        if self.is_binary:
            self.classes_ = np.unique(target)
            if len(self.classes_) != 2:  # noqa: PLR2004
                message = f'a {self.family} model needs two classes'
                raise ValueError(message)
            target = (target == self.classes_[1]).astype(float)
        self.effect_levels_ = None
        data = self._frame(x)
        data['amos_y'] = target.astype(float)
        regressors = ' + '.join(self._safe_names()) or '1'
        effects = ' + '.join(
            f'amos_fe{i}' for i in range(len(self._effects())))
        formula = f'amos_y ~ {regressors}'
        if effects:
            formula = f'{formula} | {effects}'
        vcov: Any = {'CRV1': 'amos_cluster'} if self.cluster else 'hetero'
        # pyfixest configures the root logger when it runs.
        with utilities.preserved_logging():
            if self.family == 'gaussian':
                self.results = pyfixest.feols(
                    formula, data = data, vcov = vcov)
            elif self.family == 'poisson':
                self.results = pyfixest.fepois(
                    formula, data = data, vcov = vcov)
            else:
                self.results = pyfixest.feglm(
                    formula, data = data, family = self.family, vcov = vcov)
        return self

    def get_params(
        self,
        deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Returns the parameters, as scikit-learn expects.

        Args:
            deep: not used.

        Returns:
            The parameters.

        """
        return {
            'fixed_effects': self.fixed_effects,
            'cluster': self.cluster,
            'family': self.family}

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Returns predictions (the predicted class of a two-class model).

        A row whose level of a fixed effect has no estimate is predicted with
        that fixed effect at the average of its estimated levels.

        Args:
            x: features and fixed effect columns.

        Returns:
            The predictions.

        """
        values = self._response(x)
        if self.is_binary:
            positive = (values >= 0.5).astype(int)  # noqa: PLR2004
            return np.asarray(self.classes_[positive])
        return values

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the predicted probability of each class (two classes only).

        A row whose level of a fixed effect has no estimate is predicted with
        that fixed effect at the average of its estimated levels.

        Args:
            x: features and fixed effect columns.

        Raises:
            AttributeError: if the model is not a two-class model.

        Returns:
            One column for each class, in the order of `classes_`.

        """
        if not self.is_binary:
            message = 'only a logit or probit model predicts probabilities'
            raise AttributeError(message)
        values = self._response(x)
        return np.column_stack([1 - values, values])

    def set_params(self, **parameters: Any) -> FixedEffects:
        """Sets parameters, as scikit-learn expects.

        Args:
            **parameters: "fixed_effects", "cluster", or "family".

        Returns:
            This adapter.

        """
        for key, value in parameters.items():
            setattr(self, key, value)
        return self

    """ Private Methods """

    def _effects(self) -> list[str]:
        """Returns the names of the fixed effect columns.

        Returns:
            The names in `fixed_effects`, as a `list`.

        """
        effects: list[str] = camina.listify(self.fixed_effects)
        return effects

    def _fallback(self, data: pd.DataFrame) -> np.ndarray:
        """Returns the linear predictor with unestimated levels at the mean.

        Args:
            data: rows from `_frame`.

        Returns:
            The linear predictor of each row, with the fixed effect of each
                level that has no estimate at the average of the estimated
                levels of that fixed effect.

        """
        link = np.zeros(len(data))
        for name, value in self.results.coef().items():
            link += data[str(name)].to_numpy(dtype = float) * float(value)
        for position, levels in enumerate(self._levels()):
            average = float(np.mean(list(levels.values()))) if levels else 0.0
            effects = data[f'amos_fe{position}'].map(levels)
            link += effects.fillna(average).to_numpy(dtype = float)
        return link

    def _frame(self, x: pd.DataFrame) -> pd.DataFrame:
        """Returns `x` with names that pyfixest's formulas accept.

        Args:
            x: features, fixed effect columns, and the cluster column.

        Returns:
            The features as floats named "amos_x{n}", the fixed effects named
                "amos_fe{n}", and the cluster named "amos_cluster".

        """
        frame = pd.DataFrame(index = x.index)
        for safe, name in zip(
            self._safe_names(), self.feature_names_in_, strict = True):
            frame[safe] = x[name].astype(float)
        for position, name in enumerate(self._effects()):
            frame[f'amos_fe{position}'] = x[name].astype(str)
        if self.cluster is not None and self.cluster in x.columns:
            frame['amos_cluster'] = x[self.cluster].astype(str)
        return frame

    def _levels(self) -> list[dict[str, float]]:
        """Returns the estimate of each level of each fixed effect.

        Returns:
            For each fixed effect, in the order of `fixed_effects`, the
                estimate of each of its levels, by the level as text.

        """
        if self.effect_levels_ is None:
            with utilities.preserved_logging():
                estimates = self.results.fixef()
            self.effect_levels_ = [
                {
                    str(level): float(value) for level, value in estimates.get(
                        f'C(amos_fe{position})', {}).items()}
                for position in range(len(self._effects()))]
        return self.effect_levels_

    def _response(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the model's predicted mean (a probability, for two classes).

        A row whose level of a fixed effect has no estimate is predicted with
        that fixed effect at the average of its estimated levels (see
        `_fallback`).

        Args:
            x: features and fixed effect columns.

        Returns:
            The predictions on the scale of the label.

        """
        data = self._frame(x)
        with utilities.preserved_logging():
            link = np.array(
                self.results.predict(newdata = data, type = 'link'),
                dtype = float)
        missing = np.isnan(link)
        if missing.any() and self._effects():
            link[missing] = self._fallback(data.loc[missing])
        if self.family in {'logit', 'probit'}:
            special = importlib.import_module('scipy.special')
            inverse = special.expit if self.family == 'logit' else special.ndtr
            return np.asarray(inverse(link))
        if self.family == 'poisson':
            return np.asarray(np.exp(link))
        return link

    def _safe_names(self) -> list[str]:
        """Returns the names of the features in pyfixest's formulas.

        Returns:
            A name for each feature ("amos_x0", "amos_x1", and so on),
                which pyfixest can parse whatever the features are
                called.

        """
        return [f'amos_x{n}' for n in range(len(self.feature_names_in_))]


@dataclasses.dataclass
class ProportionalHazards:
    """Adapts statsmodels' Cox regression to the scikit-learn interface.

    The label is the time until an event (such as a machine's failure) or until
    the row stopped being observed (censoring). Rows with the same time are
    handled with Efron's method. The model predicts the expected time until the
    event. The `coefficients` method returns the coefficients, hazard ratios,
    and their statistics as a table.

    Wraps:
        [`statsmodels.duration.hazard_regression.PHReg`](https://www.statsmodels.org/stable/generated/statsmodels.duration.hazard_regression.PHReg.html)
        from statsmodels.

    Args:
        event: name of the column that is 1 (or `True`) if the event
            happened and 0 if the row was censored. Defaults to `None`, which
            means that every event was observed.
        penalizer: strength of a ridge penalty on the coefficients, as in
            statsmodels' `fit_regularized` (the penalty is the number of rows
            times `penalizer` times half the sum of the squared
            coefficients). The penalty depends on the scales of the
            features, so scale them first. Defaults to 0.0.

    Attributes:
        results: the fitted statsmodels `PHRegResults`.
        feature_names_in_: the features (not the event column).

    """

    event: str | None = None
    penalizer: float = 0.0
    results: Any = dataclasses.field(default = None, init = False, repr = False)
    feature_names_in_: Any = dataclasses.field(
        default = None, init = False, repr = False)

    """ Properties """

    @property
    def coef_(self) -> np.ndarray:
        """Returns the coefficients of the features."""
        return np.asarray(self.results.params)

    """ Public Methods """

    def coefficients(self) -> pd.DataFrame:
        """Returns the coefficients and their statistics.

        Returns:
            One row for each feature, with its "coefficient", "hazard_ratio",
                "standard_error", "statistic", "p_value", "ci_lower", and
                "ci_upper" (of the coefficient).

        """
        results = self.results
        bounds = np.asarray(results.conf_int())
        return pd.DataFrame(
            {
                'coefficient': results.params,
                'hazard_ratio': np.exp(results.params),
                'standard_error': results.bse,
                'statistic': results.tvalues,
                'p_value': results.pvalues,
                'ci_lower': bounds[:, 0],
                'ci_upper': bounds[:, 1]},
            index = list(self.feature_names_in_))

    def fit(self, x: pd.DataFrame, y: pd.Series) -> ProportionalHazards:
        """Fits the model.

        Args:
            x: features and the event column.
            y: the time until the event or censoring.

        Returns:
            This adapter.

        """
        regression = utilities.import_tool(
            'statsmodels.duration.hazard_regression')
        data = self._features(x).astype(float)
        self.feature_names_in_ = np.asarray(data.columns)
        status = None
        if self.event is not None:
            status = x[self.event].astype(int).to_numpy()
        model = regression.PHReg(
            np.asarray(y, dtype = float), data.to_numpy(), status = status,
            ties = 'efron')
        if not self.penalizer:
            self.results = model.fit()
            return self
        params = model.fit_regularized(
            alpha = self.penalizer, L1_wt = 0.0).params
        # statsmodels gives no standard errors for a penalized fit. They come
        # from the curvature of the penalized log likelihood.
        information = -model.hessian(params) + (
            len(data) * self.penalizer * np.eye(len(params)))
        self.results = regression.PHRegResults(
            model, params, np.linalg.inv(information))
        return self

    def get_params(
        self,
        deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Returns the parameters, as scikit-learn expects.

        Args:
            deep: not used.

        Returns:
            The parameters.

        """
        return {'event': self.event, 'penalizer': self.penalizer}

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the expected time until the event.

        The expected time is the area under a row's survival curve, up to the
        longest time in the training rows.

        Args:
            x: features (and, optionally, the event column).

        Returns:
            The expected times.

        """
        features = self._features(x).astype(float).to_numpy()
        times, hazards, _ = self.results.baseline_cumulative_hazard[0]
        end = np.max(self.results.model.endog)
        # The survival curves are steps that start at 1 and drop at each time
        # with an event in the training rows.
        widths = np.diff(np.concatenate([[0.0], times, [end]]))
        risks = np.exp(features @ np.asarray(self.results.params))
        survival = np.exp(-np.outer(risks, np.concatenate([[0.0], hazards])))
        return survival @ widths

    def set_params(self, **parameters: Any) -> ProportionalHazards:
        """Sets parameters, as scikit-learn expects.

        Args:
            **parameters: "event" or "penalizer".

        Returns:
            This adapter.

        """
        for key, value in parameters.items():
            setattr(self, key, value)
        return self

    """ Private Methods """

    def _features(self, x: pd.DataFrame) -> pd.DataFrame:
        """Returns the columns of `x` that are features (not the event).

        Args:
            x: the columns that the model is given.

        Returns:
            `x` without the event column.

        """
        return x.drop(columns = [c for c in [self.event] if c in x.columns])


# The likelihoods of GPBoost that classify two classes. The others regress.
_GPBOOST_CLASSIFIERS: frozenset[str] = frozenset({
    'bernoulli_logit', 'bernoulli_probit'})


# It is not a dataclass: scikit-learn rebuilds a model with each of its
# parameters as a keyword, and GPBoost's booster has many, which are only
# known once gpboost is imported.
class GaussianProcessBoosting:
    """Adapts GPBoost's trees with random effects to scikit-learn.

    GPBoost boosts trees of the features, as LightGBM does, together with a
    model of the random effects of rows that are not independent: a random
    intercept for each level of each of `groups` (such as each school), a
    Gaussian process over `coordinates` (such as a latitude and a longitude),
    or both. The trees and the variances of the random effects are estimated
    together, so the trees learn the effects of the features apart from what
    the rows of a group (or of nearby places) share.

    A row of a group (or at a place) that the model has not seen is predicted
    with its random effect at zero, their average. Without the columns of
    `groups` and `coordinates` (as when a critic passes only the features),
    every row is predicted by the trees alone, as if it were of an unseen
    group. The `random_effects` method returns the variances of the random
    effects as a table.

    Wraps:
        - [`gpboost.GPBoostClassifier`](https://gpboost.readthedocs.io/en/latest/pythonapi/gpboost.GPBoostClassifier.html)
          from GPBoost, for the "bernoulli_logit" and "bernoulli_probit"
          likelihoods.
        - [`gpboost.GPBoostRegressor`](https://gpboost.readthedocs.io/en/latest/pythonapi/gpboost.GPBoostRegressor.html)
          from GPBoost, for the other likelihoods.
        - [`gpboost.GPModel`](https://gpboost.readthedocs.io/en/latest/pythonapi/gpboost.GPModel.html)
          from GPBoost, for the random effects.

    Args:
        groups: names of the columns whose levels each get a random
            intercept. Defaults to `None`.
        coordinates: names of the columns of the coordinates of a Gaussian
            process. Defaults to `None`.
        likelihood: the distribution of the label: "gaussian" (a number),
            "bernoulli_logit" or "bernoulli_probit" (two classes), or another
            of GPBoost's (such as "poisson" or "gamma"). Defaults to
            "gaussian".
        cov_function: the covariance function of the Gaussian process (such
            as "matern", "exponential", or "gaussian"). Defaults to
            "matern".
        gp_parameters: other arguments of GPBoost's `GPModel` (such as
            "gp_approx" for a large Gaussian process). Defaults to `None`.
        **parameters: parameters of GPBoost's booster, which are those of
            LightGBM's scikit-learn models (such as "n_estimators" and
            "learning_rate").

    Attributes:
        booster_: the fitted GPBoost booster.
        gp_model_: the fitted GPBoost `GPModel` of the random effects.
        classes_: the two classes of a classifier.
        feature_names_in_: the features (not the groups or coordinates).

    """

    """ Initialization Methods """

    def __init__(
        self,
        groups: Sequence[str] | str | None = None,
        coordinates: Sequence[str] | str | None = None,
        likelihood: str = 'gaussian',
        cov_function: str = 'matern',
        gp_parameters: dict[str, Any] | None = None,
        **parameters: Any) -> None:
        # The arguments are described in the class's docstring.
        self.groups = groups
        self.coordinates = coordinates
        self.likelihood = likelihood
        self.cov_function = cov_function
        self.gp_parameters = gp_parameters
        self.parameters = parameters
        self.booster_: Any = None
        self.gp_model_: Any = None
        self.classes_: Any = None
        self.feature_names_in_: Any = None

    """ Properties """

    @property
    def feature_importances_(self) -> np.ndarray:
        """Returns the importance of each feature to the trees."""
        return np.asarray(self.booster_.feature_importances_)

    @property
    def is_classifier(self) -> bool:
        """Returns whether the likelihood classifies two classes."""
        return self.likelihood in _GPBOOST_CLASSIFIERS

    @property
    def _estimator_type(self) -> str:
        """Returns the kind of estimator, for scikit-learn before 1.6."""
        return 'classifier' if self.is_classifier else 'regressor'

    """ Public Methods """

    def fit(self, x: pd.DataFrame, y: pd.Series) -> GaussianProcessBoosting:
        """Fits the trees and the random effects together.

        Args:
            x: features, and the `groups` and `coordinates` columns.
            y: labels.

        Raises:
            ValueError: if there are no `groups` or `coordinates`, or if a
                classifier's label does not have two classes.

        Returns:
            This adapter.

        """
        gpboost = utilities.import_tool('gpboost')
        if not self._extras():
            message = (
                'a GPBoost model needs "groups" or "coordinates" columns for '
                'its random effects (use lightgbm for boosting alone)')
            raise ValueError(message)
        features = self._features(x)
        self.feature_names_in_ = np.asarray(features.columns)
        target = np.asarray(y)
        if self.is_classifier:
            self.classes_, target = _two_classes(
                target,
                f'a GPBoost model with the {self.likelihood!r} likelihood')
        seed = self.parameters.get('random_state')
        self.gp_model_ = gpboost.GPModel(
            likelihood = self.likelihood,
            cov_function = self.cov_function,
            seed = seed if isinstance(seed, int) else 0,
            **self._random_effects(x),
            **(self.gp_parameters or {}))
        tool = (
            gpboost.GPBoostClassifier if self.is_classifier
            else gpboost.GPBoostRegressor)
        self.booster_ = tool(**self.parameters)
        self.booster_.fit(features, target, gp_model = self.gp_model_)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Returns the parameters, as scikit-learn expects.

        The parameters of the booster that are not set are GPBoost's
        defaults, so that every parameter that the booster accepts is named.

        Args:
            deep: not used.

        Returns:
            The parameters.

        """
        gpboost = utilities.import_tool('gpboost')
        tool = (
            gpboost.GPBoostClassifier if self.is_classifier
            else gpboost.GPBoostRegressor)
        return {
            'groups': self.groups,
            'coordinates': self.coordinates,
            'likelihood': self.likelihood,
            'cov_function': self.cov_function,
            'gp_parameters': self.gp_parameters,
            **tool().get_params(deep = False),
            **self.parameters}

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Returns predictions (the predicted class of a classifier).

        Args:
            x: features, and (to use the random effects) the `groups` and
                `coordinates` columns.

        Returns:
            The predictions.

        """
        values = self._response(x)
        if self.is_classifier:
            positive = (values >= 0.5).astype(int)  # noqa: PLR2004
            return np.asarray(self.classes_[positive])
        return values

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the predicted probability of each class (two classes only).

        Args:
            x: features, and (to use the random effects) the `groups` and
                `coordinates` columns.

        Raises:
            AttributeError: if the model is not a classifier.

        Returns:
            One column for each class, in the order of `classes_`.

        """
        if not self.is_classifier:
            message = (
                'only a GPBoost model with a bernoulli likelihood predicts '
                'probabilities')
            raise AttributeError(message)
        values = self._response(x)
        return np.column_stack([1 - values, values])

    def random_effects(self) -> pd.DataFrame:
        """Returns the estimated variances of the random effects.

        Returns:
            One row for each variance (or other parameter of the covariance,
                such as a Gaussian process's range), with its "estimate".
                GPBoost names a group's variance "Group_1", "Group_2", and so
                on, in the order of `groups`, which are named here instead.

        """
        estimates = self.gp_model_.get_cov_pars().iloc[0]
        names = {
            f'Group_{position}': name
            for position, name in enumerate(self._groups(), start = 1)}
        return pd.DataFrame(
            {'estimate': estimates.to_numpy(dtype = float)},
            index = [names.get(str(n), str(n)) for n in estimates.index])

    def score(self, x: pd.DataFrame, y: pd.Series) -> float:
        """Returns the accuracy of a classifier or the R² of a regression.

        Args:
            x: features, and the `groups` and `coordinates` columns.
            y: labels.

        Returns:
            The score, as scikit-learn's classifiers and regressors score.

        """
        return _score(self.predict(x), y, classify = self.is_classifier)

    def set_params(self, **parameters: Any) -> GaussianProcessBoosting:
        """Sets parameters, as scikit-learn expects.

        Args:
            **parameters: any parameters of the adapter or of the booster.

        Returns:
            This adapter.

        """
        own = {
            'groups', 'coordinates', 'likelihood', 'cov_function',
            'gp_parameters'}
        for key, value in parameters.items():
            if key in own:
                setattr(self, key, value)
            else:
                self.parameters[key] = value
        return self

    """ Private Methods """

    def _coordinates(self) -> list[str]:
        """Returns the names of the columns of coordinates.

        Returns:
            The names in `coordinates`, as a `list`.

        """
        coordinates: list[str] = camina.listify(self.coordinates)
        return coordinates

    def _extras(self) -> list[str]:
        """Returns the names of the columns that are not features.

        Returns:
            The names in `groups` and then in `coordinates`.

        """
        return [*self._groups(), *self._coordinates()]

    def _features(self, x: pd.DataFrame) -> pd.DataFrame:
        """Returns the columns of `x` that are features.

        Args:
            x: the columns that the model is given.

        Returns:
            `x` without the `groups` and `coordinates` columns.

        """
        return x.drop(columns = [c for c in self._extras() if c in x.columns])

    def _groups(self) -> list[str]:
        """Returns the names of the columns of groups.

        Returns:
            The names in `groups`, as a `list`.

        """
        groups: list[str] = camina.listify(self.groups)
        return groups

    def _random_effects(
        self,
        x: pd.DataFrame,
        suffix: str = '') -> dict[str, np.ndarray]:
        """Returns GPBoost's arguments of the groups and coordinates of `x`.

        The levels of the groups are passed as text, so that groups of any
        type (and a missing group, which is a level of its own) can be used.

        Args:
            x: the columns that the model is given.
            suffix: the suffix of GPBoost's arguments: "" to fit, or "_pred"
                to predict. Defaults to "".

        Returns:
            "group_data" and "gp_coords" (with `suffix`), for those that the
                model has.

        """
        arguments: dict[str, np.ndarray] = {}
        if self._groups():
            arguments[f'group_data{suffix}'] = (
                x[self._groups()].astype(str).to_numpy(dtype = str))
        if self._coordinates():
            arguments[f'gp_coords{suffix}'] = (
                x[self._coordinates()].to_numpy(dtype = float))
        return arguments

    def _response(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the predicted mean (a probability, for a classifier).

        Args:
            x: features, and (to use the random effects) the `groups` and
                `coordinates` columns.

        Returns:
            The predictions on the scale of the label.

        """
        features = x[list(self.feature_names_in_)]
        predict = (
            self.booster_.predict_proba if self.is_classifier
            else self.booster_.predict)
        if not all(c in x.columns for c in self._extras()):
            values = np.asarray(predict(features, ignore_gp_model = True))
            return values[:, 1] if values.ndim == 2 else values  # noqa: PLR2004
        predicted = predict(features, **self._random_effects(x, '_pred'))
        return np.asarray(predicted['response_mean'], dtype = float)

    """ Dunder Methods """

    def __sklearn_tags__(self) -> Any:
        """Returns scikit-learn's tags for a classifier or regressor.

        A search needs them to choose its folds and scorer.

        Returns:
            The tags (scikit-learn 1.6 and later).

        """
        return _estimator_tags(classify = self.is_classifier)


# The families of Bambi that classify two classes. The others regress.
_BAMBI_CLASSIFIERS: frozenset[str] = frozenset({'bernoulli'})
# The level of a group that a model has not seen, for rows without groups.
_UNSEEN: str = '__amos_unseen_group__'


@dataclasses.dataclass
class BayesianMixedModel:
    """Adapts Bambi's Bayesian (mixed) regressions to scikit-learn.

    Bambi writes a PyMC model from a formula, as R's lme4 and brms do, with
    weakly informative priors, and samples its posterior with PyMC's NUTS.
    Without a `formula`, the label is regressed on every feature, with a
    random intercept for each level of each of `groups` (as
    "label ~ x0 + x1 + (1|group)"). With one, the columns are passed by their
    own names, and the label is the name of `y`.

    The predictions are the posterior means of the response's mean (the
    probability of the second class, for a classifier), averaged over the
    posterior draws. Bambi gives a row of a group that the model has not
    seen draws of a new group's random effect, so its prediction is averaged
    over the groups that could have been. Without the `groups` columns (as
    when a critic passes only the features), every row is of such a group.
    The `coefficients` method returns the posterior of each common
    coefficient, and `random_effects` the standard deviation of each group's
    random effects.

    A fitted Bambi model cannot be pickled, so the adapter keeps the
    posterior and the training rows, and builds the model again (without
    sampling) when it is needed.

    Wraps:
        - [`bambi.Model`](https://bambinos.github.io/bambi/api/Model.html)
          from Bambi.

    Args:
        formula: the model's formula, in Bambi's syntax. Defaults to `None`,
            which regresses the label on every feature, with a random
            intercept for each of `groups`.
        groups: names of the columns whose levels each get a random
            intercept. Defaults to `None`.
        family: Bambi's family of the label: "gaussian" (a number),
            "bernoulli" (two classes), or another (such as "poisson",
            "negativebinomial", "gamma", or "t"). Defaults to "gaussian".
        priors: Bambi's priors, by term. Defaults to `None`, which uses
            Bambi's weakly informative priors.
        draws: number of posterior draws of each chain. Defaults to 1000.
        tune: number of tuning draws of each chain, which are discarded.
            Defaults to 1000.
        chains: number of chains. Defaults to 2.
        cores: number of chains to sample at once. Defaults to 1, which
            samples them in turn (and so in amos's parallel runs too).
        target_accept: NUTS's target acceptance rate, which a hierarchical
            model needs higher than PyMC's default to avoid divergences.
            Defaults to 0.9.
        random_state: seed of the sampler (and of the draws of new groups).
            Defaults to `None`.

    Attributes:
        posterior_: the posterior (and Bambi's other groups of results).
        classes_: the two classes of a classifier.
        feature_names_in_: the features (not the groups).
        data_: the training rows, as Bambi was given them.

    """

    formula: str | None = None
    groups: Sequence[str] | str | None = None
    family: str = 'gaussian'
    priors: dict[str, Any] | None = None
    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    cores: int = 1
    target_accept: float = 0.9
    random_state: int | None = None
    posterior_: Any = dataclasses.field(
        default = None, init = False, repr = False)
    classes_: Any = dataclasses.field(
        default = None, init = False, repr = False)
    feature_names_in_: Any = dataclasses.field(
        default = None, init = False, repr = False)
    data_: pd.DataFrame | None = dataclasses.field(
        default = None, init = False, repr = False)
    _model: Any = dataclasses.field(
        default = None, init = False, repr = False, compare = False)

    """ Properties """

    @property
    def coef_(self) -> np.ndarray:
        """Returns the posterior means of the common coefficients."""
        return np.asarray(self.coefficients()['coefficient'])

    @property
    def is_classifier(self) -> bool:
        """Returns whether the family classifies two classes."""
        return self.family in _BAMBI_CLASSIFIERS

    @property
    def _estimator_type(self) -> str:
        """Returns the kind of estimator, for scikit-learn before 1.6."""
        return 'classifier' if self.is_classifier else 'regressor'

    """ Public Methods """

    def coefficients(self) -> pd.DataFrame:
        """Returns the posterior of each common coefficient.

        Returns:
            One row for each coefficient (the intercept is "const", as in
                statsmodels' tables), with its "coefficient" (the posterior
                mean), "standard_error" (the posterior standard deviation),
                "statistic" (their ratio), "p_value" (missing, since a
                posterior has none), and "ci_lower" and "ci_upper" (the 95%
                equal-tailed credible interval).

        """
        model = self._bambi_model()
        component = model.parameters[model.family.likelihood.parent]
        posterior = self.posterior_['posterior']
        names = [
            name for name in ['Intercept', *component.common_terms]
            if name in posterior]
        rows: dict[str, np.ndarray] = {}
        for name in names:
            values = posterior[name]
            draws = np.asarray(values).reshape(
                values.sizes['chain'] * values.sizes['draw'], -1)
            levels = [d for d in values.dims if d not in ('chain', 'draw')]
            label = 'const' if name == 'Intercept' else self._original(name)
            if not levels:
                rows[label] = draws[:, 0]
                continue
            for position, level in enumerate(
                np.asarray(values[levels[0]]).ravel()):
                rows[f'{label}[{level}]'] = draws[:, position]
        return _posterior_table(rows)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> BayesianMixedModel:
        """Samples the posterior of the model.

        Args:
            x: features and the `groups` columns.
            y: labels.

        Raises:
            ValueError: if a classifier's label does not have two classes.

        Returns:
            This adapter.

        """
        self.feature_names_in_ = np.asarray(
            [c for c in x.columns if c not in self._groups()])
        target = np.asarray(y)
        if self.is_classifier:
            self.classes_, target = _two_classes(
                target, f'a Bambi model of the {self.family!r} family')
        data = self._frame(x)
        data[self._response_name(y)] = target
        self.data_ = data
        self._model = None
        with utilities.preserved_logging():
            self.posterior_ = self._bambi_model().fit(
                draws = self.draws,
                tune = self.tune,
                chains = self.chains,
                cores = self.cores,
                target_accept = self.target_accept,
                random_seed = self.random_state,
                progressbar = False)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Returns the parameters, as scikit-learn expects.

        Args:
            deep: not used.

        Returns:
            The parameters.

        """
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self) if field.init}

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Returns predictions (the predicted class of a classifier).

        Args:
            x: features, and (to use their random effects) the `groups`
                columns.

        Returns:
            The predictions.

        """
        values = self._response(x)
        if self.is_classifier:
            positive = (values >= 0.5).astype(int)  # noqa: PLR2004
            return np.asarray(self.classes_[positive])
        return values

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the predicted probability of each class (two classes only).

        Args:
            x: features, and (to use their random effects) the `groups`
                columns.

        Raises:
            AttributeError: if the model is not a classifier.

        Returns:
            One column for each class, in the order of `classes_`.

        """
        if not self.is_classifier:
            message = 'only a Bambi model of two classes predicts probabilities'
            raise AttributeError(message)
        values = self._response(x)
        return np.column_stack([1 - values, values])

    def random_effects(self) -> pd.DataFrame:
        """Returns the posterior of the standard deviation of random effects.

        Returns:
            One row for each group term (such as "1|court"), with the
                statistics of `coefficients`.

        """
        posterior = self.posterior_['posterior']
        rows = {}
        for name in posterior.data_vars:
            if '|' in str(name) and str(name).endswith('_sigma'):
                values = np.asarray(posterior[name]).ravel()
                rows[self._original(str(name).removesuffix('_sigma'))] = values
        return _posterior_table(rows)

    def score(self, x: pd.DataFrame, y: pd.Series) -> float:
        """Returns the accuracy of a classifier or the R² of a regression.

        Args:
            x: features and the `groups` columns.
            y: labels.

        Returns:
            The score, as scikit-learn's classifiers and regressors score.

        """
        return _score(self.predict(x), y, classify = self.is_classifier)

    def set_params(self, **parameters: Any) -> BayesianMixedModel:
        """Sets parameters, as scikit-learn expects.

        Args:
            **parameters: any of the parameters of the adapter.

        Returns:
            This adapter.

        """
        for key, value in parameters.items():
            setattr(self, key, value)
        return self

    """ Private Methods """

    def _bambi_model(self) -> Any:
        """Returns the Bambi model of the training rows, building it once.

        Returns:
            The model (not sampled), which shares the fitted posterior.

        """
        if self._model is None:
            bambi = utilities.import_tool('bambi')
            with utilities.preserved_logging():
                self._model = bambi.Model(
                    self._formula(),
                    self.data_,
                    family = self.family,
                    priors = self.priors)
        return self._model

    def _formula(self) -> str:
        """Returns the formula of the model.

        Returns:
            `formula`, or the label regressed on every feature with a random
                intercept for each of `groups` (with the names of `_frame`).

        """
        if self.formula is not None:
            return self.formula
        features = [f'amos_x{n}' for n in range(len(self.feature_names_in_))]
        terms = [
            *(features or ['1']),
            *(f'(1|amos_g{n})' for n in range(len(self._groups())))]
        return f'amos_y ~ {" + ".join(terms)}'

    def _frame(self, x: pd.DataFrame) -> pd.DataFrame:
        """Returns the columns of `x` as Bambi is given them.

        A group column that `x` does not have is given a level that the
        model has not seen.

        Args:
            x: features, and any of the `groups` columns.

        Returns:
            `x` itself, for a `formula`, or else the features as floats
                named "amos_x{n}" and the groups as text named "amos_g{n}".

        """
        frame = pd.DataFrame(index = x.index)
        if self.formula is not None:
            frame = x.copy()
            for name in self._groups():
                if name not in frame.columns:
                    frame[name] = _UNSEEN
            return frame
        for position, name in enumerate(self.feature_names_in_):
            frame[f'amos_x{position}'] = x[name].astype(float)
        for position, name in enumerate(self._groups()):
            frame[f'amos_g{position}'] = (
                x[name].astype(str) if name in x.columns else _UNSEEN)
        return frame

    def _groups(self) -> list[str]:
        """Returns the names of the columns of groups.

        Returns:
            The names in `groups`, as a `list`.

        """
        groups: list[str] = camina.listify(self.groups)
        return groups

    def _original(self, name: str) -> str:
        """Returns a term of the model with the names of the data.

        Args:
            name: a term, which may use the names of `_frame` (such as
                "amos_x0" or "1|amos_g0").

        Returns:
            The term with the names of the features and groups.

        """
        if self.formula is not None:
            return name
        for position, group in enumerate(self._groups()):
            name = name.replace(f'amos_g{position}', str(group))
        if name.startswith('amos_x') and name[6:].isdigit():
            return str(self.feature_names_in_[int(name[6:])])
        return name

    def _response(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the posterior mean of the response's mean.

        Args:
            x: features, and (to use their random effects) the `groups`
                columns.

        Returns:
            The mean of each row (the probability of the second class, for a
                classifier).

        """
        model = self._bambi_model()
        parent = model.family.likelihood.parent
        with utilities.preserved_logging(), warnings.catch_warnings():
            # Bambi needs "sample_new_groups" to draw the effects of new
            # groups, and warns that a later version will not.
            warnings.simplefilter('ignore')
            predicted = model.predict(
                self.posterior_,
                data = self._frame(x),
                inplace = False,
                sample_new_groups = True,
                random_seed = self.random_state)
        values = predicted['posterior'][parent]
        return np.asarray(values.mean(('chain', 'draw')), dtype = float)

    def _response_name(self, y: pd.Series) -> str:
        """Returns the name of the label in the data that Bambi is given.

        Args:
            y: labels.

        Returns:
            "amos_y", or the name of `y` (or "y") for a `formula`.

        """
        if self.formula is None:
            return 'amos_y'
        return str(y.name) if y.name is not None else 'y'

    """ Dunder Methods """

    def __getstate__(self) -> dict[str, Any]:
        """Returns the state to pickle, without the Bambi model.

        Returns:
            The attributes, with the model left out (it is built again when
                it is needed).

        """
        return {**self.__dict__, '_model': None}

    def __sklearn_tags__(self) -> Any:
        """Returns scikit-learn's tags for a classifier or regressor.

        Returns:
            The tags (scikit-learn 1.6 and later).

        """
        return _estimator_tags(classify = self.is_classifier)


# The likelihoods of `BayesianAdditiveTrees`: the inverse of the link of the
# sum of the trees to the mean of each.
_BART_LIKELIHOODS: dict[str, str] = {
    'normal': 'identity',
    'bernoulli': 'expit',
    'poisson': 'exp'}


@dataclasses.dataclass
class BayesianAdditiveTrees:
    """Adapts PyMC's Bayesian additive regression trees (BART) to scikit-learn.

    BART is a sum of `m` small trees, each with a prior that keeps it
    shallow, whose posterior PyMC samples (with pymc-bart's particle Gibbs
    sampler for the trees, and NUTS for any other parameter). Like a random
    forest, it needs no scaling and finds interactions and curves itself, and
    its posterior says how sure it is of each prediction. With the "normal"
    likelihood the trees are the mean of a number (with an estimated standard
    deviation), with "bernoulli" they are the log odds of the second of two
    classes, and with "poisson" the log of the mean of counts. The
    predictions are the posterior means.

    A fitted PyMC model cannot be pickled, so the adapter keeps the sampled
    trees of each chain, and rebuilds pymc-bart's sampler of them when it
    predicts. The importance of each feature is the share of the splits of
    the trees on it.

    Wraps:
        - [`pymc_bart.BART`](https://www.pymc.io/projects/bart/en/latest/api_reference.html)
          from pymc-bart, for the trees.
        - [`pymc.sample`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html)
          from PyMC, to sample their posterior.

    Args:
        m: number of trees. Defaults to 50.
        alpha: prior probability that a node at the root splits, which
            falls with depth. Defaults to 0.95.
        beta: how fast the probability that a node splits falls with its
            depth. Defaults to 2.0.
        likelihood: "normal", "bernoulli", or "poisson". Defaults to
            "normal".
        draws: number of posterior draws of each chain. Defaults to 1000.
        tune: number of tuning draws of each chain, which are discarded.
            Defaults to 1000.
        chains: number of chains. Defaults to 2.
        cores: number of chains to sample at once. Defaults to 1, which
            samples them in turn (and so in amos's parallel runs too).
        random_state: seed of the sampler. Defaults to `None`.

    Attributes:
        trees_: the sampled trees of each chain.
        classes_: the two classes of a classifier.
        feature_names_in_: the features.
        feature_importances_: the share of the splits on each feature.

    """

    m: int = 50
    alpha: float = 0.95
    beta: float = 2.0
    likelihood: str = 'normal'
    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    cores: int = 1
    random_state: int | None = None
    trees_: list[Any] | None = dataclasses.field(
        default = None, init = False, repr = False)
    classes_: Any = dataclasses.field(
        default = None, init = False, repr = False)
    feature_names_in_: Any = dataclasses.field(
        default = None, init = False, repr = False)
    feature_importances_: Any = dataclasses.field(
        default = None, init = False, repr = False)
    _outputs: int = dataclasses.field(
        default = 1, init = False, repr = False)
    _samplers: list[Any] | None = dataclasses.field(
        default = None, init = False, repr = False, compare = False)

    """ Properties """

    @property
    def is_classifier(self) -> bool:
        """Returns whether the likelihood classifies two classes."""
        return self.likelihood == 'bernoulli'

    @property
    def _estimator_type(self) -> str:
        """Returns the kind of estimator, for scikit-learn before 1.6."""
        return 'classifier' if self.is_classifier else 'regressor'

    """ Public Methods """

    def fit(self, x: pd.DataFrame, y: pd.Series) -> BayesianAdditiveTrees:
        """Samples the posterior of the trees.

        Args:
            x: features.
            y: labels.

        Raises:
            ValueError: if the likelihood is not known, or a classifier's
                label does not have two classes.

        Returns:
            This adapter.

        """
        pm = utilities.import_tool('pymc')
        bart = utilities.import_tool('pymc_bart')
        if self.likelihood not in _BART_LIKELIHOODS:
            message = (
                f'the likelihood of BART must be one of '
                f'{sorted(_BART_LIKELIHOODS)}, not {self.likelihood!r}')
            raise ValueError(message)
        self.feature_names_in_ = np.asarray(x.columns)
        features = np.ascontiguousarray(x.to_numpy(dtype = float))
        target = np.asarray(y)
        if self.is_classifier:
            self.classes_, target = _two_classes(
                target, 'BART with the "bernoulli" likelihood')
        target = target.astype(float)
        with utilities.preserved_logging(), pm.Model():
            trees = bart.BART(
                'mu', features, target,
                m = self.m, alpha = self.alpha, beta = self.beta)
            if self.likelihood == 'normal':
                sigma = pm.HalfNormal('sigma', float(np.std(target)) or 1.0)
                pm.Normal('y', trees, sigma, observed = target)
            elif self.likelihood == 'bernoulli':
                pm.Bernoulli('y', logit_p = trees, observed = target)
            else:
                pm.Poisson('y', mu = pm.math.exp(trees), observed = target)
            posterior = pm.sample(
                self.draws,
                tune = self.tune,
                chains = self.chains,
                cores = self.cores,
                random_seed = self.random_state,
                progressbar = False)
        operation = trees.owner.op
        self.trees_ = list(operation.all_trees)
        self._outputs = int(operation.n_outputs)
        self._samplers = None
        # pymc-bart sorts the shares from the largest, and names each by its
        # position (it does not sort labels that it is given).
        shares, labels = bart.get_variable_inclusion(posterior, features)
        importances = np.zeros(features.shape[1])
        importances[[int(label) for label in labels]] = shares
        self.feature_importances_ = importances
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Returns the parameters, as scikit-learn expects.

        Args:
            deep: not used.

        Returns:
            The parameters.

        """
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self) if field.init}

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Returns predictions (the predicted class of a classifier).

        Args:
            x: features.

        Returns:
            The predictions.

        """
        values = self._response(x)
        if self.is_classifier:
            positive = (values >= 0.5).astype(int)  # noqa: PLR2004
            return np.asarray(self.classes_[positive])
        return values

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the predicted probability of each class (two classes only).

        Args:
            x: features.

        Raises:
            AttributeError: if the model is not a classifier.

        Returns:
            One column for each class, in the order of `classes_`.

        """
        if not self.is_classifier:
            message = (
                'only BART with the "bernoulli" likelihood predicts '
                'probabilities')
            raise AttributeError(message)
        values = self._response(x)
        return np.column_stack([1 - values, values])

    def score(self, x: pd.DataFrame, y: pd.Series) -> float:
        """Returns the accuracy of a classifier or the R² of a regression.

        Args:
            x: features.
            y: labels.

        Returns:
            The score, as scikit-learn's classifiers and regressors score.

        """
        return _score(self.predict(x), y, classify = self.is_classifier)

    def set_params(self, **parameters: Any) -> BayesianAdditiveTrees:
        """Sets parameters, as scikit-learn expects.

        Args:
            **parameters: any of the parameters of the adapter.

        Returns:
            This adapter.

        """
        for key, value in parameters.items():
            setattr(self, key, value)
        return self

    """ Private Methods """

    def _response(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the posterior mean of the label's mean.

        Every posterior draw of the trees of every chain predicts each row,
        so the predictions do not depend on a random seed.

        Args:
            x: features.

        Returns:
            The mean of each row (the probability of the second class, for a
                classifier).

        """
        if self._samplers is None:
            module = utilities.import_tool('pymc_bart.pymc_bart')
            self._samplers = [
                module.PosteriorSampler.from_history(
                    batches, forest, self.m, self._outputs)
                for forest, batches in self.trees_ or []]
        features = np.ascontiguousarray(
            x[list(self.feature_names_in_)].to_numpy(dtype = float))
        draws = np.concatenate([
            sampler.sample_posterior(
                features, list(range(sampler.n_draws)), None)
            for sampler in self._samplers])
        # The draws are (draw, output, row), with one output.
        sums = draws[:, 0, :]
        link = _BART_LIKELIHOODS[self.likelihood]
        if link == 'expit':
            special = importlib.import_module('scipy.special')
            sums = special.expit(sums)
        elif link == 'exp':
            sums = np.exp(sums)
        return np.asarray(sums.mean(axis = 0), dtype = float)

    """ Dunder Methods """

    def __getstate__(self) -> dict[str, Any]:
        """Returns the state to pickle, without pymc-bart's samplers.

        Returns:
            The attributes, with the samplers left out (they are built
                again from `trees_` when they are needed).

        """
        return {**self.__dict__, '_samplers': None}

    def __sklearn_tags__(self) -> Any:
        """Returns scikit-learn's tags for a classifier or regressor.

        Returns:
            The tags (scikit-learn 1.6 and later).

        """
        return _estimator_tags(classify = self.is_classifier)


""" Private Functions """


def _by_class(frame: pd.DataFrame, labels: Sequence[str]) -> pd.Series:
    """Returns a table with a column for each class as one column.

    Args:
        frame: one row for each feature and one column for each class other
            than the reference, as statsmodels reports an `mnlogit`.
        labels: the names of those classes.

    Returns:
        The values, class by class, named "{feature} ({class})".

    """
    values: list[Any] = []
    names: list[str] = []
    for position, label in enumerate(labels):
        values.extend(frame.iloc[:, position])
        names.extend(f'{name} ({label})' for name in frame.index)
    return pd.Series(values, index = names)


def _collinear(exog: pd.DataFrame, rows: int = 250000) -> list[str]:
    """Returns the columns that are combinations of the columns before them.

    Each column is scaled to a length of 1 first, so that the scales of the
    columns do not matter. The columns are then factored by one QR
    decomposition, whose triangular factor holds how far each column is from
    the columns before it on its diagonal. A column is collinear if that is
    no more than the tolerance that NumPy's `matrix_rank` uses for all of the
    columns (from their singular values, which do not depend on rounding the
    way that inverting a matrix does), so the same columns are found on every
    computer. The factor is made a block of rows at a time (the factor so far
    and the next block are factored together), so that, unlike a singular
    value decomposition of each set of columns, it never copies every row.

    Args:
        exog: the exogenous variables, including any constant.
        rows: the number of rows in each block. Defaults to 250,000.

    Returns:
        The names of the collinear columns (none if there are missing values,
            which statsmodels reports itself).

    """
    values = exog.to_numpy(dtype = float)
    count, width = values.shape
    squares = np.zeros(width)
    for start in range(0, count, rows):
        block = values[start:start + rows]
        if np.isnan(block).any():
            return []
        squares += np.einsum('ij,ij->j', block, block)
    lengths = np.sqrt(squares)
    lengths[lengths == 0] = 1.0
    factor = np.zeros((0, width))
    for start in range(0, count, rows):
        block = values[start:start + rows] / lengths
        # With mode "r", NumPy returns only the factor, not a `QRResult`.
        factor = cast(
            'np.ndarray', np.linalg.qr(np.vstack([factor, block]), mode = 'r'))
    distances = np.zeros(width)
    diagonal = np.abs(np.diagonal(factor))
    distances[:len(diagonal)] = diagonal
    singular = np.linalg.svd(factor, compute_uv = False)
    largest = singular.max() if singular.size else 0.0
    tolerance = largest * max(count, width) * np.finfo(float).eps
    pairs = zip(exog.columns, distances, strict = True)
    return [str(name) for name, distance in pairs if distance <= tolerance]


def _estimator_tags(*, classify: bool) -> Any:
    """Returns scikit-learn's tags for a classifier or a regressor.

    An adapter that is not a scikit-learn estimator needs them for a search
    to choose its folds and scorer.

    Args:
        classify: whether the adapter is a classifier.

    Returns:
        The tags (scikit-learn 1.6 and later).

    """
    utils = importlib.import_module('sklearn.utils')
    return utils.Tags(
        estimator_type = 'classifier' if classify else 'regressor',
        target_tags = utils.TargetTags(required = True),
        classifier_tags = utils.ClassifierTags() if classify else None,
        regressor_tags = None if classify else utils.RegressorTags())


def _posterior_table(draws: dict[str, np.ndarray]) -> pd.DataFrame:
    """Returns a table of the posteriors of some parameters.

    Args:
        draws: the posterior draws of each parameter, by its name.

    Returns:
        One row for each parameter, with its "coefficient" (the posterior
            mean), "standard_error" (the posterior standard deviation),
            "statistic" (their ratio), "p_value" (missing, since a posterior
            has none), and "ci_lower" and "ci_upper" (the 95% equal-tailed
            credible interval), as in the tables of the other models.

    """
    columns = [
        'coefficient', 'standard_error', 'statistic', 'p_value', 'ci_lower',
        'ci_upper']
    rows = {}
    for name, values in draws.items():
        mean = float(np.mean(values))
        deviation = float(np.std(values, ddof = 1)) if values.size > 1 else (
            np.nan)
        lower, upper = np.quantile(values, [0.025, 0.975])
        rows[name] = [
            mean, deviation, mean / deviation if deviation else np.nan,
            np.nan, float(lower), float(upper)]
    return pd.DataFrame.from_dict(rows, orient = 'index', columns = columns)


def _score(predicted: np.ndarray, y: Any, *, classify: bool) -> float:
    """Returns the accuracy of a classifier or the R² of a regression.

    Args:
        predicted: the predictions.
        y: labels.
        classify: whether the predictions are classes.

    Returns:
        The score, as scikit-learn's classifiers and regressors score.

    """
    metrics = importlib.import_module('sklearn.metrics')
    if classify:
        return float(metrics.accuracy_score(y, predicted))
    return float(metrics.r2_score(y, predicted))


def _statsmodels_class(module: Any, name: str) -> type:
    """Returns the class in a statsmodels module named `name` in snake case.

    Wraps:
        - [`statsmodels.genmod.families`](https://www.statsmodels.org/stable/glm.html#families)
          from statsmodels, for a "family".
        - [`statsmodels.genmod.cov_struct`](https://www.statsmodels.org/stable/gee.html#dependence-structures)
          from statsmodels, for a "covariance".
        - [`statsmodels.robust.norms`](https://www.statsmodels.org/stable/rlm.html#norms)
          from statsmodels, for a "norm".

    Args:
        module: a statsmodels module (such as `statsmodels.api.families`).
        name: name of the class in snake case (such as "negative_binomial"
            for `NegativeBinomial`).

    Raises:
        ValueError: if there is no such class.

    Returns:
        The class.

    """
    title = ''.join(part.capitalize() for part in name.split('_'))
    try:
        kind: type = getattr(module, title)
    except AttributeError as error:
        message = f'{name!r} is not in {module.__name__}'
        raise ValueError(message) from error
    return kind


def _statistic(results: Any, name: str, like: Any) -> Any:
    """Returns a statistic of statsmodels results, or missing values.

    Args:
        results: fitted statsmodels results.
        name: name of the statistic (such as "bse").
        like: the coefficients, to copy the shape of.

    Returns:
        The statistic, or missing values in the shape of `like` if statsmodels
            could not find the covariance of the coefficients.

    """
    try:
        return getattr(results, name)
    except ValueError:
        return like * np.nan


def _two_classes(
    target: np.ndarray,
    model: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns the two classes of a label, and the label as 0 and 1.

    Args:
        target: labels.
        model: a name of the model, for the message of an error.

    Raises:
        ValueError: if the label does not have two classes.

    Returns:
        The classes, sorted, and 1 for each label that is the second class
            (and 0 for the first).

    """
    classes = np.unique(target)
    if len(classes) != 2:  # noqa: PLR2004
        message = f'{model} needs two classes'
        raise ValueError(message)
    return classes, (target == classes[1]).astype(int)
