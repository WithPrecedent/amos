"""Adapters that give statistical packages the scikit-learn interface.

Contents:
    Statsmodel: adapts statsmodels regressions to the scikit-learn interface.
    FixedEffects: adapts pyfixest regressions to the scikit-learn interface.
    ProportionalHazards: adapts statsmodels' Cox model to the scikit-learn
        interface.

"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable, Sequence
from typing import Any

import camina
import numpy as np
import pandas as pd

from .. import utilities

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
        # likelihood, which prints its progress unless `disp` is 0.
        model = {
            'generalized_poisson': api.GeneralizedPoisson,
            'logit': api.Logit,
            'mnlogit': api.MNLogit,
            'negative_binomial': api.NegativeBinomial,
            'poisson': api.Poisson,
            'probit': api.Probit,
            'zero_inflated_poisson': api.ZeroInflatedPoisson}[self.kind](
                endog, exog)
        try:
            return model.fit(disp = 0)
        except np.linalg.LinAlgError:
            # Newton's method fails if a feature is a combination of others.
            # BFGS still finds coefficients, though statsmodels then warns
            # that it cannot find their standard errors.
            return model.fit(method = 'bfgs', maxiter = 1000, disp = 0)

    def _exog(self, x: pd.DataFrame) -> pd.DataFrame:
        """Returns the features as floats, with a constant if one is added.

        Args:
            x: features, and any other columns, which are left out.

        Returns:
            The exogenous variables for statsmodels, without the features in
                `dropped_`.

        """
        exog = x[list(self.feature_names_in_)].astype(float)
        exog.columns = [str(c) for c in exog.columns]
        if self.constant and self.kind != 'ordinal_regression':
            exog.insert(0, 'const', 1.0)
        return exog.drop(columns = self.dropped_)

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
    each judge or year) quickly, with standard errors clustered by any
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

    The label is the time until an event (such as rearrest) or until the row
    stopped being observed (censoring). Rows with the same time are handled
    with Efron's method. The model predicts the expected time until the
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


def _collinear(exog: pd.DataFrame) -> list[str]:
    """Returns the columns that are combinations of the columns before them.

    The rank of the columns is found by singular values, which do not depend
    on rounding the way that inverting a matrix does, so the same columns are
    found on every computer. Each column is scaled to a length of 1 first, so
    that the scales of the columns do not matter.

    Args:
        exog: the exogenous variables, including any constant.

    Returns:
        The names of the collinear columns (none if there are missing values,
            which statsmodels reports itself).

    """
    values = exog.to_numpy(dtype = float)
    if np.isnan(values).any():
        return []
    lengths = np.linalg.norm(values, axis = 0)
    values = values / np.where(lengths > 0, lengths, 1.0)
    kept: list[int] = []
    collinear: list[str] = []
    for position, name in enumerate(exog.columns):
        if np.linalg.matrix_rank(values[:, [*kept, position]]) > len(kept):
            kept.append(position)
        else:
            collinear.append(str(name))
    return collinear


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
