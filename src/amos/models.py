"""Techniques that fit models and make predictions.

These are the techniques of the last step of the "analyst" stage. Each one
wraps a model from scikit-learn, xgboost, lightgbm, catboost, InterpretML,
TabPFN, statsmodels, or pyfixest. Most work for both tasks: `random_forest`,
for example, is a random forest classifier when the label is classified and a
random forest regressor when it is regressed. A model is
fitted to the training rows and predicts the test rows (or every row, if the
data has not been split). The fitted model, its predictions, and (for
classifiers) its predicted probabilities are stored in the dataset.

Every model accepts the parameters of the tool it wraps. It also accepts
"search" ("grid", "random", or "optuna"), which searches for the best values
of any parameter that is given as a list, using cross-validation on the
training rows ("cv" folds, 5 by default, and "n_iter" draws for a random or
Optuna search).

Contents:
    Model: base class for techniques that fit models.
    LabelCoded: lets a classifier that needs classes numbered 0 to n - 1 use
        any classes.
    Statsmodel: adapts statsmodels regressions to the scikit-learn interface.
    FixedEffects: adapts pyfixest regressions to the scikit-learn interface.
    ProportionalHazards: adapts statsmodels' Cox model to the scikit-learn
        interface.
    Adaboost, Baseline, BinomialBayesMixedglm, Catboost, Cox, DecisionTree,
        ElasticNet, ExplainableBoosting, ExtraTrees, Fixest, GEE,
        GeneralizedPoisson, GLM, GradientBoosting, KNN, Lasso, Lightgbm,
        Linear, Logit, LogitSM, Mixedlm, Mnlogit, NaiveBayes,
        NegativeBinomial, NeuralNetwork, OLS, OrdinalRegression, Poisson,
        Probit, QuantileRegression, RandomForest, Ridge, RobustRegression,
        SVM, Tabpfn, WLS, Xgboost, ZeroInflatedPoisson: models.

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from . import base, utilities

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
class Model(base.Operation, abc.ABC):
    """Base class for techniques that fit models and make predictions.

    `tools` maps each task that the model can do ("classify" and "regress")
    to the model class (or its import path) for that task. If `contents` is
    set, it is used for any task instead. A model class must have `fit` and
    `predict` methods, as in scikit-learn, and a classifier should have
    `predict_proba`.

    A `Model` can be used directly to wrap any such class:

    ```py
    amos.Model(name = "bayes_ridge", contents = "sklearn.linear_model.BayesianRidge")
    ```

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: the model class or its import path, used for every task.
            Defaults to `None`, in which case `tools` is used.
        parameters: keyword arguments for the model, and "search", "cv",
            "n_iter", and "scoring" for a hyperparameter search. Defaults to
            an empty `dict`.

    """

    # The model class (or its import path) for each task.
    tools: ClassVar[Mapping[str, Any]] = {}
    # Whether the model needs classes numbered 0 to n - 1 (as xgboost does).
    numbered_classes: ClassVar[bool] = False
    # Parameters that name columns (such as fixed effects or an event) that
    # the model is given along with the features.
    column_parameters: ClassVar[tuple[str, ...]] = ()
    # Whether the label must be counts: whole numbers of 0 or more.
    counts: ClassVar[bool] = False

    """ Public Methods """

    def implement(
        self,
        item: base.Dataset,
        search: str | None = None,
        cv: int = 5,
        n_iter: int = 10,
        scoring: str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Fits the model to the training rows and predicts the test rows.

        Args:
            item: the dataset to model. It must have a label.
            search: "grid" or "random" to search for the best values of any
                parameter given as a list. Defaults to `None`, which does not
                search (and passes lists to the model as they are).
            cv: number of cross-validation folds for a search. Defaults to 5.
            n_iter: number of combinations to draw for a random search.
                Defaults to 10.
            scoring: scikit-learn scorer for a search (such as "roc_auc").
                Defaults to `None`, which uses the model's `score` method.
            **kwargs: parameters for the model.

        Returns:
            The dataset, with the fitted `model`, `predictions`, and (for a
                classifier) `probabilities`.

        """
        task = self._check_task(item)
        extras = self._extra_columns(item, kwargs)
        self._check_features(item, extras)
        path = self._choose_tool(task)
        tool = utilities.import_tool(path)
        fixed, space = (
            (kwargs, {}) if search is None
            else utilities.search_space(kwargs, search))
        estimator = self._make_tool(item, fixed, tool = tool)
        space = utilities.accepted_parameters(tool, space)
        if (
            task == 'classify'
            and self.numbered_classes
            and not _is_numbered(item.classes)):
            estimator = LabelCoded(estimator = estimator)
            space = {f'estimator__{k}': v for k, v in space.items()}
        columns = [c for c in item.features if c not in extras] + extras
        y = item.y_train
        x = item.data.loc[y.index, columns]
        if space:
            estimator = self._search(
                item,
                estimator,
                x,
                y,
                space = space,
                search = search,
                cv = cv,
                n_iter = n_iter,
                scoring = scoring)
        else:
            estimator.fit(x, y)
        item.model = estimator
        self._predict(item, estimator, task, columns)
        if hasattr(estimator, 'coefficients'):
            item.tables[f'{self.name}_coefficients'] = estimator.coefficients()
        item.record(
            self.name,
            tool = utilities.describe_tool(path),
            task = task,
            parameters = utilities.parameters_of(estimator),
            search = search,
            rows = len(x))
        return item

    """ Private Methods """

    def _check_features(
        self,
        item: base.Dataset,
        extras: Sequence[str] = ()) -> None:
        """Checks that the model can use every feature of `item`.

        Models need numbers. Categorical columns (the `pandas` "category"
        type) are allowed because some models (such as lightgbm) use them
        directly.

        Args:
            item: the dataset to model.
            extras: columns named by `column_parameters`, which are not
                checked. Defaults to an empty tuple.

        Raises:
            ValueError: if a feature is a date or text.

        """
        unusable = []
        for column in item.features:
            if column in extras:
                continue
            values = item.data[column]
            kind = base._kind_of(values)
            if kind == 'dates' or (
                kind == 'categoricals'
                and not isinstance(values.dtype, pd.CategoricalDtype)):
                unusable.append(column)
        if unusable:
            message = (
                f'{self.name!r} cannot use the dates or text in {unusable}: '
                f'encode them (with an encoder such as one_hot) or remove '
                f'them (with drop_columns) first'
            )
            raise ValueError(message)

    def _check_task(self, item: base.Dataset) -> str:
        """Returns the task of `item`, checking that the model can do it.

        Args:
            item: the dataset to model.

        Raises:
            ValueError: if `item` has no task or the model cannot do it.

        Returns:
            "classify" or "regress".

        """
        task = item.task
        if task is None:
            message = f'{self.name!r} needs a dataset with a label'
            raise ValueError(message)
        if self.contents is None and task not in self.tools:
            tasks = ' or '.join(self.tools) or 'no task'
            message = (
                f'{self.name!r} can {tasks}, but the label is set to {task}'
            )
            if self.counts:
                # Whole numbers with few values are classified by default.
                message += (
                    ': set "task" to "regress" in the "general" section for '
                    'a label of counts'
                )
            raise ValueError(message)
        return task

    def _choose_tool(self, task: str) -> Any:
        """Returns the model class (or its import path) for `task`.

        Args:
            task: "classify" or "regress".

        Returns:
            `contents`, if it is set, or the tool in `tools` for `task`.

        """
        return self.contents if self.contents is not None else self.tools[task]

    def _extra_columns(
        self,
        item: base.Dataset,
        parameters: Mapping[str, Any]) -> list[str]:
        """Returns the columns named by the parameters in `column_parameters`.

        Args:
            item: the dataset being modeled.
            parameters: parameters for the model.

        Raises:
            KeyError: if a named column is not in the data.

        Returns:
            The names of the columns, without duplicates.

        """
        names: list[str] = []
        for parameter in self.column_parameters:
            for name in base._listify(parameters.get(parameter)):
                if name not in item.data.columns:
                    message = (
                        f'the {parameter} column {name!r} is not in the data')
                    raise KeyError(message)
                if name not in names:
                    names.append(name)
        return names

    def _predict(
        self,
        item: base.Dataset,
        estimator: Any,
        task: str,
        columns: Sequence[str]) -> None:
        """Stores the predictions (and probabilities) of `estimator`.

        Args:
            item: the dataset being modeled.
            estimator: the fitted model.
            task: "classify" or "regress".
            columns: the columns that the model was fitted to.

        """
        rows = item.y_test.index if item.is_split else item.data.index
        x = item.data.loc[rows, list(columns)]
        item.predictions = pd.Series(
            np.asarray(estimator.predict(x)),
            index = x.index,
            name = item.label)
        item.probabilities = None
        if task == 'classify' and hasattr(estimator, 'predict_proba'):
            classes = getattr(estimator, 'classes_', None)
            columns = list(item.classes if classes is None else classes)
            item.probabilities = pd.DataFrame(
                np.asarray(estimator.predict_proba(x)),
                index = x.index,
                columns = columns)

    def _search(
        self,
        item: base.Dataset,
        estimator: Any,
        x: pd.DataFrame,
        y: pd.Series,
        *,
        space: dict[str, Any],
        search: str | None,
        cv: int,
        n_iter: int,
        scoring: str | None) -> Any:
        """Searches for the best parameters and returns the best model.

        The results of every combination are stored in the dataset's `tables`
        as "{name}_search".

        Args:
            item: the dataset being modeled.
            estimator: the model, built with the fixed parameters.
            x: the training rows of the columns the model uses.
            y: the training labels.
            space: the parameters to search.
            search: "grid", "random", or "optuna".
            cv: number of cross-validation folds.
            n_iter: number of combinations to draw for a random or Optuna
                search.
            scoring: scikit-learn scorer, or `None`.

        Returns:
            The best model, refitted to every training row.

        """
        if search == 'optuna':
            return self._search_optuna(
                item, estimator, x, y,
                space = space, cv = cv, n_iter = n_iter, scoring = scoring)
        model_selection = importlib.import_module('sklearn.model_selection')
        if search == 'random':
            searcher = model_selection.RandomizedSearchCV(
                estimator,
                param_distributions = space,
                n_iter = n_iter,
                cv = cv,
                scoring = scoring,
                random_state = item.seed)
        else:
            searcher = model_selection.GridSearchCV(
                estimator, param_grid = space, cv = cv, scoring = scoring)
        searcher.fit(x, y)
        results = pd.DataFrame(searcher.cv_results_)
        columns = [c for c in results.columns if c.startswith('param_')]
        columns += ['mean_test_score', 'std_test_score', 'rank_test_score']
        item.tables[f'{self.name}_search'] = (
            results[columns].sort_values('rank_test_score'))
        return searcher.best_estimator_

    def _search_optuna(
        self,
        item: base.Dataset,
        estimator: Any,
        x: pd.DataFrame,
        y: pd.Series,
        *,
        space: dict[str, Any],
        cv: int,
        n_iter: int,
        scoring: str | None) -> Any:
        """Searches with Optuna and returns the best model.

        Optuna chooses each combination to try from the results of the ones
        before, so it usually finds good values in fewer tries than a random
        search. Its table has the same columns as the other searches.

        Args:
            item: the dataset being modeled.
            estimator: the model, built with the fixed parameters.
            x: the training rows of the columns the model uses.
            y: the training labels.
            space: Optuna distributions of the parameters to search.
            cv: number of cross-validation folds.
            n_iter: number of combinations to try.
            scoring: scikit-learn scorer, or `None`.

        Returns:
            The best model, refitted to every training row.

        """
        optuna = utilities.import_tool('optuna')
        integration = utilities.import_tool('optuna_integration')
        # Optuna reports every trial unless it is told to report only
        # warnings.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        searcher = integration.OptunaSearchCV(
            estimator,
            space,
            n_trials = n_iter,
            cv = cv,
            scoring = scoring,
            random_state = item.seed)
        searcher.fit(x, y)
        trials = searcher.trials_dataframe()
        results = pd.DataFrame({
            f'param_{c.removeprefix("params_")}': trials[c]
            for c in trials.columns if c.startswith('params_')})
        results['mean_test_score'] = trials['value']
        if 'user_attrs_std_test_score' in trials.columns:
            results['std_test_score'] = trials['user_attrs_std_test_score']
        results['rank_test_score'] = results['mean_test_score'].rank(
            ascending = False, method = 'min').astype(int)
        item.tables[f'{self.name}_search'] = results.sort_values(
            'rank_test_score')
        return searcher.best_estimator_


# `eq` is `False` so that wrappers are compared by identity, as scikit-learn
# estimators are.
@dataclasses.dataclass(eq = False)
class LabelCoded:
    """Lets a classifier that needs classes numbered 0 to n - 1 use any classes.

    xgboost, for example, needs classes numbered from 0. `LabelCoded` numbers
    the classes before fitting and turns predictions back into the original
    classes. Other attributes (such as `feature_importances_`) are those of
    the wrapped model.

    Args:
        estimator: the classifier to wrap.

    Attributes:
        classes_: the original classes, in sorted order.

    """

    estimator: Any
    classes_: Any = dataclasses.field(
        default = None, init = False, repr = False)

    # Marks the wrapper as a classifier for older versions of scikit-learn.
    _estimator_type: ClassVar[str] = 'classifier'

    """ Public Methods """

    def fit(self, x: Any, y: Any) -> LabelCoded:
        """Fits the wrapped classifier to `x` and the numbered classes of `y`.

        Args:
            x: features.
            y: labels.

        Returns:
            This wrapper.

        """
        self.classes_, codes = np.unique(np.asarray(y), return_inverse = True)
        self.estimator.fit(x, codes)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: FBT002
        """Returns the parameters, as scikit-learn expects.

        Args:
            deep: whether to include the parameters of the wrapped classifier
                (as "estimator__{name}"). Defaults to `True`.

        Returns:
            The parameters.

        """
        parameters: dict[str, Any] = {'estimator': self.estimator}
        if deep and hasattr(self.estimator, 'get_params'):
            for key, value in self.estimator.get_params(deep = True).items():
                parameters[f'estimator__{key}'] = value
        return parameters

    def predict(self, x: Any) -> np.ndarray:
        """Returns the predicted classes, in the original classes.

        Args:
            x: features.

        Returns:
            The predicted classes.

        """
        codes = np.asarray(self.estimator.predict(x)).astype(int)
        return np.asarray(self.classes_[codes])

    def predict_proba(self, x: Any) -> np.ndarray:
        """Returns the predicted probability of each class.

        Args:
            x: features.

        Returns:
            One column for each class, in the order of `classes_`.

        """
        return np.asarray(self.estimator.predict_proba(x))

    def score(self, x: Any, y: Any) -> float:
        """Returns the share of rows that are classified correctly.

        Args:
            x: features.
            y: labels.

        Returns:
            The accuracy.

        """
        return float(np.mean(self.predict(x) == np.asarray(y)))

    def set_params(self, **parameters: Any) -> LabelCoded:
        """Sets parameters, as scikit-learn expects.

        Args:
            **parameters: "estimator" or "estimator__{name}" parameters.

        Returns:
            This wrapper.

        """
        inner = {}
        for key, value in parameters.items():
            if key == 'estimator':
                self.estimator = value
            else:
                inner[key.removeprefix('estimator__')] = value
        if inner:
            self.estimator.set_params(**inner)
        return self

    """ Dunder Methods """

    def __getattr__(self, attribute: str) -> Any:
        """Returns an attribute of the wrapped classifier.

        Args:
            attribute: name of the attribute.

        Raises:
            AttributeError: if the wrapped classifier does not have it.

        Returns:
            The attribute of the wrapped classifier.

        """
        # Private attributes and the classifier itself are not passed on, which
        # avoids endless recursion while the wrapper is copied.
        if attribute.startswith('_') or attribute == 'estimator':
            raise AttributeError(attribute)
        return getattr(self.estimator, attribute)

    def __sklearn_tags__(self) -> Any:
        """Returns the scikit-learn tags of the wrapped classifier."""
        return self.estimator.__sklearn_tags__()


@dataclasses.dataclass
class Statsmodel:
    """Adapts a statsmodels regression to the scikit-learn interface.

    statsmodels reports the standard errors, test statistics, p-values, and
    confidence intervals that academic research usually needs. The
    `coefficients` method returns them as a table. `kind` chooses the model:

    | Kind | statsmodels model | Predicts |
    | --- | --- | --- |
    | "ols" | `OLS` | Values. |
    | "wls" | `WLS` | Values, from rows with `weights`. |
    | "glm" | `GLM` | Values, or a class (for the "binomial" `family`). |
    | "gee" | `GEE` | The same as "glm", for rows correlated within `groups`. |
    | "mixedlm" | `MixedLM` | Values, with a random intercept for each of `groups`. |
    | "quantile_regression" | `QuantReg` | A `quantile` of the label. |
    | "robust_regression" | `RLM` | Values, with less weight on outliers. |
    | "poisson" | `Poisson` | Counts. |
    | "negative_binomial" | `NegativeBinomial` | Counts that vary more than a Poisson's. |
    | "generalized_poisson" | `GeneralizedPoisson` | Counts that vary more or less than a Poisson's. |
    | "zero_inflated_poisson" | `ZeroInflatedPoisson` | Counts with more zeros than a Poisson's. |
    | "logit" | `Logit` | One of two classes. |
    | "probit" | `Probit` | One of two classes. |
    | "binomial_bayes_mixedglm" | `BinomialBayesMixedGLM` | One of two classes, with a random intercept for each of `groups`. |
    | "mnlogit" | `MNLogit` | One of any number of classes. |
    | "ordinal_regression" | `OrderedModel` | One of ordered classes. |

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

    Attributes:
        results: the fitted statsmodels results.
        classes_: the classes of a classifier, sorted.
        levels_: the classes of an "ordinal_regression", in order: the order
            of the categories of a label that is an ordered categorical, and
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

        An "mnlogit" has a row of coefficients for each class after the
        first. A feature that was left out (see `dropped_`) has a missing
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
                "{feature} ({class})". A "binomial_bayes_mixedglm" reports the
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
            deviations = pd.Series(np.asarray(results.fe_sd), index = means.index)
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
                labels = [str(c) for c in self.classes_[1:]]
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
        deep: bool = True) -> dict[str, Any]:  # noqa: ARG002, FBT002
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
                classes, or a model of counts is given other values.

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
                message = f'a {self._title} model needs a label with two classes'
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
                groups = pd.get_dummies(self._column(x, 'groups'), dtype = float)
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

    """

    fixed_effects: Sequence[str] | str | None = None
    cluster: str | None = None
    family: str = 'gaussian'
    results: Any = dataclasses.field(default = None, init = False, repr = False)
    classes_: Any = dataclasses.field(
        default = None, init = False, repr = False)
    feature_names_in_: Any = dataclasses.field(
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
        extras = [*self._effects(), *base._listify(self.cluster)]
        self.feature_names_in_ = np.asarray(
            [c for c in x.columns if c not in extras])
        target = np.asarray(y)
        if self.is_binary:
            self.classes_ = np.unique(target)
            if len(self.classes_) != 2:  # noqa: PLR2004
                message = f'a {self.family} model needs two classes'
                raise ValueError(message)
            target = (target == self.classes_[1]).astype(float)
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
        deep: bool = True) -> dict[str, Any]:  # noqa: ARG002, FBT002
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

        Rows with a fixed effect level that was not in the training rows
        cannot be predicted and are missing.

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
        """Returns the names of the fixed effect columns."""
        return base._listify(self.fixed_effects)

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

    def _response(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the model's predicted mean (a probability, for two classes).

        Args:
            x: features and fixed effect columns.

        Returns:
            The predictions on the scale of the label.

        """
        data = self._frame(x)
        with utilities.preserved_logging():
            if self.family == 'gaussian':
                return np.asarray(self.results.predict(newdata = data))
            return np.asarray(
                self.results.predict(newdata = data, type = 'response'))

    def _safe_names(self) -> list[str]:
        """Returns the names of the features in pyfixest's formulas."""
        return [f'amos_x{n}' for n in range(len(self.feature_names_in_))]


@dataclasses.dataclass
class ProportionalHazards:
    """Adapts statsmodels' Cox regression to the scikit-learn interface.

    The label is the time until an event (such as rearrest) or until the row
    stopped being observed (censoring). Rows with the same time are handled
    with Efron's method. The model predicts the expected time until the
    event. The `coefficients` method returns the coefficients, hazard ratios,
    and their statistics as a table.

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
        deep: bool = True) -> dict[str, Any]:  # noqa: ARG002, FBT002
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
        """Returns the columns of `x` that are features (not the event)."""
        return x.drop(columns = [c for c in [self.event] if c in x.columns])


""" Models """


class _Statsmodels:
    """Makes a model one kind of `Statsmodel` (see its table of kinds).

    It is a plain class rather than a `Model`, so that the library does not
    store it as a technique.

    """

    # The kind of `Statsmodel`.
    kind: ClassVar[str] = 'ols'

    def _prepare(
        self,
        item: base.Dataset,  # noqa: ARG002
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Makes the statsmodels adapter the model's kind.

        Args:
            item: the dataset to model.
            parameters: parameters for the model.

        Returns:
            The parameters, with "kind".

        """
        return {**parameters, 'kind': self.kind}


@dataclasses.dataclass
class Adaboost(Model):
    """AdaBoost: a sequence of small trees, each fixing the last one's errors."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.ensemble.AdaBoostClassifier',
        'regress': 'sklearn.ensemble.AdaBoostRegressor'}


@dataclasses.dataclass
class Baseline(Model):
    """Predicts the most common class (or the mean) for every row.

    Every model should beat this. It is the standard comparison for showing
    that a model learned something from the features.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.dummy.DummyClassifier',
        'regress': 'sklearn.dummy.DummyRegressor'}


@dataclasses.dataclass
class BinomialBayesMixedglm(_Statsmodels, Model):
    """A Bayesian logistic regression with a random intercept for each group.

    statsmodels' `BinomialBayesMixedGLM`, fitted by variational Bayes, for a
    label of two classes and rows that are correlated within groups (such as
    the decisions of each judge). Set "groups" to the column of groups (by
    default, the dataset's first group). Its table of coefficients has the
    mean and standard deviation of each coefficient's posterior, and its 95%
    credible interval. Predictions use the fixed effects only, so they suit
    groups the model has not seen.

    """

    tools: ClassVar[Mapping[str, Any]] = {'classify': Statsmodel}
    column_parameters: ClassVar[tuple[str, ...]] = ('groups',)
    kind: ClassVar[str] = 'binomial_bayes_mixedglm'

    def implement(
        self,
        item: base.Dataset,
        search: str | None = None,
        cv: int = 5,
        n_iter: int = 10,
        scoring: str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Fits the model, with the dataset's first group as "groups".

        Args:
            item: the dataset to model.
            search: as for `Model.implement`.
            cv: as for `Model.implement`.
            n_iter: as for `Model.implement`.
            scoring: as for `Model.implement`.
            **kwargs: parameters for the model.

        Returns:
            The dataset, with the fitted model and its predictions.

        """
        return super().implement(
            item,
            search = search,
            cv = cv,
            n_iter = n_iter,
            scoring = scoring,
            **_with_groups(item, kwargs))


@dataclasses.dataclass
class Catboost(Model):
    """CatBoost gradient boosting, which uses categorical features directly.

    Categorical features (the `pandas` "category" type) are passed to it as
    categories, so they need no encoder.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'catboost.CatBoostClassifier',
        'regress': 'catboost.CatBoostRegressor'}
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {
            'verbose': 0,
            # Otherwise, catboost writes files to the current folder.
            'allow_writing_files': False})

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Names the categorical features for catboost.

        Args:
            item: the dataset to model.
            parameters: parameters for the model.

        Returns:
            The parameters, with "cat_features" if there are categorical
                features.

        """
        if item.categoricals:
            parameters.setdefault('cat_features', item.categoricals)
        return parameters


@dataclasses.dataclass
class Cox(Model):
    """Cox proportional hazards regression of the time until an event.

    The label is the time until the event (such as rearrest) or until the row
    stopped being observed. Set "event" to the column that is 1 if the event
    happened and 0 if the row was censored (it is not used as a feature).
    The coefficients, hazard ratios, standard errors, p-values, and
    confidence intervals are stored in the dataset's `tables` as
    "{name}_coefficients". The predictions are the expected times until the
    event.

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': ProportionalHazards}
    column_parameters: ClassVar[tuple[str, ...]] = ('event',)


@dataclasses.dataclass
class DecisionTree(Model):
    """A single decision tree."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.tree.DecisionTreeClassifier',
        'regress': 'sklearn.tree.DecisionTreeRegressor'}


@dataclasses.dataclass
class ElasticNet(Model):
    """Linear regression with both lasso and ridge penalties."""

    tools: ClassVar[Mapping[str, Any]] = {
        'regress': 'sklearn.linear_model.ElasticNet'}


@dataclasses.dataclass
class ExplainableBoosting(Model):
    """An Explainable Boosting Machine from InterpretML.

    It is nearly as accurate as gradient boosting, but it is a sum of one
    function of each feature (and of a few pairs), so the effect of every
    feature can be shown exactly. `feature_importance` reports the importance
    of each term.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'interpret.glassbox.ExplainableBoostingClassifier',
        'regress': 'interpret.glassbox.ExplainableBoostingRegressor'}


@dataclasses.dataclass
class ExtraTrees(Model):
    """An ensemble of extremely randomized trees."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.ensemble.ExtraTreesClassifier',
        'regress': 'sklearn.ensemble.ExtraTreesRegressor'}


@dataclasses.dataclass
class Fixest(Model):
    """Regression with fixed effects and clustered standard errors (pyfixest).

    Set "fixed_effects" to the columns whose levels each get their own
    intercept (such as "judge" or "year") and "cluster" to the column to
    cluster the standard errors by (such as "court"). Those columns are
    usually the dataset's `groups`; they are not used as features. The label
    is regressed by least squares, or by a logit model if it is classified.
    Set "family" to "poisson" for counts or "probit" for a probit model. The
    coefficients, standard errors, p-values, and confidence intervals are
    stored in the dataset's `tables` as "{name}_coefficients".

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': FixedEffects,
        'regress': FixedEffects}
    column_parameters: ClassVar[tuple[str, ...]] = ('fixed_effects', 'cluster')

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the family for the task.

        Args:
            item: the dataset to model.
            parameters: parameters for the model.

        Returns:
            The parameters, with a "family".

        """
        family = 'logit' if item.task == 'classify' else 'gaussian'
        return {'family': family, **parameters}


@dataclasses.dataclass
class GEE(Model):
    """Generalized estimating equations from statsmodels, with inference.

    A `glm` for rows that are correlated within groups (such as several
    cases of one court): the coefficients are averages over the groups, and
    their standard errors allow for the correlation. Set "groups" to the
    column of groups (by default, the dataset's first group), "family" as
    for a `glm`, and "covariance" to "exchangeable" to model the correlation
    (by default, "independence").

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': Statsmodel,
        'regress': Statsmodel}
    column_parameters: ClassVar[tuple[str, ...]] = ('groups',)

    def implement(
        self,
        item: base.Dataset,
        search: str | None = None,
        cv: int = 5,
        n_iter: int = 10,
        scoring: str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Fits the model, with the dataset's first group as "groups".

        Args:
            item: the dataset to model.
            search: as for `Model.implement`.
            cv: as for `Model.implement`.
            n_iter: as for `Model.implement`.
            scoring: as for `Model.implement`.
            **kwargs: parameters for the model.

        Returns:
            The dataset, with the fitted model and its predictions.

        """
        return super().implement(
            item,
            search = search,
            cv = cv,
            n_iter = n_iter,
            scoring = scoring,
            **_with_groups(item, kwargs))

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the family for the task.

        Args:
            item: the dataset to model.
            parameters: parameters for the model.

        Returns:
            The parameters, with "kind" and "family".

        """
        family = _family(self.name, item, parameters)
        return {**parameters, 'family': family, 'kind': 'gee'}


@dataclasses.dataclass
class GeneralizedPoisson(_Statsmodels, Model):
    """Generalized Poisson regression of counts, with inference.

    Like `poisson`, for counts that vary more (or less) than a Poisson
    distribution's.

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'generalized_poisson'
    counts: ClassVar[bool] = True


@dataclasses.dataclass
class GLM(Model):
    """A generalized linear model from statsmodels, with inference.

    By default, the family is "binomial" (logistic regression without a
    penalty) for classification and "gaussian" for regression. Set "family"
    to use another (such as "poisson" for counts). The coefficients, standard
    errors, p-values, and confidence intervals are stored in the dataset's
    `tables` as "{name}_coefficients".

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': Statsmodel,
        'regress': Statsmodel}

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the family for the task.

        Args:
            item: the dataset to model.
            parameters: parameters for the model.

        Returns:
            The parameters, with "kind" and "family".

        """
        family = _family(self.name, item, parameters)
        return {**parameters, 'family': family, 'kind': 'glm'}


@dataclasses.dataclass
class GradientBoosting(Model):
    """Histogram-based gradient boosting from scikit-learn.

    It is fast on large data and handles missing values itself.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.ensemble.HistGradientBoostingClassifier',
        'regress': 'sklearn.ensemble.HistGradientBoostingRegressor'}


@dataclasses.dataclass
class KNN(Model):
    """Predicts from the k nearest training rows (5 by default)."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.neighbors.KNeighborsClassifier',
        'regress': 'sklearn.neighbors.KNeighborsRegressor'}


@dataclasses.dataclass
class Lasso(Model):
    """Linear regression with a lasso (L1) penalty."""

    tools: ClassVar[Mapping[str, Any]] = {
        'regress': 'sklearn.linear_model.Lasso'}


@dataclasses.dataclass
class Lightgbm(Model):
    """LightGBM gradient boosting."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'lightgbm.LGBMClassifier',
        'regress': 'lightgbm.LGBMRegressor'}
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'verbose': -1})


@dataclasses.dataclass
class Linear(Model):
    """Ordinary least squares regression from scikit-learn."""

    tools: ClassVar[Mapping[str, Any]] = {
        'regress': 'sklearn.linear_model.LinearRegression'}


@dataclasses.dataclass
class Logit(Model):
    """Logistic regression from scikit-learn.

    scikit-learn adds a ridge (L2) penalty by default. Use `logit_sm` for
    statsmodels' logistic regression, without a penalty and with p-values.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.linear_model.LogisticRegression'}
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'max_iter': 1000})


@dataclasses.dataclass
class LogitSM(_Statsmodels, Model):
    """Logistic regression from statsmodels, with inference.

    Unlike `logit` (scikit-learn's logistic regression), it has no penalty,
    so its coefficients have standard errors, p-values, and confidence
    intervals (as does `glm` for a label of two classes).

    """

    tools: ClassVar[Mapping[str, Any]] = {'classify': Statsmodel}
    kind: ClassVar[str] = 'logit'


@dataclasses.dataclass
class Mixedlm(_Statsmodels, Model):
    """A linear mixed model from statsmodels, with inference.

    A linear regression with a random intercept for each group (such as each
    judge), for rows that are correlated within groups. Set "groups" to the
    column of groups (by default, the dataset's first group). Predictions use
    the fixed effects only, so they suit groups the model has not seen.

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    column_parameters: ClassVar[tuple[str, ...]] = ('groups',)
    kind: ClassVar[str] = 'mixedlm'

    def implement(
        self,
        item: base.Dataset,
        search: str | None = None,
        cv: int = 5,
        n_iter: int = 10,
        scoring: str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Fits the model, with the dataset's first group as "groups".

        Args:
            item: the dataset to model.
            search: as for `Model.implement`.
            cv: as for `Model.implement`.
            n_iter: as for `Model.implement`.
            scoring: as for `Model.implement`.
            **kwargs: parameters for the model.

        Returns:
            The dataset, with the fitted model and its predictions.

        """
        return super().implement(
            item,
            search = search,
            cv = cv,
            n_iter = n_iter,
            scoring = scoring,
            **_with_groups(item, kwargs))


@dataclasses.dataclass
class Mnlogit(_Statsmodels, Model):
    """Multinomial logistic regression from statsmodels, with inference.

    For a label of any number of classes. Each class after the first has its
    own coefficients, compared with the first class.

    """

    tools: ClassVar[Mapping[str, Any]] = {'classify': Statsmodel}
    kind: ClassVar[str] = 'mnlogit'


@dataclasses.dataclass
class NaiveBayes(Model):
    """Gaussian naive Bayes."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.naive_bayes.GaussianNB'}


@dataclasses.dataclass
class NegativeBinomial(_Statsmodels, Model):
    """Negative binomial regression of counts, with inference.

    Like `poisson`, for counts that vary more than a Poisson distribution's
    (which is common). The extra variance ("alpha") is estimated.

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'negative_binomial'
    counts: ClassVar[bool] = True


@dataclasses.dataclass
class NeuralNetwork(Model):
    """A multi-layer perceptron (a simple neural network)."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.neural_network.MLPClassifier',
        'regress': 'sklearn.neural_network.MLPRegressor'}
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'max_iter': 1000})


@dataclasses.dataclass
class OLS(_Statsmodels, Model):
    """Ordinary least squares regression from statsmodels, with inference.

    The coefficients, standard errors, p-values, and confidence intervals are
    stored in the dataset's `tables` as "{name}_coefficients".

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'ols'


@dataclasses.dataclass
class OrdinalRegression(_Statsmodels, Model):
    """Ordinal regression (ordered logit or probit) from statsmodels.

    For a label of ordered classes (such as a sentence that is lenient,
    typical, or harsh). The classes are ordered as the categories of a label
    that is an ordered categorical, and otherwise sorted. Set "distribution"
    to "probit" for an ordered probit.

    """

    tools: ClassVar[Mapping[str, Any]] = {'classify': Statsmodel}
    kind: ClassVar[str] = 'ordinal_regression'


@dataclasses.dataclass
class Poisson(_Statsmodels, Model):
    """Poisson regression of counts (such as the number of arrests).

    The label must be whole numbers of 0 or more. amos classifies whole
    numbers with few values, so set "task" to "regress" for such counts.

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'poisson'
    counts: ClassVar[bool] = True


@dataclasses.dataclass
class Probit(_Statsmodels, Model):
    """Probit regression of two classes from statsmodels, with inference."""

    tools: ClassVar[Mapping[str, Any]] = {'classify': Statsmodel}
    kind: ClassVar[str] = 'probit'


@dataclasses.dataclass
class QuantileRegression(_Statsmodels, Model):
    """Quantile regression from statsmodels, with inference.

    Predicts a quantile of the label ("quantile", by default the median)
    rather than its mean, which outliers sway less.

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'quantile_regression'


@dataclasses.dataclass
class RandomForest(Model):
    """A random forest."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.ensemble.RandomForestClassifier',
        'regress': 'sklearn.ensemble.RandomForestRegressor'}


@dataclasses.dataclass
class Ridge(Model):
    """Linear regression with a ridge (L2) penalty."""

    tools: ClassVar[Mapping[str, Any]] = {
        'regress': 'sklearn.linear_model.Ridge'}


@dataclasses.dataclass
class RobustRegression(_Statsmodels, Model):
    """Robust linear regression from statsmodels, with inference.

    Gives less weight to rows with large errors (outliers), by "norm"
    (Huber's, by default).

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'robust_regression'


@dataclasses.dataclass
class SVM(Model):
    """A support vector machine (with probabilities for classification)."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.svm.SVC',
        'regress': 'sklearn.svm.SVR'}
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'probability': True})


@dataclasses.dataclass
class Tabpfn(Model):
    """TabPFN, a pretrained model that is often the most accurate on small data.

    TabPFN is a neural network trained in advance on millions of synthetic
    tables, so it is not trained on your data in the usual way. It works best
    with up to about 10,000 rows and 500 features. It needs PyTorch (install
    it with "pip install amos[tabpfn]"), downloads its weights the first time
    it is used, and runs much faster with a GPU. Check the license of its
    weights before using it.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'tabpfn.TabPFNClassifier',
        'regress': 'tabpfn.TabPFNRegressor'}


@dataclasses.dataclass
class WLS(_Statsmodels, Model):
    """Weighted least squares regression from statsmodels, with inference.

    Set "weights" to the column with the weight of each row (such as the
    inverse of its variance). The weights are not a feature.

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    column_parameters: ClassVar[tuple[str, ...]] = ('weights',)
    kind: ClassVar[str] = 'wls'


@dataclasses.dataclass
class Xgboost(Model):
    """XGBoost gradient boosting."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'xgboost.XGBClassifier',
        'regress': 'xgboost.XGBRegressor'}
    numbered_classes: ClassVar[bool] = True


@dataclasses.dataclass
class ZeroInflatedPoisson(_Statsmodels, Model):
    """Zero-inflated Poisson regression of counts, with inference.

    Like `poisson`, for counts with more zeros than a Poisson distribution
    has (such as rows that could not have had any events).

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'zero_inflated_poisson'
    counts: ClassVar[bool] = True


""" Private Functions """


def _is_numbered(classes: list[Any]) -> bool:
    """Returns whether `classes` are the integers 0 to n - 1.

    Args:
        classes: the sorted classes of a label.

    Returns:
        Whether the classes are numbered from 0 without gaps.

    """
    try:
        return [int(c) for c in classes] == list(range(len(classes))) and all(
            float(c) == int(c) for c in classes)
    except (TypeError, ValueError):
        return False


def _by_class(frame: pd.DataFrame, labels: Sequence[str]) -> pd.Series:
    """Returns a table with a column for each class as one column.

    Args:
        frame: one row for each feature and one column for each class after
            the first, as statsmodels reports an `mnlogit`.
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


def _family(name: str, item: base.Dataset, parameters: Mapping[str, Any]) -> str:
    """Returns the family of a `glm` or `gee`, checking that it suits the task.

    Args:
        name: name of the model, for the message of an error.
        item: the dataset to model.
        parameters: parameters for the model, which may name a "family".

    Raises:
        ValueError: if the label is classified and the family is not
            "binomial".

    Returns:
        The "family" in `parameters`, or else "binomial" for classification
            and "gaussian" for regression.

    """
    classify = item.task == 'classify'
    family = str(parameters.get(
        'family', 'binomial' if classify else 'gaussian'))
    if classify and family != 'binomial':
        message = (
            f'{name!r} with the {family!r} family regresses, but the label is '
            f'set to classify: use the "binomial" family, or set "task" to '
            f'"regress" in the "general" section'
        )
        raise ValueError(message)
    return family


def _statsmodels_class(module: Any, name: str) -> type:
    """Returns the class in a statsmodels module named `name` in snake case.

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


def _with_groups(
    item: base.Dataset,
    parameters: dict[str, Any]) -> dict[str, Any]:
    """Returns `parameters` with the dataset's first group as "groups".

    Args:
        item: the dataset to model.
        parameters: parameters for the model.

    Returns:
        The parameters, with "groups" if they did not name it and the dataset
            has groups.

    """
    if parameters.get('groups') is None and item.groups:
        return {**parameters, 'groups': item.groups[0]}
    return parameters
