"""Techniques that fit models and make predictions.

These are the techniques of the last step of the "analyst" stage. Each one
wraps a model from scikit-learn, xgboost, lightgbm, or statsmodels. Most work
for both tasks: `random_forest`, for example, is a random forest classifier
when the label is classified and a random forest regressor when it is
regressed. A model is fitted to the training rows and predicts the test rows
(or every row, if the data has not been split). The fitted model, its
predictions, and (for classifiers) its predicted probabilities are stored in
the dataset.

Every model accepts the parameters of the tool it wraps. It also accepts
"search" ("grid" or "random"), which searches for the best values of any
parameter that is given as a list, using cross-validation on the training
rows ("cv" folds, 5 by default, and "n_iter" draws for a random search).

Contents:
    Model: base class for techniques that fit models.
    LabelCoded: lets a classifier that needs classes numbered 0 to n - 1 use
        any classes.
    Statsmodel: adapts statsmodels regressions to the scikit-learn interface.
    Adaboost, Baseline, DecisionTree, ElasticNet, ExtraTrees, GLM,
        GradientBoosting, KNN, Lasso, Lightgbm, Linear, Logit, NaiveBayes,
        NeuralNetwork, OLS, RandomForest, Ridge, SVM, Xgboost: models.

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from . import base, utilities


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
        self._check_features(item)
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
        x, y = item.x_train, item.y_train
        if space:
            estimator = self._search(
                item,
                estimator,
                space = space,
                search = search,
                cv = cv,
                n_iter = n_iter,
                scoring = scoring)
        else:
            estimator.fit(x, y)
        item.model = estimator
        self._predict(item, estimator, task)
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

    def _check_features(self, item: base.Dataset) -> None:
        """Checks that the model can use every feature of `item`.

        Models need numbers. Categorical columns (the `pandas` "category"
        type) are allowed because some models (such as lightgbm) use them
        directly.

        Args:
            item: the dataset to model.

        Raises:
            ValueError: if a feature is a date or text.

        """
        unusable = []
        for column in item.features:
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

    def _predict(self, item: base.Dataset, estimator: Any, task: str) -> None:
        """Stores the predictions (and probabilities) of `estimator`.

        Args:
            item: the dataset being modeled.
            estimator: the fitted model.
            task: "classify" or "regress".

        """
        x = item.x_test if item.is_split else item.x
        item.predictions = pd.Series(
            np.asarray(estimator.predict(x)), index = x.index, name = item.label)
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
            space: the parameters to search.
            search: "grid" or "random".
            cv: number of cross-validation folds.
            n_iter: number of combinations to draw for a random search.
            scoring: scikit-learn scorer, or `None`.

        Returns:
            The best model, refitted to every training row.

        """
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
        searcher.fit(item.x_train, item.y_train)
        results = pd.DataFrame(searcher.cv_results_)
        columns = [c for c in results.columns if c.startswith('param_')]
        columns += ['mean_test_score', 'std_test_score', 'rank_test_score']
        item.tables[f'{self.name}_search'] = (
            results[columns].sort_values('rank_test_score'))
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
    classes_: Any = dataclasses.field(default = None, init = False, repr = False)

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
    `coefficients` method returns them as a table.

    Args:
        kind: "ols" (ordinary least squares) or "glm" (a generalized linear
            model). Defaults to "ols".
        family: family of a "glm", by the name of a class in
            `statsmodels.api.families` in snake case ("binomial", "gaussian",
            "poisson", "negative_binomial", "gamma", and so on). A "binomial"
            model is a classifier of two classes. Defaults to "gaussian".
        constant: whether to add a constant (intercept). Defaults to `True`.

    Attributes:
        results: the fitted statsmodels results.
        classes_: the two classes of a "binomial" model.

    """

    kind: str = 'ols'
    family: str = 'gaussian'
    constant: bool = True
    results: Any = dataclasses.field(default = None, init = False, repr = False)
    classes_: Any = dataclasses.field(default = None, init = False, repr = False)

    """ Properties """

    @property
    def coef_(self) -> np.ndarray:
        """Returns the coefficients of the features (not of the constant)."""
        params = self.results.params
        return np.asarray(params.drop('const', errors = 'ignore'))

    @property
    def is_binomial(self) -> bool:
        """Returns whether the model is a binomial (two-class) classifier."""
        return self.kind == 'glm' and self.family == 'binomial'

    """ Public Methods """

    def coefficients(self) -> pd.DataFrame:
        """Returns the coefficients and their statistics.

        Returns:
            One row for each coefficient, with its "coefficient",
                "standard_error", "statistic", "p_value", "ci_lower", and
                "ci_upper".

        """
        results = self.results
        intervals = results.conf_int()
        return pd.DataFrame({
            'coefficient': results.params,
            'standard_error': results.bse,
            'statistic': results.tvalues,
            'p_value': results.pvalues,
            'ci_lower': intervals.iloc[:, 0],
            'ci_upper': intervals.iloc[:, 1]})

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Statsmodel:
        """Fits the model.

        Args:
            x: features.
            y: labels.

        Raises:
            ValueError: if a binomial model's label does not have two classes.

        Returns:
            This adapter.

        """
        api = importlib.import_module('statsmodels.api')
        target = np.asarray(y)
        if self.is_binomial:
            self.classes_ = np.unique(target)
            if len(self.classes_) != 2:  # noqa: PLR2004
                message = 'a binomial model needs a label with two classes'
                raise ValueError(message)
            target = (target == self.classes_[1]).astype(float)
        exog = self._exog(x)
        if self.kind == 'ols':
            self.results = api.OLS(target.astype(float), exog).fit()
        else:
            name = ''.join(p.capitalize() for p in self.family.split('_'))
            family = getattr(api.families, name)()
            self.results = api.GLM(
                target.astype(float), exog, family = family).fit()
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
            'kind': self.kind,
            'family': self.family,
            'constant': self.constant}

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Returns predictions (the predicted class of a binomial model).

        Args:
            x: features.

        Returns:
            The predictions.

        """
        values = np.asarray(self.results.predict(self._exog(x)))
        if self.is_binomial:
            positive = (values >= 0.5).astype(int)  # noqa: PLR2004
            return np.asarray(self.classes_[positive])
        return values

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the predicted probability of each class (binomial only).

        Args:
            x: features.

        Raises:
            AttributeError: if the model is not binomial.

        Returns:
            One column for each class, in the order of `classes_`.

        """
        if not self.is_binomial:
            message = 'only a binomial model predicts probabilities'
            raise AttributeError(message)
        values = np.asarray(self.results.predict(self._exog(x)))
        return np.column_stack([1 - values, values])

    def set_params(self, **parameters: Any) -> Statsmodel:
        """Sets parameters, as scikit-learn expects.

        Args:
            **parameters: "kind", "family", or "constant".

        Returns:
            This adapter.

        """
        for key, value in parameters.items():
            setattr(self, key, value)
        return self

    """ Private Methods """

    def _exog(self, x: pd.DataFrame) -> pd.DataFrame:
        """Returns the features as floats, with a constant if one is added.

        Args:
            x: features.

        Returns:
            The exogenous variables for statsmodels.

        """
        exog = x.astype(float)
        if self.constant:
            exog = exog.copy()
            exog.insert(0, 'const', 1.0)
        return exog


""" Models """


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
class ExtraTrees(Model):
    """An ensemble of extremely randomized trees."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.ensemble.ExtraTreesClassifier',
        'regress': 'sklearn.ensemble.ExtraTreesRegressor'}


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
        family = 'binomial' if item.task == 'classify' else 'gaussian'
        return {'family': family, **parameters, 'kind': 'glm'}


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

    scikit-learn adds a ridge (L2) penalty by default. Use `glm` for logistic
    regression without a penalty and with p-values.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.linear_model.LogisticRegression'}
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'max_iter': 1000})


@dataclasses.dataclass
class NaiveBayes(Model):
    """Gaussian naive Bayes."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.naive_bayes.GaussianNB'}


@dataclasses.dataclass
class NeuralNetwork(Model):
    """A multi-layer perceptron (a simple neural network)."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.neural_network.MLPClassifier',
        'regress': 'sklearn.neural_network.MLPRegressor'}
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'max_iter': 1000})


@dataclasses.dataclass
class OLS(Model):
    """Ordinary least squares regression from statsmodels, with inference.

    The coefficients, standard errors, p-values, and confidence intervals are
    stored in the dataset's `tables` as "{name}_coefficients".

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}

    def _prepare(
        self,
        item: base.Dataset,  # noqa: ARG002
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Makes the statsmodels adapter an ordinary least squares model.

        Args:
            item: the dataset to model.
            parameters: parameters for the model.

        Returns:
            The parameters, with "kind".

        """
        return {**parameters, 'kind': 'ols'}


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
class SVM(Model):
    """A support vector machine (with probabilities for classification)."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.svm.SVC',
        'regress': 'sklearn.svm.SVR'}
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'probability': True})


@dataclasses.dataclass
class Xgboost(Model):
    """XGBoost gradient boosting."""

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'xgboost.XGBClassifier',
        'regress': 'xgboost.XGBRegressor'}
    numbered_classes: ClassVar[bool] = True


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
