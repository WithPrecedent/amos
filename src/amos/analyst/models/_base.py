"""The base class of models, and what they share.

Contents:
    Model: base class for techniques that fit models.
    LabelCoded: lets a classifier that needs classes numbered 0 to n - 1 use any
        classes.
    search_folds: the folds of a search, which keep the rows of each group
        together.

"""

from __future__ import annotations

import abc
import dataclasses
import functools
import importlib
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import camina
import numpy as np
import pandas as pd

from ...core import base, utilities


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

    A model lists by name what it assumes about the data in `assumptions`
    (such as "normal_errors") and what it needs from it in `needs` (such as
    "no_missing_values"). They are checked before it is fitted, and the
    dataset's `assumption_violations` decides what is done if the data does
    not meet one (see `base.Operation._check_conditions`). `auto_model`
    reads them, with `package` and `complexity`, to choose a model.

    A model that can be fitted on a GPU overrides `_device_parameters` to
    return the parameters that put its tool on the GPU or the CPU. It is
    fitted on a GPU if the dataset's `use_gpu` is `True` and an NVIDIA GPU is
    found (see `utilities.gpus`).

    Wraps:
        - [`sklearn.model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
          from scikit-learn, for a "grid" search.
        - [`sklearn.model_selection.RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
          from scikit-learn, for a "random" search.
        - [`optuna_integration.OptunaSearchCV`](https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.OptunaSearchCV.html)
          from Optuna, for an "optuna" search.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
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
    # Parameters that the tool accepts only through `**kwargs` (and so does
    # not list in its signature or `get_params`), which are passed to it
    # when they are set.
    keyword_parameters: ClassVar[tuple[str, ...]] = ()
    # The names of what the model assumes about the data (see the
    # `assumptions` module), which are checked before it is fitted (and, for
    # some, after) and which `auto_model` reads.
    assumptions: ClassVar[tuple[str, ...]] = ()
    # The names of what the model needs from the data (see the `needs`
    # module), which are checked before it is fitted.
    needs: ClassVar[tuple[str, ...]] = ()
    # The package that the model needs, which `auto_model` checks is
    # installed, or an empty `str` for a model that `auto_model` does not
    # know.
    package: ClassVar[str] = ''
    # How complex the model is, from 1 (the simplest, such as a linear
    # model) up. Of models that score about as well, `auto_model` chooses
    # the simplest.
    complexity: ClassVar[int] = 3
    # Fewest training rows the model learns well from (the "min_rows"
    # need), most it can be fitted to in reasonable time ("max_rows"), and
    # most features it works well with ("max_features").
    min_rows: ClassVar[int] = 0
    max_rows: ClassVar[int | None] = None
    max_features: ClassVar[int | None] = None

    """ Public Methods """

    def apply(
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
            search: "grid", "random", or "optuna" to search for the best
                values of any parameter given as a list. Defaults to `None`,
                which does not search (and passes lists to the model as they
                are).
            cv: number of cross-validation folds for a search, which keep
                the rows of each of the dataset's first groups together if
                it has groups. Defaults to 5.
            n_iter: number of combinations to try in a random or Optuna
                search. Defaults to 10.
            scoring: scikit-learn scorer for a search (such as "roc_auc").
                Defaults to `None`, which uses the model's `score` method.
            **kwargs: parameters for the model. Any "plots", "evaluators",
                or "inferers" (the techniques that replace a workflow's for
                this model) are left out.

        Returns:
            The dataset, with the fitted `model`, `predictions`, and (for a
                classifier) `probabilities`.

        """
        # The techniques that replace a workflow's plots, evaluators, and
        # inferers for this model are not parameters of its tool (see
        # `base.using_model_specific_techniques`).
        kwargs = {
            k: v for k, v in kwargs.items()
            if k not in base._MODEL_SPECIFIC_GENRES}
        task = self._check_task(item)
        extras = self._extra_columns(item, kwargs)
        self._check_features(item, extras)
        path = self._choose_tool(task)
        tool = utilities.import_tool(path)
        fixed, space = (
            (kwargs, {}) if search is None
            else utilities.search_space(kwargs, search))
        space = utilities.accepted_parameters(tool, space)
        columns = [c for c in item.features if c not in extras] + extras
        y = item.y_train
        x = item.data.loc[y.index, columns]
        fit = functools.partial(
            self._fit,
            item,
            tool,
            x,
            y,
            task = task,
            fixed = fixed,
            space = space,
            search = search,
            cv = cv,
            n_iter = n_iter,
            scoring = scoring)
        gpu = self._uses_gpu(item)
        try:
            estimator = fit(gpu = gpu)
        # Anything that stops a model from using the GPU (such as a package
        # built without GPU support, a GPU that is too old for it, or too
        # little memory) means fitting it on the CPU instead. An error that
        # has nothing to do with the GPU is raised again by the CPU's fit.
        except Exception as error:
            if not gpu:
                raise
            warnings.warn(
                f'{self.name!r} could not be fitted on the GPU, so it was '
                f'fitted on the CPU instead: {error}',
                category = base.AmosWarning,
                stacklevel = 2)
            estimator = fit(gpu = False)
        item.model = estimator
        self._predict(item, estimator, task, columns)
        if hasattr(estimator, 'coefficients'):
            item.tables[f'{self.name}_coefficients'] = estimator.coefficients()
        item.record(
            self.name,
            role = 'model',
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
            if 'count_label' in self.needs:
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

    def _device_parameters(
        self,
        *,
        gpu: bool) -> dict[str, Any] | None:  # noqa: ARG002
        """Returns the parameters that fit the model on a GPU or the CPU.

        Models that can be fitted on a GPU override this.

        Args:
            gpu: whether to fit the model on the first GPU (or on the CPU).

        Returns:
            The parameters for the tool, or `None` if the model can only be
                fitted on the CPU.

        """
        return None

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
            for name in camina.listify(parameters.get(parameter)):
                if name not in item.data.columns:
                    message = (
                        f'the {parameter} column {name!r} is not in the data')
                    raise KeyError(message)
                if name not in names:
                    names.append(name)
        return names

    def _fit(
        self,
        item: base.Dataset,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series,
        *,
        task: str,
        fixed: dict[str, Any],
        space: dict[str, Any],
        gpu: bool,
        search: str | None,
        cv: int,
        n_iter: int,
        scoring: str | None) -> Any:
        """Builds the model and fits it (or searches for the best one).

        Args:
            item: the dataset being modeled.
            tool: the model class.
            x: the training rows of the columns the model uses.
            y: the training labels.
            task: "classify" or "regress".
            fixed: the parameters that are not searched.
            space: the parameters to search, which may be empty.
            gpu: whether to fit the model on the GPU.
            search: "grid", "random", "optuna", or `None`.
            cv: number of cross-validation folds for a search.
            n_iter: number of combinations to try in a random or Optuna
                search.
            scoring: scikit-learn scorer for a search, or `None`.

        Returns:
            The fitted model.

        """
        known = {
            **(self._device_parameters(gpu = gpu) or {}),
            **{k: v for k, v in fixed.items() if k in self.keyword_parameters}}
        estimator = self._make_tool(item, fixed, tool = tool, known = known)
        if (
            task == 'classify'
            and self.numbered_classes
            and not _is_numbered(item.classes)):
            estimator = LabelCoded(estimator = estimator)
            space = {f'estimator__{k}': v for k, v in space.items()}
        if space:
            return self._search(
                item,
                estimator,
                x,
                y,
                space = space,
                search = search,
                cv = cv,
                n_iter = n_iter,
                scoring = scoring)
        estimator.fit(x, y)
        return estimator

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

    def _folds(
        self,
        item: base.Dataset,
        x: pd.DataFrame,
        y: pd.Series,
        cv: int) -> int | list[tuple[np.ndarray, np.ndarray]]:
        """Returns the folds of a search, which keep groups together.

        Args:
            item: the dataset being modeled.
            x: the training rows of the columns the model uses.
            y: the training labels.
            cv: number of cross-validation folds.

        Returns:
            The folds from `search_folds`.

        """
        return search_folds(item, x, y, cv)

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
        as "{name}_search". If the dataset has groups, the folds keep the
        rows of each of its first group together (see `_folds`).

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
        folds = self._folds(item, x, y, cv)
        if search == 'optuna':
            return self._search_optuna(
                item, estimator, x, y,
                space = space, cv = folds, n_iter = n_iter, scoring = scoring)
        model_selection = importlib.import_module('sklearn.model_selection')
        if search == 'random':
            searcher = model_selection.RandomizedSearchCV(
                estimator,
                param_distributions = space,
                n_iter = n_iter,
                cv = folds,
                scoring = scoring,
                random_state = item.seed)
        else:
            searcher = model_selection.GridSearchCV(
                estimator, param_grid = space, cv = folds, scoring = scoring)
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
        cv: int | list[tuple[np.ndarray, np.ndarray]],
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
            cv: number of cross-validation folds, or the folds (as `_folds`
                returns them).
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

    def _uses_gpu(self, item: base.Dataset) -> bool:
        """Returns whether to fit the model on a GPU.

        Args:
            item: the dataset being modeled.

        Returns:
            Whether the dataset's `use_gpu` is `True`, the model can be fitted
                on a GPU, and an NVIDIA GPU is found. If no GPU is found, it
                warns that the model is fitted on the CPU.

        """
        if not item.use_gpu or self._device_parameters(gpu = True) is None:
            return False
        if not utilities.gpus():
            warnings.warn(
                'use_gpu is true, but no NVIDIA GPU was found, so models are '
                'fitted on the CPU',
                category = base.AmosWarning,
                stacklevel = 3)
            return False
        return True


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

    def get_params(self, deep: bool = True) -> dict[str, Any]:
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
        """Returns the scikit-learn tags of the wrapped classifier.

        Returns:
            The tags of `estimator`.

        """
        return self.estimator.__sklearn_tags__()


""" Public Functions """


def search_folds(
    item: base.Dataset,
    x: pd.DataFrame,
    y: pd.Series,
    cv: int) -> int | list[tuple[np.ndarray, np.ndarray]]:
    """Returns the folds of a search, which keep groups together.

    When rows are not independent (such as the ratings of several reviewers of
    one product), a fold that splits a group puts rows that are nearly copies of
    each other on both sides, so the search scores the model on rows it has
    partly seen. So if the dataset has groups, each group of its first group
    column is kept wholly in one fold, as the group validators keep them: with
    the classes balanced across folds for a classifier, and in folds of about
    equal size for a regression. A row with a missing group is a group of its
    own.

    Wraps:
        - [`sklearn.model_selection.StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
          from scikit-learn, for a classifier.
        - [`sklearn.model_selection.GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
          from scikit-learn, for a regression.

    Args:
        item: the dataset being modeled.
        x: the training rows of the columns the model uses.
        y: the training labels.
        cv: number of cross-validation folds.

    Returns:
        `cv`, if the dataset has no groups, or else the training and test
            positions of each fold.

    """
    if not item.groups:
        return cv
    model_selection = importlib.import_module('sklearn.model_selection')
    groups = item.data.loc[x.index, item.groups[0]].astype(object)
    missing = groups.isna()
    groups[missing] = [f'__row__{label}' for label in groups.index[missing]]
    codes = pd.factorize(groups)[0]
    if item.task == 'classify':
        splitter = model_selection.StratifiedGroupKFold(
            n_splits = cv, shuffle = True, random_state = item.seed)
    else:
        splitter = model_selection.GroupKFold(n_splits = cv)
    return list(splitter.split(x, y, codes))


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
