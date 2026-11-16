"""The base class of validators, and what they share.

Contents:
    Validator: base class for techniques that cross-validate a model.

"""

from __future__ import annotations

import abc
import dataclasses
import gc
import importlib
from collections.abc import Sequence
from typing import Any, ClassVar

import camina
import chrisjen
import numpy as np
import pandas as pd

from ...core import base, utilities
from ...core import metrics as metrics_
from ...critic import evaluators
from .. import samplers


@dataclasses.dataclass
class Validator(base.Operation, abc.ABC):
    """Base class for techniques that cross-validate the dataset's model.

    `contents` is the cross-validation splitter to wrap: a class (or its
    import path) whose `split` method yields the positions of the rows to
    learn from and of the rows to score in each fold, as in scikit-learn. A
    `Validator` can be used directly to wrap any such class. Unless "shuffle"
    is false, a splitter that can shuffle the rows does, with the dataset's
    seed.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: the splitter class or its import path. Defaults to `None`.
        parameters: keyword arguments for the splitter (such as "n_splits"),
            and "metrics", "groups", and "order". Defaults to an empty
            `dict`.

    """

    # Whether the splitter keeps the rows of each group in the same fold.
    uses_groups: ClassVar[bool] = False
    # Whether the splitter keeps the share of each class in each fold.
    stratifies: ClassVar[bool] = False
    # Whether the predictions of every fold are scored together, for folds
    # too small to score alone. Otherwise, the scores of the folds are
    # averaged.
    pools: ClassVar[bool] = False

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        metrics: Sequence[str] | None = None,
        groups: str | None = None,
        order: str | None = None,
        n_jobs: int | None = 1,
        **kwargs: Any) -> base.Dataset:
        """Cross-validates the model on the training rows of `item`.

        Args:
            item: the dataset with a fitted model.
            metrics: names of the metrics to score each fold with. Defaults to
                `None`, which uses the metrics of a scorecard for the task.
            groups: name of the column that identifies each row's group, for
                a validator that keeps each group in one fold. Defaults to
                `None`, in which case the dataset's first group is used.
            order: name of the column to order the rows by (such as a date)
                before they are divided. Defaults to `None`, which uses the
                current order of the rows.
            n_jobs: how many folds to fit at once, each in its own process
                (-1 for one for each processor). Each process holds a copy
                of the training rows and of its model, so more use more
                memory. Defaults to 1, which fits one fold at a time.
            **kwargs: parameters for the splitter.

        Raises:
            ValueError: if there is no model, or if the validator needs
                groups or classes that the dataset does not have.

        Returns:
            The dataset, with a table of the scores of each fold, and the
                mean of each score in `metrics`.

        """
        model = self._check(item)
        rows = item.real_train_rows
        if order is not None:
            rows = item.data.loc[rows].sort_values(order, kind = 'stable').index
        x = item.data.loc[rows, _columns(model, item)]
        features = _features(model, item)
        y = item.y.loc[rows]
        column = groups or (item.groups[0] if item.groups else None)
        if self.uses_groups and column is None:
            message = f'{self.name!r} needs the name of a "groups" column'
            raise ValueError(message)
        labels = item.data.loc[rows, column] if self.uses_groups else None
        splitter = self._make_tool(item, kwargs)
        samplers = [
            tool for tool in item.fitted.values()
            if hasattr(tool, 'fit_resample')]
        shared = {
            'model': model,
            'samplers': samplers,
            'x': x,
            'y': y,
            'features': features}
        target = _Target(
            label = item.require_label(),
            task = item.task,
            classes = item.classes if item.task == 'classify' else [])
        # The folds are made one at a time: a validator such as
        # `leave_one_row_out` has as many folds as rows.
        divided = enumerate(splitter.split(x, y, labels), start = 1)
        folds: list[_Fold]
        if n_jobs == 1:
            folds = []
            for number, (learn, score) in divided:
                folds.append(_fit_fold(
                    number, learn, score, target = target, **shared))
                # A fitted model can be as large as its data, so the one of
                # each fold is freed before the next is fitted. The results
                # of pyfixest are in reference cycles, so they need a
                # collection. It takes a noticeable time, which is too long
                # for each of the many folds of a pooled validator, whose
                # folds are too small to need it.
                if not self.pools:
                    gc.collect()
        else:
            joblib = importlib.import_module('joblib')
            folds = list(joblib.Parallel(n_jobs = n_jobs)(
                joblib.delayed(_fit_fold)(
                    number, learn, score, target = target, **shared)
                for number, (learn, score) in divided))
        names = evaluators.Scorecard.metric_names(item, metrics = metrics)
        summarize = _pool if self.pools else _average
        table, scores = summarize(item, rows, folds, names)
        item.tables[self.name] = table
        for name, score in scores.items():
            item.metrics[f'cv_{name}'] = score['mean']
        item.record(
            self.name,
            tool = utilities.describe_tool(self.contents),
            parameters = utilities.parameters_of(splitter),
            folds = len(table),
            rows = len(rows),
            scores = scores)
        return item

    """ Private Methods """

    def _check(self, item: base.Dataset) -> Any:
        """Returns the dataset's model, checking that it can be validated.

        Args:
            item: the dataset to validate.

        Raises:
            ValueError: if there is no model, or if the validator keeps the
                share of each class in each fold and the label is not
                classified.

        Returns:
            The fitted model.

        """
        if item.model is None:
            message = f'{self.name!r} needs a model: apply a model first'
            raise ValueError(message)
        if self.stratifies and item.task != 'classify':
            message = (
                f'{self.name!r} keeps the share of each class in each fold, so '
                f'it needs a classification task, not {item.task!r}'
            )
            raise ValueError(message)
        return item.model

    def _prepare(
        self,
        item: base.Dataset,  # noqa: ARG002
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Shuffles the rows unless "shuffle" is false.

        Args:
            item: the dataset to validate.
            parameters: parameters for the splitter.

        Returns:
            The parameters. A splitter that does not shuffle is not given the
                dataset's seed, which scikit-learn does not allow.

        """
        parameters.setdefault('shuffle', True)
        if not parameters['shuffle']:
            parameters.setdefault('random_state', None)
        return parameters


""" Private Classes and Functions """


@dataclasses.dataclass
class _Fold:
    """The predictions of the copy of the model fitted for one fold.

    Args:
        number: the number of the fold, from 1.
        learned: how many rows the copy of the model learned from.
        predictions: its predictions for the rows of the fold.
        probabilities: its predicted probabilities of each class for the
            rows of the fold, or `None`.

    """

    number: int
    learned: int
    predictions: pd.Series
    probabilities: pd.DataFrame | None


@dataclasses.dataclass(frozen = True)
class _Target:
    """What the copy of the model of each fold predicts.

    A fold needs only this from the dataset, so that a fold fitted in
    another process is not sent the whole dataset.

    Args:
        label: name of the label.
        task: "classify" or "regress".
        classes: the classes of the label (for a classifier), sorted as the
            dataset sorts them.

    """

    label: str
    task: str | None
    classes: list[Any]


def _average(
    item: base.Dataset,
    rows: pd.Index,
    folds: Sequence[_Fold],
    names: Sequence[str]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Scores each fold and averages the scores.

    Args:
        item: the dataset that is validated.
        rows: the training rows that are divided into folds.
        folds: the predictions of each fold.
        names: names of the metrics to score with.

    Returns:
        A table with a row for each fold (the rows learned from and scored,
            and the score of each metric), and the mean and standard
            deviation of the scores of each metric that scored a fold.

    """
    records = []
    for fold in folds:
        scored = _dataset(item, rows, fold.predictions, fold.probabilities)
        records.append({
            'fold': fold.number,
            'train': fold.learned,
            'validation': len(fold.predictions),
            **_measure(scored, names)})
    table = pd.DataFrame(records).set_index('fold')
    scores = {
        name: {
            'mean': float(table[name].mean()),
            'sd': float(table[name].std())}
        for name in names
        if name in table.columns and table[name].notna().any()}
    return table, scores


def _columns(model: Any, item: base.Dataset) -> list[Any]:
    """Returns the columns that the model was fitted to.

    Args:
        model: the fitted model.
        item: the dataset it was fitted to.

    Returns:
        The model's features (see `_features`), and any other columns that
            the model's parameters name, such as the fixed effects of a
            `fixest` model.

    """
    columns = _features(model, item)
    get_params = getattr(model, 'get_params', None)
    parameters = get_params(deep = False) if callable(get_params) else {}
    for parameter in _column_parameters():
        for name in camina.listify(parameters.get(parameter)):
            if name in item.data.columns and name not in columns:
                columns.append(name)
    return columns


def _features(model: Any, item: base.Dataset) -> list[Any]:
    """Returns the features that the model was fitted to.

    Args:
        model: the fitted model.
        item: the dataset it was fitted to.

    Returns:
        The model's features, or the dataset's features if the model does
            not name them.

    """
    names = utilities.feature_names(model, item.data.columns)
    return list(item.features if names is None else names)


def _resampled(
    sampler: Any,
    x: pd.DataFrame,
    y: pd.Series,
    features: Sequence[Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Resamples the features of `x`, as a sampler of the analysis did.

    A model can need columns besides its features, such as the fixed effects
    of a `fixest` model, which are often text that a sampler (such as
    `smote`) cannot resample. Only the features are resampled, and each row
    that copies a row of `x` gets that row's other columns, as
    `Dataset.resample` gives it that row's groups. A made-up row has none
    of them.

    Args:
        sampler: a sampler that has not been fitted.
        x: the rows that a copy of the model learns from.
        y: their labels.
        features: the model's features.

    Returns:
        The resampled rows (with every column of `x`) and labels.

    """
    others = [c for c in x.columns if c not in set(features)]
    kept = x if not others else x[list(features)]
    x_new, y_new = sampler.fit_resample(kept, y)
    if not others:
        return x_new, y_new
    origins = getattr(sampler, 'sample_indices_', None)
    if origins is None:
        origins = samplers._leading_copies(kept, x_new)
    positions = np.asarray(origins)
    copied = positions >= 0
    x_new = x_new.copy()
    for column in others:
        values = pd.Series(pd.NA, index = x_new.index, dtype = x[column].dtype)
        values[copied] = x[column].to_numpy()[positions[copied]]
        x_new[column] = values
    return x_new[list(x.columns)], y_new


def _column_parameters() -> set[str]:
    """Returns the names of model parameters that name columns of the data.

    Returns:
        The `column_parameters` of every model in the library.

    """
    names: set[str] = set()
    for kind in chrisjen.library.get_genre('model').values():
        names.update(getattr(kind, 'column_parameters', ()))
    return names


def _dataset(
    item: base.Dataset,
    rows: pd.Index,
    predictions: pd.Series,
    probabilities: pd.DataFrame | None) -> base.Dataset:
    """Returns a dataset of the training rows, to score some predictions of.

    Args:
        item: the dataset that is validated.
        rows: the training rows that are divided into folds.
        predictions: predictions of some of `rows`.
        probabilities: predicted probabilities of the same rows, or `None`.

    Returns:
        A dataset of `rows` (with only the label and groups), split so that
            the predicted rows are the test set, with the predictions and
            probabilities.

    """
    label = item.require_label()
    scored = predictions.index
    dataset = base.Dataset(
        data = item.data.loc[rows, [label, *item.groups]],
        label = label,
        task = item.task,
        seed = item.seed,
        groups = list(item.groups))
    dataset.split(rows[~rows.isin(scored)], scored)
    dataset.predictions = predictions
    dataset.probabilities = probabilities
    return dataset


def _fit_fold(
    number: int,
    learn: np.ndarray,
    score: np.ndarray,
    *,
    model: Any,
    samplers: Sequence[Any],
    x: pd.DataFrame,
    y: pd.Series,
    features: Sequence[Any],
    target: _Target) -> _Fold:
    """Fits a copy of the model to one fold and predicts the rows it scores.

    It needs nothing else from the validator, so that folds can be fitted
    at once in other processes.

    Args:
        number: the number of the fold, from 1.
        learn: positions (in `x`) of the rows that the copy learns from.
        score: positions of the rows that it predicts.
        model: the dataset's model, which is copied without its fit.
        samplers: the samplers of the analysis, which resample the rows that
            the copy learns from, as they resampled the training rows.
        x: the columns of the training rows that the model uses.
        y: the labels of the training rows.
        features: the model's features.
        target: what the model predicts.

    Returns:
        The predictions of the copy of the model for the fold.

    """
    clone = importlib.import_module('sklearn.base').clone
    x_learn, y_learn = x.iloc[learn], y.iloc[learn]
    for sampler in samplers:
        x_learn, y_learn = _resampled(
            clone(sampler), x_learn, y_learn, features)
    estimator = clone(model).fit(x_learn, y_learn)
    predictions, probabilities = _predict(estimator, x.iloc[score], target)
    return _Fold(number, len(y_learn), predictions, probabilities)


def _measure(item: base.Dataset, names: Sequence[str]) -> dict[str, float]:
    """Returns the score of each metric in `names` for some predictions.

    Args:
        item: a dataset with predictions.
        names: names of metrics in the library.

    Raises:
        TypeError: if a name is not the name of a metric.

    Returns:
        The scores. Metrics for another task, and metrics that need
            probabilities that the model did not make, are left out. A
            metric that cannot score the rows (such as `roc_auc` of rows of
            one class) has a missing score.

    """
    values = {}
    for name in names:
        metric = chrisjen.library.borrow(name, genre = 'metric')()
        if not isinstance(metric, metrics_.Metric):
            message = f'{name!r} is not a metric'
            raise TypeError(message)
        if (
            item.task not in metric.tasks
            or (metric.uses_probabilities and item.probabilities is None)):
            continue
        try:
            values[name] = metric.measure(item)
        except ValueError:
            values[name] = np.nan
    return values


def _pool(
    item: base.Dataset,
    rows: pd.Index,
    folds: Sequence[_Fold],
    names: Sequence[str]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Scores the predictions of every fold together.

    Args:
        item: the dataset that is validated.
        rows: the training rows that are divided into folds.
        folds: the predictions of each fold.
        names: names of the metrics to score with.

    Returns:
        A table with a row for each predicted row (its fold, true label, and
            prediction), and the score of each metric that scored the
            predictions (as its "mean").

    """
    predictions = pd.concat([fold.predictions for fold in folds])
    probabilities = None
    if all(fold.probabilities is not None for fold in folds):
        probabilities = pd.concat([fold.probabilities for fold in folds])
    scored = _dataset(item, rows, predictions, probabilities)
    table = pd.DataFrame({
        'fold': np.repeat(
            [fold.number for fold in folds],
            [len(fold.predictions) for fold in folds]),
        'actual': item.y.loc[predictions.index],
        'prediction': predictions})
    scores = {
        name: {'mean': float(value)}
        for name, value in _measure(scored, names).items()
        if not np.isnan(value)}
    return table, scores


def _predict(
    estimator: Any,
    x: pd.DataFrame,
    target: _Target) -> tuple[pd.Series, pd.DataFrame | None]:
    """Returns the predictions (and probabilities) of a copy of the model.

    Args:
        estimator: the copy of the model fitted for a fold.
        x: the features of the rows of the fold.
        target: what the model predicts.

    Raises:
        ValueError: if the copy of the model could not predict some rows.

    Returns:
        The predictions, and the predicted probability of each class for a
            classifier that has them (or `None`).

    """
    predictions = pd.Series(
        np.asarray(estimator.predict(x)), index = x.index, name = target.label)
    missing = int(predictions.isna().sum())
    if missing:
        message = (
            f'a copy of the model could not predict {missing} of the '
            f'{len(x)} rows it was given (a model may not be able to '
            f'predict a group that it did not learn from)'
        )
        raise ValueError(message)
    probabilities = None
    if target.task == 'classify' and hasattr(estimator, 'predict_proba'):
        classes = getattr(estimator, 'classes_', None)
        probabilities = pd.DataFrame(
            np.asarray(estimator.predict_proba(x)),
            index = x.index,
            columns = list(target.classes if classes is None else classes))
    return predictions, probabilities
