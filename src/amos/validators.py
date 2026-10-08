"""Techniques that cross-validate a model, as the last step of the "analyst".

A validator measures how well the dataset's model does on rows that it did
not learn from, without using the test rows. It divides the training rows into
folds with one of scikit-learn's cross-validation splitters, fits a new copy
of the model to the rest of the training rows for each fold, and scores the
copy on the fold with the critic's metrics. The scores of each fold are a
table, stored in the dataset's `tables` under the validator's name, and their
means are stored in its `metrics` as "cv_{metric}" (such as "cv_roc_auc"), so
that a scorecard or an experiment's comparison shows them beside the scores
on the test rows.

Rows that a sampler made up (the dataset's `synthetic` rows) are never scored.
Instead, each sampler that was applied to the training rows is applied again
to the rows that each copy of the model learns from. Transformers are not
fitted again: a validator scores the model, given the features that the
transformers made after learning from all of the training rows.

Contents:
    Validator: base class for techniques that cross-validate a model.
    GroupKFold, GroupShuffleSplit, KFold, LeaveOneGroupOut, RepeatedKFold,
        RepeatedStratifiedKFold, ShuffleSplit, StratifiedGroupKFold,
        StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit: validators
        that wrap the cross-validation splitter of scikit-learn with the same
        name.
    LeaveOneRowOut: a validator that wraps scikit-learn's `LeaveOneOut` (the
        name `leave_one_out` is an encoder).

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
from collections.abc import Sequence
from typing import Any, ClassVar

import chrisjen
import numpy as np
import pandas as pd

from . import base, evaluators, utilities
from . import metrics as metrics_


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
            `None`, in which case it is the snake case name of the class.
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

    def implement(
        self,
        item: base.Dataset,
        metrics: Sequence[str] | None = None,
        groups: str | None = None,
        order: str | None = None,
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
            **kwargs: parameters for the splitter.

        Raises:
            ValueError: if there is no model, or if the validator needs
                groups or classes that the dataset does not have.

        Returns:
            The dataset, with a table of the scores of each fold, and the
                mean of each score in `metrics`.

        """
        model = self._check(item)
        clone = importlib.import_module('sklearn.base').clone
        training = item._train_rows()
        rows = training[~training.isin(item.synthetic)]
        if order is not None:
            rows = item.data.loc[rows].sort_values(order, kind = 'stable').index
        x = item.data.loc[rows, _columns(model, item)]
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
        folds: list[_Fold] = []
        divided = splitter.split(x, y, labels)
        for number, (learn, score) in enumerate(divided, start = 1):
            x_learn, y_learn = x.iloc[learn], y.iloc[learn]
            for sampler in samplers:
                x_learn, y_learn = clone(sampler).fit_resample(x_learn, y_learn)
            estimator = clone(model).fit(x_learn, y_learn)
            folds.append(_Fold(
                number, len(y_learn), *_predict(estimator, x.iloc[score], item)))
        names = evaluators.Scorecard()._metric_names(item, [], metrics)
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


@dataclasses.dataclass
class GroupKFold(Validator):
    """Folds that keep all of the rows of each group in the same fold."""

    contents: str = 'sklearn.model_selection.GroupKFold'
    uses_groups: ClassVar[bool] = True


@dataclasses.dataclass
class GroupShuffleSplit(Validator):
    """Repeated random splits of the groups (not the rows) into two sets."""

    contents: str = 'sklearn.model_selection.GroupShuffleSplit'
    uses_groups: ClassVar[bool] = True


@dataclasses.dataclass
class KFold(Validator):
    """Divides the rows into folds ("n_splits", 5 by default) at random."""

    contents: str = 'sklearn.model_selection.KFold'


@dataclasses.dataclass
class LeaveOneGroupOut(Validator):
    """Scores each group with a copy of the model fitted to the other groups."""

    contents: str = 'sklearn.model_selection.LeaveOneGroupOut'
    uses_groups: ClassVar[bool] = True


@dataclasses.dataclass
class LeaveOneRowOut(Validator):
    """Predicts each row with a copy of the model fitted to every other row.

    A fold of one row cannot be scored alone, so the predictions of every
    fold are scored together. The table has the prediction of each row.

    """

    contents: str = 'sklearn.model_selection.LeaveOneOut'
    pools: ClassVar[bool] = True


@dataclasses.dataclass
class RepeatedKFold(Validator):
    """`k_fold` repeated with different random folds ("n_repeats" times)."""

    contents: str = 'sklearn.model_selection.RepeatedKFold'


@dataclasses.dataclass
class RepeatedStratifiedKFold(Validator):
    """`stratified_k_fold` repeated with different random folds."""

    contents: str = 'sklearn.model_selection.RepeatedStratifiedKFold'
    stratifies: ClassVar[bool] = True


@dataclasses.dataclass
class ShuffleSplit(Validator):
    """Repeated random splits of the rows into two sets (Monte Carlo)."""

    contents: str = 'sklearn.model_selection.ShuffleSplit'


@dataclasses.dataclass
class StratifiedGroupKFold(Validator):
    """Folds that keep each group together and the classes in proportion."""

    contents: str = 'sklearn.model_selection.StratifiedGroupKFold'
    uses_groups: ClassVar[bool] = True
    stratifies: ClassVar[bool] = True


@dataclasses.dataclass
class StratifiedKFold(Validator):
    """Folds that keep the share of each class the same in every fold."""

    contents: str = 'sklearn.model_selection.StratifiedKFold'
    stratifies: ClassVar[bool] = True


@dataclasses.dataclass
class StratifiedShuffleSplit(Validator):
    """Repeated random splits into two sets, with the classes in proportion."""

    contents: str = 'sklearn.model_selection.StratifiedShuffleSplit'
    stratifies: ClassVar[bool] = True


@dataclasses.dataclass
class TimeSeriesSplit(Validator):
    """Scores later rows with copies of the model fitted to earlier rows.

    Use this when the rows are ordered in time (or set "order" to a column,
    such as a date), so that the model is never scored on rows that came
    before the rows it learned from.

    """

    contents: str = 'sklearn.model_selection.TimeSeriesSplit'


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
        The model's features (or the dataset's features, if the model does
            not name them), and any other columns that the model's parameters
            name, such as the fixed effects of a `fixest` model.

    """
    names = getattr(model, 'feature_names_in_', None)
    columns = list(item.features if names is None else names)
    get_params = getattr(model, 'get_params', None)
    parameters = get_params(deep = False) if callable(get_params) else {}
    for parameter in _column_parameters():
        for name in base._listify(parameters.get(parameter)):
            if name in item.data.columns and name not in columns:
                columns.append(name)
    return columns


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
    label = item._require_label()
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
            values[name] = metric.measure(item, **metric._keywords())
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
    item: base.Dataset) -> tuple[pd.Series, pd.DataFrame | None]:
    """Returns the predictions (and probabilities) of a copy of the model.

    Args:
        estimator: the copy of the model fitted for a fold.
        x: the features of the rows of the fold.
        item: the dataset that is validated.

    Raises:
        ValueError: if the copy of the model could not predict some rows.

    Returns:
        The predictions, and the predicted probability of each class for a
            classifier that has them (or `None`).

    """
    label = item._require_label()
    predictions = pd.Series(
        np.asarray(estimator.predict(x)), index = x.index, name = label)
    missing = int(predictions.isna().sum())
    if missing:
        message = (
            f'a copy of the model could not predict {missing} of the '
            f'{len(x)} rows it was given (a model with fixed effects, for '
            f'example, cannot predict a group that it did not learn from)'
        )
        raise ValueError(message)
    probabilities = None
    if item.task == 'classify' and hasattr(estimator, 'predict_proba'):
        classes = getattr(estimator, 'classes_', None)
        probabilities = pd.DataFrame(
            np.asarray(estimator.predict_proba(x)),
            index = x.index,
            columns = list(item.classes if classes is None else classes))
    return predictions, probabilities
