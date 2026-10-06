"""Techniques that evaluate a fitted model with tables.

These are techniques of the "critic" stage, beside the metrics. Each one adds
a table to the dataset's `tables` (under the technique's name). They evaluate
the model on the test rows (or on every row, if the data has not been split).

Contents:
    Evaluator: base class for techniques that evaluate a model with a table.
    ClassificationReport: precision, recall, and f1 for each class.
    Confusion: how many rows of each class were predicted to be each class.
    FeatureImportance: the importance that the model gives each feature.
    PermutationImportance: how much the score drops when each feature is
        shuffled.
    Scorecard: every standard metric for the task.
    ShapImportance: the mean absolute SHAP value of each feature.

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

from . import base, models, utilities
from . import metrics as metrics_


@dataclasses.dataclass
class Evaluator(base.Operation, abc.ABC):
    """Base class for techniques that evaluate a model with a table.

    A subclass writes an `evaluate` method, which returns a `DataFrame`. It is
    stored in the dataset's `tables` under the technique's name.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: not used by most evaluators. Defaults to `None`.
        parameters: keyword arguments for `evaluate`. Defaults to an empty
            `dict`.

    """

    """ Required Methods """

    @abc.abstractmethod
    def evaluate(self, item: base.Dataset, **kwargs: Any) -> pd.DataFrame:
        """Returns a table that evaluates the model of `item`.

        Args:
            item: the dataset with a fitted model.
            **kwargs: parameters for the evaluation.

        Returns:
            The table.

        """

    """ Public Methods """

    def implement(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Adds the table from `evaluate` to the tables of `item`.

        Args:
            item: the dataset with a fitted model.
            **kwargs: parameters for `evaluate`.

        Returns:
            The dataset, with the new table.

        """
        item.tables[self.name] = self.evaluate(item, **kwargs)
        item.record(self.name, table = self.name)
        return item


@dataclasses.dataclass
class ClassificationReport(Evaluator):
    """Precision, recall, f1, and the number of rows of each class."""

    def evaluate(self, item: base.Dataset, **kwargs: Any) -> pd.DataFrame:
        """Returns scikit-learn's classification report as a table.

        Args:
            item: the dataset with a fitted classifier.
            **kwargs: not used.

        Returns:
            One row for each class and for the averages.

        """
        metrics = importlib.import_module('sklearn.metrics')
        y_true, y_pred = _observed(item)
        report = metrics.classification_report(
            y_true, y_pred, output_dict = True, zero_division = 0)
        return pd.DataFrame(report).T


@dataclasses.dataclass
class Confusion(Evaluator):
    """How many rows of each class were predicted to be each class."""

    def evaluate(self, item: base.Dataset, **kwargs: Any) -> pd.DataFrame:
        """Returns the confusion matrix.

        Args:
            item: the dataset with a fitted classifier.
            **kwargs: not used.

        Returns:
            One row for each actual class and one column for each predicted
                class.

        """
        metrics = importlib.import_module('sklearn.metrics')
        y_true, y_pred = _observed(item)
        classes = item.classes
        matrix = metrics.confusion_matrix(y_true, y_pred, labels = classes)
        return pd.DataFrame(
            matrix,
            index = pd.Index(classes, name = 'actual'),
            columns = pd.Index(classes, name = 'predicted'))


@dataclasses.dataclass
class FeatureImportance(Evaluator):
    """The importance that the model itself gives each feature.

    This is the model's `feature_importances_` (for trees and boosting) or
    the absolute value of its coefficients (for linear models, averaged over
    the classes). Coefficients are only comparable if the features were
    scaled. For other models, use `permutation_importance`.

    """

    def evaluate(self, item: base.Dataset, **kwargs: Any) -> pd.DataFrame:
        """Returns the importance of each feature, largest first.

        Args:
            item: the dataset with a fitted model.
            **kwargs: not used.

        Raises:
            ValueError: if the model does not report the importance of its
                features.

        Returns:
            One row for each feature, with an "importance" column.

        """
        model = _model(item)
        importances = getattr(model, 'feature_importances_', None)
        if importances is None:
            coefficients = getattr(model, 'coef_', None)
            if coefficients is None:
                message = (
                    f'{type(model).__name__} does not report the importance of '
                    f'its features: use permutation_importance instead'
                )
                raise ValueError(message)
            importances = np.abs(np.atleast_2d(coefficients)).mean(axis = 0)
        features = list(getattr(model, 'feature_names_in_', item.features))
        return _importance_table(features, np.asarray(importances))


@dataclasses.dataclass
class PermutationImportance(Evaluator):
    """How much the model's score drops when each feature is shuffled.

    This works for any model. By default, the score is accuracy (for
    classification) or R squared (for regression), and each feature is
    shuffled 10 times.

    """

    def evaluate(
        self,
        item: base.Dataset,
        n_repeats: int = 10,
        scoring: str | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the mean and standard deviation of each feature's importance.

        Args:
            item: the dataset with a fitted model.
            n_repeats: times to shuffle each feature. Defaults to 10.
            scoring: scikit-learn scorer. Defaults to `None`, which uses
                "accuracy" or "r2" for the task.
            **kwargs: not used.

        Returns:
            One row for each feature, with "importance" and "std" columns,
                largest first.

        """
        inspection = importlib.import_module('sklearn.inspection')
        x = item.x_test if item.is_split else item.x
        y = item.y.loc[x.index]
        if scoring is None:
            scoring = 'accuracy' if item.task == 'classify' else 'r2'
        result = inspection.permutation_importance(
            _model(item),
            x,
            y,
            n_repeats = n_repeats,
            scoring = scoring,
            random_state = item.seed)
        table = _importance_table(list(x.columns), result.importances_mean)
        table['std'] = pd.Series(
            result.importances_std, index = list(x.columns))
        return table


@dataclasses.dataclass
class Scorecard(Evaluator):
    """Every standard metric for the task, in one table.

    The scores are also stored in the dataset's `metrics`, so a scorecard at
    the end of each combination in an `experiment` puts every metric in its
    comparison table. Metrics that need probabilities are skipped if the
    model does not predict them.

    """

    # The metrics in a scorecard for each task, by name.
    defaults: ClassVar[dict[str, tuple[str, ...]]] = {
        'classify': (
            'accuracy',
            'balanced_accuracy',
            'precision',
            'recall',
            'f1',
            'roc_auc',
            'log_loss'),
        'regress': ('r2', 'rmse', 'mae')}

    def evaluate(
        self,
        item: base.Dataset,
        metrics: Sequence[str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the score of each metric.

        Args:
            item: the dataset with a fitted model.
            metrics: names of the metrics to use. Defaults to `None`, which
                uses `defaults` for the task.
            **kwargs: not used.

        Returns:
            One row for each metric, with a "value" column.

        """
        names = list(metrics or self.defaults[item.task or 'classify'])
        values = {}
        for name in names:
            metric = chrisjen.library.borrow(name, genre = 'metric')()
            if not isinstance(metric, metrics_.Metric):
                message = f'{name!r} is not a metric'
                raise TypeError(message)
            if metric.uses_probabilities and item.probabilities is None:
                continue
            values[name] = metric.measure(item, **metric._keywords())
            item.metrics[name] = values[name]
        return pd.DataFrame({'value': pd.Series(values, dtype = float)})


@dataclasses.dataclass
class ShapImportance(Evaluator):
    """The mean absolute SHAP value of each feature.

    SHAP values explain each prediction as the sum of the contributions of the
    features. The `shap.Explanation` is stored in the dataset's `fitted` for
    further plots. For a classifier, the values are for the last class (the
    positive class of a binary label).

    """

    def evaluate(
        self,
        item: base.Dataset,
        rows: int = 200,
        background: int = 100,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the importance of each feature, largest first.

        Args:
            item: the dataset with a fitted model.
            rows: most rows to explain. Defaults to 200.
            background: most training rows to use as the background data.
                Defaults to 100.
            **kwargs: not used.

        Returns:
            One row for each feature, with an "importance" column.

        """
        shap = utilities.import_tool('shap')
        model = _model(item)
        if isinstance(model, models.LabelCoded):
            model = model.estimator
        train = item.x_train
        sample = train.sample(
            min(background, len(train)), random_state = item.seed)
        x = item.x_test if item.is_split else item.x
        x = x.head(rows)
        explanation = _explain(shap, model, sample, x, item.task)
        values = np.asarray(explanation.values)
        if values.ndim == 3:  # noqa: PLR2004
            values = values[:, :, -1]
        item.fitted[self.name] = explanation
        return _importance_table(list(x.columns), np.abs(values).mean(axis = 0))


""" Private Functions """


def _explain(
    shap: Any,
    model: Any,
    background: pd.DataFrame,
    x: pd.DataFrame,
    task: str | None) -> Any:
    """Returns SHAP values for `x`.

    The explainer is chosen by `shap` for the model. If `shap` does not
    recognize the model, the model's prediction function is explained
    instead.

    Args:
        shap: the `shap` module.
        model: the fitted model.
        background: background rows for the explainer.
        x: rows to explain.
        task: "classify" or "regress".

    Returns:
        A `shap.Explanation`.

    """
    try:
        explainer = shap.Explainer(model, background)
    except Exception:  # noqa: BLE001
        predict = (
            model.predict_proba
            if task == 'classify' and hasattr(model, 'predict_proba')
            else model.predict)
        explainer = shap.Explainer(predict, background)
    try:
        return explainer(x, check_additivity = False)
    except TypeError:
        return explainer(x)


def _importance_table(
    features: list[str],
    importances: np.ndarray) -> pd.DataFrame:
    """Returns a table of feature importances, largest first.

    Args:
        features: names of the features.
        importances: importance of each feature.

    Returns:
        One row for each feature (the index is named "feature"), with an
            "importance" column.

    """
    table = pd.DataFrame(
        {'importance': np.asarray(importances, dtype = float)},
        index = pd.Index(features, name = 'feature'))
    return table.sort_values('importance', ascending = False)


def _model(item: base.Dataset) -> Any:
    """Returns the fitted model of `item`.

    Args:
        item: the dataset.

    Raises:
        ValueError: if there is no model.

    Returns:
        The fitted model.

    """
    if item.model is None:
        message = 'there is no fitted model: apply a model first'
        raise ValueError(message)
    return item.model


def _observed(item: base.Dataset) -> tuple[pd.Series, pd.Series]:
    """Returns the true labels and the predictions of the predicted rows.

    Args:
        item: the dataset.

    Raises:
        ValueError: if there are no predictions.

    Returns:
        The true labels and the predictions.

    """
    if item.predictions is None:
        message = 'there are no predictions: apply a model first'
        raise ValueError(message)
    return item.y.loc[item.predictions.index], item.predictions
