"""Metrics that score a model's predictions.

These are techniques of the "critic" stage. Each one wraps a scoring function
from `sklearn.metrics`, compares the model's predictions (or predicted
probabilities) with the true labels of the same rows, and stores the score in
the dataset's `metrics` under the technique's name.

Every metric is also a `chrisjen.Criteria`, so the same name can be the
"criterion" of an `experiment` or `contest`, which keeps the combination of
techniques with the best score. Metrics for which lower is better (such as
`log_loss` and `rmse`) are negated when they are used as criteria, so the best
combination still has the highest score.

The details of each scoring function are handled for you. For a binary label,
the positive class is the last of `Dataset.classes` (such as 1 or `True`), and
the probability of that class is scored. For more than two classes,
`precision`, `recall`, and `f1` are averaged over the classes ("macro") and
`roc_auc` compares each class with the rest. Any of these can be changed with
parameters.

Contents:
    Metric: base class for metrics.
    Accuracy, AveragePrecision, BalancedAccuracy, Brier, CohenKappa, F1,
        LogLoss, Matthews, Precision, Recall, RocAuc: classification metrics.
    ExplainedVariance, MAE, MAPE, MSE, R2, RMSE: regression metrics.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, ClassVar

import chrisjen
import pandas as pd

from . import base, utilities


@dataclasses.dataclass
class Metric(base.Operation, chrisjen.Criteria, abc.ABC):
    """Base class for metrics, which score a model's predictions.

    `contents` is the scoring function (or its import path). It is called with
    the true labels, the predictions (or probabilities), and the parameters it
    accepts. A subclass can instead override `measure`.

    Args:
        name: name used to refer to the metric in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: the scoring function or its import path. Defaults to `None`.
        parameters: keyword arguments for the scoring function. Defaults to
            an empty `dict`.

    """

    # Whether a higher score is better.
    greater_is_better: ClassVar[bool] = True
    # The tasks that the metric scores.
    tasks: ClassVar[tuple[str, ...]] = ('classify',)
    # Whether the metric scores predicted probabilities (rather than
    # predictions).
    uses_probabilities: ClassVar[bool] = False

    """ Public Methods """

    def implement(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Stores the score of the model's predictions in `metrics`.

        Args:
            item: the dataset to score. It must have predictions.
            **kwargs: parameters for the scoring function.

        Returns:
            The dataset, with the score in `metrics`.

        """
        value = self.measure(item, **kwargs)
        item.metrics[self.name] = value
        item.record(
            self.name,
            tool = utilities.describe_tool(self.contents),
            value = value)
        return item

    def measure(self, item: base.Dataset, **kwargs: Any) -> float:
        """Returns the score of the model's predictions.

        Args:
            item: the dataset to score. It must have predictions.
            **kwargs: parameters for the scoring function.

        Raises:
            ValueError: if the metric does not score the dataset's task.

        Returns:
            The score.

        """
        if item.task not in self.tasks:
            message = (
                f'{self.name!r} scores {" and ".join(self.tasks)} tasks, but '
                f'the task is {item.task!r}'
            )
            raise ValueError(message)
        y_true, y_score = self._observed(item)
        if self.contents is None:
            message = f'{self.name!r} has no scoring function in contents'
            raise NotImplementedError(message)
        tool = utilities.import_tool(self.contents)
        parameters = self._prepare(item, dict(kwargs))
        arguments = utilities.accepted_parameters(tool, parameters)
        return float(tool(y_true, y_score, **arguments))

    def score(self, item: base.Dataset) -> float:
        """Returns the score of `item` for comparison. Higher is better.

        This is how `chrisjen` designs use a metric as their criteria.

        Args:
            item: the dataset to score.

        Returns:
            The score, negated if lower scores are better.

        """
        value = self.measure(item, **self._keywords())
        return value if self.greater_is_better else -value

    """ Private Methods """

    def _observed(self, item: base.Dataset) -> tuple[pd.Series, Any]:
        """Returns the true labels and the predictions to score.

        Args:
            item: the dataset to score.

        Raises:
            ValueError: if there are no predictions (or no probabilities, for
                a metric that needs them).

        Returns:
            The true labels of the predicted rows, and the predictions or
                probabilities. For a binary label, the probability of the
                positive class is returned.

        """
        if item.predictions is None:
            message = f'{self.name!r} needs predictions: apply a model first'
            raise ValueError(message)
        y_true = item.y.loc[item.predictions.index]
        if not self.uses_probabilities:
            return y_true, item.predictions
        if item.probabilities is None:
            message = (
                f'{self.name!r} needs predicted probabilities, which the model '
                f'does not make'
            )
            raise ValueError(message)
        if item.probabilities.shape[1] == 2:  # noqa: PLR2004
            return y_true, item.probabilities.iloc[:, -1]
        return y_true, item.probabilities

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Adds defaults for a binary or multiclass label.

        Only the parameters that the scoring function accepts are passed to
        it, so these defaults do not affect functions that do not use them.

        Args:
            item: the dataset to score.
            parameters: parameters for the scoring function.

        Returns:
            The parameters, with defaults for the positive class, averaging,
                and the order of the classes.

        """
        if item.task != 'classify':
            return parameters
        classes = item.classes
        if len(classes) == 2:  # noqa: PLR2004
            parameters.setdefault('pos_label', classes[-1])
        else:
            parameters.setdefault('average', 'macro')
            parameters.setdefault('multi_class', 'ovr')
            if self.uses_probabilities and item.probabilities is not None:
                parameters.setdefault('labels', list(item.probabilities.columns))
        return parameters


""" Classification Metrics """


@dataclasses.dataclass
class Accuracy(Metric):
    """The share of rows classified correctly."""

    contents: str = 'sklearn.metrics.accuracy_score'


@dataclasses.dataclass
class AveragePrecision(Metric):
    """The area under the precision-recall curve."""

    contents: str = 'sklearn.metrics.average_precision_score'
    uses_probabilities: ClassVar[bool] = True


@dataclasses.dataclass
class BalancedAccuracy(Metric):
    """The average share of each class classified correctly."""

    contents: str = 'sklearn.metrics.balanced_accuracy_score'


@dataclasses.dataclass
class Brier(Metric):
    """The mean squared error of the predicted probabilities (lower is better)."""

    contents: str = 'sklearn.metrics.brier_score_loss'
    greater_is_better: ClassVar[bool] = False
    uses_probabilities: ClassVar[bool] = True


@dataclasses.dataclass
class CohenKappa(Metric):
    """Agreement between the predictions and labels beyond chance."""

    contents: str = 'sklearn.metrics.cohen_kappa_score'


@dataclasses.dataclass
class F1(Metric):
    """The harmonic mean of precision and recall."""

    contents: str = 'sklearn.metrics.f1_score'


@dataclasses.dataclass
class LogLoss(Metric):
    """The negative log-likelihood of the true labels (lower is better)."""

    contents: str = 'sklearn.metrics.log_loss'
    greater_is_better: ClassVar[bool] = False
    uses_probabilities: ClassVar[bool] = True


@dataclasses.dataclass
class Matthews(Metric):
    """The Matthews correlation coefficient (phi for two classes)."""

    contents: str = 'sklearn.metrics.matthews_corrcoef'


@dataclasses.dataclass
class Precision(Metric):
    """The share of rows predicted to be positive that are positive."""

    contents: str = 'sklearn.metrics.precision_score'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'zero_division': 0})


@dataclasses.dataclass
class Recall(Metric):
    """The share of positive rows that are predicted to be positive."""

    contents: str = 'sklearn.metrics.recall_score'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'zero_division': 0})


@dataclasses.dataclass
class RocAuc(Metric):
    """The area under the receiver operating characteristic (ROC) curve."""

    contents: str = 'sklearn.metrics.roc_auc_score'
    uses_probabilities: ClassVar[bool] = True


""" Regression Metrics """


@dataclasses.dataclass
class ExplainedVariance(Metric):
    """The share of the variance of the label that the model explains."""

    contents: str = 'sklearn.metrics.explained_variance_score'
    tasks: ClassVar[tuple[str, ...]] = ('regress',)


@dataclasses.dataclass
class MAE(Metric):
    """The mean absolute error (lower is better)."""

    contents: str = 'sklearn.metrics.mean_absolute_error'
    greater_is_better: ClassVar[bool] = False
    tasks: ClassVar[tuple[str, ...]] = ('regress',)


@dataclasses.dataclass
class MAPE(Metric):
    """The mean absolute percentage error (lower is better)."""

    contents: str = 'sklearn.metrics.mean_absolute_percentage_error'
    greater_is_better: ClassVar[bool] = False
    tasks: ClassVar[tuple[str, ...]] = ('regress',)


@dataclasses.dataclass
class MSE(Metric):
    """The mean squared error (lower is better)."""

    contents: str = 'sklearn.metrics.mean_squared_error'
    greater_is_better: ClassVar[bool] = False
    tasks: ClassVar[tuple[str, ...]] = ('regress',)


@dataclasses.dataclass
class R2(Metric):
    """The coefficient of determination (R squared)."""

    contents: str = 'sklearn.metrics.r2_score'
    tasks: ClassVar[tuple[str, ...]] = ('regress',)


@dataclasses.dataclass
class RMSE(Metric):
    """The root mean squared error (lower is better)."""

    contents: str = 'sklearn.metrics.root_mean_squared_error'
    greater_is_better: ClassVar[bool] = False
    tasks: ClassVar[tuple[str, ...]] = ('regress',)
