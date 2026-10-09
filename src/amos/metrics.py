"""Metrics that score a model's predictions.

These are techniques of the "critic" stage. Each one wraps a scoring function
(from `sklearn.metrics` or fairlearn), compares the model's
predictions (or predicted probabilities) with the true labels of the same
rows, and stores the score in the dataset's `metrics` under the technique's
name.

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
    GroupMetric: genre of fairness metrics that compare groups of rows.
    Accuracy, AveragePrecision, BalancedAccuracy, Brier, CohenKappa, F1,
        LogLoss, Matthews, Precision, Recall, RocAuc: classification metrics.
    DemographicParity, DemographicParityRatio, EqualOpportunity,
        EqualOpportunityRatio, EqualizedOdds, EqualizedOddsRatio: fairness
        metrics.
    Concordance, ExplainedVariance, MAE, MAPE, MSE, R2, RMSE: regression
        metrics.
    concordance_index: Harrell's concordance index of predicted times.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, ClassVar

import chrisjen
import numpy as np
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
                parameters.setdefault(
                    'labels', list(item.probabilities.columns))
        return parameters


@dataclasses.dataclass
class GroupMetric(Metric, abc.ABC):
    """Genre of fairness metrics, which compare a model across groups of rows.

    Each wraps a fairlearn metric. It compares the model's predictions for
    the groups in the column named by the "group" parameter, which defaults
    to the dataset's first `groups` column. The label must have two classes;
    the positive class is the last of `Dataset.classes`. A "difference" is
    the largest gap between groups (0 is equal treatment, and lower is
    better). A "ratio" is the smallest group's value divided by the largest
    (1 is equal treatment, and higher is better).

    """

    """ Private Methods """

    def _observed(self, item: base.Dataset) -> tuple[pd.Series, Any]:
        """Returns whether each true label and prediction is the positive class.

        Args:
            item: the dataset to score.

        Raises:
            ValueError: if the label does not have two classes.

        Returns:
            The true labels and the predictions, as 1 (positive) or 0.

        """
        y_true, y_pred = super()._observed(item)
        return _positive(item, y_true, y_pred, self.name)

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Adds the group of each predicted row.

        Args:
            item: the dataset to score.
            parameters: parameters for the scoring function, which may
                include "group" (the name of a column).

        Raises:
            ValueError: if no group is given and the dataset has no groups.

        Returns:
            The parameters, with "sensitive_features".

        """
        group = _group(item, parameters.pop('group', None), self.name)
        rows: Any = (
            [] if item.predictions is None else item.predictions.index)
        parameters['sensitive_features'] = item.data.loc[rows, group]
        return parameters


""" Classification Metrics """


@dataclasses.dataclass
class Accuracy(Metric):
    """The share of rows classified correctly.

    Wraps:
        [`sklearn.metrics.accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.accuracy_score'


@dataclasses.dataclass
class AveragePrecision(Metric):
    """The area under the precision-recall curve.

    Wraps:
        [`sklearn.metrics.average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.average_precision_score'
    uses_probabilities: ClassVar[bool] = True


@dataclasses.dataclass
class BalancedAccuracy(Metric):
    """The average share of each class classified correctly.

    Wraps:
        [`sklearn.metrics.balanced_accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.balanced_accuracy_score'


@dataclasses.dataclass
class Brier(Metric):
    """The mean squared error of the predicted probabilities (lower is better).

    Wraps:
        [`sklearn.metrics.brier_score_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.brier_score_loss'
    greater_is_better: ClassVar[bool] = False
    uses_probabilities: ClassVar[bool] = True


@dataclasses.dataclass
class CohenKappa(Metric):
    """Agreement between the predictions and labels beyond chance.

    Wraps:
        [`sklearn.metrics.cohen_kappa_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.cohen_kappa_score'


@dataclasses.dataclass
class F1(Metric):
    """The harmonic mean of precision and recall.

    Wraps:
        [`sklearn.metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.f1_score'


@dataclasses.dataclass
class LogLoss(Metric):
    """The negative log-likelihood of the true labels (lower is better).

    Wraps:
        [`sklearn.metrics.log_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.log_loss'
    greater_is_better: ClassVar[bool] = False
    uses_probabilities: ClassVar[bool] = True


@dataclasses.dataclass
class Matthews(Metric):
    """The Matthews correlation coefficient (phi for two classes).

    Wraps:
        [`sklearn.metrics.matthews_corrcoef`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.matthews_corrcoef'


@dataclasses.dataclass
class Precision(Metric):
    """The share of rows predicted to be positive that are positive.

    Wraps:
        [`sklearn.metrics.precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.precision_score'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'zero_division': 0})


@dataclasses.dataclass
class Recall(Metric):
    """The share of positive rows that are predicted to be positive.

    Wraps:
        [`sklearn.metrics.recall_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.recall_score'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'zero_division': 0})


@dataclasses.dataclass
class RocAuc(Metric):
    """The area under the receiver operating characteristic (ROC) curve.

    Wraps:
        [`sklearn.metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.roc_auc_score'
    uses_probabilities: ClassVar[bool] = True


""" Fairness Metrics """


@dataclasses.dataclass
class DemographicParity(GroupMetric):
    """The largest gap between groups in the share predicted to be positive.

    Wraps:
        [`fairlearn.metrics.demographic_parity_difference`](https://fairlearn.org/v0.14/api_reference/generated/fairlearn.metrics.demographic_parity_difference.html)
        from fairlearn.

    """

    contents: str = 'fairlearn.metrics.demographic_parity_difference'
    greater_is_better: ClassVar[bool] = False


@dataclasses.dataclass
class DemographicParityRatio(GroupMetric):
    """The smallest group's share predicted positive over the largest's.

    Wraps:
        [`fairlearn.metrics.demographic_parity_ratio`](https://fairlearn.org/v0.14/api_reference/generated/fairlearn.metrics.demographic_parity_ratio.html)
        from fairlearn.

    """

    contents: str = 'fairlearn.metrics.demographic_parity_ratio'


@dataclasses.dataclass
class EqualOpportunity(GroupMetric):
    """The largest gap between groups in the true positive rate.

    Wraps:
        [`fairlearn.metrics.equal_opportunity_difference`](https://fairlearn.org/v0.14/api_reference/generated/fairlearn.metrics.equal_opportunity_difference.html)
        from fairlearn.

    """

    contents: str = 'fairlearn.metrics.equal_opportunity_difference'
    greater_is_better: ClassVar[bool] = False


@dataclasses.dataclass
class EqualOpportunityRatio(GroupMetric):
    """The smallest group's true positive rate over the largest's.

    Wraps:
        [`fairlearn.metrics.equal_opportunity_ratio`](https://fairlearn.org/v0.14/api_reference/generated/fairlearn.metrics.equal_opportunity_ratio.html)
        from fairlearn.

    """

    contents: str = 'fairlearn.metrics.equal_opportunity_ratio'


@dataclasses.dataclass
class EqualizedOdds(GroupMetric):
    """The larger gap between groups in true or false positive rates.

    Wraps:
        [`fairlearn.metrics.equalized_odds_difference`](https://fairlearn.org/v0.14/api_reference/generated/fairlearn.metrics.equalized_odds_difference.html)
        from fairlearn.

    """

    contents: str = 'fairlearn.metrics.equalized_odds_difference'
    greater_is_better: ClassVar[bool] = False


@dataclasses.dataclass
class EqualizedOddsRatio(GroupMetric):
    """The smaller ratio between groups of true or false positive rates.

    Wraps:
        [`fairlearn.metrics.equalized_odds_ratio`](https://fairlearn.org/v0.14/api_reference/generated/fairlearn.metrics.equalized_odds_ratio.html)
        from fairlearn.

    """

    contents: str = 'fairlearn.metrics.equalized_odds_ratio'


""" Regression Metrics """


@dataclasses.dataclass
class Concordance(Metric):
    """How often the model orders pairs of times correctly (Harrell's C).

    This is the usual measure of a model of the time until an event, such as
    `cox`. Censored rows are compared only where their order is known. The
    "event" parameter names the column that is 1 if the event happened; it
    defaults to the event column of a `cox` model. 0.5 is chance, and 1 is
    perfect.

    """

    contents: str = 'amos.metrics.concordance_index'
    tasks: ClassVar[tuple[str, ...]] = ('regress',)

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Adds whether each predicted row's event happened.

        Args:
            item: the dataset to score.
            parameters: parameters for the scoring function, which may
                include "event" (the name of a column).

        Returns:
            The parameters, with "event_observed" if there is an event column.

        """
        event = parameters.pop('event', None) or getattr(
            item.model, 'event', None)
        if event is not None and item.predictions is not None:
            parameters['event_observed'] = item.data.loc[
                item.predictions.index, event].astype(int)
        return parameters


@dataclasses.dataclass
class ExplainedVariance(Metric):
    """The share of the variance of the label that the model explains.

    Wraps:
        [`sklearn.metrics.explained_variance_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.explained_variance_score'
    tasks: ClassVar[tuple[str, ...]] = ('regress',)


@dataclasses.dataclass
class MAE(Metric):
    """The mean absolute error (lower is better).

    Wraps:
        [`sklearn.metrics.mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.mean_absolute_error'
    greater_is_better: ClassVar[bool] = False
    tasks: ClassVar[tuple[str, ...]] = ('regress',)


@dataclasses.dataclass
class MAPE(Metric):
    """The mean absolute percentage error (lower is better).

    Wraps:
        [`sklearn.metrics.mean_absolute_percentage_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.mean_absolute_percentage_error'
    greater_is_better: ClassVar[bool] = False
    tasks: ClassVar[tuple[str, ...]] = ('regress',)


@dataclasses.dataclass
class MSE(Metric):
    """The mean squared error (lower is better).

    Wraps:
        [`sklearn.metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.mean_squared_error'
    greater_is_better: ClassVar[bool] = False
    tasks: ClassVar[tuple[str, ...]] = ('regress',)


@dataclasses.dataclass
class R2(Metric):
    """The coefficient of determination (R squared).

    Wraps:
        [`sklearn.metrics.r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.r2_score'
    tasks: ClassVar[tuple[str, ...]] = ('regress',)


@dataclasses.dataclass
class RMSE(Metric):
    """The root mean squared error (lower is better).

    Wraps:
        [`sklearn.metrics.root_mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.metrics.root_mean_squared_error'
    greater_is_better: ClassVar[bool] = False
    tasks: ClassVar[tuple[str, ...]] = ('regress',)


""" Public Functions """


def concordance_index(
    event_times: Any,
    predicted_scores: Any,
    event_observed: Any = None) -> float:
    """Returns Harrell's concordance index of predicted times.

    Two rows can be compared if the one with the shorter time had the event,
    or if they have the same time and only one of them had the event (the
    censored row is known to have lasted at least as long). Two rows with
    events at the same time are not compared. A pair is ordered correctly if
    the row that lasted longer has the higher prediction, and a tie in the
    predictions counts as half.

    Args:
        event_times: the time until the event or censoring of each row.
        predicted_scores: the predictions, which are higher for rows that are
            predicted to last longer (such as predicted times).
        event_observed: whether the event of each row happened (1) or the row
            was censored (0). Defaults to `None`, which means every event
            happened.

    Raises:
        ValueError: if no two rows can be compared.

    Returns:
        The share of the pairs that can be compared that are ordered
            correctly: 0.5 is chance, and 1 is perfect.

    """
    times = np.asarray(event_times, dtype = float)
    scores = np.asarray(predicted_scores, dtype = float)
    if event_observed is None:
        happened = np.ones(len(times), dtype = bool)
    else:
        happened = np.asarray(event_observed).astype(bool)
    pairs = 0
    correct = 0.0
    for row in np.flatnonzero(happened):
        later = (times > times[row]) | ((times == times[row]) & ~happened)
        pairs += int(later.sum())
        correct += (scores[later] > scores[row]).sum()
        correct += (scores[later] == scores[row]).sum() / 2
    if not pairs:
        message = 'no two rows can be compared (did any event happen?)'
        raise ValueError(message)
    return float(correct / pairs)


""" Private Functions """


def _group(item: base.Dataset, group: str | None, name: str) -> str:
    """Returns the name of the column of groups to compare.

    Args:
        item: the dataset.
        group: name of the column chosen by the user, or `None`.
        name: name of the technique, for the error message.

    Raises:
        ValueError: if no group is given and the dataset has no groups.

    Returns:
        `group`, or the dataset's first group.

    """
    if group is None and item.groups:
        group = item.groups[0]
    if group is None:
        message = (
            f'{name!r} needs a group: set "groups" in the "general" section '
            f'of the settings or pass "group"'
        )
        raise ValueError(message)
    return group


def _positive(
    item: base.Dataset,
    y_true: pd.Series,
    y_pred: Any,
    name: str) -> tuple[pd.Series, pd.Series]:
    """Returns whether each true label and prediction is the positive class.

    Args:
        item: the dataset.
        y_true: the true labels.
        y_pred: the predictions of the same rows.
        name: name of the technique, for the error message.

    Raises:
        ValueError: if the label does not have two classes.

    Returns:
        The true labels and the predictions, as 1 (positive) or 0.

    """
    classes = item.classes
    if len(classes) != 2:  # noqa: PLR2004
        message = f'{name!r} needs a label with two classes'
        raise ValueError(message)
    positive = classes[-1]
    predicted = pd.Series(np.asarray(y_pred), index = y_true.index)
    return (y_true == positive).astype(int), (predicted == positive).astype(int)
