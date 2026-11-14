"""The scorecard of every branch of an experiment.

Contents:
    Scorecard: every standard metric for every branch of an analysis, ready to
        publish as csv, Markdown, LaTeX, HTML, Word, or an image.

"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any, ClassVar

import chrisjen
import pandas as pd

from .. import base
from .. import metrics as metrics_
from ._base import Evaluator, _Published


@dataclasses.dataclass
class Scorecard(_Published, Evaluator):
    """The results of every branch of an analysis, ready to publish.

    A scorecard has one row for each branch of the most recent `experiment`
    (each combination of techniques that it tried): its rank, the technique
    used at each step, and every standard metric for the task, computed from
    that branch's own predictions. If the data did not come from an
    experiment, it has one row for the final model and the techniques that
    made it. The table is stored in the dataset's `tables` under the
    technique's name, and the final model's metrics are stored in its
    `metrics`.

    A scorecard can be saved as a csv file (`to_csv`), a Markdown table
    (`to_markdown`), a LaTeX table (`to_latex`), an HTML table (`to_html`,
    which needs great_tables), a Word document (`to_word`, which needs
    python-docx), or an image (`to_image`, which needs matplotlib), or in
    several at once (`export`). If the dataset has groups and a label of two
    classes, fairness metrics are included. Make one from a dataset or an
    applied project with `create`, or use `Project.scorecard`.

    Wraps:
        - [`great_tables.GT`](https://posit-dev.github.io/great-tables/reference/GT.html)
          from great_tables, for `to_html`.
        - [`docx.Document`](https://python-docx.readthedocs.io/en/latest/api/document.html#docx.Document)
          from python-docx, for `to_word`.
        - [`matplotlib.axes.Axes.table`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.table.html)
          from matplotlib, for `to_image`.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, in which case it is "scorecard".
        contents: not used. Defaults to `None`.
        parameters: keyword arguments for `evaluate`, such as "metrics".
            Defaults to an empty `dict`.
        title: title of the scorecard in Word documents and images. Defaults
            to `None`, in which case it describes the branches (such as "8
            branches ranked by roc_auc").
        digits: digits after the decimal point when scores are written as
            text. Defaults to 3.

    Attributes:
        table: the most recent scorecard. Its columns are "rank", the steps,
            and the metrics.
        summary: a description of the most recent scorecard.

    """

    visual: ClassVar[str] = 'scorecard_plot'

    title: str | None = None
    digits: int = 3
    table: pd.DataFrame | None = dataclasses.field(default = None, repr = False)
    summary: str | None = dataclasses.field(default = None, repr = False)

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
    # Fairness metrics in a scorecard of a dataset with groups and a label of
    # two classes.
    fairness: ClassVar[tuple[str, ...]] = (
        'demographic_parity',
        'equalized_odds')
    # Genres of the techniques that are listed as steps of a dataset that did
    # not come from an experiment.
    step_genres: ClassVar[tuple[str, ...]] = (
        'splitter',
        'imputer',
        'scaler',
        'encoder',
        'mixer',
        'selector',
        'reducer',
        'sampler',
        'model',
        'validator')
    # What the table is called in the message of an empty table.
    noun: ClassVar[str] = 'scorecard'

    """ Class Methods """

    @classmethod
    def create(
        cls,
        item: base.Dataset | chrisjen.Project,
        **kwargs: Any) -> Scorecard:
        """Returns the scorecard of a dataset or of an applied project.

        Args:
            item: a `Dataset`, or a `Project` that has been applied.
            **kwargs: arguments for the scorecard (such as `title` and
                `digits`) and for `evaluate` (such as `metrics`).

        Raises:
            TypeError: if `item` is not a `Dataset` or an applied project.

        Returns:
            A scorecard, with its `table` filled in.

        """
        if isinstance(item, chrisjen.Project):
            item = item.result
        if not isinstance(item, base.Dataset):
            message = (
                f'a scorecard needs a Dataset or an applied Project, not '
                f'{type(item).__name__}'
            )
            raise TypeError(message)
        fields = {field.name for field in dataclasses.fields(cls)}
        scorecard = cls(**{k: v for k, v in kwargs.items() if k in fields})
        scorecard.evaluate(
            item, **{k: v for k, v in kwargs.items() if k not in fields})
        return scorecard

    @classmethod
    def metric_names(
        cls,
        item: base.Dataset,
        branches: Sequence[base.Branch] = (),
        metrics: Sequence[str] | None = None) -> list[str]:
        """Returns the names of the metrics that a scorecard of `item` has.

        Validators score each fold with these metrics too, unless they are
        given their own.

        Args:
            item: the dataset with a fitted model.
            branches: the branches in the scorecard. Defaults to an empty
                tuple.
            metrics: names of metrics chosen by the user. Defaults to `None`,
                which uses `defaults` for the task (and `fairness`, for a
                binary classifier with groups).

        Returns:
            `metrics`, or `defaults` for the task with the experiment's
                criterion first (if it is a metric).

        """
        if metrics:
            return list(metrics)
        names = list(cls.defaults.get(item.task or 'classify', ()))
        if (
            item.groups
            and item.task == 'classify'
            and len(item.classes) == 2):  # noqa: PLR2004
            names.extend(cls.fairness)
        criterion = branches[0].criterion if branches else None
        kind = chrisjen.library.all.get(criterion) if criterion else None
        if (
            criterion is not None
            and isinstance(kind, type)
            and issubclass(kind, metrics_.Metric)):
            names = [criterion, *(n for n in names if n != criterion)]
        return names

    """ Properties """

    @property
    def heading(self) -> str:
        """Returns `title`, or `summary` if there is no title."""
        return self.title or self.summary or 'Scorecard'

    """ Public Methods """

    def evaluate(
        self,
        item: base.Dataset,
        metrics: Sequence[str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the scorecard of `item`, which is also stored in `table`.

        Args:
            item: the dataset with a fitted model.
            metrics: names of the metrics to use. Defaults to `None`, which
                uses `defaults` for the task (with the experiment's criterion
                first, and the `fairness` metrics after them if the dataset
                has groups and two classes). Other metrics that the branches
                computed are added after them.
            **kwargs: not used.

        Returns:
            One row for each branch, best first, with a "rank" column, a
                column for each step, and a column for each metric.

        """
        branches = item.branches or [base.Branch(
            label = 'result', steps = self._history_steps(item), result = item)]
        names = self.metric_names(item, branches, metrics)
        extras = list(dict.fromkeys(
            name for branch in branches for name in branch.result.metrics
            if name not in names))
        rows = []
        for branch in branches:
            row: dict[str, Any] = dict(branch.steps)
            row.update(self._measure(branch.result, names))
            row.update({
                name: branch.result.metrics[name] for name in extras
                if name in branch.result.metrics})
            rows.append(row)
        steps = list(dict.fromkeys(k for b in branches for k in b.steps))
        table = pd.DataFrame(rows)
        columns = steps + [n for n in [*names, *extras] if n in table.columns]
        table = table.reindex(columns = columns)
        criterion = branches[0].criterion
        scores = [branch.score for branch in branches]
        if any(score is not None for score in scores):
            if criterion not in table.columns:
                table['score'] = scores
            # Sorts from best to worst. Ties keep the order of the branches.
            order = sorted(range(len(scores)), key = lambda i: _worst_first(
                scores[i]))
            table = table.iloc[order].reset_index(drop = True)
        table.insert(0, 'rank', range(1, len(table) + 1))
        self.table = table
        if item.branches:
            ranked = f' ranked by {criterion}' if criterion else ''
            self.summary = f'{len(branches)} branches{ranked}'
        else:
            self.summary = 'The final model'
        return table

    def apply(
        self,
        item: base.Dataset,
        metrics: Sequence[str] | None = None,
        **kwargs: Any) -> base.Dataset:
        """Adds the scorecard to `item` and the final model's metrics.

        Args:
            item: the dataset with a fitted model.
            metrics: names of the metrics to use. Defaults to `None`, which
                uses `defaults` for the task.
            **kwargs: other parameters for `evaluate`, and `visualize` (see
                `Evaluator.apply`).

        Returns:
            The dataset, with the scorecard in `tables` and the final model's
                scores in `metrics`.

        """
        names = list(metrics or self.defaults.get(item.task or 'classify', ()))
        item.metrics.update(self._measure(item, names))
        return super().apply(item, metrics = metrics, **kwargs)

    """ Private Methods """

    def _history_steps(self, item: base.Dataset) -> dict[str, str]:
        """Returns the techniques that made a dataset, by their genres.

        Args:
            item: a dataset that did not come from an experiment.

        Returns:
            The names of the techniques in `history` whose genres are in
                `steps`, by genre. A technique that is not in
                `chrisjen.library` is placed by the "genre" that it
                recorded, if any. Several techniques of one genre are joined
                with commas.

        """
        found: dict[str, str] = {}
        for entry in item.history:
            if entry.get('skipped'):
                continue
            technique = str(entry.get('technique'))
            try:
                genre = chrisjen.library.classify(technique)
            except ValueError:
                # A technique that is only in a project's own library (such
                # as a feature group) may record its genre.
                genre = str(entry.get('genre'))
            if genre in self.step_genres:
                found[genre] = (
                    f'{found[genre]}, {technique}' if genre in found
                    else technique)
        return {
            genre: found[genre] for genre in self.step_genres if genre in found}

    def _measure(
        self,
        item: base.Dataset,
        names: Sequence[str]) -> dict[str, float]:
        """Returns the score of each metric in `names` that applies to `item`.

        Args:
            item: a dataset with predictions.
            names: names of metrics in the library.

        Raises:
            TypeError: if a name is not the name of a metric.

        Returns:
            The scores. Metrics for another task, metrics that need
                probabilities that the model did not make, and every metric
                of a dataset without predictions are left out.

        """
        values = {}
        for name in names:
            metric = chrisjen.library.borrow(name, genre = 'metric')()
            if not isinstance(metric, metrics_.Metric):
                message = f'{name!r} is not a metric'
                raise TypeError(message)
            if (
                item.predictions is None
                or item.task not in metric.tasks
                or (metric.uses_probabilities and item.probabilities is None)):
                continue
            values[name] = metric.measure(item)
        return values

    def _shaded(self) -> int | None:
        """Returns the position of the best branch, if there are several.

        Returns:
            The position (from 0) of the row of the best branch, or
                `None` if there is only one.

        """
        return 0 if len(self._require_table()) > 1 else None


""" Private Functions """


def _worst_first(score: float | None) -> float:
    """Returns a key that sorts scores from best to worst.

    Args:
        score: a score (higher is better), or `None`.

    Returns:
        The negative of `score`, or infinity if there is no score (so that
            branches without scores come last).

    """
    return math.inf if score is None else -score
