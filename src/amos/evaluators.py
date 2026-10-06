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
    Scorecard: every standard metric for every branch of an analysis, ready
        to publish as csv, Markdown, Word, or an image.
    ShapImportance: the mean absolute SHAP value of each feature.

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
import math
import pathlib
from collections.abc import Iterable, Sequence
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
    (`to_markdown`), a Word document (`to_word`, which needs python-docx), or
    an image (`to_image`, which needs matplotlib), or in all of them at once
    (`export`). Make one from a dataset or an applied project with `create`,
    or use `Project.scorecard`.

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
    # Genres of the techniques that are listed as steps of a dataset that did
    # not come from an experiment.
    step_genres: ClassVar[tuple[str, ...]] = (
        'splitter',
        'imputer',
        'scaler',
        'encoder',
        'mixer',
        'reducer',
        'sampler',
        'model')

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
                first). Other metrics that the branches computed are added
                after them.
            **kwargs: not used.

        Returns:
            One row for each branch, best first, with a "rank" column, a
                column for each step, and a column for each metric.

        """
        branches = item.branches or [base.Branch(
            label = 'result', steps = self._history_steps(item), result = item)]
        names = self._metric_names(item, branches, metrics)
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

    def export(
        self,
        folder: pathlib.Path | str,
        *,
        name: str | None = None,
        formats: Sequence[str] = ('csv', 'md', 'docx', 'png')) -> dict[str, pathlib.Path]:
        """Saves the scorecard in several formats.

        Args:
            folder: folder to save the files in. It is created if needed.
            name: name of the files, without extensions. Defaults to `None`,
                in which case the name of the technique is used.
            formats: extensions of the files to save: "csv", "md", "docx",
                and any image format that matplotlib saves (such as "png",
                "svg", or "pdf"). Defaults to csv, md, docx, and png.

        Returns:
            The path of each file, by its format.

        """
        folder = pathlib.Path(folder)
        folder.mkdir(parents = True, exist_ok = True)
        paths = {}
        for extension in formats:
            path = folder / f'{name or self.name}.{extension}'
            if extension == 'csv':
                self.to_csv(path)
            elif extension == 'md':
                self.to_markdown(path)
            elif extension == 'docx':
                self.to_word(path)
            else:
                self.to_image(path)
            paths[extension] = path
        return paths

    def implement(
        self,
        item: base.Dataset,
        metrics: Sequence[str] | None = None,
        **kwargs: Any) -> base.Dataset:
        """Adds the scorecard to `item` and the final model's metrics.

        Args:
            item: the dataset with a fitted model.
            metrics: names of the metrics to use. Defaults to `None`, which
                uses `defaults` for the task.
            **kwargs: other parameters for `evaluate`.

        Returns:
            The dataset, with the scorecard in `tables` and the final model's
                scores in `metrics`.

        """
        names = list(metrics or self.defaults.get(item.task or 'classify', ()))
        item.metrics.update(self._measure(item, names))
        return super().implement(item, metrics = metrics, **kwargs)

    def to_csv(self, path: pathlib.Path | str | None = None) -> str:
        """Returns the scorecard as csv text, and saves it if `path` is given.

        Args:
            path: file to save the text in. Defaults to `None`.

        Returns:
            The csv text. Scores are saved at full precision.

        """
        text = self._require_table().to_csv(index = False, lineterminator = '\n')
        if path is not None:
            pathlib.Path(path).write_text(text, encoding = 'utf-8')
        return text

    def to_figure(self) -> Any:
        """Returns the scorecard drawn as a table in a `matplotlib` figure.

        The header is bold, and the best branch is shaded.

        Returns:
            A `matplotlib.figure.Figure`.

        """
        figures = utilities.import_tool('matplotlib.figure')
        text = self._text()
        rows = len(text)
        lengths = [
            max([len(str(c)), *(len(v) for v in text[c])]) + 2
            for c in text.columns]
        width = max(4.0, 0.085 * sum(lengths))
        height = 0.3 * (rows + 1) + 0.5
        figure = figures.Figure(figsize = (width, height))
        axes = figure.add_axes((0, 0, 1, 1))
        axes.set_axis_off()
        grid = axes.table(
            cellText = text.to_numpy(),
            colLabels = list(text.columns),
            colWidths = [length / sum(lengths) for length in lengths],
            cellLoc = 'center',
            bbox = (0, 0, 1, (rows + 1) / (rows + 1 + 1.6)))
        grid.auto_set_font_size(value = False)
        grid.set_fontsize(9)
        for (row, _), cell in grid.get_celld().items():
            cell.set_edgecolor('#bbbbbb')
            if row == 0:
                cell.set_text_props(weight = 'bold')
                cell.set_facecolor('#e6e6e6')
            elif row == 1 and rows > 1:
                cell.set_facecolor('#e3f1e3')
        figure.text(
            0.5, 0.98, self.heading,
            ha = 'center', va = 'top', fontsize = 11, weight = 'bold')
        return figure

    def to_image(
        self,
        path: pathlib.Path | str,
        dpi: int = 200) -> pathlib.Path:
        """Saves the scorecard as an image.

        Args:
            path: file to save the image in. Its extension (such as ".png",
                ".svg", or ".pdf") sets the format.
            dpi: dots per inch of an image with pixels. Defaults to 200.

        Returns:
            The path of the image.

        """
        path = pathlib.Path(path)
        self.to_figure().savefig(path, dpi = dpi, bbox_inches = 'tight')
        return path

    def to_markdown(self, path: pathlib.Path | str | None = None) -> str:
        """Returns the scorecard as a Markdown table, and saves it if asked.

        Numbers are aligned to the right.

        Args:
            path: file to save the table in. Defaults to `None`.

        Returns:
            The Markdown table.

        """
        text = self._text()
        numeric = self._numeric_columns()
        lines = [
            _markdown_row(str(c) for c in text.columns),
            _markdown_row(
                '---:' if c in numeric else '---' for c in text.columns)]
        lines.extend(
            _markdown_row(row) for row in text.itertuples(index = False))
        markdown = '\n'.join(lines) + '\n'
        if path is not None:
            pathlib.Path(path).write_text(markdown, encoding = 'utf-8')
        return markdown

    def to_word(self, path: pathlib.Path | str) -> pathlib.Path:
        """Saves the scorecard as a table in a Word document.

        The document has the scorecard's heading and a table in the "Table
        Grid" style, with a bold header and numbers aligned to the right.

        Args:
            path: file to save the document in (usually ending in ".docx").

        Returns:
            The path of the document.

        """
        docx = utilities.import_tool('docx')
        alignment = utilities.import_tool('docx.enum.text.WD_ALIGN_PARAGRAPH')
        text = self._text()
        numeric = self._numeric_columns()
        document = docx.Document()
        document.add_heading(self.heading, level = 2)
        grid = document.add_table(rows = len(text) + 1, cols = len(text.columns))
        grid.style = 'Table Grid'
        for column, label in enumerate(text.columns):
            cell = grid.cell(0, column)
            cell.text = str(label)
            for run in cell.paragraphs[0].runs:
                run.font.bold = True
        for row, values in enumerate(text.itertuples(index = False), start = 1):
            for column, value in enumerate(values):
                cell = grid.cell(row, column)
                cell.text = value
                if text.columns[column] in numeric:
                    cell.paragraphs[0].alignment = alignment.RIGHT
        path = pathlib.Path(path)
        document.save(str(path))
        return path

    """ Private Methods """

    def _history_steps(self, item: base.Dataset) -> dict[str, str]:
        """Returns the techniques that made a dataset, by their genres.

        Args:
            item: a dataset that did not come from an experiment.

        Returns:
            The names of the techniques in `history` whose genres are in
                `steps`, by genre. Several techniques of one genre are
                joined with commas.

        """
        found: dict[str, str] = {}
        for entry in item.history:
            technique = str(entry.get('technique'))
            try:
                genre = chrisjen.library.classify(technique)
            except ValueError:
                continue
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
            values[name] = metric.measure(item, **metric._keywords())
        return values

    def _metric_names(
        self,
        item: base.Dataset,
        branches: Sequence[base.Branch],
        metrics: Sequence[str] | None) -> list[str]:
        """Returns the names of the metrics to put in the scorecard.

        Args:
            item: the dataset with a fitted model.
            branches: the branches in the scorecard.
            metrics: names of metrics chosen by the user, or `None`.

        Returns:
            `metrics`, or `defaults` for the task with the experiment's
                criterion first (if it is a metric).

        """
        if metrics:
            return list(metrics)
        names = list(self.defaults.get(item.task or 'classify', ()))
        criterion = branches[0].criterion if branches else None
        kind = chrisjen.library.all.get(criterion) if criterion else None
        if (
            criterion is not None
            and isinstance(kind, type)
            and issubclass(kind, metrics_.Metric)):
            names = [criterion, *(n for n in names if n != criterion)]
        return names

    def _numeric_columns(self) -> set[str]:
        """Returns the names of the columns of `table` that hold numbers."""
        table = self._require_table()
        return {
            c for c in table.columns
            if pd.api.types.is_numeric_dtype(table[c].dtype)}

    def _require_table(self) -> pd.DataFrame:
        """Returns `table`.

        Raises:
            ValueError: if there is no table yet.

        Returns:
            The most recent scorecard.

        """
        if self.table is None:
            message = (
                'the scorecard is empty: make one with Scorecard.create or '
                'apply it to a dataset first'
            )
            raise ValueError(message)
        return self.table

    def _text(self) -> pd.DataFrame:
        """Returns `table` with every value written as text.

        Returns:
            The table, with scores rounded to `digits` and missing values
                shown as a dash.

        """
        digits = self.digits

        def write(value: Any) -> str:
            if isinstance(value, str):
                return value
            if value is None or pd.isna(value):
                return '—'
            if isinstance(value, bool | int | np.integer):
                return str(value)
            if isinstance(value, float | np.floating):
                return f'{value:.{digits}f}'
            return str(value)

        return self._require_table().map(write)


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


def _worst_first(score: float | None) -> float:
    """Returns a key that sorts scores from best to worst.

    Args:
        score: a score (higher is better), or `None`.

    Returns:
        The negative of `score`, or infinity if there is no score (so that
            branches without scores come last).

    """
    return math.inf if score is None else -score


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


def _markdown_row(values: Iterable[str]) -> str:
    """Returns a row of a Markdown table.

    Args:
        values: the text of each cell.

    Returns:
        The row, with any "|" in the text escaped.

    """
    cells = [str(value).replace('|', r'\|') for value in values]
    return f'| {" | ".join(cells)} |'


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
