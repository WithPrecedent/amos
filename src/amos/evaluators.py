"""Techniques that evaluate a fitted model with tables.

These are techniques of the "critic" stage, beside the metrics. Each one adds
a table to the dataset's `tables` (under the technique's name). Most evaluate
the model on the test rows (or on every row, if the data has not been split).
`factor_analysis` and `pca` instead describe the features
that the model learned from, so they work in any stage.

Contents:
    Evaluator: base class for techniques that evaluate a model with a table.
    BiasVariance: the model's expected error, as its bias and variance.
    ClassificationReport: precision, recall, and f1 for each class.
    Confusion: how many rows of each class were predicted to be each class.
    Conformal: prediction intervals (or sets) with a known rate of coverage.
    ExplainWeights: eli5's explanation of the weights of the features.
    FactorAnalysis: the hidden factors that explain the correlations of the
        features.
    Fairness: how the model does for each group, and the gaps between them.
    FeatureImportance: the importance that the model gives each feature.
    PCA: how much of the variance of the features each principal component
        has.
    PermutationImportance: how much the score drops when each feature is
        shuffled.
    RegressionTable: the coefficients (or average marginal effects) of every
        model side by side, with their standard errors and cross-validated
        metrics, ready to publish like a scorecard.
    Scorecard: every standard metric for every branch of an analysis, ready
        to publish as csv, Markdown, LaTeX, HTML, Word, or an image.
    ShapImportance: the mean absolute SHAP value of each feature.

"""

from __future__ import annotations

import abc
import dataclasses
import functools
import importlib
import itertools
import math
import pathlib
import textwrap
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar

import camina
import chrisjen
import numpy as np
import pandas as pd

from . import base, models, utilities
from . import metrics as metrics_

# The columns of the estimates of a regression table.
_ESTIMATES: tuple[str, ...] = (
    'estimate', 'standard_error', 'statistic', 'p_value', 'ci_lower',
    'ci_upper')
# The width of a character of DejaVu Sans Mono, matplotlib's monospaced font,
# as a share of the size of the font.
_MONOSPACE: float = 0.6021
# The widths of the columns of numbers in Stata's output of `regress`: the
# estimate, standard error, test statistic, p-value, and the two bounds of
# the confidence interval.
_STATA_COLUMNS: tuple[int, ...] = (11, 11, 9, 8, 13, 12)
# The names that Stata (or a reader of a regression table) uses for metrics.
_STATA_NAMES: dict[str, str] = {
    'accuracy': 'Accuracy',
    'balanced_accuracy': 'Bal. accuracy',
    'f1': 'F1',
    'log_loss': 'Log loss',
    'mae': 'MAE',
    'mse': 'MSE',
    'precision': 'Precision',
    'r2': 'R-squared',
    'recall': 'Recall',
    'rmse': 'Root MSE',
    'roc_auc': 'ROC AUC'}

# Replacements for the characters that LaTeX treats specially. Each character
# is replaced once, so the braces in a replacement are not escaped again.
_LATEX_ESCAPES: dict[int, str] = str.maketrans({
    '\\': r'\textbackslash{}',
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
    '<': r'\textless{}',
    '>': r'\textgreater{}'})


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

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
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


class _Published:
    """Saves a table as csv, Markdown, LaTeX, HTML, Word, or an image.

    A mixin for the evaluators that make a table to publish (`Scorecard` and
    `RegressionTable`). The class that uses it has a `table`, the `digits`
    shown after the decimal point, a `heading`, and a `name` (as every
    technique does), and it can
    override the hooks that shade a row (`_shaded`), draw a rule above rows
    (`_rules`), and add notes below the table (`_notes`).

    """

    digits: int
    table: pd.DataFrame | None
    # What the table is called in the message of an empty table.
    noun: ClassVar[str] = 'table'

    """ Properties """

    @property
    def heading(self) -> str:
        """Returns the title of the table."""
        raise NotImplementedError

    """ Public Methods """

    def export(
        self,
        folder: pathlib.Path | str,
        *,
        name: str | None = None,
        formats: Sequence[str] = ('csv', 'md', 'docx', 'png'),
        ) -> dict[str, pathlib.Path]:
        """Saves the table in several formats.

        Args:
            folder: folder to save the files in. It is created if needed.
            name: name of the files, without extensions. Defaults to `None`,
                in which case the name of the technique is used.
            formats: extensions of the files to save: "csv", "md", "tex",
                "html", "docx", and any image format that matplotlib saves
                (such as "png", "svg", or "pdf"). Defaults to csv, md, docx,
                and png.

        Returns:
            The path of each file, by its format.

        """
        folder = pathlib.Path(folder)
        folder.mkdir(parents = True, exist_ok = True)
        paths = {}
        for extension in formats:
            path = folder / f'{name or getattr(self, "name", self.noun)}.{extension}'
            if extension == 'csv':
                self.to_csv(path)
            elif extension == 'md':
                self.to_markdown(path)
            elif extension == 'tex':
                self.to_latex(path)
            elif extension == 'html':
                self.to_html(path)
            elif extension == 'docx':
                self.to_word(path)
            else:
                self.to_image(path)
            paths[extension] = path
        return paths

    def to_csv(self, path: pathlib.Path | str | None = None) -> str:
        """Returns the table as csv text, and saves it if `path` is given.

        Args:
            path: file to save the text in. Defaults to `None`.

        Returns:
            The csv text.

        """
        text = self._csv_table().to_csv(index = False, lineterminator = '\n')
        if path is not None:
            pathlib.Path(path).write_text(text, encoding = 'utf-8')
        return text

    def to_figure(self) -> Any:
        """Returns the table drawn in a `matplotlib` figure.

        The header is bold, and any shaded row (such as the best branch of a
        scorecard) is shaded. Notes are written below the table.

        Returns:
            A `matplotlib.figure.Figure`.

        """
        figures = utilities.import_tool('matplotlib.figure')
        text = self._text()
        notes = self._notes()
        shaded = self._shaded()
        rows = len(text)
        lengths = [
            max([len(str(c)), *(len(v) for v in text[c])]) + 2
            for c in text.columns]
        width = max(4.0, 0.085 * sum(lengths))
        extra = 0.3 if notes else 0.0
        height = 0.3 * (rows + 1) + 0.5 + extra
        figure = figures.Figure(figsize = (width, height))
        axes = figure.add_axes((0, 0, 1, 1))
        axes.set_axis_off()
        bottom = extra / height
        grid = axes.table(
            cellText = text.to_numpy(),
            colLabels = list(text.columns),
            colWidths = [length / sum(lengths) for length in lengths],
            cellLoc = 'center',
            bbox = (
                0,
                bottom,
                1,
                (1 - bottom) * (rows + 1) / (rows + 1 + 1.6)))
        grid.auto_set_font_size(value = False)
        grid.set_fontsize(9)
        for (row, _), cell in grid.get_celld().items():
            cell.set_edgecolor('#bbbbbb')
            if row == 0:
                cell.set_text_props(weight = 'bold')
                cell.set_facecolor('#e6e6e6')
            elif shaded is not None and row == shaded + 1:
                cell.set_facecolor('#e3f1e3')
        figure.text(
            0.5, 0.98, self.heading,
            ha = 'center', va = 'top', fontsize = 11, weight = 'bold')
        if notes:
            figure.text(
                0.0, bottom / 2, notes, ha = 'left', va = 'center', fontsize = 8)
        return figure

    def to_html(self, path: pathlib.Path | str | None = None) -> str:
        """Returns the table as HTML, and saves it if `path` is given.

        The table is made by great_tables, with the heading as its title,
        numbers aligned to the right, any shaded row shaded, and the notes
        as a source note.

        Args:
            path: file to save the table in. Defaults to `None`.

        Returns:
            The HTML.

        """
        great_tables = utilities.import_tool('great_tables')
        text = self._text()
        numeric = [c for c in text.columns if c in self._numeric_columns()]
        table = great_tables.GT(text).tab_header(title = self.heading)
        if numeric:
            table = table.cols_align(align = 'right', columns = numeric)
        shaded = self._shaded()
        if shaded is not None:
            table = table.tab_style(
                style = great_tables.style.fill(color = '#e3f1e3'),
                locations = great_tables.loc.body(rows = [shaded]))
        notes = self._notes()
        if notes:
            table = table.tab_source_note(source_note = notes)
        html = str(table.as_raw_html())
        if path is not None:
            pathlib.Path(path).write_text(html, encoding = 'utf-8')
        return html

    def to_image(
        self,
        path: pathlib.Path | str,
        dpi: int = 200) -> pathlib.Path:
        """Saves the table as an image.

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

    def to_latex(self, path: pathlib.Path | str | None = None) -> str:
        r"""Returns the table as LaTeX, and saves it if `path` is given.

        The table uses the booktabs package, so add `\usepackage{booktabs}`
        to the preamble of the document. Its caption is the heading, and the
        notes are a row below the bottom rule.

        Args:
            path: file to save the table in. Defaults to `None`.

        Returns:
            The LaTeX table.

        """
        text = self._text()
        numeric = self._numeric_columns()
        rules = set(self._rules())
        alignment = ''.join('r' if c in numeric else 'l' for c in text.columns)
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            rf'\caption{{{_latex(self.heading)}}}',
            rf'\begin{{tabular}}{{{alignment}}}',
            r'\toprule',
            ' & '.join(_latex(str(c)) for c in text.columns) + r' \\',
            r'\midrule']
        for position, row in enumerate(text.itertuples(index = False)):
            if position in rules:
                lines.append(r'\midrule')
            lines.append(' & '.join(_latex(value) for value in row) + r' \\')
        lines.append(r'\bottomrule')
        notes = self._notes()
        if notes:
            lines.append(
                rf'\multicolumn{{{len(text.columns)}}}{{l}}'
                rf'{{\footnotesize {_latex(notes)}}} \\')
        lines.extend([r'\end{tabular}', r'\end{table}'])
        latex = '\n'.join(lines) + '\n'
        if path is not None:
            pathlib.Path(path).write_text(latex, encoding = 'utf-8')
        return latex

    def to_markdown(self, path: pathlib.Path | str | None = None) -> str:
        """Returns the table as Markdown, and saves it if `path` is given.

        Numbers are aligned to the right, and the notes are a paragraph
        below the table.

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
        notes = self._notes()
        if notes:
            lines.extend(['', notes])
        markdown = '\n'.join(lines) + '\n'
        if path is not None:
            pathlib.Path(path).write_text(markdown, encoding = 'utf-8')
        return markdown

    def to_word(self, path: pathlib.Path | str) -> pathlib.Path:
        """Saves the table in a Word document.

        The document has the heading, a table in the "Table Grid" style
        (with a bold header and numbers aligned to the right), and the notes
        below it.

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
        grid = document.add_table(
            rows = len(text) + 1, cols = len(text.columns))
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
        notes = self._notes()
        if notes:
            document.add_paragraph(notes)
        path = pathlib.Path(path)
        document.save(str(path))
        return path

    """ Private Methods """

    def _csv_table(self) -> pd.DataFrame:
        """Returns the table that `to_csv` saves: `table` itself."""
        return self._require_table()

    def _notes(self) -> str | None:
        """Returns notes to write below the table, or `None`."""
        return None

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
            The most recent table.

        """
        if self.table is None:
            message = (
                f'the {self.noun} is empty: make one with '
                f'{type(self).__name__}.create or apply it to a dataset first'
            )
            raise ValueError(message)
        return self.table

    def _rules(self) -> list[int]:
        """Returns the positions of the rows to draw a rule above."""
        return []

    def _shaded(self) -> int | None:
        """Returns the position of the row to shade, or `None`."""
        return None

    def _text(self) -> pd.DataFrame:
        """Returns `table` with every value written as text.

        Returns:
            The table, with numbers rounded to `digits` and missing values
                shown as a dash.

        """
        return self._require_table().map(
            functools.partial(_written, digits = self.digits))


@dataclasses.dataclass
class BiasVariance(Evaluator):
    """The model's expected error on the test rows, as its bias and variance.

    mlxtend's bias-variance decomposition: copies of the model are fitted to
    `rounds` bootstrap samples of the training rows, and each predicts the
    test rows. The expected loss is their mean loss: the share of rows
    misclassified (0-1 loss), for a classifier, or the mean squared error.
    The bias is the loss of their main prediction (the most common class, or
    the mean), and the variance is how far the copies' predictions are from
    it. A model with a large bias underfits; one with a large variance
    overfits. The table has one row, with the "loss", "expected_loss",
    "bias", and "variance".

    Wraps:
        [`mlxtend.evaluate.bias_variance_decomp`](https://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/)
        from mlxtend.

    """

    def evaluate(
        self,
        item: base.Dataset,
        rounds: int = 100,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the expected loss of the model, and its bias and variance.

        Args:
            item: the dataset with a fitted scikit-learn model and test rows.
            rounds: number of bootstrap samples (and copies of the model).
                Defaults to 100.
            **kwargs: not used.

        Raises:
            TypeError: if the model is not a scikit-learn estimator.
            ValueError: if the data is not split.

        Returns:
            One row, with the loss and its parts.

        """
        evaluate = utilities.import_tool('mlxtend.evaluate')
        sklearn = importlib.import_module('sklearn.base')
        if not item.is_split:
            message = 'bias_variance needs test rows: split the data first'
            raise ValueError(message)
        model = _model(item)
        names = list(
            utilities.feature_names(model, item.data.columns) or item.features)
        y_train, y_test = item.y_train, item.y_test
        if isinstance(model, models.LabelCoded):
            model = model.estimator
        if item.task == 'classify':
            # mlxtend counts the predictions of each class by its number.
            positions = {c: i for i, c in enumerate(item.classes)}
            y_train, y_test = y_train.map(positions), y_test.map(positions)
        if not isinstance(model, sklearn.BaseEstimator):
            message = (
                f'bias_variance needs a scikit-learn model, not a '
                f'{type(model).__name__}')
            raise TypeError(message)
        loss = '0-1_loss' if item.task == 'classify' else 'mse'
        expected, bias, variance = evaluate.bias_variance_decomp(
            sklearn.clone(model),
            item.x_train[names].to_numpy(),
            y_train.to_numpy(),
            item.x_test[names].to_numpy(),
            y_test.to_numpy(),
            loss = loss,
            num_rounds = rounds,
            random_seed = item.seed)
        return pd.DataFrame({
            'loss': [loss],
            'expected_loss': [float(expected)],
            'bias': [float(bias)],
            'variance': [float(variance)]})


@dataclasses.dataclass
class ClassificationReport(Evaluator):
    """Precision, recall, f1, and the number of rows of each class.

    Wraps:
        [`sklearn.metrics.classification_report`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
        from scikit-learn.

    """

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
    """How many rows of each class were predicted to be each class.

    Wraps:
        [`sklearn.metrics.confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
        from scikit-learn.

    """

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
class Conformal(Evaluator):
    """Prediction intervals (or sets) with a known rate of coverage.

    Conformal prediction turns any model's predictions into intervals (for
    regression) or sets of classes (for classification) that contain the
    true value for at least a chosen share of rows ("confidence", 90% by
    default), without assumptions about the data's distribution. It wraps
    MAPIE's cross-conformal methods, which refit copies of the model on folds
    of the training rows ("cv", 5 by default). The table has a row for each
    test row. The share of rows covered ("coverage") and the mean width of
    the intervals ("interval_width") or size of the sets ("set_size") are
    stored in the dataset's `metrics`.

    Wraps:
        - [`mapie.classification.CrossConformalClassifier`](https://mapie.readthedocs.io/en/stable/api/classification/#mapie.classification.CrossConformalClassifier)
          from MAPIE, to classify.
        - [`mapie.regression.CrossConformalRegressor`](https://mapie.readthedocs.io/en/stable/api/regression/#mapie.regression.CrossConformalRegressor)
          from MAPIE, to regress.

    """

    def evaluate(
        self,
        item: base.Dataset,
        confidence: float = 0.9,
        cv: int = 5,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the interval (or set) of each test row.

        Args:
            item: the dataset with a fitted model.
            confidence: share of rows that the intervals should cover.
                Defaults to 0.9.
            cv: number of folds of the training rows. Defaults to 5.
            **kwargs: not used.

        Returns:
            One row for each test row, with "actual", "prediction", "lower"
                and "upper" (or "set" and "size"), and "covered".

        """
        sklearn_base = importlib.import_module('sklearn.base')
        model = _model(item)
        columns = (
            utilities.feature_names(model, item.data.columns) or item.features)
        rows = item.y_test.index if item.is_split else item.data.index
        train = item.y_train.index
        estimator = sklearn_base.clone(model)
        actual = item.y.loc[rows]
        if item.task == 'regress':
            regression = utilities.import_tool('mapie.regression')
            conformal = regression.CrossConformalRegressor(
                estimator,
                confidence_level = confidence,
                cv = cv,
                random_state = item.seed)
            conformal.fit_conformalize(
                item.data.loc[train, columns], item.y_train)
            predicted, intervals = conformal.predict_interval(
                item.data.loc[rows, columns])
            lower = intervals[:, 0, 0]
            upper = intervals[:, 1, 0]
            return pd.DataFrame({
                'actual': actual,
                'prediction': predicted,
                'lower': lower,
                'upper': upper,
                'covered': (actual >= lower) & (actual <= upper)},
                index = rows)
        classification = utilities.import_tool('mapie.classification')
        conformal = classification.CrossConformalClassifier(
            estimator,
            confidence_level = confidence,
            cv = cv,
            random_state = item.seed)
        conformal.fit_conformalize(item.data.loc[train, columns], item.y_train)
        predicted, sets = conformal.predict_set(item.data.loc[rows, columns])
        classes = np.unique(np.asarray(item.y_train))
        members = [
            [c for c, inside in zip(classes, row, strict = True) if inside]
            for row in sets[:, :, 0]]
        return pd.DataFrame({
            'actual': actual,
            'prediction': predicted,
            'set': [', '.join(str(c) for c in m) for m in members],
            'size': [len(m) for m in members],
            'covered': [
                a in m for a, m in zip(actual, members, strict = True)]},
            index = rows)

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Adds the table and stores the coverage and width (or set size).

        Args:
            item: the dataset with a fitted model.
            **kwargs: parameters for `evaluate`.

        Returns:
            The dataset, with the table in `tables` and the coverage in
                `metrics`.

        """
        super().apply(item, **kwargs)
        table = item.tables[self.name]
        item.metrics['coverage'] = float(table['covered'].mean())
        if 'lower' in table.columns:
            item.metrics['interval_width'] = float(
                (table['upper'] - table['lower']).mean())
        else:
            item.metrics['set_size'] = float(table['size'].mean())
        return item


@dataclasses.dataclass
class ExplainWeights(Evaluator):
    """eli5's explanation of the weights of the model's features.

    For a linear model, the weights are its coefficients (for each class);
    for trees and boosting, they are the importances and their spread across
    the trees. The table is eli5's, with the 20 largest weights by default.

    Wraps:
        [`eli5.explain_weights_df`](https://eli5.readthedocs.io/en/latest/autodocs/eli5.html#eli5.explain_weights_df)
        from eli5.

    """

    def evaluate(
        self,
        item: base.Dataset,
        top: int = 20,
        **kwargs: Any) -> pd.DataFrame:
        """Returns eli5's table of the model's weights.

        Args:
            item: the dataset with a fitted model.
            top: most weights to list. Defaults to 20.
            **kwargs: not used.

        Raises:
            ValueError: if eli5 cannot explain the model.

        Returns:
            eli5's table of weights.

        """
        eli5 = utilities.import_tool('eli5')
        model = _model(item)
        if isinstance(model, models.LabelCoded):
            model = model.estimator
        names = [str(n) for n in (
            utilities.feature_names(model, item.data.columns)
            or item.features)]
        table: pd.DataFrame | None = eli5.explain_weights_df(
            model, feature_names = names, top = top)
        if table is None:
            message = (
                f'eli5 cannot explain the weights of {type(model).__name__}')
            raise ValueError(message)
        return table


@dataclasses.dataclass
class FactorAnalysis(Evaluator):
    """The hidden factors that explain the correlations of the features.

    statsmodels' factor analysis of the features that the model learned from
    (the real training rows). The table has a row for each feature, with its
    loading on each factor, its communality (the share of its variance that
    the factors explain), and its uniqueness (the rest). By default, there is
    a factor for each eigenvalue of the features' correlations above 1
    (Kaiser's rule), and the loadings are rotated by varimax.

    Wraps:
        [`statsmodels.multivariate.factor.Factor`](https://www.statsmodels.org/stable/generated/statsmodels.multivariate.factor.Factor.html)
        from statsmodels.

    """

    def evaluate(
        self,
        item: base.Dataset,
        *,
        factors: int | None = None,
        method: str = 'pa',
        rotation: str | None = 'varimax',
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the loading of each feature on each factor.

        Args:
            item: the dataset.
            factors: number of factors. Defaults to `None`, which uses
                Kaiser's rule.
            method: "pa" (principal axes) or "ml" (maximum likelihood).
                Defaults to "pa".
            rotation: how statsmodels rotates the loadings, such as
                "varimax", "quartimax", "promax", or "oblimin", or `None`
                (or "none") to leave them unrotated. Defaults to "varimax".
            columns: the columns to analyze. Defaults to `None`, which uses
                the numeric and boolean features.
            **kwargs: not used.

        Returns:
            One row for each feature, with a "factor_{n}" column for each
                factor, "communality", and "uniqueness".

        """
        factor = utilities.import_tool('statsmodels.multivariate.factor')
        data = _numbers(item, columns, self.name)
        count = factors or _kaiser(data)
        results = factor.Factor(
            data, n_factor = count, method = method, missing = 'drop').fit()
        if rotation not in {None, 'none'} and count > 1:
            results.rotate(rotation)
        names = [f'factor_{n}' for n in range(1, count + 1)]
        table = pd.DataFrame(
            np.asarray(results.loadings), index = data.columns, columns = names)
        table['communality'] = np.asarray(results.communality)
        table['uniqueness'] = np.asarray(results.uniqueness)
        return table


@dataclasses.dataclass
class Fairness(Evaluator):
    """How the model does for each group, and the gaps between groups.

    It compares the groups in the column named by the "group" parameter
    (the dataset's first `groups` column by default) with fairlearn's
    `MetricFrame`: the number of rows, the share predicted to be positive
    (the selection rate), accuracy, and the true and false positive rates.
    The last two rows are the largest difference and the smallest ratio
    between groups. The label must have two classes.

    Wraps:
        [`fairlearn.metrics.MetricFrame`](https://fairlearn.org/v0.14/api_reference/generated/fairlearn.metrics.MetricFrame.html)
        from fairlearn.

    """

    def evaluate(
        self,
        item: base.Dataset,
        group: str | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the metrics of each group and the gaps between them.

        Args:
            item: the dataset with a fitted classifier.
            group: name of the column of groups. Defaults to `None`, which
                uses the dataset's first group.
            **kwargs: not used.

        Returns:
            One row for each group, then "difference" and "ratio" rows.

        """
        fairness = utilities.import_tool('fairlearn.metrics')
        scores = importlib.import_module('sklearn.metrics')
        column = metrics_._group(item, group, self.name)
        y_true, y_pred = _observed(item)
        y_true, y_pred = metrics_._positive(item, y_true, y_pred, self.name)
        frame = fairness.MetricFrame(
            metrics = {
                'count': fairness.count,
                'selection_rate': fairness.selection_rate,
                'accuracy': scores.accuracy_score,
                'true_positive_rate': fairness.true_positive_rate,
                'false_positive_rate': fairness.false_positive_rate},
            y_true = y_true,
            y_pred = y_pred,
            sensitive_features = item.data.loc[y_true.index, column])
        table: pd.DataFrame = frame.by_group.copy()
        table.index = table.index.astype(str)
        table.loc['difference'] = frame.difference()
        table.loc['ratio'] = frame.ratio()
        table.index.name = column
        return table


@dataclasses.dataclass
class FeatureImportance(Evaluator):
    """The importance that the model itself gives each feature.

    This is the model's `feature_importances_` (for trees and boosting), the
    importance of each term (for `explainable_boosting`), or the absolute
    value of its coefficients (for linear models, averaged over the classes).
    Coefficients are only comparable if the features were scaled. A feature
    that the model dropped (such as a collinear one, which pyfixest drops) has
    no coefficient and is left out. For other models, use
    `permutation_importance`.

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
        if importances is None and hasattr(model, 'term_importances'):
            # An Explainable Boosting Machine reports the importance of each
            # term (a feature or a pair of features).
            return _importance_table(
                [str(n) for n in model.term_names_],
                np.asarray(model.term_importances()))
        if importances is None:
            coefficients = getattr(model, 'coef_', None)
            if coefficients is None:
                message = (
                    f'{type(model).__name__} does not report the importance of '
                    f'its features: use permutation_importance instead'
                )
                raise ValueError(message)
            importances = np.abs(np.atleast_2d(coefficients)).mean(axis = 0)
        features = (
            utilities.feature_names(model, item.data.columns) or item.features)
        coefficients = getattr(model, 'coefficients', None)
        if len(np.ravel(importances)) != len(features) and callable(coefficients):
            # A model that drops features (such as pyfixest's, which drops
            # collinear ones) has a coefficient only for the others.
            table = coefficients().dropna(subset = ['coefficient'])
            kept = [f for f in features if f in table.index]
            return _importance_table(
                kept, np.abs(table.loc[kept, 'coefficient'].to_numpy(float)))
        return _importance_table(features, np.asarray(importances))


@dataclasses.dataclass
class PCA(Evaluator):
    """How much of the variance of the features each principal component has.

    statsmodels' principal component analysis of the features that the model
    learned from (the real training rows). Unlike the `pca_reduce` reducer, it
    does not change the data: it describes how many dimensions the features
    really have. The table has a row for each component, with its eigenvalue and the
    share of the variance it explains, and the loading of each feature on
    each component is stored in `tables` as "{name}_loadings".

    Wraps:
        [`statsmodels.multivariate.pca.PCA`](https://www.statsmodels.org/stable/generated/statsmodels.multivariate.pca.PCA.html)
        from statsmodels.

    """

    def evaluate(
        self,
        item: base.Dataset,
        *,
        components: int | None = None,
        standardize: bool = True,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the eigenvalue and share of the variance of each component.

        Args:
            item: the dataset.
            components: number of components. Defaults to `None`, which uses
                one for each column.
            standardize: whether to give each column a standard deviation of
                1 first, so that the columns with the largest values do not
                dominate. Defaults to `True`.
            columns: the columns to analyze. Defaults to `None`, which uses
                the numeric and boolean features.
            **kwargs: not used.

        Returns:
            One row for each component, with its "eigenvalue" (of the
                correlations of the columns, or of their covariances if
                `standardize` is `False`), the "share" of the variance that
                it explains, and the "cumulative" share of it and the
                components before it.

        """
        pca = utilities.import_tool('statsmodels.multivariate.pca.PCA')
        data = _numbers(item, columns, self.name)
        results = pca(
            data,
            ncomp = components,
            standardize = standardize,
            missing = 'drop-row')
        count = len(results.eigenvals)
        names = [f'component_{n}' for n in range(1, count + 1)]
        # statsmodels' eigenvalues are multiplied by the number of rows, and
        # the R² of the first n components is the share of the variance that
        # they explain together.
        rows = len(results.factors)
        cumulative = np.asarray(results.rsquare)[1:count + 1]
        item.tables[f'{self.name}_loadings'] = pd.DataFrame(
            np.asarray(results.loadings), index = data.columns, columns = names)
        return pd.DataFrame({
            'eigenvalue': np.asarray(results.eigenvals) / rows,
            'share': np.diff(cumulative, prepend = 0.0),
            'cumulative': cumulative},
            index = names)


@dataclasses.dataclass
class PermutationImportance(Evaluator):
    """How much the model's score drops when each feature is shuffled.

    This works for any model. By default, the score is accuracy (for
    classification) or R squared (for regression), and each feature is
    shuffled 10 times. Columns that the model needs besides its features
    (such as the fixed effects and cluster of a `fixest` model) are not
    shuffled: each row keeps its own values of them.

    Wraps:
        [`sklearn.inspection.permutation_importance`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html)
        from scikit-learn.

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
        model = _model(item)
        others = _other_columns(model, item)
        if others or not hasattr(model, '__sklearn_tags__'):
            model = _GivenColumns(model, item.data.loc[x.index, others], item.task)
        result = inspection.permutation_importance(
            model,
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
class RegressionTable(_Published, Evaluator):
    """The estimates of every model side by side, ready to publish.

    Like the regression tables of Stata, R, and statsmodels, it has a column
    for each model and two rows for each feature: its estimate, with stars
    for its p-value, and below it, its standard error in parentheses. The
    models are the branches of the most recent `experiment`, or the final
    model if the data did not come from one. Rows below the estimates
    describe each model: the kind of estimates, the number of observations
    that it learned from, and its cross-validated metrics.

    A model that reports its coefficients and their statistics (the
    statsmodels and pyfixest models, such as `ols`, `logit_sm`, `glm`, and
    `fixest`) is shown with them. Every other model is shown with the
    average marginal effect (AME) of each feature: the average change in the
    prediction (or in the predicted probability of each class, for a
    classifier) for a one-unit increase in the feature. These are models
    without coefficients (such as `random_forest` or `xgboost`) and the
    scikit-learn linear models, whose coefficients have no standard errors.
    The AME of a linear regression is its coefficient. For each training
    row, a feature is moved up and down by "step" of its standard
    deviations (0.1 by default), or from its lower to its higher value if it
    has only two. The standard error and 95% confidence interval of an AME
    come from refitting the model to "simulations" bootstrap samples of the
    training rows (100 by default), and its p-value comes from the normal
    distribution. Set "marginal" to true to show marginal effects for every
    model, which puts every column on the same scale.

    Features that form a set, such as the dummy variables of one category
    with its reference left out (a victim's relationship to an offender:
    "family", "partner", or "stranger", against "unknown"), cannot be moved
    one at a time: setting "family" to 1 in a row whose "stranger" is 1 makes
    a row that cannot happen. Name them in "sets" (in settings, as
    "family, partner, stranger; under_12, teen", with sets separated by
    semicolons) and the AME of each member is the average change in the
    prediction from the reference (every member 0) to that member alone (it
    1, the others 0). The members of a set must have two values each, and
    members that are not features of a model are left out of its set.

    The metrics are cross-validated on the training rows, never scored on
    the rows that the model learned from. They are the "cv_{metric}" metrics
    of a validator that the analysis already ran, or else they come from the
    validator named by "validator" (`stratified_k_fold` to classify and
    `k_fold` to regress, by default).

    A marginal effect describes the model, not the world. It is the
    model's estimated effect if every other feature is held fixed, so it is
    only causal if the features are (see the inferers to estimate a causal
    effect). It can mislead when features are correlated, since moving one
    while holding the others fixed makes rows unlike any real one. The
    bootstrap treats rows as independent, so its standard errors are too
    small for rows that are correlated within groups or over time.

    The estimates are stored in the dataset's `tables` under the technique's
    name, with one row for each model and feature, and the rows below them
    are stored as "{name}_statistics". The table can be saved in the same
    formats as a `scorecard` (csv, Markdown, LaTeX, HTML, Word, or an
    image). In the "stata" `style` (the default), Word documents, images,
    and `to_text` look like Stata's output of `regress`: a block for each
    model, with a description of the model and its statistics at the top and
    a table of its estimates below them ("Coef." or "dy/dx", "Std. Err.",
    "t" or "z", its p-value, and its 95% confidence interval), set in a
    monospaced font between rules. In the "columns" style, they put the
    models side by side, as csv, Markdown, LaTeX, and HTML always do (like
    Stata's `esttab`). Make one from a dataset, an applied project, or
    several datasets with `create`.

    Wraps:
        - [`sklearn.base.clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
          from scikit-learn, to refit each model to bootstrap samples.
        - [`scipy.stats.norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
          from SciPy, for the p-values of marginal effects.
        - [`docx.Document`](https://python-docx.readthedocs.io/en/latest/api/document.html#docx.Document)
          from python-docx, for `to_word`.
        - [`matplotlib.figure.Figure`](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html)
          from matplotlib, for `to_image`.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, in which case it is "regression_table".
        contents: not used. Defaults to `None`.
        parameters: keyword arguments for `evaluate`, such as "marginal" and
            "simulations". Defaults to an empty `dict`.
        title: title of the table in Word documents, images, LaTeX, and
            HTML. Defaults to `None`, in which case it says how many models
            there are.
        digits: digits after the decimal point when numbers are written as
            text. Defaults to 3.
        stars: the p-values below which an estimate gets one, two, and three
            stars. Defaults to 0.1, 0.05, and 0.01.
        style: how Word documents and images lay out the models: "stata"
            (a block for each model, as Stata's `regress` prints it) or
            "columns" (side by side). Defaults to "stata".

    Attributes:
        table: the most recent table, as text: a column of terms and a column
            for each model.
        estimates: the estimates of the most recent table, with one row for
            each model and term, and their "estimate", "standard_error",
            "statistic", "p_value", "ci_lower", and "ci_upper".
        statistics: the rows below the estimates in the most recent table,
            with a column for each model.

    """

    title: str | None = None
    digits: int = 3
    stars: tuple[float, ...] = (0.1, 0.05, 0.01)
    style: str = 'stata'
    table: pd.DataFrame | None = dataclasses.field(default = None, repr = False)
    estimates: pd.DataFrame | None = dataclasses.field(
        default = None, repr = False)
    statistics: pd.DataFrame | None = dataclasses.field(
        default = None, repr = False)
    # What each model's block in the "stata" style says about it, by model.
    _blocks: dict[str, _Block] = dataclasses.field(
        default_factory = dict, init = False, repr = False)

    # The cross-validated metrics of each task, by name.
    defaults: ClassVar[dict[str, tuple[str, ...]]] = {
        'classify': ('accuracy', 'roc_auc', 'log_loss'),
        'regress': ('r2', 'rmse', 'mae')}
    # The validator that cross-validates a model for each task, if the
    # analysis did not.
    validators: ClassVar[dict[str, str]] = {
        'classify': 'stratified_k_fold',
        'regress': 'k_fold'}
    # What the table is called in the message of an empty table.
    noun: ClassVar[str] = 'regression table'

    """ Class Methods """

    @classmethod
    def create(
        cls,
        item: (
            base.Dataset
            | chrisjen.Project
            | Mapping[str, base.Dataset]
            | Sequence[base.Dataset]),
        **kwargs: Any) -> RegressionTable:
        """Returns the regression table of one or more fitted models.

        Args:
            item: a `Dataset` (whose branches, or final model, are the
                columns), a `Project` that has been applied, or several
                datasets with fitted models (by the name of each column, or
                in a sequence, named by their models).
            **kwargs: arguments for the table (such as `title`, `digits`, and
                `stars`) and for `evaluate` (such as `marginal`).

        Raises:
            TypeError: if `item` is not one of these.

        Returns:
            A regression table, with its `table` filled in.

        """
        if isinstance(item, chrisjen.Project):
            item = item.result
        fields = {field.name for field in dataclasses.fields(cls)}
        table = cls(**{k: v for k, v in kwargs.items() if k in fields})
        parameters = {k: v for k, v in kwargs.items() if k not in fields}
        if isinstance(item, base.Dataset):
            table.evaluate(item, **parameters)
            return table
        datasets: list[Any] = []
        names: list[str] = []
        if isinstance(item, Mapping):
            datasets = list(item.values())
            names = [str(k) for k in item]
        elif isinstance(item, Sequence) and not isinstance(item, str):
            datasets = list(item)
        if not datasets or not all(
            isinstance(d, base.Dataset) for d in datasets):
            message = (
                f'a regression table needs a Dataset, an applied Project, or '
                f'several datasets, not {type(item).__name__}'
            )
            raise TypeError(message)
        names = names or _unique([_model_name(d) for d in datasets])
        table._tabulate(dict(zip(names, datasets, strict = True)), **parameters)
        return table

    """ Properties """

    @property
    def heading(self) -> str:
        """Returns `title`, or a description of the models."""
        if self.title:
            return self.title
        count = 0 if self.statistics is None else len(self.statistics.columns)
        if count == 1:
            return 'Estimates of the final model'
        return f'Estimates of {count} models'

    """ Public Methods """

    def evaluate(
        self,
        item: base.Dataset,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the estimates of every model of `item`.

        Args:
            item: the dataset with a fitted model. If it came from an
                experiment, each branch is a model in the table.
            **kwargs: parameters of the estimates and metrics (see
                `_tabulate`).

        Returns:
            One row for each model and term, with the "estimate",
                "standard_error", "statistic", "p_value", "ci_lower", and
                "ci_upper". The text of the table is stored in `table`.

        """
        return self._tabulate(self._columns(item), **kwargs)

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Adds the estimates and the rows below them to the tables of `item`.

        Args:
            item: the dataset with a fitted model.
            **kwargs: parameters for `evaluate`.

        Returns:
            The dataset, with the estimates in `tables` under the technique's
                name and the rows below them as "{name}_statistics".

        """
        super().apply(item, **kwargs)
        if self.statistics is not None:
            item.tables[f'{self.name}_statistics'] = self.statistics
        return item

    def to_figure(self) -> Any:
        """Returns the table drawn in a `matplotlib` figure.

        In the "stata" style, each model is a block like Stata's output of
        `regress`, in a monospaced font, with rules above and below its
        estimates and a bar after the terms. In the "columns" style, the
        models are side by side, as in a scorecard.

        Returns:
            A `matplotlib.figure.Figure`.

        """
        if self._stata_style():
            return self._stata_figure()
        return super().to_figure()

    def to_text(self, path: pathlib.Path | str | None = None) -> str:
        """Returns the table as Stata prints it, and saves it if asked.

        Each model is a block, as in Stata's output of `regress`: its name,
        the kind of estimates, and what they are estimates of on the left;
        the number of observations, the number of bootstrap samples, and the
        cross-validated metrics on the right; and below them, a row for each
        term with its estimate, standard error, test statistic, p-value, and
        95% confidence interval. A missing number is a dot, as in Stata.

        Args:
            path: file to save the text in. Defaults to `None`.

        Returns:
            The text.

        """
        lines = []
        for line in self._stata_lines():
            if line.kind in {'top', 'bottom'}:
                lines.append('-' * line.width)
            elif line.kind == 'cross':
                lines.append(
                    '-' * line.bar + '+' + '-' * (line.width - line.bar - 1))
            else:
                lines.append(line.text.rstrip())
        text = '\n'.join(lines) + '\n'
        if path is not None:
            pathlib.Path(path).write_text(text, encoding = 'utf-8')
        return text

    def to_word(self, path: pathlib.Path | str) -> pathlib.Path:
        """Saves the table in a Word document.

        In the "stata" style, the document has the heading and, for each
        model, a borderless table of its description and statistics and a
        table of its estimates with Stata's rules (above and below the
        header, below the last term, and after the terms), set in Courier
        New. In the "columns" style, the models are side by side, as in a
        scorecard.

        Args:
            path: file to save the document in (usually ending in ".docx").

        Returns:
            The path of the document.

        """
        if not self._stata_style():
            return super().to_word(path)
        docx = utilities.import_tool('docx')
        alignment = utilities.import_tool('docx.enum.text.WD_ALIGN_PARAGRAPH')
        points = utilities.import_tool('docx.shared.Pt')
        document = docx.Document()
        normal = document.styles['Normal']
        normal.font.name = 'Courier New'
        normal.font.size = points(8)
        normal.paragraph_format.space_after = points(0)
        document.add_heading(self.heading, level = 2)
        for model in self._models():
            left, right = self._stata_header(model)
            header = document.add_table(
                rows = max(len(left), len(right)), cols = 2)
            for row, text in enumerate(left):
                header.cell(row, 0).text = text
            for row, text in enumerate(right):
                header.cell(row, 1).text = text
                header.cell(row, 1).paragraphs[0].alignment = alignment.RIGHT
            document.add_paragraph()
            rows = self._stata_rows(model)
            grid = document.add_table(rows = len(rows), cols = 7)
            for number, values in enumerate(rows):
                for column, value in enumerate(values):
                    cell = grid.cell(number, column)
                    cell.text = value
                    cell.paragraphs[0].alignment = alignment.RIGHT
                edges = ['right'] if number else ['top', 'bottom']
                if number == len(rows) - 1 and number:
                    edges.append('bottom')
                _border(grid.cell(number, 0), [*edges, 'right'])
                for column in range(1, 7):
                    _border(
                        grid.cell(number, column),
                        [e for e in edges if e != 'right'])
            # The interval's header spans its two columns, as in Stata.
            interval = grid.cell(0, 5).merge(grid.cell(0, 6))
            interval.text = rows[0][5]
            interval.paragraphs[0].alignment = alignment.CENTER
            document.add_paragraph()
        notes = self._stata_notes()
        if notes:
            document.add_paragraph(notes)
        path = pathlib.Path(path)
        document.save(str(path))
        return path

    """ Private Methods """

    def _columns(self, item: base.Dataset) -> dict[str, base.Dataset]:
        """Returns the dataset of each model in the table, by its column.

        Args:
            item: the dataset with a fitted model.

        Returns:
            The whole dataset of each branch of the experiment that `item`
                came from, named by the steps that differ between branches,
                or `item` itself, named by its model.

        """
        if not item.branches:
            return {_model_name(item): item}
        kept = [
            branch for branch in item.branches
            if branch.dataset is not None and branch.dataset.model is not None]
        if not kept:
            warnings.warn(
                'the datasets of the branches are no longer in memory, so '
                'the regression table has only the final model (an '
                'experiment keeps them only if "conserve_memory" is false)',
                stacklevel = 3)
            return {_model_name(item): item}
        if len(kept) < len(item.branches):
            lost = [b.label for b in item.branches if b not in kept]
            warnings.warn(
                f'the datasets of some branches are no longer in memory, so '
                f'they are not in the regression table (an experiment keeps '
                f'them only if "conserve_memory" is false): {lost}',
                stacklevel = 3)
        names = _unique(_branch_names(kept))
        return {
            name: branch.dataset for name, branch in zip(
                names, kept, strict = True)
            if branch.dataset is not None}

    def _cross_validated(
        self,
        item: base.Dataset,
        validator: str | None,
        metrics: Sequence[str] | None) -> dict[str, float]:
        """Returns the cross-validated metrics of the model of `item`.

        Args:
            item: the dataset with a fitted model.
            validator: name of the validator to use if the analysis did not
                cross-validate the model, or `None` for `validators`.
            metrics: names of the metrics, or `None` for `defaults`.

        Returns:
            The mean of each metric across the folds, by "cv_{metric}". The
                metrics already in `item` are used if there are any, and
                `item` is not changed.

        """
        names = list(metrics or self.defaults.get(item.task or 'classify', ()))
        keys = [f'cv_{name}' for name in names]
        found = item.metrics
        if not any(key in found for key in keys):
            # The validators module imports this one, so it is imported when
            # used.
            from . import validators as validators_  # noqa: PLC0415
            name = validator or self.validators[item.task or 'classify']
            technique = chrisjen.library.borrow(name, genre = 'validator')()
            if not isinstance(technique, validators_.Validator):
                message = f'{name!r} is not a validator'
                raise TypeError(message)
            # A copy that shares the data, so that the validator's tables
            # and metrics are not added to the dataset.
            copy = dataclasses.replace(
                item,
                metrics = {},
                tables = {},
                figures = {},
                history = [],
                branches = [])
            technique.apply(copy, metrics = names)
            found = copy.metrics
        return {key: found[key] for key in keys if key in found}

    def _csv_table(self) -> pd.DataFrame:
        """Returns the table that `to_csv` saves: the text of the table."""
        return self._text()

    def _estimate(
        self,
        item: base.Dataset,
        *,
        marginal: bool,
        simulations: int,
        step: float,
        sets: Sequence[Sequence[str]] = ()) -> tuple[pd.DataFrame, str]:
        """Returns the estimates of the model of `item` and their kind.

        Args:
            item: the dataset with a fitted model.
            marginal: whether to find marginal effects even if the model
                reports its coefficients.
            simulations: number of bootstrap samples for marginal effects.
            step: standard deviations that each feature is moved up and
                down for marginal effects.
            sets: sets of features whose members are moved one at a time
                from all of them being 0. Defaults to none.

        Returns:
            One row for each term, with the columns of `estimates`, and the
                kind of estimates ("coefficients" or "marginal effects").

        """
        model = _model(item)
        if not marginal and hasattr(model, 'coefficients'):
            table = model.coefficients().rename(
                columns = {'coefficient': 'estimate'})
            return table.reindex(columns = list(_ESTIMATES)), 'coefficients'
        if item.task == 'classify' and not hasattr(model, 'predict_proba'):
            empty = pd.DataFrame(columns = list(_ESTIMATES), dtype = float)
            return empty, 'none (no predicted probabilities)'
        table = _marginal_effects(item, model, simulations, step, sets)
        return table, 'marginal effects'

    def _levels(self) -> list[float]:
        """Returns `stars` as numbers.

        Returns:
            The p-values of the stars. Text from a settings file (such as
                "0.05, 0.01, 0.001") is split at its commas.

        """
        levels = self.stars
        if isinstance(levels, str):
            levels = tuple(levels.split(','))
        return [float(level) for level in camina.listify(levels)]

    def _notes(self) -> str | None:
        """Returns the meaning of the parentheses and stars."""
        levels = sorted(self._levels(), reverse = True)
        stars = ', '.join(
            f'{"*" * count} p < {level:g}'
            for count, level in enumerate(levels, start = 1))
        notes = f'Standard errors in parentheses. {stars}.'
        if self.statistics is not None and any(
            str(kind).startswith('marginal')
            for kind in self.statistics.loc['estimates']):
            notes += (
                ' Marginal effects are the average change in the prediction '
                'for a one-unit increase in a feature, with bootstrapped '
                'standard errors.'
            )
        return notes

    def _numeric_columns(self) -> set[str]:
        """Returns the names of the columns of the models."""
        if self.statistics is None:
            return set()
        return {str(c) for c in self.statistics.columns}

    def _rules(self) -> list[int]:
        """Returns the position of the first row below the estimates."""
        if self.estimates is None or self.estimates.empty:
            return []
        terms = self.estimates.index.get_level_values('term').unique()
        return [2 * len(terms)]

    def _models(self) -> list[str]:
        """Returns the names of the models in the table, in order."""
        self._require_table()
        if self.statistics is None:
            return []
        return [str(c) for c in self.statistics.columns]

    def _stata_figure(self) -> Any:
        """Returns the "stata" style of the table in a `matplotlib` figure.

        Each character of the monospaced text is one unit of the axes, so
        the rules and the bar after the terms are drawn exactly between the
        characters.

        Returns:
            A `matplotlib.figure.Figure`, with the heading as the first text
                of its axes.

        """
        figures = utilities.import_tool('matplotlib.figure')
        lines = self._stata_lines()
        width = max([40, *(line.width for line in lines)])
        size = 9
        margin = 2
        # A rule takes less room than a line of text, as in Stata.
        heights = [
            0.6 if line.kind in {'top', 'cross', 'bottom'} else 1.0
            for line in lines]
        tops = list(itertools.accumulate([2.0, *heights]))
        depth = tops[-1] + 1
        figure = figures.Figure(figsize = (
            (width + 2 * margin) * _MONOSPACE * size / 72,
            (depth + 1) * 1.45 * size / 72))
        axes = figure.add_axes((0, 0, 1, 1))
        axes.set_axis_off()
        axes.set_xlim(-margin, width + margin)
        axes.set_ylim(depth, -1)
        axes.text(
            width / 2, 0.5, self.heading,
            ha = 'center', va = 'center', fontsize = size + 2, weight = 'bold')
        rule = {'color': 'black', 'linewidth': 0.8}
        for line, top, height in zip(lines, tops, heights, strict = False):
            middle, bottom = top + height / 2, top + height
            if line.kind in {'top', 'cross', 'bottom'}:
                axes.plot([0, line.width], [middle, middle], **rule)
            if line.kind == 'row':
                # The bar is drawn as a line, so its character is left out.
                text = f'{line.text[:line.bar]} {line.text[line.bar + 1:]}'
            else:
                text = line.text if line.kind == 'text' else ''
            if text.strip():
                axes.text(
                    0, middle, text,
                    family = 'DejaVu Sans Mono', fontsize = size,
                    ha = 'left', va = 'center')
            spans = {
                'top': (middle, bottom),
                'row': (top, bottom),
                'cross': (top, bottom),
                'bottom': (top, middle)}
            if line.kind in spans:
                axes.plot([line.bar + 0.5] * 2, list(spans[line.kind]), **rule)
        return figure

    def _stata_header(self, model: str) -> tuple[list[str], list[str]]:
        """Returns the lines above the estimates of a model's block.

        Args:
            model: the name of the model.

        Returns:
            The lines on the left (the model, its kind of estimates, and
                what they are estimates of) and on the right (the number of
                observations, of bootstrap samples, and the cross-validated
                metrics, as "{name} = {value}").

        """
        statistics = self._statistics_of(model)
        block = self._blocks.get(model, _Block())
        kind = str(statistics.get('estimates', ''))
        descriptions = {
            'coefficients': 'Coefficients',
            'marginal effects': 'Average marginal effects'}
        left = [model, descriptions.get(kind, f'No estimates: {kind}')]
        if block.expression:
            left.append(f'Expression: {block.expression}')
        pairs = [('Number of obs', statistics.get('observations'))]
        if block.replications:
            pairs.append(('Replications', block.replications))
        pairs.extend(
            (f'CV {_STATA_NAMES.get(name[3:], name[3:])}', value)
            for name, value in statistics.items()
            if str(name).startswith('cv_'))
        width = max(len(label) for label, _ in pairs) + 2
        right = [
            f'{label:<{width}} = {_stata_value(value):>10}'
            for label, value in pairs]
        return left, right

    def _stata_lines(self) -> list[_Line]:
        """Returns the lines of the "stata" style of the table.

        Returns:
            For each model, its header (see `_stata_header`), a blank line,
                a rule, the column titles, a rule crossed by the bar, a row
                for each term, and a rule, followed by the notes. Blocks are
                separated by two blank lines.

        """
        terms = self._stata_width()
        width = terms + 2 + sum(_STATA_COLUMNS)
        bar = terms + 1

        def line(kind: str, text: str = '') -> _Line:
            return _Line(kind, text, width, bar)

        lines: list[_Line] = []
        for number, model in enumerate(self._models()):
            if number:
                lines.extend([line('text'), line('text')])
            left, right = self._stata_header(model)
            room = max([0, *(len(text) for text in right)])
            for first, second in itertools.zip_longest(
                left, right, fillvalue = ''):
                lines.append(
                    line('text', f'{first:<{width - room}}{second:>{room}}'))
            lines.extend([line('text'), line('top')])
            rows = self._stata_rows(model)
            for position, row in enumerate(rows):
                numbers = ''.join(
                    f'{value:>{size}}'
                    for value, size in zip(row[1:5], _STATA_COLUMNS, strict = False))
                if position == 0:
                    # The title of the interval spans its two columns.
                    numbers += f'{row[5]:>{sum(_STATA_COLUMNS[4:])}}'
                else:
                    numbers += (
                        f'{row[5]:>{_STATA_COLUMNS[4]}}'
                        f'{row[6]:>{_STATA_COLUMNS[5]}}')
                lines.append(line('row', f'{row[0]:>{terms}} |{numbers}'))
                if position == 0:
                    lines.append(line('cross'))
            lines.append(line('bottom'))
        notes = self._stata_notes()
        if notes:
            lines.append(line('text'))
            lines.extend(
                line('text', text) for text in textwrap.wrap(notes, width))
        return lines

    def _stata_notes(self) -> str | None:
        """Returns notes on marginal effects for the "stata" style, if any."""
        if not any(
            block.coefficient == 'dy/dx' for block in self._blocks.values()):
            return None
        return (
            'dy/dx is the average change in the prediction for a one-unit '
            'increase in a feature, with the other features held fixed. Its '
            'standard error and interval come from refitting the model to '
            'bootstrap samples of the training rows (Replications).'
        )

    def _stata_rows(self, model: str) -> list[list[str]]:
        """Returns the column titles and a row for each term of a model.

        Args:
            model: the name of the model.

        Returns:
            Rows of seven texts: the term, its estimate, standard error,
                test statistic, p-value, and the bounds of its 95%
                confidence interval. The first row has the titles, with the
                interval's title in the sixth text.

        """
        block = self._blocks.get(model, _Block())
        rows = [[
            block.label,
            block.coefficient,
            'Std. Err.',
            block.test,
            f'P>|{block.test}|',
            '[95% Conf. Interval]',
            '']]
        estimates = self.estimates
        if (
            estimates is None
            or model not in estimates.index.get_level_values('model')):
            return rows
        for term, values in estimates.loc[model].iterrows():
            rows.append([
                str(term),
                _stata_number(values['estimate']),
                _stata_number(values['standard_error']),
                _fixed(values['statistic'], 2),
                _fixed(values['p_value'], 3),
                _stata_number(values['ci_lower']),
                _stata_number(values['ci_upper'])])
        return rows

    def _stata_style(self) -> bool:
        """Returns whether `style` is "stata".

        Raises:
            ValueError: if `style` is not "stata" or "columns".

        Returns:
            Whether Word documents and images use the "stata" style.

        """
        if self.style not in {'stata', 'columns'}:
            message = f'style must be "stata" or "columns", not {self.style!r}'
            raise ValueError(message)
        return self.style == 'stata'

    def _stata_width(self) -> int:
        """Returns the width of the column of terms in the "stata" style."""
        terms: list[str] = (
            [] if self.estimates is None
            else [
                str(t) for t in self.estimates.index.get_level_values('term')])
        labels = [block.label for block in self._blocks.values()]
        return max([12, *map(len, terms), *map(len, labels)])

    def _statistics_of(self, model: str) -> dict[str, Any]:
        """Returns the statistics of a model, by name."""
        if self.statistics is None:
            return {}
        return {
            str(k): v for k, v in self.statistics[model].items()
            if not (not isinstance(v, str) and pd.isna(v))}

    def _stars(self, p_value: float) -> str:
        """Returns the stars of an estimate with `p_value`."""
        if pd.isna(p_value):
            return ''
        return '*' * sum(p_value < level for level in self._levels())

    def _tabulate(
        self,
        columns: Mapping[str, base.Dataset],
        *,
        marginal: bool = False,
        simulations: int = 100,
        step: float = 0.1,
        sets: Any = None,
        validator: str | None = None,
        metrics: Sequence[str] | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Fills in the table from the dataset of each model.

        Args:
            columns: the dataset of each model, by the name of its column.
            marginal: whether to show marginal effects for every model, even
                one that reports its coefficients. Defaults to `False`.
            simulations: number of bootstrap samples for the standard errors
                of marginal effects. Defaults to 100.
            step: standard deviations that each feature is moved up and down
                to find its marginal effect. Defaults to 0.1.
            sets: sets of features (such as the dummy variables of one
                category) whose marginal effects are each from all of the set
                being 0: a list of lists of names, or text such as
                "a, b; c, d" (see `_feature_sets`). Defaults to `None`, which
                moves every feature on its own.
            validator: name of the validator that cross-validates a model if
                the analysis did not. Defaults to `None`, which uses
                `validators` for the task.
            metrics: names of the cross-validated metrics. Defaults to
                `None`, which uses `defaults` for the task.
            **kwargs: not used.

        Returns:
            The estimates (see `estimates`).

        """
        frames = []
        statistics: dict[str, dict[str, Any]] = {}
        self._blocks = {}
        grouped = _feature_sets(sets)
        for name, item in columns.items():
            table, kind = self._estimate(
                item, marginal = marginal, simulations = simulations, step = step,
                sets = grouped)
            self._blocks[name] = _block_of(item, kind, simulations)
            table.index = pd.Index([str(t) for t in table.index], name = 'term')
            frames.append(pd.concat({name: table}, names = ['model']))
            training = item._train_rows()
            statistics[name] = {
                'estimates': kind,
                'observations': int((~training.isin(item.synthetic)).sum()),
                **self._cross_validated(item, validator, metrics)}
        self.estimates = pd.concat(frames).astype(float)
        self.statistics = pd.DataFrame(statistics, dtype = object)
        self.table = self._text()
        return self.estimates

    def _text(self) -> pd.DataFrame:
        """Returns the table as text.

        Returns:
            A column of terms (with no name) and a column for each model.
                Each term has a row for its estimates (with stars) and a row
                for their standard errors (in parentheses), and the
                statistics follow. A term that a model does not have is
                blank.

        """
        if self.estimates is None or self.statistics is None:
            return self._require_table()
        digits = self.digits
        models = [str(c) for c in self.statistics.columns]
        terms = list(dict.fromkeys(
            self.estimates.index.get_level_values('term')))
        rows: list[dict[str, str]] = []
        for term in terms:
            top = {'': term}
            bottom = {'': ''}
            for model in models:
                key = (model, term)
                if key not in self.estimates.index:
                    top[model] = bottom[model] = ''
                    continue
                values = self.estimates.loc[key]
                estimate, error = values['estimate'], values['standard_error']
                top[model] = '' if pd.isna(estimate) else (
                    f'{estimate:.{digits}f}{self._stars(values["p_value"])}')
                bottom[model] = '' if pd.isna(error) else f'({error:.{digits}f})'
            rows.extend([top, bottom])
        for statistic, values in self.statistics.iterrows():
            row = {'': str(statistic)}
            row.update({
                model: _written(values[model], digits) for model in models})
            rows.append(row)
        return pd.DataFrame(rows, columns = ['', *models])


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
            **kwargs: other parameters for `evaluate`.

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
        if (
            item.groups
            and item.task == 'classify'
            and len(item.classes) == 2):  # noqa: PLR2004
            names.extend(self.fairness)
        criterion = branches[0].criterion if branches else None
        kind = chrisjen.library.all.get(criterion) if criterion else None
        if (
            criterion is not None
            and isinstance(kind, type)
            and issubclass(kind, metrics_.Metric)):
            names = [criterion, *(n for n in names if n != criterion)]
        return names

    def _shaded(self) -> int | None:
        """Returns the position of the best branch, if there are several."""
        return 0 if len(self._require_table()) > 1 else None


@dataclasses.dataclass
class ShapImportance(Evaluator):
    """The mean absolute SHAP value of each feature.

    SHAP values explain each prediction as the sum of the contributions of the
    features. The `shap.Explanation` is stored in the dataset's `fitted` for
    further plots. For a classifier, the values are for the last class (the
    positive class of a binary label).

    Wraps:
        [`shap.Explainer`](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html)
        from shap.

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
        # Columns that the model needs besides its features, such as the
        # fixed effects of a `fixest` model, are held at each row's own
        # values while its features are explained.
        others = _other_columns(model, item)
        if isinstance(model, models.LabelCoded):
            model = model.estimator
        train = item.x_train
        sample = train.sample(
            min(background, len(train)), random_state = item.seed)
        x = item.x_test if item.is_split else item.x
        x = x.head(rows)
        given = item.data.loc[x.index, others] if others else None
        explanation = _explain(shap, model, sample, x, item.task, given = given)
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
    task: str | None,
    *,
    given: pd.DataFrame | None = None) -> Any:
    """Returns SHAP values for `x`.

    The explainer is chosen by `shap` for the model. If `shap` does not
    recognize the model, the model's prediction function is explained
    instead. If the model needs columns besides the features (`given`, such
    as the fixed effects of a `fixest` model), each row is explained with
    those columns held at its own values (see `_explain_given`).

    Wraps:
        [`shap.Explainer`](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html)
        from shap.

    Args:
        shap: the `shap` module.
        model: the fitted model.
        background: background rows for the explainer.
        x: rows to explain.
        task: "classify" or "regress".
        given: the other columns of the rows of `x`, which are not explained.
            Defaults to `None`.

    Returns:
        A `shap.Explanation`.

    """
    if given is not None and len(given.columns):
        return _explain_given(shap, _prediction(model, task), background, x, given)
    try:
        explainer = shap.Explainer(model, background)
    except Exception:  # noqa: BLE001
        explainer = shap.Explainer(_prediction(model, task), background)
    try:
        return explainer(x, check_additivity = False)
    except TypeError:
        return explainer(x)


def _explain_given(
    shap: Any,
    predict: Any,
    background: pd.DataFrame,
    x: pd.DataFrame,
    given: pd.DataFrame) -> Any:
    """Returns SHAP values for `x`, each with its own `given` columns held fixed.

    A model with fixed effects predicts a row from its features and its
    fixed effects (such as its court), but only the features are explained.
    Each row is explained by its own explainer of the prediction function
    with that row's fixed effects, so its SHAP values are the effects of its
    features given its fixed effects.

    Args:
        shap: the `shap` module.
        predict: the model's prediction function.
        background: background rows (of the features) for the explainers.
        x: rows to explain.
        given: the other columns of the rows of `x`.

    Returns:
        A `shap.Explanation` of every row of `x`.

    """
    columns = list(x.columns)
    parts = []
    for position in range(len(x)):
        held = given.iloc[position]

        def predicted(values: Any, held: pd.Series = held) -> Any:
            frame = pd.DataFrame(np.asarray(values), columns = columns)
            for column, value in held.items():
                frame[column] = value
            return predict(frame)

        explainer = shap.Explainer(predicted, background)
        parts.append(explainer(x.iloc[[position]]))
    return shap.Explanation(
        values = np.concatenate([np.asarray(p.values) for p in parts]),
        base_values = np.concatenate([np.asarray(p.base_values) for p in parts]),
        data = x.to_numpy(),
        feature_names = columns)


def _prediction(model: Any, task: str | None) -> Any:
    """Returns the model's prediction function: probabilities, if it classifies."""
    if task == 'classify' and hasattr(model, 'predict_proba'):
        return model.predict_proba
    return model.predict


@dataclasses.dataclass
class _GivenColumns:
    """Predicts from features alone, with the model's other columns given.

    scikit-learn's `permutation_importance` passes the model only the
    features of the rows it scores (with one of them shuffled), in the same
    order each time. This adds the other columns that the model needs (such
    as the fixed effects of a `fixest` model) of those rows, by position. It
    also has the tags of a scikit-learn classifier or regressor, which its
    scorers need and which models that are not scikit-learn estimators
    (such as `FixedEffects`) lack.

    Args:
        model: the fitted model.
        given: the other columns of the rows that are scored, in order.
        task: "classify" or "regress".

    """

    model: Any
    given: pd.DataFrame
    task: str | None

    """ Properties """

    @property
    def classes_(self) -> Any:
        """Returns the model's classes."""
        return self.model.classes_

    @property
    def _estimator_type(self) -> str:
        """Returns the kind of estimator, for scikit-learn before 1.6."""
        return 'classifier' if self.task == 'classify' else 'regressor'

    """ Public Methods """

    def fit(self, x: Any, y: Any) -> _GivenColumns:  # noqa: ARG002
        """Does nothing, since the model is fitted.

        scikit-learn's scorers check that an estimator has `fit`.

        Args:
            x: not used.
            y: not used.

        Returns:
            This wrapper.

        """
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the model's predictions.

        Args:
            x: the features of the rows of `given`, in order.

        Returns:
            The predictions.

        """
        return np.asarray(self.model.predict(_with_given(x, self.given)))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Returns the model's predicted probability of each class.

        Args:
            x: the features of the rows of `given`, in order.

        Returns:
            One column for each class.

        """
        return np.asarray(
            self.model.predict_proba(_with_given(x, self.given)))

    """ Dunder Methods """

    def __sklearn_tags__(self) -> Any:
        """Returns scikit-learn's tags for a classifier or regressor.

        Returns:
            The tags (scikit-learn 1.6 and later).

        """
        utils = importlib.import_module('sklearn.utils')
        classify = self.task == 'classify'
        return utils.Tags(
            estimator_type = self._estimator_type,
            target_tags = utils.TargetTags(required = True),
            classifier_tags = utils.ClassifierTags() if classify else None,
            regressor_tags = None if classify else utils.RegressorTags())


def _other_columns(model: Any, item: base.Dataset) -> list[Any]:
    """Returns the columns that the model needs besides its features.

    Args:
        model: the fitted model.
        item: the dataset it was fitted to.

    Returns:
        The columns that the model's parameters name, such as the fixed
            effects and cluster of a `fixest` model (see
            `validators._columns`), or an empty `list`.

    """
    # The validators module imports this one, so it is imported when used.
    from . import validators  # noqa: PLC0415
    features = set(validators._features(model, item))
    return [c for c in validators._columns(model, item) if c not in features]


def _with_given(x: Any, given: pd.DataFrame) -> pd.DataFrame:
    """Returns the features `x` with the columns of `given`, by position.

    Args:
        x: features of the rows of `given`, in the same order.
        given: other columns of the same rows.

    Raises:
        ValueError: if `x` and `given` do not have the same number of rows.

    Returns:
        `x` with the columns of `given`.

    """
    frame = pd.DataFrame(x).copy()
    if len(frame) != len(given):
        message = (
            f'{len(frame)} rows cannot be given the other columns of '
            f'{len(given)} rows')
        raise ValueError(message)
    for column in given.columns:
        frame[column] = given[column].to_numpy()
    return frame


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


def _kaiser(data: pd.DataFrame) -> int:
    """Returns the number of eigenvalues of the correlations above 1.

    Args:
        data: numeric columns.

    Returns:
        The number of factors that Kaiser's rule keeps (at least 1).

    """
    eigenvalues = np.linalg.eigvalsh(data.dropna().corr().to_numpy())
    return max(1, int((eigenvalues > 1).sum()))


def _numbers(
    item: base.Dataset,
    columns: Sequence[str] | None,
    use: str) -> pd.DataFrame:
    """Returns the numeric columns of the real training rows, as floats.

    Args:
        item: the dataset.
        columns: the columns to use, or `None` for the numeric and boolean
            features.
        use: what uses the columns (for the message of an error).

    Raises:
        ValueError: if fewer than two of the columns vary.

    Returns:
        The columns that vary, in the training rows that are not `synthetic`.

    """
    training = item._train_rows()
    rows = training[~training.isin(item.synthetic)]
    if columns is None:
        kinds = {*item.numerics, *item.booleans}
        columns = [c for c in item.features if c in kinds]
    data = item.data.loc[rows, list(columns)].astype(float)
    data = data.loc[:, data.std() > 0]
    if data.shape[1] < 2:  # noqa: PLR2004
        message = f'{use} needs at least two numeric columns that vary'
        raise ValueError(message)
    return data


def _latex(text: str) -> str:
    """Returns `text` with the characters that LaTeX treats specially escaped.

    Args:
        text: text for a LaTeX table.

    Returns:
        The escaped text. A dash for a missing value becomes "---".

    """
    if text == '—':
        return '---'
    return text.translate(_LATEX_ESCAPES)


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


@dataclasses.dataclass(frozen = True)
class _Move:
    """How a feature is moved to find its marginal effect.

    A feature is moved up and down by `shift` from each row's value, or, if
    `shift` is `None`, set to `low` and then to `high` in every row. A
    feature in a set of features has `others`: the rest of its set, which
    are set to their lower values in both, so that it moves from the set's
    reference to itself alone.

    Args:
        shift: how far the feature is moved each way, or `None`.
        low: the lower value of a feature with two values.
        high: the higher value of a feature with two values.
        width: the distance between the two values of the feature, which the
            change in the prediction is divided by.
        others: the other features of its set, with the value each is set to.
            Defaults to none.

    """

    shift: float | None
    low: Any
    high: Any
    width: float
    others: tuple[tuple[str, Any], ...] = ()

    def apply(self, x: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Returns the rows of `x` with `feature` moved down, then up.

        Args:
            x: the rows.
            feature: the feature to move.

        Returns:
            Twice as many rows as `x`: those moved down, then those moved up.

        """
        low, high = x.copy(), x.copy()
        if self.shift is None:
            low[feature] = self.low
            high[feature] = self.high
        else:
            low[feature] = x[feature] - self.shift
            high[feature] = x[feature] + self.shift
        for other, value in self.others:
            low[other] = value
            high[other] = value
        return pd.concat([low, high], ignore_index = True)


def _branch_names(branches: Sequence[base.Branch]) -> list[str]:
    """Returns a short name for each branch.

    Args:
        branches: branches of an experiment.

    Returns:
        The techniques of the steps that differ between the branches, joined
            by " / ", or the technique of a branch's last step if no step
            differs.

    """
    steps = list(dict.fromkeys(k for b in branches for k in b.steps))
    differ = [
        step for step in steps
        if len({b.steps.get(step) for b in branches}) > 1]
    names = []
    for branch in branches:
        if differ:
            names.append(' / '.join(
                str(branch.steps.get(step, '')) for step in differ))
        elif branch.steps:
            names.append(str(list(branch.steps.values())[-1]))
        else:
            names.append(branch.label)
    return names


def _draw(y: pd.Series, rng: np.random.Generator, task: str | None) -> Any:
    """Returns the positions of the rows of a bootstrap sample.

    Args:
        y: the labels of the rows.
        rng: the random number generator.
        task: "classify" or "regress".

    Returns:
        As many positions as rows, drawn with replacement. To classify, each
            class keeps its number of rows, so that every class is learned.

    """
    if task != 'classify':
        return rng.integers(0, len(y), len(y))
    codes = pd.factorize(y)[0]
    positions = [
        rng.choice(np.flatnonzero(codes == code), size = int(count))
        for code, count in zip(
            *np.unique(codes, return_counts = True), strict = True)]
    return rng.permutation(np.concatenate(positions))


def _effects(
    model: Any,
    x: pd.DataFrame,
    moves: dict[str, _Move],
    task: str | None) -> pd.Series:
    """Returns the average marginal effect of each feature in `moves`.

    Args:
        model: the fitted model.
        x: the rows to average over.
        moves: how each feature is moved.
        task: "classify" or "regress".

    Returns:
        The average change in the prediction for a one-unit increase in each
            feature, or, for a classifier, in the predicted probability of
            the positive class (of two) or of each class (named
            "{feature} ({class})").

    """
    effects: dict[str, float] = {}
    count = len(x)
    for feature, move in moves.items():
        predicted, labels = _outcomes(model, move.apply(x, feature), task)
        changes = predicted[count:] - predicted[:count]
        if np.isnan(changes).all():
            averages = np.full(changes.shape[1], np.nan)
        else:
            averages = np.nanmean(changes, axis = 0) / move.width
        if labels is None:
            effects[feature] = float(averages[0])
        else:
            effects.update({
                f'{feature} ({label})': float(value)
                for label, value in zip(labels, averages, strict = True)})
    return pd.Series(effects, dtype = float)


def _feature_sets(sets: Any) -> list[list[str]]:
    """Returns sets of features from the forms that "sets" can take.

    Args:
        sets: `None`; a list of lists of feature names; or text with sets
            separated by semicolons and names by commas ("a, b; c, d"), as a
            string or as the list that a settings file makes of it by
            splitting it at the commas (["a", "b; c", "d"]).

    Returns:
        A list of sets, each a list of names. Empty sets are left out.

    """
    if sets is None:
        return []
    if isinstance(sets, str) or all(isinstance(s, str) for s in sets):
        text = sets if isinstance(sets, str) else ', '.join(sets)
        parts = [part.split(',') for part in text.split(';')]
    else:
        parts = [list(members) for members in sets]
    found = [[str(m).strip() for m in part if str(m).strip()] for part in parts]
    return [members for members in found if members]


def _marginal_effects(
    item: base.Dataset,
    model: Any,
    simulations: int,
    step: float,
    sets: Sequence[Sequence[str]] = ()) -> pd.DataFrame:
    """Returns the average marginal effects of the features of a model.

    Wraps:
        - [`sklearn.base.clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
          from scikit-learn.
        - [`scipy.stats.norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
          from SciPy.

    Args:
        item: the dataset with a fitted model.
        model: the fitted model.
        simulations: number of bootstrap samples.
        step: standard deviations that each feature is moved up and down.
        sets: sets of features whose members are moved one at a time from
            all of them being at their lower values. Defaults to none.

    Returns:
        One row for each term (see `_effects`), with the columns of
            `RegressionTable.estimates`. The estimates average over the real
            training rows, and their standard errors and 95% intervals come
            from refitting copies of the model to bootstrap samples of those
            rows. Each sampler that was applied to the training rows is
            applied again to each sample, as a validator does.

    """
    # The validators module imports this one, so it is imported when used.
    from . import validators  # noqa: PLC0415
    clone = importlib.import_module('sklearn.base').clone
    stats = importlib.import_module('scipy.stats')
    training = item._train_rows()
    rows = training[~training.isin(item.synthetic)]
    columns = validators._columns(model, item)
    x = item.data.loc[rows, columns].reset_index(drop = True)
    y = item.y.loc[rows].reset_index(drop = True)
    moves = _moves(x, _movable(model, item, columns), step, sets)
    estimates = _effects(model, x, moves, item.task)
    draws = []
    if len(estimates) and simulations > 1:
        rng = np.random.default_rng(item.seed)
        samplers = [
            tool for tool in item.fitted.values()
            if hasattr(tool, 'fit_resample')]
        for _ in range(simulations):
            positions = _draw(y, rng, item.task)
            x_draw = x.iloc[positions].reset_index(drop = True)
            y_draw = y.iloc[positions].reset_index(drop = True)
            x_learn, y_learn = x_draw, y_draw
            for sampler in samplers:
                x_learn, y_learn = clone(sampler).fit_resample(x_learn, y_learn)
            estimator = clone(model).fit(x_learn, y_learn)
            effects = _effects(estimator, x_draw, moves, item.task)
            draws.append(effects.reindex(estimates.index).to_numpy())
    values = estimates.to_numpy()
    if draws:
        sample = np.vstack(draws)
        errors = np.nanstd(sample, axis = 0, ddof = 1)
        lower, upper = np.nanpercentile(sample, [2.5, 97.5], axis = 0)
    else:
        errors = lower = upper = np.full(len(values), np.nan)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        statistic = np.where(errors > 0, values / errors, np.nan)
    return pd.DataFrame(
        {
            'estimate': values,
            'standard_error': errors,
            'statistic': statistic,
            'p_value': 2 * stats.norm.sf(np.abs(statistic)),
            'ci_lower': lower,
            'ci_upper': upper},
        index = estimates.index)


def _model_name(item: base.Dataset) -> str:
    """Returns the name of the technique that fitted the model of `item`.

    Args:
        item: a dataset with a fitted model.

    Returns:
        The name of the last model in `history` (the last entry with a
            "search", which only models record), or "model".

    """
    for entry in reversed(item.history):
        if 'search' in entry:
            return str(entry.get('technique'))
    return 'model'


def _movable(
    model: Any,
    item: base.Dataset,
    columns: Sequence[Any]) -> list[str]:
    """Returns the features that a marginal effect can be found for.

    Args:
        model: the fitted model.
        item: the dataset it was fitted to.
        columns: the columns it was fitted to.

    Returns:
        The model's features that are numbers or booleans and vary.
            Categorical columns, and columns that the model's parameters name
            (such as fixed effects) but that are not its features, are left
            out.

    """
    names = utilities.feature_names(model, item.data.columns)
    features = list(item.features if names is None else names)
    movable = []
    for feature in features:
        if feature not in columns or feature not in item.features:
            continue
        values = item.data[feature]
        if (
            isinstance(values.dtype, pd.CategoricalDtype)
            or not (
                pd.api.types.is_numeric_dtype(values.dtype)
                or pd.api.types.is_bool_dtype(values.dtype))
            or values.nunique() < 2):  # noqa: PLR2004
            continue
        movable.append(feature)
    return movable


def _moves(
    x: pd.DataFrame,
    features: Sequence[str],
    step: float,
    sets: Sequence[Sequence[str]] = ()) -> dict[str, _Move]:
    """Returns how each feature is moved to find its marginal effect.

    Args:
        x: the rows that the model learned from.
        features: the features to move.
        step: standard deviations that each feature is moved up and down.
        sets: sets of features whose members are each moved with the rest
            of their set at its lower values. Members that are not in
            `features` are left out. Defaults to none.

    Raises:
        ValueError: if a member of a set has more than two values, or a
            feature is in more than one set.

    Returns:
        A `_Move` for each feature: from false to true for a boolean, from
            the lower to the higher value for a feature of two values, and
            up and down by `step` standard deviations for any other. A member
            of a set also sets the rest of its set to their lower values.

    """
    moves = _single_moves(x, features, step)
    seen: set[str] = set()
    for members in sets:
        present = [m for m in dict.fromkeys(members) if m in moves]
        for member in present:
            if member in seen:
                message = f'{member!r} is in more than one set of features'
                raise ValueError(message)
            seen.add(member)
            if moves[member].shift is not None:
                message = (
                    f'{member!r} is in a set of features, but it has more '
                    f'than two values'
                )
                raise ValueError(message)
        for member in present:
            others = tuple(
                (other, moves[other].low) for other in present if other != member)
            moves[member] = dataclasses.replace(moves[member], others = others)
    return moves


def _single_moves(
    x: pd.DataFrame,
    features: Sequence[str],
    step: float) -> dict[str, _Move]:
    """Returns how each feature is moved on its own (see `_moves`).

    Args:
        x: the rows that the model learned from.
        features: the features to move.
        step: standard deviations that each feature is moved up and down.

    Returns:
        A `_Move` for each feature that varies.

    """
    moves = {}
    for feature in features:
        values = x[feature]
        if pd.api.types.is_bool_dtype(values.dtype):
            moves[feature] = _Move(None, low = False, high = True, width = 1.0)
            continue
        distinct = values.dropna().unique()
        if len(distinct) == 2:  # noqa: PLR2004
            low, high = sorted(distinct)
            moves[feature] = _Move(
                None, low = low, high = high, width = float(high - low))
            continue
        shift = step * float(values.std())
        if shift > 0:
            moves[feature] = _Move(
                shift, low = None, high = None, width = 2 * shift)
    return moves


def _outcomes(
    model: Any,
    x: pd.DataFrame,
    task: str | None) -> tuple[np.ndarray, list[str] | None]:
    """Returns the predictions whose changes are marginal effects.

    Args:
        model: the fitted model.
        x: the rows to predict.
        task: "classify" or "regress".

    Returns:
        A column of predictions (or of the predicted probability of the
            positive class, for a classifier of two classes), or a column for
            the predicted probability of each class and the names of the
            classes.

    """
    if task == 'classify':
        probabilities = np.asarray(model.predict_proba(x), dtype = float)
        if probabilities.shape[1] <= 2:  # noqa: PLR2004
            return probabilities[:, -1:], None
        classes = getattr(model, 'classes_', range(probabilities.shape[1]))
        return probabilities, [str(c) for c in classes]
    predictions = np.asarray(model.predict(x), dtype = float)
    return predictions.reshape(len(x), -1)[:, :1], None


def _unique(names: Sequence[str]) -> list[str]:
    """Returns `names`, with a number added to each repeat of a name.

    Args:
        names: names of columns.

    Returns:
        The names, with " (2)", " (3)", and so on added to repeats.

    """
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f'{name} ({seen[name]})')
    return unique


def _written(value: Any, digits: int) -> str:
    """Returns `value` written as text for a published table.

    Args:
        value: a value of a table.
        digits: digits after the decimal point of a number.

    Returns:
        Text as it is, whole numbers as they are, other numbers rounded to
            `digits`, and a dash for a missing value.

    """
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return '—'
    if isinstance(value, bool | int | np.integer):
        return str(value)
    if isinstance(value, float | np.floating):
        return f'{value:.{digits}f}'
    return str(value)


@dataclasses.dataclass(frozen = True)
class _Block:
    """What a model's block in the "stata" style of a table says about it.

    Args:
        label: the name of the label (the dependent variable), which titles
            the column of terms. Defaults to "y".
        coefficient: the title of the column of estimates: "Coef." or
            "dy/dx" (for marginal effects). Defaults to "Coef.".
        test: "t" or "z", the test statistic. Defaults to "z".
        expression: what marginal effects are changes in, such as
            "Pr(target = 1)", or `None`. Defaults to `None`.
        replications: the number of bootstrap samples, or `None`. Defaults to
            `None`.

    """

    label: str = 'y'
    coefficient: str = 'Coef.'
    test: str = 'z'
    expression: str | None = None
    replications: int | None = None


@dataclasses.dataclass(frozen = True)
class _Line:
    """A line of the "stata" style of a regression table.

    Args:
        kind: "text", "row" (a row of estimates, with a bar after the term),
            "top" and "bottom" (rules), or "cross" (a rule crossed by the
            bar).
        text: the text of a "text" or "row" line.
        width: the width of the table, in characters.
        bar: the position of the bar after the terms, in characters.

    """

    kind: str
    text: str
    width: int
    bar: int


def _block_of(item: base.Dataset, kind: str, simulations: int) -> _Block:
    """Returns what a model's block in the "stata" style says about it.

    Args:
        item: the dataset with the fitted model.
        kind: the kind of its estimates ("coefficients", "marginal
            effects", or "none ...").
        simulations: the number of bootstrap samples of marginal effects.

    Returns:
        The block. Coefficients have a t statistic if the model's results
            use the t distribution (as least squares and pyfixest do), and
            a z statistic otherwise.

    """
    label = str(item.label)
    if kind == 'coefficients':
        model = item.model
        results = getattr(model, 'results', None)
        uses_t = (
            bool(getattr(results, 'use_t', False))
            or isinstance(model, models.FixedEffects))
        return _Block(label, 'Coef.', 't' if uses_t else 'z')
    expression = f'Predicted {label}'
    if item.task == 'classify':
        classes = item.classes
        expression = (
            f'Pr({label} = {classes[-1]})' if len(classes) == 2  # noqa: PLR2004
            else f'Pr({label} = each class)')
    replications = simulations if kind == 'marginal effects' else None
    return _Block(
        label,
        'dy/dx',
        'z',
        expression,
        replications if replications and replications > 1 else None)


def _border(cell: Any, edges: Sequence[str]) -> None:
    """Draws single black borders on some edges of a cell of a Word table.

    Args:
        cell: a `docx` table cell.
        edges: "top", "left", "bottom", or "right".

    """
    if not edges:
        return
    element = utilities.import_tool('docx.oxml.OxmlElement')
    qualified = utilities.import_tool('docx.oxml.ns.qn')
    borders = element('w:tcBorders')
    # Word needs the edges in this order.
    for edge in ('top', 'left', 'bottom', 'right'):
        if edge in edges:
            line = element(f'w:{edge}')
            line.set(qualified('w:val'), 'single')
            line.set(qualified('w:sz'), '6')
            line.set(qualified('w:space'), '0')
            line.set(qualified('w:color'), '000000')
            borders.append(line)
    cell._tc.get_or_add_tcPr().append(borders)


def _fixed(value: Any, digits: int) -> str:
    """Returns `value` with `digits` after the decimal point, or a dot."""
    if value is None or pd.isna(value):
        return '.'
    return f'{value:.{digits}f}'


def _stata_number(value: Any) -> str:
    """Returns a number as Stata's `%9.0g` format writes it.

    Args:
        value: the number.

    Returns:
        The number with as many significant digits (up to 7) as fit in 8
            characters and a sign, without a 0 before the decimal point
            (such as "-.0065879"), or a dot if it is missing.

    """
    if value is None or pd.isna(value):
        return '.'
    text = ''
    for digits in range(7, 0, -1):
        text = f'{value:.{digits}g}'
        if text.startswith(('0.', '-0.')):
            text = text.replace('0.', '.', 1)
        # A sign takes the ninth character, so a number has eight.
        if len(text.removeprefix('-')) <= 8:  # noqa: PLR2004
            break
    return text


def _stata_value(value: Any) -> str:
    """Returns a statistic as Stata writes it above a table.

    Args:
        value: a whole number, another number, text, or `None`.

    Returns:
        A whole number as it is, another number with 4 digits after the
            decimal point, text as it is, or a dot if it is missing.

    """
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return '.'
    if isinstance(value, bool | int | np.integer):
        return str(value)
    return f'{value:.4f}'
