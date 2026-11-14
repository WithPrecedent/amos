"""The regression table of one or more models.

Contents:
    RegressionTable: the coefficients (or average marginal effects) of every
        model side by side, with their standard errors and cross-validated
        metrics, ready to publish like a scorecard.

"""

from __future__ import annotations

import dataclasses
import importlib
import itertools
import pathlib
import textwrap
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import camina
import chrisjen
import numpy as np
import pandas as pd

from .. import base, models, utilities
from ._base import Evaluator, _Published, _model, _written

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

    visual: ClassVar[str] = 'estimates_plot'

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
        names = names or _unique(
            [d.model_name or 'model' for d in datasets])
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
            **kwargs: parameters for `evaluate`, and `visualize` (see
                `Evaluator.apply`).

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
            return {item.model_name or 'model': item}
        kept = [
            branch for branch in item.branches
            if branch.dataset is not None and branch.dataset.model is not None]
        if not kept:
            warnings.warn(
                'the datasets of the branches are no longer in memory, so '
                'the regression table has only the final model (an '
                'experiment keeps them only if "conserve_memory" is false)',
                category = base.AmosWarning,
                stacklevel = 3)
            return {item.model_name or 'model': item}
        if len(kept) < len(item.branches):
            lost = [b.label for b in item.branches if b not in kept]
            warnings.warn(
                f'the datasets of some branches are no longer in memory, so '
                f'they are not in the regression table (an experiment keeps '
                f'them only if "conserve_memory" is false): {lost}',
                category = base.AmosWarning,
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

        Raises:
            TypeError: if `validator` does not name a validator.

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
            from .. import validators as validators_  # noqa: PLC0415
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
        """Returns the table that `to_csv` saves: the text of the table.

        Returns:
            The table as it is written, with each estimate and its
                standard error (or interval) as text.

        """
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
        """Returns the meaning of the parentheses and stars.

        Returns:
            The notes to write below the table.

        """
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
        """Returns the names of the columns of the models.

        Returns:
            The names of the columns that hold the estimates of each
                model.

        """
        if self.statistics is None:
            return set()
        return {str(c) for c in self.statistics.columns}

    def _rules(self) -> list[int]:
        """Returns the position of the first row below the estimates.

        Returns:
            The position (from 0) of the row of the first statistic, in a
                `list` that is empty if there are no statistics.

        """
        if self.estimates is None or self.estimates.empty:
            return []
        terms = self.estimates.index.get_level_values('term').unique()
        return [2 * len(terms)]

    def _models(self) -> list[str]:
        """Returns the names of the models in the table, in order.

        Returns:
            The names of the models.

        """
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
            """Returns a line of the table, as wide as the table.

            Args:
                kind: the kind of line (such as "text" or "rule").
                text: the text of the line. Defaults to an empty `str`.

            Returns:
                The line.

            """
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
        """Returns notes on marginal effects for the "stata" style, if any.

        Returns:
            The notes, or `None` if no model's estimates are marginal
                effects.

        """
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
        """Returns the width of the column of terms in the "stata" style.

        Returns:
            The number of characters of the longest term or statistic,
                and at least the width that Stata uses.

        """
        terms: list[str] = (
            [] if self.estimates is None
            else [
                str(t) for t in self.estimates.index.get_level_values('term')])
        labels = [block.label for block in self._blocks.values()]
        return max([12, *map(len, terms), *map(len, labels)])

    def _statistics_of(self, model: str) -> dict[str, Any]:
        """Returns the statistics of a model, by name.

        Args:
            model: name of a model in the table.

        Returns:
            The statistics of the model that are not missing, by name.

        """
        if self.statistics is None:
            return {}
        return {
            str(k): v for k, v in self.statistics[model].items()
            if not (not isinstance(v, str) and pd.isna(v))}

    def _stars(self, p_value: float) -> str:
        """Returns the stars of an estimate with `p_value`.

        Args:
            p_value: the p-value of the estimate.

        Returns:
            One star for each significance level that `p_value` is below,
                or an empty `str` if it is missing.

        """
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
            statistics[name] = {
                'estimates': kind,
                'observations': len(item.real_train_rows),
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


""" Private Functions """


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
            by " / ". If no step differs (as when only one branch is left),
            the name of a branch's model, or else the technique of its last
            step, which may be a validator after the model.

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
        elif branch.dataset is not None and branch.dataset.model is not None:
            names.append(branch.dataset.model_name or 'model')
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
    from .. import validators  # noqa: PLC0415
    clone = importlib.import_module('sklearn.base').clone
    stats = importlib.import_module('scipy.stats')
    rows = item.real_train_rows
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
    """Returns `value` with `digits` after the decimal point, or a dot.

    Args:
        value: a number, or a missing value.
        digits: number of digits after the decimal point.

    Returns:
        The number as text, or "." (as Stata writes it) if it is
            missing.

    """
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
