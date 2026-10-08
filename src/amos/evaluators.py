"""Techniques that evaluate a fitted model with tables.

These are techniques of the "critic" stage, beside the metrics. Each one adds
a table to the dataset's `tables` (under the technique's name). Most evaluate
the model on the test rows (or on every row, if the data has not been split).
`factor_analysis` and `pca` instead describe the features
that the model learned from, so they work in any stage.

Contents:
    Evaluator: base class for techniques that evaluate a model with a table.
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
    Scorecard: every standard metric for every branch of an analysis, ready
        to publish as csv, Markdown, LaTeX, HTML, Word, or an image.
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
    '^': r'\textasciicircum{}'})


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
        columns = list(getattr(model, 'feature_names_in_', item.features))
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

    def implement(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Adds the table and stores the coverage and width (or set size).

        Args:
            item: the dataset with a fitted model.
            **kwargs: parameters for `evaluate`.

        Returns:
            The dataset, with the table in `tables` and the coverage in
                `metrics`.

        """
        super().implement(item, **kwargs)
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
        names = [str(n) for n in getattr(
            model, 'feature_names_in_', item.features)]
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
    Coefficients are only comparable if the features were scaled. For other
    models, use `permutation_importance`.

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
        features = list(getattr(model, 'feature_names_in_', item.features))
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
    (`to_markdown`), a LaTeX table (`to_latex`), an HTML table (`to_html`,
    which needs great_tables), a Word document (`to_word`, which needs
    python-docx), or an image (`to_image`, which needs matplotlib), or in
    several at once (`export`). If the dataset has groups and a label of two
    classes, fairness metrics are included. Make one from a dataset or an
    applied project with `create`, or use `Project.scorecard`.

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

    def export(
        self,
        folder: pathlib.Path | str,
        *,
        name: str | None = None,
        formats: Sequence[str] = ('csv', 'md', 'docx', 'png'),
        ) -> dict[str, pathlib.Path]:
        """Saves the scorecard in several formats.

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
            path = folder / f'{name or self.name}.{extension}'
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
        text = self._require_table().to_csv(
            index = False, lineterminator = '\n')
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

    def to_html(self, path: pathlib.Path | str | None = None) -> str:
        """Returns the scorecard as an HTML table, and saves it if asked.

        The table is made by great_tables, with the heading as its title,
        numbers aligned to the right, and the best branch shaded.

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
        if len(text) > 1:
            table = table.tab_style(
                style = great_tables.style.fill(color = '#e3f1e3'),
                locations = great_tables.loc.body(rows = [0]))
        html = str(table.as_raw_html())
        if path is not None:
            pathlib.Path(path).write_text(html, encoding = 'utf-8')
        return html

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

    def to_latex(self, path: pathlib.Path | str | None = None) -> str:
        r"""Returns the scorecard as a LaTeX table, and saves it if asked.

        The table uses the booktabs package, so add `\usepackage{booktabs}`
        to the preamble of the document. Its caption is the heading.

        Args:
            path: file to save the table in. Defaults to `None`.

        Returns:
            The LaTeX table.

        """
        text = self._text()
        numeric = self._numeric_columns()
        alignment = ''.join('r' if c in numeric else 'l' for c in text.columns)
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            rf'\caption{{{_latex(self.heading)}}}',
            rf'\begin{{tabular}}{{{alignment}}}',
            r'\toprule',
            ' & '.join(_latex(str(c)) for c in text.columns) + r' \\',
            r'\midrule']
        lines.extend(
            ' & '.join(_latex(value) for value in row) + r' \\'
            for row in text.itertuples(index = False))
        lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
        latex = '\n'.join(lines) + '\n'
        if path is not None:
            pathlib.Path(path).write_text(latex, encoding = 'utf-8')
        return latex

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
