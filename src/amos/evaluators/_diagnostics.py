"""Evaluators of how well a model predicts and what it learned.

Contents:
    BiasVariance: the model's expected error, as its bias and variance.
    ClassificationReport: precision, recall, and f1 for each class.
    Confusion: how many rows of each class were predicted to be each class.
    Conformal: prediction intervals (or sets) with a known rate of coverage.
    ExplainWeights: eli5's explanation of the weights of the features.
    FactorAnalysis: the hidden factors that explain the correlations of the
        features.
    Fairness: how the model does for each group, and the gaps between them.
    FeatureImportance: the importance that the model gives each feature.
    PCA: how much of the variance of the features each principal component has.
    PermutationImportance: how much the score drops when each feature is
        shuffled.
    ShapImportance: the mean absolute SHAP value of each feature.

"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .. import base, models, utilities
from .. import metrics as metrics_
from ._base import Evaluator, _model


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

    visual: ClassVar[str] = 'bias_variance_plot'

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
            IncompatibleTechniqueError: if the model is not from scikit-learn.

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
            raise base.IncompatibleTechniqueError(message)
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

    visual: ClassVar[str] = 'classification_report_heatmap'

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

    visual: ClassVar[str] = 'confusion_heatmap'

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

    visual: ClassVar[str] = 'prediction_intervals'

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
            **kwargs: parameters for `evaluate`, and `visualize` (see
                `Evaluator.apply`).

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

    visual: ClassVar[str] = 'weights_plot'

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
            IncompatibleTechniqueError: if eli5 cannot explain the model.

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
            raise base.IncompatibleTechniqueError(message)
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

    visual: ClassVar[str] = 'loadings_heatmap'

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

    visual: ClassVar[str] = 'fairness_plot'

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

    visual: ClassVar[str] = 'importance_plot'

    def evaluate(self, item: base.Dataset, **kwargs: Any) -> pd.DataFrame:
        """Returns the importance of each feature, largest first.

        Args:
            item: the dataset with a fitted model.
            **kwargs: not used.

        Raises:
            IncompatibleTechniqueError: if the model does not report the
                importance of its features.

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
                raise base.IncompatibleTechniqueError(message)
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

    visual: ClassVar[str] = 'scree_plot'

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

    visual: ClassVar[str] = 'importance_plot'

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

    visual: ClassVar[str] = 'importance_plot'

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
            """Returns the predictions with the other columns held.

            Args:
                values: the values of the explained columns, which shap
                    varies.
                held: the values of the other columns, which are held
                    at those of the explained row.

            Returns:
                The model's predictions.

            """
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
    """Returns the model's prediction function.

    Args:
        model: a fitted model.
        task: "classify" or "regress".

    Returns:
        The model's `predict_proba` if it classifies (and has one), and
            otherwise its `predict`.

    """
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
    from .. import validators  # noqa: PLC0415
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
    rows = item.real_train_rows
    if columns is None:
        kinds = {*item.numerics, *item.booleans}
        columns = [c for c in item.features if c in kinds]
    data = item.data.loc[rows, list(columns)].astype(float)
    data = data.loc[:, data.std() > 0]
    if data.shape[1] < 2:  # noqa: PLR2004
        message = f'{use} needs at least two numeric columns that vary'
        raise ValueError(message)
    return data


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
