"""Tests that amos gives the same results as each package it wraps.

The README lists the packages that `amos` wraps. For each one, a test here
applies an `amos` technique and checks that the result is the same as using
the package directly, with the same data, parameters, and seed. The first test
checks that every package in the README's table has such a test.
"""

from __future__ import annotations

import importlib
import pathlib
import re

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_mixed, requires

import amos

ROOT = pathlib.Path(__file__).parent.parent
# The packages that do not use their own names when imported.
IMPORT_NAMES = {
    'imbalanced-learn': 'imblearn',
    'python-docx': 'docx',
    'scikit-learn': 'sklearn'}


def _readme_packages() -> list[str]:
    """Returns the packages in the first column of the README's package table."""
    text = (ROOT / 'README.md').read_text(encoding = 'utf-8')
    table = text.split('| Package | What `amos` uses it for |')[1]
    rows = [r for r in table.split('\n\n')[0].splitlines()[2:] if r]
    packages = []
    for row in rows:
        first = row.split('|')[1]
        packages.extend(re.findall(r'\[([\w-]+)\]\(', first))
    return packages


def test_every_package_in_the_readme_has_a_test() -> None:
    packages = _readme_packages()
    assert 'scikit-learn' in packages
    assert len(packages) == 10
    for package in packages:
        name = IMPORT_NAMES.get(package, package.replace('-', '_'))
        assert callable(globals().get(f'test_{name}')), package


def _mixed() -> amos.Dataset:
    """Returns the mixed dataset, split, with its missing ages filled."""
    dataset = amos.Dataset(make_mixed(), label = 'outcome', seed = SEED)
    amos.cleaners.DropColumns().apply(dataset, columns = ['joined'])
    amos.splitters.Stratified().apply(dataset)
    return amos.transformers.MedianImpute().apply(dataset)


def test_sklearn(classified: amos.Dataset) -> None:
    preprocessing = importlib.import_module('sklearn.preprocessing')
    linear_model = importlib.import_module('sklearn.linear_model')
    metrics = importlib.import_module('sklearn.metrics')
    x_train, x_test = classified.x_train, classified.x_test
    y_train, y_test = classified.y_train, classified.y_test
    amos.transformers.Standard().apply(classified)
    amos.models.Logit().apply(classified)
    amos.metrics.RocAuc().apply(classified)
    scaler = preprocessing.StandardScaler().fit(x_train)
    expected = scaler.transform(x_test)
    np.testing.assert_allclose(classified.x_test.to_numpy(), expected)
    model = linear_model.LogisticRegression(
        max_iter = 1000, random_state = SEED).fit(
            scaler.transform(x_train), y_train)
    probabilities = model.predict_proba(expected)
    np.testing.assert_allclose(
        classified.probabilities.to_numpy(), probabilities)
    assert classified.metrics['roc_auc'] == pytest.approx(
        metrics.roc_auc_score(y_test, probabilities[:, 1]))


def test_category_encoders() -> None:
    requires('category_encoders')
    encoders = importlib.import_module('category_encoders')
    dataset = _mixed()
    region = dataset.data[['region']].copy()
    train = dataset.x_train[['region']]
    codes = dataset.y_train.map({'no': 0, 'yes': 1})
    amos.transformers.Target().apply(dataset, columns = ['region'])
    expected = encoders.TargetEncoder().fit(train, codes).transform(region)
    np.testing.assert_allclose(
        dataset.data['region'].to_numpy(), expected['region'].to_numpy())


def test_imblearn(classified: amos.Dataset) -> None:
    requires('imblearn')
    over_sampling = importlib.import_module('imblearn.over_sampling')
    x_train, y_train = classified.x_train, classified.y_train
    test = classified.data.loc[classified.test].copy()
    amos.samplers.Smote().apply(classified)
    x, y = over_sampling.SMOTE(random_state = SEED).fit_resample(
        x_train, y_train)
    np.testing.assert_allclose(classified.x_train.to_numpy(), x.to_numpy())
    np.testing.assert_array_equal(classified.y_train.to_numpy(), y.to_numpy())
    pd.testing.assert_frame_equal(classified.data.loc[classified.test], test)


def test_xgboost(classified: amos.Dataset) -> None:
    requires('xgboost')
    xgboost = importlib.import_module('xgboost')
    amos.models.Xgboost().apply(classified, n_estimators = 10)
    model = xgboost.XGBClassifier(n_estimators = 10, random_state = SEED)
    model.fit(classified.x_train, classified.y_train)
    np.testing.assert_allclose(
        classified.probabilities.to_numpy(),
        model.predict_proba(classified.x_test))


def test_lightgbm(regressed: amos.Dataset) -> None:
    requires('lightgbm')
    lightgbm = importlib.import_module('lightgbm')
    amos.models.Lightgbm().apply(regressed, n_estimators = 10)
    model = lightgbm.LGBMRegressor(
        n_estimators = 10, verbose = -1, random_state = SEED)
    model.fit(regressed.x_train, regressed.y_train)
    np.testing.assert_allclose(
        regressed.predictions.to_numpy(), model.predict(regressed.x_test))


def test_statsmodels(regressed: amos.Dataset) -> None:
    requires('statsmodels')
    api = importlib.import_module('statsmodels.api')
    amos.models.OLS().apply(regressed)
    table = regressed.tables['ols_coefficients']
    results = api.OLS(
        regressed.y_train.astype(float),
        api.add_constant(regressed.x_train.astype(float))).fit()
    np.testing.assert_allclose(table['coefficient'], results.params)
    np.testing.assert_allclose(table['standard_error'], results.bse)
    np.testing.assert_allclose(table['p_value'], results.pvalues)
    np.testing.assert_allclose(
        table[['ci_lower', 'ci_upper']].to_numpy(),
        results.conf_int().to_numpy())


def test_statsmodels_logistic_regression(classified: amos.Dataset) -> None:
    requires('statsmodels')
    api = importlib.import_module('statsmodels.api')
    amos.models.GLM().apply(classified)
    results = api.GLM(
        classified.y_train.astype(float),
        api.add_constant(classified.x_train.astype(float)),
        family = api.families.Binomial()).fit()
    np.testing.assert_allclose(
        classified.tables['glm_coefficients']['coefficient'], results.params)
    np.testing.assert_allclose(
        classified.probabilities[1],
        results.predict(api.add_constant(classified.x_test.astype(float))))


def test_shap(fitted: amos.Dataset) -> None:
    requires('shap')
    shap = importlib.import_module('shap')
    amos.evaluators.ShapImportance().apply(fitted, rows = 40, background = 40)
    background = fitted.x_train.sample(40, random_state = SEED)
    explanation = shap.Explainer(fitted.model, background)(
        fitted.x_test.head(40))
    expected = pd.Series(
        np.abs(explanation.values).mean(axis = 0),
        index = fitted.x_test.columns)
    table = fitted.tables['shap_importance']
    np.testing.assert_allclose(
        table['importance'], expected.loc[table.index])


def test_matplotlib(classified: amos.Dataset, tmp_path: pathlib.Path) -> None:
    requires('matplotlib')
    figures = importlib.import_module('matplotlib.figure')
    amos.plots.Histograms().apply(classified, columns = ['x0'], bins = 5)
    figure = classified.figures['histograms']
    assert isinstance(figure, figures.Figure)
    heights = [bar.get_height() for bar in figure.axes[0].patches]
    counts, _ = np.histogram(classified.data['x0'], bins = 5)
    np.testing.assert_array_equal(heights, counts)
    path = tmp_path / 'histograms.png'
    figure.savefig(path)
    assert path.read_bytes().startswith(b'\x89PNG')


def test_docx(fitted: amos.Dataset, tmp_path: pathlib.Path) -> None:
    requires('docx')
    docx = importlib.import_module('docx')
    scorecard = amos.evaluators.Scorecard.create(fitted)
    path = scorecard.to_word(tmp_path / 'scorecard.docx')
    table = docx.Document(str(path)).tables[0]
    cells = [[cell.text for cell in row.cells] for row in table.rows]
    expected = scorecard.table.columns.tolist()
    assert cells[0] == expected
    assert float(cells[1][expected.index('roc_auc')]) == pytest.approx(
        scorecard.table.loc[0, 'roc_auc'], abs = 0.0005)


def test_seaborn(fitted: amos.Dataset) -> None:
    requires('matplotlib')
    requires('seaborn')
    amos.plots.CorrelationHeatmap().apply(fitted)
    axes = fitted.figures['correlation_heatmap'].axes[0]
    expected = amos.describers.Correlations().describe(fitted)
    drawn = np.asarray(axes.collections[0].get_array())
    np.testing.assert_allclose(
        drawn.reshape(expected.shape), expected.to_numpy())
    amos.plots.ConfusionHeatmap().apply(fitted)
    axes = fitted.figures['confusion_heatmap'].axes[0]
    matrix = amos.evaluators.Confusion().evaluate(fitted)
    numbers = [int(text.get_text()) for text in axes.texts]
    assert numbers == matrix.to_numpy().flatten().tolist()
