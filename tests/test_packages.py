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
    'DoubleML': 'doubleml',
    'InterpretML': 'interpret',
    'MAPIE': 'mapie',
    'Optuna': 'optuna',
    'Polars': 'polars',
    'SciencePlots': 'scienceplots',
    'TabPFN': 'tabpfn',
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
    assert len(packages) == 22
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


def test_scienceplots(fitted: amos.Dataset) -> None:
    requires('matplotlib')
    requires('scienceplots')
    importlib.import_module('scienceplots')
    library = importlib.import_module('matplotlib.style').library
    colors = importlib.import_module('matplotlib.colors')
    amos.plots.RocCurve().apply(fitted)
    figure = fitted.figures['roc_curve']
    axes = figure.axes[0]
    # The figure has the size and fonts of the "nature" style, the ticks of
    # the "science" style, and the first color of the "bright" cycle.
    nature, science = library['nature'], library['science']
    assert figure.get_size_inches()[0] == pytest.approx(
        nature['figure.figsize'][0])
    assert axes.xaxis.label.get_fontsize() == nature['font.size']
    tick = axes.xaxis.get_major_ticks()[0]
    assert tick._tickdir == science['xtick.direction']
    bright = library['bright']['axes.prop_cycle'].by_key()['color']
    assert colors.to_hex(axes.get_lines()[0].get_color()) == (
        colors.to_hex(bright[0]))


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


def test_catboost(classified: amos.Dataset) -> None:
    requires('catboost')
    catboost = importlib.import_module('catboost')
    amos.models.Catboost().apply(classified, iterations = 20)
    model = catboost.CatBoostClassifier(
        iterations = 20,
        verbose = 0,
        allow_writing_files = False,
        random_state = SEED)
    model.fit(classified.x_train, classified.y_train)
    np.testing.assert_allclose(
        classified.probabilities.to_numpy(),
        model.predict_proba(classified.x_test))


def test_skrub() -> None:
    requires('skrub')
    skrub = importlib.import_module('skrub')
    dataset = _mixed()
    region = dataset.data['region'].copy()
    train = dataset.x_train['region']
    amos.transformers.MinHash().apply(
        dataset, columns = ['region'], n_components = 3)
    expected = skrub.MinHashEncoder(n_components = 3).fit(train).transform(
        region)
    np.testing.assert_allclose(
        dataset.data[list(expected.columns)].to_numpy(), expected.to_numpy())


def test_pyfixest(regressed: amos.Dataset) -> None:
    requires('pyfixest')
    pyfixest = importlib.import_module('pyfixest')
    regressed.data['judge'] = [f'j{i % 5}' for i in range(len(regressed.data))]
    regressed.groups = ['judge']
    amos.models.Fixest().apply(regressed, fixed_effects = 'judge')
    train = regressed.data.loc[regressed.train].rename(
        columns = lambda c: c.replace(' ', '_'))
    expected = pyfixest.feols(
        'target ~ x0 + x1 + x2 + x3 + x4 | judge',
        data = train,
        vcov = 'hetero').tidy()
    table = regressed.tables['fixest_coefficients']
    np.testing.assert_allclose(table['coefficient'], expected['Estimate'])
    np.testing.assert_allclose(table['standard_error'], expected['Std. Error'])


def test_optuna(classified: amos.Dataset) -> None:
    requires('optuna_integration')
    optuna = importlib.import_module('optuna')
    integration = importlib.import_module('optuna_integration')
    ensemble = importlib.import_module('sklearn.ensemble')
    x_train, y_train = classified.x_train, classified.y_train
    amos.models.RandomForest().apply(
        classified, search = 'optuna', n_estimators = [5, 15], n_iter = 3,
        cv = 3)
    search = integration.OptunaSearchCV(
        ensemble.RandomForestClassifier(random_state = SEED),
        {'n_estimators': optuna.distributions.IntDistribution(5, 15)},
        n_trials = 3,
        cv = 3,
        random_state = SEED)
    search.fit(x_train, y_train)
    table = classified.tables['random_forest_search']
    assert sorted(table['param_n_estimators']) == sorted(
        search.trials_dataframe()['params_n_estimators'])
    assert classified.model.n_estimators == search.best_params_['n_estimators']


def test_fairlearn() -> None:
    requires('fairlearn')
    fairness = importlib.import_module('fairlearn.metrics')
    dataset = _mixed()
    dataset.groups = ['member']
    amos.transformers.OneHot().apply(dataset)
    amos.models.Logit().apply(dataset)
    amos.metrics.EqualizedOdds().apply(dataset)
    rows = dataset.predictions.index
    expected = fairness.equalized_odds_difference(
        (dataset.y.loc[rows] == 'yes').astype(int),
        (dataset.predictions == 'yes').astype(int),
        sensitive_features = dataset.data.loc[rows, 'member'])
    assert dataset.metrics['equalized_odds'] == pytest.approx(expected)


def test_mapie(fitted_regression: amos.Dataset) -> None:
    requires('mapie')
    regression = importlib.import_module('mapie.regression')
    linear_model = importlib.import_module('sklearn.linear_model')
    amos.evaluators.Conformal().apply(fitted_regression, cv = 3)
    conformal = regression.CrossConformalRegressor(
        linear_model.LinearRegression(),
        confidence_level = 0.9,
        cv = 3,
        random_state = SEED)
    conformal.fit_conformalize(
        fitted_regression.x_train, fitted_regression.y_train)
    _, intervals = conformal.predict_interval(fitted_regression.x_test)
    table = fitted_regression.tables['conformal']
    np.testing.assert_allclose(table['lower'], intervals[:, 0, 0])
    np.testing.assert_allclose(table['upper'], intervals[:, 1, 0])


def test_interpret(classified: amos.Dataset) -> None:
    requires('interpret')
    glassbox = importlib.import_module('interpret.glassbox')
    amos.models.ExplainableBoosting().apply(classified, interactions = 0)
    model = glassbox.ExplainableBoostingClassifier(
        interactions = 0, random_state = SEED)
    model.fit(classified.x_train, classified.y_train)
    np.testing.assert_allclose(
        classified.probabilities.to_numpy(),
        model.predict_proba(classified.x_test))


def test_doubleml(regressed: amos.Dataset) -> None:
    requires('doubleml')
    doubleml = importlib.import_module('doubleml')
    linear_model = importlib.import_module('sklearn.linear_model')
    model_selection = importlib.import_module('sklearn.model_selection')
    data = regressed.data.copy()
    amos.effects.PartiallyLinear().apply(
        regressed, treatment = 'x0', outcome_model = 'linear', n_folds = 3)
    controls = ['x1', 'x2', 'x3', 'x4']
    estimator = doubleml.DoubleMLPLR(
        doubleml.DoubleMLData(
            data.astype(float), y_col = 'target', d_cols = 'x0',
            x_cols = controls),
        linear_model.LinearRegression(),
        linear_model.LinearRegression(),
        n_folds = 3)
    folds = model_selection.KFold(
        n_splits = 3, shuffle = True, random_state = SEED)
    estimator.set_sample_splitting(list(folds.split(np.arange(len(data)))))
    estimator.fit()
    table = regressed.tables['partially_linear']
    assert table.loc['x0', 'coefficient'] == pytest.approx(
        estimator.summary.loc['x0', 'coef'])
    assert table.loc['x0', 'standard_error'] == pytest.approx(
        estimator.summary.loc['x0', 'std err'])


def test_statsmodels_cox(regressed: amos.Dataset) -> None:
    requires('statsmodels')
    regression = importlib.import_module(
        'statsmodels.duration.hazard_regression')
    amos.models.Cox().apply(regressed)
    train = regressed.data.loc[regressed.train]
    expected = regression.PHReg(
        train['target'], train.drop(columns = 'target'),
        ties = 'efron').fit()
    table = regressed.tables['cox_coefficients']
    np.testing.assert_allclose(table['coefficient'], expected.params)
    np.testing.assert_allclose(table['standard_error'], expected.bse)


def test_tabpfn(classified: amos.Dataset) -> None:
    requires('tabpfn')
    tabpfn = importlib.import_module('tabpfn')
    amos.models.Tabpfn().apply(classified)
    model = tabpfn.TabPFNClassifier(random_state = SEED)
    model.fit(classified.x_train, classified.y_train)
    np.testing.assert_allclose(
        classified.probabilities.to_numpy(),
        model.predict_proba(classified.x_test),
        rtol = 1e-4)


def test_polars() -> None:
    requires('polars')
    polars = importlib.import_module('polars')
    data = _mixed().data
    frame = polars.from_pandas(data.reset_index(drop = True))
    dataset = amos.Dataset.create(frame, label = 'outcome')
    pd.testing.assert_frame_equal(
        dataset.data, frame.to_pandas(), check_dtype = False)


def test_great_tables(fitted: amos.Dataset) -> None:
    requires('great_tables')
    great_tables = importlib.import_module('great_tables')
    scorecard = amos.evaluators.Scorecard.create(fitted)
    html = scorecard.to_html()
    expected = great_tables.GT(scorecard._text()).as_raw_html()
    for value in scorecard._text().iloc[0]:
        assert value in html
        assert value in expected


def test_eli5(fitted: amos.Dataset) -> None:
    requires('eli5')
    eli5 = importlib.import_module('eli5')
    amos.evaluators.ExplainWeights().apply(fitted, top = 5)
    expected = eli5.explain_weights_df(
        fitted.model, feature_names = list(fitted.x_train.columns), top = 5)
    pd.testing.assert_frame_equal(fitted.tables['explain_weights'], expected)
