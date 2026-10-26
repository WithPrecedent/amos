"""Tests the RegressionTable of the critic stage and its exports."""

from __future__ import annotations

import copy
import os
import pathlib
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_numeric, make_regression, requires

import amos
from amos.evaluators import RegressionTable


def _regressed(model: str = 'ols', **parameters: Any) -> amos.Dataset:
    """Returns a split regression dataset with a fitted `model`."""
    dataset = amos.Dataset(make_regression(), label = 'target', seed = SEED)
    amos.splitters.TrainTest().apply(dataset)
    return amos.library.borrow(model)().apply(dataset, **parameters)


def _settings() -> dict[str, Any]:
    """Returns settings for a study with four branches and a table."""
    return {
        'general': {'label': 'target', 'seed': SEED},
        'study_project': {'study_workers': 'analyst, critic'},
        'analyst': {
            'design': 'experiment',
            'criterion': 'roc_auc',
            'steps': 'split, scale, model',
            'split_techniques': 'stratified',
            'scale_techniques': 'standard, none',
            'model_techniques': 'logit_sm, random_forest'},
        'critic': {'techniques': 'regression_table'},
        'regression_table_parameters': {'simulations': 5},
        'random_forest_parameters': {'n_estimators': 10}}


@pytest.fixture
def project(tmp_path: pathlib.Path) -> amos.Project:
    """An applied project whose experiment has four branches."""
    os.chdir(tmp_path)
    return amos.Project.create(_settings(), item = make_numeric(), id = 'run')


def test_a_model_with_coefficients_shows_them() -> None:
    dataset = _regressed('ols')
    table = RegressionTable.create(dataset)
    expected = dataset.model.coefficients()
    estimates = table.estimates.loc['ols']
    assert list(estimates.index) == [str(t) for t in expected.index]
    np.testing.assert_allclose(
        estimates['estimate'], expected['coefficient'])
    np.testing.assert_allclose(
        estimates['standard_error'], expected['standard_error'])
    statistics = table.statistics['ols']
    assert statistics['estimates'] == 'coefficients'
    assert statistics['observations'] == len(dataset.train)
    assert list(table.statistics.index) == [
        'estimates', 'observations', 'cv_r2', 'cv_rmse', 'cv_mae']
    assert table.heading == 'Estimates of the final model'


def test_the_marginal_effects_of_a_linear_regression_are_its_coefficients(
    ) -> None:
    dataset = _regressed('linear')
    table = RegressionTable.create(dataset, simulations = 20)
    estimates = table.estimates.loc['linear']
    np.testing.assert_allclose(
        estimates['estimate'], dataset.model.coef_, rtol = 1e-6)
    assert (estimates['standard_error'] > 0).all()
    assert (estimates['ci_lower'] < estimates['estimate']).all()
    assert (estimates['ci_upper'] > estimates['estimate']).all()
    assert table.statistics.loc['estimates', 'linear'] == 'marginal effects'


def test_marginal_shows_effects_for_a_model_with_coefficients() -> None:
    dataset = _regressed('ols')
    table = RegressionTable.create(dataset, marginal = True, simulations = 5)
    estimates = table.estimates.loc['ols']
    # The constant does not change with any feature, so it has no effect.
    assert 'const' not in estimates.index
    coefficients = dataset.model.coefficients()['coefficient']
    np.testing.assert_allclose(
        estimates['estimate'], coefficients.drop('const'), rtol = 1e-6)


def _categorical() -> amos.Dataset:
    """Returns a dataset whose "a" and "b" are dummies of one category."""
    rng = np.random.default_rng(SEED)
    rows = 3000
    group = rng.choice(['reference', 'a', 'b'], rows, p = [0.4, 0.3, 0.3])
    data = pd.DataFrame({
        'a': (group == 'a').astype(int),
        'b': (group == 'b').astype(int),
        'z': rng.normal(size = rows)})
    odds = -2 + 3 * data['a'] + 3 * data['b'] + 0.5 * data['z']
    data['y'] = (rng.random(rows) < 1 / (1 + np.exp(-odds))).astype(int)
    dataset = amos.Dataset(data, label = 'y', seed = SEED)
    return amos.library.borrow('logit')().apply(dataset, C = 1e6)


def _changes(dataset: amos.Dataset) -> dict[str, float]:
    """Returns the change from the reference to "a" and to "b" alone."""
    x = dataset.data[['a', 'b', 'z']]

    def mean(a: int, b: int) -> float:
        moved = x.assign(a = a, b = b)
        return float(dataset.model.predict_proba(moved)[:, 1].mean())

    return {'a': mean(1, 0) - mean(0, 0), 'b': mean(0, 1) - mean(0, 0)}


def test_sets_move_dummies_from_their_reference() -> None:
    dataset = _categorical()
    expected = _changes(dataset)
    grouped = RegressionTable.create(
        dataset, marginal = True, simulations = 0, sets = [['a', 'b']])
    estimates = grouped.estimates.loc['logit', 'estimate']
    for name, change in expected.items():
        assert estimates[name] == pytest.approx(change)
    alone = RegressionTable.create(dataset, marginal = True, simulations = 0)
    single = alone.estimates.loc['logit', 'estimate']
    assert abs(single['a'] - expected['a']) > 0.01
    assert single['z'] == pytest.approx(estimates['z'])
    written = RegressionTable.create(
        dataset, marginal = True, simulations = 0, sets = 'a, b; q')
    np.testing.assert_allclose(
        written.estimates.loc['logit', 'estimate'], estimates)


def test_sets_need_features_of_two_values() -> None:
    dataset = _categorical()
    with pytest.raises(ValueError, match = 'more than two values'):
        RegressionTable.create(
            dataset, marginal = True, simulations = 0, sets = [['a', 'z']])
    with pytest.raises(ValueError, match = 'more than one set'):
        RegressionTable.create(
            dataset, marginal = True, simulations = 0,
            sets = [['a', 'b'], ['b']])


def test_models_without_coefficients_show_marginal_effects() -> None:
    dataset = _regressed('random_forest', n_estimators = 10)
    table = RegressionTable.create(dataset, simulations = 5)
    estimates = table.estimates.loc['random_forest']
    assert list(estimates.index) == dataset.features
    assert estimates.notna().all().all()


def test_the_effects_of_a_binary_classifier_are_on_its_probability(
    fitted: amos.Dataset) -> None:
    table = RegressionTable.create(fitted, simulations = 5)
    estimates = table.estimates.loc['logit']
    assert list(estimates.index) == fitted.features
    assert (estimates['estimate'].abs() < 1).all()
    assert list(table.statistics.index[2:]) == [
        'cv_accuracy', 'cv_roc_auc', 'cv_log_loss']


def test_each_class_has_effects(multiclass: amos.Dataset) -> None:
    amos.models.Logit().apply(multiclass)
    table = RegressionTable.create(multiclass, simulations = 5)
    estimates = table.estimates.loc['logit', 'estimate']
    assert list(estimates.index[:3]) == ['x0 (0)', 'x0 (1)', 'x0 (2)']
    # The probabilities of the classes add up to 1, so their changes cancel.
    totals = estimates.groupby(lambda term: term.split(' ')[0]).sum()
    np.testing.assert_allclose(totals, 0, atol = 1e-9)


def test_features_with_two_values_change_from_one_to_the_other() -> None:
    data = make_regression()
    rng = np.random.default_rng(SEED)
    data['treated'] = rng.random(len(data)) > 0.5
    data['coded'] = (rng.random(len(data)) > 0.5).astype(int) * 3
    data['target'] += 20 * data['treated'] + 4 * data['coded']
    dataset = amos.Dataset(data, label = 'target', seed = SEED)
    amos.models.Linear().apply(dataset)
    estimates = RegressionTable.create(
        dataset, simulations = 5).estimates.loc['linear', 'estimate']
    coefficients = pd.Series(dataset.model.coef_, index = dataset.features)
    assert estimates['treated'] == pytest.approx(coefficients['treated'])
    # A feature of 0 and 3 changes by 3, so its effect is per unit.
    assert estimates['coded'] == pytest.approx(coefficients['coded'])


def test_a_classifier_without_probabilities_has_no_effects(
    classified: amos.Dataset) -> None:
    model = amos.Model(
        name = 'ridge_classifier',
        contents = 'sklearn.linear_model.RidgeClassifier')
    model.apply(classified)
    table = RegressionTable.create(classified, simulations = 5)
    assert table.estimates.empty
    kind = table.statistics.loc['estimates', 'ridge_classifier']
    assert kind.startswith('none')


def test_the_bootstrap_is_reproducible() -> None:
    first = RegressionTable.create(
        _regressed('random_forest', n_estimators = 5), simulations = 5)
    second = RegressionTable.create(
        _regressed('random_forest', n_estimators = 5), simulations = 5)
    pd.testing.assert_frame_equal(first.estimates, second.estimates)


def test_the_metrics_of_a_validator_are_used() -> None:
    dataset = _regressed('ols')
    amos.validators.KFold().apply(dataset, n_splits = 3)
    table = RegressionTable.create(dataset)
    assert table.statistics.loc['cv_r2', 'ols'] == dataset.metrics['cv_r2']


def test_cross_validating_does_not_change_the_dataset() -> None:
    dataset = _regressed('ols')
    metrics = dict(dataset.metrics)
    tables = set(dataset.tables)
    history = len(dataset.history)
    RegressionTable.create(dataset, metrics = ['r2'])
    assert dataset.metrics == metrics
    assert set(dataset.tables) == tables
    assert len(dataset.history) == history


def test_each_branch_of_an_experiment_is_a_column(
    project: amos.Project) -> None:
    result = project.result
    assert list(result.tables['regression_table_statistics'].columns) == [
        'none / logit_sm', 'standard / logit_sm',
        'none / random_forest', 'standard / random_forest']
    estimates = result.tables['regression_table']
    assert estimates.index.names == ['model', 'term']
    assert 'const' in estimates.loc['none / logit_sm'].index
    assert 'const' not in estimates.loc['none / random_forest'].index
    assert result.history[-1]['technique'] == 'regression_table'


def test_branches_keep_a_weak_reference_to_their_datasets(
    project: amos.Project) -> None:
    analyst = {node.name: node for node in project.workflow}['analyst']
    branch = project.result.branches[0]
    assert branch.dataset is analyst.results[branch.label]
    assert branch.dataset.model is not None
    assert pickle.loads(pickle.dumps(branch)).dataset is None
    assert copy.deepcopy(branch).dataset is None


def test_branches_that_are_gone_leave_the_final_model(
    project: amos.Project) -> None:
    result = copy.deepcopy(project.result)
    with pytest.warns(UserWarning, match = 'no longer in memory'):
        table = RegressionTable.create(result, simulations = 5)
    assert list(table.statistics.columns) == ['random_forest']


def test_several_datasets_are_columns() -> None:
    ols, linear = _regressed('ols'), _regressed('linear')
    table = RegressionTable.create(
        {'(1)': ols, '(2)': linear}, simulations = 5)
    assert list(table.statistics.columns) == ['(1)', '(2)']
    assert table.heading == 'Estimates of 2 models'
    table = RegressionTable.create([linear, linear], simulations = 5)
    assert list(table.statistics.columns) == ['linear', 'linear (2)']


def test_create_needs_datasets() -> None:
    with pytest.raises(TypeError, match = 'needs a Dataset'):
        RegressionTable.create(42)
    with pytest.raises(TypeError, match = 'needs a Dataset'):
        RegressionTable.create([42])
    with pytest.raises(ValueError, match = 'regression table is empty'):
        RegressionTable().to_markdown()


def test_the_text_has_two_rows_for_each_term() -> None:
    table = RegressionTable.create(_regressed('ols'))
    text = table.table
    assert list(text.columns) == ['', 'ols']
    assert list(text[''][:4]) == ['const', '', 'x0', '']
    assert text.loc[0, 'ols'].endswith('***')
    assert text.loc[1, 'ols'].startswith('(')
    assert text.loc[1, 'ols'].endswith(')')
    assert list(text[''][12:]) == list(table.statistics.index)
    assert text.loc[13, 'ols'] == '150'


def test_stars_and_digits_can_be_set() -> None:
    table = RegressionTable.create(
        _regressed('ols'), digits = 1, stars = (0.01, 0.001, 1e-9))
    assert table.table.loc[2, 'ols'] == (
        f'{table.estimates.loc[("ols", "x0"), "estimate"]:.1f}***')
    assert 'p < 1e-09' in table._notes()


def test_to_markdown_has_notes(tmp_path: pathlib.Path) -> None:
    table = RegressionTable.create(_regressed('linear'), simulations = 5)
    text = table.to_markdown(tmp_path / 'table.md')
    lines = text.splitlines()
    assert lines[0] == '|  | linear |'
    assert lines[1] == '| --- | ---: |'
    assert lines[-1].startswith('Standard errors in parentheses. * p < 0.1')
    assert 'Marginal effects are the average change' in lines[-1]
    assert (tmp_path / 'table.md').read_text(encoding = 'utf-8') == text


def test_to_latex_draws_a_rule_above_the_statistics() -> None:
    table = RegressionTable.create(_regressed('ols'))
    lines = table.to_latex().splitlines()
    rules = [i for i, line in enumerate(lines) if line == r'\midrule']
    assert len(rules) == 2
    assert lines[rules[1] + 1].startswith('estimates & coefficients')
    assert r'p \textless{} 0.1' in table.to_latex()


def test_to_csv_saves_the_text(tmp_path: pathlib.Path) -> None:
    table = RegressionTable.create(_regressed('ols'))
    table.to_csv(tmp_path / 'table.csv')
    read = pd.read_csv(tmp_path / 'table.csv', dtype = str).fillna('')
    assert list(read.iloc[:, 1]) == list(table.table['ols'])


def test_to_text_looks_like_stata() -> None:
    ols, forest = _regressed('ols'), _regressed('random_forest', n_estimators = 5)
    table = RegressionTable.create(
        {'ols': ols, 'forest': forest}, simulations = 5)
    lines = table.to_text().splitlines()
    assert lines[0].startswith('ols')
    assert lines[0].endswith('Number of obs   =        150')
    assert lines[1].startswith('Coefficients')
    assert 'CV R-squared' in lines[1]
    width = len(lines[5])
    assert lines[5] == '-' * width
    assert lines[6] == (
        '      target |      Coef.  Std. Err.        t   P>|t|'
        '     [95% Conf. Interval]')
    assert lines[7] == '-' * 13 + '+' + '-' * (width - 14)
    assert lines[8].startswith('       const |   499.9293   .8297661')
    assert lines[14] == '-' * width
    start = next(i for i, line in enumerate(lines) if line.startswith('forest'))
    forest_block = lines[start:]
    assert forest_block[1].startswith('Average marginal effects')
    assert 'Replications    =          5' in forest_block[1]
    assert forest_block[2].startswith('Expression: Predicted target')
    assert ' dy/dx ' in table.to_text()
    assert table.to_text().splitlines()[-1].endswith('(Replications).')


def test_coefficients_of_a_classifier_have_a_z_statistic(
    classified: amos.Dataset) -> None:
    requires('statsmodels')
    amos.library.borrow('logit_sm')().apply(classified)
    text = RegressionTable.create(classified).to_text()
    assert 'Std. Err.        z   P>|z|' in text
    assert 'dy/dx' not in text
    assert 'Expression' not in text


def test_stata_numbers() -> None:
    write = amos.evaluators._stata_number
    assert write(0.9286648) == '.9286648'
    assert write(-0.0941676) == '-.0941676'
    assert write(-2.3374731) == '-2.337473'
    assert write(0.047888391) == '.0478884'
    assert write(25.7459) == '25.7459'
    assert write(float('nan')) == '.'


def test_to_word_looks_like_stata(tmp_path: pathlib.Path) -> None:
    requires('docx')
    docx = pytest.importorskip('docx')
    table = RegressionTable.create(
        {'ols': _regressed('ols'), 'linear': _regressed('linear')},
        title = 'Table 3', simulations = 5)
    document = docx.Document(str(table.to_word(tmp_path / 'table.docx')))
    assert document.paragraphs[0].text == 'Table 3'
    assert document.paragraphs[-1].text.startswith('dy/dx is the average')
    # A table of statistics and a table of estimates for each model.
    assert len(document.tables) == 4
    header, grid = document.tables[0], document.tables[1]
    assert header.cell(0, 0).text == 'ols'
    assert header.cell(0, 1).text.startswith('Number of obs')
    assert [c.text for c in grid.rows[0].cells][:5] == [
        'target', 'Coef.', 'Std. Err.', 't', 'P>|t|']
    assert grid.cell(0, 5).text == '[95% Conf. Interval]'
    assert len(grid.rows) == 1 + 6
    borders = grid.cell(0, 0)._tc.xml
    assert 'w:top' in borders
    assert 'w:right' in borders


def test_to_image_looks_like_stata(tmp_path: pathlib.Path) -> None:
    requires('matplotlib')
    table = RegressionTable.create(_regressed('ols'), title = 'Table 3')
    path = table.to_image(tmp_path / 'table.png', dpi = 72)
    assert path.read_bytes().startswith(b'\x89PNG')
    axes = table.to_figure().axes[0]
    texts = [text.get_text() for text in axes.texts]
    assert texts[0] == 'Table 3'
    # The bar after the terms is a line, not a character.
    row = next(text for text in texts if text.lstrip().startswith('x0'))
    assert '|' not in row
    # Three rules and the bar, drawn in pieces.
    assert len(axes.lines) > 3


def test_the_columns_style_puts_the_models_side_by_side(
    tmp_path: pathlib.Path) -> None:
    requires('docx')
    requires('matplotlib')
    docx = pytest.importorskip('docx')
    table = RegressionTable.create(
        _regressed('ols'), title = 'Table 3', style = 'columns')
    document = docx.Document(str(table.to_word(tmp_path / 'table.docx')))
    assert document.paragraphs[-1].text.startswith('Standard errors')
    assert len(document.tables) == 1
    assert len(document.tables[0].rows) == len(table.table) + 1
    figure = table.to_figure()
    assert figure.texts[0].get_text() == 'Table 3'
    assert figure.texts[1].get_text().startswith('Standard errors')
    table.style = 'stata_like'
    with pytest.raises(ValueError, match = 'style must be'):
        table.to_figure()


def test_project_export_includes_the_regression_table(
    project: amos.Project) -> None:
    folder = project.export()
    names = {path.name for path in folder.iterdir()}
    assert {'regression_table.csv', 'regression_table.md'} <= names
    assert (folder / 'tables' / 'regression_table.csv').is_file()
    assert (folder / 'tables' / 'regression_table_statistics.csv').is_file()
