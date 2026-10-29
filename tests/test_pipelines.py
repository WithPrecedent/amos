"""Tests the pipelines module."""

from __future__ import annotations

import copy
import pathlib

import chrisjen
import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_numeric, requires

import amos
from amos import pipelines


def _expanded(settings: dict, name: str | None = None) -> dict:
    """Returns a copy of `settings` with its pipelines filled in."""
    expanded = copy.deepcopy(settings)
    pipelines.expand(expanded, name)
    return expanded


def _cases(rows: int = 400) -> pd.DataFrame:
    """Returns court cases with a rare outcome and a feature of many levels."""
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({
        'severity': rng.normal(size = rows),
        'prior': rng.poisson(1, rows).astype(float),
        'court': rng.choice(['state', 'federal', 'tribal'], rows),
        'judge': [f'judge {j}' for j in rng.integers(0, 30, rows)]})
    data.loc[data.index[:30], 'prior'] = np.nan
    score = data['severity'] + (data['court'] == 'federal') - 2.0
    data['reversed'] = (rng.random(rows) < 1 / (1 + np.exp(-score))).astype(
        int)
    data['sentence'] = 12 + 3 * data['severity'] + rng.normal(size = rows)
    data['appeals'] = rng.negative_binomial(
        2, 2 / (2 + np.exp(0.5 + 0.4 * data['severity'])))
    return data


def _chosen(item: amos.Dataset, name: str) -> list[str]:
    """Returns the techniques that the chooser `name` chose."""
    table = item.tables[name]
    return table.index[table['chosen'].to_numpy(dtype = bool)].tolist()


""" Expanding settings """


def test_a_pipeline_in_general_fills_in_the_whole_project() -> None:
    settings = {'general': {'label': 'reversed', 'pipeline': 'classification'}}
    expanded = _expanded(settings)
    assert expanded['classification_project'] == {
        'classification_workers': ['analyst', 'critic', 'artist']}
    analyst = expanded['analyst']
    assert analyst['design'] == 'experiment'
    assert analyst['criterion'] == 'balanced_accuracy'
    assert analyst['sample_techniques'] == ['auto']
    assert expanded['critic'] == {
        'techniques': ['stratified_k_fold', 'auto_evaluator']}
    assert expanded['artist'] == {'techniques': ['auto_plot']}
    assert expanded['random_forest_parameters'] == {'n_estimators': 300}
    assert expanded['general']['task'] == 'classify'


def test_expanding_does_not_change_the_settings_given() -> None:
    settings = {
        'general': {'label': 'reversed', 'pipeline': 'classification'},
        'analyst': {'model_techniques': ['logit']}}
    original = copy.deepcopy(settings)
    project = amos.Project.create(settings, automatic = False)
    assert settings == original
    assert project.idea['analyst']['design'] == 'experiment'


def test_expanding_twice_changes_nothing() -> None:
    once = _expanded({'general': {'pipeline': 'regression_inference'}})
    assert _expanded(once) == once


def test_settings_written_by_the_user_win() -> None:
    settings = {
        'general': {'label': 'reversed', 'pipeline': 'classification'},
        'analyst': {'model_techniques': 'logit, xgboost', 'criterion': 'f1'},
        'random_forest_parameters': {'max_depth': 3}}
    expanded = _expanded(settings)
    assert expanded['analyst']['model_techniques'] == 'logit, xgboost'
    assert expanded['analyst']['criterion'] == 'f1'
    assert expanded['analyst']['split_techniques'] == ['stratified']
    assert expanded['random_forest_parameters'] == {
        'n_estimators': 300, 'max_depth': 3}


def test_a_pipeline_in_the_project_fills_in_only_its_workers() -> None:
    settings = {
        'general': {'label': 'reversed'},
        'cases_project': {
            'cases_workers': 'explorer, analyst', 'pipeline': 'regression'},
        'explorer': {'techniques': 'summarize'}}
    expanded = _expanded(settings)
    assert expanded['cases_project']['cases_workers'] == 'explorer, analyst'
    assert expanded['explorer'] == {'techniques': 'summarize'}
    assert expanded['analyst']['criterion'] == 'rmse'
    assert 'critic' not in expanded
    assert 'artist' not in expanded


def test_a_worker_can_name_its_own_pipeline() -> None:
    settings = {
        'general': {'pipeline': 'regression'},
        'critic': {'pipeline': 'regression_inference'}}
    expanded = _expanded(settings, 'cases')
    assert 'cases_project' in expanded
    assert expanded['analyst']['design'] == 'experiment'
    assert expanded['auto_model_parameters']['choices'] == [
        'robust_regression', 'ols']
    settings = {
        'general': {'label': 'sentence'},
        'cases_project': {'cases_workers': 'analyst, critic'},
        'analyst': {'pipeline': 'regression_inference'}}
    expanded = _expanded(settings)
    assert expanded['analyst']['model_techniques'] == ['auto']
    assert 'critic' not in expanded
    assert expanded['general']['task'] == 'regress'


def test_the_project_name_is_used_for_a_new_project_section() -> None:
    settings = {'general': {'pipeline': 'classification'}}
    project = amos.Project.create(settings, name = 'cases', automatic = False)
    assert project.idea['cases_project'] == {
        'cases_workers': ['analyst', 'critic', 'artist']}
    project = amos.Project.create(settings, automatic = False)
    assert project.name == 'classification'


def test_panel_pipelines_use_the_groups_as_fixed_effects() -> None:
    settings = {
        'general': {
            'pipeline': 'panel_regression', 'groups': 'judge, year'}}
    expanded = _expanded(settings)
    assert expanded['fixest_parameters'] == {
        'fixed_effects': ['judge', 'year'], 'cluster': 'judge'}
    assert expanded['auto_encode_parameters'] == {'drop': 'first'}
    with pytest.raises(ValueError, match = 'set "groups"'):
        _expanded({'general': {'pipeline': 'panel_regression'}})


@pytest.mark.parametrize(('settings', 'message'), [
    ({'general': {'pipeline': 'clustering'}},
     "no pipeline named 'clustering'.*classification, regression"),
    ({'general': {'pipeline': 'regression'},
      'cases_project': {'cases_workers': 'explorer'},
      'explorer': {'pipeline': 'regression'}},
     "named in the 'explorer' section"),
    ({'general': {'pipeline': 'regression', 'task': 'classify'}},
     'sets "task" to \'classify\''),
    ({'general': {'pipeline': 'regression'}, 'critic': {
        'pipeline': 'classification'}},
     'different tasks')])
def test_mistakes_in_pipelines_are_explained(
    settings: dict,
    message: str) -> None:
    with pytest.raises(ValueError, match = message):
        _expanded(settings)


def test_every_pipeline_names_techniques_in_the_library() -> None:
    names = set(amos.library.all)
    for pipeline in pipelines.PIPELINES.values():
        expanded = _expanded(
            {'general': {'pipeline': pipeline.name, 'groups': 'judge'}})
        idea = chrisjen.Idea.create(expanded)
        for stage in pipelines._STAGES:
            for technique in idea.techniques.get(stage, []):
                assert technique in names, (pipeline.name, technique)
        for step in idea.steps.get('analyst', []):
            for technique in idea.techniques[step]:
                assert technique in names | {'auto'}, (pipeline.name, step)
        for model in expanded.get('auto_model_parameters', {}).get(
                'choices', []):
            assert model in names, (pipeline.name, model)


def test_a_new_pipeline_can_be_named(monkeypatch: pytest.MonkeyPatch) -> None:
    requires('matplotlib')
    mine = amos.Pipeline(
        name = 'quick',
        task = 'classify',
        description = 'A quick logit model.',
        analyst = {
            'steps': ['split', 'model'],
            'split_techniques': ['stratified'],
            'model_techniques': ['logit']},
        critic = {'techniques': ['confusion']})
    monkeypatch.setitem(pipelines.PIPELINES, 'quick', mine)
    settings = {'general': {'label': 'target', 'pipeline': 'quick'}}
    project = amos.Project.create(settings, item = make_numeric())
    assert 'confusion' in project.result.tables
    assert 'auto_plot' in project.result.tables


def test_a_pipeline_can_be_named_in_a_settings_file(
    tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'study.ini'
    path.write_text(
        '[general]\nlabel = sentence\nseed = 43\n'
        'pipeline = regression_inference\n\n'
        '[analyst]\nencode_techniques = one_hot\n',
        encoding = 'utf-8')
    project = amos.Project.create(path, automatic = False)
    assert project.name == 'regression_inference'
    assert project.idea['analyst']['encode_techniques'] == 'one_hot'
    assert project.idea['analyst']['split_techniques'] == ['train_test']


""" Applying pipelines """


def test_the_classification_pipeline_adapts_to_the_data() -> None:
    requires('imblearn')
    requires('category_encoders')
    data = _cases().drop(columns = ['sentence', 'appeals'])
    settings = {
        'general': {
            'label': 'reversed', 'seed': SEED, 'pipeline': 'classification'},
        'cases_project': {'cases_workers': 'analyst, critic'},
        'analyst': {'model_techniques': 'logit, baseline'}}
    result = amos.Project.create(settings, item = data).result
    assert _chosen(result, 'auto_imputer') == ['iterative_impute']
    encoders = next(
        record['encoders'] for record in result.history
        if record['technique'] == 'auto_encoder')
    assert encoders == {'court': 'one_hot', 'judge': 'target'}
    assert _chosen(result, 'auto_sampler') != ['none']
    assert len(result.tables['analyst_comparison']) == 2
    assert 'confusion' in result.tables
    assert 'cv_balanced_accuracy' in result.metrics


def test_the_inference_pipelines_choose_a_model_that_suits_the_label() -> None:
    requires('statsmodels')
    data = _cases()
    for label, pipeline, model in [
            ('reversed', 'classification_inference', 'logit_sm'),
            ('sentence', 'regression_inference', 'ols'),
            ('appeals', 'count_inference', 'negative_binomial')]:
        settings = {
            'general': {'label': label, 'seed': SEED, 'pipeline': pipeline},
            'cases_project': {'cases_workers': 'analyst, critic'}}
        others = [c for c in ('reversed', 'sentence', 'appeals') if c != label]
        result = amos.Project.create(
            settings, item = data.drop(columns = others)).result
        assert _chosen(result, 'auto_model') == [model], pipeline
        assert 'regression_table' in _chosen(result, 'auto_evaluator')
        # Each category is encoded against its first level.
        courts = [c for c in result.features if c.startswith('court_')]
        assert courts == ['court_state', 'court_tribal']
        assert 'auto_sampler' not in result.tables


def test_the_panel_pipeline_fits_fixed_effects() -> None:
    requires('pyfixest')
    data = _cases().drop(columns = ['reversed', 'appeals'])
    settings = {
        'general': {
            'label': 'sentence', 'seed': SEED, 'groups': 'judge',
            'pipeline': 'panel_regression'},
        'cases_project': {'cases_workers': 'analyst, critic'}}
    result = amos.Project.create(settings, item = data).result
    assert 'fixest_coefficients' in result.tables
    assert 'judge' not in result.features
    assert 'regression_table' in _chosen(result, 'auto_evaluator')
