"""Tests the pipelines module."""

from __future__ import annotations

import copy
import pathlib

import chrisjen
import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_numeric, package_of, requires, techniques

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


def _required(pipeline: amos.Pipeline) -> dict:
    """Returns parameters sections that give a pipeline what it requires."""
    return {
        f'{technique}_parameters': {name: name for name in names}
        for technique, names in pipeline.required.items()}


def test_every_pipeline_names_techniques_in_the_library() -> None:
    names = set(amos.library.all)
    for pipeline in pipelines.PIPELINES.values():
        expanded = _expanded({
            'general': {'pipeline': pipeline.name, 'groups': 'judge'},
            **_required(pipeline)})
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
        for chooser in ('auto_evaluator', 'auto_plot'):
            for technique in expanded.get(f'{chooser}_parameters', {}).get(
                    'choices', []):
                assert technique in names, (pipeline.name, technique)


def test_every_model_has_a_pipeline_for_each_task() -> None:
    covered = {
        (model, pipeline.task)
        for pipeline in pipelines.PIPELINES.values()
        if len(pipeline.models) == 1 and 'design' not in pipeline.analyst
        for model in pipeline.models}
    for name, kind in techniques('model'):
        for task in kind.tools:
            assert (name, task) in covered, (name, task)


@pytest.mark.parametrize(('name', 'scale', 'impute', 'sample'), [
    ('random_forest_classification', None, True, False),
    ('xgboost_classification', None, False, True),
    ('lightgbm_regression', None, False, False),
    ('logit_classification', 'standard', True, False),
    ('svm_regression', 'standard', True, False),
    ('knn_classification', 'standard', True, True),
    ('neural_network_classification', 'standard', True, True),
    ('naive_bayes_classification', 'gauss', True, True),
    ('explainable_boosting_classification', None, False, False),
    ('ols_regression', None, True, False),
    ('logit_sm_classification', None, True, False)])
def test_model_pipelines_prepare_the_data_as_their_models_need(
    name: str,
    scale: str | None,
    impute: bool,
    sample: bool) -> None:
    analyst = pipelines.PIPELINES[name].analyst
    steps = analyst['steps']
    assert analyst.get('scale_techniques') == (
        None if scale is None else [scale])
    assert ('impute' in steps) == impute
    assert ('sample' in steps) == sample
    assert steps[-1] == 'model'


def test_model_pipelines_weigh_or_resample_rare_classes() -> None:
    forest = pipelines.PIPELINES['random_forest_classification'].parameters
    assert forest['random_forest']['class_weight'] == 'balanced_subsample'
    assert 'class_weight' not in pipelines.PIPELINES[
        'random_forest_regression'].parameters['random_forest']
    catboost = pipelines.PIPELINES['catboost_classification'].parameters
    assert catboost['catboost'] == {'auto_class_weights': 'Balanced'}
    # Models for inference keep the real share of each class.
    for name in ('logit_sm_classification', 'probit_classification'):
        assert 'sample' not in pipelines.PIPELINES[name].analyst['steps']
        assert pipelines.PIPELINES[name].parameters['auto_encode'] == {
            'drop': 'first'}


def test_model_pipelines_score_their_searches_for_the_task() -> None:
    logit = pipelines.PIPELINES['logit_classification'].parameters['logit']
    assert logit['search'] == 'grid'
    assert logit['scoring'] == 'balanced_accuracy'
    lasso = pipelines.PIPELINES['lasso_regression'].parameters['lasso']
    assert lasso['scoring'] == 'neg_root_mean_squared_error'


def test_model_pipelines_choose_the_tools_that_explain_their_models() -> None:
    forest = pipelines.PIPELINES['random_forest_regression']
    assert 'shap_importance' in forest.parameters['auto_evaluator']['choices']
    assert 'shap_beeswarm' in forest.parameters['auto_plot']['choices']
    ols = pipelines.PIPELINES['ols_regression']
    assert 'regression_table' in ols.parameters['auto_evaluator']['choices']
    assert {'qq_plot', 'influence_plot', 'coefficient_plot'} <= set(
        ols.parameters['auto_plot']['choices'])
    assert pipelines.PIPELINES['neural_network_regression'].artist == {
        'techniques': ['auto_plot', 'learning_curve']}
    assert 'shape_functions' in pipelines.PIPELINES[
        'explainable_boosting_regression'].parameters['auto_plot']['choices']


def test_pipeline_names_follow_the_model_and_task() -> None:
    assert pipelines.PIPELINES['quantile_inference'].models == [
        'quantile_regression']
    assert pipelines.PIPELINES['ordinal_inference'].task == 'classify'
    # A project named for a pipeline must not share a technique's name.
    assert not set(pipelines.PIPELINES) & set(amos.library.all)
    assert pipelines.PIPELINES['classification_inference'].models == [
        'ordinal_regression', 'logit_sm', 'mnlogit']


def test_grouped_pipelines_need_groups() -> None:
    with pytest.raises(ValueError, match = 'correlated: set "groups"'):
        _expanded({'general': {'pipeline': 'mixedlm_regression'}})
    expanded = _expanded({
        'general': {'pipeline': 'gee_regression', 'groups': 'judge'}})
    assert expanded['analyst']['split_techniques'] == ['group_split']
    assert expanded['critic']['techniques'][0] == 'group_k_fold'
    assert expanded['gee_parameters'] == {'covariance': 'exchangeable'}


def test_required_parameters_are_checked_and_shared() -> None:
    with pytest.raises(ValueError, match = "'event' in the 'cox_parameters'"):
        _expanded({'general': {'pipeline': 'cox_regression'}})
    expanded = _expanded({
        'general': {'pipeline': 'cox_regression'},
        'cox_parameters': {'event': 'arrested'},
        'survival_curves_parameters': {'at_risk': True}})
    assert expanded['survival_curves_parameters'] == {
        'event': 'arrested', 'at_risk': True}
    assert expanded['artist']['techniques'] == [
        'auto_plot', 'survival_curves']
    assert _expanded(expanded) == expanded


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


# Labels of `_study`, and the models whose pipelines model each one (the
# others classify "reversed" or regress "sentence").
_LABELS: dict[str, frozenset[str]] = {
    'reversed': frozenset(),
    'sentence': frozenset(),
    'level': frozenset({'mnlogit', 'ordinal_regression'}),
    'appeals': frozenset({
        'generalized_poisson', 'negative_binomial', 'poisson',
        'zero_inflated_poisson'}),
    'days': frozenset({'cox'})}
# The columns that the parameters required by a pipeline name.
_COLUMNS: dict[str, str] = {'event': 'arrested', 'weights': 'weight'}
# Parameters that make the slowest models quick to test: fewer bags of an
# explainable boosting machine, and fewer threads for xgboost (whose threads
# wait on each other for small data on a busy computer).
_LIGHT: dict[str, dict] = {
    'explainable_boosting': {'outer_bags': 2},
    'xgboost': {'n_jobs': 2}}


def _study(rows: int = 160) -> pd.DataFrame:
    """Returns cases with a label for every kind of model.

    "reversed" has two classes, "level" has three ordered classes,
    "sentence" is a number, "appeals" is overdispersed counts, and "days" is
    the time until "arrested" (or until the case was censored). "weight" is
    a weight for each row and "judge" is a group.
    """
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({
        'severity': rng.normal(size = rows),
        'prior': rng.poisson(1, rows).astype(float),
        'court': rng.choice(['state', 'federal', 'tribal'], rows),
        'judge': [f'judge {j}' for j in rng.integers(0, 10, rows)]})
    data.loc[data.index[:15], 'prior'] = np.nan
    score = data['severity'] + (data['court'] == 'federal') - 1.0
    data['reversed'] = (rng.random(rows) < 1 / (1 + np.exp(-score))).astype(
        int)
    data['level'] = pd.cut(
        score + rng.normal(size = rows), [-np.inf, -1.5, -0.5, np.inf],
        labels = ['low', 'mid', 'high'], ordered = True)
    data['sentence'] = 12 + 3 * data['severity'] + rng.normal(size = rows)
    data['appeals'] = rng.negative_binomial(
        2, 2 / (2 + np.exp(0.5 + 0.4 * data['severity'])))
    data['weight'] = rng.uniform(0.5, 2.0, rows)
    data['arrested'] = (rng.random(rows) < 0.7).astype(int)
    data['days'] = 1 + rng.exponential(100 * np.exp(-0.3 * data['severity']))
    return data


def _quick(pipeline: amos.Pipeline) -> dict:
    """Returns parameters that make a pipeline's model quick to fit.

    A search tries one setting in two folds.
    """
    model = pipeline.models[0]
    values = pipeline.parameters.get(model, {})
    quick = dict(_LIGHT.get(model, {}))
    if 'search' in values:
        quick.update({
            'cv': 2, 'n_iter': 1,
            **{k: v[:1] for k, v in values.items() if isinstance(v, list)}})
    return {f'{model}_parameters': quick} if quick else {}


_MODEL_PIPELINE_NAMES: list[str] = [
    pipeline.name for pipeline in pipelines._MODEL_PIPELINES]


@pytest.mark.parametrize('name', _MODEL_PIPELINE_NAMES)
def test_every_model_pipeline_runs(name: str) -> None:
    requires('matplotlib')
    pipeline = pipelines.PIPELINES[name]
    model = pipeline.models[0]
    requires(package_of(dict(techniques('model'))[model]))
    label = next(
        (label for label, models in _LABELS.items() if model in models),
        'reversed' if pipeline.task == 'classify' else 'sentence')
    data = _study().drop(columns = [c for c in _LABELS if c != label])
    general = {'label': label, 'seed': SEED, 'pipeline': name}
    if pipeline.grouped:
        general['groups'] = 'judge'
    else:
        data = data.drop(columns = ['judge'])
    settings = {
        'general': general,
        **{f'{technique}_parameters': {key: _COLUMNS[key] for key in names}
           for technique, names in pipeline.required.items()},
        **_quick(pipeline)}
    result = amos.Project.create(settings, item = data).result
    fitted = [
        entry['technique'] for entry in result.history if 'search' in entry]
    assert fitted[-1] == model
    assert 'auto_evaluator' in result.tables
    assert result.figures
