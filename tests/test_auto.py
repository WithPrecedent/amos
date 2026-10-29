"""Tests the auto module."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_numeric, make_regression, requires

import amos
from amos import auto


def _split(data: pd.DataFrame, label: str = 'target', **kwargs: object) -> amos.Dataset:
    """Returns a dataset of `data`, split into training and test rows."""
    dataset = amos.Dataset(data, label = label, seed = SEED, **kwargs)
    if dataset.task == 'classify':
        return amos.splitters.Stratified().apply(dataset)
    return amos.splitters.TrainTest().apply(dataset)


def _counts(rows: int = 400, overdispersed: bool = True) -> pd.DataFrame:
    """Returns two features and a label of counts that depends on them."""
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size = rows)
    x2 = rng.normal(size = rows)
    mean = np.exp(0.5 + 0.6 * x1 - 0.4 * x2)
    if overdispersed:
        counts = rng.negative_binomial(1, 1 / (1 + mean))
    else:
        counts = rng.poisson(mean)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'count': counts})


def _chosen(item: amos.Dataset, name: str) -> list[str]:
    """Returns the techniques that the chooser `name` chose."""
    table = item.tables[name]
    return table.index[table['chosen'].to_numpy(dtype = bool)].tolist()


""" Profile """


def test_profile_describes_classes() -> None:
    dataset = _split(make_numeric(rows = 300, weights = [0.85]))
    found = auto.profile(dataset)
    assert found.task == 'classify'
    assert found.rows == len(dataset.train)
    assert (found.features, found.numerics, found.binaries) == (6, 6, 0)
    assert found.classes == 2
    assert found.balance is not None and found.balance < 0.3
    assert found.minority == dataset.y_train.value_counts().min()
    assert not found.missing and not found.ordinal
    assert found.normal is None


def test_profile_finds_ordered_classes() -> None:
    data = make_numeric(rows = 300, classes = 3)
    found = auto.profile(_split(data))
    assert found.ordinal
    data['target'] = data['target'].map({0: 'a', 1: 'b', 2: 'c'})
    assert not auto.profile(_split(data)).ordinal


def test_profile_finds_overdispersed_counts() -> None:
    requires('statsmodels')
    found = auto.profile(_split(_counts(), label = 'count', task = 'regress'))
    assert found.counts and found.overdispersed
    assert found.dispersion is not None and found.dispersion > 1.5
    found = auto.profile(_split(
        _counts(overdispersed = False), label = 'count', task = 'regress'))
    assert found.counts and not found.overdispersed


def test_profile_tests_the_normality_of_errors() -> None:
    data = make_regression(rows = 300)
    found = auto.profile(_split(data))
    assert found.normal and not found.counts
    assert found.normality is not None
    rng = np.random.default_rng(SEED)
    data['target'] = data['x0'] + rng.lognormal(0, 1, len(data))
    found = auto.profile(_split(data))
    assert found.normal is False
    assert found.skewness is not None and found.skewness > 1


def test_profile_finds_collinear_unscaled_and_missing_features() -> None:
    data = make_regression(rows = 200)
    data['copy'] = data['x0'] * 1000 + 0.001
    data.loc[data.index[:5], 'x1'] = np.nan
    found = auto.profile(_split(data))
    assert found.collinear and found.vif is not None and found.vif > 10
    assert found.unscaled and found.missing


def test_profile_is_plain_values_for_the_history() -> None:
    found = auto.profile(_split(make_numeric()))
    assert json.loads(json.dumps(found.to_dict())) == found.to_dict()


def test_profile_ignores_synthetic_rows() -> None:
    requires('imblearn')
    dataset = _split(make_numeric(rows = 300, weights = [0.85]))
    before = auto.profile(dataset)
    amos.samplers.RandomOver().apply(dataset)
    after = auto.profile(dataset)
    assert after.rows == before.rows
    assert after.balance == before.balance


""" Resolving "auto" """


@pytest.mark.parametrize(('step', 'genre'), [
    ('sample', 'sampler'), ('resample', 'sampler'), ('model', 'model'),
    ('fit', 'model'), ('plot', 'plot'), ('draw', 'plot'),
    ('impute', 'imputer'), ('encode', 'encoder'), ('evaluate', 'evaluator')])
def test_auto_is_resolved_from_its_step(step: str, genre: str) -> None:
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'demo_project': {'demo_workers': ['analyst']},
        'analyst': {'steps': [step], f'{step}_techniques': 'auto'},
    }
    project = amos.Project.create(settings, automatic = False)
    nodes = amos.interface._nodes(project.workflow)
    found = [node for node in nodes if isinstance(node, auto.Auto)]
    assert [node.genre for node in found] == [genre]


@pytest.mark.parametrize(('worker', 'genre'), [
    ('artist', 'plot'), ('critic', 'evaluator')])
def test_auto_is_resolved_from_its_worker(worker: str, genre: str) -> None:
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'demo_project': {'demo_workers': [worker]},
        worker: {'techniques': 'auto'},
    }
    project = amos.Project.create(settings, automatic = False)
    found = [
        node for node in amos.interface._nodes(project.workflow)
        if isinstance(node, auto.Auto)]
    assert [node.genre for node in found] == [genre]


def test_auto_in_an_unknown_step_is_an_error_when_drafted() -> None:
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'demo_project': {'demo_workers': ['analyst']},
        'analyst': {'steps': ['scale'], 'scale_techniques': 'auto'},
    }
    with pytest.raises(ValueError, match = "for 'scale'"):
        amos.Project.create(settings, automatic = False)


def test_auto_without_a_genre_is_an_error() -> None:
    with pytest.raises(ValueError, match = 'does not know'):
        auto.Auto().apply(_split(make_numeric()))
    with pytest.raises(ValueError, match = 'not \'scaler\''):
        auto.Auto(genre = 'scaler').apply(_split(make_numeric()))


def test_named_choosers_need_no_resolving() -> None:
    requires('statsmodels')
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'demo_project': {'demo_workers': ['analyst']},
        'analyst': {
            'steps': ['split', 'estimate_effects'],
            'split_techniques': 'train_test',
            'estimate_effects_techniques': 'auto_model'},
        'auto_model_estimate_effects_parameters': {
            'choices': ['linear', 'ridge'], 'folds': 1},
    }
    project = amos.Project.create(settings, item = make_regression())
    assert _chosen(project.result, 'auto_model') == ['linear']


""" AutoImputer """


def _gaps(missing: float, rows: int = 300) -> pd.DataFrame:
    """Returns numeric features (and a category) with some values missing."""
    data = make_numeric(rows = rows)
    rng = np.random.default_rng(SEED)
    data.loc[rng.random(rows) < missing, 'x0'] = np.nan
    data['court'] = rng.choice(['state', 'federal'], rows).astype(object)
    data.loc[data.index[:5], 'court'] = np.nan
    return data


def test_auto_imputer_leaves_complete_data_alone() -> None:
    dataset = _split(make_numeric())
    before = dataset.data.copy()
    auto.AutoImputer().apply(dataset)
    assert _chosen(dataset, 'auto_imputer') == ['none']
    assert dataset.data.equals(before)


def test_auto_imputer_uses_the_median_for_a_few_missing_values() -> None:
    dataset = _split(_gaps(missing = 0.02))
    auto.AutoImputer().apply(dataset)
    table = dataset.tables['auto_imputer']
    assert _chosen(dataset, 'auto_imputer') == ['median_impute', 'mode_impute']
    assert 'few values are missing' in table.loc['iterative_impute', 'reason']
    assert not dataset.data[dataset.features].isna().to_numpy().any()


def test_auto_imputer_models_features_with_many_missing_values() -> None:
    dataset = _split(_gaps(missing = 0.2))
    auto.AutoImputer().apply(dataset)
    table = dataset.tables['auto_imputer']
    assert _chosen(dataset, 'auto_imputer') == [
        'iterative_impute', 'mode_impute']
    assert 'filled the features it fills' in table.loc[
        'median_impute', 'reason']
    assert not dataset.data[dataset.features].isna().to_numpy().any()


def test_auto_imputer_warns_about_features_it_cannot_fill() -> None:
    dataset = _split(_gaps(missing = 0.02))
    with pytest.warns(UserWarning, match = r"could not fill \['court'\]"):
        auto.AutoImputer().apply(dataset, choices = ['median_impute'])
    assert dataset.data['court'].isna().any()


""" AutoEncoder """


def _levels(rows: int = 400, classes: int = 2) -> pd.DataFrame:
    """Returns features with few levels, many levels, and free text."""
    rng = np.random.default_rng(SEED)
    words = np.array(['the', 'court', 'held', 'that', 'motion', 'denied'])
    return pd.DataFrame({
        'x': rng.normal(size = rows),
        'court': rng.choice(['state', 'federal', 'tribal'], rows),
        'judge': [f'judge {j}' for j in rng.integers(0, 40, rows)],
        'note': [' '.join(rng.choice(words, 5)) for _ in range(rows)],
        'target': rng.integers(0, classes, rows)})


def test_auto_encoder_suits_the_levels_of_each_feature() -> None:
    requires('category_encoders')
    requires('skrub')
    dataset = _split(_levels())
    auto.AutoEncoder().apply(dataset)
    assert dataset.history[-1]['encoders'] == {
        'court': 'one_hot', 'judge': 'target', 'note': 'tfidf'}
    table = dataset.tables['auto_encoder']
    assert 'court (3 levels)' in table.loc['one_hot', 'reason']
    assert 'judge (40 levels)' in table.loc['target', 'reason']
    assert {'court_state', 'judge'} <= set(dataset.features)
    assert not dataset.categoricals


def test_auto_encoder_counts_levels_for_a_label_of_many_classes() -> None:
    requires('category_encoders')
    dataset = _split(_levels(classes = 3).drop(columns = 'note'))
    auto.AutoEncoder().apply(dataset)
    table = dataset.tables['auto_encoder']
    assert dataset.history[-1]['encoders']['judge'] == 'count'
    assert 'not 3 classes' in table.loc['target', 'reason']


def test_auto_encoder_limits_dummy_variables_without_other_encoders() -> None:
    dataset = _split(_levels().drop(columns = 'note'))
    auto.AutoEncoder().apply(dataset, choices = 'one_hot', levels = 5)
    judges = [c for c in dataset.features if c.startswith('judge_')]
    assert len(judges) == 5
    assert 'at most 5 dummy variables' in dataset.tables[
        'auto_encoder'].loc['one_hot', 'reason']


def test_auto_encoder_passes_parameters_to_the_encoders() -> None:
    dataset = _split(_levels().drop(columns = ['judge', 'note']))
    auto.AutoEncoder().apply(dataset, drop = 'first')
    assert len([c for c in dataset.features if c.startswith('court_')]) == 2


def test_auto_encoder_splits_dates_and_leaves_numbers_alone() -> None:
    requires('skrub')
    data = make_numeric()
    data['filed'] = pd.date_range('2020-01-01', periods = len(data))
    dataset = _split(data)
    auto.AutoEncoder().apply(dataset)
    assert _chosen(dataset, 'auto_encoder') == ['date_parts']
    assert not dataset.dates
    dataset = _split(make_numeric())
    auto.AutoEncoder().apply(dataset)
    assert _chosen(dataset, 'auto_encoder') == ['none']


def test_auto_encoder_explains_why_no_encoder_suits_a_feature() -> None:
    dataset = _split(_levels(classes = 3).drop(columns = 'note'))
    with pytest.raises(
        ValueError, match = "no encoder for 'court': target needs a label"):
        auto.AutoEncoder().apply(dataset, choices = ['target'])


""" AutoSampler """


def test_auto_sampler_does_not_resample_balanced_classes() -> None:
    dataset = _split(make_numeric(rows = 200))
    rows = len(dataset.train)
    auto.AutoSampler().apply(dataset)
    assert len(dataset.train) == rows
    assert _chosen(dataset, 'auto_sampler') == ['none']
    assert 'balanced' in dataset.tables['auto_sampler'].loc['none', 'reason']
    record = dataset.history[-1]
    assert record['technique'] == 'auto_sampler'
    assert record['chosen'] == []
    assert record['profile']['balance'] > 0.5


def test_auto_sampler_does_not_resample_a_regressed_label() -> None:
    dataset = _split(make_regression())
    auto.AutoSampler().apply(dataset)
    assert _chosen(dataset, 'auto_sampler') == ['none']


def test_auto_sampler_uses_smote_for_numeric_features() -> None:
    requires('imblearn')
    dataset = _split(make_numeric(rows = 400, weights = [0.85]))
    before = dataset.y_train.value_counts()
    auto.AutoSampler().apply(dataset)
    chosen = _chosen(dataset, 'auto_sampler')
    assert chosen in (['smote'], ['smote_tomek'])
    after = dataset.y_train.value_counts()
    assert after.min() / after.max() > before.min() / before.max()
    assert dataset.history[-2]['technique'] == chosen[0]
    assert dataset.history[-1]['chosen'] == chosen


def test_auto_sampler_duplicates_rows_for_binary_or_missing_features() -> None:
    requires('imblearn')
    data = make_numeric(rows = 400, weights = [0.85])
    binary = data.copy()
    for column in ['x0', 'x1', 'x2', 'x3']:
        binary[column] = (binary[column] > 0).astype(int)
    dataset = _split(binary)
    auto.AutoSampler().apply(dataset)
    assert _chosen(dataset, 'auto_sampler') == ['random_over']
    assert 'binary' in dataset.tables['auto_sampler'].loc['smote', 'reason']
    missing = data.copy()
    missing.loc[missing.index[:10], 'x0'] = np.nan
    dataset = _split(missing)
    auto.AutoSampler().apply(dataset)
    assert _chosen(dataset, 'auto_sampler') == ['random_over']
    assert 'missing' in dataset.tables['auto_sampler'].loc['smote', 'reason']


def test_auto_sampler_needs_enough_rare_rows_for_smote() -> None:
    requires('imblearn')
    data = make_numeric(rows = 200, weights = [0.97])
    dataset = _split(data)
    auto.AutoSampler().apply(dataset)
    reasons = dataset.tables['auto_sampler']['reason']
    if dataset.y_train.value_counts().min() < 6:
        assert _chosen(dataset, 'auto_sampler') == ['random_over']
        assert 'at least 6 rows' in reasons['smote']
    assert 'too few rows' in reasons['random_under']


def test_auto_sampler_chooses_from_choices_in_order() -> None:
    requires('imblearn')
    dataset = _split(make_numeric(rows = 400, weights = [0.85]))
    auto.AutoSampler().apply(dataset, choices = ['random_over', 'smote'])
    table = dataset.tables['auto_sampler']
    assert list(table.index) == ['random_over', 'smote']
    assert _chosen(dataset, 'auto_sampler') == ['random_over']
    assert 'preferred' in table.loc['smote', 'reason']


""" AutoModel """


def test_auto_model_records_every_model_it_considered() -> None:
    dataset = _split(make_numeric(rows = 200))
    auto.AutoModel().apply(
        dataset, choices = ['logit', 'random_forest', 'naive_bayes'])
    table = dataset.tables['auto_model']
    assert list(table.index) == ['logit', 'random_forest', 'naive_bayes']
    # Two of the features are combinations of the others.
    assert 'collinear' in table.loc['naive_bayes', 'reason']
    table = table.drop(index = 'naive_bayes')
    assert list(table.columns) == [
        'chosen', 'reason', 'score', 'standard_error']
    assert table['chosen'].sum() == 1
    assert table['score'].notna().all()
    chosen = _chosen(dataset, 'auto_model')[0]
    assert dataset.model is not None
    assert dataset.predictions is not None
    assert dataset.history[-2]['technique'] == chosen
    record = dataset.history[-1]
    assert record['technique'] == 'auto_model'
    assert record['chosen'] == [chosen]
    assert (record['metric'], record['folds']) == ('balanced_accuracy', 5)


def test_auto_model_considers_only_models_for_the_task() -> None:
    dataset = _split(make_regression())
    auto.AutoModel().apply(dataset, candidates = 1)
    table = dataset.tables['auto_model']
    assert 'logit' not in table.index and 'naive_bayes' not in table.index
    assert 'linear' in table.index
    assert table['score'].isna().all()
    assert _chosen(dataset, 'auto_model') == ['linear']


def test_auto_model_rules_out_linear_models_for_non_normal_errors() -> None:
    data = make_regression(rows = 300)
    rng = np.random.default_rng(SEED)
    data['target'] = data['x0'] + rng.lognormal(0, 1, len(data))
    dataset = _split(data)
    auto.AutoModel().apply(
        dataset, choices = ['linear', 'random_forest'], folds = 3)
    reason = dataset.tables['auto_model'].loc['linear', 'reason']
    assert reason.startswith('assumes normal errors')
    assert _chosen(dataset, 'auto_model') == ['random_forest']


def test_auto_model_rules_out_linear_models_for_collinear_features() -> None:
    data = make_regression()
    data['copy'] = data['x0'] * 2 + 0.01
    dataset = _split(data)
    auto.AutoModel().apply(dataset, choices = ['linear', 'random_forest'])
    reason = dataset.tables['auto_model'].loc['linear', 'reason']
    assert 'collinear' in reason


def test_auto_model_rules_out_naive_bayes_for_skewed_features() -> None:
    data = make_numeric(rows = 200)
    for column in ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']:
        data[column] = np.exp(data[column] * 2)
    dataset = _split(data)
    auto.AutoModel().apply(
        dataset, choices = ['naive_bayes', 'random_forest'], folds = 3)
    reason = dataset.tables['auto_model'].loc['naive_bayes', 'reason']
    assert reason.startswith('assumes normal features')


def test_auto_model_chooses_count_models_by_their_dispersion() -> None:
    requires('statsmodels')
    choices = ['poisson', 'negative_binomial', 'linear']
    dataset = _split(_counts(), label = 'count', task = 'regress')
    auto.AutoModel().apply(dataset, choices = choices, folds = 3)
    table = dataset.tables['auto_model']
    assert 'overdispersed' in table.loc['poisson', 'reason']
    assert _chosen(dataset, 'auto_model') == ['negative_binomial']
    dataset = _split(
        _counts(overdispersed = False), label = 'count', task = 'regress')
    auto.AutoModel().apply(dataset, choices = choices, folds = 3)
    table = dataset.tables['auto_model']
    assert table.loc['negative_binomial', 'reason'].startswith(
        'is for overdispersed counts')
    assert _chosen(dataset, 'auto_model')[0] in ('poisson', 'linear')


def test_auto_model_rules_out_robust_regression_for_counts() -> None:
    requires('statsmodels')
    dataset = _split(_counts(), label = 'count', task = 'regress')
    auto.AutoModel().apply(
        dataset, choices = ['robust_regression', 'random_forest'], folds = 1)
    reason = dataset.tables['auto_model'].loc['robust_regression', 'reason']
    assert reason == 'is for continuous errors, but the label is counts'


def test_auto_model_uses_models_that_handle_missing_values() -> None:
    data = make_numeric(rows = 200)
    data.loc[data.index[:20], 'x0'] = np.nan
    dataset = _split(data)
    auto.AutoModel().apply(
        dataset, choices = ['logit', 'gradient_boosting'], folds = 1)
    table = dataset.tables['auto_model']
    assert 'missing values' in table.loc['logit', 'reason']
    assert _chosen(dataset, 'auto_model') == ['gradient_boosting']


def test_auto_model_explains_why_no_model_suits_the_data(
    mixed: amos.Dataset) -> None:
    dataset = amos.splitters.Stratified().apply(mixed)
    with pytest.raises(ValueError, match = 'no model that suits'):
        auto.AutoModel().apply(dataset, choices = ['logit', 'random_forest'])


def test_auto_model_needs_features_on_similar_scales_for_some_models() -> None:
    data = make_numeric(rows = 200)
    data['x0'] = data['x0'] * 1000
    dataset = _split(data)
    auto.AutoModel().apply(dataset, choices = ['knn', 'random_forest'])
    assert 'similar scales' in dataset.tables['auto_model'].loc['knn', 'reason']
    assert _chosen(dataset, 'auto_model') == ['random_forest']


def test_auto_model_cross_validates_only_some_candidates() -> None:
    dataset = _split(make_numeric(rows = 200))
    auto.AutoModel().apply(
        dataset,
        choices = ['logit', 'random_forest', 'decision_tree'],
        candidates = 2,
        folds = 3)
    table = dataset.tables['auto_model']
    assert table['score'].notna().tolist() == [True, True, False]
    assert 'only the first 2' in table.loc['decision_tree', 'reason']


def test_auto_model_without_cross_validation_chooses_the_first() -> None:
    dataset = _split(make_numeric(rows = 200))
    auto.AutoModel().apply(
        dataset, choices = ['random_forest', 'logit'], folds = 1)
    assert _chosen(dataset, 'auto_model') == ['random_forest']
    assert 'metric' not in dataset.history[-1]


def test_auto_model_passes_parameters_to_the_model() -> None:
    dataset = _split(make_numeric(rows = 200))
    auto.AutoModel().apply(
        dataset, choices = ['random_forest'], n_estimators = 7)
    assert dataset.model.n_estimators == 7


def test_auto_model_skips_models_that_cannot_be_fitted() -> None:
    dataset = _split(make_numeric(rows = 200))
    with pytest.warns(UserWarning, match = "left out 'logit'"):
        auto.AutoModel().apply(
            dataset,
            choices = ['logit', 'random_forest'],
            folds = 3,
            max_iter = 'not a number')
    table = dataset.tables['auto_model']
    assert table.loc['logit', 'reason'].startswith('could not be fitted')
    assert _chosen(dataset, 'auto_model') == ['random_forest']


def test_auto_model_names_unknown_choices() -> None:
    dataset = _split(make_numeric(rows = 200))
    auto.AutoModel().apply(
        dataset, choices = ['no_such_model', 'standard', 'logit'], folds = 1)
    table = dataset.tables['auto_model']
    assert table.loc['no_such_model', 'reason'] == 'is not a technique'
    assert table.loc['standard', 'reason'] == 'is not a model'


def test_best_prefers_the_simplest_model_within_one_standard_error() -> None:
    scores = {'logit': (0.90, 0.02), 'random_forest': (0.91, 0.02)}
    assert auto._best(scores, better = True, simplest = True) == 'logit'
    assert auto._best(scores, better = True, simplest = False) == (
        'random_forest')
    scores = {'logit': (0.80, 0.02), 'random_forest': (0.91, 0.02)}
    assert auto._best(scores, better = True, simplest = True) == (
        'random_forest')
    errors = {'linear': (10.0, 1.0), 'random_forest': (9.5, 1.0)}
    assert auto._best(errors, better = False, simplest = True) == 'linear'
    assert auto._best(errors, better = False, simplest = False) == (
        'random_forest')


def test_auto_model_chooses_inference_models_by_the_classes() -> None:
    requires('statsmodels')
    choices = ['ordinal_regression', 'logit_sm', 'mnlogit']
    dataset = _split(make_numeric(rows = 300))
    auto.AutoModel().apply(dataset, choices = choices, folds = 1)
    table = dataset.tables['auto_model']
    assert _chosen(dataset, 'auto_model') == ['logit_sm']
    assert table.loc['mnlogit', 'reason'] == 'is for 3 or more classes, not 2'
    data = make_numeric(rows = 300, classes = 3)
    data['target'] = data['target'].map({0: 'a', 1: 'b', 2: 'c'})
    dataset = _split(data)
    auto.AutoModel().apply(dataset, choices = choices, folds = 1)
    table = dataset.tables['auto_model']
    assert _chosen(dataset, 'auto_model') == ['mnlogit']
    assert table.loc['logit_sm', 'reason'] == 'is for 2 classes, not 3'


""" AutoEvaluator """


def test_auto_evaluator_suits_a_classifier() -> None:
    requires('shap')
    dataset = _split(make_numeric(rows = 200))
    amos.models.RandomForest().apply(dataset, n_estimators = 20)
    auto.AutoEvaluator().apply(dataset)
    chosen = _chosen(dataset, 'auto_evaluator')
    assert chosen == [
        'scorecard', 'classification_report', 'confusion',
        'feature_importance', 'permutation_importance', 'shap_importance']
    assert {'confusion', 'permutation_importance', 'shap_importance'} <= set(
        dataset.tables)
    reason = dataset.tables['auto_evaluator'].loc['regression_table', 'reason']
    assert reason == 'random_forest has no coefficients with standard errors'


def test_auto_evaluator_tabulates_coefficients() -> None:
    requires('statsmodels')
    dataset = _split(make_regression())
    amos.models.OLS().apply(dataset)
    auto.AutoEvaluator().apply(dataset)
    chosen = _chosen(dataset, 'auto_evaluator')
    assert 'regression_table' in chosen
    assert 'classification_report' not in chosen
    assert 'permutation_importance' not in chosen
    assert 'regression_table' in dataset.tables


def test_auto_evaluator_needs_predictions() -> None:
    with pytest.raises(ValueError, match = 'needs the predictions'):
        auto.AutoEvaluator().apply(_split(make_numeric()))


""" AutoPlot """


@pytest.fixture
def _plotting() -> None:
    """Skips a test if the plotting packages cannot be imported."""
    requires('matplotlib')
    requires('seaborn')


@pytest.mark.usefixtures('_plotting')
def test_auto_plot_draws_the_data_before_a_model() -> None:
    data = make_numeric(rows = 200)
    data.loc[data.index[:5], 'x0'] = np.nan
    dataset = _split(data)
    auto.AutoPlot().apply(dataset)
    drawn = _chosen(dataset, 'auto_plot')
    assert {'label_plot', 'missing_heatmap', 'histograms', 'box_plots'} <= (
        set(drawn))
    assert 'pair_plot' not in drawn
    assert 'count_plots' not in drawn
    assert set(dataset.figures) == set(drawn)
    assert dataset.history[-1]['chosen'] == drawn


@pytest.mark.usefixtures('_plotting')
def test_auto_plot_draws_a_classifier() -> None:
    dataset = _split(make_numeric(rows = 300, weights = [0.8]))
    amos.models.RandomForest().apply(dataset)
    auto.AutoPlot().apply(dataset, choices = [
        'confusion_heatmap', 'roc_curve', 'precision_recall_curve',
        'calibration_curve', 'qq_plot', 'importance_plot', 'histograms'])
    table = dataset.tables['auto_plot']
    assert _chosen(dataset, 'auto_plot') == [
        'confusion_heatmap', 'roc_curve', 'precision_recall_curve',
        'importance_plot']
    assert 'histograms' not in table.index
    assert 'qq_plot' not in table.index
    assert 'at least 100' in table.loc['calibration_curve', 'reason']


@pytest.mark.usefixtures('_plotting')
def test_auto_plot_draws_the_assumptions_of_a_linear_regression() -> None:
    requires('statsmodels')
    dataset = _split(make_regression())
    amos.models.OLS().apply(dataset)
    auto.AutoPlot().apply(dataset)
    drawn = _chosen(dataset, 'auto_plot')
    assert drawn[:2] == ['actual_vs_predicted', 'residual_plot']
    assert {'qq_plot', 'influence_plot', 'coefficient_plot'} <= set(drawn)
    assert 'partial_dependence' not in drawn
    reason = dataset.tables['auto_plot'].loc['partial_dependence', 'reason']
    assert reason.endswith('is linear, so its coefficients show its shape')


@pytest.mark.usefixtures('_plotting')
def test_auto_plot_draws_the_data_with_a_model_if_asked() -> None:
    dataset = _split(make_regression())
    amos.models.Linear().apply(dataset)
    auto.AutoPlot().apply(
        dataset, data = True, choices = ['histograms', 'residual_plot'])
    assert _chosen(dataset, 'auto_plot') == ['histograms', 'residual_plot']


@pytest.mark.usefixtures('_plotting')
def test_auto_plot_leaves_out_plots_that_cannot_be_drawn(
    monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _split(make_regression())
    amos.models.Linear().apply(dataset)

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError('cannot draw this')

    monkeypatch.setattr(amos.plots.ResidualPlot, 'draw', fail)
    with pytest.warns(UserWarning, match = "left out 'residual_plot'"):
        auto.AutoPlot().apply(
            dataset, choices = ['actual_vs_predicted', 'residual_plot'])
    table = dataset.tables['auto_plot']
    assert _chosen(dataset, 'auto_plot') == ['actual_vs_predicted']
    assert table.loc['residual_plot', 'reason'] == (
        'could not be drawn: cannot draw this')
    with pytest.raises(ValueError, match = 'cannot draw this'), pytest.warns(
        UserWarning):
        auto.AutoPlot().apply(dataset, choices = ['residual_plot'])


""" Projects """


@pytest.mark.usefixtures('_plotting')
def test_a_project_can_choose_every_kind_of_technique() -> None:
    requires('imblearn')
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'demo_project': {'demo_workers': ['analyst', 'artist']},
        'analyst': {
            'steps': ['split', 'scale', 'sample', 'model'],
            'split_techniques': 'stratified',
            'scale_techniques': 'standard',
            'sample_techniques': 'auto',
            'model_techniques': 'auto'},
        'auto_model_parameters': {
            'choices': ['logit', 'random_forest'], 'folds': 3},
        'artist': {'techniques': 'auto'},
    }
    data = make_numeric(rows = 300, weights = [0.85])
    project = amos.Project.create(settings, item = data)
    result = project.result
    assert set(result.tables) >= {'auto_sampler', 'auto_model', 'auto_plot'}
    assert _chosen(result, 'auto_sampler') in (['smote'], ['smote_tomek'])
    assert _chosen(result, 'auto_model')[0] in ('logit', 'random_forest')
    assert 'precision_recall_curve' in result.figures
    techniques = [record['technique'] for record in result.history]
    assert techniques.count('auto_sampler') == 1
    assert techniques.count('auto_plot') == 1
    second = amos.Project.create(settings, item = data).result
    assert second.predictions.equals(result.predictions)


def test_an_experiment_can_compare_auto_with_other_techniques() -> None:
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'demo_project': {'demo_workers': ['analyst']},
        'analyst': {
            'design': 'experiment',
            'criterion': 'accuracy',
            'steps': ['split', 'model'],
            'split_techniques': 'stratified',
            'model_techniques': 'auto, baseline'},
        'auto_model_parameters': {
            'choices': ['naive_bayes', 'random_forest'], 'folds': 3},
    }
    project = amos.Project.create(settings, item = make_numeric())
    table = project.result.tables['analyst_comparison']
    assert list(table.index) == [
        'stratified_split > auto_model', 'stratified_split > baseline_model']
