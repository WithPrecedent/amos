"""Tests the validators module and the synthetic rows of a dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_numeric, make_regression, requires

import amos

VALIDATORS = sorted(amos.library.get_genre('validator').items())


def _grouped() -> amos.Dataset:
    """Returns a split binary dataset with a group column and a fitted model."""
    data = make_numeric(rows = 240)
    data['court'] = np.repeat(list('abcdefgh'), 30)
    dataset = amos.Dataset(
        data, label = 'target', seed = SEED, groups = ['court'])
    amos.splitters.Stratified().apply(dataset)
    amos.models.SkLogit().apply(dataset)
    return dataset


@pytest.mark.parametrize(('name', 'kind'), VALIDATORS)
def test_every_validator_scores_the_training_rows(
    name: str,
    kind: type[amos.Validator]) -> None:
    dataset = _grouped()
    test = dataset.data.loc[dataset.test].copy()
    model = dataset.model
    kind().apply(dataset, metrics = ['accuracy', 'roc_auc'])
    table = dataset.tables[name]
    assert {'cv_accuracy', 'cv_roc_auc'} <= set(dataset.metrics)
    assert 0 <= dataset.metrics['cv_accuracy'] <= 1
    record = dataset.history[-1]
    assert record['technique'] == name
    assert record['tool'].startswith('sklearn.model_selection.')
    assert record['rows'] == len(dataset.train)
    assert record['folds'] == len(table)
    # The test rows and the fitted model are not changed.
    pd.testing.assert_frame_equal(dataset.data.loc[dataset.test], test)
    assert dataset.model is model
    if kind.pools:
        assert len(table) == len(dataset.train)
        assert list(table.columns) == ['fold', 'actual', 'prediction']
    else:
        assert list(table.columns) == [
            'train', 'validation', 'accuracy', 'roc_auc']


def test_folds_are_shuffled_with_the_seed() -> None:
    first, second = _grouped(), _grouped()
    amos.validators.KFold().apply(first)
    amos.validators.KFold().apply(second)
    pd.testing.assert_frame_equal(
        first.tables['k_fold'], second.tables['k_fold'])
    # A splitter that does not shuffle is not given the seed, which
    # scikit-learn would refuse.
    amos.validators.KFold().apply(first, shuffle = False, n_splits = 3)
    assert first.history[-1]['folds'] == 3


def test_a_regression_is_scored_with_its_own_metrics(
    regressed: amos.Dataset) -> None:
    amos.models.Linear().apply(regressed)
    amos.validators.RepeatedKFold().apply(
        regressed, n_splits = 3, n_repeats = 2)
    assert len(regressed.tables['repeated_k_fold']) == 6
    assert {'cv_r2', 'cv_rmse', 'cv_mae'} <= set(regressed.metrics)
    scores = regressed.history[-1]['scores']['r2']
    assert set(scores) == {'mean', 'sd'}
    with pytest.raises(ValueError, match = 'classification task'):
        amos.validators.StratifiedKFold().apply(regressed)


def test_validators_check_what_they_need(classified: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'apply a model first'):
        amos.validators.KFold().apply(classified)
    amos.models.SkLogit().apply(classified)
    with pytest.raises(ValueError, match = 'groups'):
        amos.validators.GroupKFold().apply(classified)
    with pytest.raises(KeyError):
        amos.validators.KFold().apply(classified, metrics = ['missing'])


def test_a_group_column_can_be_named(classified: amos.Dataset) -> None:
    amos.models.SkLogit().apply(classified)
    # The model did not learn from "court", so the validator does not use it.
    classified.data['court'] = np.resize(list('abcde'), len(classified.data))
    amos.validators.LeaveOneGroupOut().apply(classified, groups = 'court')
    assert len(classified.tables['leave_one_group_out']) == 5


def test_time_series_split_orders_the_rows(regressed: amos.Dataset) -> None:
    amos.models.Linear().apply(regressed)
    # The rows are in the reverse of the order of "when".
    regressed.data['when'] = pd.date_range(
        '2020-01-01', periods = len(regressed.data))[::-1]
    amos.validators.TimeSeriesSplit().apply(regressed, n_splits = 4)
    unordered = regressed.tables['time_series_split']
    amos.validators.TimeSeriesSplit().apply(
        regressed, n_splits = 4, order = 'when')
    ordered = regressed.tables['time_series_split']
    # Each fold learns from all of the rows before it.
    assert ordered['train'].is_monotonic_increasing
    assert ordered['validation'].nunique() == 1
    assert not ordered['r2'].equals(unordered['r2'])


def test_leave_one_row_out_scores_every_prediction_together(
    classified: amos.Dataset) -> None:
    amos.models.SkLogit().apply(classified)
    amos.validators.LeaveOneRowOut().apply(classified)
    table = classified.tables['leave_one_row_out']
    assert table.index.equals(classified.train)
    accuracy = (table['actual'] == table['prediction']).mean()
    assert classified.metrics['cv_accuracy'] == pytest.approx(accuracy)
    assert 0 < classified.metrics['cv_f1'] <= 1
    assert 'sd' not in classified.history[-1]['scores']['f1']


def test_samplers_are_applied_again_and_their_rows_are_not_scored() -> None:
    requires('imblearn')
    dataset = amos.Dataset(
        make_numeric(rows = 300, weights = [0.85]),
        label = 'target',
        seed = SEED)
    amos.splitters.Stratified().apply(dataset)
    real = len(dataset.train)
    amos.samplers.Smote().apply(dataset)
    assert len(dataset.synthetic) == len(dataset.train) - real
    amos.models.SkLogit().apply(dataset)
    amos.validators.StratifiedKFold().apply(dataset)
    table = dataset.tables['stratified_k_fold']
    assert table['validation'].sum() == real
    assert dataset.history[-1]['rows'] == real
    # Each copy of the model learned from rows that smote balanced again.
    assert (table['train'] > real - table['validation']).all()


def test_the_synthetic_rows_of_a_dataset() -> None:
    requires('imblearn')
    dataset = amos.Dataset(
        make_numeric(rows = 300, weights = [0.85]),
        label = 'target',
        seed = SEED)
    amos.splitters.Stratified().apply(dataset)
    real = len(dataset.train)
    amos.samplers.RandomOver().apply(dataset)
    # Each extra copy of a row is made up, and the first copy is real.
    assert len(dataset.train) - len(dataset.synthetic) == real
    amos.samplers.RandomUnder().apply(dataset)
    kept = dataset.train[~dataset.train.isin(dataset.synthetic)]
    assert len(kept) <= real
    dataset.replace(dataset.data.drop(index = dataset.synthetic[:5]))
    assert not dataset.synthetic.isin(dataset.synthetic[:0]).any()
    assert dataset.synthetic.isin(dataset.data.index).all()


def test_a_model_without_new_groups_cannot_be_validated_by_group() -> None:
    requires('pyfixest')
    data = make_regression(rows = 240)
    data['court'] = np.repeat(list('abcdef'), 40)
    dataset = amos.Dataset(
        data, label = 'target', seed = SEED, groups = ['court'])
    amos.splitters.TrainTest().apply(dataset)
    amos.models.Fixest().apply(dataset, fixed_effects = 'court')
    amos.validators.KFold().apply(dataset)
    assert 'cv_r2' in dataset.metrics
    with pytest.raises(ValueError, match = 'could not predict'):
        amos.validators.GroupKFold().apply(dataset, n_splits = 3)


def test_a_model_with_numbered_classes_is_validated() -> None:
    requires('xgboost')
    dataset = amos.Dataset(
        make_numeric().assign(target = lambda d: d['target'].map(
            {0: 'no', 1: 'yes'})),
        label = 'target',
        seed = SEED)
    amos.splitters.Stratified().apply(dataset)
    amos.models.Xgboost().apply(dataset, n_estimators = 20)
    assert isinstance(dataset.model, amos.models.LabelCoded)
    amos.validators.StratifiedKFold().apply(dataset, n_splits = 3)
    assert 'cv_roc_auc' in dataset.metrics


def test_an_experiment_compares_the_scores_of_each_branch() -> None:
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'study_project': {'study_workers': 'analyst, critic'},
        'analyst': {
            'design': 'experiment',
            'criterion': 'accuracy',
            'steps': 'split, model, validate',
            'split_techniques': 'stratified',
            'model_techniques': 'sk_logit, baseline',
            'validate_techniques': 'stratified_k_fold'},
        'critic': {'techniques': 'scorecard'}}
    project = amos.Project.create(settings, item = make_numeric(), id = 'run')
    table = project.scorecard.table
    assert set(table['validate']) == {'stratified_k_fold'}
    assert table['cv_accuracy'].notna().all()
    comparison = project.result.tables['analyst_comparison']
    assert 'cv_accuracy' in comparison.columns
    # The comparison ranks the branches by the score on the test rows.
    assert table.loc[0, 'model'] == 'sk_logit'
