"""Tests the transformers module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_mixed, requires

import amos
from amos import transformers

SCALERS = sorted(amos.library.get_genre('scaler').items())
ENCODERS = sorted(amos.library.get_genre('encoder').items())
IMPUTERS = sorted(amos.library.get_genre('imputer').items())
MIXERS = sorted(amos.library.get_genre('mixer').items())
REDUCERS = sorted(amos.library.get_genre('reducer').items())


def _split_mixed() -> amos.Dataset:
    """Returns the mixed dataset, split, with its missing ages filled."""
    dataset = amos.Dataset(make_mixed(), label = 'outcome', seed = SEED)
    amos.splitters.Stratified().apply(dataset)
    return transformers.MedianImpute().apply(dataset)


@pytest.mark.parametrize(('name', 'kind'), SCALERS)
def test_scalers_change_only_numeric_features(
    name: str,
    kind: type[amos.Scaler]) -> None:
    dataset = _split_mixed()
    before = dataset.data[['region', 'member', 'outcome']].copy()
    kind().apply(dataset)
    assert dataset.numerics == ['age', 'income']
    assert not dataset.data[['age', 'income']].isna().any().any()
    pd.testing.assert_frame_equal(
        dataset.data[['region', 'member', 'outcome']], before)
    assert dataset.history[-1]['technique'] == name
    assert dataset.history[-1]['columns'] == ['age', 'income']
    assert name in dataset.fitted


@pytest.mark.parametrize(('name', 'kind'), ENCODERS)
def test_encoders_make_categories_numeric(
    name: str,
    kind: type[amos.Encoder]) -> None:
    if kind.contents.startswith('category_encoders'):
        requires('category_encoders')
    dataset = _split_mixed()
    kind().apply(dataset)
    assert dataset.categoricals == []
    assert dataset.numerics[0] == 'age'
    assert dataset.history[-1]['technique'] == name
    assert dataset.history[-1]['columns'] == ['region']


@pytest.mark.parametrize(('name', 'kind'), IMPUTERS)
def test_imputers_fill_missing_values(
    name: str,
    kind: type[amos.Imputer]) -> None:
    dataset = amos.Dataset(make_mixed(), label = 'outcome', seed = SEED)
    amos.splitters.TrainTest().apply(dataset)
    kind().apply(dataset)
    assert not dataset.data['age'].isna().any()
    assert dataset.history[-1]['columns'] == ['age']


def test_mode_impute_fills_text() -> None:
    dataset = amos.Dataset(make_mixed(), label = 'outcome')
    dataset.data.loc[[0, 1], 'region'] = None
    transformers.ModeImpute().apply(dataset)
    assert dataset.history[-1]['columns'] == ['age', 'region']
    assert not dataset.data['region'].isna().any()


@pytest.mark.parametrize(('name', 'kind'), MIXERS)
def test_mixers_add_features(name: str, kind: type[amos.Mixer]) -> None:
    dataset = _split_mixed()
    kind().apply(dataset)
    assert len(dataset.numerics) > 2
    assert dataset.history[-1]['technique'] == name


@pytest.mark.parametrize(('name', 'kind'), REDUCERS)
def test_reducers_select_or_combine_features(
    name: str,
    kind: type[amos.Reducer],
    classified: amos.Dataset) -> None:
    classified.data['constant'] = 1.0
    kind().apply(classified, **({'k': 3} if name == 'k_best' else {}))
    assert 0 < len(classified.features) < 7
    assert classified.history[-1]['technique'] == name


def test_k_best_keeps_every_feature_if_there_are_fewer_than_k(
    classified: amos.Dataset) -> None:
    transformers.KBest().apply(classified, k = 50)
    assert len(classified.features) == 6


def test_k_best_scores_by_the_task(regressed: amos.Dataset) -> None:
    transformers.KBest().apply(regressed, k = 2)
    tool = regressed.fitted['k_best']
    assert tool.score_func.__name__ == 'f_regression'
    assert len(regressed.features) == 2


def test_k_best_imports_a_scoring_function(classified: amos.Dataset) -> None:
    transformers.KBest().apply(
        classified,
        k = 2,
        score_func = 'sklearn.feature_selection.mutual_info_classif')
    tool = classified.fitted['k_best']
    assert tool.score_func.__name__ == 'mutual_info_classif'


def test_transformers_learn_from_the_training_rows_only(
    classified: amos.Dataset) -> None:
    transformers.Standard().apply(classified)
    train = classified.x_train.mean()
    test = classified.x_test.mean()
    assert np.allclose(train, 0)
    assert not np.allclose(test, 0)


def test_columns_can_be_chosen(classified: amos.Dataset) -> None:
    original = classified.data['x1'].copy()
    transformers.MinMax().apply(classified, columns = ['x0'])
    assert classified.data['x0'].min() >= -0.5
    pd.testing.assert_series_equal(classified.data['x1'], original)


def test_no_columns_are_recorded(classified: amos.Dataset) -> None:
    transformers.OneHot().apply(classified)
    assert classified.history[-1]['columns'] == []
    assert 'note' in classified.history[-1]


def test_any_transformer_can_be_wrapped(classified: amos.Dataset) -> None:
    technique = amos.Transformer(
        name = 'yeo_johnson',
        contents = 'sklearn.preprocessing.PowerTransformer')
    technique.apply(classified)
    assert classified.history[-1]['tool'] == (
        'sklearn.preprocessing.PowerTransformer')


def test_output_without_names_is_named(classified: amos.Dataset) -> None:
    class Halves:
        def fit(self, x):
            return self

        def transform(self, x):
            return np.asarray(x)[:, :2] / 2

    amos.Transformer(name = 'halves', contents = Halves).apply(classified)
    assert 'halves_0' in classified.features
    assert 'halves_1' in classified.features
