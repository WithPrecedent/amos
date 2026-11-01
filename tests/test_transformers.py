"""Tests the transformers module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_mixed, package_of, requires, techniques

import amos
from amos import transformers

SCALERS = sorted(amos.library.get_genre('scaler').items())
# `date_parts` encodes dates, so it is tested on its own.
ENCODERS = [(n, k) for n, k in techniques('encoder') if n != 'date_parts']
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
    requires(package_of(kind))
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


def test_interactions_of_formulas() -> None:
    data = pd.DataFrame({
        'offender_race': pd.Categorical(['W', 'B', 'W', None]),
        'victim_race': ['B', 'B', 'W', 'W'],
        'age': [20.0, 30.0, np.nan, 50.0],
        'income': [1.0, 2.0, 3.0, 4.0],
        'member': [True, False, True, True],
        'won': [0, 1, 0, 1]})
    dataset = amos.Dataset(data, label = 'won')
    transformers.Interactions().apply(
        dataset,
        race = 'offender_race * victim_race',
        formulas = {'wealth': 'age * `income`'},
        joined = 'member * victim_race')
    result = dataset.data
    assert isinstance(result['race'].dtype, pd.CategoricalDtype)
    assert result['race'].tolist()[:3] == [
        'w_offender_b_victim', 'b_offender_b_victim', 'w_offender_w_victim']
    assert pd.isna(result['race'].iloc[3])
    assert result['wealth'].tolist()[:2] == [20.0, 60.0]
    assert pd.isna(result['wealth'].iloc[2])
    assert result['joined_b_victim_race'].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert result['joined_w_victim_race'].tolist() == [0.0, 0.0, 1.0, 1.0]
    # The columns that the interactions are made from are kept.
    assert {'offender_race', 'victim_race', 'age'} <= set(dataset.features)
    assert 'race' in dataset.categoricals
    with pytest.raises(ValueError, match = 'two or more columns'):
        transformers.Interactions().apply(dataset, bad = 'age *')
    with pytest.raises(KeyError, match = 'not in the data'):
        transformers.Interactions().apply(dataset, bad = 'age * missing')


def test_roles_are_the_parts_of_names_that_differ() -> None:
    roles = transformers._roles
    assert roles(['offender_race', 'victim_race']) == ['offender', 'victim']
    assert roles(['race_offender', 'race_victim']) == ['offender', 'victim']
    assert roles(['race', 'offender_race']) == ['race', 'offender_race']
    assert roles(['region']) == ['region']


@pytest.mark.parametrize(('name', 'kind'), REDUCERS)
def test_reducers_combine_features(
    name: str,
    kind: type[amos.Reducer],
    classified: amos.Dataset) -> None:
    classified.data['constant'] = 1.0
    kind().apply(classified)
    assert 0 < len(classified.features) < 7
    assert classified.history[-1]['technique'] == name


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


def test_date_parts_lets_models_use_dates(mixed: amos.Dataset) -> None:
    requires('skrub')
    amos.splitters.Stratified().apply(mixed)
    transformers.DateParts().apply(mixed)
    assert mixed.dates == []
    assert 'joined_year' in mixed.numerics
    assert mixed.history[-1]['columns'] == ['joined']


def test_column_wise_fits_a_copy_to_each_column() -> None:
    requires('skrub')
    data = make_mixed()
    data['county'] = data['region'].str.upper()
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    transformers.MinHash().apply(
        dataset, columns = ['region', 'county'], n_components = 2)
    tool = dataset.fitted['min_hash']
    assert isinstance(tool, transformers.ColumnWise)
    assert set(tool.transformers_) == {'region', 'county'}
    assert list(tool.get_feature_names_out()) == [
        'region_0', 'region_1', 'county_0', 'county_1']
