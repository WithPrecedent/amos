"""Tests what every transformer does, and the base classes of transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import SEED, make_mixed, requires

import amos


def test_transformers_learn_from_the_training_rows_only(
    classified: amos.Dataset) -> None:
    amos.scalers.Standard().apply(classified)
    train = classified.x_train.mean()
    test = classified.x_test.mean()
    assert np.allclose(train, 0)
    assert not np.allclose(test, 0)


def test_columns_can_be_chosen(classified: amos.Dataset) -> None:
    original = classified.data['x1'].copy()
    amos.scalers.MinMax().apply(classified, columns = ['x0'])
    assert classified.data['x0'].min() >= -0.5
    pd.testing.assert_series_equal(classified.data['x1'], original)


def test_no_columns_are_recorded(classified: amos.Dataset) -> None:
    amos.encoders.OneHot().apply(classified)
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


def test_column_wise_fits_a_copy_to_each_column() -> None:
    requires('skrub')
    data = make_mixed()
    data['county'] = data['region'].str.upper()
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    amos.encoders.MinHash().apply(
        dataset, columns = ['region', 'county'], n_components = 2)
    tool = dataset.fitted['min_hash']
    assert isinstance(tool, amos.transformers.ColumnWise)
    assert set(tool.transformers_) == {'region', 'county'}
    assert list(tool.get_feature_names_out()) == [
        'region_0', 'region_1', 'county_0', 'county_1']
