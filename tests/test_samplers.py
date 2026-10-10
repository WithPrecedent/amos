"""Tests the samplers module."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from conftest import SEED, make_numeric, requires

import amos

SAMPLERS = sorted(amos.library.get_genre('sampler').items())


def _imbalanced() -> amos.Dataset:
    """Returns a split dataset in which one class is 15% of the rows."""
    dataset = amos.Dataset(
        make_numeric(rows = 300, weights = [0.85]),
        label = 'target',
        seed = SEED)
    return amos.splitters.Stratified().apply(dataset)


@pytest.mark.parametrize(('name', 'kind'), SAMPLERS)
def test_samplers_change_only_the_training_rows(
    name: str,
    kind: type[amos.Sampler]) -> None:
    requires('imblearn')
    dataset = _imbalanced()
    test = dataset.data.loc[dataset.test].copy()
    before = dataset.y_train.value_counts()
    kind().apply(dataset)
    after = dataset.y_train.value_counts()
    pd.testing.assert_frame_equal(dataset.data.loc[dataset.test], test)
    # The classes are closer to balanced than before.
    assert after.min() / after.max() > before.min() / before.max()
    record = dataset.history[-1]
    assert record['technique'] == name
    assert record['tool'].startswith('imblearn.')
    assert record['before'] == before.to_dict()
    assert len(dataset.train.intersection(dataset.test)) == 0


def test_samplers_record_classes_that_can_be_exported() -> None:
    requires('imblearn')
    dataset = _imbalanced()
    # The classes of `pandas` booleans that allow missing values are `numpy`
    # values, which a json file cannot have as keys.
    dataset.data['target'] = dataset.data['target'].astype('boolean')
    amos.samplers.RandomUnder().apply(dataset)
    record = dataset.history[-1]
    for counts in (record['before'], record['after']):
        assert {type(label) for label in counts} == {bool}
        assert {type(count) for count in counts.values()} == {int}
        assert set(json.loads(json.dumps(counts))) == {'false', 'true'}
    assert record['after'][False] == record['after'][True]


def test_samplers_are_reproducible() -> None:
    requires('imblearn')
    first = amos.samplers.Smote().apply(_imbalanced())
    second = amos.samplers.Smote().apply(_imbalanced())
    pd.testing.assert_frame_equal(first.data, second.data)
