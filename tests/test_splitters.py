"""Tests the splitters module."""

from __future__ import annotations

import pytest
from conftest import SEED, make_mixed, make_numeric

import amos
from amos import splitters


def test_train_test_is_reproducible() -> None:
    first = splitters.TrainTest().apply(
        amos.Dataset(make_numeric(), label = 'target', seed = SEED))
    second = splitters.TrainTest().apply(
        amos.Dataset(make_numeric(), label = 'target', seed = SEED))
    assert list(first.test) == list(second.test)
    assert len(first.test) == 50
    assert first.history[-1] == {
        'technique': 'train_test',
        'test_size': 0.25,
        'train': 150,
        'test': 50,
        'seed': SEED}


def test_train_test_without_shuffling_uses_the_last_rows() -> None:
    dataset = amos.Dataset(make_numeric(), label = 'target')
    splitters.TrainTest().apply(dataset, shuffle = False, test_size = 0.1)
    assert list(dataset.test) == list(range(180, 200))


def test_stratified_keeps_the_shares_of_the_classes() -> None:
    dataset = amos.Dataset(
        make_numeric(weights = [0.9]), label = 'target', seed = SEED)
    splitters.Stratified().apply(dataset)
    train = dataset.y_train.mean()
    test = dataset.y_test.mean()
    assert train == pytest.approx(test, abs = 0.02)


def test_group_split_keeps_groups_together() -> None:
    data = make_numeric()
    data['judge'] = [i % 20 for i in range(200)]
    dataset = amos.Dataset(data, label = 'target', seed = SEED)
    splitters.GroupSplit().apply(dataset, groups = 'judge')
    train = set(dataset.data.loc[dataset.train, 'judge'])
    test = set(dataset.data.loc[dataset.test, 'judge'])
    assert train.isdisjoint(test)
    with pytest.raises(ValueError, match = 'groups'):
        splitters.GroupSplit().apply(dataset)


def test_time_split_tests_the_latest_rows() -> None:
    data = make_mixed().sample(frac = 1, random_state = 1)
    dataset = amos.Dataset(data, label = 'outcome')
    splitters.TimeSplit().apply(dataset, order = 'joined', test_size = 0.2)
    latest = dataset.data.loc[dataset.train, 'joined'].max()
    assert (dataset.data.loc[dataset.test, 'joined'] > latest).all()
    assert len(dataset.test) == 40


@pytest.mark.parametrize('size', [0, 1, 1.5])
def test_test_size_must_be_a_share(size: float) -> None:
    dataset = amos.Dataset(make_numeric(), label = 'target')
    with pytest.raises(ValueError, match = 'between 0 and 1'):
        splitters.TrainTest().apply(dataset, test_size = size)
