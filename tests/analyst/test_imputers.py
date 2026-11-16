"""Tests the imputers."""

from __future__ import annotations

import pytest
from conftest import SEED, make_mixed

import amos

IMPUTERS = sorted(amos.library.get_genre('imputer').items())


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
    amos.imputers.ModeImpute().apply(dataset)
    assert dataset.history[-1]['columns'] == ['age', 'region']
    assert not dataset.data['region'].isna().any()
