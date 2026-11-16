"""Tests the scalers."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import split_mixed

import amos

SCALERS = sorted(amos.library.get_genre('scaler').items())


@pytest.mark.parametrize(('name', 'kind'), SCALERS)
def test_scalers_change_only_numeric_features(
    name: str,
    kind: type[amos.Scaler]) -> None:
    dataset = split_mixed()
    before = dataset.data[['region', 'member', 'outcome']].copy()
    kind().apply(dataset)
    assert dataset.numerics == ['age', 'income']
    assert not dataset.data[['age', 'income']].isna().any().any()
    pd.testing.assert_frame_equal(
        dataset.data[['region', 'member', 'outcome']], before)
    assert dataset.history[-1]['technique'] == name
    assert dataset.history[-1]['columns'] == ['age', 'income']
    assert name in dataset.fitted
