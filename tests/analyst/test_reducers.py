"""Tests the reducers."""

from __future__ import annotations

import pytest

import amos

REDUCERS = sorted(amos.library.get_genre('reducer').items())


@pytest.mark.parametrize(('name', 'kind'), REDUCERS)
def test_reducers_combine_features(
    name: str,
    kind: type[amos.Reducer],
    classified: amos.Dataset) -> None:
    classified.data['constant'] = 1.0
    kind().apply(classified)
    assert 0 < len(classified.features) < 7
    assert classified.history[-1]['technique'] == name
