"""Tests the encoders."""

from __future__ import annotations

import pytest
from conftest import package_of, requires, split_mixed, techniques

import amos

# `date_parts` encodes dates, so it is tested on its own.
ENCODERS = [(n, k) for n, k in techniques('encoder') if n != 'date_parts']


@pytest.mark.parametrize(('name', 'kind'), ENCODERS)
def test_encoders_make_categories_numeric(
    name: str,
    kind: type[amos.Encoder]) -> None:
    requires(package_of(kind))
    dataset = split_mixed()
    kind().apply(dataset)
    assert dataset.categoricals == []
    assert dataset.numerics[0] == 'age'
    assert dataset.history[-1]['technique'] == name
    assert dataset.history[-1]['columns'] == ['region']


def test_date_parts_lets_models_use_dates(mixed: amos.Dataset) -> None:
    requires('skrub')
    amos.splitters.Stratified().apply(mixed)
    amos.encoders.DateParts().apply(mixed)
    assert mixed.dates == []
    assert 'joined_year' in mixed.numerics
    assert mixed.history[-1]['columns'] == ['joined']
