"""Tests the mixers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import split_mixed

import amos
from amos.analyst.mixers import _interactions

MIXERS = sorted(amos.library.get_genre('mixer').items())


@pytest.mark.parametrize(('name', 'kind'), MIXERS)
def test_mixers_add_features(name: str, kind: type[amos.Mixer]) -> None:
    dataset = split_mixed()
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
    amos.mixers.Interactions().apply(
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
        amos.mixers.Interactions().apply(dataset, bad = 'age *')
    with pytest.raises(KeyError, match = 'not in the data'):
        amos.mixers.Interactions().apply(dataset, bad = 'age * missing')


def test_roles_are_the_parts_of_names_that_differ() -> None:
    roles = _interactions._roles
    assert roles(['offender_race', 'victim_race']) == ['offender', 'victim']
    assert roles(['race_offender', 'race_victim']) == ['offender', 'victim']
    assert roles(['race', 'offender_race']) == ['race', 'offender_race']
    assert roles(['region']) == ['region']
