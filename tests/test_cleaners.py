"""Tests the cleaners module."""

from __future__ import annotations

import pandas as pd
import pytest

import amos
from amos import cleaners


def test_drop_columns(mixed: amos.Dataset) -> None:
    cleaners.DropColumns().apply(mixed, columns = ['joined', 'income'])
    assert mixed.features == ['age', 'region', 'member']
    with pytest.raises(KeyError, match = 'not in the data'):
        cleaners.DropColumns().apply(mixed, columns = 'missing')
    with pytest.raises(KeyError, match = 'label'):
        cleaners.DropColumns().apply(mixed, columns = 'outcome')


def test_drop_constant(mixed: amos.Dataset) -> None:
    mixed.data['constant'] = 1
    cleaners.DropConstant().apply(mixed)
    assert 'constant' not in mixed.data.columns
    assert 'age' in mixed.data.columns


def test_drop_duplicates_records_rows(mixed: amos.Dataset) -> None:
    mixed.data = pd.concat([mixed.data, mixed.data.head(5)])
    cleaners.DropDuplicates().apply(mixed)
    assert len(mixed.data) == 200
    assert mixed.history[-1] == {
        'technique': 'drop_duplicates', 'rows': [205, 200], 'columns': [6, 6]}
    cleaners.DropDuplicates().apply(mixed, columns = ['region'])
    assert len(mixed.data) == 4


def test_drop_missing(mixed: amos.Dataset) -> None:
    missing = int(mixed.data['age'].isna().sum())
    cleaners.DropMissing().apply(mixed, columns = ['income'])
    assert len(mixed.data) == 200
    cleaners.DropMissing().apply(mixed)
    assert len(mixed.data) == 200 - missing


def test_drop_missing_keeps_the_split_consistent(
    mixed: amos.Dataset) -> None:
    amos.splitters.TrainTest().apply(mixed)
    cleaners.DropMissing().apply(mixed)
    assert len(mixed.x_train) + len(mixed.x_test) == len(mixed.data)


def test_filter_rows(mixed: amos.Dataset) -> None:
    cleaners.FilterRows().apply(mixed)
    assert len(mixed.data) == 200
    cleaners.FilterRows().apply(mixed, query = 'income > 30000')
    assert (mixed.data['income'] > 30000).all()


def test_filter_rows_by_conditions_for_columns(mixed: amos.Dataset) -> None:
    cleaners.FilterRows().apply(mixed, region = 'not in [north, south]')
    assert set(mixed.data['region'].dropna()) <= {'east', 'west'}
    cleaners.FilterRows().apply(mixed, region = 'not east')
    assert set(mixed.data['region'].dropna()) == {'west'}
    before = len(mixed.data)
    cleaners.FilterRows().apply(
        mixed, query = 'income > 30000', member = True, age = '>= 30')
    data = mixed.data
    assert len(data) < before
    assert (data['income'] > 30000).all()
    assert data['member'].all()
    assert (data['age'] >= 30).all()
    with pytest.raises(KeyError, match = 'not in the data'):
        cleaners.FilterRows().apply(mixed, missing = 1)


def test_filter_rows_condition_queries() -> None:
    query = cleaners._condition_query
    assert query('state', "not in ['GU']") == "`state` not in ['GU']"
    assert query('code', 'in [1, 2]') == '`code` in [1, 2]'
    assert query('flag', 'not 1') == '`flag` != 1'
    assert query('age', '>= 18') == '`age` >= 18'
    assert query('race', 'W') == '`race` == W'
    assert query('race', ['W', 'B']) == "`race` in ['W', 'B']"
    assert query('count', 2) == '`count` == 2'
    assert query('index', 'indigo') == '`index` == indigo'


def test_keep_columns_always_keeps_the_label(mixed: amos.Dataset) -> None:
    cleaners.KeepColumns().apply(mixed)
    assert len(mixed.data.columns) == 6
    cleaners.KeepColumns().apply(mixed, columns = ['age', 'region'])
    assert list(mixed.data.columns) == ['age', 'region', 'outcome']


def test_rename_columns_renames_the_label(mixed: amos.Dataset) -> None:
    cleaners.RenameColumns().apply(
        mixed, names = {'outcome': 'result', 'age': 'years'})
    assert mixed.label == 'result'
    assert 'years' in mixed.features
    assert mixed.y.name == 'result'


def test_rename_columns_by_parameter(mixed: amos.Dataset) -> None:
    mixed.kinds['age'] = 'numerics'
    cleaners.RenameColumns().apply(
        mixed, names = {'income': 'earnings'}, age = 'years')
    assert {'years', 'earnings'} <= set(mixed.features)
    assert mixed.kinds == {'years': 'numerics'}
