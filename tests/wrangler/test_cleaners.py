"""Tests the cleaners module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import amos
from amos import cleaners
from amos.wrangler.cleaners import _rows


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
    query = _rows._condition_query
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


def test_clean_numbers(cases: amos.Dataset) -> None:
    cases.data['notes'] = pd.Series(
        ['12%', 'about 3 years', '-1.5e3', '.5'], dtype = 'str')
    cleaners.CleanNumbers().apply(cases, columns = ['damages', 'notes'])
    assert cases.data['damages'].tolist()[0::2] == [1200.0, 350.5]
    assert cases.data['damages'].tolist()[3] == 15000.0
    assert pd.isna(cases.data['damages'].iloc[1])
    assert cases.data['notes'].tolist() == [12.0, 3.0, -1500.0, 0.5]
    assert 'damages' in cases.numerics
    data = pd.DataFrame({'amount': ['1.234,50', '7'], 'count': [1, 2]})
    result = cleaners.CleanNumbers().apply(
        data, columns = ['amount', 'count'], thousands = '.', decimal = ',')
    assert result.data['amount'].tolist() == [1234.5, 7.0]
    assert result.data['count'].dtype == 'int64'
    with pytest.raises(ValueError, match = 'must differ'):
        cleaners.CleanNumbers().apply(data, thousands = ',', decimal = ',')


def test_clean_booleans(cases: amos.Dataset) -> None:
    cases.data['flag'] = [1, 0, 2, np.nan]
    cleaners.CleanBooleans().apply(cases, columns = ['appealed', 'flag'])
    assert cases.data['appealed'].dtype == 'boolean'
    assert cases.data['appealed'].tolist()[:2] == [True, False]
    assert cases.data['appealed'].isna().tolist() == [
        False, False, True, True]
    assert cases.data['flag'].tolist()[:2] == [True, False]
    assert cases.data['flag'].isna().sum() == 2
    assert 'appealed' in cases.booleans
    cleaners.CleanBooleans().apply(
        cases, columns = 'disposition', true_values = 'affirmed.',
        false_values = ['vacated in part'])
    assert cases.data['disposition'].tolist()[1:3] == [True, False]


def test_clean_dates(cases: amos.Dataset) -> None:
    cleaners.CleanDates().apply(cases, columns = ['decided'])
    assert cases.data['decided'].tolist()[:2] == [
        pd.Timestamp('2020-01-05'), pd.Timestamp('2021-03-04')]
    assert cases.data['decided'].isna().tolist() == [
        False, False, True, True]
    assert cases.dates == ['decided']
    data = pd.DataFrame({'when': ['January 5, 2010', '2020-03-04']})
    result = cleaners.CleanDates().apply(
        data, columns = 'when', date_format = 'mixed')
    assert result.data['when'].dt.year.tolist() == [2010, 2020]


def test_cleaners_record_changed_columns(cases: amos.Dataset) -> None:
    cleaners.CleanNumbers().apply(cases, columns = ['damages'])
    assert cases.history[-1] == {
        'technique': 'clean_numbers', 'rows': [4, 4], 'columns': [7, 7],
        'changed': ['damages']}
    for kind in (
        cleaners.CleanBooleans, cleaners.CleanDates, cleaners.CleanNumbers):
        before = cases.data.copy()
        kind().apply(cases)
        assert 'changed' not in cases.history[-1], kind
        pd.testing.assert_frame_equal(cases.data, before)
    assert amos.library.classify('clean_numbers') == 'cleaner'
