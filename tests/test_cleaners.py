"""Tests the cleaners module."""

from __future__ import annotations

import pandas as pd
import pytest

import amos
from amos import cleaners


def test_auto_categorize(mixed: amos.Dataset) -> None:
    mixed.data['few'] = [1, 2] * 100
    cleaners.AutoCategorize().apply(mixed)
    assert isinstance(mixed.data['region'].dtype, pd.CategoricalDtype)
    assert isinstance(mixed.data['few'].dtype, pd.CategoricalDtype)
    assert mixed.data['member'].dtype == bool
    assert pd.api.types.is_float_dtype(mixed.data['income'].dtype)
    assert 'few' in mixed.categoricals


def test_convert_types(mixed: amos.Dataset) -> None:
    cleaners.ConvertTypes().apply(mixed, types = {'member': 'int64'})
    assert mixed.data['member'].dtype == 'int64'
    assert 'member' in mixed.numerics


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


def test_strip_text(mixed: amos.Dataset) -> None:
    mixed.data['region'] = mixed.data['region'].str.upper() + '  '
    mixed.data.loc[0, 'region'] = '   '
    cleaners.StripText().apply(mixed, lowercase = True)
    assert set(mixed.data['region'].dropna()) == {
        'north', 'south', 'east', 'west'}
    assert pd.isna(mixed.data.loc[0, 'region'])
    assert mixed.data['outcome'].iloc[1] in {'yes', 'no'}
