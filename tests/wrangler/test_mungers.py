"""Tests the mungers module."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import amos
from amos import mungers


def test_auto_categorize(mixed: amos.Dataset) -> None:
    mixed.data['few'] = [1, 2] * 100
    mungers.AutoCategorize().apply(mixed)
    assert isinstance(mixed.data['region'].dtype, pd.CategoricalDtype)
    assert isinstance(mixed.data['few'].dtype, pd.CategoricalDtype)
    assert mixed.data['member'].dtype == bool
    assert pd.api.types.is_float_dtype(mixed.data['income'].dtype)
    assert 'few' in mixed.categoricals


def test_convert_types(mixed: amos.Dataset) -> None:
    mungers.ConvertTypes().apply(mixed, types = {'member': 'int64'})
    assert mixed.data['member'].dtype == 'int64'
    assert 'member' in mixed.numerics


def test_strip_text(mixed: amos.Dataset) -> None:
    mixed.data['region'] = mixed.data['region'].str.upper() + '  '
    mixed.data.loc[0, 'region'] = '   '
    mungers.StripText().apply(mixed, lowercase = True)
    assert set(mixed.data['region'].dropna()) == {
        'north', 'south', 'east', 'west'}
    assert pd.isna(mixed.data.loc[0, 'region'])
    assert mixed.data['outcome'].iloc[1] in {'yes', 'no'}


def test_a_munger_records_what_it_changed_and_created(
    cases: amos.Dataset) -> None:
    mungers.StripText().apply(cases)
    assert cases.history[-1] == {
        'technique': 'strip_text',
        'changed': ['appealed'],
        'created': []}
    mungers.DeriveColumns().apply(
        cases, expressions = {'caption_length': 'caption.str.len()'})
    assert cases.history[-1] == {
        'technique': 'derive_columns', 'changed': [],
        'created': ['caption_length']}


def test_a_munger_cannot_remove_rows_or_columns(cases: amos.Dataset) -> None:

    @dataclasses.dataclass
    class FirstRows(amos.Munger):

        def munge(self, data, **kwargs):
            return data.head(2)

    @dataclasses.dataclass
    class NoDamages(amos.Munger):

        def munge(self, data, **kwargs):
            return data.drop(columns = 'damages')

    with pytest.raises(ValueError, match = 'only changes and adds columns'):
        FirstRows().apply(cases)
    with pytest.raises(ValueError, match = 'only changes and adds columns'):
        NoDamages().apply(cases)
    assert len(cases.data) == 4


def test_normalize_text(cases: amos.Dataset) -> None:
    cases.data.loc[0, 'caption'] = ' Café  v.\n O’Brien '
    mungers.NormalizeText().apply(
        cases,
        columns = ['caption'],
        case = 'upper',
        remove_accents = True,
        remove_punctuation = True)
    assert cases.data['caption'].tolist()[:3] == [
        'CAFE V OBRIEN', 'JONES V ACME CORP', 'IN RE DOE']
    mungers.NormalizeText().apply(cases)
    assert cases.data['appealed'].tolist()[1] == 'no'
    with pytest.raises(ValueError, match = 'case must be'):
        mungers.NormalizeText().apply(cases, case = 'snake')


def test_map_values(cases: amos.Dataset) -> None:
    mungers.MapValues().apply(cases, values = {'n/a': None, 'maybe': None})
    assert pd.isna(cases.data['damages'].iloc[1])
    assert pd.isna(cases.data['appealed'].iloc[2])
    assert cases.history[-1]['changed'] == ['damages', 'appealed']
    mungers.AutoCategorize().apply(cases, columns = ['won'])
    mungers.MapValues().apply(
        cases, columns = ['won'], values = {'yes': 'plaintiff'})
    assert isinstance(cases.data['won'].dtype, pd.CategoricalDtype)
    assert cases.data['won'].tolist() == ['plaintiff', 'no', 'plaintiff', 'no']
    data = pd.DataFrame({'age': [30, 99, 41]})
    result = mungers.MapValues().apply(data, values = {99: None})
    assert result.data['age'].isna().tolist() == [False, True, False]
    assert pd.api.types.is_float_dtype(result.data['age'].dtype)


def test_coalesce() -> None:
    data = pd.DataFrame({
        'filed': ['2020-01-01', None, None],
        'decided': ['2020-02-01', '2021-02-01', None]})
    result = mungers.Coalesce().apply(
        data.copy(), columns = ['filed', 'decided'], name = 'date')
    assert result.data['date'].tolist()[:2] == ['2020-01-01', '2021-02-01']
    assert pd.isna(result.data['date'].iloc[2])
    result = mungers.Coalesce().apply(data, columns = ['filed', 'decided'])
    assert result.data['filed'].tolist()[1] == '2021-02-01'
    assert result.history[-1]['changed'] == ['filed']


def test_combine_flags() -> None:
    data = pd.DataFrame({
        'criminal_search': [True, False, False],
        'criminal_sentence': pd.array([True, None, False], dtype = 'boolean'),
        'criminal_count': [0, 3, 0],
        'court': ['a', 'b', 'c']})
    columns = ['criminal_search', 'criminal_sentence', 'criminal_count']
    dataset = amos.Dataset(data)
    for how, expected in (
        ('any', [True, True, False]),
        ('all', [False, False, False]),
        ('count', [2, 1, 0])):
        mungers.CombineFlags().apply(
            dataset, columns = columns, name = how, how = how)
        assert dataset.data[how].tolist() == expected
    with pytest.raises(TypeError, match = 'clean_booleans'):
        mungers.CombineFlags().apply(
            dataset, columns = ['court'], name = 'x')
    with pytest.raises(ValueError, match = 'how must be'):
        mungers.CombineFlags().apply(
            dataset, columns = columns, name = 'x', how = 'most')
    with pytest.raises(ValueError, match = 'needs a name'):
        mungers.CombineFlags().apply(dataset, columns = columns)


def test_derive_columns(mixed: amos.Dataset) -> None:
    mungers.DeriveColumns().apply(
        mixed,
        expressions = {
            'income_k': 'income / 1000',
            'rich': 'income_k > 30',
            'older': '`age` >= 50'})
    assert np.allclose(mixed.data['income_k'] * 1000, mixed.data['income'])
    assert mixed.data['rich'].dtype == bool
    assert mixed.booleans == ['member', 'rich', 'older']
    with pytest.raises(ValueError, match = 'assignment'):
        mungers.DeriveColumns().apply(
            mixed, expressions = {'x': 'y = income * 2'})


def test_derive_columns_by_parameter_with_bare_values(
    mixed: amos.Dataset) -> None:
    mungers.DeriveColumns().apply(
        mixed,
        north = 'region == north',
        north_member = 'north and member',
        wealthy = "region in ['east', 'west'] and income > 30000")
    data = mixed.data
    assert (data['north'] == (data['region'] == 'north')).all()
    assert (data['north_member'] == (data['north'] & data['member'])).all()
    assert data['wealthy'].dtype == bool


def test_set_kinds() -> None:
    data = pd.DataFrame({
        'flag': [True, False, True, True],
        'count': [0, 1, 1, 0],
        'age': [17.0, 30.0, 45.0, 120.0],
        'big': [1, 2, 3, 40_000],
        'share': [0.5, 0.25, np.nan, 1.0],
        'precise': [0.1, 1 / 3, 2.0, 1e39],
        'code': [1, 2, 1, 3],
        'race': ['W', 'B', 'W', 'A'],
        'note': ['a', 'b', 'c', 'd'],
        'won': [True, False, False, True]})
    dataset = amos.Dataset(data, label = 'won', groups = ['note'])
    mungers.SetKinds().apply(
        dataset,
        dummy = 'count',
        categorical = ['code', 'race', 'race'],
        numerical = 'flag')
    types = dataset.data.dtypes
    assert types['count'] == 'int8'
    assert types['flag'] == 'int8'
    assert types['age'] == 'int8'
    assert types['big'] == 'int32'
    assert types['share'] == 'float32'
    assert types['precise'] == 'float64'
    assert isinstance(types['code'], pd.CategoricalDtype)
    assert isinstance(types['race'], pd.CategoricalDtype)
    assert types['won'] == 'int8'
    assert types['note'] == object or pd.api.types.is_string_dtype(
        types['note'])
    assert dataset.kinds['count'] == 'booleans'
    assert dataset.kinds['flag'] == 'numerics'
    assert dataset.kinds['won'] == 'booleans'
    assert 'note' not in dataset.kinds
    assert dataset.booleans == ['count']
    assert dataset.numerics == ['flag', 'age', 'big', 'share', 'precise']
    assert dataset.classes == [0, 1]


def test_set_kinds_checks_dummies_and_kinds() -> None:
    data = pd.DataFrame({
        'gap': [1.0, np.nan, 0.0],
        'many': [0, 1, 2],
        'text': ['1', '2', 'x']})
    with pytest.raises(ValueError, match = 'missing values'):
        mungers.SetKinds().apply(data.copy(), dummy = 'gap')
    with pytest.raises(ValueError, match = 'other than 0 and 1'):
        mungers.SetKinds().apply(data.copy(), dummy = 'many')
    with pytest.raises(ValueError, match = 'not numbers'):
        mungers.SetKinds().apply(data.copy(), numerical = 'text')
    with pytest.raises(ValueError, match = 'two kinds'):
        mungers.SetKinds().apply(
            data.copy(), dummy = 'many', numerical = 'many')
    with pytest.raises(ValueError, match = 'does not know'):
        mungers.SetKinds().apply(data.copy(), text = 'text')
    with pytest.raises(KeyError, match = 'not in the data'):
        mungers.SetKinds().apply(data.copy(), dummy = 'missing')


def test_map_values_subclasses_map_their_columns() -> None:
    data = pd.DataFrame({
        'offender_race': ['W', 'B', None],
        'victim_race': pd.Categorical(['B', 'B', 'W']),
        'state': ['ny', 'ca', 'tx'],
        'city': ['a', 'b', 'c']})
    races = {'W': 'white', 'B': 'black'}

    @dataclasses.dataclass
    class MapCodes(mungers.MapValues):
        value_maps: ClassVar[dict[str, Any]] = {
            'offender_race': races, 'victim_race': races,
            'state': str.upper, 'county': str.upper}

    @dataclasses.dataclass
    class MapBadCodes(mungers.MapValues):
        value_maps: ClassVar[dict[str, Any]] = {'state': 5}

    try:
        assert amos.library.classify('map_codes') == 'munger'
        for columns in (['offender_race', 'victim_race', 'state'], None):
            changed = MapCodes().apply(data.copy(), columns = columns).data
            assert changed['offender_race'].tolist()[:2] == ['white', 'black']
            assert pd.isna(changed['offender_race'].iloc[2])
            assert isinstance(
                changed['victim_race'].dtype, pd.CategoricalDtype)
            assert changed['victim_race'].tolist() == [
                'black', 'black', 'white']
            assert changed['state'].tolist() == ['NY', 'CA', 'TX']
            assert changed['city'].tolist() == ['a', 'b', 'c']
        with pytest.raises(TypeError, match = 'a dict or a function'):
            MapBadCodes().apply(data.copy())
        with pytest.raises(KeyError, match = 'value_maps'):
            MapCodes().apply(data.copy(), columns = 'city')
    finally:
        amos.library.delete('map_codes')
        amos.library.delete('map_bad_codes')
    with pytest.raises(KeyError, match = 'value_maps'):
        mungers.MapValues().apply(data, columns = 'state')
    unchanged = mungers.MapValues().apply(data.copy()).data
    pd.testing.assert_frame_equal(unchanged, data)


def test_mungers_check_their_columns(cases: amos.Dataset) -> None:
    with pytest.raises(KeyError, match = 'not in the data'):
        mungers.NormalizeText().apply(cases, columns = ['missing'])
    cases.data['number'] = range(4)
    with pytest.raises(TypeError, match = 'is not text'):
        mungers.NormalizeText().apply(cases, columns = ['number'])


def test_mungers_change_nothing_without_their_parameters(
    cases: amos.Dataset) -> None:
    before = cases.data.copy()
    for kind in (
        mungers.Coalesce, mungers.CombineFlags, mungers.DeriveColumns,
        mungers.MapValues, mungers.ConvertTypes):
        kind().apply(cases)
        assert cases.history[-1]['changed'] == [], kind
    pd.testing.assert_frame_equal(cases.data, before)


def test_mungers_keep_rows_with_repeated_labels(cases: amos.Dataset) -> None:
    cases.data.index = [0, 0, 1, 1]
    mungers.MapValues().apply(cases, values = {'n/a': None})
    mungers.NormalizeText().apply(
        cases, columns = ['caption'], case = 'lower')
    assert pd.isna(cases.data['damages'].tolist()[1])
    assert cases.data['caption'].tolist()[:2] == [
        'united states v. smith', 'jones v. acme corp.']
    assert list(cases.data.index) == [0, 0, 1, 1]


def test_mungers_work_in_a_project(cases: amos.Dataset) -> None:
    settings = {
        'general': {'label': 'won'},
        'cases_project': {'cases_workers': 'wrangler'},
        'wrangler': {
            'techniques': 'strip_text, clean_numbers, auto_categorize'},
        'strip_text_parameters': {'lowercase': True},
        'clean_numbers_parameters': {'columns': 'damages'},
        'auto_categorize_parameters': {'columns': ['court']}}
    project = amos.Project.create(settings, item = cases.data)
    result = project.result
    assert result.data['disposition'].tolist()[0] == 'reversed and remanded'
    assert 'damages' in result.numerics
    assert [h['technique'] for h in result.history] == [
        'strip_text', 'clean_numbers', 'auto_categorize']
    assert amos.library.classify('strip_text') == 'munger'
