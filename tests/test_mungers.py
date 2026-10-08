"""Tests the mungers module."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

import amos
from amos import mungers


@pytest.fixture
def cases() -> amos.Dataset:
    """A small dataset of court cases, with text to munge."""
    data = pd.DataFrame({
        'caption': [
            'United States v. Smith', 'Jones  v.  Acme Corp.',
            'In re Doe', None],
        'disposition': [
            'REVERSED and remanded', 'Affirmed.', 'vacated in part', None],
        'court': [
            'First Circuit', 'second circuit', 'Ninth Circuit', 'Ninth'],
        'damages': ['$1,200', 'n/a', '$350.50', '15,000'],
        'decided': ['2020-01-05', '2021-03-04', 'unknown', None],
        'appealed': ['Yes', ' no', 'maybe', None],
        'won': ['yes', 'no', 'yes', 'no']})
    return amos.Dataset(data.astype('str'), label = 'won')


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
    mungers.FlagPatterns().apply(
        cases, column = 'disposition', patterns = {'reversed': 'revers'})
    assert cases.history[-1] == {
        'technique': 'flag_patterns', 'changed': [], 'created': ['reversed']}


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


def test_flag_patterns(cases: amos.Dataset) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mungers.FlagPatterns().apply(
            cases,
            column = 'disposition',
            patterns = {
                'reversed': ['revers', 'vacat'],
                'remanded': '(remand)'},
            ignorecase = True)
    assert cases.data['reversed'].tolist() == [True, False, True, False]
    assert cases.data['remanded'].tolist() == [True, False, False, False]
    assert cases.booleans == ['reversed', 'remanded']
    mungers.FlagPatterns().apply(
        cases, column = 'disposition', patterns = {'upper': 'REVERSED'})
    assert cases.data['upper'].tolist() == [True, False, False, False]


def test_flag_patterns_searches_categories(cases: amos.Dataset) -> None:
    mungers.AutoCategorize().apply(cases, columns = ['court'])
    mungers.FlagPatterns().apply(
        cases, column = 'court', patterns = {'ninth': 'Ninth'})
    assert cases.data['ninth'].tolist() == [False, False, True, True]


def test_count_patterns(cases: amos.Dataset) -> None:
    mungers.CountPatterns().apply(
        cases,
        column = 'caption',
        patterns = {'words': r'\S+', 'versus': [r'\bv\.', r'\bin re\b']},
        ignorecase = True)
    assert cases.data['words'].tolist() == [4, 4, 3, 0]
    assert cases.data['versus'].tolist() == [1, 1, 1, 0]
    assert cases.data['words'].dtype == 'int64'


def test_map_patterns(cases: amos.Dataset) -> None:
    mungers.MapPatterns().apply(
        cases,
        column = 'court',
        patterns = {'first': 1, 'second': 2, 'ninth circuit': 9},
        name = 'circuit',
        ignorecase = True)
    assert cases.data['circuit'].tolist()[:3] == [1, 2, 9]
    assert pd.isna(cases.data['circuit'].iloc[3])
    mungers.MapPatterns().apply(
        cases,
        column = 'disposition',
        patterns = {'revers|vacat': 'reversed', '.': 'other'},
        default = 'unknown',
        ignorecase = True)
    assert cases.data['disposition'].tolist() == [
        'reversed', 'other', 'reversed', 'unknown']


def test_extract_pattern(cases: amos.Dataset) -> None:
    mungers.ExtractPattern().apply(
        cases, column = 'caption', pattern = r'(?i)\bv\.', name = 'versus')
    assert cases.data['versus'].tolist()[:2] == ['v.', 'v.']
    assert cases.data['versus'].isna().tolist() == [False, False, True, True]
    mungers.ExtractPattern().apply(
        cases, column = 'decided', pattern = r'(\d{4})-', name = 'year')
    assert cases.data['year'].tolist()[:2] == ['2020', '2021']
    mungers.ExtractPattern().apply(
        cases,
        column = 'decided',
        pattern = r'(?P<month>\d{2})-(?P<day>\d{2})$')
    assert cases.data['month'].tolist()[:2] == ['01', '03']
    assert cases.data['day'].tolist()[:2] == ['05', '04']
    with pytest.raises(ValueError, match = 'need names'):
        mungers.ExtractPattern().apply(
            cases, column = 'decided', pattern = r'(\d+)-(\d+)')


def test_extract_all(cases: amos.Dataset) -> None:
    mungers.ExtractAll().apply(
        cases, column = 'caption', pattern = r'\b[A-Z]\w+', name = 'names')
    assert cases.data['names'].tolist()[:3] == [
        'United, States, Smith', 'Jones, Acme, Corp', 'In, Doe']
    assert pd.isna(cases.data['names'].iloc[3])
    mungers.ExtractAll().apply(
        cases, column = 'damages', pattern = r'(\d+)', separator = '')
    assert cases.data['damages'].iloc[0] == '1200'
    assert pd.isna(cases.data['damages'].iloc[1])
    with pytest.raises(ValueError, match = 'keep one'):
        mungers.ExtractAll().apply(
            cases, column = 'caption', pattern = r'(\w)(\w)')


def test_split_text(cases: amos.Dataset) -> None:
    mungers.SplitText().apply(
        cases,
        column = 'caption',
        pattern = r'\s+v\.\s+',
        names = ['party1', 'party2'])
    assert cases.data['party1'].tolist()[:3] == [
        'United States', 'Jones', 'In re Doe']
    assert cases.data['party2'].tolist()[:2] == ['Smith', 'Acme Corp.']
    assert cases.data['party2'].isna().tolist()[2:] == [True, True]
    assert 'caption' in cases.data.columns
    mungers.SplitText().apply(
        cases, column = 'court', pattern = ';', names = 'a, b'.split(', '))
    assert cases.data['b'].isna().all()
    with pytest.raises(ValueError, match = 'has groups'):
        mungers.SplitText().apply(
            cases, column = 'caption', pattern = '(v)', names = ['a', 'b'])
    with pytest.raises(ValueError, match = 'two or more'):
        mungers.SplitText().apply(
            cases, column = 'caption', pattern = 'v', names = 'a')


def test_replace_text(cases: amos.Dataset) -> None:
    mungers.ReplaceText().apply(
        cases,
        columns = ['caption'],
        patterns = {r'\s+': ' ', r'(\w+) v\. (\w+)': r'\2 v. \1'})
    assert cases.data['caption'].tolist()[:2] == [
        'United Smith v. States', 'Acme v. Jones Corp.']
    mungers.ReplaceText().apply(
        cases, patterns = {'[$,]': None, 'circuit': 'Cir.'}, ignorecase = True)
    assert cases.data['damages'].tolist()[0] == '1200'
    assert cases.data['court'].tolist()[0] == 'First Cir.'


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


def test_parse_numbers(cases: amos.Dataset) -> None:
    cases.data['notes'] = pd.Series(
        ['12%', 'about 3 years', '-1.5e3', '.5'], dtype = 'str')
    mungers.ParseNumbers().apply(cases, columns = ['damages', 'notes'])
    assert cases.data['damages'].tolist()[0::2] == [1200.0, 350.5]
    assert cases.data['damages'].tolist()[3] == 15000.0
    assert pd.isna(cases.data['damages'].iloc[1])
    assert cases.data['notes'].tolist() == [12.0, 3.0, -1500.0, 0.5]
    assert 'damages' in cases.numerics
    data = pd.DataFrame({'amount': ['1.234,50', '7'], 'count': [1, 2]})
    result = mungers.ParseNumbers().apply(
        data, columns = ['amount', 'count'], thousands = '.', decimal = ',')
    assert result.data['amount'].tolist() == [1234.5, 7.0]
    assert result.data['count'].dtype == 'int64'
    with pytest.raises(ValueError, match = 'must differ'):
        mungers.ParseNumbers().apply(data, thousands = ',', decimal = ',')


def test_parse_booleans(cases: amos.Dataset) -> None:
    cases.data['flag'] = [1, 0, 2, np.nan]
    mungers.ParseBooleans().apply(cases, columns = ['appealed', 'flag'])
    assert cases.data['appealed'].dtype == 'boolean'
    assert cases.data['appealed'].tolist()[:2] == [True, False]
    assert cases.data['appealed'].isna().tolist() == [
        False, False, True, True]
    assert cases.data['flag'].tolist()[:2] == [True, False]
    assert cases.data['flag'].isna().sum() == 2
    assert 'appealed' in cases.booleans
    mungers.ParseBooleans().apply(
        cases, columns = 'disposition', true_values = 'affirmed.',
        false_values = ['vacated in part'])
    assert cases.data['disposition'].tolist()[1:3] == [True, False]


def test_parse_dates(cases: amos.Dataset) -> None:
    mungers.ParseDates().apply(cases, columns = ['decided'])
    assert cases.data['decided'].tolist()[:2] == [
        pd.Timestamp('2020-01-05'), pd.Timestamp('2021-03-04')]
    assert cases.data['decided'].isna().tolist() == [
        False, False, True, True]
    assert cases.dates == ['decided']
    data = pd.DataFrame({'when': ['January 5, 2010', '2020-03-04']})
    result = mungers.ParseDates().apply(
        data, columns = 'when', date_format = 'mixed')
    assert result.data['when'].dt.year.tolist() == [2010, 2020]


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
    with pytest.raises(TypeError, match = 'parse_booleans'):
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


def test_mungers_check_their_columns_and_patterns(
    cases: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'is not valid'):
        mungers.FlagPatterns().apply(
            cases, column = 'caption', patterns = {'x': '(unclosed'})
    with pytest.raises(ValueError, match = 'needs the name of a column'):
        mungers.FlagPatterns().apply(cases, patterns = {'x': 'v'})
    with pytest.raises(KeyError, match = 'not in the data'):
        mungers.CountPatterns().apply(
            cases, column = 'missing', patterns = {'x': 'v'})
    cases.data['number'] = range(4)
    with pytest.raises(TypeError, match = 'is not text'):
        mungers.ExtractPattern().apply(
            cases, column = 'number', pattern = '1')


def test_mungers_change_nothing_without_their_parameters(
    cases: amos.Dataset) -> None:
    before = cases.data.copy()
    for kind in (
        mungers.Coalesce, mungers.CombineFlags, mungers.CountPatterns,
        mungers.DeriveColumns, mungers.ExtractAll, mungers.ExtractPattern,
        mungers.FlagPatterns, mungers.MapPatterns, mungers.MapValues,
        mungers.ParseBooleans, mungers.ParseDates, mungers.ParseNumbers,
        mungers.ReplaceText, mungers.SplitText, mungers.ConvertTypes):
        kind().apply(cases)
        assert cases.history[-1]['changed'] == [], kind
    pd.testing.assert_frame_equal(cases.data, before)


def test_mungers_keep_rows_with_repeated_labels(cases: amos.Dataset) -> None:
    cases.data.index = [0, 0, 1, 1]
    mungers.FlagPatterns().apply(
        cases, column = 'disposition', patterns = {'reversed': 'REVERSED'})
    mungers.SplitText().apply(
        cases, column = 'caption', pattern = r' v\. ', names = ['p1', 'p2'])
    mungers.MapValues().apply(cases, values = {'n/a': None})
    mungers.ExtractPattern().apply(
        cases, column = 'decided', pattern = r'\d{4}', name = 'year')
    assert cases.data['reversed'].tolist() == [True, False, False, False]
    assert cases.data['p2'].tolist()[:2] == ['Smith', 'Acme Corp.']
    assert cases.data['year'].tolist()[:2] == ['2020', '2021']
    assert list(cases.data.index) == [0, 0, 1, 1]


def test_mungers_work_in_a_project(cases: amos.Dataset) -> None:
    settings = {
        'general': {'label': 'won'},
        'cases_project': {'cases_workers': 'wrangler'},
        'wrangler': {
            'techniques': 'flag_patterns, parse_numbers, auto_categorize'},
        'flag_patterns_parameters': {
            'column': 'disposition',
            'patterns': {'reversed': 'revers|vacat'},
            'ignorecase': True},
        'parse_numbers_parameters': {'columns': 'damages'},
        'auto_categorize_parameters': {'columns': ['court']}}
    project = amos.Project.create(settings, item = cases.data)
    result = project.result
    assert result.data['reversed'].tolist() == [True, False, True, False]
    assert 'damages' in result.numerics
    assert [h['technique'] for h in result.history] == [
        'flag_patterns', 'parse_numbers', 'auto_categorize']
    assert amos.library.classify('flag_patterns') == 'munger'
