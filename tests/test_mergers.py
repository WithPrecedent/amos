"""Tests the mergers module."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import SEED

import amos
import nagata
from amos import mergers


@pytest.fixture(autouse = True)
def _work_in(tmp_path: pathlib.Path) -> None:
    """Runs each test in its own folder."""
    os.chdir(tmp_path)


@pytest.fixture
def cases() -> amos.Dataset:
    """A small dataset with a row for each judge on each court case."""
    data = pd.DataFrame({
        'case': ['a1', 'a1', 'a2', 'a2', 'a3', 'a4'],
        'court': [
            'First Circuit', 'First Circuit', 'Ninth Circuit',
            'Ninth Circuit', 'first circuit ', None],
        'year': [2001, 2001, 2015, 2015, 2019, 2010],
        'judge': ['Lynch', 'Selya', 'Kozinski', 'Lynch', 'LYNCH', 'Doe'],
        'reversed': [True, True, False, False, True, False]})
    return amos.Dataset(data, label = 'reversed', seed = SEED)


@pytest.fixture
def courts() -> pd.DataFrame:
    """A table with a row for each court."""
    return pd.DataFrame({
        'court': ['First Circuit', 'Ninth Circuit', 'Second Circuit'],
        'number': [1, 9, 2],
        'western': [False, True, False]})


@pytest.fixture
def judges() -> pd.DataFrame:
    """A table with a row for each court that each judge has served on."""
    return pd.DataFrame({
        'name': ['Lynch', 'Selya', 'Kozinski', 'Lynch', 'Lynch'],
        'bench': [
            'First Circuit', 'First Circuit', 'Ninth Circuit',
            'Ninth Circuit', 'Ninth Circuit'],
        'began': [1995, 1986, 1985, 1990, 2012],
        'ended': [np.nan, 2021, 2017, 2000, np.nan],
        'party': [-1, 1, 1, 1, -1],
        'woman': [True, False, False, False, True]})


def test_merge_keys_adds_the_columns_of_the_matching_row(
    cases: amos.Dataset,
    courts: pd.DataFrame) -> None:
    before = cases.data.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mergers.MergeKeys().apply(cases, source = courts, on = 'court')
    data = cases.data
    assert data['number'].tolist()[:4] == [1, 1, 9, 9]
    assert data['number'].isna().tolist() == [
        False, False, False, False, True, True]
    assert data['western'].tolist()[:4] == [False, False, True, True]
    # The rows, their order, and the columns that were there are not changed.
    pd.testing.assert_frame_equal(data[before.columns], before)
    assert (cases.label, cases.task) == ('reversed', 'classify')
    assert cases.history[-1] == {
        'technique': 'merge_keys',
        'source': 'DataFrame',
        'rows': 6,
        'matched': 4,
        'created': ['number', 'western']}


def test_merge_keys_with_several_keys_of_other_names(
    cases: amos.Dataset,
    judges: pd.DataFrame) -> None:
    served = judges.drop(index = [3, 4])
    mergers.MergeKeys().apply(
        cases,
        source = served,
        on = ['judge', 'court'],
        other_on = ['name', 'bench'])
    assert cases.data['party'].tolist()[:3] == [-1, 1, 1]
    assert cases.data['party'].isna().tolist()[3:] == [True, True, True]
    # The keys of the other table repeat those of the data, so they are not
    # added.
    assert cases.history[-1]['created'] == [
        'began', 'ended', 'party', 'woman']
    with pytest.raises(ValueError, match = 'as many keys'):
        mergers.MergeKeys().apply(
            cases, source = served, on = ['judge', 'court'], other_on = 'name')
    with pytest.raises(ValueError, match = 'needs the key columns'):
        mergers.MergeKeys().apply(cases, source = served)
    with pytest.raises(KeyError, match = 'not in the data'):
        mergers.MergeKeys().apply(cases, source = served, on = 'name')
    with pytest.raises(KeyError, match = 'not in the other table'):
        mergers.MergeKeys().apply(cases, source = served, on = 'judge')


def test_merge_keys_can_ignore_case_and_spaces(
    cases: amos.Dataset,
    courts: pd.DataFrame) -> None:
    mergers.MergeKeys().apply(
        cases, source = courts, on = 'court', ignorecase = True)
    assert cases.data['number'].tolist()[:5] == [1, 1, 9, 9, 1]
    assert cases.history[-1]['matched'] == 5
    # The keys of the data are not changed.
    assert cases.data['court'].tolist()[4] == 'first circuit '


def test_missing_keys_match_nothing(cases: amos.Dataset) -> None:
    other = pd.DataFrame({
        'court': pd.Series([None, 'First Circuit'], dtype = 'str'),
        'number': [0, 1]})
    mergers.MergeKeys().apply(cases, source = other, on = 'court')
    assert cases.data['number'].tolist()[:2] == [1, 1]
    assert pd.isna(cases.data['number'].iloc[5])


def test_merge_keys_refuses_repeated_keys_unless_told_which_to_use(
    cases: amos.Dataset,
    judges: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match = 'merge_summary'):
        mergers.MergeKeys().apply(
            cases, source = judges, on = 'judge', other_on = 'name')
    first = mergers.MergeKeys().apply(
        cases.data.copy(),
        source = judges,
        on = 'judge',
        other_on = 'name',
        duplicates = 'first',
        columns = 'began')
    assert first.data['began'].tolist()[:4] == [1995, 1986, 1985, 1995]
    last = mergers.MergeKeys().apply(
        cases.data.copy(),
        source = judges,
        on = 'judge',
        other_on = 'name',
        duplicates = 'last',
        columns = 'began')
    assert last.data['began'].tolist()[:4] == [2012, 1986, 1985, 2012]
    with pytest.raises(ValueError, match = 'duplicates must be one of'):
        mergers.MergeKeys().apply(
            cases, source = judges, on = 'judge', other_on = 'name',
            duplicates = 'all')


def test_keys_that_are_numbers_of_different_types_match() -> None:
    data = pd.DataFrame({'number': [1, 9, 3], 'big': [2**60 + 1, 2**60, 5]})
    other = pd.DataFrame({
        'number': pd.array([1.0, 9.0, 2.5]),
        'nullable': pd.array([1, 9, None], dtype = 'Int64'),
        'big': [2**60 + 1, 7, 5],
        'name': ['first', 'ninth', 'none']})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        floats = mergers.MergeKeys().apply(
            data.copy(), source = other, on = 'number', columns = 'name')
        nullable = mergers.MergeKeys().apply(
            data.copy(), source = other, on = 'number',
            other_on = 'nullable', columns = 'name')
        # Whole numbers of the same type are not made into floats, which
        # could not tell these two apart.
        exact = mergers.MergeKeys().apply(
            data.copy(), source = other, on = 'big', columns = 'name')
    assert floats.data['name'].tolist()[:2] == ['first', 'ninth']
    assert nullable.data['name'].tolist()[:2] == ['first', 'ninth']
    assert pd.isna(floats.data['name'].iloc[2])
    assert exact.data['name'].tolist()[0] == 'first'
    assert pd.isna(exact.data['name'].iloc[1])


def test_keys_of_different_kinds_are_explained(
    cases: amos.Dataset,
    courts: pd.DataFrame) -> None:
    courts['year'] = ['2001', '2015', '2019']
    with pytest.raises(TypeError, match = 'numbers in the data.*parse_numbers'):
        mergers.MergeKeys().apply(cases, source = courts, on = 'year')


def test_categorical_keys_match_text(
    cases: amos.Dataset,
    courts: pd.DataFrame) -> None:
    cases.data['court'] = cases.data['court'].astype('category')
    courts['court'] = courts['court'].astype('category')
    mergers.MergeKeys().apply(cases, source = courts, on = 'court')
    assert cases.data['number'].tolist()[:4] == [1, 1, 9, 9]
    assert isinstance(cases.data['court'].dtype, pd.CategoricalDtype)


def test_columns_prefix_and_indicator(
    cases: amos.Dataset,
    courts: pd.DataFrame) -> None:
    mergers.MergeKeys().apply(
        cases,
        source = courts,
        on = 'court',
        columns = ['western', 'court'],
        prefix = 'court_',
        indicator = 'found')
    assert cases.history[-1]['created'] == [
        'court_western', 'court_court', 'found']
    assert cases.data['found'].tolist() == [
        True, True, True, True, False, False]
    assert cases.data['found'].dtype == bool
    assert cases.data['court_court'].tolist()[:3] == [
        'First Circuit', 'First Circuit', 'Ninth Circuit']
    with pytest.raises(KeyError, match = 'not in the other table'):
        mergers.MergeKeys().apply(
            cases, source = courts, on = 'court', columns = 'judges')
    with pytest.raises(ValueError, match = 'set "prefix"'):
        mergers.MergeKeys().apply(
            cases, source = courts, on = 'court', columns = 'court')
    with pytest.raises(ValueError, match = 'set "prefix"'):
        mergers.MergeKeys().apply(
            cases, source = courts, on = 'court', indicator = 'year')


def test_columns_that_the_data_has_are_not_added_unless_named(
    cases: amos.Dataset,
    courts: pd.DataFrame) -> None:
    courts['year'] = [1891, 1891, 1891]
    mergers.MergeKeys().apply(cases, source = courts, on = 'court')
    assert cases.history[-1]['created'] == ['number', 'western']
    assert cases.data['year'].tolist()[:2] == [2001, 2001]
    mergers.MergeKeys().apply(
        cases, source = courts, on = 'court', columns = 'year',
        prefix = 'founded_')
    assert cases.data['founded_year'].tolist()[:2] == [1891, 1891]


def test_types_only_change_to_hold_missing_values(
    cases: amos.Dataset,
    courts: pd.DataFrame) -> None:
    matched = amos.Dataset(cases.data.iloc[:4].copy())
    mergers.MergeKeys().apply(matched, source = courts, on = 'court')
    assert matched.data['number'].dtype == 'int64'
    assert matched.data['western'].dtype == bool
    assert matched.booleans == ['reversed', 'western']
    mergers.MergeKeys().apply(cases, source = courts, on = 'court')
    assert cases.data['number'].dtype == 'float64'
    assert cases.data['western'].dtype == 'boolean'
    assert cases.booleans == ['western']
    assert cases.data['western'].isna().tolist()[4:] == [True, True]


def test_merge_nearest_looks_before_after_or_both() -> None:
    data = pd.DataFrame({'day': [1.0, 5.0, 10.0, np.nan, 20.0]})
    other = pd.DataFrame({'day': [0, 4, 12], 'rating': ['a', 'b', 'c']})
    found = {}
    for direction in ('before', 'after', 'nearest'):
        dataset = mergers.MergeNearest().apply(
            data.copy(), source = other, column = 'day',
            direction = direction)
        found[direction] = dataset.data['rating'].tolist()
    assert found['before'][:3] == ['a', 'b', 'b']
    assert found['before'][4] == 'c'
    assert found['after'][:3] == ['b', 'c', 'c']
    assert pd.isna(found['after'][4])
    assert found['nearest'][:3] == ['a', 'b', 'c']
    assert all(pd.isna(values[3]) for values in found.values())
    near = mergers.MergeNearest().apply(
        data.copy(), source = other, column = 'day', tolerance = 1)
    assert near.data['rating'].tolist()[:2] == ['a', 'b']
    assert near.data['rating'].isna().tolist()[2:] == [True, True, True]
    assert near.history[-1]['matched'] == 2
    with pytest.raises(ValueError, match = 'direction must be one of'):
        mergers.MergeNearest().apply(
            data, source = other, column = 'day', direction = 'backward')
    with pytest.raises(ValueError, match = 'needs a "column"'):
        mergers.MergeNearest().apply(data, source = other)


def test_merge_nearest_matches_dates_within_keys(
    cases: amos.Dataset) -> None:
    cases.data['decided'] = pd.to_datetime([
        '2001-05-05', '2001-05-05', '2015-01-01', '2015-01-01',
        '2019-01-15', '2010-01-01'])
    ratings = pd.DataFrame({
        'court': ['First Circuit'] * 3 + ['Ninth Circuit'],
        'rated': pd.to_datetime([
            '2000-01-01', '2019-01-01', '2001-05-05',
            '2014-12-01']).astype('datetime64[s]'),
        'rating': [3, 5, 4, 2]})
    mergers.MergeNearest().apply(
        cases,
        source = ratings,
        column = 'decided',
        other_column = 'rated',
        on = 'court',
        ignorecase = True)
    assert cases.data['rating'].tolist()[:5] == [4, 4, 2, 2, 5]
    assert pd.isna(cases.data['rating'].iloc[5])
    # The date of the row that matched is added too.
    assert cases.data['rated'].iloc[0] == pd.Timestamp('2001-05-05')
    recent = mergers.MergeNearest().apply(
        cases.data.drop(columns = ['rating', 'rated']),
        source = ratings,
        column = 'decided',
        other_column = 'rated',
        on = 'court',
        tolerance = 31)
    assert recent.data['rating'].isna().tolist() == [
        False, False, False, False, True, True]
    hours = mergers.MergeNearest().apply(
        cases.data.drop(columns = ['rating', 'rated']),
        source = ratings,
        column = 'decided',
        other_column = 'rated',
        tolerance = '12h')
    assert hours.history[-1]['matched'] == 2
    with pytest.raises(TypeError, match = 'dates in the data'):
        mergers.MergeNearest().apply(
            cases, source = ratings, column = 'decided',
            other_column = 'rating')
    with pytest.raises(TypeError, match = 'text in the data'):
        mergers.MergeNearest().apply(
            cases, source = ratings, column = 'court',
            other_column = 'rated')


def test_merge_ranges_finds_the_range_within_keys(
    cases: amos.Dataset,
    judges: pd.DataFrame) -> None:
    mergers.MergeRanges().apply(
        cases,
        source = judges,
        column = 'year',
        start = 'began',
        end = 'ended',
        on = ['judge', 'court'],
        other_on = ['name', 'bench'],
        ignorecase = True,
        columns = ['party', 'began'],
        prefix = 'judge_')
    # Judge Lynch of the Ninth Circuit served twice, and the case of 2015
    # was during the second term, which has no end.
    assert cases.data['judge_began'].tolist()[:5] == [
        1995, 1986, 1985, 2012, 1995]
    assert cases.data['judge_party'].tolist()[:5] == [-1, 1, 1, -1, -1]
    assert pd.isna(cases.data['judge_party'].iloc[5])
    assert cases.history[-1]['matched'] == 5


def test_merge_ranges_without_keys_or_one_of_the_limits() -> None:
    data = pd.DataFrame({'year': [1975, 1977, 1981, 2030, np.nan]})
    terms = pd.DataFrame({
        'president': ['Ford', 'Carter', 'Reagan', 'Nobody'],
        'began': [1974, 1977, 1981, np.nan],
        'ended': [1976, 1980, 1988, np.nan]})
    both = mergers.MergeRanges().apply(
        data.copy(), source = terms, column = 'year', start = 'began',
        end = 'ended')
    assert both.data['president'].tolist()[:3] == ['Ford', 'Carter', 'Reagan']
    # A range with neither a start nor an end matches nothing.
    assert both.data['president'].isna().tolist()[3:] == [True, True]
    assert both.history[-1]['created'] == ['president', 'began', 'ended']
    with pytest.raises(ValueError, match = 'more than one range'):
        mergers.MergeRanges().apply(
            data.copy(), source = terms, column = 'year', start = 'began')
    latest = mergers.MergeRanges().apply(
        data.copy(), source = terms, column = 'year', start = 'began',
        duplicates = 'last')
    assert latest.data['president'].tolist()[:4] == [
        'Ford', 'Carter', 'Reagan', 'Reagan']
    earliest = mergers.MergeRanges().apply(
        data.copy(), source = terms, column = 'year', end = 'ended',
        duplicates = 'first')
    assert earliest.data['president'].tolist()[:3] == [
        'Ford', 'Carter', 'Reagan']
    with pytest.raises(ValueError, match = '"start" or "end"'):
        mergers.MergeRanges().apply(data, source = terms, column = 'year')
    with pytest.raises(TypeError, match = 'text in the other table'):
        mergers.MergeRanges().apply(
            data, source = terms, column = 'year', start = 'president')


def test_merge_ranges_gives_the_same_matches_a_part_at_a_time(
    cases: amos.Dataset,
    judges: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch) -> None:
    parameters = {
        'source': judges, 'column': 'year', 'start': 'began',
        'end': 'ended', 'duplicates': 'last', 'columns': 'party'}
    whole = mergers.MergeRanges().apply(cases.data.copy(), **parameters)
    keyed = mergers.MergeRanges().apply(
        cases.data.copy(), on = 'judge', other_on = 'name', **parameters)
    monkeypatch.setattr(mergers, '_PAIRS', 3)
    parts = mergers.MergeRanges().apply(cases.data.copy(), **parameters)
    keyed_parts = mergers.MergeRanges().apply(
        cases.data.copy(), on = 'judge', other_on = 'name', **parameters)
    pd.testing.assert_frame_equal(parts.data, whole.data)
    pd.testing.assert_frame_equal(keyed_parts.data, keyed.data)
    assert whole.history[-1]['matched'] == 6


def test_merge_summary_adds_summaries_of_the_matching_rows(
    courts: pd.DataFrame,
    judges: pd.DataFrame) -> None:
    dataset = mergers.MergeSummary().apply(
        courts.copy(),
        source = judges,
        on = 'court',
        other_on = 'bench',
        count = 'judges',
        prefix = 'bench_')
    data = dataset.data
    assert data['bench_judges'].tolist()[:2] == [2, 3]
    assert data['bench_party'].tolist()[:2] == [0, 1 / 3]
    # The mean of a boolean is the share that are true.
    assert data['bench_woman'].tolist()[:2] == [0.5, 1 / 3]
    assert data['bench_began'].tolist()[:2] == [1990.5, 1995 + 2 / 3]
    # A court with no judges in the other table has missing summaries.
    assert data.iloc[2, 3:].isna().all()
    assert dataset.history[-1] == {
        'technique': 'merge_summary',
        'source': 'DataFrame',
        'rows': 3,
        'matched': 2,
        'created': [
            'bench_began', 'bench_ended', 'bench_party', 'bench_woman',
            'bench_judges']}


def test_merge_summary_takes_the_summaries_it_is_told_to(
    courts: pd.DataFrame,
    judges: pd.DataFrame) -> None:
    judges['bench'] = judges['bench'].str.upper()
    several = mergers.MergeSummary().apply(
        courts.copy(),
        source = judges,
        on = 'court',
        other_on = 'bench',
        ignorecase = True,
        how = ['min', 'max'],
        columns = ['began_min', 'began_max'])
    assert several.data['began_min'].tolist()[:2] == [1986, 1985]
    assert several.data['began_max'].tolist()[:2] == [1995, 2012]
    chosen = mergers.MergeSummary().apply(
        courts.copy(),
        source = judges,
        on = 'court',
        other_on = 'bench',
        ignorecase = True,
        how = {'name': ['list', 'nunique'], 'woman': 'any'})
    assert chosen.data['name_list'].tolist()[:2] == [
        ['Lynch', 'Selya'], ['Kozinski', 'Lynch', 'Lynch']]
    assert chosen.data['name_nunique'].tolist()[:2] == [2, 2]
    assert chosen.data['woman'].tolist()[:2] == [True, True]
    assert chosen.history[-1]['created'] == [
        'name_list', 'name_nunique', 'woman']
    with pytest.raises(KeyError, match = 'not in the other table'):
        mergers.MergeSummary().apply(
            courts, source = judges, on = 'court', other_on = 'bench',
            how = {'age': 'mean'})
    with pytest.raises(ValueError, match = 'summary must be one of'):
        mergers.MergeSummary().apply(
            courts, source = judges, on = 'court', other_on = 'bench',
            how = 'average')


def test_a_merger_needs_a_source(cases: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'nothing to merge'):
        mergers.MergeKeys().apply(cases, on = 'court')


def test_the_source_can_be_a_dataset_or_columns(
    cases: amos.Dataset,
    courts: pd.DataFrame) -> None:
    mergers.MergeKeys().apply(
        cases, source = amos.Dataset(courts), on = 'court',
        columns = 'number')
    assert cases.history[-1]['source'] == 'Dataset'
    mergers.MergeKeys().apply(
        cases, source = courts.to_dict('list'), on = 'court',
        columns = 'western')
    assert cases.history[-1]['source'] == 'dict'
    assert cases.data['western'].tolist()[:4] == [False, False, True, True]


def test_a_merger_loads_a_file_with_the_clerk(
    cases: amos.Dataset,
    courts: pd.DataFrame) -> None:
    clerk = nagata.FileManager(root_folder = '.', input_folder = 'data')
    courts.to_csv('data/courts.data', index = False, sep = ';')
    with pytest.raises(ValueError, match = 'file_format'):
        mergers.MergeKeys(clerk = clerk).apply(
            cases, source = 'courts.data', on = 'court')
    mergers.MergeKeys(clerk = clerk).apply(
        cases,
        source = 'courts.data',
        on = 'court',
        reader = {'file_format': 'csv', 'sep': ';'})
    assert cases.data['number'].tolist()[:4] == [1, 1, 9, 9]
    assert cases.history[-1]['source'] == 'courts.data'
    with pytest.raises(FileNotFoundError, match = 'no file'):
        mergers.MergeKeys(clerk = clerk).apply(
            cases, source = 'missing.csv', on = 'court')


def test_mergers_keep_the_rows_of_split_data(
    cases: amos.Dataset,
    courts: pd.DataFrame) -> None:
    cases.split(train = [0, 1, 2, 3], test = [4, 5])
    mergers.MergeKeys().apply(cases, source = courts, on = 'court')
    assert list(cases.train) == [0, 1, 2, 3]
    assert list(cases.test) == [4, 5]
    assert 'number' in cases.x_train.columns
    assert len(cases.x_test) == 2


def test_mergers_keep_rows_with_repeated_labels(
    cases: amos.Dataset,
    courts: pd.DataFrame,
    judges: pd.DataFrame) -> None:
    cases.data.index = [0, 0, 1, 1, 2, 2]
    mergers.MergeKeys().apply(cases, source = courts, on = 'court')
    mergers.MergeRanges().apply(
        cases, source = judges, column = 'year', start = 'began',
        end = 'ended', on = 'judge', other_on = 'name',
        duplicates = 'first', columns = 'party')
    mergers.MergeNearest().apply(
        cases, source = judges, column = 'year', other_column = 'began',
        columns = 'woman')
    assert list(cases.data.index) == [0, 0, 1, 1, 2, 2]
    assert cases.data['number'].tolist()[:4] == [1, 1, 9, 9]
    assert cases.data['party'].tolist()[:3] == [-1, 1, 1]
    assert cases.data['woman'].tolist()[:2] == [True, True]


def test_a_merger_only_has_to_match_rows(
    cases: amos.Dataset,
    courts: pd.DataFrame) -> None:

    @dataclasses.dataclass
    class MergeStart(amos.Merger):
        """Matches rows whose keys start with the same letters."""

        def match(self, data, other, on = None, letters = 1, **kwargs):
            starts = other[on].str[:letters].tolist()
            return [
                starts.index(key[:letters])
                if isinstance(key, str) and key[:letters] in starts else -1
                for key in data[on]]

        def prepare(self, other, **kwargs):
            return other.assign(circuit = other['court'].str.upper())

    @dataclasses.dataclass
    class MergeBadly(amos.Merger):

        def match(self, data, other, **kwargs):
            return [0, len(other)]

    MergeStart().apply(cases, source = courts, on = 'court')
    assert cases.data['circuit'].tolist()[:4] == [
        'FIRST CIRCUIT', 'FIRST CIRCUIT', 'NINTH CIRCUIT', 'NINTH CIRCUIT']
    assert cases.history[-1] == {
        'technique': 'merge_start',
        'source': 'DataFrame',
        'rows': 6,
        'matched': 4,
        'created': ['number', 'western', 'circuit']}
    assert amos.library.classify('merge_start') == 'merger'
    with pytest.raises(ValueError, match = 'for each of the 6 rows'):
        MergeBadly().apply(cases, source = courts)


def test_mergers_work_in_a_project(
    cases: amos.Dataset,
    courts: pd.DataFrame,
    judges: pd.DataFrame) -> None:
    pathlib.Path('data').mkdir()
    cases.data.to_csv('data/cases.csv', index = False)
    courts.to_csv('data/courts.csv', index = False)
    settings = {
        'general': {'label': 'reversed', 'seed': SEED},
        'files': {'input_folder': 'data'},
        'cases_project': {'cases_workers': 'wrangler'},
        'wrangler': {
            'steps': 'load, merge',
            'load_techniques': 'load_file',
            'merge_techniques': 'merge_keys, merge_ranges'},
        'load_file_parameters': {'source': 'cases.csv'},
        'merge_keys_parameters': {
            'source': 'courts.csv', 'on': 'court', 'indicator': 'found'},
        'merge_ranges_parameters': {
            'source': judges,
            'column': 'year',
            'start': 'began',
            'end': 'ended',
            'on': ['judge', 'court'],
            'other_on': ['name', 'bench'],
            'columns': ['party']}}
    project = amos.Project.create(settings, id = 'run')
    result = project.result
    assert result.data['number'].tolist()[:4] == [1, 1, 9, 9]
    assert result.data['party'].tolist()[:4] == [-1, 1, 1, -1]
    assert [entry['technique'] for entry in result.history] == [
        'load_file', 'merge_keys', 'merge_ranges']
    assert result.history[1]['source'] == 'courts.csv'
    assert result.history[1]['matched'] == 4
    nodes = amos.interface._nodes(project.workflow)
    merger = next(n for n in nodes if isinstance(n, amos.Merger))
    assert merger.clerk is project.clerk
    assert amos.library.classify('merge_summary') == 'merger'
    project.export()
