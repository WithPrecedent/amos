"""Tests the shapers module."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import SEED

import amos
from amos import shapers


@pytest.fixture
def states() -> amos.Dataset:
    """A small "wide" dataset, with a column of each variable for each year."""
    data = pd.DataFrame({
        'state': ['Kansas', 'Missouri', 'Iowa'],
        'region': ['plains', 'midwest', 'midwest'],
        'income_2019': [51, 48, 53],
        'income_2020': [52, 49, 55],
        'tax_2019': [0.05, 0.04, 0.06],
        'tax_2020': [0.05, 0.05, 0.06]})
    return amos.Dataset(data, seed = SEED)


@pytest.fixture
def votes() -> amos.Dataset:
    """A small dataset with a row for each judge on each court case."""
    data = pd.DataFrame({
        'case': ['a1', 'a1', 'a1', 'a2', 'a2', 'a3'],
        'year': [2001, 2001, 2001, 2015, 2015, 2019],
        'court': ['First', 'First', 'First', 'Ninth', 'Ninth', 'First'],
        'judge': ['Lynch', 'Selya', 'Boudin', 'Kozinski', 'Reinhardt', 'Lynch'],
        'woman': [True, False, False, False, False, True],
        'age': [55, 67, 62, 65, 84, 73],
        'reversed': [True, True, True, False, False, True]})
    return amos.Dataset(data, label = 'reversed', seed = SEED)


def test_wide_to_long_stacks_columns(states: amos.Dataset) -> None:
    shapers.WideToLong().apply(
        states,
        columns = ['income_2019', 'income_2020'],
        name = 'measure',
        value = 'income')
    data = states.data
    assert list(data.columns) == [
        'state', 'region', 'tax_2019', 'tax_2020', 'measure', 'income']
    # The rows of each state stay together, in the order of the columns.
    assert data['state'].tolist() == [
        'Kansas', 'Kansas', 'Missouri', 'Missouri', 'Iowa', 'Iowa']
    assert data['measure'].tolist() == ['income_2019', 'income_2020'] * 3
    assert data['income'].tolist() == [51, 52, 48, 49, 53, 55]
    assert data['income'].dtype == 'int64'
    assert data['tax_2019'].tolist() == [0.05, 0.05, 0.04, 0.04, 0.06, 0.06]
    assert list(data.index) == list(range(6))
    assert states.history[-1] == {
        'technique': 'wide_to_long', 'rows': [3, 6], 'columns': [6, 6]}


def test_wide_to_long_names_its_columns_as_usual(
    states: amos.Dataset) -> None:
    shapers.WideToLong().apply(states, columns = 'tax_2019')
    assert list(states.data.columns)[-2:] == ['variable', 'value']
    assert states.data['variable'].tolist() == ['tax_2019'] * 3


def test_wide_to_long_stacks_sets_of_columns(states: amos.Dataset) -> None:
    states.data = states.data.drop(columns = 'tax_2020')
    shapers.WideToLong().apply(
        states, stubs = ['income', 'tax'], name = 'year')
    data = states.data
    assert list(data.columns) == ['state', 'region', 'year', 'income', 'tax']
    # Names that are all numbers are made into numbers.
    assert data['year'].tolist() == [2019, 2020] * 3
    assert data['year'].dtype == 'int64'
    assert data['income'].tolist() == [51, 52, 48, 49, 53, 55]
    # A set that lacks a column is missing there.
    assert data['tax'].tolist()[::2] == [0.05, 0.04, 0.06]
    assert data['tax'].isna().tolist() == [False, True] * 3
    assert states.numerics == ['year', 'income', 'tax']


def test_wide_to_long_with_other_separators_and_names() -> None:
    data = pd.DataFrame({
        'id': [1, 2], 'vote.first': ['yes', 'no'], 'vote.second': ['no', 'no']})
    dataset = shapers.WideToLong().apply(
        data, stubs = 'vote', separator = '.', name = 'round')
    assert dataset.data['round'].tolist() == [
        'first', 'second', 'first', 'second']
    assert dataset.data['vote'].tolist() == ['yes', 'no', 'no', 'no']


def test_wide_to_long_checks_its_columns(states: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'not both'):
        shapers.WideToLong().apply(
            states, columns = 'tax_2019', stubs = 'income')
    with pytest.raises(KeyError, match = "begins with 'sales_'"):
        shapers.WideToLong().apply(states, stubs = 'sales')
    with pytest.raises(KeyError, match = 'not in the data'):
        shapers.WideToLong().apply(states, columns = 'sales_2019')
    with pytest.raises(ValueError, match = "two columns named 'region'"):
        shapers.WideToLong().apply(
            states, stubs = 'income', name = 'region')
    with pytest.raises(ValueError, match = "two columns named 'state'"):
        shapers.WideToLong().apply(
            states, columns = 'tax_2019', value = 'state')
    assert states.data.shape == (3, 6)


def test_long_to_wide_makes_a_column_for_each_name(
    votes: amos.Dataset) -> None:
    votes.data['seat'] = [1, 2, 3, 1, 2, 1]
    shapers.LongToWide().apply(
        votes,
        names = 'seat',
        stubs = ['judge', 'age'],
        ids = ['case', 'year', 'reversed'])
    data = votes.data
    assert list(data.columns) == [
        'case', 'year', 'reversed', 'judge_1', 'judge_2', 'judge_3',
        'age_1', 'age_2', 'age_3']
    assert data['case'].tolist() == ['a1', 'a2', 'a3']
    assert data['judge_1'].tolist() == ['Lynch', 'Kozinski', 'Lynch']
    assert data['judge_3'].tolist()[0] == 'Boudin'
    # A cell that no row fills is missing.
    assert data['judge_3'].isna().tolist() == [False, True, True]
    assert data['age_2'].tolist()[:2] == [67, 84]
    assert pd.isna(data['age_2'].iloc[2])
    assert data['year'].dtype == 'int64'
    assert (votes.label, votes.task) == ('reversed', 'classify')
    assert votes.history[-1] == {
        'technique': 'long_to_wide', 'rows': [6, 3], 'columns': [8, 9]}


def test_long_to_wide_names_columns_for_values_alone() -> None:
    data = pd.DataFrame({
        'state': ['Kansas', 'Kansas', 'Iowa', 'Iowa'],
        'measure': ['tax', 'income', 'income', 'tax'],
        'value': [0.05, 51, 53, 0.06]})
    dataset = shapers.LongToWide().apply(
        data, names = 'measure', values = 'value')
    # The other columns say which row each row belongs to, the rows are in
    # the order in which they first appear, and the columns are in order.
    assert list(dataset.data.columns) == ['state', 'income', 'tax']
    assert dataset.data['state'].tolist() == ['Kansas', 'Iowa']
    assert dataset.data['income'].tolist() == [51, 53]
    assert dataset.data['tax'].tolist() == [0.05, 0.06]


def test_long_to_wide_writes_numbers_and_categories_in_names() -> None:
    data = pd.DataFrame({
        'state': ['Kansas', 'Kansas'],
        'year': [2019.0, 2020.0],
        'party': pd.Categorical(['red', 'blue'], ['red', 'blue', 'green']),
        'income': [51, 52]})
    years = shapers.LongToWide().apply(
        data, names = 'year', stubs = 'income', ids = 'state',
        separator = '')
    assert list(years.data.columns) == ['state', 'income2019', 'income2020']
    parties = shapers.LongToWide().apply(
        data, names = 'party', values = 'income', ids = 'state')
    assert list(parties.data.columns) == ['state', 'blue', 'red']
    nothing = shapers.LongToWide().apply(
        data[['year', 'income']], names = 'year', values = 'income')
    assert nothing.data.to_dict('list') == {'2019': [51], '2020': [52]}


def test_long_to_wide_checks_its_rows(votes: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = '"values".*or "stubs"'):
        shapers.LongToWide().apply(votes, names = 'judge')
    with pytest.raises(ValueError, match = '"values".*or "stubs"'):
        shapers.LongToWide().apply(
            votes, names = 'judge', values = 'age', stubs = 'woman')
    with pytest.raises(ValueError, match = 'two rows cannot fill one cell'):
        shapers.LongToWide().apply(
            votes, names = 'court', values = 'age', ids = 'year')
    with pytest.raises(ValueError, match = "two columns named 'First'"):
        shapers.LongToWide().apply(
            votes.data.assign(First = 1).drop(index = [1, 2, 4]),
            names = 'court', values = 'age')
    votes.data['court'] = votes.data['court'].mask(votes.data['year'] == 2019)
    with pytest.raises(ValueError, match = "1 rows have no value in 'court'"):
        shapers.LongToWide().apply(
            votes, names = 'court', values = 'age', ids = 'case')
    assert votes.data.shape == (6, 7)


def test_reshaping_wide_to_long_and_back_changes_nothing(
    states: amos.Dataset) -> None:
    before = states.data.copy()
    shapers.WideToLong().apply(
        states, stubs = ['income', 'tax'], name = 'year')
    assert states.data.shape == (6, 5)
    shapers.LongToWide().apply(
        states, names = 'year', stubs = ['income', 'tax'])
    pd.testing.assert_frame_equal(states.data, before)
    shapers.WideToLong().apply(states, columns = ['tax_2019', 'tax_2020'])
    shapers.LongToWide().apply(states, names = 'variable', values = 'value')
    pd.testing.assert_frame_equal(states.data, before[states.data.columns])


def test_lists_to_rows_splits_text() -> None:
    data = pd.DataFrame({
        'case': ['a1', 'a2', 'a3', 'a4'],
        'panel': ['Lynch; Selya;Boudin', 'Kozinski;; Reinhardt ', ' ; ', None],
        'reversed': [True, False, True, False]})
    dataset = amos.Dataset(data, label = 'reversed')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        shapers.ListsToRows().apply(dataset, column = 'panel', separator = ';')
    assert dataset.data['panel'].tolist()[:5] == [
        'Lynch', 'Selya', 'Boudin', 'Kozinski', 'Reinhardt']
    # A row with no items is kept, with a missing item.
    assert dataset.data['case'].tolist() == [
        'a1', 'a1', 'a1', 'a2', 'a2', 'a3', 'a4']
    assert dataset.data['panel'].isna().tolist()[5:] == [True, True]
    assert dataset.data['reversed'].tolist() == [
        True, True, True, False, False, True, False]
    assert list(dataset.data.index) == list(range(7))
    assert dataset.categoricals == ['case', 'panel']
    assert dataset.history[-1] == {
        'technique': 'lists_to_rows', 'rows': [4, 7], 'columns': [3, 3]}
    with pytest.raises(ValueError, match = 'needs a "separator"'):
        shapers.ListsToRows().apply(data, column = 'panel')


def test_lists_to_rows_splits_lists_and_numbers_their_items() -> None:
    data = pd.DataFrame({
        'case': ['a1', 'a2', 'a3', 'a4'],
        'panel': [['Lynch', 'Selya'], ('Kozinski',), [], None]})
    dataset = shapers.ListsToRows().apply(
        data, column = 'panel', name = 'judge', position = 'seat')
    assert list(dataset.data.columns) == ['case', 'panel', 'judge', 'seat']
    assert dataset.data['judge'].tolist()[:3] == ['Lynch', 'Selya', 'Kozinski']
    assert dataset.data['judge'].isna().tolist()[3:] == [True, True]
    assert dataset.data['seat'].tolist()[:3] == [1, 2, 1]
    assert dataset.data['seat'].isna().tolist()[3:] == [True, True]
    # The lists are kept beside their items.
    assert dataset.data['panel'].iloc[1] == ['Lynch', 'Selya']
    numbers = shapers.ListsToRows().apply(
        pd.DataFrame({'years': [[2001, 2002], [2015]]}), column = 'years')
    assert numbers.data['years'].tolist() == [2001, 2002, 2015]
    assert numbers.numerics == ['years']
    mixed = shapers.ListsToRows().apply(
        pd.DataFrame({'panel': [['Lynch', 'Selya'], 'Kozinski | Reinhardt']}),
        column = 'panel', separator = '|')
    assert mixed.data['panel'].tolist() == [
        'Lynch', 'Selya', 'Kozinski', 'Reinhardt']


def test_lists_to_rows_checks_the_names_of_its_columns() -> None:
    data = pd.DataFrame({'case': ['a1'], 'panel': [['Lynch']], 'seat': [0]})
    with pytest.raises(ValueError, match = "two columns named 'case'"):
        shapers.ListsToRows().apply(data, column = 'panel', name = 'case')
    with pytest.raises(ValueError, match = "two columns named 'seat'"):
        shapers.ListsToRows().apply(data, column = 'panel', position = 'seat')
    with pytest.raises(KeyError, match = 'not in the data'):
        shapers.ListsToRows().apply(data, column = 'judges')


def test_repeated_rows_keep_an_index_with_a_name() -> None:
    data = pd.DataFrame(
        {'panel': ['Lynch;Selya', 'Kozinski'], 'tax_1': [1, 2], 'tax_2': [3, 4]},
        index = pd.Index(['a1', 'a2'], name = 'case'))
    listed = shapers.ListsToRows().apply(
        data.copy(), column = 'panel', separator = ';')
    assert listed.data['case'].tolist() == ['a1', 'a1', 'a2']
    assert list(listed.data.index) == [0, 1, 2]
    stacked = shapers.WideToLong().apply(data.copy(), stubs = 'tax')
    assert stacked.data['case'].tolist() == ['a1', 'a1', 'a2', 'a2']
    # An index without a name (or with the name of a column) is not kept.
    unnamed = shapers.ListsToRows().apply(
        data.rename_axis(None), column = 'panel', separator = ';')
    assert list(unnamed.data.columns) == ['panel', 'tax_1', 'tax_2']
    taken = shapers.ListsToRows().apply(
        data.rename_axis('tax_1'), column = 'panel', separator = ';')
    assert taken.data['tax_1'].tolist() == [1, 1, 2]


def test_group_rows_summarizes_columns_as_usual(votes: amos.Dataset) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        shapers.GroupRows().apply(votes, by = 'case', count = 'judges')
    data = votes.data
    assert list(data.columns) == [
        'case', 'year', 'court', 'judge', 'woman', 'age', 'reversed',
        'judges']
    assert data['case'].tolist() == ['a1', 'a2', 'a3']
    # A column that is the same throughout each group keeps its values.
    assert data['year'].tolist() == [2001, 2015, 2019]
    assert data['year'].dtype == 'int64'
    assert data['reversed'].tolist() == [True, False, True]
    assert data['reversed'].dtype == bool
    # The mean is taken of numbers and booleans that vary.
    assert data['age'].tolist() == [184 / 3, 74.5, 73]
    assert data['woman'].tolist() == [1 / 3, 0, 1]
    # The first value is taken of anything else.
    assert data['judge'].tolist() == ['Lynch', 'Kozinski', 'Lynch']
    assert data['judges'].tolist() == [3, 2, 1]
    assert (votes.label, votes.task) == ('reversed', 'classify')
    assert votes.history[-1] == {
        'technique': 'group_rows', 'rows': [6, 3], 'columns': [7, 8]}


def test_group_rows_takes_the_summaries_it_is_told_to(
    votes: amos.Dataset) -> None:
    chosen = shapers.GroupRows().apply(
        votes.data.copy(),
        by = ['court', 'year'],
        how = {
            'judge': 'list', 'age': ['min', 'max'], 'woman': 'any',
            'case': 'nunique'})
    assert list(chosen.data.columns) == [
        'court', 'year', 'case', 'judge', 'woman', 'age_min', 'age_max',
        'reversed']
    assert chosen.data['judge'].tolist()[0] == ['Lynch', 'Selya', 'Boudin']
    assert chosen.data['age_min'].tolist() == [55, 65, 73]
    assert chosen.data['age_max'].tolist() == [67, 84, 73]
    assert chosen.data['woman'].tolist() == [True, False, True]
    assert chosen.data['case'].tolist() == [1, 1, 1]
    every = shapers.GroupRows().apply(
        votes.data[['court', 'age', 'woman']], by = 'court', how = 'sum')
    assert every.data.to_dict('list') == {
        'court': ['First', 'Ninth'], 'age': [257, 149], 'woman': [2, 0]}
    several = shapers.GroupRows().apply(
        votes.data[['court', 'age']], by = 'court', how = ['count', 'median'])
    assert several.data.to_dict('list') == {
        'court': ['First', 'Ninth'],
        'age_count': [4, 2],
        'age_median': [64.5, 74.5]}
    alone = shapers.GroupRows().apply(votes.data[['court']], by = 'court')
    assert alone.data.to_dict('list') == {'court': ['First', 'Ninth']}


def test_group_rows_checks_its_summaries(votes: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'summary must be one of'):
        shapers.GroupRows().apply(votes, by = 'case', how = 'average')
    with pytest.raises(TypeError, match = "the mean of 'court'"):
        shapers.GroupRows().apply(votes, by = 'case', how = 'mean')
    with pytest.raises(ValueError, match = 'cannot be summarized'):
        shapers.GroupRows().apply(votes, by = 'case', how = {'case': 'first'})
    with pytest.raises(KeyError, match = 'not in the data'):
        shapers.GroupRows().apply(votes, by = 'case', how = {'name': 'first'})
    with pytest.raises(ValueError, match = "two columns named 'age'"):
        shapers.GroupRows().apply(votes, by = 'case', count = 'age')
    votes.data['age_max'] = 0
    with pytest.raises(ValueError, match = "two columns named 'age_max'"):
        shapers.GroupRows().apply(
            votes, by = 'case', how = {'age': ['min', 'max']})


def test_group_rows_keeps_rows_without_a_group_together(
    votes: amos.Dataset) -> None:
    votes.data['court'] = votes.data['court'].mask(votes.data['year'] == 2015)
    shapers.GroupRows().apply(votes, by = 'court', how = 'first')
    assert votes.data['court'].tolist()[0] == 'First'
    assert pd.isna(votes.data['court'].iloc[1])
    assert votes.data['judge'].tolist() == ['Lynch', 'Kozinski']


def test_lists_to_rows_and_group_rows_are_opposites() -> None:
    data = pd.DataFrame({
        'case': ['a1', 'a2'],
        'year': [2001, 2015],
        'panel': [['Lynch', 'Selya', 'Boudin'], ['Kozinski', 'Reinhardt']]})
    dataset = shapers.ListsToRows().apply(data.copy(), column = 'panel')
    assert len(dataset.data) == 5
    shapers.GroupRows().apply(dataset, by = 'case', how = {'panel': 'list'})
    pd.testing.assert_frame_equal(dataset.data, data)


def test_shapers_do_nothing_without_parameters(votes: amos.Dataset) -> None:
    before = votes.data.copy()
    for kind in (
        shapers.GroupRows, shapers.ListsToRows, shapers.LongToWide,
        shapers.WideToLong):
        kind().apply(votes)
        assert votes.history[-1]['rows'] == [6, 6], kind
    pd.testing.assert_frame_equal(votes.data, before)


def test_shapers_come_before_the_split(votes: amos.Dataset) -> None:
    amos.splitters.TrainTest().apply(votes)
    with pytest.raises(ValueError, match = 'before the data is split'):
        shapers.GroupRows().apply(votes, by = 'case')
    assert len(votes.data) == 6


def test_shapers_need_the_label_and_groups_of_the_new_shape(
    votes: amos.Dataset) -> None:
    votes.groups = ['court']
    votes.data['seat'] = [1, 2, 3, 1, 2, 1]
    with pytest.raises(KeyError, match = 'no columns.*court.*"groups"'):
        shapers.LongToWide().apply(
            votes, names = 'seat', stubs = 'judge', ids = ['case', 'reversed'])
    with pytest.raises(KeyError, match = 'no columns.*reversed.*"label"'):
        shapers.LongToWide().apply(
            votes, names = 'seat', stubs = 'judge', ids = 'case', groups = [])
    assert votes.data.shape == (6, 8)
    assert (votes.label, votes.groups) == ('reversed', ['court'])
    shapers.LongToWide().apply(
        votes,
        names = 'seat',
        stubs = 'age',
        ids = ['case', 'year'],
        label = 'age_1',
        groups = 'year')
    assert (votes.label, votes.task) == ('age_1', 'regress')
    assert votes.groups == ['year']
    assert votes.features == ['case', 'age_2', 'age_3']


def test_shapers_set_the_task_they_are_given(votes: amos.Dataset) -> None:
    shapers.GroupRows().apply(
        votes, by = 'case', how = {'reversed': 'mean'}, task = 'regress')
    assert (votes.label, votes.task) == ('reversed', 'regress')
    with pytest.raises(ValueError, match = 'task must be one of'):
        shapers.GroupRows().apply(votes, by = 'case', task = 'cluster')
    with pytest.raises(ValueError, match = 'cannot also be a group'):
        shapers.GroupRows().apply(votes, by = 'case', groups = 'reversed')


def test_a_shaper_only_has_to_reshape_the_data(votes: amos.Dataset) -> None:

    @dataclasses.dataclass
    class FirstOfEach(amos.Shaper):
        """Keeps the first row of each group."""

        def shape(self, data, by = None, **kwargs):
            return data.drop_duplicates(by).reset_index(drop = True)

    FirstOfEach().apply(votes, by = 'case')
    assert votes.data['judge'].tolist() == ['Lynch', 'Kozinski', 'Lynch']
    assert votes.history[-1] == {
        'technique': 'first_of_each', 'rows': [6, 3], 'columns': [7, 7]}
    assert amos.library.classify('first_of_each') == 'shaper'


def test_shapers_work_in_a_project() -> None:
    cases = pd.DataFrame({
        'case': ['a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
        'panel': [
            'Lynch; Selya; Boudin', 'Kozinski; Reinhardt', 'Lynch; Barron',
            'Selya; Lynch', 'Reinhardt; Kozinski', 'Barron'],
        'reversed': [True, False, True, True, False, False]})
    judges = pd.DataFrame({
        'judge': ['Lynch', 'Selya', 'Boudin', 'Kozinski', 'Reinhardt'],
        'party': [-1, 1, 1, 1, -1],
        'woman': [True, False, False, False, False]})
    settings = {
        'general': {'label': 'reversed', 'seed': SEED},
        'cases_project': {'cases_workers': 'wrangler, explorer'},
        'wrangler': {
            'steps': 'split, merge, group',
            'split_techniques': 'lists_to_rows',
            'merge_techniques': 'merge_keys',
            'group_techniques': 'group_rows'},
        'lists_to_rows_parameters': {
            'column': 'panel', 'separator': ';', 'name': 'judge'},
        'merge_keys_parameters': {
            'source': judges, 'on': 'judge', 'prefix': 'panel_'},
        'group_rows_parameters': {
            'by': 'case', 'how': {'judge': 'list'}, 'count': 'judges'},
        'explorer': {'techniques': 'label_balance'}}
    project = amos.Project.create(settings, item = cases)
    result = project.result
    assert list(result.data.columns) == [
        'case', 'panel', 'reversed', 'judge', 'panel_party', 'panel_woman',
        'judges']
    assert result.data['judges'].tolist() == [3, 2, 2, 2, 2, 1]
    assert result.data['panel_woman'].tolist()[:2] == [1 / 3, 0]
    # Judge Barron is not in the other table, so the means leave him out.
    assert result.data['panel_party'].tolist()[2] == -1
    assert np.isnan(result.data['panel_party'].iloc[5])
    assert result.data['reversed'].tolist() == cases['reversed'].tolist()
    assert [entry['technique'] for entry in result.history] == [
        'lists_to_rows', 'merge_keys', 'group_rows', 'label_balance']
    assert amos.library.classify('wide_to_long') == 'shaper'
    # The data that the project was given is not changed.
    assert cases.shape == (6, 3)
