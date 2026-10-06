"""Tests the base module: Dataset and Operation."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
from conftest import SEED, make_mixed, make_numeric

import amos


def test_create_from_a_data_frame_infers_the_task() -> None:
    dataset = amos.Dataset.create(make_mixed(), label = 'outcome', seed = 3)
    assert dataset.label == 'outcome'
    assert dataset.task == 'classify'
    assert dataset.seed == 3
    assert not dataset.is_split


@pytest.mark.parametrize(('item', 'columns'), [
    (np.zeros((3, 2)), [0, 1]),
    ({'a': [1, 2], 'b': [3, 4]}, ['a', 'b']),
    (pd.Series([1, 2], name = 'a'), ['a']),
])
def test_create_from_other_types(item: Any, columns: list[Any]) -> None:
    assert list(amos.Dataset.create(item).data.columns) == columns


def test_create_from_a_scikit_learn_dataset_uses_its_target() -> None:
    cancer = sklearn.datasets.load_breast_cancer(as_frame = True)
    dataset = amos.Dataset.create(cancer)
    assert dataset.label == 'target'
    assert dataset.task == 'classify'
    assert dataset.data.shape == (569, 31)


@pytest.mark.parametrize('suffix', ['.csv', '.tsv', '.json', '.pkl'])
def test_create_from_a_file(tmp_path: pathlib.Path, suffix: str) -> None:
    data = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
    path = tmp_path / f'data{suffix}'
    if suffix == '.csv':
        data.to_csv(path, index = False)
    elif suffix == '.tsv':
        data.to_csv(path, index = False, sep = '\t')
    elif suffix == '.json':
        data.to_json(path)
    else:
        data.to_pickle(path)
    dataset = amos.Dataset.create(str(path), label = 'a')
    assert dataset.data.shape == (3, 2)
    assert dataset.task == 'classify'


def test_create_raises_for_a_missing_or_unknown_file(
    tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        amos.Dataset.create(tmp_path / 'missing.csv')
    unknown = tmp_path / 'data.unknown'
    unknown.write_text('a', encoding = 'utf-8')
    with pytest.raises(ValueError, match = 'not supported'):
        amos.Dataset.create(unknown)


def test_create_raises_for_an_unsupported_type() -> None:
    with pytest.raises(TypeError, match = 'cannot be made'):
        amos.Dataset.create(42)


def test_create_fills_in_what_a_dataset_lacks() -> None:
    dataset = amos.Dataset(make_numeric())
    same = amos.Dataset.create(dataset, label = 'target', seed = 5)
    assert same is dataset
    assert dataset.label == 'target'
    assert dataset.task == 'classify'
    assert dataset.seed == 5
    # What a dataset already has is kept.
    amos.Dataset.create(dataset, label = 'x0', seed = 9)
    assert dataset.label == 'target'
    assert dataset.seed == 5
    with pytest.raises(KeyError):
        amos.Dataset.create(amos.Dataset(make_numeric()), label = 'missing')


def test_invalid_label_or_task_raises() -> None:
    with pytest.raises(KeyError):
        amos.Dataset(make_numeric(), label = 'missing')
    with pytest.raises(ValueError, match = 'task must be one of'):
        amos.Dataset(make_numeric(), label = 'target', task = 'cluster')


@pytest.mark.parametrize(('values', 'task'), [
    ([True, False, True], 'classify'),
    (['a', 'b', 'a'], 'classify'),
    ([0, 1, 2, 1], 'classify'),
    (list(range(20)), 'regress'),
    ([0.0, 1.0, 1.0], 'classify'),
    ([0.5, 1.5, 2.5], 'regress'),
])
def test_infer_task(values: list[Any], task: str) -> None:
    dataset = amos.Dataset(pd.DataFrame({'y': values}), label = 'y')
    assert dataset.task == task


def test_kinds_of_features(mixed: amos.Dataset) -> None:
    assert mixed.features == ['age', 'income', 'region', 'member', 'joined']
    assert mixed.numerics == ['age', 'income']
    assert mixed.categoricals == ['region']
    assert mixed.booleans == ['member']
    assert mixed.classes == ['no', 'yes']


def test_unsortable_classes_keep_their_order() -> None:
    dataset = amos.Dataset(
        pd.DataFrame({'y': [1, 'a', 1]}, dtype = object),
        label = 'y',
        task = 'classify')
    assert dataset.classes == [1, 'a']


def test_rows_before_and_after_a_split(mixed: amos.Dataset) -> None:
    assert len(mixed.x_train) == len(mixed.data)
    with pytest.raises(ValueError, match = 'has not been split'):
        _ = mixed.x_test
    mixed.split(mixed.data.index[:150], mixed.data.index[150:])
    assert len(mixed.x_train) == 150
    assert len(mixed.y_test) == 50
    assert 'outcome' not in mixed.x_test.columns


def test_split_rejects_overlapping_or_unknown_rows(
    mixed: amos.Dataset) -> None:
    with pytest.raises(ValueError, match = 'both'):
        mixed.split([0, 1], [1, 2])
    with pytest.raises(ValueError, match = 'not in the data'):
        mixed.split([0, 1], [10_000])


def test_y_without_a_label_raises() -> None:
    dataset = amos.Dataset(make_numeric())
    with pytest.raises(ValueError, match = 'no label'):
        _ = dataset.y


def test_replace_keeps_the_split_to_rows_that_remain(
    classified: amos.Dataset) -> None:
    test = list(classified.test)
    classified.replace(classified.data.drop(index = test[:5]))
    assert len(classified.test) == len(test) - 5
    with pytest.raises(KeyError):
        classified.replace(classified.data.drop(columns = 'target'))


def test_resample_gives_new_labels_and_keeps_the_test_rows(
    classified: amos.Dataset) -> None:
    test = classified.data.loc[classified.test].copy()
    x = pd.concat([classified.x_train, classified.x_train.head(10)])
    y = pd.concat([classified.y_train, classified.y_train.head(10)])
    classified.resample(x.reset_index(drop = True), y)
    assert len(classified.x_train) == len(x)
    assert classified.train.min() > classified.data.index.difference(
        classified.train).max()
    pd.testing.assert_frame_equal(classified.data.loc[classified.test], test)


def test_resample_with_text_labels_in_the_index() -> None:
    data = make_numeric(rows = 20)
    data.index = [f'row{i}' for i in range(20)]
    dataset = amos.Dataset(data, label = 'target')
    dataset.resample(dataset.x.head(5), dataset.y.head(5))
    assert list(dataset.data.index) == [f'resampled_{i}' for i in range(5)]


def test_update_features(mixed: amos.Dataset) -> None:
    new = pd.DataFrame({'age_squared': mixed.data['age'] ** 2})
    mixed.update_features(['age'], new)
    assert 'age' not in mixed.data.columns
    # The new column takes the place of the column it replaces.
    assert mixed.features[0] == 'age_squared'
    with pytest.raises(ValueError, match = 'same index'):
        mixed.update_features(['income'], new.reset_index(drop = True).head(3))
    with pytest.raises(ValueError, match = 'same names'):
        mixed.update_features(['income'], new)


def test_repr(fitted: amos.Dataset) -> None:
    text = repr(fitted)
    assert text.startswith('Dataset(rows=200, columns=7')
    assert "label='target'" in text
    assert 'model=LogisticRegression' in text


def test_operation_names_and_records() -> None:
    @dataclasses.dataclass
    class AddOne(amos.Operation):
        def implement(self, item, amount = 1, **kwargs):
            item.data = item.data + amount
            return item

    technique = AddOne()
    assert technique.name == 'add_one'
    result = technique.apply(pd.DataFrame({'a': [1, 2]}), amount = 2)
    assert list(result.data['a']) == [3, 4]
    assert result.history == [{'technique': 'add_one', 'tool': None}]
    assert amos.library.all['add_one'] is AddOne


def test_make_tool_uses_the_seed_and_accepted_parameters(
    classified: amos.Dataset) -> None:
    technique = amos.models.RandomForest()
    tool = technique._make_tool(
        classified,
        {'n_estimators': 5, 'unrelated': 1},
        tool = 'sklearn.ensemble.RandomForestClassifier')
    assert tool.n_estimators == 5
    assert tool.random_state == SEED
    with pytest.raises(NotImplementedError, match = 'has no tool'):
        amos.Transformer()._make_tool(classified, {})
