"""Tests the groups of a dataset, Polars input, and dates."""

from __future__ import annotations

import os
import pathlib

import pandas as pd
import pytest
from conftest import SEED, make_mixed, make_numeric, requires

import amos


def _grouped(**kwargs: object) -> amos.Dataset:
    """Returns numeric data with a "court" group."""
    data = make_numeric()
    data['court'] = [f'c{i % 4}' for i in range(len(data))]
    return amos.Dataset(
        data, label = 'target', seed = SEED, groups = ['court'], **kwargs)


def test_groups_are_not_features() -> None:
    dataset = _grouped()
    assert dataset.groups == ['court']
    assert 'court' not in dataset.features
    assert 'court' not in dataset.x.columns
    assert "groups=['court']" in repr(dataset)


def test_one_group_can_be_a_name() -> None:
    dataset = amos.Dataset(make_mixed(), label = 'outcome', groups = 'region')
    assert dataset.groups == ['region']
    assert dataset.categoricals == []


def test_invalid_groups_raise() -> None:
    with pytest.raises(KeyError, match = 'not columns'):
        amos.Dataset(make_numeric(), label = 'target', groups = ['missing'])
    with pytest.raises(ValueError, match = 'cannot also be a group'):
        amos.Dataset(make_numeric(), label = 'target', groups = ['target'])


def test_create_adds_groups() -> None:
    dataset = amos.Dataset.create(
        make_mixed(), label = 'outcome', groups = ['region'])
    assert dataset.groups == ['region']
    plain = amos.Dataset(make_mixed(), label = 'outcome')
    assert amos.Dataset.create(plain, groups = 'region').groups == ['region']


def test_transformers_leave_groups_alone() -> None:
    dataset = _grouped()
    amos.splitters.Stratified().apply(dataset)
    amos.transformers.OneHot().apply(dataset)
    assert 'court' in dataset.data.columns
    amos.models.Logit().apply(dataset)
    assert 'court' not in dataset.model.feature_names_in_


def test_replace_needs_the_groups() -> None:
    dataset = _grouped()
    with pytest.raises(KeyError, match = 'groups'):
        amos.cleaners.DropColumns().apply(dataset, columns = ['court'])


def test_keep_columns_keeps_the_groups() -> None:
    dataset = _grouped()
    amos.cleaners.KeepColumns().apply(dataset, columns = ['x0'])
    assert list(dataset.data.columns) == ['x0', 'target', 'court']


def test_rename_columns_renames_the_groups() -> None:
    dataset = _grouped()
    amos.cleaners.RenameColumns().apply(dataset, names = {'court': 'venue'})
    assert dataset.groups == ['venue']
    assert 'venue' in dataset.data.columns


def test_group_split_uses_the_first_group() -> None:
    dataset = _grouped()
    amos.splitters.GroupSplit().apply(dataset, test_size = 0.25)
    train = set(dataset.data.loc[dataset.train, 'court'])
    test = set(dataset.data.loc[dataset.test, 'court'])
    assert train.isdisjoint(test)


def test_samplers_keep_the_groups_of_copied_rows() -> None:
    requires('imblearn')
    data = make_numeric(rows = 300, weights = [0.85])
    data['court'] = [f'c{i % 4}' for i in range(len(data))]
    for kind in (amos.samplers.RandomUnder, amos.samplers.RandomOver):
        dataset = amos.Dataset(
            data.copy(), label = 'target', seed = SEED, groups = ['court'])
        amos.splitters.Stratified().apply(dataset)
        kind().apply(dataset)
        assert not dataset.data.loc[dataset.train, 'court'].isna().any()
    dataset = amos.Dataset(
        data.copy(), label = 'target', seed = SEED, groups = ['court'])
    amos.splitters.Stratified().apply(dataset)
    original = len(dataset.train)
    amos.samplers.Smote().apply(dataset)
    groups = dataset.data.loc[dataset.train, 'court']
    # SMOTE returns the original rows first, so only its new rows lack groups.
    assert groups.notna().sum() == original
    assert groups.isna().sum() == len(dataset.train) - original


def test_resample_without_origins_leaves_groups_empty() -> None:
    dataset = _grouped()
    dataset.resample(dataset.x.head(3), dataset.y.head(3))
    assert dataset.data['court'].isna().all()


def test_the_general_section_sets_the_groups(tmp_path: pathlib.Path) -> None:
    os.chdir(tmp_path)
    data = make_numeric()
    data['court'] = [f'c{i % 4}' for i in range(len(data))]
    settings = {
        'general': {'label': 'target', 'seed': SEED, 'groups': 'court'},
        'study_project': {'techniques': 'stratified, logit'}}
    project = amos.Project.create(settings, item = data)
    assert project.result.groups == ['court']


def test_experiments_keep_the_groups_of_each_branch() -> None:
    dataset = _grouped()
    amos.splitters.Stratified().apply(dataset)
    experiment = amos.Experiment(
        name = 'models', criteria = amos.metrics.Accuracy())
    experiment.populate([[amos.models.Baseline(), amos.models.Logit()]])
    result = experiment.apply(dataset)
    for branch in result.branches:
        assert branch.result.groups == ['court']
        assert 'court' in branch.result.data.columns


def test_polars_data() -> None:
    requires('polars')
    polars = pytest.importorskip('polars')
    frame = polars.from_pandas(make_numeric())
    dataset = amos.Dataset.create(frame, label = 'target')
    assert isinstance(dataset.data, pd.DataFrame)
    assert dataset.data.shape == (200, 7)
    lazy = amos.Dataset.create(frame.lazy(), label = 'target')
    assert lazy.data.shape == (200, 7)


def test_dates_are_a_kind_of_feature(mixed: amos.Dataset) -> None:
    assert mixed.dates == ['joined']
