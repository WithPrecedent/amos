"""Tests the savers module."""

from __future__ import annotations

import pathlib
from typing import Any

import pandas as pd
import pytest
from conftest import SEED, make_numeric

import amos
from amos import savers
from amos.wrangler.savers import _base


@pytest.fixture
def clerk(tmp_path: pathlib.Path) -> amos.Clerk:
    """A clerk whose workspace is the test's own folder."""
    return amos.Clerk(root_folder = tmp_path)


def _settings(**workers: Any) -> dict[str, Any]:
    """Returns the settings of a project named "cancer" with `workers`."""
    return {
        'general': {'label': 'target', 'seed': SEED},
        'cancer_project': {'cancer_workers': list(workers)},
        **workers}


def test_save_data_saves_the_dataset_and_a_table(
    classified: amos.Dataset,
    clerk: amos.Clerk) -> None:
    saver = savers.SaveData(clerk = clerk)
    before = classified.data.copy()
    result = saver.apply(classified)
    folder = pathlib.Path(clerk.interim_folder).resolve()
    dataset_path = folder / 'save_data.joblib'
    table_path = folder / 'save_data.parquet'
    assert result is classified
    pd.testing.assert_frame_equal(result.data, before)
    assert result.history[-1] == {
        'technique': 'save_data',
        'folder': str(folder),
        'file_name': 'save_data',
        'rows': 200,
        'columns': 7,
        'files': [str(dataset_path), str(table_path)]}
    pd.testing.assert_frame_equal(pd.read_parquet(table_path), before)


def test_a_saved_dataset_goes_on_from_where_it_was_saved(
    classified: amos.Dataset,
    clerk: amos.Clerk) -> None:
    fitted = amos.models.Logit().apply(classified)
    savers.SaveData(clerk = clerk).apply(fitted)
    path = pathlib.Path(clerk.interim_folder) / 'save_data.joblib'
    loaded = amos.Dataset.create(path)
    assert isinstance(loaded, amos.Dataset)
    assert loaded.label == 'target'
    assert loaded.seed == SEED
    assert list(loaded.test_rows) == list(fitted.test_rows)
    pd.testing.assert_series_equal(loaded.predictions, fitted.predictions)
    # The saved history ends with the saver's entry, without its files,
    # which were listed once they were saved.
    assert [entry['technique'] for entry in loaded.history] == [
        'stratified', 'logit', 'save_data']
    assert 'files' not in loaded.history[-1]
    assert loaded.model.predict(loaded.x_test).tolist() == (
        fitted.model.predict(fitted.x_test).tolist())


def test_save_data_takes_a_table_format_folder_and_file_name(
    classified: amos.Dataset,
    clerk: amos.Clerk) -> None:
    saver = savers.SaveData(clerk = clerk)
    saver.apply(
        classified, folder = 'output', file_name = 'cases', table_format = 'csv')
    output = pathlib.Path(clerk.output_folder)
    assert (output / 'cases.joblib').is_file()
    table = pd.read_csv(output / 'cases.csv', index_col = 0)
    assert table.shape == classified.data.shape
    saver.apply(classified, folder = 'checkpoints', table_format = False)
    root = pathlib.Path(clerk.root_folder)
    assert sorted(p.name for p in (root / 'checkpoints').iterdir()) == [
        'save_data.joblib']
    assert classified.history[-1]['files'] == [
        str((root / 'checkpoints' / 'save_data.joblib').resolve())]


def test_save_data_rejects_a_format_that_the_clerk_does_not_know(
    classified: amos.Dataset,
    clerk: amos.Clerk) -> None:
    with pytest.raises(ValueError, match = 'not a format that the clerk'):
        savers.SaveData(clerk = clerk).apply(
            classified, table_format = 'nonsense')


def test_save_data_warns_if_the_table_cannot_be_saved(
    clerk: amos.Clerk) -> None:
    # Parquet needs each column to hold one type.
    dataset = amos.Dataset(pd.DataFrame({'mixed': [1, 'x']}))
    with pytest.warns(amos.AmosWarning, match = 'not its data'):
        savers.SaveData(clerk = clerk).apply(dataset)
    folder = pathlib.Path(clerk.interim_folder).resolve()
    assert dataset.history[-1]['files'] == [str(folder / 'save_data.joblib')]


def test_a_project_saves_and_goes_on_from_its_dataset() -> None:
    data = make_numeric()
    settings = _settings(
        wrangler = {'techniques': ['drop_duplicates', 'save_data']},
        analyst = {
            'steps': ['split', 'model'],
            'split_techniques': 'stratified',
            'model_techniques': 'logit'})
    project = amos.Project.create(settings, item = data)
    path = pathlib.Path(project.clerk.interim_folder) / 'cancer_save_data.joblib'
    assert path.is_file()
    # A later project starts after the saver, from the saved file.
    later = _settings(analyst = settings['analyst'])
    resumed = amos.Project.create(later, item = 'cancer_save_data.joblib')
    assert [entry['technique'] for entry in resumed.result.history] == [
        'drop_duplicates', 'save_data', 'stratified', 'logit']
    assert resumed.result.metrics == project.result.metrics
    # A loader goes on from it too.
    loading = {
        'general': {},
        'cancer_project': {'cancer_workers': ['wrangler']},
        'wrangler': {'techniques': 'load_file'},
        'load_file_parameters': {'source': 'cancer_save_data.joblib'}}
    loaded = amos.Project.create(loading).result
    assert loaded.label == 'target'
    assert loaded.seed == SEED
    assert [entry['technique'] for entry in loaded.history] == [
        'drop_duplicates', 'save_data', 'load_file']


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_save_data_saves_every_branch_of_an_experiment(n_jobs: int) -> None:
    settings = _settings(analyst = {
        'steps': ['split', 'scale', 'save', 'model'],
        'design': 'experiment',
        'criterion': 'accuracy',
        'n_jobs': n_jobs,
        'split_techniques': 'stratified',
        'scale_techniques': ['standard', 'min_max'],
        'save_techniques': 'save_data',
        'model_techniques': 'logit'})
    project = amos.Project.create(settings, item = make_numeric())
    folder = pathlib.Path(project.clerk.interim_folder)
    names = sorted(path.stem for path in folder.glob('*.joblib'))
    assert names == [
        'cancer_save_data_stratified_split-min_max_scale-save_data_save-'
        'logit_model',
        'cancer_save_data_stratified_split-standard_scale-save_data_save-'
        'logit_model']


def test_a_long_branch_label_is_cut_and_hashed() -> None:
    label = ' > '.join(f'technique_{number}' for number in range(30))
    safe = _base._safe_label(label)
    assert len(safe) == _base._MAX_BRANCH_LENGTH
    assert safe.startswith('technique_0-technique_1-')
    assert safe != _base._safe_label(label + ' > other')
    assert _base._safe_label('standard > logit') == 'standard-logit'
