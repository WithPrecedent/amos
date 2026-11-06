"""Tests the interface module (Project) and the reports module."""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_numeric, make_regression, requires

import amos
import chrisjen
import nagata


def _settings(**extra: Any) -> dict[str, Any]:
    """Returns settings for a small classification project."""
    return {
        'general': {'label': 'target', 'seed': SEED},
        'study_project': {'study_workers': 'explorer, analyst, critic'},
        'explorer': {'techniques': 'summarize'},
        'analyst': {
            'design': 'experiment',
            'criterion': 'accuracy',
            'steps': 'split, scale, model',
            'split_techniques': 'stratified',
            'scale_techniques': 'standard, none',
            'model_techniques': 'logit, baseline'},
        'critic': {'techniques': 'scorecard, confusion'},
        **extra}


@pytest.fixture(autouse = True)
def _work_in(tmp_path: pathlib.Path) -> None:
    """Runs each test in its own folder."""
    os.chdir(tmp_path)


def test_create_and_apply() -> None:
    project = amos.Project.create(
        _settings(), item = make_numeric(), id = 'run')
    result = project.result
    assert isinstance(result, amos.Dataset)
    assert result.label == 'target'
    assert result.seed == SEED
    assert project.name == 'study'
    assert {'summarize', 'analyst_comparison', 'scorecard', 'confusion'} <= set(
        result.tables)
    workers = {node.name: node for node in project.workflow}
    assert isinstance(workers['analyst'], amos.Experiment)
    assert len(workers['analyst'].results) == 4
    assert project.result.history[-1]['technique'] == 'confusion'


def test_the_report_describes_the_project() -> None:
    project = amos.Project.create(
        _settings(), item = make_numeric(), id = 'run')
    text = project.report.contents
    assert isinstance(project.report, amos.Findings)
    assert text.startswith('project: study\nid: run\n')
    assert 'workflow: explorer > analyst > critic' in text
    assert 'label: target (classify)' in text
    assert f'seed: {SEED}' in text
    assert 'split: 150 training rows and 50 test rows' in text
    assert 'analyst: the best of 4 paths was stratified_split' in text
    assert 'tables: summarize, analyst_comparison, scorecard, confusion' in text


def test_the_report_counts_every_path_when_memory_is_conserved() -> None:
    settings = _settings()
    settings['analyst']['conserve_memory'] = True
    project = amos.Project.create(settings, item = make_numeric(), id = 'run')
    workers = {node.name: node for node in project.workflow}
    assert len(workers['analyst'].results) == 1
    text = project.report.contents
    assert 'analyst: the best of 4 paths was stratified_split' in text


def test_the_report_of_a_result_that_is_not_a_dataset() -> None:
    project = chrisjen.Project.create(
        {'count_project': {'techniques': 'none'}},
        item = 3,
        report = amos.Findings(),
        clerk = nagata.FileManager())
    assert 'result: 3' in project.report.contents


def test_applying_twice_gives_the_same_result() -> None:
    data = make_numeric()
    project = amos.Project.create(_settings(), item = data)
    first = project.result.metrics.copy()
    second = project.apply().metrics
    assert first == second
    pd.testing.assert_frame_equal(project.item.data, data)


def test_a_project_without_data_is_not_applied() -> None:
    project = amos.Project.create(_settings())
    assert project.result is None
    assert project.workflow is not None
    result = project.apply(make_numeric())
    assert result.label == 'target'
    with pytest.raises(ValueError, match = 'no data'):
        project.apply()


def test_a_settings_file_and_a_data_file(tmp_path: pathlib.Path) -> None:
    (tmp_path / 'data').mkdir()
    make_numeric().to_csv(tmp_path / 'data' / 'cases.csv', index = False)
    (tmp_path / 'study.ini').write_text(
        '[general]\nlabel = target\nseed = 43\n\n'
        '[files]\nroot_folder = .\ninput_folder = data\n'
        'output_folder = results\n\n'
        '[study_project]\nstudy_techniques = stratified, logit\n',
        encoding = 'utf-8')
    project = amos.Project.create('study.ini', item = 'cases.csv')
    assert project.result.model is not None
    assert pathlib.Path(project.clerk.output_folder).name == 'results'


@pytest.mark.parametrize('name', ['cases.tab', 'cases.csv.gz'])
def test_a_data_file_is_read_as_load_file_reads_it(name: str) -> None:
    # Only nagata's formats read ".tab" and compressed files.
    pathlib.Path('data').mkdir()
    make_numeric().to_csv(
        pathlib.Path('data') / name,
        index = False,
        sep = '	' if name.endswith('.tab') else ',')
    project = amos.Project.create(
        _settings(files = {'input_folder': 'data'}),
        item = name,
        automatic = False)
    assert project.item.data.shape == make_numeric().shape
    assert project.item.label == 'target'
    with pytest.raises(FileNotFoundError, match = 'no file'):
        amos.Project.create(_settings(), item = 'missing.csv')
    pathlib.Path('cases.unknown').write_text('a', encoding = 'utf-8')
    with pytest.raises(ValueError, match = 'load_file'):
        amos.Project.create(_settings(), item = 'cases.unknown')


def test_the_general_section_sets_the_task() -> None:
    settings = _settings()
    settings['general']['task'] = 'regress'
    settings['analyst']['model_techniques'] = 'linear'
    settings['analyst']['criterion'] = 'r2'
    settings['critic']['techniques'] = 'scorecard'
    project = amos.Project.create(settings, item = make_numeric())
    assert project.result.task == 'regress'
    assert 'rmse' in project.result.metrics


def test_a_clerk_can_be_passed(tmp_path: pathlib.Path) -> None:
    clerk = nagata.FileManager(root_folder = tmp_path / 'elsewhere')
    project = amos.Project.create(_settings(), clerk = clerk)
    assert project.clerk is clerk
    project = amos.Project.create(_settings(), clerk = tmp_path / 'root')
    assert pathlib.Path(project.clerk.root_folder) == tmp_path / 'root'


def test_export_saves_everything(tmp_path: pathlib.Path) -> None:
    requires('matplotlib')
    requires('seaborn')
    requires('docx')
    settings = _settings()
    settings['critic']['techniques'] = 'scorecard, confusion, confusion_heatmap'
    project = amos.Project.create(settings, item = make_numeric(), id = 'run')
    folder = project.export(data = True)
    # Each run is saved in a folder of the workspace's "studies" folder.
    assert folder == tmp_path / 'studies' / 'run'
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*.*'))
    assert files == [
        'data.csv',
        'environment.json',
        'figures/confusion_heatmap.png',
        'history.json',
        'metrics.csv',
        'predictions.csv',
        'report.txt',
        'scorecard.csv',
        'scorecard.docx',
        'scorecard.md',
        'scorecard.png',
        'settings.json',
        'study_replication.json',
        'tables/analyst_comparison.csv',
        'tables/confusion.csv',
        'tables/scorecard.csv',
        'tables/summarize.csv']
    winning = json.loads(
        (folder / 'study_replication.json').read_text('utf-8'))
    assert winning == dict(project.winning_settings())
    settings = json.loads((folder / 'settings.json').read_text('utf-8'))
    assert settings['general']['seed'] == SEED
    history = json.loads((folder / 'history.json').read_text('utf-8'))
    assert history == json.loads(json.dumps(
        project.result.history, default = str))
    predictions = pd.read_csv(folder / 'predictions.csv', index_col = 0)
    assert list(predictions.columns) == [
        'actual', 'predicted', 'probability_0', 'probability_1']
    report = (folder / 'report.txt').read_text('utf-8')
    assert report == project.report.contents
    image = importlib.import_module('PIL.Image')
    with image.open(folder / 'figures/confusion_heatmap.png') as figure:
        assert round(figure.info['dpi'][0]) == amos.options._FIGURE_DPI


def test_export_saves_a_history_with_keys_that_json_cannot_store() -> None:
    project = amos.Project.create(
        _settings(), item = make_numeric(), id = 'run')
    # json refuses keys that are `numpy` values or tuples.
    project.result.record(
        'note', counts = {np.bool_(True): 1, (1, 2): 2, 3: [{np.int64(4): 5}]})
    folder = project.export()
    history = json.loads((folder / 'history.json').read_text('utf-8'))
    assert history[-1] == {
        'technique': 'note',
        'counts': {'true': 1, '(1, 2)': 2, '3': [{'4': 5}]}}


def test_export_after_each_apply_if_the_settings_say_so(
    tmp_path: pathlib.Path) -> None:
    settings = _settings(files = {'export_results': True})
    amos.Project.create(settings, item = make_numeric(), id = 'auto')
    assert (tmp_path / 'studies' / 'auto' / 'report.txt').is_file()


def test_export_before_apply_raises() -> None:
    project = amos.Project.create(_settings())
    with pytest.raises(ValueError, match = 'apply the project'):
        project.export()


def test_a_project_exports_a_plot_that_was_skipped_in_its_history() -> None:
    settings = _settings(
        analyst = {
            'design': 'experiment',
            'criterion': 'accuracy',
            'conserve_memory': True,
            'steps': 'split, model',
            'split_techniques': 'stratified',
            'model_techniques': 'logit, baseline'},
        critic = {'techniques': 'scorecard, coefficient_plot'})
    project = amos.Project.create(settings, item = make_numeric(), id = 'run')
    folder = project.export()
    assert 'coefficient_plot' not in project.result.figures
    history = json.loads(
        (folder / 'history.json').read_text(encoding = 'utf-8'))
    entry = next(e for e in history if e.get('skipped'))
    assert entry['technique'] == 'coefficient_plot'
    assert 'no table of coefficients' in entry['reason']


def test_winning_settings_keep_only_the_winning_path() -> None:
    settings = _settings(
        logit_parameters = {'max_iter': 500},
        baseline_parameters = {'strategy': 'prior'},
        standard_scale_parameters = {'with_mean': True})
    project = amos.Project.create(settings, item = make_numeric())
    workers = {node.name: node for node in project.workflow}
    assert workers['analyst'].winner == (
        'stratified_split > none_scale > logit_model')
    winning = project.winning_settings()
    assert winning is not None
    assert winning['analyst'] == {
        'steps': 'split, scale, model',
        'split_techniques': 'stratified',
        'scale_techniques': 'none',
        'model_techniques': 'logit'}
    assert winning['logit_parameters'] == {'max_iter': 500}
    assert 'standard_scale_parameters' not in winning
    assert 'baseline_parameters' not in winning
    # The other workers and the settings of the project are kept.
    assert winning['explorer'] == settings['explorer']
    assert winning['critic'] == settings['critic']
    assert winning['general'] == settings['general']
    # The settings were copied, not changed.
    assert project.idea['analyst']['design'] == 'experiment'
    again = amos.Project.create(winning, item = make_numeric())
    replica = {node.name: node for node in again.workflow}
    assert isinstance(replica['analyst'], chrisjen.Flow)
    assert not isinstance(replica['analyst'], chrisjen.Contest)
    pd.testing.assert_series_equal(
        again.result.predictions, project.result.predictions)


def test_winning_settings_pin_the_values_a_search_chose() -> None:
    settings = _settings(
        random_forest_parameters = {'n_estimators': [5, 20], 'cv': 3},
        random_forest_model_parameters = {
            'search': 'grid', 'max_depth': [2, 4]})
    settings['analyst']['model_techniques'] = 'baseline, random_forest'
    project = amos.Project.create(settings, item = make_numeric())
    winning = project.winning_settings()
    assert winning is not None
    assert winning['analyst']['model_techniques'] == 'random_forest'
    assert 'random_forest_search' in project.result.tables
    best = project.result.model.get_params()
    assert winning['random_forest_parameters'] == {
        'n_estimators': best['n_estimators']}
    assert winning['random_forest_model_parameters'] == {
        'max_depth': best['max_depth']}
    again = amos.Project.create(winning, item = make_numeric())
    assert 'random_forest_search' not in again.result.tables
    pd.testing.assert_series_equal(
        again.result.predictions, project.result.predictions)


def test_winning_settings_of_a_contest_without_steps() -> None:
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'study_project': {'study_workers': 'splitter, modeler'},
        'splitter': {'techniques': 'stratified'},
        'modeler': {
            'design': 'contest',
            'criterion': 'accuracy',
            'conserve_memory': True,
            'techniques': 'logit, baseline'},
        'baseline_parameters': {'strategy': 'prior'}}
    project = amos.Project.create(settings, item = make_numeric())
    winning = project.winning_settings()
    assert winning is not None
    assert winning['modeler'] == {'techniques': 'logit'}
    assert 'baseline_parameters' not in winning
    assert winning['splitter'] == settings['splitter']


def test_winning_settings_of_a_contest_inside_an_experiment() -> None:
    settings = _settings(
        analyst = {
            'design': 'experiment',
            'criterion': 'accuracy',
            'steps': 'split, scale, modeler',
            'split_techniques': 'stratified',
            'scale_techniques': 'standard, none'},
        modeler = {
            'design': 'experiment',
            'criterion': 'accuracy',
            'techniques': 'logit, baseline'})
    project = amos.Project.create(settings, item = make_numeric())
    # The inner experiment is applied only as a copy in each path, so its
    # winner is found in the history.
    workers = {node.name: node for node in project.workflow}
    assert workers['analyst'].winner.endswith(' > modeler')
    winning = project.winning_settings()
    assert winning is not None
    assert winning['modeler'] == {'techniques': 'logit'}
    assert 'design' not in winning['analyst']
    again = amos.Project.create(winning, item = make_numeric())
    pd.testing.assert_series_equal(
        again.result.predictions, project.result.predictions)


def test_winning_settings_drop_the_pipeline() -> None:
    settings = {
        'general': {'label': 'target', 'seed': SEED, 'pipeline': 'regression'}}
    project = amos.Project.create(settings, item = make_regression())
    winning = project.winning_settings()
    assert winning is not None
    assert 'pipeline' not in winning['general']
    assert 'design' not in winning['analyst']
    assert isinstance(winning['analyst']['model_techniques'], str)
    # The pipeline's parameters of a model that lost are dropped too.
    if winning['analyst']['model_techniques'] != 'random_forest':
        assert 'random_forest_parameters' not in winning
    again = amos.Project.create(winning, item = make_regression())
    assert 'design' not in again.idea['analyst']
    pd.testing.assert_series_equal(
        again.result.predictions, project.result.predictions)


def test_winning_settings_without_a_contest() -> None:
    settings = _settings(
        analyst = {
            'steps': 'split, model',
            'split_techniques': 'stratified',
            'model_techniques': 'logit'})
    project = amos.Project.create(settings, item = make_numeric(), id = 'run')
    assert project.winning_settings() is None
    assert project.export_replication() is None
    folder = project.export()
    assert not list(folder.glob('*_replication.*'))


def test_winning_settings_before_apply_raise() -> None:
    project = amos.Project.create(_settings())
    with pytest.raises(ValueError, match = 'apply the project'):
        project.winning_settings()


def test_the_chosen_parameters_of_a_wrapped_classifier() -> None:
    linear_model = importlib.import_module('sklearn.linear_model')
    wrapped = amos.models.LabelCoded(
        estimator = linear_model.LogisticRegression(C = 2.0))
    parameters = amos.utilities.parameters_of(wrapped)
    assert list(parameters) == ['estimator']
    chosen = amos.interface._chosen_parameters(parameters)
    assert chosen['C'] == 2.0
    assert 'estimator' not in chosen


@pytest.mark.parametrize('file_format', ['json', 'ini', 'toml', 'yaml', 'py'])
def test_export_replication_in_each_format(file_format: str) -> None:
    if file_format == 'toml':
        requires('tomli_w')
    if file_format == 'yaml':
        requires('yaml')
    project = amos.Project.create(
        _settings(), item = make_numeric(), id = 'run')
    path = project.export_replication(file_format = file_format)
    assert path == (
        pathlib.Path(project.clerk.output_folder) / 'run'
        / f'study_replication.{file_format}')
    again = amos.Project.create(path, item = make_numeric())
    assert 'design' not in again.idea['analyst']
    pd.testing.assert_series_equal(
        again.result.predictions, project.result.predictions)


def test_export_replication_takes_a_folder_and_a_name(
    tmp_path: pathlib.Path) -> None:
    requires('yaml')
    project = amos.Project.create(_settings(), item = make_numeric())
    path = project.export_replication(
        tmp_path / 'shared', file_name = 'winner', file_format = '.YAML')
    assert path == tmp_path / 'shared' / 'winner.yaml'
    assert path.is_file()


def test_export_saves_the_replication_in_the_format_of_the_settings() -> None:
    settings = _settings(files = {'replication_format': 'ini'})
    project = amos.Project.create(settings, item = make_numeric(), id = 'run')
    folder = project.export()
    assert (folder / 'study_replication.ini').is_file()
    assert not (folder / 'study_replication.json').exists()
    # A format passed to `export` takes precedence over the settings.
    folder = project.export(replication_format = 'py')
    assert (folder / 'study_replication.py').is_file()


def test_export_replication_in_an_unknown_format_raises() -> None:
    project = amos.Project.create(_settings(), item = make_numeric())
    with pytest.raises(ValueError, match = "cannot be saved as 'xml'"):
        project.export_replication(file_format = 'xml')


def test_export_replication_names_the_extra_for_a_missing_writer(
    monkeypatch: pytest.MonkeyPatch) -> None:
    project = amos.Project.create(_settings(), item = make_numeric())
    # A `None` in `sys.modules` makes importing the package fail.
    monkeypatch.setitem(sys.modules, 'tomli_w', None)
    with pytest.raises(ImportError, match = r'amos\[settings\]'):
        project.export_replication(file_format = 'toml')
