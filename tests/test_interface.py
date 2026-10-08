"""Tests the interface module (Project) and the reports module."""

from __future__ import annotations

import importlib
import json
import os
import pathlib
from typing import Any

import pandas as pd
import pytest
from conftest import SEED, make_numeric, requires

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
            'model_techniques': 'sk_logit, baseline'},
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
        '[study_project]\nstudy_techniques = stratified, sk_logit\n',
        encoding = 'utf-8')
    project = amos.Project.create('study.ini', item = 'cases.csv')
    assert project.result.model is not None
    assert pathlib.Path(project.clerk.output_folder).name == 'results'


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
    assert folder == tmp_path / 'run'
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
        'tables/analyst_comparison.csv',
        'tables/confusion.csv',
        'tables/scorecard.csv',
        'tables/summarize.csv']
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


def test_export_after_each_apply_if_the_settings_say_so(
    tmp_path: pathlib.Path) -> None:
    settings = _settings(files = {'export_results': True})
    amos.Project.create(settings, item = make_numeric(), id = 'auto')
    assert (tmp_path / 'auto' / 'report.txt').is_file()


def test_export_before_apply_raises() -> None:
    project = amos.Project.create(_settings())
    with pytest.raises(ValueError, match = 'apply the project'):
        project.export()
