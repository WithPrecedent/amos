"""Tests the Scorecard of the critic stage and its exports."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

import pandas as pd
import pytest
from conftest import SEED, make_numeric, requires

import amos
import chrisjen
from amos.evaluators import Scorecard


def _settings(**analyst: Any) -> dict[str, Any]:
    """Returns settings for a study with four branches."""
    return {
        'general': {'label': 'target', 'seed': SEED},
        'study_project': {'study_workers': 'analyst, critic'},
        'analyst': {
            'design': 'experiment',
            'criterion': 'roc_auc',
            'steps': 'split, scale, model',
            'split_techniques': 'stratified',
            'scale_techniques': 'standard, none',
            'model_techniques': 'sk_logit, baseline',
            **analyst},
        'critic': {'techniques': 'scorecard'}}


@pytest.fixture
def project(tmp_path: pathlib.Path) -> amos.Project:
    """An applied project whose experiment has four branches."""
    os.chdir(tmp_path)
    return amos.Project.create(_settings(), item = make_numeric(), id = 'run')


def test_scorecard_has_a_row_for_each_branch(project: amos.Project) -> None:
    table = project.result.tables['scorecard']
    assert list(table.columns) == [
        'rank', 'split', 'scale', 'model', 'roc_auc', 'accuracy',
        'balanced_accuracy', 'precision', 'recall', 'f1', 'log_loss']
    assert list(table['rank']) == [1, 2, 3, 4]
    assert set(table['scale']) == {'standard', 'none'}
    assert set(table['split']) == {'stratified'}
    # The best branch is first, and its scores are the final model's.
    assert table['roc_auc'].is_monotonic_decreasing
    assert table.loc[0, 'model'] == 'sk_logit'
    assert table.loc[0, 'f1'] == pytest.approx(project.result.metrics['f1'])
    assert list(table['model'].iloc[2:]) == ['baseline', 'baseline']


def test_scorecard_matches_the_experiment(project: amos.Project) -> None:
    analyst = {node.name: node for node in project.workflow}['analyst']
    table = project.result.tables['scorecard']
    expected = sorted(analyst.scores.values(), reverse = True)
    assert list(table['roc_auc']) == pytest.approx(expected)
    assert [b.label for b in project.result.branches] == list(analyst.results)
    branch = project.result.branches[0]
    assert list(branch.steps) == ['split', 'scale', 'model']
    assert branch.steps['scale'] in {'standard', 'none'}
    assert branch.criterion == 'roc_auc'
    assert list(branch.result.data.columns) == ['target']


def test_a_lower_is_better_criterion_ranks_from_lowest(
    tmp_path: pathlib.Path) -> None:
    os.chdir(tmp_path)
    project = amos.Project.create(
        _settings(criterion = 'log_loss'), item = make_numeric())
    table = project.result.tables['scorecard']
    assert list(table.columns[4:6]) == ['log_loss', 'accuracy']
    assert table['log_loss'].is_monotonic_increasing


def test_metrics_and_digits_can_be_set(tmp_path: pathlib.Path) -> None:
    os.chdir(tmp_path)
    settings = _settings()
    settings['critic']['scorecard_digits'] = 2
    settings['scorecard_parameters'] = {'metrics': ['f1', 'accuracy']}
    project = amos.Project.create(settings, item = make_numeric())
    scorecard = project.scorecard
    assert scorecard.digits == 2
    # The criterion that ranked the branches is kept after the metrics.
    assert list(scorecard.table.columns[4:]) == ['f1', 'accuracy', 'roc_auc']
    assert '| 0.' in scorecard.to_markdown()
    assert '0.000' not in scorecard.to_markdown()


def test_a_dataset_without_branches_has_one_row(fitted: amos.Dataset) -> None:
    amos.transformers.Standard().apply(fitted)
    scorecard = Scorecard.create(fitted)
    table = scorecard.table
    assert len(table) == 1
    assert list(table.columns[:4]) == ['rank', 'splitter', 'scaler', 'model']
    assert table.loc[0, 'model'] == 'sk_logit'
    assert scorecard.heading == 'The final model'


def test_the_scorecard_technique_stores_the_final_metrics(
    fitted: amos.Dataset) -> None:
    Scorecard().apply(fitted, metrics = ['accuracy', 'roc_auc'])
    assert set(fitted.metrics) == {'accuracy', 'roc_auc'}
    assert fitted.history[-1] == {'technique': 'scorecard', 'table': 'scorecard'}


def test_metrics_that_need_probabilities_are_skipped(
    fitted: amos.Dataset) -> None:
    fitted.probabilities = None
    table = Scorecard.create(fitted).table
    assert 'roc_auc' not in table.columns
    assert 'accuracy' in table.columns


def test_regression_scorecard(fitted_regression: amos.Dataset) -> None:
    table = Scorecard.create(fitted_regression).table
    assert list(table.columns[-3:]) == ['r2', 'rmse', 'mae']


def test_only_metrics_can_be_scored(fitted: amos.Dataset) -> None:
    with pytest.raises(KeyError):
        Scorecard.create(fitted, metrics = ['standard'])


def test_branches_built_in_code_are_named_by_genre(
    classified: amos.Dataset) -> None:
    experiment = amos.Experiment(
        name = 'models', criteria = amos.metrics.Accuracy())
    experiment.populate([
        [amos.transformers.Standard(), chrisjen.NullVertex()],
        [amos.models.Baseline(), amos.models.SkLogit()]])
    result = experiment.apply(classified)
    table = Scorecard.create(result).table
    assert list(table.columns[:3]) == ['rank', 'scaler', 'model']
    assert set(table['scaler']) == {'standard', 'none'}
    assert Scorecard.create(result).heading == '4 branches ranked by accuracy'


def test_other_criteria_add_a_score_column(classified: amos.Dataset) -> None:
    def accuracy(item: amos.Dataset) -> float:
        return amos.metrics.Accuracy().measure(item)

    experiment = amos.Experiment(
        name = 'models', criteria = chrisjen.Criteria(contents = accuracy))
    experiment.populate([[amos.models.Baseline(), amos.models.SkLogit()]])
    scorecard = Scorecard.create(experiment.apply(classified))
    assert scorecard.table.columns[-1] == 'score'
    assert scorecard.table['score'].is_monotonic_decreasing
    assert scorecard.table.loc[0, 'model'] == 'sk_logit'
    assert scorecard.heading == '2 branches'


def test_create_needs_a_dataset_or_an_applied_project() -> None:
    with pytest.raises(TypeError, match = 'needs a Dataset'):
        Scorecard.create(42)
    with pytest.raises(ValueError, match = 'empty'):
        Scorecard().to_markdown()
    project = amos.Project.create(_settings())
    with pytest.raises(ValueError, match = 'apply the project'):
        _ = project.scorecard


def test_to_csv(project: amos.Project, tmp_path: pathlib.Path) -> None:
    scorecard = project.scorecard
    text = scorecard.to_csv(tmp_path / 'card.csv')
    read = pd.read_csv(tmp_path / 'card.csv')
    assert text.splitlines()[0].startswith('rank,split,scale,model,roc_auc')
    pd.testing.assert_frame_equal(read, scorecard.table, check_dtype = False)


def test_to_markdown(project: amos.Project, tmp_path: pathlib.Path) -> None:
    text = project.scorecard.to_markdown(tmp_path / 'card.md')
    lines = text.splitlines()
    assert lines[0].startswith('| rank | split | scale | model | roc_auc |')
    assert lines[1].startswith('| ---: | --- | --- | --- | ---: |')
    assert len(lines) == 2 + 4
    assert lines[2].startswith('| 1 | stratified | ')
    assert lines[2].count('|') == 12
    assert (tmp_path / 'card.md').read_text(encoding = 'utf-8') == text


def test_markdown_escapes_pipes(fitted: amos.Dataset) -> None:
    scorecard = Scorecard.create(fitted)
    scorecard.table.loc[0, 'model'] = 'a|b'
    assert '| a\\|b |' in scorecard.to_markdown()


def test_to_word(project: amos.Project, tmp_path: pathlib.Path) -> None:
    requires('docx')
    docx = pytest.importorskip('docx')
    path = project.scorecard.to_word(tmp_path / 'card.docx')
    document = docx.Document(str(path))
    assert document.paragraphs[0].text == '4 branches ranked by roc_auc'
    table = document.tables[0]
    assert len(table.rows) == 5
    assert [c.text for c in table.rows[0].cells][:4] == [
        'rank', 'split', 'scale', 'model']
    assert table.rows[1].cells[0].text == '1'


def test_to_image(project: amos.Project, tmp_path: pathlib.Path) -> None:
    requires('matplotlib')
    scorecard = project.scorecard
    path = scorecard.to_image(tmp_path / 'card.png', dpi = 72)
    assert path.read_bytes().startswith(b'\x89PNG')
    scorecard.to_image(tmp_path / 'card.svg')
    assert '<svg' in (tmp_path / 'card.svg').read_text(encoding = 'utf-8')
    figure = scorecard.to_figure()
    assert figure.texts[0].get_text() == '4 branches ranked by roc_auc'


def test_title(project: amos.Project) -> None:
    scorecard = Scorecard.create(project, title = 'Table 2')
    assert scorecard.heading == 'Table 2'
    assert scorecard.to_figure().texts[0].get_text() == 'Table 2'


def test_export(project: amos.Project, tmp_path: pathlib.Path) -> None:
    requires('docx')
    requires('matplotlib')
    paths = project.scorecard.export(tmp_path / 'cards', name = 'table_2')
    assert sorted(p.name for p in paths.values()) == [
        'table_2.csv', 'table_2.docx', 'table_2.md', 'table_2.png']
    assert all(path.is_file() for path in paths.values())
    paths = project.scorecard.export(tmp_path, formats = ['pdf'])
    assert paths['pdf'].name == 'scorecard.pdf'


def test_project_export_includes_the_scorecard(project: amos.Project) -> None:
    requires('docx')
    requires('matplotlib')
    folder = project.export()
    names = {path.name for path in folder.iterdir()}
    assert {
        'scorecard.csv', 'scorecard.md', 'scorecard.docx',
        'scorecard.png'} <= names


def test_word_needs_python_docx(
    project: amos.Project,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch) -> None:
    # A `None` in `sys.modules` makes importing that module fail.
    monkeypatch.setitem(sys.modules, 'docx', None)
    with pytest.raises(ImportError, match = r'amos\[word\]'):
        project.scorecard.to_word(tmp_path / 'card.docx')


def test_to_latex(project: amos.Project, tmp_path: pathlib.Path) -> None:
    text = project.scorecard.to_latex(tmp_path / 'card.tex')
    lines = text.splitlines()
    assert lines[0] == r'\begin{table}[htbp]'
    assert lines[2] == r'\caption{4 branches ranked by roc\_auc}'
    assert lines[3].startswith(r'\begin{tabular}{rlll')
    assert r'rank & split & scale & model & roc\_auc' in lines[5]
    assert lines[-1] == r'\end{table}'
    assert (tmp_path / 'card.tex').read_text(encoding = 'utf-8') == text


def test_to_html(project: amos.Project, tmp_path: pathlib.Path) -> None:
    requires('great_tables')
    html = project.scorecard.to_html(tmp_path / 'card.html')
    assert '4 branches ranked by roc_auc' in html
    assert '<table' in html
    assert 'stratified' in html
    assert (tmp_path / 'card.html').read_text(encoding = 'utf-8') == html


def test_export_latex_and_html(
    project: amos.Project,
    tmp_path: pathlib.Path) -> None:
    requires('great_tables')
    paths = project.scorecard.export(tmp_path, formats = ['tex', 'html'])
    assert paths['tex'].read_text(encoding = 'utf-8').startswith(r'\begin')
    assert paths['html'].is_file()
