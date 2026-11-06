"""Tests the figures that explorers and critics draw of their tables.

Every describer, evaluator, and inferer takes `visualize`, and the
"visualize_explorers" and "visualize_critics" settings of the "general"
section (the dataset's fields of the same names) replace it if they are set.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, make_numeric, requires, techniques

import amos
from amos import base, describers, evaluators, inferers, plots

# The parameters of each describer that the tests apply, by its name.
DESCRIBERS: dict[str, dict[str, Any]] = {
    'association_rules': {'min_support': 0.05, 'min_threshold': 0.5},
    'correlations': {},
    'describe': {},
    'frequencies': {},
    'label_balance': {},
    'missing_values': {},
    'summarize': {}}
# The packages and parameters of each evaluator that the tests apply.
EVALUATORS: dict[str, tuple[str | None, dict[str, Any]]] = {
    'bias_variance': ('mlxtend', {'rounds': 5}),
    'classification_report': (None, {}),
    'confusion': (None, {}),
    'conformal': ('mapie', {}),
    'explain_weights': ('eli5', {}),
    'factor_analysis': ('statsmodels', {'factors': 2}),
    'fairness': ('fairlearn', {}),
    'feature_importance': (None, {}),
    'pca': (None, {}),
    'permutation_importance': (None, {'n_repeats': 2}),
    'regression_table': (None, {'simulations': 5}),
    'scorecard': (None, {}),
    'shap_importance': ('shap', {'rows': 30, 'background': 30})}


@pytest.fixture(autouse = True)
def _plotting() -> None:
    """Skips the tests if the plotting packages cannot be imported."""
    requires('matplotlib')
    requires('seaborn')


def _drawn(item: amos.Dataset, name: str, tmp_path: pathlib.Path) -> None:
    """Checks that `name` drew a figure that can be saved, and recorded it."""
    figure = item.figures[name]
    assert figure.axes
    path = tmp_path / f'{name}.png'
    figure.savefig(path)
    assert path.stat().st_size > 0
    entry = [e for e in item.history if e['technique'] == name][-1]
    assert entry['figure'] == name
    assert 'figure_error' not in entry


def _grouped() -> amos.Dataset:
    """Returns a split classification dataset with a group, fitted by a logit.

    Returns:
        The dataset.

    """
    data = make_numeric()
    data['court'] = np.where(data['x0'] > 0, 'north', 'south')
    dataset = amos.Dataset(
        data, label = 'target', seed = SEED, groups = ['court'])
    amos.splitters.Stratified().apply(dataset)
    return amos.models.Logit().apply(dataset)


def test_every_explorer_and_critic_has_a_visual() -> None:
    names = [
        name for genre in ('describer', 'evaluator', 'inferer')
        for name, _ in techniques(genre)]
    plot_names = {name for name, _ in techniques('plot')}
    for name in names:
        visual = amos.library.borrow(name).visual
        assert visual in plot_names, name


@pytest.mark.parametrize('name', list(DESCRIBERS))
def test_each_explorer_draws_its_table(
    name: str,
    mixed: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    if name == 'association_rules':
        requires('mlxtend')
    amos.library.borrow(name)().apply(
        mixed, visualize = True, **DESCRIBERS[name])
    _drawn(mixed, name, tmp_path)


@pytest.mark.parametrize('name', list(EVALUATORS))
def test_each_evaluator_draws_its_table(
    name: str,
    tmp_path: pathlib.Path) -> None:
    package, parameters = EVALUATORS[name]
    if package is not None:
        requires(package)
    dataset = _grouped()
    amos.library.borrow(name)().apply(
        dataset, visualize = True, **parameters)
    _drawn(dataset, name, tmp_path)


def test_kaplan_meier_draws_the_curves_of_its_groups(
    tmp_path: pathlib.Path) -> None:
    requires('statsmodels')
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame({
        'days': rng.exponential(100, 120),
        'arm': rng.choice(['a', 'b'], 120)})
    dataset = amos.Dataset(data, label = 'days', seed = SEED)
    describers.KaplanMeier().apply(dataset, group = 'arm', visualize = True)
    _drawn(dataset, 'kaplan_meier', tmp_path)
    labels = dataset.figures['kaplan_meier'].axes[0].get_legend_handles_labels()
    assert labels[1] == ['a', 'b']


def test_an_effect_is_drawn_as_a_coefficient_plot(
    tmp_path: pathlib.Path) -> None:
    requires('doubleml')
    rng = np.random.default_rng(SEED)
    age = rng.normal(40, 10, 300)
    treated = (rng.random(300) < 0.5).astype(int)
    data = pd.DataFrame({
        'age': age,
        'treated': treated,
        'outcome': 2 * treated + 0.1 * age + rng.normal(size = 300)})
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    inferers.PartiallyLinear().apply(
        dataset, treatment = 'treated', outcome_model = 'linear',
        treatment_model = 'logit', n_folds = 3, visualize = True)
    _drawn(dataset, 'partially_linear', tmp_path)
    axes = dataset.figures['partially_linear'].axes[0]
    assert [t.get_text() for t in axes.get_yticklabels()] == ['treated']


def test_causal_links_are_drawn_as_a_graph(tmp_path: pathlib.Path) -> None:
    requires('tigramite')
    requires('networkx')
    rng = np.random.default_rng(2)
    weeks = np.zeros((200, 2))
    for week in range(1, 200):
        weeks[week, 0] = 0.7 * weeks[week - 1, 0] + rng.normal()
        weeks[week, 1] = 0.6 * weeks[week - 1, 0] + rng.normal()
    data = pd.DataFrame(weeks, columns = ['filings', 'hearings'])
    dataset = amos.Dataset(data, label = 'hearings', seed = SEED)
    inferers.PCMCIPlus().apply(
        dataset, max_lag = 2, alpha = 0.01, visualize = True)
    _drawn(dataset, 'pcmci_plus', tmp_path)
    texts = {
        t.get_text() for t in dataset.figures['pcmci_plus'].axes[0].texts}
    assert {'filings', 'hearings'} <= texts


def test_nothing_is_drawn_by_default(fitted: amos.Dataset) -> None:
    evaluators.Confusion().apply(fitted)
    assert 'confusion' not in fitted.figures
    assert 'figure' not in fitted.history[-1]


def test_visualize_can_be_a_parameter_in_the_settings(
    fitted: amos.Dataset) -> None:
    evaluators.Confusion(parameters = {'visualize': 'true'}).apply(fitted)
    assert 'confusion' in fitted.figures


def test_visualize_must_be_true_or_false(fitted: amos.Dataset) -> None:
    with pytest.raises(TypeError, match = 'visualize must be true or false'):
        evaluators.Confusion().apply(fitted, visualize = 'sometimes')


@pytest.mark.parametrize(('setting', 'argument', 'drawn'), [
    (None, False, False),
    (None, True, True),
    (True, False, True),
    (False, True, False)])
def test_the_setting_of_a_stage_replaces_the_argument(
    setting: bool | None,
    argument: bool,
    drawn: bool,
    fitted: amos.Dataset) -> None:
    fitted.visualize_critics = setting
    fitted.visualize_explorers = not drawn
    evaluators.Confusion().apply(fitted, visualize = argument)
    assert ('confusion' in fitted.figures) is drawn
    fitted.visualize_explorers = setting
    fitted.visualize_critics = not drawn
    describers.MissingValues().apply(fitted, visualize = argument)
    assert ('missing_values' in fitted.figures) is drawn


def test_the_settings_of_a_stage_are_not_set_by_default() -> None:
    dataset = amos.Dataset()
    assert dataset.visualize_explorers is None
    assert dataset.visualize_critics is None


@pytest.mark.parametrize(('value', 'expected'), [
    (True, True), (False, False), ('True', True), ('false', False),
    (np.bool_(True), True), (None, None)])
def test_the_settings_of_a_stage_are_true_or_false(
    value: Any,
    expected: bool | None) -> None:
    dataset = amos.Dataset(
        visualize_explorers = value, visualize_critics = value)
    assert dataset.visualize_explorers is expected
    assert dataset.visualize_critics is expected


def test_the_settings_of_a_stage_must_be_true_or_false() -> None:
    with pytest.raises(TypeError, match = 'visualize_critics must be true'):
        amos.Dataset(visualize_critics = 'often')


def test_create_replaces_the_settings_of_a_stage_if_they_are_passed(
    mixed: amos.Dataset) -> None:
    mixed.visualize_critics = True
    same = amos.Dataset.create(mixed, visualize_explorers = 'true')
    assert same.visualize_explorers is True
    assert same.visualize_critics is True
    made = amos.Dataset.create(make_numeric(), visualize_critics = False)
    assert made.visualize_critics is False
    assert made.visualize_explorers is None


def test_a_figure_that_fails_keeps_the_table(
    fitted: amos.Dataset,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        message = 'no room to draw'
        raise RuntimeError(message)

    monkeypatch.setattr(plots.ConfusionHeatmap, 'draw', fail)
    with caplog.at_level(logging.WARNING, logger = base.__name__):
        evaluators.Confusion().apply(fitted, visualize = True)
    assert 'confusion' in fitted.tables
    assert 'confusion' not in fitted.figures
    assert fitted.history[-1]['figure_error'] == 'no room to draw'
    assert 'figure' not in fitted.history[-1]
    assert "the figure of 'confusion' was not drawn" in caplog.text


def test_a_figure_has_the_style_of_every_plot(fitted: amos.Dataset) -> None:
    evaluators.Confusion().apply(fitted, visualize = True)
    plots.ConfusionHeatmap().apply(fitted)
    visual = fitted.figures['confusion']
    plot = fitted.figures['confusion_heatmap']
    assert list(visual.get_size_inches()) == list(plot.get_size_inches())
    # The figure of the table is not a step of its own in the history.
    assert [e['technique'] for e in fitted.history].count('confusion') == 1


def _project(general: dict[str, Any]) -> amos.Project:
    """Returns an applied project with an explorer and a critic.

    Args:
        general: settings of the "general" section, besides the label and
            seed.

    Returns:
        The project.

    """
    settings = {
        'general': {'label': 'target', 'seed': SEED, **general},
        'study_project': {'study_workers': 'explorer, analyst, critic'},
        'explorer': {'techniques': 'summarize, label_balance'},
        'analyst': {'steps': 'split, model', 'split_techniques': 'stratified',
                    'model_techniques': 'logit'},
        'critic': {'techniques': 'confusion'}}
    return amos.Project.create(settings, item = make_numeric())


def test_the_general_section_turns_on_the_figures_of_a_stage() -> None:
    project = _project({'visualize_explorers': True})
    figures = project.result.figures
    assert {'summarize', 'label_balance'} <= set(figures)
    assert 'confusion' not in figures
    project = _project({'visualize_critics': 'true'})
    assert set(project.result.figures) == {'confusion'}


def test_the_general_section_turns_off_the_figures_of_a_stage() -> None:
    settings = {
        'general': {
            'label': 'target', 'seed': SEED, 'visualize_explorers': False},
        'study_project': {'study_workers': 'explorer'},
        'explorer': {'techniques': 'summarize'},
        'summarize_parameters': {'visualize': True}}
    project = amos.Project.create(settings, item = make_numeric())
    assert project.result.figures == {}
    settings['general'].pop('visualize_explorers')
    project = amos.Project.create(settings, item = make_numeric())
    assert set(project.result.figures) == {'summarize'}
