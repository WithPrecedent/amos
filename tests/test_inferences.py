"""Tests the inferences module (DoubleML, DoWhy, causalml, and tigramite)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, package_of, requires, techniques

import amos
from amos import inferences

FAST = {'outcome_model': 'linear', 'treatment_model': 'logit', 'n_folds': 3}
COLUMNS = [
    'coefficient', 'standard_error', 'statistic', 'p_value', 'ci_lower',
    'ci_upper']


def _treated(rows: int = 400, effect: float = 2.0) -> pd.DataFrame:
    """Returns data in which a binary treatment raises the label by `effect`.

    Older people are more often treated and have higher labels, so a naive
    comparison of the treated and untreated would overstate the effect.
    """
    rng = np.random.default_rng(11)
    age = rng.normal(40, 10, rows)
    income = rng.normal(0, 1, rows)
    treated = (rng.random(rows) < 1 / (1 + np.exp(-(age - 40) / 5))).astype(int)
    outcome = effect * treated + 0.1 * age + income + rng.normal(0, 1, rows)
    return pd.DataFrame({
        'age': age, 'income': income, 'treated': treated, 'outcome': outcome})


def _instrumented(rows: int = 600) -> pd.DataFrame:
    """Returns data in which an instrument moves a treatment worth 2.

    "lottery" (the instrument) changes the treatment but not the label, and
    "need" affects both.
    """
    rng = np.random.default_rng(3)
    need = rng.normal(size = rows)
    other = rng.normal(size = rows)
    lottery = (rng.random(rows) < 0.5).astype(int)
    treated = ((need + 1.2 * lottery + rng.normal(size = rows)) > 0.6)
    outcome = 2 * treated + need + 0.5 * other + rng.normal(size = rows)
    return pd.DataFrame({
        'need': need, 'other': other, 'lottery': lottery,
        'treated': treated.astype(int), 'outcome': outcome})


def _weeks(rows: int = 300) -> pd.DataFrame:
    """Returns a time series in which filings cause hearings and then backlog.

    Filings raise hearings a week later (by 0.6), which raise the backlog two
    weeks after that (by 0.8), so filings raise the backlog by 0.48 three
    weeks later.
    """
    rng = np.random.default_rng(2)
    weeks = np.zeros((rows, 3))
    for week in range(2, rows):
        weeks[week, 0] = 0.7 * weeks[week - 1, 0] + rng.normal()
        weeks[week, 1] = (
            0.5 * weeks[week - 1, 1] + 0.6 * weeks[week - 1, 0] + rng.normal())
        weeks[week, 2] = (
            0.4 * weeks[week - 1, 2] + 0.8 * weeks[week - 2, 1] + rng.normal())
    return pd.DataFrame(weeks, columns = ['filings', 'hearings', 'backlog'])


def _covers(table: pd.DataFrame, row: object, value: float) -> bool:
    """Returns whether the confidence interval of `row` contains `value`."""
    return bool(table.loc[row, 'ci_lower'] < value < table.loc[row, 'ci_upper'])


def test_each_package_is_a_genre_of_inferences() -> None:
    genres = {
        'causalml': {'dr_learner', 's_learner', 't_learner', 'tmle', 'x_learner'},
        'doubleml': {
            'difference_in_differences', 'interactive_iv',
            'interactive_regression', 'partially_linear', 'partially_linear_iv',
            'partially_linear_panel', 'partially_logistic', 'potential_outcomes',
            'quantile_effects', 'regression_discontinuity', 'sample_selection'},
        'dowhy': {
            'distance_matching', 'doubly_robust', 'glm_adjustment',
            'instrumental_variable', 'propensity_matching',
            'propensity_stratification', 'propensity_weighting',
            'regression_adjustment'},
        'tigramite': {'lpcmci', 'pcmci', 'pcmci_plus', 'time_series_effect'}}
    for genre, names in genres.items():
        found = techniques(genre)
        assert {name for name, _ in found} == names, genre
        assert {package_of(kind) for _, kind in found} == {genre}
    assert len(techniques('inference')) == 28


# DoubleML


@pytest.mark.parametrize('kind', [
    inferences.PartiallyLinear, inferences.InteractiveRegression])
def test_doubleml_finds_the_true_effect(kind: type[amos.Inference]) -> None:
    requires('doubleml')
    dataset = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    kind().apply(dataset, treatment = 'treated', **FAST)
    table = dataset.tables[kind().name]
    assert list(table.columns) == COLUMNS
    assert _covers(table, 'treated', 2.0)
    record = dataset.history[-1]
    assert record['treatment'] == 'treated'
    assert record['effect'] == pytest.approx(table.loc['treated', 'coefficient'])
    assert kind().name in dataset.fitted


@pytest.mark.parametrize(('kind', 'treatment_model'), [
    (inferences.PartiallyLinearIV, 'linear'),
    (inferences.InteractiveIV, 'logit')])
def test_doubleml_finds_the_effect_through_an_instrument(
    kind: type[amos.Inference], treatment_model: str) -> None:
    requires('doubleml')
    dataset = amos.Dataset(_instrumented(), label = 'outcome', seed = SEED)
    kind().apply(
        dataset, treatment = 'treated', instrument = 'lottery',
        outcome_model = 'linear', treatment_model = treatment_model,
        n_folds = 3)
    assert _covers(dataset.tables[kind().name], 'treated', 2.0)


def test_doubleml_instrumented_inferences_need_an_instrument() -> None:
    requires('doubleml')
    dataset = amos.Dataset(_instrumented(), label = 'outcome', seed = SEED)
    with pytest.raises(ValueError, match = 'instrument'):
        inferences.InteractiveIV().apply(dataset, treatment = 'treated', **FAST)


def test_interactive_regression_finds_the_effect_on_the_treated() -> None:
    requires('doubleml')
    dataset = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    inferences.InteractiveRegression().apply(
        dataset, treatment = 'treated', score = 'ATTE', **FAST)
    assert _covers(dataset.tables['interactive_regression'], 'treated', 2.0)


def test_partially_logistic_finds_a_rise_in_the_odds() -> None:
    requires('doubleml')
    data = _treated(rows = 600)
    data['outcome'] = (data['outcome'] > data['outcome'].median()).astype(int)
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    inferences.PartiallyLogistic().apply(
        dataset, treatment = 'treated', treatment_model = 'logit', n_folds = 3)
    assert dataset.tables['partially_logistic'].loc['treated', 'ci_lower'] > 0
    # Its outcome model must also regress the log-odds.
    with pytest.raises(ValueError, match = 'cannot regress'):
        inferences.PartiallyLogistic().apply(
            dataset, treatment = 'treated', outcome_model = 'logit')


def test_partially_logistic_needs_a_label_with_two_classes() -> None:
    requires('doubleml')
    dataset = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    with pytest.raises(ValueError, match = 'two classes'):
        inferences.PartiallyLogistic().apply(
            dataset, treatment = 'treated', **FAST)


def test_potential_outcomes_compares_each_level_to_the_reference() -> None:
    requires('doubleml')
    rng = np.random.default_rng(5)
    levels = rng.integers(0, 3, 600)
    need = rng.normal(size = 600)
    data = pd.DataFrame({
        'need': need,
        'sentence': np.array(['fine', 'probation', 'prison'])[levels],
        'outcome': 1.0 * levels + need + rng.normal(size = 600)})
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    inferences.PotentialOutcomes().apply(
        dataset, treatment = 'sentence', outcome_model = 'linear',
        treatment_model = 'logit', n_folds = 3)
    table = dataset.tables['potential_outcomes']
    assert list(table.index) == ['prison vs fine', 'probation vs fine']
    assert _covers(table, 'prison vs fine', 2.0)
    assert _covers(table, 'probation vs fine', 1.0)
    inferences.PotentialOutcomes().apply(
        dataset, treatment = 'sentence', reference = 'probation',
        outcome_model = 'linear', treatment_model = 'logit', n_folds = 3)
    table = dataset.tables['potential_outcomes']
    assert list(table.index) == ['fine vs probation', 'prison vs probation']
    with pytest.raises(ValueError, match = 'levels'):
        inferences.PotentialOutcomes().apply(
            dataset, treatment = 'sentence', reference = 'parole', **FAST)


def test_quantile_effects_has_a_row_for_each_quantile() -> None:
    requires('doubleml')
    dataset = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    inferences.QuantileEffects().apply(
        dataset, treatment = 'treated', outcome_model = 'logit',
        treatment_model = 'logit', n_folds = 3)
    table = dataset.tables['quantile_effects']
    assert list(table.index) == [0.25, 0.5, 0.75]
    assert table.index.name == 'quantile'
    assert (table['coefficient'] > 0).all()
    inferences.QuantileEffects().apply(
        dataset, treatment = 'treated', quantiles = [0.5], score = 'cvar',
        **FAST)
    assert list(dataset.tables['quantile_effects'].index) == [0.5]
    with pytest.raises(ValueError, match = 'score'):
        inferences.QuantileEffects().apply(
            dataset, treatment = 'treated', score = 'median', **FAST)
    with pytest.raises(ValueError, match = 'instrument'):
        inferences.QuantileEffects().apply(
            dataset, treatment = 'treated', score = 'local_quantile', **FAST)


def test_difference_in_differences_finds_the_effect_on_the_treated() -> None:
    requires('doubleml')
    rng = np.random.default_rng(7)
    need = rng.normal(size = 800)
    group = (need + rng.normal(size = 800) > 0).astype(int)
    after = (rng.random(800) < 0.5).astype(int)
    outcome = (
        after + 0.8 * group + 1.5 * group * after + 0.5 * need
        + rng.normal(size = 800))
    # Repeated cross sections in two years: different rows in each.
    data = pd.DataFrame({
        'need': need, 'group': group, 'year': 2019 + after,
        'outcome': outcome})
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    inferences.DifferenceInDifferences().apply(
        dataset, treatment = 'group', time = 'year', **FAST)
    assert _covers(dataset.tables['difference_in_differences'], 'group', 1.5)
    periods = dataset.tables['difference_in_differences_periods']
    assert list(periods.index) == [0.0]
    with pytest.raises(ValueError, match = 'time'):
        inferences.DifferenceInDifferences().apply(
            dataset, treatment = 'group', **FAST)
    with pytest.raises(ValueError, match = 'comparison'):
        inferences.DifferenceInDifferences().apply(
            dataset, treatment = 'group', time = 'year',
            comparison = 'everyone', **FAST)


def test_difference_in_differences_with_staggered_treatments() -> None:
    requires('doubleml')
    rng = np.random.default_rng(8)
    units, years = 300, 5
    first = rng.choice([3.0, 4.0, np.nan], units)
    need = rng.normal(size = units)
    rows = [
        (f'court {u}', year, first[u], need[u],
         0.5 * year + need[u] + 1.5 * (year >= first[u]) + rng.normal())
        for u in range(units) for year in range(1, years + 1)]
    panel = pd.DataFrame(
        rows, columns = ['court', 'year', 'first', 'need', 'outcome'])
    dataset = amos.Dataset(panel, label = 'outcome', seed = SEED)
    inferences.DifferenceInDifferences().apply(
        dataset, treatment = 'first', time = 'year', unit = 'court', **FAST)
    assert _covers(dataset.tables['difference_in_differences'], 'first', 1.5)
    periods = dataset.tables['difference_in_differences_periods']
    assert list(periods.index) == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert periods.index.name == 'periods_since'
    # Before the treatment, there is no effect.
    assert _covers(periods, -1.0, 0.0)
    assert dataset.history[-1]['unit'] == 'court'
    inferences.DifferenceInDifferences().apply(
        dataset, treatment = 'first', time = 'year', unit = 'court',
        comparison = 'not_yet_treated', **FAST)
    assert _covers(dataset.tables['difference_in_differences'], 'first', 1.5)


@pytest.mark.parametrize('fuzzy', [False, True])
def test_regression_discontinuity_finds_the_effect_at_the_cutoff(
    fuzzy: bool) -> None:  # noqa: FBT001
    requires('doubleml')
    requires('rdrobust')
    rng = np.random.default_rng(10)
    score = rng.uniform(40, 60, 1000)
    need = rng.normal(size = 1000)
    above = score >= 50
    if fuzzy:
        chance = np.where(above, 0.8, 0.2)
        treated = (rng.random(1000) < chance).astype(int)
    else:
        treated = above.astype(int)
    data = pd.DataFrame({
        'score': score, 'need': need, 'treated': treated,
        'outcome': (
            2 * treated + 0.1 * score + 0.5 * need
            + rng.normal(0, 0.5, 1000))})
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    inferences.RegressionDiscontinuity().apply(
        dataset, treatment = 'treated', running = 'score', cutoff = 50, **FAST)
    table = dataset.tables['regression_discontinuity']
    assert list(table.columns) == COLUMNS
    assert _covers(table, 'treated', 2.0)
    assert dataset.fitted['regression_discontinuity'].fuzzy is fuzzy
    with pytest.raises(ValueError, match = 'running'):
        inferences.RegressionDiscontinuity().apply(
            dataset, treatment = 'treated', **FAST)


def test_sample_selection_uses_only_the_seen_labels() -> None:
    requires('doubleml')
    data = _instrumented().drop(columns = 'lottery')
    rng = np.random.default_rng(9)
    seen = ((data['other'] + rng.normal(size = len(data))) > -0.3).astype(int)
    data['seen'] = seen
    data['outcome'] = data['outcome'].where(seen == 1)
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    inferences.SampleSelection().apply(
        dataset, treatment = 'treated', selected = 'seen', **FAST)
    assert _covers(dataset.tables['sample_selection'], 'treated', 2.0)
    with pytest.raises(ValueError, match = 'selected'):
        inferences.SampleSelection().apply(
            dataset, treatment = 'treated', **FAST)


def test_partially_linear_panel_removes_each_units_own_level() -> None:
    requires('doubleml')
    rng = np.random.default_rng(4)
    units, periods = 120, 4
    unit = np.repeat(np.arange(units), periods)
    level = rng.normal(size = units)[unit]
    need = rng.normal(size = units * periods) + level
    dose = 0.5 * need + rng.normal(size = units * periods)
    data = pd.DataFrame({
        'court': [f'court {u}' for u in unit],
        'year': np.tile(np.arange(periods), units),
        'need': need,
        'dose': dose,
        'outcome': 1.2 * dose + need + level + rng.normal(size = units * periods)})
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    inferences.PartiallyLinearPanel().apply(
        dataset, treatment = 'dose', unit = 'court', time = 'year',
        outcome_model = 'linear', treatment_model = 'linear', n_folds = 3)
    table = dataset.tables['partially_linear_panel']
    assert list(table.index) == ['dose']
    assert _covers(table, 'dose', 1.2)


def test_doubleml_is_reproducible() -> None:
    requires('doubleml')
    first = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    second = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    for dataset in (first, second):
        inferences.PartiallyLinear().apply(
            dataset, treatment = 'treated', n_folds = 3,
            outcome_model = 'random_forest')
    pd.testing.assert_frame_equal(
        first.tables['partially_linear'], second.tables['partially_linear'])


def test_a_classified_label_and_text_treatment() -> None:
    requires('doubleml')
    data = _treated()
    data['outcome'] = (data['outcome'] > data['outcome'].median()).astype(int)
    data['treated'] = data['treated'].map({0: 'control', 1: 'program'})
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    inferences.InteractiveRegression().apply(
        dataset, treatment = 'treated', outcome_model = 'logit', n_folds = 3)
    effect = dataset.tables['interactive_regression'].loc['treated', 'coefficient']
    # The program raises the chance of a high outcome.
    assert 0 < effect < 1


def test_learners_are_amos_models() -> None:
    forest = inferences._learner('random_forest', 'classify', SEED)
    assert type(forest).__name__ == 'RandomForestClassifier'
    assert forest.random_state == SEED
    logit = inferences._learner('logit', 'classify', None)
    assert logit.max_iter == 1000
    with pytest.raises(ValueError, match = 'cannot regress'):
        inferences._learner('logit', 'regress', SEED)
    with pytest.raises(KeyError, match = 'standard'):
        inferences._learner('standard', 'regress', SEED)


def test_inferences_need_a_treatment() -> None:
    dataset = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    for kind in (
        inferences.PartiallyLinear, inferences.RegressionAdjustment,
        inferences.TLearner, inferences.TimeSeriesEffect):
        with pytest.raises(ValueError, match = 'treatment'):
            kind().apply(dataset)


def test_inferences_need_numbers() -> None:
    requires('doubleml')
    data = _treated()
    data['court'] = 'state'
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    with pytest.raises(ValueError, match = 'court'):
        inferences.PartiallyLinear().apply(dataset, treatment = 'treated')


def test_interactive_regression_needs_two_treatment_values() -> None:
    requires('doubleml')
    dataset = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    with pytest.raises(ValueError, match = 'to have 2 values'):
        inferences.InteractiveRegression().apply(
            dataset, treatment = 'income', n_folds = 2)


# causalml


CAUSALML = [
    inferences.SLearner, inferences.TLearner, inferences.XLearner,
    inferences.DRLearner, inferences.TMLE]


@pytest.mark.parametrize('kind', CAUSALML)
def test_causalml_finds_the_true_effect(kind: type[amos.Inference]) -> None:
    requires('causalml')
    dataset = amos.Dataset(
        _instrumented().drop(columns = 'lottery'), label = 'outcome', seed = SEED)
    kind().apply(
        dataset, treatment = 'treated', outcome_model = 'linear',
        treatment_model = 'logit')
    table = dataset.tables[kind().name]
    assert list(table.columns) == COLUMNS
    # Each row is a group of the treatment, compared to the control.
    assert table.index.name == 'treated'
    assert _covers(table, '1', 2.0)
    if kind is inferences.TMLE:
        assert 'tmle_effects' not in dataset.tables
    else:
        assert len(dataset.tables[f'{kind().name}_effects']) == len(dataset.data)


def test_causalml_compares_each_group_to_the_control() -> None:
    requires('causalml')
    rng = np.random.default_rng(6)
    group = rng.integers(0, 3, 600)
    need = rng.normal(size = 600)
    data = pd.DataFrame({
        'need': need,
        'program': np.array(['none', 'low', 'high'])[group],
        'outcome': 1.0 * group + need + rng.normal(size = 600)})
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    inferences.TLearner().apply(
        dataset, treatment = 'program', control = 'none',
        outcome_model = 'linear')
    table = dataset.tables['t_learner']
    assert sorted(table.index) == ['high', 'low']
    assert _covers(table, 'low', 1.0)
    assert _covers(table, 'high', 2.0)
    with pytest.raises(ValueError, match = 'two values'):
        inferences.TMLE().apply(
            dataset, treatment = 'program', control = 'none',
            outcome_model = 'linear')


def test_causalml_keeps_the_style_of_matplotlib() -> None:
    requires('causalml')
    # causalml sets matplotlib's style to "fivethirtyeight" when it is first
    # imported, so this imports it in a new interpreter.
    code = (
        'import matplotlib, amos; '
        'colors = matplotlib.rcParams["axes.prop_cycle"]; '
        'amos.inferences._import_causalml(); '
        'assert matplotlib.rcParams["axes.prop_cycle"] == colors')
    subprocess.run([sys.executable, '-c', code], check = True)  # noqa: S603


@pytest.mark.skipif(
    sys.version_info < (3, 13), reason = 'causalml installs before 3.13')
def test_causalml_explains_that_it_needs_an_older_python() -> None:
    dataset = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    with pytest.raises(ImportError, match = '3.11 or 3.12'):
        inferences.TLearner().apply(dataset, treatment = 'treated')


# DoWhy


DOWHY = [
    inferences.RegressionAdjustment, inferences.GLMAdjustment,
    inferences.DoublyRobust, inferences.PropensityMatching,
    inferences.PropensityStratification, inferences.PropensityWeighting,
    inferences.DistanceMatching]


@pytest.mark.parametrize('kind', DOWHY)
def test_dowhy_finds_the_true_effect(kind: type[amos.Inference]) -> None:
    requires('dowhy')
    dataset = amos.Dataset(
        _instrumented().drop(columns = 'lottery'), label = 'outcome', seed = SEED)
    kind().apply(dataset, treatment = 'treated', simulations = 20)
    table = dataset.tables[kind().name]
    assert list(table.columns) == COLUMNS
    assert _covers(table, 'treated', 2.0)


def test_dowhy_finds_the_effect_through_an_instrument() -> None:
    requires('dowhy')
    dataset = amos.Dataset(_instrumented(), label = 'outcome', seed = SEED)
    inferences.InstrumentalVariable().apply(
        dataset, treatment = 'treated', instrument = 'lottery',
        simulations = 20)
    assert _covers(dataset.tables['instrumental_variable'], 'treated', 2.0)
    with pytest.raises(ValueError, match = 'instrument'):
        inferences.InstrumentalVariable().apply(dataset, treatment = 'treated')


def test_dowhy_refutes_its_estimates() -> None:
    requires('dowhy')
    dataset = amos.Dataset(
        _instrumented().drop(columns = 'lottery'), label = 'outcome', seed = SEED)
    inferences.RegressionAdjustment().apply(
        dataset, treatment = 'treated', simulations = 10,
        refuters = 'placebo_treatment_refuter, random_common_cause')
    refutations = dataset.tables['regression_adjustment_refutations']
    assert len(refutations) == 2
    with pytest.raises(ValueError, match = 'refuters'):
        inferences.RegressionAdjustment().apply(
            dataset, treatment = 'treated', refuters = 'bootstrap')


def test_dowhy_explains_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, 'dowhy', None)
    dataset = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    with pytest.raises(ImportError, match = 'causal extra'):
        inferences.RegressionAdjustment().apply(dataset, treatment = 'treated')


# tigramite


def test_pcmci_finds_the_true_links() -> None:
    requires('tigramite')
    series = amos.Dataset(_weeks(), label = 'backlog', seed = SEED)
    inferences.PCMCI().apply(series, max_lag = 3, alpha = 0.01)
    table = series.tables['pcmci']
    assert list(table.columns) == [
        'cause', 'effect', 'lag', 'link', 'strength', 'p_value']
    assert table[['cause', 'effect', 'lag']].values.tolist() == [
        ['filings', 'filings', 1], ['filings', 'hearings', 1],
        ['hearings', 'hearings', 1], ['hearings', 'backlog', 2],
        ['backlog', 'backlog', 1]]
    assert series.history[-1]['links'] == 5


@pytest.mark.parametrize('kind', [inferences.PCMCIPlus, inferences.LPCMCI])
def test_other_discoveries_find_the_causes_between_series(
    kind: type[amos.Inference]) -> None:
    requires('tigramite')
    series = amos.Dataset(_weeks(), label = 'backlog', seed = SEED)
    kind().apply(series, max_lag = 3, alpha = 0.01)
    links = series.tables[kind().name][['cause', 'effect', 'lag']].values.tolist()
    assert ['filings', 'hearings', 1] in links
    assert ['hearings', 'backlog', 2] in links


def test_pcmci_uses_other_tests() -> None:
    requires('tigramite')
    series = amos.Dataset(_weeks(), label = 'backlog', seed = SEED)
    inferences.PCMCI().apply(
        series, max_lag = 2, alpha = 0.01, test = 'robust_parcorr')
    links = series.tables['pcmci'][['cause', 'effect', 'lag']].values.tolist()
    assert ['filings', 'hearings', 1] in links
    with pytest.raises(ValueError, match = 'test'):
        inferences.PCMCI().apply(series, test = 'granger')


def test_time_series_effect_finds_the_effect_through_a_chain() -> None:
    requires('tigramite')
    series = amos.Dataset(_weeks(), label = 'backlog', seed = SEED)
    inferences.TimeSeriesEffect().apply(
        series, treatment = 'filings', lag = 3, max_lag = 3, alpha = 0.01,
        simulations = 50)
    table = series.tables['time_series_effect']
    assert list(table.index) == ['filings (lag 3)']
    assert _covers(table, 'filings (lag 3)', 0.6 * 0.8)
    assert 'time_series_effect_links' in series.tables
    with pytest.raises(ValueError, match = 'lag'):
        inferences.TimeSeriesEffect().apply(
            series, treatment = 'filings', lag = 4, max_lag = 3)


def test_time_series_need_no_missing_values() -> None:
    requires('tigramite')
    data = _weeks()
    data.loc[5, 'hearings'] = np.nan
    series = amos.Dataset(data, label = 'backlog', seed = SEED)
    with pytest.raises(ValueError, match = 'missing'):
        inferences.PCMCI().apply(series, max_lag = 2)
