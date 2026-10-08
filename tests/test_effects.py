"""Tests the effects module (DoubleML)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import SEED, requires

import amos
from amos import effects


@pytest.fixture(autouse = True)
def _doubleml() -> None:
    """Skips the tests if DoubleML cannot be imported."""
    requires('doubleml')


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


@pytest.mark.parametrize('kind', [
    effects.PartiallyLinear, effects.InteractiveRegression])
def test_effects_find_the_true_effect(kind: type[amos.Effect]) -> None:
    dataset = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    kind().apply(
        dataset, treatment = 'treated', outcome_model = 'linear',
        treatment_model = 'sk_logit', n_folds = 3)
    table = dataset.tables[kind().name]
    assert list(table.columns) == [
        'coefficient', 'standard_error', 'statistic', 'p_value', 'ci_lower',
        'ci_upper']
    assert table.loc['treated', 'ci_lower'] < 2.0 < table.loc[
        'treated', 'ci_upper']
    record = dataset.history[-1]
    assert record['treatment'] == 'treated'
    assert record['effect'] == pytest.approx(table.loc['treated', 'coefficient'])
    assert kind().name in dataset.fitted


def test_effects_are_reproducible() -> None:
    first = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    second = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    for dataset in (first, second):
        effects.PartiallyLinear().apply(
            dataset, treatment = 'treated', n_folds = 3,
            outcome_model = 'random_forest')
    pd.testing.assert_frame_equal(
        first.tables['partially_linear'], second.tables['partially_linear'])


def test_a_classified_label_and_text_treatment() -> None:
    data = _treated()
    data['outcome'] = (data['outcome'] > data['outcome'].median()).astype(int)
    data['treated'] = data['treated'].map({0: 'control', 1: 'program'})
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    effects.InteractiveRegression().apply(
        dataset, treatment = 'treated', outcome_model = 'sk_logit', n_folds = 3)
    effect = dataset.tables['interactive_regression'].loc['treated', 'coefficient']
    # The program raises the chance of a high outcome.
    assert 0 < effect < 1


def test_learners_are_amos_models() -> None:
    forest = effects._learner('random_forest', 'classify', SEED)
    assert type(forest).__name__ == 'RandomForestClassifier'
    assert forest.random_state == SEED
    logit = effects._learner('sk_logit', 'classify', None)
    assert logit.max_iter == 1000
    with pytest.raises(ValueError, match = 'cannot regress'):
        effects._learner('sk_logit', 'regress', SEED)
    with pytest.raises(KeyError, match = 'standard'):
        effects._learner('standard', 'regress', SEED)


def test_effects_need_a_treatment() -> None:
    dataset = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    with pytest.raises(ValueError, match = 'treatment'):
        effects.PartiallyLinear().apply(dataset)


def test_effects_need_numbers() -> None:
    data = _treated()
    data['court'] = 'state'
    dataset = amos.Dataset(data, label = 'outcome', seed = SEED)
    with pytest.raises(ValueError, match = 'court'):
        effects.PartiallyLinear().apply(dataset, treatment = 'treated')


def test_interactive_regression_needs_two_treatment_values() -> None:
    dataset = amos.Dataset(_treated(), label = 'outcome', seed = SEED)
    with pytest.raises(ValueError, match = 'two values'):
        effects.InteractiveRegression().apply(
            dataset, treatment = 'income', n_folds = 2)
