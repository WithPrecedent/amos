"""Tests the describers module."""

from __future__ import annotations

import pytest
from conftest import requires

import amos
from amos import describers


def test_association_rules(mixed: amos.Dataset) -> None:
    requires('mlxtend')
    describers.AssociationRules().apply(
        mixed, min_support = 0.05, min_threshold = 0.5)
    table = mixed.tables['association_rules']
    assert {'antecedents', 'consequents', 'support', 'confidence',
            'lift'} <= set(table.columns)
    assert table['lift'].is_monotonic_decreasing
    assert (table['confidence'] >= 0.5).all()
    assert (table['support'] >= 0.05).all()
    items = {
        i for side in ('antecedents', 'consequents')
        for rule in table[side] for i in rule.split(', ')}
    # The values of "region", "member" (where it is true), and the label.
    assert 'outcome = yes' in items
    assert 'member' in items
    assert items <= {
        'member', 'outcome = no', 'outcome = yes', 'region = east',
        'region = north', 'region = south', 'region = west'}
    south = table[
        (table['antecedents'] == 'region = south')
        & (table['consequents'] == 'outcome = yes')]
    data = mixed.data
    share = (data.loc[data['region'] == 'south', 'outcome'] == 'yes').mean()
    assert south['confidence'].iloc[0] == pytest.approx(share)


def test_association_rules_of_chosen_columns(mixed: amos.Dataset) -> None:
    requires('mlxtend')
    describers.AssociationRules().apply(
        mixed, columns = ['region', 'outcome'], min_support = 0.05,
        metric = 'lift', min_threshold = 1.2, max_len = 2)
    table = mixed.tables['association_rules']
    assert not table.empty
    assert (table['lift'] >= 1.2).all()
    assert not table['antecedents'].str.contains(', ').any()
    assert 'member' not in set(table['antecedents'])


def test_association_rules_without_common_items(mixed: amos.Dataset) -> None:
    requires('mlxtend')
    with pytest.raises(ValueError, match = 'lower "min_support"'):
        describers.AssociationRules().apply(
            mixed, columns = ['region'], min_support = 0.9)


def test_correlations(classified: amos.Dataset) -> None:
    describers.Correlations().apply(classified, method = 'spearman')
    table = classified.tables['correlations']
    assert list(table.columns) == [*classified.features, 'target']
    assert table.loc['x0', 'x0'] == pytest.approx(1.0)


def test_describe(mixed: amos.Dataset) -> None:
    describers.Describe().apply(mixed)
    table = mixed.tables['describe']
    assert list(table.index) == list(mixed.data.columns)
    assert mixed.history[-1] == {'technique': 'describe', 'table': 'describe'}


def test_frequencies(mixed: amos.Dataset) -> None:
    describers.Frequencies().apply(mixed)
    table = mixed.tables['frequencies']
    assert list(table.columns) == ['column', 'value', 'count', 'share']
    assert set(table['column']) == {'region', 'member'}
    regions = table[table['column'] == 'region']
    assert regions['count'].sum() == 200
    assert regions['share'].sum() == pytest.approx(1.0)


def test_frequencies_of_all_includes_dummies(mixed: amos.Dataset) -> None:
    mixed.data['flag'] = (mixed.data['income'] > 30000).astype('int8')
    mixed.kinds['flag'] = 'booleans'
    describers.Frequencies().apply(mixed, columns = 'all')
    table = mixed.tables['frequencies']
    assert set(table['column']) == {'region', 'member', 'flag'}
    describers.Frequencies().apply(mixed, columns = 'region')
    assert set(mixed.tables['frequencies']['column']) == {'region'}


def test_frequencies_without_categorical_columns(
    classified: amos.Dataset) -> None:
    describers.Frequencies().apply(classified)
    assert classified.tables['frequencies'].empty


def test_label_balance(mixed: amos.Dataset) -> None:
    describers.LabelBalance().apply(mixed)
    table = mixed.tables['label_balance']
    assert list(table.index) == ['no', 'yes']
    assert table['count'].sum() == 200


def test_missing_values(mixed: amos.Dataset) -> None:
    describers.MissingValues().apply(mixed)
    table = mixed.tables['missing_values']
    assert table.loc['age', 'missing'] == mixed.data['age'].isna().sum()
    assert table.loc['income', 'share'] == 0


def test_summarize(mixed: amos.Dataset) -> None:
    describers.Summarize().apply(mixed)
    table = mixed.tables['summarize']
    assert list(table.index) == ['age', 'income', 'member']
    assert list(table.columns) == [
        'count', 'missing', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'max',
        'skew', 'kurtosis']
    assert table.loc['age', 'count'] + table.loc['age', 'missing'] == 200


def test_summarize_includes_a_numeric_label(regressed: amos.Dataset) -> None:
    describers.Summarize().apply(regressed)
    assert 'target' in regressed.tables['summarize'].index
