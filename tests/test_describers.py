"""Tests the describers module."""

from __future__ import annotations

import pytest

import amos
from amos import describers


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
