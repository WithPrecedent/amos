"""Tests the utilities and options modules."""

from __future__ import annotations

import importlib
import logging
import statistics

import pytest
import scipy.stats
import sklearn.preprocessing
from conftest import requires

import amos
from amos import options, utilities


def test_accepted_parameters_uses_the_signature() -> None:
    accepted = utilities.accepted_parameters(
        sklearn.preprocessing.MinMaxScaler, {'clip': True, 'other': 1})
    assert accepted == {'clip': True}


def test_accepted_parameters_uses_get_params_for_any_keyword() -> None:
    requires('xgboost')
    xgboost = importlib.import_module('xgboost')
    accepted = utilities.accepted_parameters(
        xgboost.XGBClassifier, {'max_depth': 3, 'columns': ['a']})
    assert accepted == {'max_depth': 3}


def test_accepted_parameters_keeps_everything_if_it_cannot_tell() -> None:
    def anything(**kwargs):
        return kwargs

    class Opaque:
        # Makes `inspect.signature` fail.
        __signature__ = 'not a signature'

        def __call__(self, **kwargs):
            return kwargs

    assert utilities.accepted_parameters(anything, {'a': 1}) == {'a': 1}
    assert utilities.accepted_parameters(Opaque(), {'a': 1}) == {'a': 1}


def test_describe_tool() -> None:
    assert utilities.describe_tool(None) is None
    assert utilities.describe_tool('a.b') == 'a.b'
    assert utilities.describe_tool(statistics.fmean) == 'statistics.fmean'
    scaler = sklearn.preprocessing.MinMaxScaler()
    assert utilities.describe_tool(scaler).endswith('MinMaxScaler')


def test_environment_records_python_and_packages() -> None:
    environment = utilities.environment()
    assert set(environment) == {'python', 'platform', 'packages'}
    assert environment['packages']['amos'] == amos.__version__
    assert 'pandas' in environment['packages']


def test_import_tool() -> None:
    assert utilities.import_tool(len) is len
    assert utilities.import_tool('statistics.fmean') is statistics.fmean


def test_import_tool_names_the_extra_for_a_missing_package(
    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(options._EXTRAS, 'not_a_real_package', 'demo')
    with pytest.raises(ImportError, match = r'pip install amos\[demo\]'):
        utilities.import_tool('not_a_real_package.Tool')
    with pytest.raises(ImportError):
        utilities.import_tool('another_missing_package.Tool')


def test_parameters_of() -> None:
    scaler = sklearn.preprocessing.MinMaxScaler(clip = True)
    assert utilities.parameters_of(scaler)['clip'] is True
    assert utilities.parameters_of(object()) == {}


def test_grid_search_space() -> None:
    fixed, space = utilities.search_space(
        {'max_depth': [3, 5], 'n_estimators': 10}, 'grid')
    assert fixed == {'n_estimators': 10}
    assert space == {'max_depth': [3, 5]}


def test_random_search_space_makes_ranges() -> None:
    _, space = utilities.search_space({
        'depth': [3, 9],
        'rate': [0.1, 0.01],
        'kind': ['gini', 'entropy'],
        'flags': [True, False],
        'three': [1, 2, 3]}, 'random')
    assert isinstance(space['depth'].dist, type(scipy.stats.randint))
    assert space['depth'].support() == (3, 9)
    assert space['rate'].support() == pytest.approx((0.01, 0.1))
    assert space['kind'] == ['gini', 'entropy']
    assert space['flags'] == [True, False]
    assert space['three'] == [1, 2, 3]


def test_search_space_rejects_unknown_searches() -> None:
    with pytest.raises(ValueError, match = 'grid'):
        utilities.search_space({}, 'bayes')


def test_optuna_search_space() -> None:
    requires('optuna')
    _, space = utilities.search_space({
        'depth': [3, 9],
        'rate': [0.001, 0.1],
        'share': [0.2, 0.8],
        'kind': ['gini', 'entropy']}, 'optuna')
    assert type(space['depth']).__name__ == 'IntDistribution'
    assert (space['depth'].low, space['depth'].high) == (3, 9)
    assert space['rate'].log
    assert not space['share'].log
    assert type(space['kind']).__name__ == 'CategoricalDistribution'


def test_preserved_logging_restores_the_root_logger() -> None:
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    with utilities.preserved_logging():
        logging.basicConfig(level = logging.INFO, force = True)
        assert root.level == logging.INFO
    assert root.level == level
    assert root.handlers == handlers
