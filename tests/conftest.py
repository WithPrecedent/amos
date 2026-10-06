"""Shared fixtures for the tests."""

from __future__ import annotations

import copy
import importlib
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

import amos
import chrisjen

SEED = 43


@pytest.fixture(autouse = True)
def restore_library() -> Iterator[None]:
    """Removes the classes that a test adds to the library."""
    saved = copy.deepcopy(chrisjen.library.contents)
    yield
    chrisjen.library.contents.clear()
    chrisjen.library.contents.update(saved)


def requires(package: str) -> None:
    """Skips a test if `package` cannot be imported.

    Some optional packages (such as xgboost on macOS without OpenMP) are
    installed but fail when imported, so any error skips the test.
    """
    try:
        importlib.import_module(package)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f'{package} cannot be imported: {error}')


def make_mixed(rows: int = 200, seed: int = 7) -> pd.DataFrame:
    """Returns data with every kind of column and a text label.

    "age" has missing values, "region" is text, "member" is boolean,
    "joined" is a date, and "outcome" is "yes" or "no".
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, rows)
    income = rng.lognormal(10, 0.5, rows)
    region = rng.choice(['north', 'south', 'east', 'west'], rows)
    member = rng.random(rows) > 0.5
    score = (
        0.05 * age
        + 1.5 * (region == 'south')
        + 0.8 * member
        + rng.normal(0, 1, rows))
    outcome = np.where(score > np.median(score), 'yes', 'no')
    age[rng.random(rows) < 0.1] = np.nan
    return pd.DataFrame({
        'age': age,
        'income': income,
        'region': pd.Series(region, dtype = 'str'),
        'member': member,
        'joined': pd.date_range('2020-01-01', periods = rows, freq = 'D'),
        'outcome': outcome})


def make_numeric(
    rows: int = 200,
    classes: int = 2,
    weights: list[float] | None = None,
    seed: int = 7) -> pd.DataFrame:
    """Returns numeric features and an integer label of `classes` classes."""
    datasets = importlib.import_module('sklearn.datasets')
    x, y = datasets.make_classification(
        n_samples = rows,
        n_features = 6,
        n_informative = 4,
        n_classes = classes,
        weights = weights,
        random_state = seed)
    data = pd.DataFrame(x, columns = [f'x{i}' for i in range(6)])
    data['target'] = y
    return data


def make_regression(rows: int = 200, seed: int = 7) -> pd.DataFrame:
    """Returns numeric features and a continuous label."""
    datasets = importlib.import_module('sklearn.datasets')
    x, y = datasets.make_regression(
        n_samples = rows,
        n_features = 5,
        noise = 10,
        random_state = seed)
    data = pd.DataFrame(x, columns = [f'x{i}' for i in range(5)])
    data['target'] = y + 500
    return data


@pytest.fixture
def mixed() -> amos.Dataset:
    """A dataset with every kind of column and a text label."""
    return amos.Dataset(make_mixed(), label = 'outcome', seed = SEED)


@pytest.fixture
def classified() -> amos.Dataset:
    """A split binary classification dataset with numeric features."""
    dataset = amos.Dataset(make_numeric(), label = 'target', seed = SEED)
    return amos.splitters.Stratified().apply(dataset)


@pytest.fixture
def multiclass() -> amos.Dataset:
    """A split classification dataset with three classes."""
    dataset = amos.Dataset(
        make_numeric(rows = 300, classes = 3), label = 'target', seed = SEED)
    return amos.splitters.Stratified().apply(dataset)


@pytest.fixture
def regressed() -> amos.Dataset:
    """A split regression dataset."""
    dataset = amos.Dataset(make_regression(), label = 'target', seed = SEED)
    return amos.splitters.TrainTest().apply(dataset)


@pytest.fixture
def fitted(classified: amos.Dataset) -> amos.Dataset:
    """The `classified` dataset with a fitted logistic regression."""
    return amos.models.Logit().apply(classified)


@pytest.fixture
def fitted_regression(regressed: amos.Dataset) -> amos.Dataset:
    """The `regressed` dataset with a fitted linear regression."""
    return amos.models.Linear().apply(regressed)
