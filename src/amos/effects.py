"""Techniques that estimate the causal effect of a treatment.

These techniques answer a different question from models. A model predicts
the label; an effect estimates how much a treatment (such as a program, a
policy, or a type of ruling) changes the label, after accounting for the other
features. They use double (or debiased) machine learning, from DoubleML: one
model of the label and one of the treatment, each fitted to some folds of the
rows and applied to the others, so that flexible models can control for the
features without biasing the estimate.

Unlike a model, an effect uses every row (it does not need a split) and makes
no predictions. Its estimate, standard error, test statistic, p-value, and 95%
confidence interval are stored in the dataset's `tables` under the
technique's name. These estimates are causal only if every feature that
affects both the treatment and the label is among the features.

Contents:
    Effect: base class for techniques that estimate the effect of a
        treatment.
    InteractiveRegression: the average effect of a treatment with two values.
    PartiallyLinear: the effect of a treatment that adds to the label.

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
from typing import Any

import chrisjen
import numpy as np
import pandas as pd

from . import base, models, utilities


@dataclasses.dataclass
class Effect(base.Operation, abc.ABC):
    """Base class for techniques that estimate the effect of a treatment.

    Parameters (all set in the settings or passed to `apply`):

    | Parameter | Meaning |
    | --- | --- |
    | `treatment` | Name of the column with the treatment. Required. |
    | `outcome_model` | Name of an `amos` model for the label (by default, `random_forest`). |
    | `treatment_model` | Name of an `amos` model for the treatment (by default, the outcome model). |
    | `n_folds` | Number of folds for cross-fitting (5 by default). |

    A subclass writes `_estimator`, which builds the DoubleML estimator.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, in which case it is the snake case name of the
            class.
        contents: not used. Defaults to `None`.
        parameters: keyword arguments for `implement`. Defaults to an empty
            `dict`.

    """

    """ Required Methods """

    @abc.abstractmethod
    def _estimator(
        self,
        doubleml: Any,
        data: Any,
        outcome: Any,
        treatment: Any,
        n_folds: int) -> Any:
        """Returns the DoubleML estimator.

        Args:
            doubleml: the `doubleml` module.
            data: a `doubleml.DoubleMLData` of the label, treatment, and
                features.
            outcome: the model of the label.
            treatment: the model of the treatment.
            n_folds: number of folds for cross-fitting.

        Returns:
            The estimator, ready to fit.

        """

    """ Public Methods """

    def implement(
        self,
        item: base.Dataset,
        treatment: str | None = None,
        outcome_model: str = 'random_forest',
        treatment_model: str | None = None,
        n_folds: int = 5,
        **kwargs: Any) -> base.Dataset:
        """Estimates the effect of `treatment` on the label.

        Args:
            item: the dataset. It must have a label.
            treatment: name of the column with the treatment. Defaults to
                `None`, but it is required.
            outcome_model: name of an `amos` model for the label. Defaults to
                "random_forest".
            treatment_model: name of an `amos` model for the treatment.
                Defaults to `None`, which uses `outcome_model`.
            n_folds: number of folds for cross-fitting. Defaults to 5.
            **kwargs: not used.

        Raises:
            ValueError: if there is no treatment, or a feature is not a number.

        Returns:
            The dataset, with the estimate in `tables`.

        """
        if treatment is None:
            message = f'{self.name!r} needs the name of a "treatment" column'
            raise ValueError(message)
        doubleml = utilities.import_tool('doubleml')
        label = item._require_label()
        controls = [c for c in item.features if c != treatment]
        unusable = [
            c for c in controls
            if base._kind_of(item.data[c]) not in {'numerics', 'booleans'}]
        if unusable:
            message = (
                f'{self.name!r} needs numbers, but {unusable} are not: encode '
                f'or remove them first'
            )
            raise ValueError(message)
        data = item.data[controls].astype(float)
        data[label] = _numbers(item.y)
        data[treatment] = _numbers(item.data[treatment])
        treatment_task = (
            'classify' if data[treatment].nunique() == 2 else 'regress')  # noqa: PLR2004
        estimator = self._estimator(
            doubleml,
            doubleml.DoubleMLData(
                data, y_col = label, d_cols = treatment, x_cols = controls),
            self._outcome_model(outcome_model, item),
            _learner(
                treatment_model or outcome_model, treatment_task, item.seed),
            n_folds)
        # Folds drawn with the dataset's seed make the estimate reproducible.
        estimator.set_sample_splitting(_folds(len(data), n_folds, item.seed))
        estimator.fit()
        table = estimator.summary.iloc[:, :6].copy()
        table.columns = [
            'coefficient', 'standard_error', 'statistic', 'p_value',
            'ci_lower', 'ci_upper']
        item.tables[self.name] = table
        item.fitted[self.name] = estimator
        item.record(
            self.name,
            tool = utilities.describe_tool(type(estimator)),
            treatment = treatment,
            outcome_model = outcome_model,
            treatment_model = treatment_model or outcome_model,
            n_folds = n_folds,
            effect = float(table['coefficient'].iloc[0]))
        return item

    """ Private Methods """

    def _outcome_model(self, name: str, item: base.Dataset) -> Any:
        """Returns the model of the label.

        Args:
            name: name of an `amos` model.
            item: the dataset.

        Returns:
            A regressor of the label (as a number).

        """
        return _learner(name, 'regress', item.seed)


@dataclasses.dataclass
class InteractiveRegression(Effect):
    """The average effect of a treatment that has two values (such as 0 and 1).

    It allows the effect to differ from row to row, and estimates its average
    over every row (the average treatment effect), with DoubleML's
    interactive regression model. The model of the label is a classifier if
    the label has two classes.

    """

    """ Private Methods """

    def _estimator(
        self,
        doubleml: Any,
        data: Any,
        outcome: Any,
        treatment: Any,
        n_folds: int) -> Any:
        """Returns DoubleML's interactive regression model (`DoubleMLIRM`).

        Args:
            doubleml: the `doubleml` module.
            data: a `doubleml.DoubleMLData` of the label, treatment, and
                features.
            outcome: the model of the label.
            treatment: the model of the treatment (a classifier).
            n_folds: number of folds for cross-fitting.

        Raises:
            ValueError: if the treatment does not have two values.

        Returns:
            The estimator, ready to fit.

        """
        values = np.unique(np.asarray(data.d))
        if len(values) != 2:  # noqa: PLR2004
            message = f'{self.name!r} needs a treatment with two values'
            raise ValueError(message)
        return doubleml.DoubleMLIRM(
            data, outcome, treatment, n_folds = n_folds, score = 'ATE')

    def _outcome_model(self, name: str, item: base.Dataset) -> Any:
        """Returns the model of the label.

        Args:
            name: name of an `amos` model.
            item: the dataset.

        Returns:
            A classifier if the label has two classes, and otherwise a
                regressor.

        """
        binary = item.task == 'classify' and len(item.classes) == 2  # noqa: PLR2004
        return _learner(name, 'classify' if binary else 'regress', item.seed)


@dataclasses.dataclass
class PartiallyLinear(Effect):
    """The effect of a treatment that adds to the label in the same way for all.

    It assumes that the treatment changes the label by the same amount for
    every row, while the features may affect both in any way, with DoubleML's
    partially linear regression model. The treatment can be a number or have
    two values.

    """

    """ Private Methods """

    def _estimator(
        self,
        doubleml: Any,
        data: Any,
        outcome: Any,
        treatment: Any,
        n_folds: int) -> Any:
        """Returns DoubleML's partially linear regression model (`DoubleMLPLR`).

        Args:
            doubleml: the `doubleml` module.
            data: a `doubleml.DoubleMLData` of the label, treatment, and
                features.
            outcome: the model of the label (a regressor).
            treatment: the model of the treatment.
            n_folds: number of folds for cross-fitting.

        Returns:
            The estimator, ready to fit.

        """
        return doubleml.DoubleMLPLR(data, outcome, treatment, n_folds = n_folds)


""" Private Functions """


def _folds(
    rows: int,
    n_folds: int,
    seed: int | None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns folds of the rows for cross-fitting.

    Args:
        rows: number of rows.
        n_folds: number of folds.
        seed: seed for shuffling the rows.

    Returns:
        The positions of the rows used to fit and to apply the models, for
            each fold.

    """
    model_selection = importlib.import_module('sklearn.model_selection')
    splitter = model_selection.KFold(
        n_splits = n_folds, shuffle = True, random_state = seed)
    return list(splitter.split(np.arange(rows)))


def _learner(name: str, task: str, seed: int | None) -> Any:
    """Returns an unfitted model, chosen by the name of an `amos` model.

    Args:
        name: name of an `amos` model (such as "random_forest").
        task: "classify" or "regress".
        seed: seed for the model, if it takes one.

    Raises:
        TypeError: if `name` is not the name of a model.
        ValueError: if the model cannot do `task`.

    Returns:
        The model, built with the `amos` model's default parameters.

    """
    technique = chrisjen.library.borrow(name, genre = 'model')()
    if not isinstance(technique, models.Model):
        message = f'{name!r} is not a model'
        raise TypeError(message)
    tool = technique.contents or technique.tools.get(task)
    if tool is None:
        message = f'the model {name!r} cannot {task}'
        raise ValueError(message)
    built = utilities.import_tool(tool)
    parameters = {**technique._keywords(), 'random_state': seed}
    return built(**utilities.accepted_parameters(built, parameters))


def _numbers(values: pd.Series) -> pd.Series:
    """Returns `values` as numbers.

    Args:
        values: a column.

    Returns:
        The column as floats. A column with two values that are not numbers
            becomes 1 for the last value (in sorted order) and 0 for the
            other.

    """
    if pd.api.types.is_numeric_dtype(values.dtype):
        return values.astype(float)
    levels = sorted(values.dropna().unique().tolist())
    numbers: pd.Series = (values == levels[-1]).astype(float)
    return numbers
