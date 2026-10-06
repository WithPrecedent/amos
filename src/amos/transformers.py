"""Techniques that learn from the training rows and then change every row.

These are the preprocessing techniques of the "analyst" stage. Each one wraps a
transformer from scikit-learn or category_encoders. A transformer is fitted to
the training rows only and then changes every row, so nothing is learned from
the test rows. Unlike scikit-learn, every transformer keeps the data in a
`pandas.DataFrame` with named columns, can be limited to some `columns`, and
is given the label when it is fitted (which target encoders need).

Each genre chooses the columns it changes by default:

| Genre | Default columns |
| --- | --- |
| `Imputer` | Features with missing values. |
| `Scaler` | Numeric features. |
| `Encoder` | Categorical and text features. |
| `Mixer` | Numeric features. |
| `Reducer` | Numeric and boolean features. |

Pass `columns` as a parameter to choose others.

Contents:
    Transformer: base class for techniques that fit and transform data.
    Imputer: genre of techniques that fill missing values.
    Scaler: genre of techniques that rescale numbers.
    Encoder: genre of techniques that turn categories into numbers.
    Mixer: genre of techniques that make new features from combinations.
    Reducer: genre of techniques that select or reduce features.
    IterativeImpute, KnnImpute, MeanImpute, MedianImpute, ModeImpute:
        imputers.
    Bins, Binarize, Gauss, MaxAbs, MinMax, Normalize, Quantile, Robust,
        Standard: scalers.
    BackwardDifference, BaseN, Binary, CatBoost, Count, Hashing, Helmert,
        JamesStein, LeaveOneOut, MEstimate, OneHot, Ordinal,
        PolynomialCoding, SumCoding, Target, WeightOfEvidence: encoders.
    Interactions, Polynomial, Splines: mixers.
    KBest, PCA, SelectPercentile, VarianceThreshold: reducers.

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
import inspect
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from . import base, utilities


@dataclasses.dataclass
class Transformer(base.Operation, abc.ABC):
    """Base class for techniques that fit and transform data.

    `contents` is the transformer to wrap: a class (or its import path) with
    `fit` and `transform` methods, as in scikit-learn. It is built with the
    technique's parameters (those it accepts), fitted to the training rows of
    the chosen columns, and then used to transform those columns in every
    row. The columns it produces replace the columns it was given. The fitted
    transformer is stored in the dataset's `fitted`.

    A `Transformer` can be used directly to wrap any transformer:

    ```py
    amos.Transformer(
        name = "yeo_johnson",
        contents = "sklearn.preprocessing.PowerTransformer")
    ```

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: the transformer class or its import path. Defaults to
            `None`.
        parameters: keyword arguments for the transformer. A "columns"
            parameter chooses the columns to transform. Defaults to an empty
            `dict`.

    """

    # The kinds of features (see `Dataset`) that are transformed by default.
    kinds: ClassVar[tuple[str, ...]] = ('numerics', 'categoricals', 'booleans')

    """ Public Methods """

    def implement(
        self,
        item: base.Dataset,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> base.Dataset:
        """Fits the transformer to the training rows and transforms every row.

        Args:
            item: the dataset to transform.
            columns: names of the columns to transform. Defaults to `None`,
                which uses `select`.
            **kwargs: parameters for the transformer.

        Returns:
            The transformed dataset.

        """
        columns = self.select(item) if columns is None else list(columns)
        if not columns:
            item.record(
                self.name,
                tool = utilities.describe_tool(self.contents),
                columns = [],
                note = 'there were no columns to transform')
            return item
        tool = self._make_tool(item, kwargs)
        _use_pandas_output(tool)
        self._fit(tool, item.x_train[columns], _target(item))
        values = self._to_frame(
            tool.transform(item.data[columns]), tool, columns, item.data.index)
        item.update_features(columns, values)
        item.fitted[self.name] = tool
        item.record(
            self.name,
            tool = utilities.describe_tool(
                tool if self.contents is None else self.contents),
            parameters = utilities.parameters_of(tool),
            columns = columns,
            created = list(values.columns))
        return item

    def select(self, item: base.Dataset) -> list[str]:
        """Returns the columns to transform when `columns` is not given.

        Args:
            item: the dataset to transform.

        Returns:
            The features of the kinds in `kinds`, in order.

        """
        chosen = set()
        for kind in self.kinds:
            chosen.update(getattr(item, kind))
        return [c for c in item.features if c in chosen]

    """ Private Methods """

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Fits `tool` to the training rows.

        Args:
            tool: the built transformer.
            x: the training rows of the columns to transform.
            y: the training labels, or `None` if there is no label.

        """
        if y is not None and _accepts_target(tool.fit):
            tool.fit(x, y)
        else:
            tool.fit(x)

    def _to_frame(
        self,
        values: Any,
        tool: Any,
        columns: list[str],
        index: pd.Index) -> pd.DataFrame:
        """Returns the output of a transformer as a `DataFrame`.

        Args:
            values: the output of the transformer's `transform` method.
            tool: the fitted transformer.
            columns: names of the columns that were transformed.
            index: index of the data.

        Returns:
            The output, with `index` and named columns.

        """
        if isinstance(values, pd.DataFrame):
            frame = values.copy()
            frame.index = index
            frame.columns = [str(c) for c in frame.columns]
            return frame
        if hasattr(values, 'toarray'):
            values = values.toarray()
        array = np.asarray(values)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        try:
            names = [str(n) for n in tool.get_feature_names_out(columns)]
        except (AttributeError, TypeError, ValueError):
            if array.shape[1] == len(columns):
                names = list(columns)
            else:
                names = [f'{self.name}_{i}' for i in range(array.shape[1])]
        return pd.DataFrame(array, index = index, columns = names)


@dataclasses.dataclass
class Imputer(Transformer, abc.ABC):
    """Genre of techniques that fill missing values.

    By default, an imputer fills the numeric features that have missing
    values. `ModeImpute` fills features of every kind.

    """

    kinds: ClassVar[tuple[str, ...]] = ('numerics',)

    def select(self, item: base.Dataset) -> list[str]:
        """Returns the features of `kinds` that have missing values.

        Args:
            item: the dataset to transform.

        Returns:
            Names of the columns to fill.

        """
        missing = item.data.columns[item.data.isna().any()]
        return [c for c in super().select(item) if c in missing]


@dataclasses.dataclass
class Scaler(Transformer, abc.ABC):
    """Genre of techniques that rescale the numeric features."""

    kinds: ClassVar[tuple[str, ...]] = ('numerics',)


@dataclasses.dataclass
class Encoder(Transformer, abc.ABC):
    """Genre of techniques that turn categorical and text features into numbers.

    Supervised encoders (such as `target`) are fitted with the training labels.
    A label of text is given to them as the position of each class in
    `Dataset.classes`.

    """

    kinds: ClassVar[tuple[str, ...]] = ('categoricals',)


@dataclasses.dataclass
class Mixer(Transformer, abc.ABC):
    """Genre of techniques that make new features from the numeric features."""

    kinds: ClassVar[tuple[str, ...]] = ('numerics',)


@dataclasses.dataclass
class Reducer(Transformer, abc.ABC):
    """Genre of techniques that select features or reduce their number."""

    kinds: ClassVar[tuple[str, ...]] = ('numerics', 'booleans')


""" Imputers """


@dataclasses.dataclass
class IterativeImpute(Imputer):
    """Fills missing values by modeling each feature from the others."""

    contents: str = 'sklearn.impute.IterativeImputer'

    def implement(
        self,
        item: base.Dataset,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> base.Dataset:
        """Enables scikit-learn's experimental imputer and then applies it.

        Args:
            item: the dataset to transform.
            columns: names of the columns to transform. Defaults to `None`,
                which uses `select`.
            **kwargs: parameters for the transformer.

        Returns:
            The transformed dataset.

        """
        importlib.import_module('sklearn.experimental.enable_iterative_imputer')
        return super().implement(item, columns = columns, **kwargs)


@dataclasses.dataclass
class KnnImpute(Imputer):
    """Fills missing values with the average of the most similar rows."""

    contents: str = 'sklearn.impute.KNNImputer'


@dataclasses.dataclass
class MeanImpute(Imputer):
    """Fills missing values with the mean of the training rows."""

    contents: str = 'sklearn.impute.SimpleImputer'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'strategy': 'mean'})


@dataclasses.dataclass
class MedianImpute(Imputer):
    """Fills missing values with the median of the training rows."""

    contents: str = 'sklearn.impute.SimpleImputer'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'strategy': 'median'})


@dataclasses.dataclass
class ModeImpute(Imputer):
    """Fills missing values with the most common value of the training rows.

    Unlike the other imputers, this fills features of every kind.

    """

    kinds: ClassVar[tuple[str, ...]] = ('numerics', 'categoricals', 'booleans')

    contents: str = 'sklearn.impute.SimpleImputer'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'strategy': 'most_frequent'})


""" Scalers """


@dataclasses.dataclass
class Bins(Scaler):
    """Sorts each feature into bins with about the same number of rows."""

    contents: str = 'sklearn.preprocessing.KBinsDiscretizer'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {
            'n_bins': 5,
            'encode': 'ordinal',
            'strategy': 'quantile',
            # Only passed to versions of scikit-learn that accept it.
            'quantile_method': 'averaged_inverted_cdf'})


@dataclasses.dataclass
class Binarize(Scaler):
    """Makes each feature 1 if it is above a threshold (0 by default)."""

    contents: str = 'sklearn.preprocessing.Binarizer'


@dataclasses.dataclass
class Gauss(Scaler):
    """Makes each feature more like a normal distribution (Yeo-Johnson)."""

    contents: str = 'sklearn.preprocessing.PowerTransformer'


@dataclasses.dataclass
class MaxAbs(Scaler):
    """Divides each feature by its largest absolute value."""

    contents: str = 'sklearn.preprocessing.MaxAbsScaler'


@dataclasses.dataclass
class MinMax(Scaler):
    """Rescales each feature to the range from 0 to 1."""

    contents: str = 'sklearn.preprocessing.MinMaxScaler'


@dataclasses.dataclass
class Normalize(Scaler):
    """Rescales each row to a length of 1."""

    contents: str = 'sklearn.preprocessing.Normalizer'


@dataclasses.dataclass
class Quantile(Scaler):
    """Replaces each value with its quantile in the training rows."""

    contents: str = 'sklearn.preprocessing.QuantileTransformer'

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Uses no more quantiles than there are training rows.

        Args:
            tool: the built transformer.
            x: the training rows of the columns to transform.
            y: the training labels, or `None` if there is no label.

        """
        if tool.n_quantiles > len(x):
            tool.set_params(n_quantiles = len(x))
        super()._fit(tool, x, y)


@dataclasses.dataclass
class Robust(Scaler):
    """Centers on the median and scales by the interquartile range."""

    contents: str = 'sklearn.preprocessing.RobustScaler'


@dataclasses.dataclass
class Standard(Scaler):
    """Centers each feature on 0 with a standard deviation of 1."""

    contents: str = 'sklearn.preprocessing.StandardScaler'


""" Encoders """


@dataclasses.dataclass
class BackwardDifference(Encoder):
    """Compares each category to the one before it."""

    contents: str = 'category_encoders.BackwardDifferenceEncoder'


@dataclasses.dataclass
class BaseN(Encoder):
    """Writes the number of each category in base N (4 by default)."""

    contents: str = 'category_encoders.BaseNEncoder'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'base': 4})


@dataclasses.dataclass
class Binary(Encoder):
    """Writes the number of each category in binary digits."""

    contents: str = 'category_encoders.BinaryEncoder'


@dataclasses.dataclass
class CatBoost(Encoder):
    """Target encoding in the manner of CatBoost, which limits leakage."""

    contents: str = 'category_encoders.CatBoostEncoder'


@dataclasses.dataclass
class Count(Encoder):
    """Replaces each category with how often it is in the training rows."""

    contents: str = 'category_encoders.CountEncoder'


@dataclasses.dataclass
class Hashing(Encoder):
    """Hashes the categories into a fixed number of columns."""

    contents: str = 'category_encoders.HashingEncoder'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'max_process': 1})


@dataclasses.dataclass
class Helmert(Encoder):
    """Compares each category to the mean of the categories before it."""

    contents: str = 'category_encoders.HelmertEncoder'


@dataclasses.dataclass
class JamesStein(Encoder):
    """Target encoding shrunk toward the overall mean (James-Stein)."""

    contents: str = 'category_encoders.JamesSteinEncoder'


@dataclasses.dataclass
class LeaveOneOut(Encoder):
    """Target encoding that leaves out each row's own label."""

    contents: str = 'category_encoders.LeaveOneOutEncoder'


@dataclasses.dataclass
class MEstimate(Encoder):
    """Target encoding shrunk toward the overall mean by m rows."""

    contents: str = 'category_encoders.MEstimateEncoder'


@dataclasses.dataclass
class OneHot(Encoder):
    """Makes a column of 0s and 1s for each category (dummy variables).

    Categories that are not in the training rows are all 0s.

    """

    contents: str = 'sklearn.preprocessing.OneHotEncoder'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {
            'handle_unknown': 'ignore',
            'sparse_output': False})


@dataclasses.dataclass
class Ordinal(Encoder):
    """Numbers the categories (0, 1, 2, ...) in sorted order.

    Categories that are not in the training rows are -1.

    """

    contents: str = 'sklearn.preprocessing.OrdinalEncoder'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {
            'handle_unknown': 'use_encoded_value',
            'unknown_value': -1})


@dataclasses.dataclass
class PolynomialCoding(Encoder):
    """Contrasts categories as an ordered (polynomial) sequence."""

    contents: str = 'category_encoders.PolynomialEncoder'


@dataclasses.dataclass
class SumCoding(Encoder):
    """Compares each category to the mean of all categories (effect coding)."""

    contents: str = 'category_encoders.SumEncoder'


@dataclasses.dataclass
class Target(Encoder):
    """Replaces each category with the mean label of its training rows."""

    contents: str = 'category_encoders.TargetEncoder'


@dataclasses.dataclass
class WeightOfEvidence(Encoder):
    """Replaces each category with its weight of evidence (binary labels)."""

    contents: str = 'category_encoders.WOEEncoder'


""" Mixers """


@dataclasses.dataclass
class Interactions(Mixer):
    """Adds the product of each pair of features."""

    contents: str = 'sklearn.preprocessing.PolynomialFeatures'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {
            'interaction_only': True,
            'include_bias': False})


@dataclasses.dataclass
class Polynomial(Mixer):
    """Adds the squares and products of the features (degree 2 by default)."""

    contents: str = 'sklearn.preprocessing.PolynomialFeatures'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'include_bias': False})


@dataclasses.dataclass
class Splines(Mixer):
    """Replaces each feature with a set of spline curves."""

    contents: str = 'sklearn.preprocessing.SplineTransformer'


""" Reducers """


@dataclasses.dataclass
class KBest(Reducer):
    """Keeps the k features (10 by default) most related to the label.

    The default test is an F test suited to the task (`f_classif` or
    `f_regression`). Set "score_func" to another function or its import path.

    """

    contents: str = 'sklearn.feature_selection.SelectKBest'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'k': 10})

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Keeps every feature if there are no more than k.

        Args:
            tool: the built transformer.
            x: the training rows of the columns to transform.
            y: the training labels, or `None` if there is no label.

        """
        if isinstance(tool.k, int) and tool.k > x.shape[1]:
            tool.set_params(k = 'all')
        super()._fit(tool, x, y)

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the transformer.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class PCA(Reducer):
    """Replaces the features with their principal components.

    By default, it keeps enough components to explain 95% of the variance.
    Scale the features first.

    """

    contents: str = 'sklearn.decomposition.PCA'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'n_components': 0.95})


@dataclasses.dataclass
class SelectPercentile(Reducer):
    """Keeps the features (50% by default) most related to the label.

    The default test is an F test suited to the task (`f_classif` or
    `f_regression`). Set "score_func" to another function or its import path.

    """

    contents: str = 'sklearn.feature_selection.SelectPercentile'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'percentile': 50})

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the transformer.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class VarianceThreshold(Reducer):
    """Removes features whose variance is at or below a threshold (0)."""

    contents: str = 'sklearn.feature_selection.VarianceThreshold'


""" Private Functions """


def _accepts_target(method: Any) -> bool:
    """Returns whether a `fit` method takes a label (`y`).

    Args:
        method: the `fit` method of a transformer.

    Returns:
        Whether `method` has a `y` parameter or takes any positional
            arguments after the features.

    """
    try:
        parameters = list(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return True
    if any(p.name == 'y' for p in parameters):
        return True
    return any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters)


def _target(item: base.Dataset) -> pd.Series | None:
    """Returns the training labels to fit a transformer with.

    Args:
        item: the dataset.

    Returns:
        The training labels, with text classes replaced by their position in
            `Dataset.classes`, or `None` if there is no label.

    """
    if item.label is None:
        return None
    y = item.y_train
    if item.task == 'classify' and not pd.api.types.is_numeric_dtype(y.dtype):
        positions = {c: i for i, c in enumerate(item.classes)}
        y = y.map(positions)
    return y


def _use_pandas_output(tool: Any) -> None:
    """Asks a scikit-learn transformer to return a `DataFrame`, if it can.

    Args:
        tool: a built transformer.

    """
    if hasattr(tool, 'set_output'):
        try:
            tool.set_output(transform = 'pandas')
        except (ValueError, TypeError, AttributeError):
            # Some transformers (such as those with sparse output) cannot
            # return a `DataFrame`. Their output is converted by `_to_frame`.
            return


def _with_score_function(
    item: base.Dataset,
    parameters: dict[str, Any]) -> dict[str, Any]:
    """Returns `parameters` with a scoring function for feature selection.

    Args:
        item: the dataset.
        parameters: parameters for a feature selector.

    Returns:
        The parameters, with "score_func" imported if it was an import path,
            or chosen for the task if it was not given.

    """
    score = parameters.get('score_func')
    if score is None:
        name = 'f_classif' if item.task == 'classify' else 'f_regression'
        score = f'sklearn.feature_selection.{name}'
    return {**parameters, 'score_func': utilities.import_tool(score)}
