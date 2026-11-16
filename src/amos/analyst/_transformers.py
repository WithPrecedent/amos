"""The base class of techniques that fit and transform data.

Imputers, scalers, encoders, mixers, reducers, and most selectors are
transformers: preprocessing techniques of the "analyst" stage, each in its
own step (such as `amos.scalers`). Each one wraps a transformer from
scikit-learn, category_encoders, or skrub. A transformer is fitted to the
training rows only and then changes every row, so nothing is learned from
the test rows. Unlike scikit-learn, every transformer keeps the data in a
`pandas.DataFrame` with named columns, can be limited to some `columns`, and
is given the label when it is fitted (which target encoders need).

Each genre chooses the columns it changes by default:

| Genre | Default columns |
| --- | --- |
| `Imputer` | Features with missing values. |
| `Scaler` | Numeric features. |
| `Encoder` | Categorical and text features (dates, for `date_parts`). |
| `Mixer` | Numeric features. |
| `Reducer` | Numeric and boolean features. |

Pass `columns` as a parameter to choose others.

Contents:
    Transformer: base class for techniques that fit and transform data.
    ColumnWise: applies a transformer that takes one column to each of several
        columns.

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

from ..core import base, utilities


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
            an empty `str`, which uses the snake case name of the class.
        contents: the transformer class or its import path. Defaults to
            `None`.
        parameters: keyword arguments for the transformer. A "columns"
            parameter chooses the columns to transform. Defaults to an empty
            `dict`.

    """

    # The kinds of features (see `Dataset`) that are transformed by default.
    kinds: ClassVar[tuple[str, ...]] = ('numerics', 'categoricals', 'booleans')
    # Whether the tool takes one column at a time (as skrub's encoders do), so
    # that a copy of it is fitted to each column.
    columnwise: ClassVar[bool] = False

    """ Public Methods """

    def apply(
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
        if self.columnwise:
            tool = ColumnWise(transformer = tool)
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


# `eq` is `False` so that adapters are compared by identity, as scikit-learn
# transformers are.
@dataclasses.dataclass(eq = False)
class ColumnWise:
    """Applies a transformer that takes one column to each of several columns.

    skrub's encoders, for example, encode one column at a time. A copy of the
    transformer is fitted to each column, and their outputs are joined.

    Args:
        transformer: the transformer to copy for each column.

    Attributes:
        transformers_: the fitted copy for each column, by its name.

    """

    transformer: Any
    transformers_: dict[str, Any] = dataclasses.field(
        default_factory = dict, init = False, repr = False)

    """ Public Methods """

    def fit(self, x: pd.DataFrame, y: Any = None) -> ColumnWise:
        """Fits a copy of the transformer to each column of `x`.

        Args:
            x: the columns to transform.
            y: the labels, passed to transformers that take them. Defaults to
                `None`.

        Returns:
            This adapter.

        """
        sklearn_base = importlib.import_module('sklearn.base')
        self.transformers_ = {}
        for column in x.columns:
            transformer = sklearn_base.clone(self.transformer)
            if y is not None and _accepts_target(transformer.fit):
                transformer.fit(x[column], y)
            else:
                transformer.fit(x[column])
            self.transformers_[column] = transformer
        return self

    def get_feature_names_out(self, *args: Any) -> np.ndarray:
        """Returns the names of the columns that `transform` makes.

        Args:
            *args: not used.

        Returns:
            The names, in order.

        """
        names = [
            name for transformer in self.transformers_.values()
            for name in transformer.get_feature_names_out()]
        return np.asarray(names, dtype = object)

    def get_params(
        self,
        deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Returns the parameters, as scikit-learn expects.

        Args:
            deep: not used.

        Returns:
            The parameters.

        """
        return {'transformer': self.transformer}

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transforms each column with its fitted copy and joins the results.

        Args:
            x: the columns to transform.

        Returns:
            The joined output, with the index of `x`.

        """
        outputs = []
        for column, transformer in self.transformers_.items():
            output = transformer.transform(x[column])
            if not isinstance(output, pd.DataFrame):
                output = pd.DataFrame(
                    np.asarray(output),
                    columns = transformer.get_feature_names_out())
            output.index = x.index
            outputs.append(output)
        return pd.concat(outputs, axis = 1)


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


# The public names say that they are in `amos.transformers`, where they are
# imported from, so that records, pickles, and the documentation name them
# as `amos.transformers.Name`. `amos` makes `amos.transformers` importable.
Transformer.__module__ = 'amos.transformers'
ColumnWise.__module__ = 'amos.transformers'
