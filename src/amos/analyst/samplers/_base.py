"""The base class of samplers, and what they share.

Contents:
    Sampler: base class for techniques that resample the training rows.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ...core import base, utilities


@dataclasses.dataclass
class Sampler(base.Operation, abc.ABC):
    """Base class for techniques that resample the training rows.

    `contents` is the sampler to wrap: a class (or its import path) with a
    `fit_resample` method, as in imbalanced-learn. A `Sampler` can be used
    directly to wrap any such class.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: the sampler class or its import path. Defaults to `None`.
        parameters: keyword arguments for the sampler. Defaults to an empty
            `dict`.

    """

    # The names of what the technique assumes about the data (see the
    # `assumptions` module) and needs from it (see the `needs` module).
    assumptions: ClassVar[tuple[str, ...]] = ()
    needs: ClassVar[tuple[str, ...]] = ()

    """ Public Methods """

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Resamples the training rows of `item`.

        Args:
            item: the dataset to resample. It must have a label.
            **kwargs: parameters for the sampler.

        Returns:
            The resampled dataset.

        """
        before = _class_counts(item.y_train)
        tool = self._make_tool(item, kwargs)
        x, y = tool.fit_resample(item.x_train, item.y_train)
        origins = getattr(tool, 'sample_indices_', None)
        if origins is None:
            origins = _leading_copies(item.x_train, x)
        item.resample(x, y, origins = origins)
        item.fitted[self.name] = tool
        item.record(
            self.name,
            tool = utilities.describe_tool(
                tool if self.contents is None else self.contents),
            parameters = utilities.parameters_of(tool),
            before = before,
            after = _class_counts(item.y_train))
        return item


""" Private Functions """


def _class_counts(labels: pd.Series) -> dict[Any, int]:
    """Returns the number of rows of each class, as plain Python values.

    The classes of some labels are `numpy` values (such as those of `pandas`
    booleans that allow missing values), which a json file cannot have as
    keys, so a history that recorded them could not be exported.

    Args:
        labels: the labels of the rows to count.

    Returns:
        The number of rows of each class, from the most common to the least.

    """
    return {
        (label.item() if isinstance(label, np.generic) else label): int(count)
        for label, count in labels.value_counts().items()}


def _leading_copies(
    original: pd.DataFrame,
    resampled: pd.DataFrame) -> np.ndarray:
    """Returns the position of the original row that each resampled row copies.

    Samplers that report which rows they kept (in `sample_indices_`) do not
    need this. SMOTE and the other over-samplers instead return the original
    rows first, in order, followed by the synthetic rows. Samplers that then
    remove rows (`smote_tomek` and `smote_enn`) return only some of the
    original rows, so if the first rows are not every original row, each
    resampled row that has the same values as an original row is taken to
    copy it (a synthetic row is made between two rows, so it almost never
    equals one).

    Args:
        original: the training rows given to the sampler.
        resampled: the rows it returned, with the same columns.

    Returns:
        For each resampled row, its position in `original`, or -1 if it is
            synthetic.

    """
    positions = np.full(len(resampled), -1)
    count = len(original)
    if len(resampled) >= count and np.array_equal(
        resampled.iloc[:count].to_numpy(dtype = float),
        original.to_numpy(dtype = float),
        equal_nan = True):
        positions[:count] = np.arange(count)
        return positions
    hashes = pd.util.hash_pandas_object(
        original.astype(float), index = False).to_numpy()
    first = pd.Series(np.arange(count), index = hashes)
    first = first[~first.index.duplicated()]
    made = pd.util.hash_pandas_object(
        resampled[original.columns].astype(float), index = False).to_numpy()
    return pd.Series(made).map(first).fillna(-1).astype(int).to_numpy()
