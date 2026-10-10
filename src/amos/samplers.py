"""Techniques that resample the training rows to balance the classes.

These are techniques of the "analyst" stage for classification tasks in which
one class is rare. Each wraps a sampler from imbalanced-learn, which adds rows
of the rare classes (over-sampling), removes rows of the common classes
(under-sampling), or both. Only the training rows are resampled, so models are
still evaluated on the real (and imbalanced) test rows. The features must be
numbers, so put a sampler after any encoders.

Contents:
    Sampler: base class for techniques that resample the training rows.
    Adasyn: adds synthetic rows where the rare class is hardest to learn.
    BorderlineSmote: adds synthetic rows near the border between classes.
    NearMiss: removes common rows that are far from the rare rows.
    RandomOver: duplicates random rows of the rare classes.
    RandomUnder: removes random rows of the common classes.
    Smote: adds synthetic rows between rare rows and their neighbors.
    SmoteEnn: `smote`, then removes rows that their neighbors misclassify.
    SmoteTomek: `smote`, then removes pairs of rows from different classes
        that are each other's nearest neighbors.
    TomekLinks: removes common rows that are nearest neighbors of rare rows.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from . import base, utilities


@dataclasses.dataclass
class Sampler(base.Operation, abc.ABC):
    """Base class for techniques that resample the training rows.

    `contents` is the sampler to wrap: a class (or its import path) with a
    `fit_resample` method, as in imbalanced-learn. A `Sampler` can be used
    directly to wrap any such class.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: the sampler class or its import path. Defaults to `None`.
        parameters: keyword arguments for the sampler. Defaults to an empty
            `dict`.

    """

    """ Public Methods """

    def implement(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
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


@dataclasses.dataclass
class Adasyn(Sampler):
    """Adds synthetic rows where the rare class is hardest to learn.

    Wraps:
        [`imblearn.over_sampling.ADASYN`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.over_sampling.ADASYN'


@dataclasses.dataclass
class BorderlineSmote(Sampler):
    """Adds synthetic rows near the border between the classes.

    Wraps:
        [`imblearn.over_sampling.BorderlineSMOTE`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.over_sampling.BorderlineSMOTE'


@dataclasses.dataclass
class NearMiss(Sampler):
    """Removes rows of the common classes that are far from the rare rows.

    Wraps:
        [`imblearn.under_sampling.NearMiss`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.NearMiss.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.under_sampling.NearMiss'


@dataclasses.dataclass
class RandomOver(Sampler):
    """Duplicates random rows of the rare classes.

    Wraps:
        [`imblearn.over_sampling.RandomOverSampler`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.over_sampling.RandomOverSampler'


@dataclasses.dataclass
class RandomUnder(Sampler):
    """Removes random rows of the common classes.

    Wraps:
        [`imblearn.under_sampling.RandomUnderSampler`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.under_sampling.RandomUnderSampler'


@dataclasses.dataclass
class Smote(Sampler):
    """Adds synthetic rows between rare rows and their nearest neighbors.

    Wraps:
        [`imblearn.over_sampling.SMOTE`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.over_sampling.SMOTE'


@dataclasses.dataclass
class SmoteEnn(Sampler):
    """Applies `smote` and then removes rows that their neighbors misclassify.

    Wraps:
        [`imblearn.combine.SMOTEENN`](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.combine.SMOTEENN'


@dataclasses.dataclass
class SmoteTomek(Sampler):
    """Applies `smote` and then removes Tomek links.

    Wraps:
        [`imblearn.combine.SMOTETomek`](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.combine.SMOTETomek'


@dataclasses.dataclass
class TomekLinks(Sampler):
    """Removes rows of the common class that are paired with rare rows.

    Wraps:
        [`imblearn.under_sampling.TomekLinks`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.TomekLinks.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.under_sampling.TomekLinks'


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
    rows first, in order, followed by the synthetic rows.

    Args:
        original: the training rows given to the sampler.
        resampled: the rows it returned.

    Returns:
        For each resampled row, its position in `original`, or -1 if it is
            synthetic. If the first rows are not the original rows, every row
            is -1.

    """
    positions = np.full(len(resampled), -1)
    count = len(original)
    if len(resampled) >= count and np.array_equal(
        resampled.iloc[:count].to_numpy(dtype = float),
        original.to_numpy(dtype = float),
        equal_nan = True):
        positions[:count] = np.arange(count)
    return positions
