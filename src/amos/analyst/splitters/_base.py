"""The base class of splitters.

Contents:
    Splitter: base class for techniques that split data.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any

import pandas as pd

from ...core import base, options


@dataclasses.dataclass
class Splitter(base.Operation, abc.ABC):
    """Base class for techniques that split data into training and test sets.

    A subclass writes a `divide` method, which returns the index labels of the
    training and test rows.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used by most splitters. Defaults to `None`.
        parameters: keyword arguments for `divide`. Defaults to an empty
            `dict`.

    """

    """ Required Methods """

    @abc.abstractmethod
    def divide(
        self,
        item: base.Dataset,
        test_size: float,
        **kwargs: Any) -> tuple[pd.Index, pd.Index]:
        """Returns the index labels of the training and test rows.

        Args:
            item: the dataset to split.
            test_size: share of the rows to put in the test set.
            **kwargs: parameters for the split.

        Returns:
            The training rows and the test rows.

        """

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        test_size: float | None = None,
        **kwargs: Any) -> base.Dataset:
        """Splits `item` into training and test sets.

        Args:
            item: the dataset to split.
            test_size: share of the rows to put in the test set. Defaults to
                `None`, which uses `options._DEFAULT_TEST_SIZE` (read when the
                splitter is applied, so a change to it takes effect).
            **kwargs: parameters for `divide`.

        Raises:
            ValueError: if `test_size` is not between 0 and 1.

        Returns:
            The split dataset.

        """
        if test_size is None:
            test_size = options._DEFAULT_TEST_SIZE
        if not 0 < test_size < 1:
            message = f'test_size must be between 0 and 1, not {test_size}'
            raise ValueError(message)
        train, test = self.divide(item, test_size, **kwargs)
        item.split(train, test)
        item.record(
            self.name,
            test_size = test_size,
            train = len(train),
            test = len(test),
            seed = item.seed)
        return item
