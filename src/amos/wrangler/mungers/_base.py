"""The base class of mungers.

Contents:
    Munger: base class for techniques that change the values in columns.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any

import pandas as pd

from ...core import base


@dataclasses.dataclass
class Munger(base.Operation, abc.ABC):
    """Base class for techniques that change the values in columns.

    A subclass writes a `munge` method, which takes the data and returns it
    with changed or new columns, and the same rows. The dataset's `history`
    records the columns whose values or types changed ("changed") and the new
    columns ("created").

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used by most mungers. Defaults to `None`.
        parameters: keyword arguments for `munge`. Defaults to an empty
            `dict`.

    """

    """ Required Methods """

    @abc.abstractmethod
    def munge(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Returns `data` with changed or new columns.

        Args:
            data: the data to change. It may be changed in place.
            **kwargs: parameters for munging.

        Returns:
            The changed data, with the same rows.

        """

    """ Public Methods """

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Changes the columns of `item`.

        Args:
            item: the dataset to change.
            **kwargs: parameters for `munge`.

        Raises:
            ValueError: if `munge` changed the rows or removed columns.

        Returns:
            The changed dataset.

        """
        after = self.munge(item.data.copy(), **kwargs)
        return base._change_columns(item, after, self.name, 'munger')
