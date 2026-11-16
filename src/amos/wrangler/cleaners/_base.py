"""The base class of cleaners.

Contents:
    Cleaner: base class for techniques that clean data.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any

import pandas as pd

from ...core import base


@dataclasses.dataclass
class Cleaner(base.Operation, abc.ABC):
    """Base class for techniques that clean data.

    A subclass writes a `clean` method, which takes the data and returns the
    cleaned data. The dataset's `history` records how many rows and columns
    there were before and after ("rows" and "columns") and, if the rows are
    the same and the values or type of any kept column changed, those columns
    ("changed").

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used by most cleaners. Defaults to `None`.
        parameters: keyword arguments for `clean`. Defaults to an empty
            `dict`.

    """

    """ Required Methods """

    @abc.abstractmethod
    def clean(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Returns the cleaned `data`.

        Args:
            data: the data to clean. It may be changed in place.
            **kwargs: parameters for cleaning.

        Returns:
            The cleaned data.

        """

    """ Public Methods """

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Cleans the data of `item`.

        Args:
            item: the dataset to clean.
            **kwargs: parameters for `clean`.

        Returns:
            The cleaned dataset.

        """
        before = item.data
        item.replace(self.clean(before.copy(), **kwargs))
        after = item.data
        details: dict[str, Any] = {
            'rows': [before.shape[0], after.shape[0]],
            'columns': [before.shape[1], after.shape[1]]}
        if after.index.equals(before.index):
            changed = [
                c for c in before.columns
                if c in after.columns and not after[c].equals(before[c])]
            if changed:
                details['changed'] = changed
        item.record(self.name, **details)
        return item
