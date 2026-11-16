"""The base class of describers, and what they share.

Contents:
    Describer: base class for techniques that describe data with a table.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, ClassVar

import pandas as pd

from ...core import base


@dataclasses.dataclass
class Describer(base.Operation, base.Visualizer, abc.ABC):
    """Base class for techniques that describe data with a table.

    A subclass writes a `describe` method, which returns a `DataFrame`. It is
    stored in the dataset's `tables` under the technique's name. A subclass
    names the plot that draws its table as `visual`, which `apply` uses if
    it is given `visualize` (see `base.Visualizer`).

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used by most describers. Defaults to `None`.
        parameters: keyword arguments for `describe`. Defaults to an empty
            `dict`.

    """

    stage_setting: ClassVar[str] = 'visualize_explorers'

    """ Required Methods """

    @abc.abstractmethod
    def describe(self, item: base.Dataset, **kwargs: Any) -> pd.DataFrame:
        """Returns a table that describes `item`.

        Args:
            item: the dataset to describe.
            **kwargs: parameters for the description.

        Returns:
            The table.

        """

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        *,
        visualize: bool = False,
        **kwargs: Any) -> base.Dataset:
        """Adds the table from `describe` to the tables of `item`.

        Args:
            item: the dataset to describe.
            visualize: whether to draw the table as a figure (with the plot
                named by `visual`), which is added to the figures of `item`
                under the technique's name. The dataset's
                `visualize_explorers` (the setting of the "general" section)
                replaces it, if it is set. Defaults to `False`.
            **kwargs: parameters for `describe`.

        Returns:
            The dataset, with the new table (and figure).

        """
        item.tables[self.name] = self.describe(item, **kwargs)
        item.record(self.name, table = self.name)
        self._visualize(item, visualize, kwargs)
        return item
