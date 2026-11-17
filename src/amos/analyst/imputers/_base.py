"""The base class of imputers.

Contents:
    Imputer: genre of techniques that fill missing values.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import ClassVar

from ...core import base
from .._transformers import Transformer


@dataclasses.dataclass
class Imputer(Transformer, abc.ABC):
    """Base class for techniques that fill missing values.

    By default, an imputer fills the numeric features that have missing
    values. `ModeImpute` fills features of every kind.

    """

    kinds: ClassVar[tuple[str, ...]] = ('numerics',)
    # The names of what the technique assumes about the data (see the
    # `assumptions` module) and needs from it (see the `needs` module).
    assumptions: ClassVar[tuple[str, ...]] = ()
    needs: ClassVar[tuple[str, ...]] = ()

    def select(self, item: base.Dataset) -> list[str]:
        """Returns the features of `kinds` that have missing values.

        Args:
            item: the dataset to transform.

        Returns:
            Names of the columns to fill.

        """
        missing = item.data.columns[item.data.isna().any()]
        return [c for c in super().select(item) if c in missing]
