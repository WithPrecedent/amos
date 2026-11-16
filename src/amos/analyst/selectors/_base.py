"""The base class of selectors, and what they share.

Contents:
    Selector: genre of techniques that keep some of the features.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .. import _transformers as transformers


@dataclasses.dataclass
class Selector(transformers.Transformer, abc.ABC):
    """Base class for techniques that keep some of the features.

    A selector chooses from the numeric and boolean features by default, and
    the features it does not choose from are kept. A selector that would keep
    none of the features raises an error instead.

    """

    kinds: ClassVar[tuple[str, ...]] = ('numerics', 'booleans')

    """ Private Methods """

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Fits the selector, checking that it keeps a feature.

        Args:
            tool: the built selector.
            x: the training rows of the columns to choose from.
            y: the training labels, or `None` if there is no label.

        Raises:
            ValueError: if the selector keeps none of the features.

        """
        super()._fit(tool, x, y)
        support = getattr(tool, 'get_support', None)
        if callable(support) and not np.asarray(support()).any():
            message = (
                f'{self.name!r} kept none of the {x.shape[1]} features: '
                f'loosen its parameters (such as "alpha" or "threshold")'
            )
            raise ValueError(message)

