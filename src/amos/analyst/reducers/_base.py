"""The base class of reducers.

Contents:
    Reducer: genre of techniques that reduce the number of features.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import ClassVar

from .._transformers import Transformer


@dataclasses.dataclass
class Reducer(Transformer, abc.ABC):
    """Base class for techniques that reduce the number of features.

    Unlike a selector (see the `selectors` module), a reducer can replace the
    features with new ones, such as their principal components.

    """

    kinds: ClassVar[tuple[str, ...]] = ('numerics', 'booleans')
