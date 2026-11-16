"""The base class of scalers.

Contents:
    Scaler: genre of techniques that rescale numbers.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import ClassVar

from .._transformers import Transformer


@dataclasses.dataclass
class Scaler(Transformer, abc.ABC):
    """Base class for techniques that rescale the numeric features."""

    kinds: ClassVar[tuple[str, ...]] = ('numerics',)
