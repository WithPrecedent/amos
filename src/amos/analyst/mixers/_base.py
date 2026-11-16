"""The base class of mixers.

Contents:
    Mixer: genre of techniques that make new features from combinations.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import ClassVar

from .._transformers import Transformer


@dataclasses.dataclass
class Mixer(Transformer, abc.ABC):
    """Base class for techniques that make new features from numeric ones."""

    kinds: ClassVar[tuple[str, ...]] = ('numerics',)
