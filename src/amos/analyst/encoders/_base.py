"""The base class of encoders.

Contents:
    Encoder: genre of techniques that turn categories into numbers.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import ClassVar

from .._transformers import Transformer


@dataclasses.dataclass
class Encoder(Transformer, abc.ABC):
    """Base class for techniques that turn categories and text into numbers.

    Supervised encoders (such as `target`) are fitted with the training labels.
    A label of text is given to them as the position of each class in
    `Dataset.classes`.

    """

    kinds: ClassVar[tuple[str, ...]] = ('categoricals',)
