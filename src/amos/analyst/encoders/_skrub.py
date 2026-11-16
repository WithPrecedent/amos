"""Encoders of text and dates from skrub.

Contents:
    DateParts: encoders.
    Gap: encoders.
    MinHash: encoders.
    Tfidf: encoders.

"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from ._base import Encoder

""" Encoders """


@dataclasses.dataclass
class DateParts(Encoder):
    """Splits dates and times into parts: year, month, day, and so on.

    This is the only encoder that changes dates (and only dates) by default,
    so it lets models use them. It wraps skrub's `DatetimeEncoder`.

    Wraps:
        [`skrub.DatetimeEncoder`](https://skrub-data.org/stable/reference/generated/skrub.DatetimeEncoder.html)
        from skrub.

    """

    kinds: ClassVar[tuple[str, ...]] = ('dates',)
    columnwise: ClassVar[bool] = True

    contents: str = 'skrub.DatetimeEncoder'


@dataclasses.dataclass
class Gap(Encoder):
    """Encodes messy text as a mix of topics of its substrings.

    Useful for text with typos or variations (such as "St. Louis" and
    "Saint Louis"). Each new column is named for the most common
    substrings of its topic, so it can be interpreted. It wraps skrub's
    `GapEncoder` (10 topics by default).

    Wraps:
        [`skrub.GapEncoder`](https://skrub-data.org/stable/reference/generated/skrub.GapEncoder.html)
        from skrub.

    """

    columnwise: ClassVar[bool] = True

    contents: str = 'skrub.GapEncoder'


@dataclasses.dataclass
class MinHash(Encoder):
    """Encodes messy text by hashing its substrings, which is fast and robust.

    It wraps skrub's `MinHashEncoder` (30 columns for each column by
    default).

    Wraps:
        [`skrub.MinHashEncoder`](https://skrub-data.org/stable/reference/generated/skrub.MinHashEncoder.html)
        from skrub.

    """

    columnwise: ClassVar[bool] = True

    contents: str = 'skrub.MinHashEncoder'


@dataclasses.dataclass
class Tfidf(Encoder):
    """Encodes text by the TF-IDF of its substrings, reduced with SVD.

    A strong default for text categories with many distinct values. It wraps
    skrub's `StringEncoder` (30 columns for each column by default).

    Wraps:
        [`skrub.StringEncoder`](https://skrub-data.org/stable/reference/generated/skrub.StringEncoder.html)
        from skrub.

    """

    columnwise: ClassVar[bool] = True

    contents: str = 'skrub.StringEncoder'
