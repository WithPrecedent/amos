"""Encoders from category_encoders.

Contents:
    BackwardDifference: encoders.
    BaseN: encoders.
    Binary: encoders.
    CatBoost: encoders.
    Count: encoders.
    Hashing: encoders.
    Helmert: encoders.
    JamesStein: encoders.
    LeaveOneOut: encoders.
    MEstimate: encoders.
    PolynomialCoding: encoders.
    SumCoding: encoders.
    Target: encoders.
    WeightOfEvidence: encoders.

"""

from __future__ import annotations

import dataclasses

from ...core import base
from ._base import Encoder

""" Encoders """


@dataclasses.dataclass
class BackwardDifference(Encoder):
    """Compares each category to the one before it.

    Wraps:
        [`category_encoders.BackwardDifferenceEncoder`](https://contrib.scikit-learn.org/category_encoders/backward_difference.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.BackwardDifferenceEncoder'


@dataclasses.dataclass
class BaseN(Encoder):
    """Writes the number of each category in base N (4 by default).

    Wraps:
        [`category_encoders.BaseNEncoder`](https://contrib.scikit-learn.org/category_encoders/basen.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.BaseNEncoder'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'base': 4})


@dataclasses.dataclass
class Binary(Encoder):
    """Writes the number of each category in binary digits.

    Wraps:
        [`category_encoders.BinaryEncoder`](https://contrib.scikit-learn.org/category_encoders/binary.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.BinaryEncoder'


@dataclasses.dataclass
class CatBoost(Encoder):
    """Target encoding in the manner of CatBoost, which limits leakage.

    Wraps:
        [`category_encoders.CatBoostEncoder`](https://contrib.scikit-learn.org/category_encoders/catboost.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.CatBoostEncoder'


@dataclasses.dataclass
class Count(Encoder):
    """Replaces each category with how often it is in the training rows.

    Wraps:
        [`category_encoders.CountEncoder`](https://contrib.scikit-learn.org/category_encoders/count.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.CountEncoder'


@dataclasses.dataclass
class Hashing(Encoder):
    """Hashes the categories into a fixed number of columns.

    Wraps:
        [`category_encoders.HashingEncoder`](https://contrib.scikit-learn.org/category_encoders/hashing.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.HashingEncoder'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'max_process': 1})


@dataclasses.dataclass
class Helmert(Encoder):
    """Compares each category to the mean of the categories before it.

    Wraps:
        [`category_encoders.HelmertEncoder`](https://contrib.scikit-learn.org/category_encoders/helmert.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.HelmertEncoder'


@dataclasses.dataclass
class JamesStein(Encoder):
    """Target encoding shrunk toward the overall mean (James-Stein).

    Wraps:
        [`category_encoders.JamesSteinEncoder`](https://contrib.scikit-learn.org/category_encoders/jamesstein.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.JamesSteinEncoder'


@dataclasses.dataclass
class LeaveOneOut(Encoder):
    """Target encoding that leaves out each row's own label.

    Wraps:
        [`category_encoders.LeaveOneOutEncoder`](https://contrib.scikit-learn.org/category_encoders/leaveoneout.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.LeaveOneOutEncoder'


@dataclasses.dataclass
class MEstimate(Encoder):
    """Target encoding shrunk toward the overall mean by m rows.

    Wraps:
        [`category_encoders.MEstimateEncoder`](https://contrib.scikit-learn.org/category_encoders/mestimate.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.MEstimateEncoder'


@dataclasses.dataclass
class PolynomialCoding(Encoder):
    """Contrasts categories as an ordered (polynomial) sequence.

    Wraps:
        [`category_encoders.PolynomialEncoder`](https://contrib.scikit-learn.org/category_encoders/polynomial.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.PolynomialEncoder'


@dataclasses.dataclass
class SumCoding(Encoder):
    """Compares each category to the mean of all categories (effect coding).

    Wraps:
        [`category_encoders.SumEncoder`](https://contrib.scikit-learn.org/category_encoders/sum.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.SumEncoder'


@dataclasses.dataclass
class Target(Encoder):
    """Replaces each category with the mean label of its training rows.

    Wraps:
        [`category_encoders.TargetEncoder`](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.TargetEncoder'


@dataclasses.dataclass
class WeightOfEvidence(Encoder):
    """Replaces each category with its weight of evidence (binary labels).

    Wraps:
        [`category_encoders.WOEEncoder`](https://contrib.scikit-learn.org/category_encoders/woe.html)
        from category_encoders.

    """

    contents: str = 'category_encoders.WOEEncoder'
