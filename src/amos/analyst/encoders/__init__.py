"""Techniques that turn categories, text, and dates into numbers.

These are preprocessing techniques of the "analyst" stage. Each one wraps an
encoder from scikit-learn, category_encoders, or skrub, which is fitted to
the training rows only (and is given the label, which target encoders need)
and then encodes every row, so nothing is learned from the test rows. By
default, an encoder changes the categorical and text features (dates, for
`date_parts`). Pass `columns` as a parameter to choose others.

Contents:
    Encoder: base class for techniques that turn categories and text into
        numbers.
    BackwardDifference: compares each category to the one before it.
    BaseN: writes the number of each category in base N (4 by default).
    Binary: writes the number of each category in binary digits.
    CatBoost: target encoding in the manner of CatBoost, which limits leakage.
    Count: replaces each category with how often it is in the training rows.
    Hashing: hashes the categories into a fixed number of columns.
    Helmert: compares each category to the mean of the categories before it.
    JamesStein: target encoding shrunk toward the overall mean (James-Stein).
    LeaveOneOut: target encoding that leaves out each row's own label.
    MEstimate: target encoding shrunk toward the overall mean by m rows.
    PolynomialCoding: contrasts categories as an ordered (polynomial) sequence.
    SumCoding: compares each category to the mean of all categories (effect
        coding).
    Target: replaces each category with the mean label of its training rows.
    WeightOfEvidence: replaces each category with its weight of evidence (binary
        labels).
    OneHot: makes a column of 0s and 1s for each category (dummy variables).
    Ordinal: numbers the categories (0, 1, 2, ...) in sorted order.
    DateParts: splits dates and times into parts: year, month, day, and so on.
    Gap: encodes messy text as a mix of topics of its substrings.
    MinHash: encodes messy text by hashing its substrings, which is fast and
        robust.
    Tfidf: encodes text by the TF-IDF of its substrings, reduced with SVD.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'BackwardDifference',
    'BaseN',
    'Binary',
    'CatBoost',
    'Count',
    'DateParts',
    'Encoder',
    'Gap',
    'Hashing',
    'Helmert',
    'JamesStein',
    'LeaveOneOut',
    'MEstimate',
    'MinHash',
    'OneHot',
    'Ordinal',
    'PolynomialCoding',
    'SumCoding',
    'Target',
    'Tfidf',
    'WeightOfEvidence']

from ._base import Encoder
from ._category_encoders import (
    BackwardDifference,
    BaseN,
    Binary,
    CatBoost,
    Count,
    Hashing,
    Helmert,
    JamesStein,
    LeaveOneOut,
    MEstimate,
    PolynomialCoding,
    SumCoding,
    Target,
    WeightOfEvidence,
)
from ._sklearn import OneHot, Ordinal
from ._skrub import DateParts, Gap, MinHash, Tfidf

# The public names say that they are in `amos.encoders` (rather than in the
# private module that defines them, or in `amos.analyst.encoders`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.encoders.Name`. `amos` makes `amos.encoders` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.encoders'
