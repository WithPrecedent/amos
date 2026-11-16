"""Samplers from imbalanced-learn.

Contents:
    Adasyn: adds synthetic rows where the rare class is hardest to learn.
    BorderlineSmote: adds synthetic rows near the border between classes.
    NearMiss: removes common rows that are far from the rare rows.
    RandomOver: duplicates random rows of the rare classes.
    RandomUnder: removes random rows of the common classes.
    Smote: adds synthetic rows between rare rows and their neighbors.
    SmoteEnn: `smote`, then removes rows that their neighbors misclassify.
    SmoteTomek: `smote`, then removes pairs of rows from different classes that
        are each other's nearest neighbors.
    TomekLinks: removes common rows that are nearest neighbors of rare rows.

"""

from __future__ import annotations

import dataclasses

from ._base import Sampler


@dataclasses.dataclass
class Adasyn(Sampler):
    """Adds synthetic rows where the rare class is hardest to learn.

    Wraps:
        [`imblearn.over_sampling.ADASYN`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.over_sampling.ADASYN'


@dataclasses.dataclass
class BorderlineSmote(Sampler):
    """Adds synthetic rows near the border between the classes.

    Wraps:
        [`imblearn.over_sampling.BorderlineSMOTE`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.over_sampling.BorderlineSMOTE'


@dataclasses.dataclass
class NearMiss(Sampler):
    """Removes rows of the common classes that are far from the rare rows.

    Wraps:
        [`imblearn.under_sampling.NearMiss`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.NearMiss.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.under_sampling.NearMiss'


@dataclasses.dataclass
class RandomOver(Sampler):
    """Duplicates random rows of the rare classes.

    Wraps:
        [`imblearn.over_sampling.RandomOverSampler`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.over_sampling.RandomOverSampler'


@dataclasses.dataclass
class RandomUnder(Sampler):
    """Removes random rows of the common classes.

    Wraps:
        [`imblearn.under_sampling.RandomUnderSampler`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.under_sampling.RandomUnderSampler'


@dataclasses.dataclass
class Smote(Sampler):
    """Adds synthetic rows between rare rows and their nearest neighbors.

    Wraps:
        [`imblearn.over_sampling.SMOTE`](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.over_sampling.SMOTE'


@dataclasses.dataclass
class SmoteEnn(Sampler):
    """Applies `smote` and then removes rows that their neighbors misclassify.

    Wraps:
        [`imblearn.combine.SMOTEENN`](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.combine.SMOTEENN'


@dataclasses.dataclass
class SmoteTomek(Sampler):
    """Applies `smote` and then removes Tomek links.

    Wraps:
        [`imblearn.combine.SMOTETomek`](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.combine.SMOTETomek'


@dataclasses.dataclass
class TomekLinks(Sampler):
    """Removes rows of the common class that are paired with rare rows.

    Wraps:
        [`imblearn.under_sampling.TomekLinks`](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.TomekLinks.html)
        from imbalanced-learn.

    """

    contents: str = 'imblearn.under_sampling.TomekLinks'
