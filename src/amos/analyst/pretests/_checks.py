"""Pretests of the assumptions and needs of techniques.

Contents:
    CheckAssumptions: checks every assumption and need of the techniques
        that has a pretest.
    CheckDispersion: checks whether counts vary as a Poisson model assumes.
    CheckFeatures: checks whether the features have missing values or are
        not numbers.
    CheckLabel: checks whether the label is the kind that techniques are
        for.
    CheckLinearity: checks whether the label is linear in the features.
    CheckMulticollinearity: checks whether features are nearly collinear.
    CheckNormality: checks whether the errors and the features are normal.
    CheckSampleSize: checks whether there are enough (or too many) rows and
        features.
    CheckScales: checks whether the features are on similar scales.

"""

from __future__ import annotations

import dataclasses

from ._base import Pretest


@dataclasses.dataclass
class CheckAssumptions(Pretest):
    """Checks every assumption and need of the techniques that has a pretest.

    It runs every other pretest, but only for the conditions that the
    techniques have, so its table has a row for each condition of each
    technique (or, without techniques, for each condition that has a
    pretest). Name it as a step of the "analyst" stage before the models:

    ```ini
    [analyst]
    steps = split, scale, check, model
    check_techniques = check_assumptions
    ```

    """

    def _check(self) -> str | None:
        """Returns `None`, which checks every condition that has a pretest.

        Returns:
            `None`.

        """
        return None


@dataclasses.dataclass
class CheckDispersion(Pretest):
    """Checks whether counts vary as a Poisson model assumes.

    It checks "equidispersion": the Pearson dispersion of a Poisson
    regression of the counts must be 1.5 or less.

    """


@dataclasses.dataclass
class CheckFeatures(Pretest):
    """Checks whether the features have missing values or are not numbers.

    It checks "no_missing_values" and "numeric_features" (no categorical
    features, text, or dates).

    """


@dataclasses.dataclass
class CheckLabel(Pretest):
    """Checks whether the label is the kind that techniques are for.

    It checks "binary_label" (two classes), "multiclass_label" (three or
    more), "ordinal_label" (three or more ordered classes), and
    "count_label" (whole numbers of 0 or more).

    """


@dataclasses.dataclass
class CheckLinearity(Pretest):
    """Checks whether the label (or its log-odds) is linear in the features.

    It checks "linearity" with Ramsey's RESET test for a regressed label and
    Box and Tidwell's test for a label of two classes (see
    `assumptions.Linearity`).

    """


@dataclasses.dataclass
class CheckMulticollinearity(Pretest):
    """Checks whether any feature is nearly a linear function of the others.

    It checks "no_multicollinearity": the largest variance inflation factor
    of the numeric features must be 10 or less.

    """


@dataclasses.dataclass
class CheckNormality(Pretest):
    """Checks whether the errors of a regression and the features are normal.

    It checks "normal_errors" (the errors of a least squares fit of the
    label to the numeric features) and "normal_features" (most features are
    not skewed or heavy-tailed).

    """


@dataclasses.dataclass
class CheckSampleSize(Pretest):
    """Checks whether there are enough (or too many) rows and features.

    It checks "more_rows_than_features", and "min_rows", "max_rows", and
    "max_features", whose limits are attributes of each technique.

    """


@dataclasses.dataclass
class CheckScales(Pretest):
    """Checks whether the features are on similar scales.

    It checks "similar_scales": the largest standard deviation of the
    numeric features must be no more than 10 times the smallest.

    """
