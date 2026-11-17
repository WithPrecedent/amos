"""Pretests of the assumptions and needs of techniques.

Contents:
    CheckAssumptions: checks every assumption and need of the techniques
        that has a pretest.
    CheckContinuity: checks whether rows bunch on one side of a cutoff.
    CheckDispersion: checks whether counts vary as a Poisson model assumes.
    CheckFeatures: checks whether the features have missing values or are
        not numbers.
    CheckInstrument: checks whether an instrument is strong enough.
    CheckLabel: checks whether the label is the kind that techniques are
        for.
    CheckLinearity: checks whether the label is linear in the features.
    CheckMissingness: checks whether values are missing completely at
        random.
    CheckMulticollinearity: checks whether features are nearly collinear.
    CheckNormality: checks whether the errors and the features are normal.
    CheckOverlap: checks whether every row could have been treated or not.
    CheckParallelTrends: checks whether treated units trended differently
        before treatment.
    CheckProportionalOdds: checks whether features have the same effect at
        every cutoff between ordered classes.
    CheckSampleSize: checks whether there are enough (or too many) rows and
        features.
    CheckScales: checks whether the features are on similar scales.
    CheckStationarity: checks whether each time series is stationary.

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
class CheckContinuity(Pretest):
    """Checks whether rows bunch on one side of a cutoff.

    It checks "continuity" for a regression discontinuity, with a binomial
    test of the rows just above and just below the "cutoff" of the
    "running" column (a simple form of McCrary's density test). Pass the
    inferer's "running" and "cutoff" if it is not the workflow's.

    """


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
class CheckInstrument(Pretest):
    """Checks whether an instrument is strong enough.

    It checks the relevance part of "valid_instrument": the F statistic of
    the "instrument" in the first-stage regression of the "treatment" must
    be at least 10. Pass the inferer's "treatment" and "instrument" if it is
    not the workflow's.

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
class CheckMissingness(Pretest):
    """Checks whether values are missing completely at random.

    It checks "missing_completely_at_random" with Little's test and the
    standardized differences between the rows with and without each missing
    value.

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
class CheckOverlap(Pretest):
    """Checks whether every row could have been treated or not.

    It checks "overlap": no more than 5% of the rows may have a propensity
    score (from a logistic regression of the "treatment" on the features)
    below 0.05 or above 0.95. Pass the inferer's "treatment" if it is not
    the workflow's.

    """


@dataclasses.dataclass
class CheckParallelTrends(Pretest):
    """Checks whether treated units trended differently before treatment.

    It checks "parallel_trends" for a difference in differences, with a t
    test of the trends of the treated and other units in the periods before
    the treatment. Pass the inferer's "time" and "treatment" if it is not
    the workflow's.

    """


@dataclasses.dataclass
class CheckProportionalOdds(Pretest):
    """Checks whether features have the same effect at every cutoff.

    It checks "proportional_odds" for an ordinal regression with Brant's
    test, which compares the coefficients of a logistic regression for each
    cutoff between ordered classes.

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


@dataclasses.dataclass
class CheckStationarity(Pretest):
    """Checks whether each time series is stationary.

    It checks "stationarity" with the augmented Dickey-Fuller and KPSS tests
    of the label and each numeric feature (with statsmodels).

    """
