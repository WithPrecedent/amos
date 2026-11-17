"""What techniques assume about the data, and the tests of each assumption.

Each assumption is a subclass of `conditions.Assumption`, named in snake
case, which a technique names in its `assumptions`. Those with a pretest
are checked on the real rows before a technique is applied, and those with
an audit are checked from a fitted model (see the `conditions` module).
Some assumptions cannot be tested from the data at all (such as that every
confounder is measured), and some only in part (such as the relevance, but
not the exclusion, of an instrument): those are listed with the techniques
that make them, and the part that can be tested is.

The tests need only numpy, scipy, and scikit-learn, except the test of
stationarity (statsmodels). With many rows, a test rejects even trivial
departures from an assumption, so an assumption fails only if its test
rejects it (at the 5% level) and the departure is large enough to matter
(see the thresholds below), as `auto.profile` judges the normality of
errors.

Contents:
    NormalErrors: the errors of a regression are normal.
    NormalFeatures: the features are normal.
    NoMulticollinearity: no feature is nearly a linear function of the
        others.
    SimilarScales: the features are on similar scales.
    Linearity: the label (or its log-odds) is a linear function of the
        features.
    ConstantVariance: the errors have the same variance for every row.
    IndependentErrors: the errors of different rows are independent.
    Equidispersion: the variance of counts equals their mean.
    MissingAtRandom: whether a value is missing depends only on values that
        are observed.
    Unconfoundedness: every common cause of the treatment and the outcome
        is among the features.
    CausalSufficiency: every common cause of the variables is measured.
    MissingCompletelyAtRandom: whether a value is missing has nothing to do
        with any value.
    ValidInstrument: the instrument changes the treatment and affects the
        outcome only through it.
    ParallelTrends: without the treatment, the treated and other rows would
        have changed alike.
    Continuity: the outcome changes smoothly at the cutoff except for the
        treatment.
    Overlap: every row could have been treated or not.
    Stationarity: the distribution of each series does not change over
        time.
    ProportionalOdds: each feature has the same effect on every cutoff
        between ordered classes.
    ProportionalHazards: each feature multiplies the hazard by the same
        amount at every time.
    box_tidwell_test: Box and Tidwell's test that log-odds are linear.
    brant_test: Brant's test that features have the same effect at every
        cutoff between ordered classes.
    breusch_pagan_test: Breusch and Pagan's test of constant variance.
    clustering_test: the intraclass correlation of errors within groups.
    density_test: a binomial test of the rows just above and below a
        cutoff.
    first_stage_test: the F test of the relevance of instruments.
    little_test: Little's test that values are missing completely at
        random.
    missing_difference: how much the rows with a missing value differ from
        the others.
    overlap_test: the share of rows whose propensity score is extreme.
    pretrend_test: a test of the trends of treated and other units before
        the treatment.
    reset_test: Ramsey's test that a regression is linear.
    schoenfeld_test: Grambsch and Therneau's test of proportional hazards.
    serial_correlation_test: Durbin and Watson's statistic and Breusch and
        Godfrey's test of serially correlated errors.
    stationarity_test: the augmented Dickey-Fuller and KPSS tests of a
        series.
    _MATERIAL: smallest share of variance (or deviance) that a departure
        from linearity or constant variance must explain to matter.
    _AUTOCORRELATION: smallest correlation of the errors of rows next to
        each other that matters.
    _CLUSTERED: smallest intraclass correlation of the errors that matters.
    _NORMAL_SHARE: largest share of features that may be skewed or
        heavy-tailed for features to count as normal.
    _LINEARITY: the name under which the test of linearity is cached.
    _IMBALANCE: largest standardized difference between the rows with and
        without a missing value.
    _WEAK_INSTRUMENT: smallest first-stage F statistic of a relevant
        instrument.
    _TREND_DRIFT: largest difference in the trends before the treatment.
    _DENSITY_SHARE: largest difference from one half in the share of rows
        above a cutoff.
    _NEARBY: fewest rows near a cutoff to test.
    _EXTREME_PROPENSITY, _EXTREME_SHARE: the propensity scores that are
        extreme, and the largest share of rows that may have them.
    _ODDS_DIFFERENCE: largest difference between a feature's coefficients
        at two cutoffs.
    _HAZARD_CORRELATION: largest correlation of Schoenfeld residuals with
        time.
    _EM_STEPS, _EM_TOLERANCE: the limits of expectation maximization.

"""
from __future__ import annotations

import dataclasses
import importlib
import warnings
from typing import Any, ClassVar

import camina
import numpy as np
import pandas as pd

from . import base, conditions, utilities

# Smallest share of the variance of the label (for linearity), of the squared
# errors (for constant variance), or of the deviance (for log-odds) that a
# departure must explain to matter.
_MATERIAL: float = 0.02
# Smallest correlation of the errors of rows next to each other (in the
# dataset's order) that matters. It is about the usual rule of thumb that
# Durbin and Watson's statistic should be between 1.5 and 2.5 (it is about 2
# times 1 minus the correlation).
_AUTOCORRELATION: float = 0.2
# Smallest intraclass correlation of the errors within groups that matters.
_CLUSTERED: float = 0.05
# Largest share of the features that may be skewed or heavy-tailed for the
# features to count as normal.
_NORMAL_SHARE: float = 0.5
# The name under which the test of linearity is kept in a cache, so that it is
# run once for every technique checked at the same time.
_LINEARITY: str = 'linearity'
# Largest standardized difference in a feature's mean between the rows with
# and without a missing value for values to count as missing completely at
# random (0.1 is the usual threshold of balance).
_IMBALANCE: float = 0.1
# Smallest first-stage F statistic of instruments that are not weak (Staiger
# and Stock's rule of thumb).
_WEAK_INSTRUMENT: float = 10.0
# Largest difference in the trends of the treated and other units before the
# treatment, in standard deviations of the label, for parallel trends.
_TREND_DRIFT: float = 0.1
# Largest difference from one half in the share of the rows near a cutoff
# that are above it, for no jump in the density at the cutoff.
_DENSITY_SHARE: float = 0.1
# Fewest rows near a cutoff for the test of the density.
_NEARBY: int = 20
# Propensity score below which (or above 1 minus which) a row's treatment is
# nearly certain, and the largest share of such rows for overlap.
_EXTREME_PROPENSITY: float = 0.05
_EXTREME_SHARE: float = 0.05
# Largest difference between a standardized feature's coefficients at two
# cutoffs between ordered classes, for proportional odds.
_ODDS_DIFFERENCE: float = 0.25
# Largest correlation of a feature's Schoenfeld residuals with the rank of
# the time, for proportional hazards.
_HAZARD_CORRELATION: float = 0.1
# Most steps of expectation maximization, and the change in the estimates
# below which they have settled.
_EM_STEPS: int = 200
_EM_TOLERANCE: float = 1e-6


""" Assumptions With Tests """


@dataclasses.dataclass(frozen = True)
class NormalErrors(conditions.Assumption):
    """The errors of a regression are normal.

    Before a model is fitted, the errors are those of a least squares fit of
    the label to the numeric features (see `auto.profile`). After it is
    fitted, they are the model's own errors on the rows it predicted. With
    many rows, normal errors matter little (the standard errors are about
    right without them), so they fail only if D'Agostino and Pearson's test
    rejects normality and they are materially skewed (beyond 1) or
    heavy-tailed (excess kurtosis beyond 3).

    """

    description: ClassVar[str] = 'the errors of a regression are normal'
    pretested_by: ClassVar[str] = 'check_normality'
    audited_by: ClassVar[str] = 'audit_residuals'
    affects: ClassVar[str] = 'inference'
    tests_used: ClassVar[str] = (
        "D'Agostino and Pearson's test, with the skewness and excess "
        "kurtosis of the errors")

    def audit(self, item: base.Dataset, technique: Any) -> conditions.Verdict:
        """Tests whether the errors of the fitted model are normal.

        Args:
            item: the dataset, with a fitted model and its predictions.
            technique: the model's technique, or `None`.

        Returns:
            The verdict.

        """
        auto = _auto()
        residuals = _residuals(item)
        if residuals is None:
            return self._verdict(
                technique, self.audited_by, None,
                'is only tested for a regression with predictions')
        shape = auto.error_shape(residuals.to_numpy())
        return self._shape_verdict(technique, self.audited_by, shape)

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether the errors of a least squares fit are normal.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        if found.task != 'regress':
            return self._verdict(
                technique, self.pretested_by, None,
                'is only tested for a regressed label')
        if found.normal is None:
            return self._verdict(
                technique, self.pretested_by, None,
                'has too few rows (or errors that do not vary) to test')
        shape = _auto().ErrorShape(
            skewness = found.skewness or 0.0,
            kurtosis = found.kurtosis or 0.0,
            normality = found.normality,
            normal = bool(found.normal),
            outliers = 0.0)
        return self._shape_verdict(technique, self.pretested_by, shape)

    def _shape_verdict(
        self,
        technique: Any,
        check: str,
        shape: Any) -> conditions.Verdict:
        """Returns the verdict for the shape of some errors.

        Args:
            technique: the technique, or `None`.
            check: name of the pretest or audit.
            shape: the `auto.ErrorShape` of the errors, or `None`.

        Returns:
            The verdict.

        """
        if shape is None:
            return self._verdict(
                technique, check, None,
                'has too few errors (or errors that do not vary) to test')
        values = (
            f'skewness {shape.skewness:.2f}, excess kurtosis '
            f'{shape.kurtosis:.2f}')
        if shape.outliers:
            values += f', {shape.outliers:.1%} outliers'
        if shape.normal:
            detail = f'has errors that are about normal ({values})'
        else:
            detail = f'assumes normal errors, but they are not ({values})'
        return self._verdict(
            technique, check, shape.normal, detail,
            statistic = shape.skewness,
            p_value = shape.normality,
            threshold = _auto()._ALPHA)


@dataclasses.dataclass(frozen = True)
class NormalFeatures(conditions.Assumption):
    """The features are normal.

    A feature is not normal if it is binary or materially skewed or
    heavy-tailed. The features are not normal if more than half of them are
    not.

    """

    description: ClassVar[str] = 'the features are normal'
    pretested_by: ClassVar[str] = 'check_normality'
    tests_used: ClassVar[str] = (
        'the skewness and excess kurtosis of each feature')

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether most of the features are normal.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        share = float(found.skewed)
        passed = share <= _NORMAL_SHARE
        detail = (
            f'has features that are mostly normal ({share:.0%} are not)'
            if passed else
            f'assumes normal features, but {share:.0%} of them are not')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = share, threshold = _NORMAL_SHARE)


@dataclasses.dataclass(frozen = True)
class NoMulticollinearity(conditions.Assumption):
    """No feature is nearly a linear function of the others.

    The largest variance inflation factor of the numeric features must be
    10 or less. Collinear features make a model's coefficients unstable and
    their standard errors large.

    """

    description: ClassVar[str] = (
        'no feature is nearly a linear function of the others')
    pretested_by: ClassVar[str] = 'check_multicollinearity'
    tests_used: ClassVar[str] = (
        'the variance inflation factor of each numeric feature')

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether any feature's variance inflation factor is large.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        limit = _auto()._VIF
        if found.vif is None:
            return self._verdict(
                technique, self.pretested_by, None,
                'has fewer than two numeric features to test')
        passed = not found.collinear
        detail = (
            f'has features that are not collinear (the largest variance '
            f'inflation factor is {found.vif:.1f})' if passed else
            f'assumes features that are not collinear, but the largest '
            f'variance inflation factor is {found.vif:.1f}')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = found.vif, threshold = limit)


@dataclasses.dataclass(frozen = True)
class SimilarScales(conditions.Assumption):
    """The features are on similar scales.

    The largest standard deviation of the numeric features must be no more
    than 10 times the smallest. Models that measure distances or penalize
    coefficients treat a feature on a large scale as more important. Put a
    scaler before them.

    """

    description: ClassVar[str] = 'the features are on similar scales'
    pretested_by: ClassVar[str] = 'check_scales'
    affects: ClassVar[str] = 'prediction'
    tests_used: ClassVar[str] = (
        'the ratio of the largest to the smallest standard deviation of the '
        'numeric features')

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether the numeric features are on similar scales.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        passed = not found.unscaled
        detail = (
            'has features on similar scales' if passed else
            'needs features on similar scales (scale them first)')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            threshold = _auto()._SCALES)


@dataclasses.dataclass(frozen = True)
class Linearity(conditions.Assumption):
    """The label (or, for two classes, its log-odds) is linear in the features.

    For a regressed label, Ramsey's RESET test adds the squares and cubes of
    the predictions of a least squares fit and tests whether they explain
    more of the label. For a label of two classes, Box and Tidwell's test
    adds x times the log of x for each feature whose values are all
    positive and tests whether they explain more of the log-odds. Either
    fails if the test rejects linearity and the added terms explain more
    than 2% of what the fit left unexplained. A label of more classes is not
    tested.

    """

    description: ClassVar[str] = (
        'the label (or its log-odds) is a linear function of the features')
    pretested_by: ClassVar[str] = 'check_linearity'
    tests_used: ClassVar[str] = (
        "Ramsey's RESET test (for a regressed label) or Box and Tidwell's "
        "test (for two classes)")

    def pretest(
        self,
        item: base.Dataset,
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:
        """Tests whether the label is linear in the features.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together. The
                test is run once and kept in it.

        Returns:
            The verdict.

        """
        if _LINEARITY not in cache:
            cache[_LINEARITY] = _test_linearity(item, found)
        result = cache[_LINEARITY]
        if isinstance(result, str):
            return self._verdict(technique, self.pretested_by, None, result)
        statistic, p_value, gain, test = result
        alpha = _auto()._ALPHA
        passed = not (p_value < alpha and gain > _MATERIAL)
        target = 'log-odds' if found.task == 'classify' else 'label'
        detail = (
            f'has a {target} that is about linear in the features ({test} '
            f'p = {p_value:.3f}, {gain:.1%} more explained)' if passed else
            f'assumes a {target} that is linear in the features, but it is '
            f'not ({test} p = {p_value:.3f}, {gain:.1%} more explained by '
            f'nonlinear terms)')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = statistic, p_value = p_value, threshold = alpha)


@dataclasses.dataclass(frozen = True)
class ConstantVariance(conditions.Assumption):
    """The errors have the same variance for every row (homoscedasticity).

    After a model is fitted, Breusch and Pagan's test regresses its squared
    errors on the numeric features and their squares. It fails if the test
    rejects constant variance and they explain more than 2% of the variance
    of the squared errors. Without constant variance, the standard errors are
    wrong (use robust standard errors or weighted least squares).

    """

    description: ClassVar[str] = (
        'the errors have the same variance for every row')
    audited_by: ClassVar[str] = 'audit_errors'
    affects: ClassVar[str] = 'inference'
    tests_used: ClassVar[str] = (
        "Breusch and Pagan's test, with the squares of the features (as in "
        "White's test)")

    def audit(self, item: base.Dataset, technique: Any) -> conditions.Verdict:
        """Tests whether the variance of the model's errors is constant.

        Args:
            item: the dataset, with a fitted model and its predictions.
            technique: the model's technique, or `None`.

        Returns:
            The verdict.

        """
        residuals = _residuals(item)
        if residuals is None:
            return self._verdict(
                technique, self.audited_by, None,
                'is only tested for a regression with predictions')
        x = _numeric_features(item, residuals.index)
        result = breusch_pagan_test(residuals.to_numpy(), x.to_numpy())
        if result is None:
            return self._verdict(
                technique, self.audited_by, None,
                'has too few rows or numeric features to test')
        statistic, p_value, share = result
        alpha = _auto()._ALPHA
        passed = not (p_value < alpha and share > _MATERIAL)
        detail = (
            f'has errors of about constant variance (Breusch-Pagan p = '
            f'{p_value:.3f})' if passed else
            f'assumes errors of constant variance, but the features explain '
            f'{share:.1%} of the variance of the squared errors '
            f'(Breusch-Pagan p = {p_value:.3f})')
        return self._verdict(
            technique, self.audited_by, passed, detail,
            statistic = statistic, p_value = p_value, threshold = alpha)


@dataclasses.dataclass(frozen = True)
class IndependentErrors(conditions.Assumption):
    """The errors of different rows are independent.

    After a model is fitted, its errors are tested in two ways, as the data
    allows. If the dataset has an `order` (such as a date), the errors are
    put in that order (within each group, if it has groups) and Breusch and
    Godfrey's test asks whether each is correlated with the one before it;
    Durbin and Watson's statistic is reported too (near 2 is independent).
    If the dataset has groups, the intraclass correlation of the errors
    within its first group column is tested. It fails if a test rejects
    independence and the correlation is large enough to matter (0.2 for
    rows next to each other, 0.05 within groups). A dataset with neither
    cannot be tested. Errors that are not independent make the standard
    errors too small (use clustered standard errors, or a model of groups
    such as `gee` or `mixedlm`).

    """

    description: ClassVar[str] = 'the errors of different rows are independent'
    audited_by: ClassVar[str] = 'audit_independence'
    affects: ClassVar[str] = 'inference'
    tests_used: ClassVar[str] = (
        "Breusch and Godfrey's test and Durbin and Watson's statistic (in "
        "the dataset's order), and the intraclass correlation of the errors "
        "(within its groups)")

    def audit(self, item: base.Dataset, technique: Any) -> conditions.Verdict:
        """Tests whether the model's errors are serially or group correlated.

        Args:
            item: the dataset, with a fitted model and its predictions.
            technique: the model's technique, or `None`.

        Returns:
            The verdict.

        """
        residuals = _residuals(item)
        if residuals is None:
            return self._verdict(
                technique, self.audited_by, None,
                'is only tested for a regression with predictions')
        ordered = item._order_column()
        if not ordered and not item.groups:
            return self._verdict(
                technique, self.audited_by, None,
                'cannot be tested without an order or groups (set "order" '
                'or "groups" in the "general" section)')
        alpha = _auto()._ALPHA
        found: list[str] = []
        failed: list[str] = []
        statistic = p_value = None
        groups = (
            item.data.loc[residuals.index, item.groups[0]]
            if item.groups else None)
        if ordered:
            serial = serial_correlation_test(
                residuals,
                item.data.loc[residuals.index, ordered[0]],
                groups)
            if serial is not None:
                watson, _statistic, p_value, rho = serial
                statistic = watson
                text = (
                    f'Durbin-Watson {watson:.2f}, Breusch-Godfrey p = '
                    f'{p_value:.3f}')
                found.append(text)
                if p_value < alpha and abs(rho) > _AUTOCORRELATION:
                    failed.append(
                        f'rows next to each other in {ordered[0]!r} are '
                        f'correlated ({text})')
        if groups is not None:
            clustered = clustering_test(residuals.to_numpy(), groups)
            if clustered is not None:
                icc, f_value, p_group = clustered
                text = (
                    f'intraclass correlation {icc:.2f} within '
                    f'{item.groups[0]!r}, p = {p_group:.3f}')
                found.append(text)
                if statistic is None:
                    statistic, p_value = f_value, p_group
                if p_group < alpha and icc > _CLUSTERED:
                    failed.append(f'rows of a group are correlated ({text})')
        if not found:
            return self._verdict(
                technique, self.audited_by, None,
                'has too few rows (or groups) to test')
        passed = not failed
        detail = (
            f'has independent errors ({"; ".join(found)})' if passed else
            f'assumes independent errors, but {"; and ".join(failed)}')
        return self._verdict(
            technique, self.audited_by, passed, detail,
            statistic = statistic, p_value = p_value, threshold = alpha)


@dataclasses.dataclass(frozen = True)
class Equidispersion(conditions.Assumption):
    """The variance of counts equals their mean, as a Poisson model assumes.

    It fails if the Pearson dispersion of a Poisson regression (see
    `auto.profile`) is above 1.5. Overdispersed counts make the standard
    errors too small (use a negative binomial model).

    """

    description: ClassVar[str] = 'the variance of counts equals their mean'
    pretested_by: ClassVar[str] = 'check_dispersion'
    affects: ClassVar[str] = 'inference'
    tests_used: ClassVar[str] = 'the Pearson dispersion of a Poisson regression'

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether counts are overdispersed.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        if not found.counts or found.dispersion is None:
            return self._verdict(
                technique, self.pretested_by, None,
                'is only tested for a label of counts')
        passed = not found.overdispersed
        detail = (
            f'has counts whose variance is about their mean (dispersion '
            f'{found.dispersion:.2f})' if passed else
            f'assumes that the variance of the counts equals their mean, '
            f'but they are overdispersed (dispersion {found.dispersion:.2f})')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = found.dispersion, threshold = _auto()._DISPERSION)


""" Assumptions That Cannot Be Tested """


@dataclasses.dataclass(frozen = True)
class MissingAtRandom(conditions.Assumption):
    """Whether a value is missing depends only on values that are observed.

    Filling a value from the other features (as an iterative or nearest
    neighbors imputer does) is unbiased if this is so. Whether a value's
    being missing depends on the value itself cannot be seen, since the
    value is missing, so this cannot be tested from the data.

    """

    description: ClassVar[str] = (
        'whether a value is missing depends only on values that are observed')


@dataclasses.dataclass(frozen = True)
class Unconfoundedness(conditions.Assumption):
    """Every common cause of the treatment and the outcome is a feature.

    Also called ignorability, or no unmeasured confounding. A cause that was
    not measured leaves no trace in the data, so this cannot be tested: it
    is argued from how the data came to be.

    """

    description: ClassVar[str] = (
        'every common cause of the treatment and the outcome is a feature')
    affects: ClassVar[str] = 'inference'


@dataclasses.dataclass(frozen = True)
class CausalSufficiency(conditions.Assumption):
    """Every common cause of the variables is measured.

    A method of causal discovery that assumes it can mistake a hidden common
    cause for a link between two variables. Like unconfoundedness, it
    cannot be tested from the data.

    """

    description: ClassVar[str] = (
        'every common cause of the variables is measured')
    affects: ClassVar[str] = 'inference'


""" Assumptions That Can Be Partly Tested """


@dataclasses.dataclass(frozen = True)
class MissingCompletelyAtRandom(conditions.Assumption):
    """Whether a value is missing has nothing to do with any value.

    Filling a value with the mean, median, or most common value of its
    column (or dropping its row) is unbiased only if this is so. Little's
    test asks whether the means of the numeric features differ between the
    patterns of missing values (with the means and covariances estimated by
    expectation maximization). It fails if the test rejects it and the rows
    that lack a feature differ from the rest by more than 0.1 standard
    deviations in the mean of another feature. A test can show that values
    are not missing completely at random, but not that they are.

    """

    description: ClassVar[str] = (
        'whether a value is missing has nothing to do with any value')
    pretested_by: ClassVar[str] = 'check_missingness'
    tests_used: ClassVar[str] = (
        "Little's test, and the standardized differences in the means of the "
        "features between the rows with and without each missing value")

    def pretest(
        self,
        item: base.Dataset,
        technique: Any,
        found: Any,  # noqa: ARG002
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether values are missing completely at random.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        x = _numeric_features(item, _auto().profiled_rows(item))
        if not x.isna().to_numpy().any():
            return self._verdict(
                technique, self.pretested_by, None,
                'has no missing values in the numeric features to test')
        result = little_test(x.to_numpy())
        if result is None:
            return self._verdict(
                technique, self.pretested_by, None,
                'has too few rows (or patterns of missing values) to test')
        statistic, p_value, _freedom = result
        difference = missing_difference(x)
        alpha = _auto()._ALPHA
        passed = not (p_value < alpha and difference > _IMBALANCE)
        detail = (
            f"has values that may be missing completely at random (Little's "
            f'test p = {p_value:.3f})' if passed else
            f"assumes values missing completely at random, but Little's test "
            f'rejects it (p = {p_value:.3f}) and the rows that lack a feature '
            f'differ from the rest by {difference:.2f} standard deviations')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = statistic, p_value = p_value, threshold = alpha)


@dataclasses.dataclass(frozen = True)
class ValidInstrument(conditions.Assumption):
    """The instrument changes the treatment and affects the outcome only so.

    The instrument must be relevant (it changes the treatment), and it must
    be excluded (it affects the outcome only through the treatment, and
    shares no cause with the outcome). Relevance is tested: the first-stage
    regression of the "treatment" on the "instrument" (and the other
    features) must have an F statistic for the instruments of at least 10
    (Staiger and Stock's rule), or the instrument is weak. Exclusion cannot
    be tested from the data.

    """

    description: ClassVar[str] = (
        'the instrument changes the treatment and affects the outcome only '
        'through it')
    pretested_by: ClassVar[str] = 'check_instrument'
    affects: ClassVar[str] = 'inference'
    tests_used: ClassVar[str] = (
        'the F statistic of the instruments in the first-stage regression of '
        'the treatment (whether the instrument affects the outcome only '
        'through the treatment cannot be tested)')

    def pretest(
        self,
        item: base.Dataset,
        technique: Any,
        found: Any,  # noqa: ARG002
        cache: dict[str, Any]) -> conditions.Verdict:
        """Tests whether the instrument is strong enough.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together, with
                the "parameters" of the technique.

        Returns:
            The verdict.

        """
        parameters = conditions.parameters_of(technique, cache)
        treatment = parameters.get('treatment')
        instruments = [str(i) for i in camina.listify(
            parameters.get('instrument'))]
        columns = [treatment, *instruments]
        if not treatment or not instruments or not all(
            column in item.data.columns for column in columns):
            return self._verdict(
                technique, self.pretested_by, None,
                'needs the "treatment" and "instrument" columns to test')
        controls = [c for c in _numeric_names(item) if c not in columns]
        frame = item.data.loc[_real_rows(item), [*columns, *controls]]
        frame = frame.apply(pd.to_numeric, errors = 'coerce').dropna()
        result = first_stage_test(
            frame[treatment].to_numpy(),
            frame[instruments].to_numpy(),
            frame[controls].to_numpy())
        if result is None:
            return self._verdict(
                technique, self.pretested_by, None,
                'has too few rows (or instruments that do not vary) to test')
        f_value, p_value = result
        passed = f_value >= _WEAK_INSTRUMENT
        detail = (
            f'has a relevant instrument (first-stage F = {f_value:.1f})'
            if passed else
            f'assumes a relevant instrument, but it is weak (first-stage F '
            f'= {f_value:.1f}, below {_WEAK_INSTRUMENT:.0f})')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = f_value, p_value = p_value,
            threshold = _WEAK_INSTRUMENT)


@dataclasses.dataclass(frozen = True)
class ParallelTrends(conditions.Assumption):
    """Without the treatment, the treated and other rows would change alike.

    A difference in differences attributes any difference in how the two
    groups changed to the treatment. What would have happened after the
    treatment cannot be seen, but the trends before it can be compared: in
    the periods before each unit's treatment, a regression of the label on
    the "time", whether the unit is ever treated, and their product tests
    whether the treated units' label already trended differently. It fails
    if the test rejects parallel trends and the difference adds up to more
    than 0.1 standard deviations of the label over those periods. It needs
    at least two periods before the treatment.

    """

    description: ClassVar[str] = (
        'without the treatment, the treated and other rows would have '
        'changed alike')
    pretested_by: ClassVar[str] = 'check_parallel_trends'
    affects: ClassVar[str] = 'inference'
    tests_used: ClassVar[str] = (
        'a t test of whether the treated and other units trended differently '
        'before the treatment (which cannot show how they would have changed '
        'after it)')

    def pretest(
        self,
        item: base.Dataset,
        technique: Any,
        found: Any,  # noqa: ARG002
        cache: dict[str, Any]) -> conditions.Verdict:
        """Tests whether the trends before the treatment were parallel.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together, with
                the "parameters" of the technique.

        Returns:
            The verdict.

        """
        parameters = conditions.parameters_of(technique, cache)
        time = parameters.get('time')
        treatment = parameters.get('treatment')
        if not (time and treatment and item.has_label) or not all(
            c in item.data.columns for c in (time, treatment)):
            return self._verdict(
                technique, self.pretested_by, None,
                'needs the "time" and "treatment" columns and a label to test')
        rows = _real_rows(item)
        result = pretrend_test(
            pd.to_numeric(item.data.loc[rows, time], errors = 'coerce'),
            pd.to_numeric(item.data.loc[rows, treatment], errors = 'coerce'),
            pd.to_numeric(item.y.loc[rows], errors = 'coerce'))
        if result is None:
            return self._verdict(
                technique, self.pretested_by, None,
                'needs at least two periods before the treatment to test')
        t_value, p_value, drift = result
        alpha = _auto()._ALPHA
        passed = not (p_value < alpha and drift > _TREND_DRIFT)
        detail = (
            f'has parallel trends before the treatment (p = {p_value:.3f})'
            if passed else
            f'assumes parallel trends, but before the treatment the treated '
            f'units trended differently by {drift:.2f} standard deviations '
            f'(p = {p_value:.3f})')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = t_value, p_value = p_value, threshold = alpha)


@dataclasses.dataclass(frozen = True)
class Continuity(conditions.Assumption):
    """The outcome changes smoothly at the cutoff except for the treatment.

    A regression discontinuity attributes a jump at the cutoff to the
    treatment, so nothing else may jump there. Rows that sort themselves to
    one side of the cutoff (such as by retaking a test) would make the
    number of rows jump, which is tested: of the rows within a quarter of a
    standard deviation of the "running" column from the "cutoff" (0 by
    default), a binomial test asks whether as many are above as below. It
    fails if the test rejects it and more than 60% are on one side. That
    the outcome itself would change smoothly cannot be tested.

    """

    description: ClassVar[str] = (
        'the outcome would change smoothly at the cutoff without the '
        'treatment')
    pretested_by: ClassVar[str] = 'check_continuity'
    affects: ClassVar[str] = 'inference'
    tests_used: ClassVar[str] = (
        "a binomial test of the numbers of rows just above and just below "
        "the cutoff (a simple form of McCrary's density test, which cannot "
        "show that the outcome would change smoothly)")

    def pretest(
        self,
        item: base.Dataset,
        technique: Any,
        found: Any,  # noqa: ARG002
        cache: dict[str, Any]) -> conditions.Verdict:
        """Tests whether rows bunch on one side of the cutoff.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together, with
                the "parameters" of the technique.

        Returns:
            The verdict.

        """
        parameters = conditions.parameters_of(technique, cache)
        running = parameters.get('running')
        if not running or running not in item.data.columns:
            return self._verdict(
                technique, self.pretested_by, None,
                'needs the "running" column to test')
        cutoff = float(parameters.get('cutoff') or 0.0)
        values = pd.to_numeric(
            item.data.loc[_real_rows(item), running], errors = 'coerce')
        result = density_test(values.dropna().to_numpy(), cutoff)
        if result is None:
            return self._verdict(
                technique, self.pretested_by, None,
                'has too few rows near the cutoff to test')
        share, p_value, count = result
        alpha = _auto()._ALPHA
        passed = not (p_value < alpha and abs(share - 0.5) > _DENSITY_SHARE)
        detail = (
            f'has as many rows just above the cutoff as below ({share:.0%} '
            f'of {count} above, p = {p_value:.3f})' if passed else
            f'assumes no jump at the cutoff, but {share:.0%} of the {count} '
            f'rows near it are above it (p = {p_value:.3f})')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = share, p_value = p_value, threshold = alpha)


""" Assumptions That Can Be Tested Without a Model """


@dataclasses.dataclass(frozen = True)
class Overlap(conditions.Assumption):
    """Every row could have been treated or not (positivity).

    Rows whose features make treatment certain (or impossible) have nothing
    to be compared with, so their effects are extrapolated. A logistic
    regression of a "treatment" of two values on the numeric features finds
    each row's propensity score (its chance of treatment). It fails if more
    than 5% of the rows have a score below 0.05 or above 0.95.

    """

    description: ClassVar[str] = 'every row could have been treated or not'
    pretested_by: ClassVar[str] = 'check_overlap'
    affects: ClassVar[str] = 'inference'
    tests_used: ClassVar[str] = (
        'the share of rows whose propensity score (from a logistic '
        'regression of the treatment on the features) is below 0.05 or above '
        '0.95')

    def pretest(
        self,
        item: base.Dataset,
        technique: Any,
        found: Any,  # noqa: ARG002
        cache: dict[str, Any]) -> conditions.Verdict:
        """Tests whether the propensity scores are rarely extreme.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together, with
                the "parameters" of the technique.

        Returns:
            The verdict.

        """
        parameters = conditions.parameters_of(technique, cache)
        treatment = parameters.get('treatment')
        if not treatment or treatment not in item.data.columns:
            return self._verdict(
                technique, self.pretested_by, None,
                'needs the "treatment" column to test')
        excluded = {treatment, *map(str, camina.listify(
            parameters.get('instrument')))}
        controls = [c for c in _numeric_names(item) if c not in excluded]
        frame = item.data.loc[_real_rows(item), [treatment, *controls]]
        frame = frame.dropna()
        values = frame[treatment]
        if values.nunique() != 2:  # noqa: PLR2004
            return self._verdict(
                technique, self.pretested_by, None,
                'is only tested for a treatment of two values')
        treated = (values == sorted(values.unique())[1]).to_numpy()
        share = overlap_test(treated, frame[controls].to_numpy(dtype = float))
        passed = share <= _EXTREME_SHARE
        detail = (
            f'has rows that could each have been treated or not ({share:.1%} '
            f'have extreme propensity scores)' if passed else
            f'assumes that every row could have been treated or not, but '
            f'{share:.1%} of the rows have propensity scores below '
            f'{_EXTREME_PROPENSITY} or above {1 - _EXTREME_PROPENSITY}')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = share, threshold = _EXTREME_SHARE)


@dataclasses.dataclass(frozen = True)
class Stationarity(conditions.Assumption):
    """The distribution of each series does not change over time.

    The label and each numeric feature, in the order of the rows, are tested
    with the augmented Dickey-Fuller test (whose null is a unit root) and
    the KPSS test (whose null is stationarity), from statsmodels. A series
    is not stationary if the first does not reject a unit root and the
    second rejects stationarity. It fails if any series is not stationary.

    """

    description: ClassVar[str] = (
        'the distribution of each series does not change over time')
    pretested_by: ClassVar[str] = 'check_stationarity'
    tests_used: ClassVar[str] = (
        'the augmented Dickey-Fuller and KPSS tests of each series (it is not '
        'stationary if the first does not reject a unit root and the second '
        'rejects stationarity)')

    def pretest(
        self,
        item: base.Dataset,
        technique: Any,
        found: Any,  # noqa: ARG002
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether each series is stationary.

        Args:
            item: the dataset, whose rows are a time series in order.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        if not utilities.importable('statsmodels'):
            return self._verdict(
                technique, self.pretested_by, None,
                'needs statsmodels to test')
        rows = _real_rows(item)
        columns = _numeric_names(item)
        if item.has_label and pd.api.types.is_numeric_dtype(item.y.dtype):
            columns = [item.require_label(), *columns]
        results = {
            column: stationarity_test(
                pd.to_numeric(item.data.loc[rows, column], errors = 'coerce')
                .dropna().to_numpy(dtype = float))
            for column in columns}
        tested = {c: r for c, r in results.items() if r is not None}
        if not tested:
            return self._verdict(
                technique, self.pretested_by, None,
                'has too few rows (or series that vary) to test')
        alpha = _auto()._ALPHA
        drifting = [
            column for column, (adf, kpss) in tested.items()
            if adf > alpha and kpss < alpha]
        passed = not drifting
        detail = (
            f'has stationary series ({len(tested)} tested)' if passed else
            f'assumes stationary series, but {drifting} are not (augmented '
            f'Dickey-Fuller and KPSS tests)')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = len(drifting), threshold = alpha)


@dataclasses.dataclass(frozen = True)
class ProportionalOdds(conditions.Assumption):
    """Each feature has the same effect on every cutoff between classes.

    An ordinal regression has one coefficient for each feature, which is the
    same for the odds of being above each class. Brant's test fits a
    logistic regression for each cutoff between ordered classes (whether a
    row is above it) and asks whether their coefficients (of standardized
    features) differ. It fails if the test rejects it and some feature's
    coefficients differ by more than 0.25.

    """

    description: ClassVar[str] = (
        'each feature has the same effect on the odds of being above every '
        'class')
    pretested_by: ClassVar[str] = 'check_proportional_odds'
    tests_used: ClassVar[str] = (
        "Brant's test, which compares the coefficients of a logistic "
        "regression for each cutoff between ordered classes")

    def pretest(
        self,
        item: base.Dataset,
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether the features have the same effect at every cutoff.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        classes = found.classes or 0
        if found.task != 'classify' or classes < 3:  # noqa: PLR2004
            return self._verdict(
                technique, self.pretested_by, None,
                'is only tested for three or more ordered classes')
        rows = _auto().profiled_rows(item)
        frame = _numeric_features(item, rows).join(
            item.y.loc[rows].rename('__label__')).dropna()
        label = frame.pop('__label__')
        dtype = item.y.dtype
        order = (
            list(dtype.categories)
            if isinstance(dtype, pd.CategoricalDtype) and dtype.ordered
            else sorted(label.unique()))
        codes = label.map({value: i for i, value in enumerate(order)})
        result = brant_test(
            frame.to_numpy(dtype = float), codes.to_numpy(dtype = int))
        if result is None:
            return self._verdict(
                technique, self.pretested_by, None,
                'has too few rows (or features that vary) to test')
        statistic, p_value, difference = result
        alpha = _auto()._ALPHA
        passed = not (p_value < alpha and difference > _ODDS_DIFFERENCE)
        detail = (
            f"has features with about the same effect at every cutoff "
            f"(Brant's test p = {p_value:.3f})" if passed else
            f"assumes that each feature has the same effect at every cutoff, "
            f"but their coefficients differ by up to {difference:.2f} "
            f"(Brant's test p = {p_value:.3f})")
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = statistic, p_value = p_value, threshold = alpha)


""" Assumptions That Are Tested From a Fitted Model """


@dataclasses.dataclass(frozen = True)
class ProportionalHazards(conditions.Assumption):
    """Each feature multiplies the hazard by the same amount at every time.

    After a Cox model (`cox`) is fitted, Grambsch and Therneau's test asks
    whether the Schoenfeld residuals of each feature are correlated with the
    rank of the time of the event, which they would be if the feature's
    effect changed over time. It fails if the test rejects it for some
    feature and the correlation is larger than 0.1.

    """

    description: ClassVar[str] = (
        'each feature multiplies the hazard by the same amount at every time')
    audited_by: ClassVar[str] = 'audit_proportional_hazards'
    tests_used: ClassVar[str] = (
        "Grambsch and Therneau's test of the correlation of each feature's "
        "Schoenfeld residuals with the rank of the time")

    def audit(self, item: base.Dataset, technique: Any) -> conditions.Verdict:
        """Tests whether each feature's effect is constant over time.

        Args:
            item: the dataset, with a fitted Cox model.
            technique: the model's technique, or `None`.

        Returns:
            The verdict.

        """
        results: Any = getattr(item.model, 'results', None)
        residuals = getattr(results, 'schoenfeld_residuals', None)
        if results is None or residuals is None:
            return self._verdict(
                technique, self.audited_by, None,
                'needs a fitted Cox model (cox) to test')
        times = np.asarray(results.model.endog, dtype = float)
        tested = schoenfeld_test(np.asarray(residuals, dtype = float), times)
        if not tested:
            return self._verdict(
                technique, self.audited_by, None,
                'has too few events to test')
        known = getattr(item.model, 'feature_names_in_', None)
        names = (
            [f'x{i}' for i in range(len(tested))] if known is None
            else [str(name) for name in known])
        alpha = _auto()._ALPHA
        failed = [
            name for name, (correlation, _, p_value) in zip(
                names, tested, strict = False)
            if p_value < alpha and abs(correlation) > _HAZARD_CORRELATION]
        smallest = min(p_value for _, _, p_value in tested)
        passed = not failed
        detail = (
            f'has features whose effects are about constant over time '
            f'(smallest p = {smallest:.3f})' if passed else
            f'assumes proportional hazards, but the effects of {failed} '
            f'change over time (Grambsch and Therneau)')
        return self._verdict(
            technique, self.audited_by, passed, detail,
            p_value = smallest, threshold = alpha)


""" Public Functions """


def box_tidwell_test(
    x: pd.DataFrame,
    y: pd.Series) -> tuple[float, float, float] | None:
    """Tests whether the log-odds of two classes are linear in the features.

    Box and Tidwell's test adds x times the log of x for each feature whose
    values are all positive (and not only 0s and 1s), and compares the fits
    of logistic regressions with and without them by a likelihood ratio.
    The regressions are not penalized (in effect).

    Args:
        x: the numeric features of the rows, without missing values.
        y: the label of the same rows, with two classes.

    Returns:
        The likelihood ratio statistic, its p-value, and the share of the
            deviance of the linear fit that the added terms explain, or
            `None` if no feature can be tested or the classes are
            separated.

    """
    stats = importlib.import_module('scipy.stats')
    linear_model = importlib.import_module('sklearn.linear_model')
    metrics = importlib.import_module('sklearn.metrics')
    varied = x.loc[:, x.std() > 0]
    positive = [
        c for c in varied.columns
        if (varied[c] > 0).all() and varied[c].nunique() > 2]  # noqa: PLR2004
    if not positive or y.nunique() != 2:  # noqa: PLR2004
        return None
    codes = (y == sorted(y.unique())[1]).astype(int).to_numpy()
    base_terms = _standardized(varied.to_numpy(dtype = float))
    added = varied[positive].to_numpy(dtype = float)
    added = _standardized(added * np.log(added))
    likelihoods = []
    for design in (base_terms, np.column_stack([base_terms, added])):
        model = linear_model.LogisticRegression(C = 1e10, max_iter = 1000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(design, codes)
        probabilities = model.predict_proba(design)
        likelihoods.append(-metrics.log_loss(
            codes, probabilities, normalize = False))
    null, full = likelihoods
    if -null < 1e-6:  # noqa: PLR2004
        return None
    statistic = max(2 * (full - null), 0.0)
    p_value = float(stats.chi2.sf(statistic, len(positive)))
    return float(statistic), p_value, float((full - null) / -null)


def brant_test(
    x: np.ndarray,
    codes: np.ndarray) -> tuple[float, float, float] | None:
    """Tests whether features have the same effect at every cutoff.

    Brant's test fits a logistic regression (in effect not penalized) of
    whether a row is above each cutoff between ordered classes, on the
    standardized features, and tests whether the coefficients of the
    cutoffs are equal with a Wald test, from the covariance of the
    coefficients of every pair of regressions.

    Args:
        x: the numeric features, without missing values.
        codes: the position of each row's class among the ordered classes
            (0 for the lowest).

    Returns:
        The chi-squared statistic, its p-value, and the largest difference
            between the coefficients of a feature at two cutoffs, or `None`
            if there are fewer than three classes or no feature varies.

    """
    stats = importlib.import_module('scipy.stats')
    linear_model = importlib.import_module('sklearn.linear_model')
    values = np.asarray(x, dtype = float)
    values = _standardized(values[:, values.std(axis = 0) > 0])
    rows, width = values.shape
    cutoffs = int(codes.max()) if len(codes) else 0
    if width == 0 or cutoffs < 2 or rows <= width + cutoffs:  # noqa: PLR2004
        return None
    design = np.column_stack([np.ones(rows), values])
    slopes, chances = [], []
    for cutoff in range(cutoffs):
        above = (codes > cutoff).astype(int)
        if above.min() == above.max():
            return None
        model = linear_model.LogisticRegression(C = 1e10, max_iter = 1000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(values, above)
        slopes.append(model.coef_[0])
        chances.append(model.predict_proba(values)[:, 1])
    inverses = [
        np.linalg.pinv((design * (p * (1 - p))[:, None]).T @ design)
        for p in chances]
    size = cutoffs * width
    covariance = np.zeros((size, size))
    for first in range(cutoffs):
        for second in range(first, cutoffs):
            # Being above the higher cutoff means being above the lower one.
            weights = chances[second] - chances[first] * chances[second]
            block = (
                inverses[first] @ (design * weights[:, None]).T @ design
                @ inverses[second])[1:, 1:]
            covariance[
                first * width:(first + 1) * width,
                second * width:(second + 1) * width] = block
            covariance[
                second * width:(second + 1) * width,
                first * width:(first + 1) * width] = block.T
    contrasts = np.zeros(((cutoffs - 1) * width, size))
    for cutoff in range(1, cutoffs):
        place = (cutoff - 1) * width
        contrasts[place:place + width, :width] = np.eye(width)
        contrasts[place:place + width, cutoff * width:(cutoff + 1) * width] = (
            -np.eye(width))
    stacked = np.concatenate(slopes)
    differences = contrasts @ stacked
    statistic = float(
        differences @ np.linalg.pinv(contrasts @ covariance @ contrasts.T)
        @ differences)
    freedom = (cutoffs - 1) * width
    spread = np.ptp(np.vstack(slopes), axis = 0)
    return (
        statistic, float(stats.chi2.sf(statistic, freedom)),
        float(spread.max()))


def breusch_pagan_test(
    residuals: np.ndarray,
    x: np.ndarray) -> tuple[float, float, float] | None:
    """Tests whether the variance of errors depends on the features.

    Breusch and Pagan's test (in Koenker's form, which does not assume
    normal errors) regresses the squared errors on the features and their
    squares (as White's test does, without the products of pairs of
    features), so that it finds variance that grows away from the middle of
    a feature as well as with it. Its statistic is the number of rows times
    the R squared of that fit.

    Args:
        residuals: the errors.
        x: the numeric features of the same rows.

    Returns:
        The statistic, its p-value, and the R squared, or `None` if there
            are too few rows or no features that vary.

    """
    stats = importlib.import_module('scipy.stats')
    complete = np.isfinite(residuals) & np.isfinite(x).all(axis = 1)
    squared = residuals[complete] ** 2
    features = _standardized(np.asarray(x[complete], dtype = float))
    design = _design(np.column_stack([features, features ** 2]))
    rows, columns = design.shape
    if columns < 2 or rows <= columns + 1:  # noqa: PLR2004
        return None
    share = _r_squared(design, squared)
    statistic = rows * share
    return float(statistic), float(stats.chi2.sf(statistic, columns - 1)), share


def clustering_test(
    residuals: np.ndarray,
    groups: pd.Series) -> tuple[float, float, float] | None:
    """Tests whether the errors of rows in the same group are correlated.

    The intraclass correlation (ICC(1)) is found from a one-way analysis of
    variance of the errors by group, whose F test is its test.

    Args:
        residuals: the errors.
        groups: the group of each row (a missing group is a group of its
            own row).

    Returns:
        The intraclass correlation, the F statistic, and its p-value, or
            `None` if there are fewer than two groups or no group has more
            than one row.

    """
    stats = importlib.import_module('scipy.stats')
    labels = np.array(groups.astype(object).to_numpy(), copy = True)
    missing = pd.isna(labels)
    labels[missing] = [f'__row__{i}' for i in np.flatnonzero(missing)]
    codes = pd.factorize(labels)[0]
    sizes = np.bincount(codes)
    count, rows = len(sizes), len(residuals)
    if count < 2 or rows <= count:  # noqa: PLR2004
        return None
    means = np.bincount(codes, weights = residuals) / sizes
    grand = residuals.mean()
    between = float((sizes * (means - grand) ** 2).sum() / (count - 1))
    within = float(((residuals - means[codes]) ** 2).sum() / (rows - count))
    if within <= 0:
        return None
    typical = (rows - (sizes ** 2).sum() / rows) / (count - 1)
    icc = (between - within) / (between + (typical - 1) * within)
    f_value = between / within
    p_value = float(stats.f.sf(f_value, count - 1, rows - count))
    return float(icc), float(f_value), p_value


def density_test(
    values: np.ndarray,
    cutoff: float) -> tuple[float, float, int] | None:
    """Tests whether as many rows are just above a cutoff as just below it.

    The rows within a quarter of a standard deviation of the cutoff are
    counted on each side, and a binomial test asks whether half are above
    it, as they would be (about) if the density of the values did not jump
    at the cutoff. It is a simple form of McCrary's density test.

    Args:
        values: the running variable of each row.
        cutoff: the cutoff.

    Returns:
        The share of the nearby rows that are at or above the cutoff, the
            p-value of the binomial test, and the number of nearby rows, or
            `None` if there are fewer than 20 of them.

    """
    stats = importlib.import_module('scipy.stats')
    numbers = np.asarray(values, dtype = float)
    numbers = numbers[np.isfinite(numbers)]
    if len(numbers) < 2:  # noqa: PLR2004
        return None
    width = 0.25 * numbers.std()
    near = numbers[np.abs(numbers - cutoff) <= width]
    if width <= 0 or len(near) < _NEARBY:
        return None
    above = int((near >= cutoff).sum())
    p_value = float(stats.binomtest(above, len(near), 0.5).pvalue)
    return above / len(near), p_value, len(near)


def first_stage_test(
    treatment: np.ndarray,
    instruments: np.ndarray,
    controls: np.ndarray) -> tuple[float, float] | None:
    """Tests whether instruments predict the treatment (their relevance).

    The treatment is regressed on the instruments and the controls, and on
    the controls alone, and an F test compares the fits. Below 10, the
    instruments are weak (Staiger and Stock's rule of thumb).

    Args:
        treatment: the treatment of each row.
        instruments: the instruments of the same rows, one column for each.
        controls: the other features of the same rows (which may have no
            columns).

    Returns:
        The F statistic of the instruments and its p-value, or `None` if
            there are too few rows or no instrument varies.

    """
    stats = importlib.import_module('scipy.stats')
    y = np.asarray(treatment, dtype = float)
    others = np.asarray(controls, dtype = float).reshape(len(y), -1)
    restricted = _design(others)
    full = _design(np.column_stack([others, np.asarray(
        instruments, dtype = float).reshape(len(y), -1)]))
    rows, columns = full.shape
    count = columns - restricted.shape[1]
    if count < 1 or rows <= columns + 1:
        return None
    left = _residual_sum(full, y)
    before = _residual_sum(restricted, y)
    if left <= 0:
        return np.inf, 0.0
    f_value = ((before - left) / count) / (left / (rows - columns))
    return float(f_value), float(stats.f.sf(f_value, count, rows - columns))


def little_test(x: np.ndarray) -> tuple[float, float, int] | None:
    """Tests whether values are missing completely at random.

    Little's test estimates the means and covariances of the columns by
    expectation maximization (as if they were normal), and then compares
    the means of the observed columns in each pattern of missing values
    with them. Its statistic is the sum, over the patterns, of the number
    of rows times the squared Mahalanobis distance between the means.

    Args:
        x: the numeric columns, with missing values as `nan`.

    Returns:
        The chi-squared statistic, its p-value, and its degrees of freedom,
            or `None` if there are too few rows or patterns to test.

    """
    stats = importlib.import_module('scipy.stats')
    values = np.asarray(x, dtype = float)
    observed = ~np.isnan(values)
    values = values[:, observed.sum(axis = 0) > 1]
    values = values[~np.isnan(values).all(axis = 1)]
    rows, width = values.shape
    if width == 0 or rows <= width + 1:
        return None
    mean, covariance = _moments(values)
    patterns, codes = _patterns(values)
    statistic = 0.0
    freedom = -width
    for position, pattern in enumerate(patterns):
        seen = np.flatnonzero(pattern)
        if not len(seen):
            continue
        group = values[codes == position][:, seen]
        difference = group.mean(axis = 0) - mean[seen]
        inverse = np.linalg.pinv(covariance[np.ix_(seen, seen)])
        statistic += len(group) * float(difference @ inverse @ difference)
        freedom += len(seen)
    if freedom <= 0:
        return None
    return statistic, float(stats.chi2.sf(statistic, freedom)), freedom


def missing_difference(x: pd.DataFrame) -> float:
    """Returns how much rows with a missing value differ from the others.

    For each column with missing values, the mean of each other column is
    compared between the rows that lack it and the rows that have it, as a
    standardized difference (the difference of the means over their pooled
    standard deviation).

    Args:
        x: the numeric columns.

    Returns:
        The largest absolute standardized difference, or 0 if none can be
            found.

    """
    largest = 0.0
    for column in x.columns[x.isna().any()]:
        lacking = x[column].isna()
        for other in x.columns.drop(column):
            first = x.loc[lacking, other].dropna()
            second = x.loc[~lacking, other].dropna()
            if len(first) < 2 or len(second) < 2:  # noqa: PLR2004
                continue
            pooled = np.sqrt((first.var() + second.var()) / 2)
            if pooled > 0:
                largest = max(
                    largest, abs(first.mean() - second.mean()) / pooled)
    return float(largest)


def overlap_test(treated: np.ndarray, x: np.ndarray) -> float:
    """Returns the share of rows whose propensity score is extreme.

    A logistic regression of the treatment on the standardized features
    finds each row's propensity score (its chance of treatment).

    Args:
        treated: whether each row was treated.
        x: the numeric features of the same rows (which may have no
            columns).

    Returns:
        The share of the rows whose propensity score is below 0.05 or above
            0.95.

    """
    linear_model = importlib.import_module('sklearn.linear_model')
    values = np.asarray(x, dtype = float).reshape(len(treated), -1)
    values = values[:, values.std(axis = 0) > 0]
    if values.shape[1] == 0:
        scores = np.full(len(treated), np.mean(treated))
    else:
        model = linear_model.LogisticRegression(max_iter = 1000)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(_standardized(values), treated)
        scores = model.predict_proba(_standardized(values))[:, 1]
    extreme = (scores < _EXTREME_PROPENSITY) | (
        scores > 1 - _EXTREME_PROPENSITY)
    return float(extreme.mean())


def pretrend_test(
    time: pd.Series,
    first: pd.Series,
    label: pd.Series) -> tuple[float, float, float] | None:
    """Tests whether treated units trended differently before treatment.

    The rows before each unit's treatment (and the rows of units never
    treated, in the periods before the last treatment) are kept, and the
    label is regressed on the time, whether the unit is ever treated, and
    their product. A t test asks whether the product's coefficient (the
    difference in the trends) is 0.

    Args:
        time: the period of each row.
        first: the period in which each row's unit was first treated
            (missing or infinite for units never treated), or, with two
            periods, 1 for treated units and 0 for the rest.
        label: the label of each row.

    Returns:
        The t statistic, its p-value, and the difference in the trends over
            the periods before the treatment, in standard deviations of the
            label, or `None` if the treated units have fewer than two
            periods before their treatment.

    """
    stats = importlib.import_module('scipy.stats')
    frame = pd.DataFrame({
        'time': time.to_numpy(dtype = float),
        'first': first.to_numpy(dtype = float),
        'label': label.to_numpy(dtype = float)})
    frame = frame.dropna(subset = ['time', 'label'])
    periods = np.sort(frame['time'].unique())
    binary = frame['first'].dropna().isin([0, 1]).all()
    if len(periods) == 2 and binary:  # noqa: PLR2004
        return None
    ever = np.isfinite(frame['first'].to_numpy())
    if not ever.any() or ever.all():
        return None
    last = frame.loc[ever, 'first'].max()
    before = np.where(
        ever, frame['time'] < frame['first'], frame['time'] < last)
    kept = frame[before]
    treated = np.isfinite(kept['first'].to_numpy())
    if kept.loc[treated, 'time'].nunique() < 2:  # noqa: PLR2004
        return None
    centered = kept['time'].to_numpy() - kept['time'].mean()
    design = np.column_stack([
        np.ones(len(kept)), centered, treated, centered * treated])
    y = kept['label'].to_numpy()
    rows = len(kept)
    if rows <= design.shape[1] + 1:
        return None
    coefficients = np.linalg.lstsq(design, y, rcond = None)[0]
    variance = _residual_sum(design, y) / (rows - design.shape[1])
    errors = np.sqrt(np.diag(variance * np.linalg.pinv(design.T @ design)))
    if errors[3] == 0:
        return None
    t_value = float(coefficients[3] / errors[3])
    p_value = float(2 * stats.t.sf(abs(t_value), rows - design.shape[1]))
    span = np.ptp(kept.loc[treated, 'time'].to_numpy())
    spread = y.std()
    drift = abs(coefficients[3]) * span / spread if spread > 0 else 0.0
    return t_value, p_value, float(drift)


def reset_test(
    x: np.ndarray,
    y: np.ndarray) -> tuple[float, float, float] | None:
    """Tests whether a regression is linear in its features.

    Ramsey's RESET test adds the squares and cubes of the predictions of a
    least squares fit to the features and tests whether they explain more of
    the label, with an F test.

    Args:
        x: the numeric features, without missing values.
        y: the label of the same rows.

    Returns:
        The F statistic, its p-value, and the share of the remaining
            variance of the label that the added terms explain, or `None` if
            there are too few rows or the fit is perfect.

    """
    stats = importlib.import_module('scipy.stats')
    design = _design(x)
    rows, columns = design.shape
    if rows <= columns + 3:
        return None
    coefficients = np.linalg.lstsq(design, y, rcond = None)[0]
    fitted = design @ coefficients
    remaining = float(((y - fitted) ** 2).sum())
    spread = fitted.std()
    if remaining <= 0 or spread == 0:
        return None
    scaled = (fitted - fitted.mean()) / spread
    larger = np.column_stack([design, scaled ** 2, scaled ** 3])
    coefficients = np.linalg.lstsq(larger, y, rcond = None)[0]
    left = float(((y - larger @ coefficients) ** 2).sum())
    freedom = rows - columns - 2
    f_value = ((remaining - left) / 2) / (left / freedom) if left > 0 else (
        np.inf)
    p_value = float(stats.f.sf(f_value, 2, freedom))
    return float(f_value), p_value, (remaining - left) / remaining


def schoenfeld_test(
    residuals: np.ndarray,
    times: np.ndarray) -> list[tuple[float, float, float]]:
    """Tests whether each feature's effect on the hazard changes over time.

    Grambsch and Therneau's test correlates each feature's Schoenfeld
    residuals (of the rows with an event) with the rank of their times. Its
    statistic is the number of events times the squared correlation, which
    is chi-squared with one degree of freedom.

    Args:
        residuals: the Schoenfeld residuals, one column for each feature
            (missing for censored rows).
        times: the time of each row.

    Returns:
        For each feature: the correlation, the statistic, and its p-value,
            or an empty `list` if there are fewer than three events.

    """
    stats = importlib.import_module('scipy.stats')
    values = np.asarray(residuals, dtype = float)
    if values.ndim == 1:
        values = values[:, None]
    events = ~np.isnan(values).any(axis = 1)
    values, ranks = values[events], stats.rankdata(np.asarray(times)[events])
    if len(values) < 3:  # noqa: PLR2004
        return []
    found = []
    for column in values.T:
        if column.std() == 0:
            found.append((0.0, 0.0, 1.0))
            continue
        correlation = float(np.corrcoef(column, ranks)[0, 1])
        statistic = len(column) * correlation ** 2
        found.append((
            correlation, statistic, float(stats.chi2.sf(statistic, 1))))
    return found


def serial_correlation_test(
    residuals: pd.Series,
    order: pd.Series,
    groups: pd.Series | None = None,
) -> tuple[float, float, float, float] | None:
    """Tests whether errors of rows next to each other are correlated.

    The errors are put in `order` (within each group, if there are groups),
    and Breusch and Godfrey's test regresses each error on the one before it
    (0 for the first row of each group). Its statistic is the number of rows
    times the R squared of that fit. Durbin and Watson's statistic (the sum
    of the squared differences of errors next to each other over the sum of
    the squared errors) is found as well: near 2 means independent errors,
    below 2 positively correlated ones, and above 2 negatively correlated
    ones.

    Args:
        residuals: the errors, by row.
        order: the value that puts each row in order.
        groups: the group of each row, or `None` if there are no groups.

    Returns:
        Durbin and Watson's statistic, Breusch and Godfrey's statistic, its
            p-value, and the correlation of the errors of rows next to each
            other, or `None` if there are too few rows.

    """
    stats = importlib.import_module('scipy.stats')
    frame = pd.DataFrame({
        'error': residuals.to_numpy(dtype = float),
        'order': order.to_numpy(),
        'group': 0 if groups is None else groups.astype(object).to_numpy()})
    frame = frame.dropna(subset = ['error', 'order'])
    frame = frame.sort_values(['group', 'order'], kind = 'stable')
    errors = frame['error'].to_numpy()
    if len(errors) < 4 or not errors.any():  # noqa: PLR2004
        return None
    same = (frame['group'].to_numpy()[1:] == frame['group'].to_numpy()[:-1])
    differences = np.diff(errors)[same]
    watson = float((differences ** 2).sum() / (errors ** 2).sum())
    lagged = np.concatenate([[0.0], np.where(same, errors[:-1], 0.0)])
    share = _r_squared(_design(lagged[:, None]), errors)
    statistic = len(errors) * share
    p_value = float(stats.chi2.sf(statistic, 1))
    pairs = np.flatnonzero(same)
    rho = (
        float(np.corrcoef(errors[pairs], errors[pairs + 1])[0, 1])
        if len(pairs) > 2 else 0.0)  # noqa: PLR2004
    return watson, float(statistic), p_value, 0.0 if np.isnan(rho) else rho


def stationarity_test(values: np.ndarray) -> tuple[float, float] | None:
    """Tests whether a series is stationary.

    Args:
        values: the series, in order, without missing values.

    Returns:
        The p-value of the augmented Dickey-Fuller test (whose null is a
            unit root) and of the KPSS test (whose null is stationarity),
            from statsmodels, or `None` if there are fewer than 20 values or
            they do not vary.

    """
    if len(values) < _auto()._TESTABLE or np.std(values) == 0:
        return None
    stattools = importlib.import_module('statsmodels.tsa.stattools')
    with warnings.catch_warnings():
        # KPSS warns when its p-value is beyond the table it is read from.
        warnings.simplefilter('ignore')
        unit_root = float(stattools.adfuller(values, autolag = 'AIC')[1])
        stationary = float(stattools.kpss(
            values, regression = 'c', nlags = 'auto')[1])
    return unit_root, stationary


""" Private Functions """


def _auto() -> Any:
    """Returns the `auto` module, which imports this one.

    Returns:
        `amos.core.auto`.

    """
    return importlib.import_module('amos.core.auto')


def _design(x: np.ndarray) -> np.ndarray:
    """Returns the design matrix of a least squares fit: a constant and `x`.

    Columns of `x` that do not vary are left out.

    Args:
        x: the features, one column for each.

    Returns:
        A column of 1s and the columns of `x` that vary.

    """
    values = np.asarray(x, dtype = float)
    if values.ndim == 1:
        values = values[:, None]
    varied = values[:, values.std(axis = 0) > 0] if len(values) else values
    return np.column_stack([np.ones(len(values)), varied])


def _moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the means and covariances of columns with missing values.

    They are estimated by expectation maximization, as if the columns were
    normal: each missing value is replaced by its expected value given the
    observed values of its row, and the covariances are corrected for what
    those expectations leave out, until the estimates settle.

    Args:
        values: the columns, with missing values as `nan`.

    Returns:
        The mean of each column and their covariance matrix.

    """
    rows, width = values.shape
    mean = np.nanmean(values, axis = 0)
    covariance = np.diag(np.nanvar(values, axis = 0) + 1e-9)
    patterns, codes = _patterns(values)
    for _ in range(_EM_STEPS):
        total = np.zeros(width)
        products = np.zeros((width, width))
        for position, pattern in enumerate(patterns):
            group = values[codes == position]
            seen, unseen = np.flatnonzero(pattern), np.flatnonzero(~pattern)
            filled = group.copy()
            if len(unseen):
                if len(seen):
                    weights = covariance[np.ix_(unseen, seen)] @ np.linalg.pinv(
                        covariance[np.ix_(seen, seen)])
                    filled[:, unseen] = mean[unseen] + (
                        group[:, seen] - mean[seen]) @ weights.T
                    left = covariance[np.ix_(unseen, unseen)] - (
                        weights @ covariance[np.ix_(seen, unseen)])
                else:
                    filled[:, unseen] = mean[unseen]
                    left = covariance[np.ix_(unseen, unseen)]
                products[np.ix_(unseen, unseen)] += len(group) * left
            total += filled.sum(axis = 0)
            products += filled.T @ filled
        updated = total / rows
        changed = products / rows - np.outer(updated, updated)
        settled = (
            np.abs(updated - mean).max() < _EM_TOLERANCE
            and np.abs(changed - covariance).max() < _EM_TOLERANCE)
        mean, covariance = updated, changed
        if settled:
            break
    return mean, covariance


def _numeric_names(item: base.Dataset) -> list[str]:
    """Returns the names of the numeric and boolean features.

    Args:
        item: the dataset.

    Returns:
        The names, in order.

    """
    return item.numerics + item.booleans


def _numeric_features(item: base.Dataset, rows: pd.Index) -> pd.DataFrame:
    """Returns the numeric and boolean features of some rows, as floats.

    Args:
        item: the dataset.
        rows: the index labels of the rows.

    Returns:
        The features that are numbers or booleans.

    """
    return item.data.loc[rows, _numeric_names(item)].astype(float)


def _patterns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the patterns of missing values and the pattern of each row.

    Args:
        values: the columns, with missing values as `nan`.

    Returns:
        Each distinct pattern (whether each column is observed) and the
            position of each row's pattern among them.

    """
    patterns, codes = np.unique(
        ~np.isnan(values), axis = 0, return_inverse = True)
    return patterns, np.asarray(codes).ravel()


def _r_squared(design: np.ndarray, y: np.ndarray) -> float:
    """Returns the R squared of a least squares fit of `y` to `design`.

    Args:
        design: the design matrix, with a column of 1s.
        y: the values to fit.

    Returns:
        The share of the variance of `y` that the fit explains.

    """
    total = float(((y - y.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    coefficients = np.linalg.lstsq(design, y, rcond = None)[0]
    left = float(((y - design @ coefficients) ** 2).sum())
    return max(0.0, 1 - left / total)


def _real_rows(item: base.Dataset) -> pd.Index:
    """Returns every row that a sampler did not make up.

    Inferers use every real row (not only the training rows), so their
    assumptions are tested on them.

    Args:
        item: the dataset.

    Returns:
        The index labels of the real rows.

    """
    return item.data.index[~item.data.index.isin(item.synthetic)]


def _residual_sum(design: np.ndarray, y: np.ndarray) -> float:
    """Returns the sum of the squared errors of a least squares fit.

    Args:
        design: the design matrix, with a column of 1s.
        y: the values to fit.

    Returns:
        The sum of the squared differences of `y` and the fit.

    """
    coefficients = np.linalg.lstsq(design, y, rcond = None)[0]
    return float(((y - design @ coefficients) ** 2).sum())


def _residuals(item: base.Dataset) -> pd.Series | None:
    """Returns the errors of a fitted regression on the rows it predicted.

    Args:
        item: the dataset, with a fitted model and its predictions (of the
            test rows, or of every row if the data is not split).

    Returns:
        The label minus the prediction of each row (without missing
            values), or `None` if the label is not regressed or there are no
            predictions.

    """
    if item.task != 'regress' or item.predictions is None:
        return None
    rows = item.predictions.index
    actual = pd.to_numeric(item.y.loc[rows], errors = 'coerce')
    errors = actual - pd.to_numeric(item.predictions, errors = 'coerce')
    return errors.dropna().astype(float)


def _standardized(values: np.ndarray) -> np.ndarray:
    """Returns each column of `values` with a mean of 0 and a deviation of 1.

    Args:
        values: the columns.

    Returns:
        The standardized columns (a column that does not vary is all 0s).

    """
    spread = values.std(axis = 0)
    spread[spread == 0] = 1.0
    return (values - values.mean(axis = 0)) / spread


def _test_linearity(
    item: base.Dataset,
    found: Any) -> tuple[float, float, float, str] | str:
    """Tests whether the label (or its log-odds) is linear in the features.

    The real training rows that `auto.profile` describes are used, with
    their numeric and boolean features (rows with missing values are left
    out).

    Args:
        item: the dataset.
        found: its profile.

    Returns:
        The statistic, p-value, share explained by nonlinear terms, and the
            name of the test, or the reason it could not be run.

    """
    if found.task not in ('classify', 'regress'):
        return 'needs a label to test'
    if found.task == 'classify' and found.classes != 2:  # noqa: PLR2004
        return 'is only tested for a regressed label or two classes'
    rows = _auto().profiled_rows(item)
    x = _numeric_features(item, rows)
    frame = x.join(item.y.loc[rows].rename('__label__')).dropna()
    label = frame.pop('__label__')
    if frame.empty or not frame.shape[1]:
        return 'has no numeric features to test'
    if found.task == 'regress':
        result = reset_test(frame.to_numpy(), label.to_numpy(dtype = float))
        test = 'RESET'
    else:
        result = box_tidwell_test(frame, label)
        test = 'Box-Tidwell'
    if result is None:
        return 'has too few rows (or no features that can be tested)'
    return (*result, test)
