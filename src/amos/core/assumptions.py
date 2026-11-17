"""What techniques assume about the data, and the tests of each assumption.

Each assumption is a subclass of `conditions.Assumption`, named in snake
case, which a technique names in its `assumptions`. Those with a pretest
are checked on the real training rows before a technique is applied, and
those with an audit are checked from the errors of a fitted model (see the
`conditions` module). An assumption without either cannot be tested from
the data, but it is listed with the techniques that make it.

The tests need only numpy, scipy, and scikit-learn. With many rows, a test
rejects even trivial departures from an assumption, so an assumption fails
only if its test rejects it (at the 5% level) and the departure is large
enough to matter (see the thresholds below), as `auto.profile` judges the
normality of errors.

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
    MissingCompletelyAtRandom: whether a value is missing has nothing to do
        with any value.
    MissingAtRandom: whether a value is missing depends only on values that
        are observed.
    Unconfoundedness: every common cause of the treatment and the outcome
        is among the features.
    Overlap: every row could have been treated or not.
    ValidInstrument: the instrument changes the treatment and affects the
        outcome only through it.
    ParallelTrends: without the treatment, the treated and other rows would
        have changed alike.
    Continuity: the outcome changes smoothly at the cutoff except for the
        treatment.
    Stationarity: the distribution of each series does not change over
        time.
    CausalSufficiency: every common cause of the variables is measured.
    ProportionalHazards: each feature multiplies the hazard by the same
        amount at every time.
    ProportionalOdds: each feature has the same effect on every cutoff
        between ordered classes.
    reset_test: Ramsey's test that a regression is linear.
    box_tidwell_test: Box and Tidwell's test that log-odds are linear.
    breusch_pagan_test: Breusch and Pagan's test of constant variance.
    serial_correlation_test: Durbin and Watson's statistic and Breusch and
        Godfrey's test of serially correlated errors.
    clustering_test: the intraclass correlation of errors within groups.
    _MATERIAL: smallest share of variance (or deviance) that a departure
        from linearity or constant variance must explain to matter.
    _AUTOCORRELATION: smallest correlation of the errors of rows next to
        each other that matters.
    _CLUSTERED: smallest intraclass correlation of the errors that matters.
    _NORMAL_SHARE: largest share of features that may be skewed or
        heavy-tailed for features to count as normal.
    _LINEARITY: the name under which the test of linearity is cached.

"""

from __future__ import annotations

import dataclasses
import importlib
import warnings
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from . import base, conditions

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


""" Assumptions Without Tests """


@dataclasses.dataclass(frozen = True)
class MissingCompletelyAtRandom(conditions.Assumption):
    """Whether a value is missing has nothing to do with any value.

    Filling a value with the mean, median, or most common value of its
    column (or dropping its row) is unbiased only if this is so.

    """

    description: ClassVar[str] = (
        'whether a value is missing has nothing to do with any value')


@dataclasses.dataclass(frozen = True)
class MissingAtRandom(conditions.Assumption):
    """Whether a value is missing depends only on values that are observed.

    Filling a value from the other features (as an iterative or nearest
    neighbors imputer does) is unbiased if this is so.

    """

    description: ClassVar[str] = (
        'whether a value is missing depends only on values that are observed')


@dataclasses.dataclass(frozen = True)
class Unconfoundedness(conditions.Assumption):
    """Every common cause of the treatment and the outcome is a feature.

    Also called ignorability, or no unmeasured confounding. It cannot be
    tested from the data: it is argued from how the data came to be.

    """

    description: ClassVar[str] = (
        'every common cause of the treatment and the outcome is a feature')
    affects: ClassVar[str] = 'inference'


@dataclasses.dataclass(frozen = True)
class Overlap(conditions.Assumption):
    """Every row could have been treated or not (positivity).

    Rows whose features make treatment certain (or impossible) have nothing
    to be compared with, so their effects are extrapolated.

    """

    description: ClassVar[str] = 'every row could have been treated or not'
    affects: ClassVar[str] = 'inference'


@dataclasses.dataclass(frozen = True)
class ValidInstrument(conditions.Assumption):
    """The instrument changes the treatment and affects the outcome only so.

    The instrument must be relevant (it changes the treatment), and it must
    be excluded (it affects the outcome only through the treatment, and
    shares no cause with the outcome).

    """

    description: ClassVar[str] = (
        'the instrument changes the treatment and affects the outcome only '
        'through it')
    affects: ClassVar[str] = 'inference'


@dataclasses.dataclass(frozen = True)
class ParallelTrends(conditions.Assumption):
    """Without the treatment, the treated and other rows would change alike.

    A difference in differences attributes any difference in how the two
    groups changed to the treatment.

    """

    description: ClassVar[str] = (
        'without the treatment, the treated and other rows would have '
        'changed alike')
    affects: ClassVar[str] = 'inference'


@dataclasses.dataclass(frozen = True)
class Continuity(conditions.Assumption):
    """The outcome changes smoothly at the cutoff except for the treatment.

    A regression discontinuity attributes a jump at the cutoff to the
    treatment, so nothing else (such as rows sorting themselves to one side)
    may jump there.

    """

    description: ClassVar[str] = (
        'the outcome would change smoothly at the cutoff without the '
        'treatment')
    affects: ClassVar[str] = 'inference'


@dataclasses.dataclass(frozen = True)
class Stationarity(conditions.Assumption):
    """The distribution of each series does not change over time."""

    description: ClassVar[str] = (
        'the distribution of each series does not change over time')


@dataclasses.dataclass(frozen = True)
class CausalSufficiency(conditions.Assumption):
    """Every common cause of the variables is measured.

    A method of causal discovery that assumes it can mistake a hidden common
    cause for a link between two variables.

    """

    description: ClassVar[str] = (
        'every common cause of the variables is measured')
    affects: ClassVar[str] = 'inference'


@dataclasses.dataclass(frozen = True)
class ProportionalHazards(conditions.Assumption):
    """Each feature multiplies the hazard by the same amount at every time."""

    description: ClassVar[str] = (
        'each feature multiplies the hazard by the same amount at every time')


@dataclasses.dataclass(frozen = True)
class ProportionalOdds(conditions.Assumption):
    """Each feature has the same effect on every cutoff between classes.

    An ordinal regression has one coefficient for each feature, which is the
    same for the odds of being above each class.

    """

    description: ClassVar[str] = (
        'each feature has the same effect on the odds of being above every '
        'class')


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


def _numeric_features(item: base.Dataset, rows: pd.Index) -> pd.DataFrame:
    """Returns the numeric and boolean features of some rows, as floats.

    Args:
        item: the dataset.
        rows: the index labels of the rows.

    Returns:
        The features that are numbers or booleans.

    """
    columns = item.numerics + item.booleans
    return item.data.loc[rows, columns].astype(float)


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
