"""Audits of the assumptions of fitted models.

Contents:
    AuditAssumptions: checks every assumption of the model that has an
        audit.
    AuditErrors: checks whether the model's errors have constant variance.
    AuditIndependence: checks whether the model's errors are independent.
    AuditProportionalHazards: checks whether each feature's effect on the
        hazard of a Cox model is constant over time.
    AuditResiduals: checks whether the model's errors are normal.

"""

from __future__ import annotations

import dataclasses

from ._base import Audit


@dataclasses.dataclass
class AuditAssumptions(Audit):
    """Checks every assumption of the model that has an audit.

    It runs every other audit, but only for the assumptions that the
    dataset's model makes, so a model that makes none (such as a random
    forest) has an empty table. Name it in the "critic" stage:

    ```ini
    [critic]
    techniques = audit_assumptions, scorecard
    ```

    """

    def _check(self) -> str | None:
        """Returns `None`, which checks every assumption that has an audit.

        Returns:
            `None`.

        """
        return None


@dataclasses.dataclass
class AuditErrors(Audit):
    """Checks whether the model's errors have constant variance.

    It checks "constant_variance" with Breusch and Pagan's test, which
    regresses the squared errors on the numeric features (see
    `assumptions.ConstantVariance`).

    """


@dataclasses.dataclass
class AuditIndependence(Audit):
    """Checks whether the model's errors are independent.

    It checks "independent_errors": with Breusch and Godfrey's test (and
    Durbin and Watson's statistic) of the errors in the dataset's `order`,
    and with the intraclass correlation of the errors within its first
    group column (see `assumptions.IndependentErrors`). A dataset with
    neither an order nor groups cannot be tested.

    """


@dataclasses.dataclass
class AuditProportionalHazards(Audit):
    """Checks whether each feature's effect on a Cox model's hazard is constant.

    It checks "proportional_hazards" of a fitted `cox` model with Grambsch
    and Therneau's test of the correlation of each feature's Schoenfeld
    residuals with the rank of the time (see
    `assumptions.ProportionalHazards`).

    """


@dataclasses.dataclass
class AuditResiduals(Audit):
    """Checks whether the model's errors are normal.

    It checks "normal_errors" with D'Agostino and Pearson's test and the
    skewness and excess kurtosis of the errors, and reports the share that
    are outliers (see `assumptions.NormalErrors`).

    """
