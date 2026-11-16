"""Describers of survival over time.

Contents:
    KaplanMeier: the share of rows without an event over time (survival).
    survival_curves: fits a Kaplan-Meier estimator for each group.
    survival_steps: the steps of a Kaplan-Meier curve, with its confidence
        interval.

"""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ...core import base, utilities
from ._base import Describer


@dataclasses.dataclass
class KaplanMeier(Describer):
    """Estimates the share of rows still without an event over time.

    The label is the time until the event (such as a patient's relapse) or until
    the row stopped being observed. Set "event" to the column that is 1 if the
    event happened and 0 if the row was censored (by default, every event was
    observed), and "group" to a column to make a curve for each group. It wraps
    statsmodels' `SurvfuncRight`.

    Wraps:
        [`statsmodels.duration.survfunc.SurvfuncRight`](https://www.statsmodels.org/stable/generated/statsmodels.duration.survfunc.SurvfuncRight.html)
        from statsmodels.

    """

    visual: ClassVar[str] = 'survival_curves'

    def describe(
        self,
        item: base.Dataset,
        event: str | None = None,
        group: str | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Returns the estimated survival at each time.

        Args:
            item: the dataset to describe. Its label is the time.
            event: name of the column that is 1 if the event happened.
                Defaults to `None`, which means every event was observed.
            group: name of a column of groups. Defaults to `None`, which
                makes one curve for every row.
            **kwargs: not used.

        Returns:
            One row for each time in each group (0, each time with an event,
                and the last time), with "group", "time", "survival" (the
                share without the event), and "at_risk".

        """
        tables = []
        for name, curve in survival_curves(item, event, group).items():
            steps = survival_steps(curve)
            steps.insert(0, 'group', name)
            tables.append(steps[['group', 'time', 'survival', 'at_risk']])
        return pd.concat(tables, ignore_index = True)

    """ Private Methods """

    def _visual_parameters(
        self,
        item: base.Dataset,  # noqa: ARG002
        parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Returns the parameters of `survival_curves`.

        Args:
            item: the dataset, with the technique's table.
            parameters: the parameters that the technique was applied with.

        Returns:
            The "event" and "group" that the curves were estimated with.

        """
        return {
            'event': parameters.get('event'),
            'group': parameters.get('group')}


""" Public Functions """


def survival_curves(
    item: base.Dataset,
    event: str | None = None,
    group: str | None = None) -> dict[str, Any]:
    """Returns a fitted Kaplan-Meier estimator for each group.

    `kaplan_meier` and `survival_curves` (in the plots) both use this.

    Wraps:
        [`statsmodels.duration.survfunc.SurvfuncRight`](https://www.statsmodels.org/stable/generated/statsmodels.duration.survfunc.SurvfuncRight.html)
        from statsmodels.

    Args:
        item: the dataset. Its label is the time until the event.
        event: name of the column that is 1 if the event happened. Defaults
            to `None`, which means every event was observed.
        group: name of a column of groups. Defaults to `None`, which makes
            one estimate, called "all", for every row.

    Returns:
        The fitted statsmodels `SurvfuncRight` of each group, by its name.

    """
    survfunc = utilities.import_tool('statsmodels.api.SurvfuncRight')
    times = item.y.astype(float)
    if event is None:
        observed = pd.Series(1, index = item.data.index)
    else:
        observed = item.data[event].astype(int)
    if group is None:
        groups = {'all': item.data.index}
    else:
        groups = {
            str(name): rows
            for name, rows in item.data.groupby(group).groups.items()}
    return {
        name: survfunc(
            times.loc[rows].to_numpy(), observed.loc[rows].to_numpy(),
            title = name)
        for name, rows in groups.items()}


def survival_steps(curve: Any) -> pd.DataFrame:
    """Returns the steps of a Kaplan-Meier curve from `survival_curves`.

    The curve starts at a time of 0 and ends at the last time that a row was
    observed. The confidence interval is the usual 95% interval of the log of
    minus the log of the survival, with Greenwood's variance.

    Args:
        curve: a fitted statsmodels `SurvfuncRight`.

    Returns:
        One row for each time (0, each time with an event, and the last time),
            with "time", "survival", "ci_lower", "ci_upper", and "at_risk"
            (the number of rows whose time is at least that time).

    """
    observed = np.sort(np.asarray(curve.time, dtype = float))
    events = np.asarray(curve.surv_times, dtype = float)
    survival = np.asarray(curve.surv_prob, dtype = float)
    at_risk = np.asarray(curve.n_risk, dtype = float)
    happened = np.asarray(curve.n_events, dtype = float)
    z = statistics.NormalDist().inv_cdf(0.975)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        greenwood = np.cumsum(happened / (at_risk * (at_risk - happened)))
        logged = np.log(survival)
        spread = z * np.sqrt(greenwood) / -logged
        lower = np.exp(-np.exp(np.log(-logged) + spread))
        upper = np.exp(-np.exp(np.log(-logged) - spread))
    # Once every row at risk has had the event, the survival is surely 0.
    lower[survival == 0] = 0.0
    upper[survival == 0] = 0.0
    times = np.unique(np.concatenate([[0.0], events, observed[-1:]]))
    # Before the first event, nobody has had it. Afterward, each time takes
    # the step of the last event at or before it.
    steps = np.searchsorted(events, times, side = 'right')
    return pd.DataFrame({
        'time': times,
        'survival': np.concatenate([[1.0], survival])[steps],
        'ci_lower': np.concatenate([[1.0], lower])[steps],
        'ci_upper': np.concatenate([[1.0], upper])[steps],
        'at_risk': len(observed) - np.searchsorted(observed, times)})
