"""Techniques that check the assumptions and needs of other techniques.

These are techniques of the "analyst" stage. Each pretest checks some of
the assumptions and needs (see the `conditions` module) of the techniques
that come after it, on the real training rows, and adds a table of what it
found to the dataset's `tables` under its name. `check_assumptions` runs
every pretest that the techniques need. Put a pretest after the steps that
change the features (such as "scale" and "encode") and before the models,
so that it checks the data that they will see:

```ini
[general]
label = target
assumption_violations = warn

[analyst]
steps = split, scale, check, model
split_techniques = stratified
scale_techniques = standard
check_techniques = check_assumptions
model_techniques = linear, knn
```

Without `techniques`, a pretest in a `Project` checks the techniques of the
workflow. A pretest only reports: amos runs the pretests of each technique
again just before it is applied (whether or not the workflow has a
pretest), and the "assumption_violations" setting of the "general" section
decides what it does when the data does not meet one: "raise" an error,
"warn" (the default), "ignore" it, or "skip" the branch of an experiment.

Contents:
    Pretest: base class for techniques that check the assumptions and needs
        of techniques before they are applied.
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

import types

__all__: list[str] = [
    'CheckAssumptions',
    'CheckContinuity',
    'CheckDispersion',
    'CheckFeatures',
    'CheckInstrument',
    'CheckLabel',
    'CheckLinearity',
    'CheckMissingness',
    'CheckMulticollinearity',
    'CheckNormality',
    'CheckOverlap',
    'CheckParallelTrends',
    'CheckProportionalOdds',
    'CheckSampleSize',
    'CheckScales',
    'CheckStationarity',
    'Pretest']

from ._base import Pretest
from ._checks import (
    CheckAssumptions,
    CheckContinuity,
    CheckDispersion,
    CheckFeatures,
    CheckInstrument,
    CheckLabel,
    CheckLinearity,
    CheckMissingness,
    CheckMulticollinearity,
    CheckNormality,
    CheckOverlap,
    CheckParallelTrends,
    CheckProportionalOdds,
    CheckSampleSize,
    CheckScales,
    CheckStationarity,
)

# The public names say that they are in `amos.pretests` (rather than in the
# private module that defines them, or in `amos.analyst.pretests`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.pretests.Name`. `amos` makes `amos.pretests` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.pretests'
