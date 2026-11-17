"""Techniques that check the assumptions of a fitted model from its errors.

These are techniques of the "critic" stage. Some assumptions of a model can
only be checked after it is fitted: whether its errors are normal
(`audit_residuals`), have constant variance (`audit_errors`), and are
independent (`audit_independence`). Each audit adds a table of what it
found to the dataset's `tables` under its name, and `audit_assumptions`
runs the audits of every assumption that the dataset's model makes:

```ini
[general]
label = target
order = date

[critic]
techniques = audit_assumptions, scorecard
```

The "assumption_violations" setting of the "general" section decides what
amos does when the data does not meet an assumption that the model makes:
"raise" an error, "warn" (the default), or "ignore" it. With "skip", an
audit in a branch of an experiment leaves the branch out, but a critic after
an experiment (which sees only the winning branch) warns.

Contents:
    Audit: base class for techniques that check the assumptions of a fitted
        model from its errors.
    AuditAssumptions: checks every assumption of the model that has an
        audit.
    AuditErrors: checks whether the model's errors have constant variance.
    AuditIndependence: checks whether the model's errors are independent.
    AuditProportionalHazards: checks whether each feature's effect on the
        hazard of a Cox model is constant over time.
    AuditResiduals: checks whether the model's errors are normal.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'Audit',
    'AuditAssumptions',
    'AuditErrors',
    'AuditIndependence',
    'AuditProportionalHazards',
    'AuditResiduals']

from ._audits import (
    AuditAssumptions,
    AuditErrors,
    AuditIndependence,
    AuditProportionalHazards,
    AuditResiduals,
)
from ._base import Audit

# The public names say that they are in `amos.audits` (rather than in the
# private module that defines them, or in `amos.critic.audits`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.audits.Name`. `amos` makes `amos.audits` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.audits'
