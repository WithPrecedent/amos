"""Techniques that choose which features a model uses.

These are techniques of the "analyst" stage, usually in a step of their own
(such as "select") after the features are encoded and scaled. Each one keeps
some of the features and removes the rest. Most are transformers (see the
`transformers` module), so they learn which features to keep from the
training rows only, choose from the numeric and boolean features by default,
and can be limited to some `columns`.

Some selectors choose features with a model, which is their "estimator"
parameter: the name of a model in the library (such as "random_forest"),
which is built for the task with its default parameters, or an estimator,
its class, or its import path. By default, it is a logistic regression (to
classify) or a linear regression, so scale the features first. Those that
compare sets of features score each set by cross-validation on the training
rows, with "scoring" (the name of a scikit-learn scorer, by default accuracy
or R²) and "cv" folds (5 by default) that keep the rows of each of the
dataset's first groups together.

`feature_group` instead keeps the features that the settings list, so an
experiment can compare groups of features that you chose. Name each group
in a "feature_groups" section of the settings, and then name the groups as
the techniques of a step:

```ini
[analyst]
steps = split, select, model
select_techniques = demographics, medical
model_techniques = logit

[feature_groups]
demographics = age, sex, race
medical = visits, diagnoses, insured
```

Features listed as "always_use" in that section are not a group of their
own: every group keeps them too, so they are in every group's path.

Contents:
    Selector: base class for techniques that keep some of the features.
    FeatureGroup: keeps the features that the settings list.
    register_feature_groups: adds the feature groups of the settings to the
        library.
    ExhaustiveSelection: keeps the best of every set of 1 to 3 features (by
        cross-validation).
    RecursiveElimination: removes the least important feature of a model, again
        and again.
    RecursiveEliminationCV: removes features as `recursive_elimination` does and
        keeps the number that scores best.
    SelectFromModel: keeps the features that a model finds most important.
    SequentialSelection: adds (or removes) features with mlxtend while the score
        improves.
    SequentialSelectionSK: adds (or removes) features with scikit-learn while
        the score improves.
    GenericUnivariate: keeps the features most related to the label, by a mode
        that the parameters choose.
    KBest: keeps the k features most related to the label.
    SelectFdr: keeps the features related to the label, controlling the false
        discovery rate.
    SelectFpr: keeps the features related to the label, controlling the false
        positive rate.
    SelectFwe: keeps the features related to the label, controlling the family-
        wise error rate.
    SelectPercentile: keeps a share of the features most related to the label.
    VarianceThreshold: removes features whose variance is at or below a
        threshold.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'ExhaustiveSelection',
    'FeatureGroup',
    'GenericUnivariate',
    'KBest',
    'RecursiveElimination',
    'RecursiveEliminationCV',
    'SelectFdr',
    'SelectFpr',
    'SelectFromModel',
    'SelectFwe',
    'SelectPercentile',
    'Selector',
    'SequentialSelection',
    'SequentialSelectionSK',
    'VarianceThreshold',
    'register_feature_groups']

from ._base import Selector
from ._groups import FeatureGroup, register_feature_groups
from ._model_based import (
    ExhaustiveSelection,
    RecursiveElimination,
    RecursiveEliminationCV,
    SelectFromModel,
    SequentialSelection,
    SequentialSelectionSK,
)
from ._univariate import (
    GenericUnivariate,
    KBest,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    SelectPercentile,
    VarianceThreshold,
)

# The public names say that they are in `amos.selectors` (rather than in the
# private module that defines them, or in `amos.analyst.selectors`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.selectors.Name`. `amos` makes `amos.selectors` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.selectors'
