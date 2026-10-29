"""Techniques that choose other techniques to suit the data.

Name "auto" in place of a technique, and `amos` chooses the techniques that
suit the data when the workflow reaches it. "auto" works in a step that
fills missing values, encodes, samples, models, evaluates, or plots (the kind
of step is found from its name, such as "impute", "encode", "sample",
"model", or "plot", or from the name of a worker such as "critic" or
"artist"), or name the chooser yourself: `auto_imputer`, `auto_encoder`,
`auto_sampler`, `auto_model`, `auto_evaluator`, or `auto_plot`.

```ini
[analyst]
steps = split, impute, encode, sample, model
split_techniques = stratified
impute_techniques = auto
encode_techniques = auto
sample_techniques = auto
model_techniques = auto

[critic]
techniques = auto

[artist]
techniques = auto
```

Each chooser first profiles the real training rows (see `profile`): the
balance of the classes, the distribution of a regression's errors, whether
the label is counts, missing values, the kinds and scales of the features,
and how collinear they are. It then rules out the techniques whose needs or
assumptions the data does not meet, and records why in a table in the
dataset's `tables` (named for the chooser, such as "auto_model"), with a row
for each technique it considered. The profile and the choice are also in the
dataset's `history`, so a paper can report how each technique was chosen.

Contents:
    Profile: what the training rows are like.
    profile: returns the profile of a dataset.
    resolve: tells each "auto" technique in a workflow which kind of
        technique to choose, from the name of its step or worker.
    Chooser: base class for techniques that choose other techniques.
    Auto: chooses the techniques of the step that it is in.
    AutoEncoder: encodes each categorical feature in the way that suits how
        many levels it has, and dates by their parts.
    AutoEvaluator: applies the evaluators that suit the task and the model.
    AutoImputer: fills missing values in the ways that suit the features, or
        not at all.
    AutoModel: rules out models whose assumptions the data does not meet,
        cross-validates the rest, and fits the best.
    AutoPlot: draws the figures that suit the data and the model.
    AutoSampler: chooses a sampler that suits the balance of the classes and
        the features, or none.

"""

from __future__ import annotations

import abc
import copy
import dataclasses
import functools
import importlib
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

import camina
import chrisjen
import numpy as np
import pandas as pd

from . import base, models, plots, transformers

# Names of the steps (and workers) whose "auto" technique each chooser
# replaces, by the genre of the techniques it chooses. A step named
# "{technique}_{step}" in a workflow is matched by its step.
_STEPS: dict[str, frozenset[str]] = {
    'imputer': frozenset({
        'fill', 'impute', 'imputer', 'imputers', 'imputation'}),
    'encoder': frozenset({'encode', 'encoder', 'encoders', 'encoding'}),
    'sampler': frozenset({
        'balance', 'resample', 'sample', 'sampler', 'samplers', 'sampling'}),
    'model': frozenset({
        'estimate', 'estimator', 'fit', 'learn', 'model', 'modeler',
        'modeling', 'models', 'predict'}),
    'evaluator': frozenset({
        'assess', 'critic', 'critique', 'evaluate', 'evaluation',
        'evaluator', 'evaluators'}),
    'plot': frozenset({
        'artist', 'chart', 'charts', 'draw', 'figure', 'figures', 'graph',
        'plot', 'plots', 'plotting', 'visualize'})}
# Share of the values of a numeric feature that must be missing (in the
# training rows) for `auto_imputer` to model each feature from the others
# rather than fill it with its median.
_MANY_MISSING: float = 0.05
# Most levels of a categorical feature that `auto_encoder` makes dummy
# variables for. A feature with more is encoded in fewer columns.
_LEVELS: int = 15
# Fewest words in the typical (median) value of a categorical feature with
# many levels for `auto_encoder` to treat it as free text.
_WORDS: int = 3
# Encoders that learn from the label, which must have two classes or be
# regressed (a label of more classes would be encoded as the position of
# each class, which means nothing).
_SUPERVISED: frozenset[str] = frozenset({
    'cat_boost', 'james_stein', 'leave_one_out', 'm_estimate', 'target',
    'weight_of_evidence'})
# The imputers that `auto_imputer` considers, in the order it prefers them.
_IMPUTERS: tuple[str, ...] = (
    'iterative_impute', 'median_impute', 'mode_impute')
# The encoders that `auto_encoder` considers for each kind of feature, in the
# order it prefers them: dates, categories with few levels, categories with
# many, and free text.
_ENCODERS: dict[str, tuple[str, ...]] = {
    'dates': ('date_parts',),
    'few': ('one_hot',),
    'many': ('target', 'count', 'one_hot'),
    'text': ('tfidf', 'target', 'count', 'one_hot')}
# The rarest class of a balanced label has at least this share of the rows of
# its most common class, so it is not resampled.
_BALANCED: float = 0.5
# Fewest rows of the rarest class for SMOTE and its variants, which make rows
# between a row and its 5 nearest neighbors in its class.
_NEIGHBORS: int = 6
# Most rows of the rarest class for which a sampler adds rows. With more,
# rows of the common classes are removed instead, which is faster and still
# leaves plenty to learn from.
_PLENTY: int = 10_000
# Share of the rows of the rare classes whose 5 nearest neighbors are mostly
# of other classes, above which the classes overlap.
_OVERLAP: float = 0.5
# Share of the features that are binary (such as dummy variables), above
# which SMOTE (which makes values between those of two rows) is not used.
_BINARY: float = 0.5
# Absolute skewness and excess kurtosis beyond which a distribution is
# materially different from a normal one.
_SKEW: float = 1.0
_KURTOSIS: float = 3.0
# Significance level of the test of normality.
_ALPHA: float = 0.05
# Fewest values for a test of normality (D'Agostino and Pearson's).
_TESTABLE: int = 20
# Variance inflation factor above which a feature is collinear with others.
_VIF: float = 10.0
# Ratio of the largest to the smallest standard deviation of the numeric
# features above which they are not on similar scales.
_SCALES: float = 10.0
# Pearson dispersion (the variance of counts relative to their mean, given
# the features) above which counts are overdispersed.
_DISPERSION: float = 1.5
# How much larger than a Poisson model expects the share of zeros must be
# for counts to have excess zeros.
_ZEROS: float = 0.1
# Robust z-score (from the median and the median absolute deviation) beyond
# which an error is an outlier, and the share of outliers above which a
# regression's errors have them.
_OUTLIER_SCORE: float = 3.5
_OUTLIERS: float = 0.01
# Most rows used to profile a dataset (a random sample of them, with the
# dataset's seed, is used if there are more).
_PROFILE_ROWS: int = 10_000
# Fewest predicted rows for a calibration curve, whose bins need enough rows
# to estimate a share.
_CALIBRATION_ROWS: int = 100
# Most numeric features for a pair plot, which has a panel for each pair.
_PAIRS: int = 5
# Models whose standard errors assume normal errors, so a Q-Q plot checks
# them.
_NORMAL_ERRORS: frozenset[str] = frozenset({
    'elastic_net', 'fixest', 'glm', 'lasso', 'linear', 'mixedlm', 'ols',
    'ridge', 'wls'})
# Models made of decision trees.
_TREES: frozenset[str] = frozenset({
    'adaboost', 'catboost', 'decision_tree', 'extra_trees',
    'gradient_boosting', 'lightgbm', 'random_forest', 'xgboost'})
# Models whose predictions are not a linear function of the features.
_NONLINEAR: frozenset[str] = _TREES | {
    'explainable_boosting', 'knn', 'neural_network', 'svm', 'tabpfn'}


@dataclasses.dataclass(frozen = True)
class _Traits:
    """What a model needs from the data, and what it assumes about it.

    Args:
        tasks: the tasks that the model can do.
        package: the package that the model needs.
        complexity: how complex the model is, from 1 (the simplest, such as a
            linear model) up. Of models that score about as well, the
            simplest is chosen.
        missing: whether it can use missing values in the features.
        categories: whether it can use categorical features directly.
        scaled: whether it needs features on similar scales.
        normal_errors: whether it assumes that a regression's errors are
            normal.
        normal_features: whether it assumes that the features are normal.
        independent: whether it assumes that the features are not collinear.
        wide: whether it can use more features than there are rows.
        counts: the kind of counts it models: "equidispersed" (a variance
            equal to the mean), "overdispersed", or "zero_inflated". `None`
            for a model that is not only for counts.
        ordinal: whether it is only for ordered classes.
        outliers: whether it is only for errors with outliers (for a model
            that is less precise without them).
        min_rows: fewest training rows it learns well from.
        max_rows: most training rows it can be fitted to in reasonable time.
        max_features: most features it works well with.
        min_classes: fewest classes of a classified label it can model.
        max_classes: most classes of a classified label it can model.

    """

    tasks: tuple[str, ...]
    package: str
    complexity: int
    missing: bool = False
    categories: bool = False
    scaled: bool = False
    normal_errors: bool = False
    normal_features: bool = False
    independent: bool = False
    wide: bool = True
    counts: str | None = None
    ordinal: bool = False
    outliers: bool = False
    min_rows: int = 0
    max_rows: int | None = None
    max_features: int | None = None
    min_classes: int = 0
    max_classes: int | None = None


_BOTH: tuple[str, ...] = ('classify', 'regress')
# The models that `auto_model` considers, in the order it prefers them.
_MODELS: dict[str, _Traits] = {
    'poisson': _Traits(
        ('regress',), 'statsmodels', 1, wide = False,
        counts = 'equidispersed'),
    'negative_binomial': _Traits(
        ('regress',), 'statsmodels', 1, wide = False,
        counts = 'overdispersed'),
    'zero_inflated_poisson': _Traits(
        ('regress',), 'statsmodels', 2, wide = False,
        counts = 'zero_inflated'),
    'ordinal_regression': _Traits(
        ('classify',), 'statsmodels', 1, wide = False, ordinal = True),
    'robust_regression': _Traits(
        ('regress',), 'statsmodels', 1, wide = False, outliers = True),
    'linear': _Traits(
        ('regress',), 'sklearn', 1, normal_errors = True, independent = True,
        wide = False),
    'logit': _Traits(('classify',), 'sklearn', 1, scaled = True),
    'ridge': _Traits(('regress',), 'sklearn', 1, scaled = True),
    'random_forest': _Traits(_BOTH, 'sklearn', 3, missing = True),
    'gradient_boosting': _Traits(
        _BOTH, 'sklearn', 3, missing = True, categories = True),
    'lightgbm': _Traits(
        _BOTH, 'lightgbm', 3, missing = True, categories = True),
    'xgboost': _Traits(_BOTH, 'xgboost', 3, missing = True),
    'catboost': _Traits(
        _BOTH, 'catboost', 3, missing = True, categories = True),
    'lasso': _Traits(('regress',), 'sklearn', 1, scaled = True),
    'naive_bayes': _Traits(
        ('classify',), 'sklearn', 1, normal_features = True,
        independent = True),
    'svm': _Traits(_BOTH, 'sklearn', 2, scaled = True, max_rows = 10_000),
    'knn': _Traits(
        _BOTH, 'sklearn', 2, scaled = True, max_rows = 50_000,
        max_features = 30),
    'neural_network': _Traits(
        _BOTH, 'sklearn', 4, scaled = True, min_rows = 1_000)}
# Models whose needs and assumptions `auto_model` knows, but which it only
# considers when they are named in "choices". They report coefficients with
# standard errors, so they suit a study of the features more than a
# prediction. `ols` does not need normal errors: its estimates are unbiased
# without them, and with enough rows its standard errors are about right
# (the Q-Q plot of `auto_plot` checks them).
_OTHER_MODELS: dict[str, _Traits] = {
    'ols': _Traits(
        ('regress',), 'statsmodels', 1, independent = True, wide = False),
    'logit_sm': _Traits(
        ('classify',), 'statsmodels', 1, wide = False, max_classes = 2),
    'probit': _Traits(
        ('classify',), 'statsmodels', 1, wide = False, max_classes = 2),
    'mnlogit': _Traits(
        ('classify',), 'statsmodels', 2, wide = False, min_classes = 3)}
# The samplers that `auto_sampler` considers, in the order it prefers them.
_SAMPLERS: tuple[str, ...] = (
    'random_under', 'smote_tomek', 'smote', 'random_over')


@dataclasses.dataclass
class Profile:
    """What the training rows of a dataset are like.

    Made by `profile` from the real training rows (not the rows that a
    sampler made up), or from a random sample of `_PROFILE_ROWS` of them.

    Args:
        task: "classify", "regress", or `None` if there is no label.
        rows: number of training rows.
        features: number of features.
        numerics: number of numeric features (not booleans).
        binaries: number of binary features: booleans and numbers that are
            only 0 and 1 (such as dummy variables).
        categories: number of categorical features (the `pandas` "category"
            type), which some models use directly.
        text: number of features of text or dates, which models cannot use.
        missing: whether any feature has a missing value.
        unscaled: whether the numeric features are on very different scales
            (the largest standard deviation is more than `_SCALES` times the
            smallest).
        vif: the largest variance inflation factor of the numeric features,
            or `None` if there are fewer than two.
        skewed: share of the numeric and binary features whose distribution
            is materially not normal (binary features never are).
        classes: number of classes of a classified label.
        minority: number of rows of the rarest class.
        balance: rows of the rarest class divided by rows of the most common.
        ordinal: whether the classes are three or more ordered whole numbers
            (such as ratings).
        counts: whether a regressed label is counts (whole numbers of 0 or
            more).
        dispersion: for counts, the Pearson dispersion of a Poisson
            regression (or the variance divided by the mean, without
            statsmodels). Above 1, the counts vary more than a Poisson model
            assumes.
        excess_zeros: for counts, whether there are many more zeros than a
            Poisson model expects.
        skewness: skewness of a regression's errors (the residuals of a
            least squares fit of the numeric features).
        kurtosis: excess kurtosis of the errors.
        normality: p-value of D'Agostino and Pearson's test that the errors
            are normal, or `None` if there are too few rows to test.
        normal: whether the errors are normal: they are not if the test
            rejects normality (or there are too few rows to test) and they
            are materially skewed or heavy-tailed.
        outliers: whether the errors have outliers.

    """

    task: str | None = None
    rows: int = 0
    features: int = 0
    numerics: int = 0
    binaries: int = 0
    categories: int = 0
    text: int = 0
    missing: bool = False
    unscaled: bool = False
    vif: float | None = None
    skewed: float = 0.0
    classes: int | None = None
    minority: int | None = None
    balance: float | None = None
    ordinal: bool = False
    counts: bool = False
    dispersion: float | None = None
    excess_zeros: bool = False
    skewness: float | None = None
    kurtosis: float | None = None
    normality: float | None = None
    normal: bool | None = None
    outliers: bool = False

    """ Properties """

    @property
    def collinear(self) -> bool:
        """Returns whether some numeric features are nearly collinear."""
        return self.vif is not None and self.vif > _VIF

    @property
    def overdispersed(self) -> bool:
        """Returns whether counts vary more than a Poisson model assumes."""
        return self.dispersion is not None and self.dispersion > _DISPERSION

    @property
    def wide(self) -> bool:
        """Returns whether there are at least as many features as rows."""
        return self.features >= self.rows

    """ Public Methods """

    def to_dict(self) -> dict[str, Any]:
        """Returns the profile as plain Python values, for the history.

        Returns:
            The fields of the profile, with floats rounded to 4 places.

        """
        return {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in dataclasses.asdict(self).items()}


def profile(item: base.Dataset) -> Profile:
    """Returns what the real training rows of `item` are like.

    Args:
        item: the dataset to profile.

    Returns:
        The profile of the training rows that a sampler did not make up.

    """
    rows = _real_rows(item)
    if len(rows) > _PROFILE_ROWS:
        generator = np.random.default_rng(item.seed)
        rows = pd.Index(generator.choice(
            rows.to_numpy(), size = _PROFILE_ROWS, replace = False))
    x = item.data.loc[rows, item.features]
    kinds = {column: base._kind_of(x[column]) for column in x.columns}
    numerics = [c for c, k in kinds.items() if k == 'numerics']
    booleans = [c for c, k in kinds.items() if k == 'booleans']
    categories = [
        c for c, k in kinds.items()
        if k == 'categoricals' and isinstance(x[c].dtype, pd.CategoricalDtype)]
    matrix = x[numerics + booleans].astype(float)
    binary = [c for c in matrix.columns if _is_binary(matrix[c])]
    continuous = matrix.drop(columns = binary)
    found = Profile(
        task = item.task,
        rows = len(rows),
        features = len(kinds),
        numerics = len(numerics),
        binaries = len(binary),
        categories = len(categories),
        text = len(kinds) - len(numerics) - len(booleans) - len(categories),
        missing = bool(x.isna().to_numpy().any()),
        unscaled = _unscaled(continuous),
        vif = _largest_vif(continuous),
        skewed = _skewed_share(continuous, len(binary)))
    if item.label is None:
        return found
    y = item.y.loc[rows]
    if item.task == 'classify':
        _profile_classes(found, y)
    elif item.task == 'regress':
        _profile_errors(found, y, matrix)
    return found


def resolve(worker: Any) -> None:
    """Tells each "auto" technique in a workflow which genre to choose from.

    The genre of an `Auto` technique in a step is found from the name of the
    step (such as "sample", "model", or "plot"). The genre of one that is a
    technique of a worker without steps is found from the name of the worker
    (such as "artist"). An `Auto` technique that already has a genre keeps
    it.

    Args:
        worker: a workflow (a `chrisjen.Worker`).

    Raises:
        ValueError: if the genre of an "auto" technique cannot be found from
            its name.

    """
    if not isinstance(worker, chrisjen.Worker):
        return
    for node in worker:
        if isinstance(node, chrisjen.Step):
            contents = node.contents
            if isinstance(contents, Auto) and contents.genre is None:
                step = str(node.name).removeprefix(f'{contents.name}_')
                contents.genre = _genre_of(step)
            else:
                resolve(contents)
        elif isinstance(node, Auto) and node.genre is None:
            node.genre = _genre_of(str(worker.name))
        else:
            resolve(node)


@dataclasses.dataclass
class Chooser(base.Operation, abc.ABC):
    """Base class for techniques that choose other techniques.

    A chooser profiles the dataset (see `profile`), chooses the techniques
    of one genre (`chooses`) that suit it, and applies them. It adds a table to the
    dataset's `tables`, named for the chooser, with a row for each technique
    it considered: whether it was chosen and why (or why not). Its record in
    the dataset's `history` has the profile and the choice.

    Every chooser accepts "choices", the names of the techniques it may
    choose from (by default, those it knows). Any other parameters are passed
    to the techniques it applies, each of which uses those it accepts.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is the snake case name of the class.
        contents: not used. Defaults to `None`.
        parameters: "choices", and parameters for the techniques it applies.
            Defaults to an empty `dict`.

    """

    # The genre of the techniques that the chooser chooses.
    chooses: ClassVar[str] = ''

    """ Private Methods """

    def _apply_each(
        self,
        item: base.Dataset,
        rows: Sequence[tuple[str, bool, str]],
        parameters: Mapping[str, Any],
        *,
        done: str) -> tuple[pd.DataFrame, list[str]]:
        """Applies each chosen technique to `item`, leaving out those that fail.

        Args:
            item: the dataset.
            rows: for each technique considered: its name, whether it was
                chosen, and why (or why not).
            parameters: parameters for the techniques. Each uses those it
                accepts.
            done: what a technique that works is (such as "drawn"), for the
                reason of one that fails.

        Raises:
            Exception: the error of the first technique, if none could be
                applied.

        Returns:
            The table of choices, with a technique that failed marked as not
                chosen and its error as the reason, and the names of the
                techniques that were applied.

        """
        table = pd.DataFrame(
            {'chosen': [chosen for _, chosen, _ in rows],
             'reason': [reason for _, _, reason in rows]},
            index = pd.Index([name for name, _, _ in rows]))
        applied = []
        errors = []
        for name in table.index[table['chosen'].to_numpy(dtype = bool)]:
            try:
                _borrow(name)().apply(item, **parameters)
            # A technique that cannot be applied (because its tool is
            # missing or cannot handle this model, for example) is left out,
            # and the reason is kept.
            except Exception as error:  # noqa: BLE001
                errors.append(error)
                table.loc[name, 'chosen'] = False
                table.loc[name, 'reason'] = f'could not be {done}: {error}'
                warnings.warn(
                    f'{self.name!r} left out {name!r}, which could not be '
                    f'{done}: {error}',
                    stacklevel = 3)
                continue
            applied.append(name)
        if errors and not applied:
            raise errors[0]
        return table, applied

    def _record(
        self,
        item: base.Dataset,
        table: pd.DataFrame,
        found: Profile,
        chosen: Sequence[str],
        **details: Any) -> None:
        """Adds the table of choices to `item` and records the choice.

        Args:
            item: the dataset.
            table: a row for each technique considered.
            found: the profile of the dataset.
            chosen: names of the techniques that were chosen.
            **details: more to record.

        """
        table.index.name = 'technique'
        item.tables[self.name] = table
        item.record(
            self.name,
            genre = self.chooses,
            chosen = list(chosen),
            **details,
            profile = found.to_dict())


@dataclasses.dataclass
class Auto(Chooser):
    """Chooses the techniques of the step that it is in.

    This is the technique named "auto" in settings. When a project is
    drafted, its `genre` is found from the name of its step (such as
    "impute", "encode", "sample", "model", or "plot") or of its worker (such
    as "critic" or "artist"), and it applies the chooser for that genre:
    `auto_imputer`, `auto_encoder`, `auto_sampler`, `auto_model`,
    `auto_evaluator`, or `auto_plot`. Its table is named for that chooser.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            `None`, in which case it is "auto".
        contents: not used. Defaults to `None`.
        parameters: parameters for the chooser. Defaults to an empty `dict`.
        genre: "imputer", "encoder", "sampler", "model", "evaluator", or
            "plot". Defaults to `None`, in which case it is found from the
            name of its step when a project is drafted.

    """

    genre: str | None = None

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Applies the chooser for `genre` to `item`.

        Args:
            item: the dataset.
            **kwargs: parameters for the chooser.

        Raises:
            ValueError: if `genre` is not set or has no chooser.

        Returns:
            The dataset, changed by the techniques that the chooser applied.

        """
        if self.genre is None:
            message = (
                f'{self.name!r} does not know which kind of technique to '
                f'choose: set its genre or use it in a step named for one '
                f'(such as "model"), or use {_chooser_names()}'
            )
            raise ValueError(message)
        kind = _CHOOSERS.get(self.genre)
        if kind is None:
            message = (
                f'"auto" can choose {_names(sorted(_CHOOSERS))}, not '
                f'{self.genre!r} techniques'
            )
            raise ValueError(message)
        result: base.Dataset = kind().apply(item, **kwargs)
        return result


@dataclasses.dataclass
class AutoEncoder(Chooser):
    """Encodes each categorical feature in the way that suits it.

    Models need numbers, so every feature of text or categories is encoded,
    each with the first of these that suits it (the levels of a feature are
    counted in the training rows):

    * `one_hot` for a feature with 15 levels or fewer ("levels"): a dummy
      variable for each level, which every model can use and whose
      coefficients are easy to read.
    * `tfidf` (with skrub) for free text: a feature with more levels whose
      typical value has 3 words or more. It encodes the words of each value.
    * `target` (with category_encoders) for a feature with more levels, if
      the label has two classes or is regressed: the mean label of each
      level, shrunk toward the overall mean for rare levels. Dummy variables
      for so many levels would make the data wide and each level rare.
    * `count` (with category_encoders) for a feature with more levels and a
      label of more classes (or none): how often each level is in the
      training rows.
    * `one_hot` with at most "levels" columns for a feature with more levels,
      if category_encoders is not installed. Its rarest levels share one
      column.

    Dates are split into their parts (year, month, day, and so on) by
    `date_parts`, with skrub.

    The table (named "auto_encoder") has a row for each encoder considered
    (or a row for "none"), with whether it was chosen and why (or why not).
    The history records the encoder of each feature.

    """

    chooses: ClassVar[str] = 'encoder'

    def apply(
        self,
        item: base.Dataset,
        choices: Sequence[str] | str | None = None,
        levels: int = _LEVELS,
        **kwargs: Any) -> base.Dataset:
        """Encodes the categorical features and dates of `item`.

        Args:
            item: the dataset to encode.
            choices: names of the encoders to choose from, in the order to
                prefer them, for every feature. Defaults to `None`, which uses
                those in `_ENCODERS` for each kind of feature.
            levels: most levels of a feature for dummy variables. Defaults to
                15.
            **kwargs: parameters for the encoders (such as "drop" for
                `one_hot`). Each encoder uses those it accepts.

        Raises:
            ValueError: if no encoder suits a feature.

        Returns:
            The dataset, with its categorical features and dates encoded.

        """
        found = profile(item)
        rows = _real_rows(item)
        counts = {
            column: int(item.data.loc[rows, column].nunique())
            for column in item.categoricals}
        encoders: dict[str, str] = {}
        reasons: dict[str, str] = {}
        for column in item.dates + item.categoricals:
            kind = (
                'dates' if column in item.dates
                else _feature_kind(item.data.loc[rows, column], levels))
            names = (
                _ENCODERS[kind] if choices is None
                else camina.listify(choices))
            tried = {name: _screen_encoder(name, kind, found) for name in names}
            chosen = next(
                (name for name, reason in tried.items() if reason is None),
                None)
            if chosen is None:
                message = (
                    f'{self.name!r} found no encoder for {column!r}: '
                    + '; '.join(f'{n} {r}' for n, r in tried.items()))
                raise ValueError(message)
            encoders[column] = chosen
            for name, reason in tried.items():
                if name == chosen:
                    break
                reasons.setdefault(name, cast('str', reason))
        if not encoders:
            table = pd.DataFrame(
                {'chosen': [True],
                 'reason': ['there are no categorical features or dates']},
                index = pd.Index(['none']))
            self._record(item, table, found, [])
            return item
        groups: dict[str, list[str]] = {}
        for column, name in encoders.items():
            groups.setdefault(name, []).append(column)
        for name, columns in groups.items():
            parameters = dict(kwargs)
            if any(counts.get(column, 0) > levels for column in columns):
                parameters.setdefault('max_categories', levels)
            _borrow(name)().apply(item, columns = columns, **parameters)
        explained = {
            name: _encoder_suits(name, columns, counts, levels, found)
            for name, columns in groups.items()}
        for name, reason in reasons.items():
            explained.setdefault(name, reason)
        table = pd.DataFrame(
            {'chosen': [name in groups for name in explained],
             'reason': list(explained.values())},
            index = pd.Index(list(explained)))
        self._record(item, table, found, list(groups), encoders = encoders)
        return item


@dataclasses.dataclass
class AutoEvaluator(Chooser):
    """Applies the evaluators that suit the task and the model.

    It needs the predictions of a fitted model, and applies those of these
    that suit it, in this order:

    * `scorecard`, which compares every branch (or the model) on every
      metric.
    * `classification_report` and `confusion`, for a classified label.
    * `regression_table`, for a model that reports coefficients with
      standard errors (such as `ols`, `logit_sm`, or `fixest`), unless the
      data came from an experiment. The table would have a column for each
      of its models, and estimate the effects of each model without
      coefficients by refitting it many times.
    * `feature_importance`, for a model with its own importances (or
      coefficients).
    * `permutation_importance`, which works for any scikit-learn model,
      when there is no regression table (which describes the features
      better).
    * `shap_importance`, for a model made of trees (with shap installed).
    * `fairness`, for a label of two classes with groups (with fairlearn
      installed).

    Apply it before `auto_plot`, which draws the importances it finds.
    Evaluators that cannot be applied are left out, with a warning, and the
    reason is in the table (named "auto_evaluator").

    """

    chooses: ClassVar[str] = 'evaluator'

    def apply(
        self,
        item: base.Dataset,
        choices: Sequence[str] | str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Applies the evaluators that suit `item`.

        Args:
            item: the dataset, with predictions.
            choices: names of the evaluators to choose from. Defaults to
                `None`, which uses every evaluator that suits the model.
            **kwargs: parameters for the evaluators (such as "n_repeats").
                Each evaluator uses those it accepts.

        Raises:
            ValueError: if there are no predictions.
            Exception: the error of the first evaluator, if none could be
                applied.

        Returns:
            The dataset, with the evaluators' tables and metrics.

        """
        if item.predictions is None:
            message = f'{self.name!r} needs the predictions of a fitted model'
            raise ValueError(message)
        found = profile(item)
        rows = _evaluators(item, found)
        if choices is not None:
            allowed = set(camina.listify(choices))
            rows = [row for row in rows if row[0] in allowed]
        table, applied = self._apply_each(item, rows, kwargs, done = 'applied')
        self._record(item, table, found, applied)
        return item


@dataclasses.dataclass
class AutoImputer(Chooser):
    """Fills missing values in the ways that suit the features.

    Features without missing values are left alone, and so is a dataset
    without any. The numeric features with missing values are filled by the
    first of these that suits them:

    * `iterative_impute`, if a numeric feature misses more than 5% of its
      values in the training rows, there are at least two numeric features,
      and there are no more than 10,000 training rows. It models each
      feature from the others, which keeps their relationships better than
      filling every gap with one value.
    * `median_impute`, which fills each gap with the median of the training
      rows, which skewed values and outliers do not pull as they pull the
      mean.

    The categorical and boolean features are filled with their most common
    values by `mode_impute`.

    The table (named "auto_imputer") has a row for each imputer considered
    (or a row for "none"), with whether it was chosen and why (or why not).

    """

    chooses: ClassVar[str] = 'imputer'

    def apply(
        self,
        item: base.Dataset,
        choices: Sequence[str] | str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Fills the missing values of the features of `item`.

        Args:
            item: the dataset to fill.
            choices: names of the imputers to choose from, in the order to
                prefer them. Defaults to `None`, which uses `_IMPUTERS`.
            **kwargs: parameters for the imputers. Each imputer uses those
                it accepts.

        Returns:
            The dataset, with its missing values filled.

        """
        found = profile(item)
        missing = [
            column for column in item.features
            if base._kind_of(item.data[column]) != 'dates'
            and bool(item.data[column].isna().any())]
        if not missing:
            table = pd.DataFrame(
                {'chosen': [True], 'reason': ['no feature has missing values']},
                index = pd.Index(['none']))
            self._record(item, table, found, [])
            return item
        shares = item.data.loc[_real_rows(item), missing].isna().mean()
        kinds = {column: base._kind_of(item.data[column]) for column in missing}
        numeric = [c for c in missing if kinds[c] == 'numerics']
        most = float(shares[numeric].max()) if numeric else 0.0
        names = list(_IMPUTERS) if choices is None else camina.listify(choices)
        left = list(missing)
        filled: list[str] = []
        explained: dict[str, str] = {}
        for name in names:
            reason = _screen_imputer(
                name, len(item.numerics), len(_real_rows(item)), most)
            if reason is None:
                imputer = cast('type[transformers.Imputer]', _borrow(name))
                fills = imputer.kinds
                columns = [c for c in left if kinds[c] in fills]
                if columns:
                    _borrow(name)().apply(item, columns = columns, **kwargs)
                    left = [c for c in left if c not in columns]
                    filled.append(name)
                    explained[name] = _imputer_suits(name, columns, shares)
                    continue
                reason = (
                    f'suits the data, but {_names(filled, "and")} filled '
                    f'the features it fills' if filled
                    else 'no feature that it fills has missing values')
            explained[name] = reason
        if left:
            warnings.warn(
                f'{self.name!r} could not fill {left}: none of {names} fills '
                f'them',
                stacklevel = 2)
        table = pd.DataFrame(
            {'chosen': [name in filled for name in explained],
             'reason': list(explained.values())},
            index = pd.Index(list(explained)))
        self._record(item, table, found, filled)
        return item


@dataclasses.dataclass
class AutoModel(Chooser):
    """Fits the best of the models whose needs and assumptions the data meets.

    The models it considers (in the order it prefers them) are in
    `_MODELS`, with what each needs and assumes. It rules out the models
    that cannot do the task or use the features (missing values, categorical
    features, more features than rows, or features on very different scales
    for models that need them on similar scales), whose package is not
    installed, or whose assumptions the data does not meet:

    * `linear` assumes normal errors and features that are not collinear,
      so it is ruled out when the residuals of a least squares fit are
      materially skewed or heavy-tailed, or a feature's variance inflation
      factor is above 10.
    * `naive_bayes` assumes normal, independent features.
    * `poisson` assumes counts whose variance equals their mean, so it is
      ruled out for overdispersed counts (and for a label that is not
      counts). `negative_binomial` is for overdispersed counts and
      `zero_inflated_poisson` for counts with more zeros than a Poisson model
      expects.
    * `robust_regression` is for continuous errors with outliers, and
      `ordinal_regression` for three or more ordered classes.
    * `logit`, `ridge`, `lasso`, `svm`, `knn`, and `neural_network` need
      features on similar scales (put a scaler before them).
    * `svm`, `knn`, and `neural_network` are ruled out for data that is too
      large, too small, or too wide for them.

    It also knows the needs of `ols` (features that are not collinear),
    `logit_sm` and `probit` (two classes), and `mnlogit` (three or more
    classes), which report coefficients with standard errors, but it only
    considers them when they are named in "choices".

    The first `candidates` of the rest are cross-validated on the training
    rows (with `stratified_k_fold` for classification and `k_fold` for
    regression), and scored with `metric`. The model with the best score is
    fitted to the training rows, or, if `simplest` is true, the simplest
    model whose score is within one standard error of the best (the
    one-standard-error rule), which is easier to interpret and less likely to
    overfit.

    The table (named "auto_model") has a row for each model considered, with
    whether it was chosen, why it was ruled out (or what it suits), and its
    cross-validated score and standard error.

    """

    chooses: ClassVar[str] = 'model'

    def apply(
        self,
        item: base.Dataset,
        choices: Sequence[str] | str | None = None,
        candidates: int = 4,
        folds: int = 5,
        metric: str | None = None,
        *,
        simplest: bool = True,
        **kwargs: Any) -> base.Dataset:
        """Chooses a model and fits it to the training rows of `item`.

        Args:
            item: the dataset to model. It must have a label.
            choices: names of the models to choose from, in the order to
                prefer them. Defaults to `None`, which uses the models in
                `_MODELS` that can do the task.
            candidates: most models to cross-validate. Defaults to 4.
            folds: number of cross-validation folds. Defaults to 5. If it is
                less than 2 (or the rarest class has fewer than 2 rows), the
                first model that suits the data is chosen without
                cross-validation.
            metric: name of the metric to score each model with. Defaults to
                `None`, which uses "balanced_accuracy" for classification and
                "rmse" for regression.
            simplest: whether to choose the simplest model whose score is
                within one standard error of the best. Defaults to `True`.
            **kwargs: parameters for the models. Each model uses those it
                accepts.

        Raises:
            ValueError: if the dataset has no label, or no model suits it.

        Returns:
            The dataset, with the chosen model fitted and its predictions.

        """
        if item.task is None:
            message = f'{self.name!r} needs a dataset with a label'
            raise ValueError(message)
        found = profile(item)
        names = (
            [n for n, t in _MODELS.items() if found.task in t.tasks]
            if choices is None else camina.listify(choices))
        reasons = {name: _screen_model(name, found) for name in names}
        suited = [name for name, reason in reasons.items() if reason is None]
        if not suited:
            message = (
                f'{self.name!r} found no model that suits the data: '
                + '; '.join(f'{n} {r}' for n, r in reasons.items()))
            raise ValueError(message)
        table = pd.DataFrame(
            {'chosen': False, 'reason': [
                _suits(name, found) if reason is None else reason
                for name, reason in reasons.items()],
             'score': np.nan, 'standard_error': np.nan},
            index = pd.Index(list(reasons)))
        tried = suited[:max(candidates, 1)]
        for name in suited[len(tried):]:
            table.loc[name, 'reason'] = (
                f'suits the data, but only the first {len(tried)} were '
                f'cross-validated')
        chosen = tried[0]
        details: dict[str, Any] = {}
        splits = _splits(item, folds)
        if len(tried) > 1 and splits > 1:
            metric = metric or (
                'balanced_accuracy' if item.task == 'classify' else 'rmse')
            scores = {}
            for name in tried:
                try:
                    scores[name] = _cross_validate(
                        item, name, metric, splits, kwargs)
                # Any error means that the model cannot be fitted to this
                # data, so it is left out (and the reason is kept).
                except Exception as error:  # noqa: BLE001
                    table.loc[name, 'reason'] = f'could not be fitted: {error}'
                    warnings.warn(
                        f'{self.name!r} left out {name!r}, which could not be '
                        f'fitted: {error}',
                        stacklevel = 2)
                    continue
                table.loc[name, ['score', 'standard_error']] = scores[name]
            if not scores:
                message = f'{self.name!r} could not fit any of {tried}'
                raise ValueError(message)
            better = getattr(_borrow(metric), 'greater_is_better', True)
            chosen = _best(scores, better = better, simplest = simplest)
            details = {'metric': metric, 'folds': splits}
        table.loc[chosen, 'chosen'] = True
        _borrow(chosen)().apply(item, **kwargs)
        self._record(item, table, found, [chosen], **details)
        return item


@dataclasses.dataclass
class AutoPlot(Chooser):
    """Draws the figures that suit the data and the model.

    Before a model is fitted (or if "data" is true), it draws the data: the
    label, the numeric features (histograms, their correlations, box plots
    by class, and a pair plot if there are few), the categorical features,
    and where values are missing. After a model is fitted, it draws the
    model:

    * For classification, the confusion matrix and, if the model predicts
      probabilities, the ROC curve, the calibration curve (with enough test
      rows), and the precision-recall curve (if the classes are
      imbalanced, when it says more than the ROC curve).
    * For regression, the actual labels and the residuals against the
      predictions, and, for a model whose standard errors assume normal
      errors (such as `ols`), a Q-Q plot of the residuals and, for `ols` and
      `glm`, the influence of each row.
    * Coefficients with their confidence intervals, if there are any; the
      importance of the features, if the model has them; partial dependence
      for models that are not linear; SHAP values for models made of trees
      (with shap installed); a decision tree's splits; an explainable
      boosting model's shape functions; a hyperparameter search; and
      fairness across groups (with fairlearn installed).

    Plots that cannot be drawn are left out, with a warning, and the reason
    is in the table (named "auto_plot").

    """

    chooses: ClassVar[str] = 'plot'

    def apply(
        self,
        item: base.Dataset,
        choices: Sequence[str] | str | None = None,
        data: bool | None = None,
        **kwargs: Any) -> base.Dataset:
        """Draws the figures that suit `item`.

        Args:
            item: the dataset to draw.
            choices: names of the plots to choose from. Defaults to `None`,
                which uses every plot that suits the data.
            data: whether to draw the data. Defaults to `None`, which draws
                the data if there are no predictions.
            **kwargs: parameters for the plots (such as "style"). Each plot
                uses those it accepts.

        Raises:
            Exception: the error of the first plot, if none could be drawn.

        Returns:
            The dataset, with the new figures.

        """
        found = profile(item)
        rows = _plots(item, found, data = data)
        if choices is not None:
            allowed = set(camina.listify(choices))
            rows = [row for row in rows if row[0] in allowed]
        table, drawn = self._apply_each(item, rows, kwargs, done = 'drawn')
        self._record(item, table, found, drawn)
        return item


@dataclasses.dataclass
class AutoSampler(Chooser):
    """Chooses a sampler that suits the classes and the features, or none.

    A regressed label, or classes that are already balanced (the rarest
    class has at least half the rows of the most common), are not
    resampled. Otherwise, the first of these that suits the data is applied:

    * `random_under`, if the rarest class has so many rows (10,000) that
      removing rows of the common classes still leaves plenty.
    * `smote_tomek`, if the classes overlap (most of the nearest neighbors
      of most rare rows are of other classes), since it removes pairs of
      rows from different classes that are each other's nearest neighbors
      after it adds rows.
    * `smote`, which adds rows between rare rows and their neighbors.
    * `random_over`, which duplicates rare rows. SMOTE and its variants need
      numeric features without missing values, mostly not binary (they make
      values between those of two rows, which a dummy variable cannot
      have), and at least 6 rows of the rarest class.

    The table (named "auto_sampler") has a row for each sampler considered
    (or a row for "none"), with whether it was chosen and why (or why not).

    """

    chooses: ClassVar[str] = 'sampler'

    def apply(
        self,
        item: base.Dataset,
        choices: Sequence[str] | str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Resamples the training rows of `item`, if they need it.

        Args:
            item: the dataset to resample.
            choices: names of the samplers to choose from, in the order to
                prefer them. Defaults to `None`, which uses `_SAMPLERS`.
            **kwargs: parameters for the sampler (such as
                "sampling_strategy"). Each sampler uses those it accepts.

        Raises:
            ValueError: if the dataset has no label.

        Returns:
            The dataset, resampled if it needed to be.

        """
        if item.task is None:
            message = f'{self.name!r} needs a dataset with a label'
            raise ValueError(message)
        found = profile(item)
        skip = _no_sampling(found)
        if skip is not None:
            table = pd.DataFrame(
                {'chosen': [True], 'reason': [skip]},
                index = pd.Index(['none']))
            self._record(item, table, found, [])
            return item
        overlap = _overlap(item)
        names = (
            list(_SAMPLERS) if choices is None else camina.listify(choices))
        reasons = {
            name: _screen_sampler(name, found, overlap) for name in names}
        chosen = next(
            (name for name, reason in reasons.items() if reason is None),
            None)
        if chosen is None:
            message = (
                f'{self.name!r} found no sampler that suits the data: '
                + '; '.join(f'{n} {r}' for n, r in reasons.items()))
            raise ValueError(message)
        explained = {
            name: (
                _sampler_suits(name, found, overlap) if name == chosen
                else f'suits the data, but {chosen} is preferred'
                if reason is None else reason)
            for name, reason in reasons.items()}
        table = pd.DataFrame(
            {'chosen': [name == chosen for name in reasons],
             'reason': list(explained.values())},
            index = pd.Index(list(reasons)))
        _borrow(chosen)().apply(item, **kwargs)
        self._record(item, table, found, [chosen])
        return item


# The chooser for each genre, which an `Auto` technique applies, in the order
# they are usually applied.
_CHOOSERS: dict[str, type[Chooser]] = {
    'imputer': AutoImputer,
    'encoder': AutoEncoder,
    'sampler': AutoSampler,
    'model': AutoModel,
    'evaluator': AutoEvaluator,
    'plot': AutoPlot}


""" Private Functions """


def _best(
    scores: Mapping[str, tuple[float, float]],
    *,
    better: bool,
    simplest: bool) -> str:
    """Returns the name of the model to choose from its scores.

    Args:
        scores: the mean and standard error of each model's scores, in the
            order the models are preferred.
        better: whether higher scores are better.
        simplest: whether to choose the simplest model within one standard
            error of the best, rather than the best.

    Returns:
        The name of the chosen model. Ties go to the model listed first.

    """
    sign = 1.0 if better else -1.0
    signed = {name: sign * mean for name, (mean, _) in scores.items()}
    best = max(signed, key = signed.__getitem__)
    if not simplest:
        return best
    error = scores[best][1]
    threshold = signed[best] - (0.0 if math.isnan(error) else error)
    close = [name for name in scores if signed[name] >= threshold]
    return min(close, key = _complexity)


def _borrow(name: str) -> type[base.Operation]:
    """Returns the technique class named `name`.

    Args:
        name: name of a technique in the library.

    Returns:
        The class.

    """
    return cast('type[base.Operation]', chrisjen.library.borrow(name))


def _chooser_names() -> str:
    """Returns the names of the choosers, as a list in words.

    Returns:
        The name of the chooser of each genre, such as "auto_model", in the
            order of `_CHOOSERS`.

    """
    return _names([f'auto_{genre}' for genre in _CHOOSERS])


def _complexity(name: str) -> int:
    """Returns how complex a model is, from 1 (the simplest) up.

    Args:
        name: name of a model.

    Returns:
        Its complexity in `_MODELS` or `_OTHER_MODELS`, or 3 for a model that
            is in neither.

    """
    traits = _traits_of(name)
    return 3 if traits is None else traits.complexity


def _cross_validate(
    item: base.Dataset,
    name: str,
    metric: str,
    splits: int,
    parameters: Mapping[str, Any]) -> tuple[float, float]:
    """Returns the cross-validated score of a model on the training rows.

    The model is fitted to a copy of `item`, which a validator then scores.

    Args:
        item: the dataset to model.
        name: name of the model.
        metric: name of the metric to score with.
        splits: number of folds.
        parameters: parameters for the model.

    Raises:
        ValueError: if the metric could not score the model.

    Returns:
        The mean score of the folds and its standard error.

    """
    trial = copy.deepcopy(item)
    _borrow(name)().apply(trial, **parameters)
    validator = (
        'stratified_k_fold' if item.task == 'classify' else 'k_fold')
    _borrow(validator)().apply(trial, metrics = [metric], n_splits = splits)
    scores = trial.history[-1].get('scores', {}).get(metric)
    if scores is None:
        message = f'{metric!r} could not score {name!r}'
        raise ValueError(message)
    error = scores['sd'] / math.sqrt(splits)
    return float(scores['mean']), float(error)


def _described(columns: Sequence[str], counts: Mapping[str, int]) -> str:
    """Returns the names of features with how many levels each has.

    Args:
        columns: names of the features.
        counts: number of levels of each categorical feature.

    Returns:
        The features as a list in words, such as "court (3 levels) and judge
            (212 levels)". A feature without a count is named alone.

    """
    return _names(
        [f'{column} ({counts[column]} levels)' if column in counts else column
         for column in columns],
        'and')


def _encoder_suits(
    name: str,
    columns: Sequence[str],
    counts: Mapping[str, int],
    levels: int,
    found: Profile) -> str:
    """Returns why an encoder suits the features that it encoded.

    Args:
        name: name of the encoder.
        columns: names of the features it encoded.
        counts: number of levels of each categorical feature.
        levels: most levels of a feature for dummy variables.
        found: the profile of the dataset.

    Returns:
        The reason.

    """
    features = _described(columns, counts)
    if name == 'date_parts':
        return f'splits {features} into their parts (year, month, and so on)'
    if name == 'tfidf':
        return f'encodes the words of {features}, which are free text'
    if name in _SUPERVISED:
        return (
            f'encodes {features} by the mean label of each level, since '
            f'dummy variables for more than {levels} levels would make each '
            f'level rare')
    if name == 'count':
        why = (
            'there is no label' if found.task is None
            else f'the label has {found.classes} classes, too many to '
            f'encode by the mean label')
        return (
            f'encodes {features} by how often each level occurs, since they '
            f'have more than {levels} levels and {why}')
    if name == 'one_hot':
        few = [c for c in columns if counts.get(c, 0) <= levels]
        many = [c for c in columns if counts.get(c, 0) > levels]
        reasons = []
        if few:
            reasons.append(
                f'makes dummy variables for {_described(few, counts)}, which '
                f'have {levels} levels or fewer')
        if many:
            reasons.append(
                f'makes at most {levels} dummy variables for '
                f'{_described(many, counts)}, whose rarest levels share one')
        return '; '.join(reasons)
    return f'encodes {features}'


def _evaluators(
    item: base.Dataset,
    found: Profile) -> list[tuple[str, bool, str]]:
    """Returns the evaluators that suit the model of `item`, and why (or not).

    Args:
        item: the dataset with predictions.
        found: its profile.

    Returns:
        For each evaluator considered: its name, whether it suits the model,
            and why (or why not).

    """
    model = _model_name(item)
    fitted = plots._unwrapped(item) if item.model is not None else None
    classify = found.task == 'classify'
    coefficients = _has_coefficients(item)
    # A regression table has a column for each branch of an experiment, and
    # estimates the marginal effects of a model without coefficients by
    # refitting it many times, which is slow for several such models.
    tabulated = coefficients and len(item.branches) < 2  # noqa: PLR2004
    # scikit-learn shuffles the features only of its own estimators.
    shuffled = not tabulated and hasattr(item.model, '__sklearn_tags__')
    importances = (
        hasattr(fitted, 'feature_importances_') or hasattr(fitted, 'coef_'))
    trees = model in _TREES and _importable('shap')
    fair = (
        classify and found.classes == 2 and bool(item.groups)  # noqa: PLR2004
        and _importable('fairlearn'))
    return [
        ('scorecard', True, 'compares the models on every metric'),
        ('classification_report', classify,
         'reports precision, recall, and F1 for each class' if classify
         else 'needs a classified label'),
        ('confusion', classify,
         'counts which classes the model confuses' if classify
         else 'needs a classified label'),
        ('regression_table', tabulated,
         'reports the coefficients with their standard errors, ready to '
         'publish' if tabulated
         else f'{model} has no coefficients with standard errors'
         if not coefficients
         else 'the models of an experiment are compared by the scorecard '
         '(add regression_table to the critic to estimate the effects of '
         'each)'),
        ('feature_importance', importances,
         f'reports the importance that {model} gives each feature'
         if importances
         else f'{model} does not report the importance of features'),
        ('permutation_importance', shuffled,
         'measures how much the score drops when each feature is shuffled, '
         'which works for any scikit-learn model' if shuffled
         else 'the coefficients and their standard errors describe the '
         'features' if tabulated
         else f'needs a scikit-learn model, not {model}'),
        ('shap_importance', trees,
         'explains how each feature moves each prediction' if trees
         else 'needs a model of trees and shap'),
        ('fairness', fair,
         'compares the model across the groups' if fair
         else 'needs a label of two classes, groups, and fairlearn')]


def _feature_kind(values: pd.Series, levels: int) -> str:
    """Returns the kind of a categorical feature, for choosing its encoder.

    Args:
        values: the values of the feature in the training rows.
        levels: most levels of a feature for dummy variables.

    Returns:
        "few" if it has no more than `levels` levels, "text" if it has more
            and its typical value has at least `_WORDS` words, and "many"
            otherwise.

    """
    values = values.dropna()
    if values.nunique() <= levels:
        return 'few'
    words = values.astype(str).str.count(r'\S+')
    if words.median() >= _WORDS:
        return 'text'
    return 'many'


def _genre_of(name: str) -> str:
    """Returns the genre of the techniques that a step or worker applies.

    Args:
        name: name of the step or worker.

    Raises:
        ValueError: if the name is not one of the names in `_STEPS`.

    Returns:
        "sampler", "model", or "plot".

    """
    for genre, names in _STEPS.items():
        if name in names:
            return genre
    message = (
        f'"auto" cannot tell which techniques to choose for {name!r}: name '
        f'the step for them (such as "encode", "sample", "model", or '
        f'"plot") or use {_chooser_names()}'
    )
    raise ValueError(message)


def _has_coefficients(item: base.Dataset) -> bool:
    """Returns whether the model of `item` reported coefficients with intervals.

    Args:
        item: the dataset.

    Returns:
        Whether any of its tables has coefficients with confidence intervals
            (as the statsmodels and pyfixest models make).

    """
    return any(
        set(plots._INTERVAL_COLUMNS) <= set(table.columns)
        for table in item.tables.values())


def _imputer_suits(
    name: str,
    columns: Sequence[str],
    shares: pd.Series) -> str:
    """Returns why an imputer suits the features that it filled.

    Args:
        name: name of the imputer.
        columns: names of the features it filled.
        shares: share of the values of each feature with missing values that
            are missing in the training rows.

    Returns:
        The reason.

    """
    features = _names(list(columns), 'and')
    if name == 'iterative_impute':
        filled = shares[list(columns)]
        worst = str(filled.idxmax())
        verb = 'is' if len(columns) == 1 else 'are each'
        return (
            f'{filled.max():.0%} of the values of {worst} are missing, so '
            f'{features} {verb} modeled from the other features, which keeps '
            f'their relationships')
    if name == 'median_impute':
        return (
            f'fills {features} with the median of each, which skewed values '
            f'and outliers do not pull')
    if name == 'mode_impute':
        return f'fills {features} with the most common value of each'
    return f'fills {features}'


@functools.cache
def _importable(package: str) -> bool:
    """Returns whether `package` can be imported.

    Args:
        package: name of the package.

    Returns:
        Whether importing it succeeds.

    """
    try:
        importlib.import_module(package)
    # Some optional packages are installed but fail when they are imported
    # (such as xgboost without OpenMP).
    except Exception:  # noqa: BLE001
        return False
    return True


def _is_binary(column: pd.Series) -> bool:
    """Returns whether a numeric column has only the values 0 and 1.

    Args:
        column: the column, as floats.

    Returns:
        Whether every value that is not missing is 0 or 1.

    """
    values = column.dropna()
    return bool(values.isin([0.0, 1.0]).all())


def _largest_vif(matrix: pd.DataFrame) -> float | None:
    """Returns the largest variance inflation factor of the columns.

    Args:
        matrix: numeric columns.

    Returns:
        The largest diagonal value of the inverse of the correlation matrix
            of the columns that vary (in the rows without missing values), or
            `None` if fewer than two columns vary. Perfectly collinear columns
            have a factor of infinity.

    """
    complete = matrix.dropna()
    complete = complete.loc[:, complete.std() > 0]
    if complete.shape[1] < 2 or len(complete) <= complete.shape[1]:  # noqa: PLR2004
        return None
    correlations = np.corrcoef(complete.to_numpy(), rowvar = False)
    if np.linalg.matrix_rank(correlations) < len(correlations):
        return math.inf
    return float(np.max(np.diag(np.linalg.inv(correlations))))


def _names(items: Sequence[str], conjunction: str = 'or') -> str:
    """Returns the items as a list in words, such as "a, b, or c".

    Args:
        items: the items.
        conjunction: the word before the last item. Defaults to "or".

    Returns:
        The items joined by commas and `conjunction`.

    """
    if len(items) < 3:  # noqa: PLR2004
        return f' {conjunction} '.join(items)
    return f'{", ".join(items[:-1])}, {conjunction} {items[-1]}'


def _no_sampling(found: Profile) -> str | None:
    """Returns why the training rows need no resampling, if they do not.

    Args:
        found: the profile of the dataset.

    Returns:
        The reason, or `None` if the classes should be resampled.

    """
    if found.task != 'classify':
        return 'only the classes of a classified label are resampled'
    balance = found.balance or 0.0
    if balance >= _BALANCED:
        return (
            f'the classes are balanced: the rarest has {balance:.0%} of '
            f'the rows of the most common')
    return None


def _overlap(item: base.Dataset) -> float | None:
    """Returns the share of rare rows whose neighbors are mostly of others.

    The rare rows are those of every class but the most common. The numeric
    features are standardized before the neighbors are found.

    Args:
        item: the dataset.

    Returns:
        The share, or `None` if the features are not all numbers without
            missing values.

    """
    rows = _real_rows(item)
    if len(rows) > _PROFILE_ROWS:
        generator = np.random.default_rng(item.seed)
        rows = pd.Index(generator.choice(
            rows.to_numpy(), size = _PROFILE_ROWS, replace = False))
    x = item.data.loc[rows, item.features]
    if any(base._kind_of(x[c]) not in {'numerics', 'booleans'} for c in x):
        return None
    matrix = x.astype(float)
    if matrix.isna().to_numpy().any() or matrix.empty:
        return None
    spread = matrix.std().replace(0, 1).fillna(1)
    values = ((matrix - matrix.mean()) / spread).to_numpy()
    y = item.y.loc[rows].to_numpy()
    common = pd.Series(y).value_counts().index[0]
    rare = np.flatnonzero(y != common)
    count = min(5, len(y) - 1)
    if count < 1 or len(rare) == 0:
        return None
    neighbors = importlib.import_module('sklearn.neighbors')
    finder = neighbors.NearestNeighbors(n_neighbors = count + 1).fit(values)
    _, nearest = finder.kneighbors(values[rare])
    # The nearest row of each rare row is itself.
    others = y[nearest[:, 1:]] != y[rare][:, None]
    return float(np.mean(others.mean(axis = 1) > 0.5))  # noqa: PLR2004


def _plots(
    item: base.Dataset,
    found: Profile,
    *,
    data: bool | None) -> list[tuple[str, bool, str]]:
    """Returns the plots that suit `item`, and why (or why not).

    Args:
        item: the dataset to draw.
        found: its profile.
        data: whether to draw the data, or `None` to draw it only if there
            are no predictions.

    Returns:
        For each plot considered: its name, whether it suits the data, and
            why (or why not).

    """
    predicted = item.predictions is not None
    rows: list[tuple[str, bool, str]] = []
    if data or (data is None and not predicted):
        rows.extend(_data_plots(item, found))
    if predicted:
        rows.extend(_model_plots(item, found))
    return rows


def _data_plots(
    item: base.Dataset,
    found: Profile) -> list[tuple[str, bool, str]]:
    """Returns the plots of the data that suit `item`, and why (or why not).

    Args:
        item: the dataset to draw.
        found: its profile.

    Returns:
        For each plot of the data: its name, whether it suits the data, and
            why (or why not).

    """
    numerics = len(item.numerics)
    categoricals = len(item.categoricals)
    missing = bool(item.data.isna().to_numpy().any())
    return [
        ('label_plot', item.label is not None,
         'shows the distribution of the label' if item.label is not None
         else 'there is no label'),
        ('missing_heatmap', missing,
         'shows where values are missing' if missing
         else 'no values are missing'),
        ('histograms', numerics > 0,
         f'shows the distribution of the {numerics} numeric features'
         if numerics else 'there are no numeric features'),
        ('correlation_heatmap', numerics > 1,
         'shows the correlations between the numeric features'
         if numerics > 1 else 'there are fewer than two numeric features'),
        ('box_plots', found.task == 'classify' and numerics > 0,
         'compares the numeric features across the classes'
         if found.task == 'classify' and numerics
         else 'needs a classified label and numeric features'),
        ('pair_plot', 1 < numerics <= _PAIRS,
         'shows each pair of the numeric features'
         if 1 < numerics <= _PAIRS
         else f'needs 2 to {_PAIRS} numeric features, not {numerics}'),
        ('count_plots', categoricals > 0,
         f'shows the values of the {categoricals} categorical features'
         if categoricals else 'there are no categorical features')]


def _model_plots(
    item: base.Dataset,
    found: Profile) -> list[tuple[str, bool, str]]:
    """Returns the plots of the model that suit `item`, and why (or why not).

    Args:
        item: the dataset with predictions.
        found: its profile.

    Returns:
        For each plot of the model: its name, whether it suits the model, and
            why (or why not).

    """
    model = _model_name(item)
    fitted = plots._unwrapped(item) if item.model is not None else None
    rows: list[tuple[str, bool, str]] = []
    if found.task == 'classify':
        probabilities = item.probabilities is not None
        no_probabilities = 'the model does not predict probabilities'
        count = 0 if item.predictions is None else len(item.predictions)
        balance = found.balance or 0.0
        rows.extend([
            ('confusion_heatmap', True,
             'shows which classes the model confuses'),
            ('roc_curve', probabilities,
             'shows the tradeoff between true and false positives'
             if probabilities else no_probabilities),
            ('precision_recall_curve',
             probabilities and balance < _BALANCED,
             'shows precision and recall, which say more than the ROC curve '
             'when the classes are imbalanced'
             if probabilities and balance < _BALANCED
             else 'the classes are balanced' if probabilities
             else no_probabilities),
            ('calibration_curve',
             probabilities and count >= _CALIBRATION_ROWS,
             'shows whether the predicted probabilities can be trusted'
             if probabilities and count >= _CALIBRATION_ROWS
             else f'needs at least {_CALIBRATION_ROWS} predicted rows'
             if probabilities else no_probabilities)])
    elif found.task == 'regress':
        normal = model in _NORMAL_ERRORS
        influence = (
            isinstance(fitted, models.Statsmodel)
            and getattr(fitted, 'kind', None) in plots._INFLUENCE)
        rows.extend([
            ('actual_vs_predicted', True,
             'shows how close the predictions are to the label'),
            ('residual_plot', True,
             ('shows whether the errors depend on the prediction (such as '
              'unequal variance or a missed curve)')),
            ('qq_plot', normal and _importable('statsmodels'),
             f'{model} assumes normal errors, which a Q-Q plot checks'
             if normal else f'{model} does not assume normal errors'),
            ('influence_plot', influence,
             'shows the rows that sway the coefficients' if influence
             else 'needs an ols or glm model')])
    coefficients = _has_coefficients(item)
    importances = (
        any(name in item.tables for name in plots._IMPORTANCE_LABELS)
        or hasattr(fitted, 'feature_importances_')
        or hasattr(fitted, 'coef_'))
    # Partial dependence is drawn for the most important features.
    nonlinear = (
        model in _NONLINEAR and hasattr(fitted, 'get_params') and importances)
    trees = model in _TREES and _importable('shap')
    searched = any(name.endswith('_search') for name in item.tables)
    fair = (
        found.task == 'classify' and bool(item.groups)
        and _importable('fairlearn'))
    rows.extend([
        ('coefficient_plot', coefficients,
         'shows the coefficients and their confidence intervals'
         if coefficients else 'there are no coefficients'),
        ('importance_plot', importances,
         'shows the most important features' if importances
         else f'{model} does not report the importance of features'),
        ('partial_dependence', nonlinear,
         f'shows how the prediction of {model} changes with the most '
         f'important features' if nonlinear
         else f'{model} is linear, so its coefficients show its shape'
         if model not in _NONLINEAR
         else f'{model} does not report the importance of features'),
        ('shap_beeswarm', trees,
         'shows how each feature moves each prediction' if trees
         else 'needs a model of trees and shap'),
        ('tree_plot', model == 'decision_tree',
         'shows the splits of the tree' if model == 'decision_tree'
         else 'needs a decision_tree model'),
        ('shape_functions', model == 'explainable_boosting',
         'shows the shape function of each feature'
         if model == 'explainable_boosting'
         else 'needs an explainable_boosting model'),
        ('search_plot', searched,
         'shows the score of each try of the search' if searched
         else 'the model was not searched'),
        ('fairness_plot', fair,
         'compares the model across the groups' if fair
         else 'needs a classifier, groups, and fairlearn')])
    return rows


def _model_name(item: base.Dataset) -> str | None:
    """Returns the name of the last model applied to `item`.

    Args:
        item: the dataset.

    Returns:
        The name of the last technique in the history that is a model, or
            `None` if there is none.

    """
    names = set(chrisjen.utilities._flatten_dict(
        chrisjen.library.get_genre('model')))
    for record in reversed(item.history):
        if record.get('technique') in names:
            return str(record['technique'])
    return None


def _profile_classes(found: Profile, y: pd.Series) -> None:
    """Adds what a classified label is like to `found`.

    Args:
        found: the profile to add to.
        y: the labels of the training rows.

    """
    counts = y.value_counts()
    found.classes = len(counts)
    if counts.empty:
        return
    found.minority = int(counts.min())
    found.balance = float(counts.min() / counts.max())
    numeric = (
        pd.api.types.is_numeric_dtype(y.dtype)
        and not pd.api.types.is_bool_dtype(y.dtype))
    if numeric and found.classes > 2:  # noqa: PLR2004
        values = y.dropna().astype(float)
        found.ordinal = bool((values == values.round()).all())


def _profile_errors(
    found: Profile,
    y: pd.Series,
    matrix: pd.DataFrame) -> None:
    """Adds what a regressed label and its errors are like to `found`.

    The errors are the residuals of a least squares fit of the label to the
    numeric and boolean features (in the rows without missing values), or
    the label minus its mean if there are too few rows for the fit.

    Args:
        found: the profile to add to.
        y: the labels of the training rows.
        matrix: the numeric and boolean features of the same rows, as floats.

    """
    stats = importlib.import_module('scipy.stats')
    complete = matrix.join(y.rename('__label__').astype(float)).dropna()
    labels = complete.pop('__label__').to_numpy()
    complete = complete.loc[:, complete.std() > 0]
    design = np.column_stack([np.ones(len(complete)), complete.to_numpy()])
    if len(labels) > design.shape[1] + 1:
        coefficients = np.linalg.lstsq(design, labels, rcond = None)[0]
        residuals = labels - design @ coefficients
    else:
        residuals = labels - labels.mean()
    if len(residuals) > 2 and np.std(residuals) > 0:  # noqa: PLR2004
        found.skewness = float(stats.skew(residuals))
        found.kurtosis = float(stats.kurtosis(residuals))
        if len(residuals) >= _TESTABLE:
            found.normality = float(stats.normaltest(residuals).pvalue)
        material = (
            abs(found.skewness) > _SKEW or abs(found.kurtosis) > _KURTOSIS)
        rejected = found.normality is None or found.normality < _ALPHA
        found.normal = not (material and rejected)
        median = np.median(residuals)
        spread = 1.4826 * np.median(np.abs(residuals - median))
        if spread > 0:
            scores = np.abs(residuals - median) / spread
            found.outliers = bool(
                np.mean(scores > _OUTLIER_SCORE) > _OUTLIERS)
    values = y.dropna().astype(float)
    found.counts = bool(
        len(values) > 0 and (values >= 0).all()
        and (values == values.round()).all())
    if found.counts:
        _profile_counts(found, labels, design)


def _profile_counts(
    found: Profile,
    labels: np.ndarray,
    design: np.ndarray) -> None:
    """Adds the dispersion and excess zeros of counts to `found`.

    The dispersion is the Pearson statistic of a Poisson regression divided
    by its residual degrees of freedom (with statsmodels), which is about 1
    if the counts vary as a Poisson model assumes. Without statsmodels, or
    if the regression cannot be fitted, it is the variance of the counts
    divided by their mean.

    Args:
        found: the profile to add to.
        labels: the counts, in the rows without missing values.
        design: the features of the same rows, with a column of ones first.

    """
    mean = float(np.mean(labels)) if len(labels) else 0.0
    if mean <= 0:
        return
    means = np.full(len(labels), mean)
    found.dispersion = float(np.var(labels, ddof = 1) / mean)
    if _importable('statsmodels') and len(labels) > design.shape[1] + 1:
        api = importlib.import_module('statsmodels.api')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit = api.GLM(
                    labels, design, family = api.families.Poisson()).fit()
            found.dispersion = float(fit.pearson_chi2 / fit.df_resid)
            means = np.asarray(fit.fittedvalues, dtype = float)
        # A regression that does not converge leaves the simpler estimate.
        except Exception:  # noqa: BLE001, S110
            pass
    expected = float(np.mean(np.exp(-means)))
    observed = float(np.mean(labels == 0))
    found.excess_zeros = observed - expected > _ZEROS


def _real_rows(item: base.Dataset) -> pd.Index:
    """Returns the training rows that a sampler did not make up.

    Args:
        item: the dataset.

    Returns:
        The index labels of the rows.

    """
    rows = item._train_rows()
    return rows[~rows.isin(item.synthetic)]


def _sampler_suits(name: str, found: Profile, overlap: float | None) -> str:
    """Returns why a sampler suits the data.

    Args:
        name: name of the sampler.
        found: the profile of the dataset.
        overlap: share of the rare rows whose neighbors are mostly of other
            classes.

    Returns:
        The reason.

    """
    balance = f'the rarest class has {found.balance or 0.0:.0%} of the rows'
    if name == 'random_under':
        return (
            f'{balance}, and with {found.minority} rows of it, removing rows '
            f'of the common classes leaves plenty')
    if name == 'smote_tomek':
        return (
            f'{balance}, and the classes overlap ({overlap or 0.0:.0%} of '
            f'the rare rows are mostly near other classes)')
    if name == 'smote':
        return f'{balance}, and the features are numeric'
    return f'{balance} of the most common'


def _screen_encoder(name: str, kind: str, found: Profile) -> str | None:
    """Returns why an encoder does not suit a feature, or `None` if it does.

    Args:
        name: name of the encoder.
        kind: the kind of the feature: "dates", or "few", "many", or "text"
            for a categorical feature (see `_feature_kind`).
        found: the profile of the dataset.

    Returns:
        The reason the encoder is ruled out, or `None`.

    """
    try:
        technique = _borrow(name)
    except KeyError:
        return 'is not a technique'
    if not issubclass(technique, transformers.Encoder):
        return 'is not an encoder'
    dates = 'dates' in technique.kinds
    if kind == 'dates' and not dates:
        return 'cannot encode dates'
    if kind != 'dates' and dates:
        return 'only encodes dates'
    source = getattr(technique, 'contents', None)
    package = source.split('.')[0] if isinstance(source, str) else None
    if package is not None and not _importable(package):
        return f'needs {package}, which is not installed'
    if name in _SUPERVISED:
        classes = found.classes or 0
        if found.task is None:
            return 'needs a label to learn from'
        if found.task == 'classify' and classes > 2:  # noqa: PLR2004
            return (
                f'needs a label of two classes or a regressed one, not '
                f'{found.classes} classes')
        if name == 'weight_of_evidence' and found.task == 'regress':
            return 'needs a label of two classes'
    return None


def _screen_imputer(
    name: str,
    numerics: int,
    rows: int,
    most: float) -> str | None:
    """Returns why an imputer does not suit the data, or `None` if it does.

    Args:
        name: name of the imputer.
        numerics: number of numeric features.
        rows: number of real training rows.
        most: largest share of the values of a numeric feature that are
            missing in the training rows.

    Returns:
        The reason the imputer is ruled out, or `None`.

    """
    try:
        technique = _borrow(name)
    except KeyError:
        return 'is not a technique'
    if not issubclass(technique, transformers.Imputer):
        return 'is not an imputer'
    if name != 'iterative_impute':
        return None
    if most == 0:
        return 'no numeric feature has missing values'
    if numerics < 2:  # noqa: PLR2004
        return 'needs at least two numeric features to model each from others'
    if rows > _PLENTY:
        return f'is slow for more than {_PLENTY} rows'
    if most <= _MANY_MISSING:
        return (
            f'few values are missing (at most {most:.0%} of a numeric '
            f'feature), so filling them with one value changes little')
    return None


def _screen_model(name: str, found: Profile) -> str | None:
    """Returns why a model does not suit the data, or `None` if it does.

    Args:
        name: name of the model.
        found: the profile of the dataset.

    Returns:
        The reason the model is ruled out, or `None`.

    """
    try:
        kind = _borrow(name)
    except KeyError:
        return 'is not a technique'
    if not issubclass(kind, models.Model):
        return 'is not a model'
    task = found.task or ''
    tools = getattr(kind, 'tools', {})
    traits = _traits_of(name) or _Traits(
        tuple(tools) or _BOTH, 'sklearn', 3, missing = True,
        categories = True)
    if task not in traits.tasks or (tools and task not in tools):
        return f'cannot {task}'
    reason = None
    if not _importable(traits.package):
        reason = f'needs {traits.package}, which is not installed'
    elif found.text:
        reason = 'cannot use text or dates (encode them first)'
    elif found.missing and not traits.missing:
        reason = 'cannot use missing values (impute them first)'
    elif found.categories and not traits.categories:
        reason = 'cannot use categorical features (encode them first)'
    elif not traits.wide and found.wide:
        reason = (
            f'needs more rows than features ({found.rows} rows and '
            f'{found.features} features)')
    elif traits.scaled and found.unscaled:
        reason = 'needs features on similar scales (scale them first)'
    else:
        reason = _screen_assumptions(traits, found)
    return reason


def _screen_assumptions(traits: _Traits, found: Profile) -> str | None:
    """Returns which assumption of a model the data does not meet, if any.

    Args:
        traits: what the model needs and assumes.
        found: the profile of the dataset.

    Returns:
        The reason the model is ruled out, or `None`.

    """
    classes = found.classes or 0
    if traits.max_classes is not None and classes > traits.max_classes:
        return f'is for {traits.max_classes} classes, not {classes}'
    if classes < traits.min_classes:
        return f'is for {traits.min_classes} or more classes, not {classes}'
    if traits.ordinal and not found.ordinal:
        return 'is for three or more ordered classes'
    if traits.counts is not None and not found.counts:
        return 'is for a label of counts'
    if traits.counts == 'equidispersed' and found.overdispersed:
        return (
            f'assumes that the variance of the counts equals their mean, but '
            f'they are overdispersed (dispersion {found.dispersion:.2f})')
    if traits.counts == 'overdispersed' and not found.overdispersed:
        return (
            f'is for overdispersed counts, but the dispersion is '
            f'{found.dispersion or 0.0:.2f}')
    if traits.counts == 'zero_inflated' and not found.excess_zeros:
        return 'is for counts with more zeros than a Poisson model expects'
    if traits.outliers and found.counts:
        return 'is for continuous errors, but the label is counts'
    if traits.outliers and not found.outliers:
        return 'is for errors with outliers, and there are few'
    if traits.normal_errors and found.normal is False:
        return (
            f'assumes normal errors, but they are not (skewness '
            f'{found.skewness or 0.0:.2f}, excess kurtosis '
            f'{found.kurtosis or 0.0:.2f})')
    if traits.normal_features and found.skewed > 0.5:  # noqa: PLR2004
        return (
            f'assumes normal features, but {found.skewed:.0%} of them are '
            f'not')
    if traits.independent and found.collinear:
        return (
            f'assumes features that are not collinear, but the largest '
            f'variance inflation factor is {found.vif:.1f}')
    if found.rows < traits.min_rows:
        return f'needs at least {traits.min_rows} rows, not {found.rows}'
    if traits.max_rows is not None and found.rows > traits.max_rows:
        return f'is too slow for more than {traits.max_rows} rows'
    if traits.max_features is not None and found.features > traits.max_features:
        return (
            f'works poorly with more than {traits.max_features} features')
    return None


def _screen_sampler(
    name: str,
    found: Profile,
    overlap: float | None) -> str | None:
    """Returns why a sampler does not suit the data, or `None` if it does.

    Args:
        name: name of the sampler.
        found: the profile of the dataset.
        overlap: share of the rare rows whose neighbors are mostly of other
            classes, or `None` if it could not be found.

    Returns:
        The reason the sampler is ruled out, or `None`.

    """
    try:
        kind = _borrow(name)
    except KeyError:
        return 'is not a technique'
    if chrisjen.library.classify(kind) != 'sampler':
        return 'is not a sampler'
    minority = found.minority or 0
    if name == 'random_under':
        if minority < _PLENTY:
            return (
                f'would leave too few rows: the rarest class has {minority}, '
                f'fewer than {_PLENTY}')
        return None
    if name == 'random_over':
        return None
    if found.text or found.categories:
        return 'needs numeric features (encode the others first)'
    if found.missing:
        return 'cannot use missing values (impute them first)'
    if found.features and found.binaries / found.features > _BINARY:
        return (
            'would make values between 0 and 1 for binary features, which '
            'are most of the features')
    if minority < _NEIGHBORS:
        return (
            f'needs at least {_NEIGHBORS} rows of the rarest class, not '
            f'{minority}')
    overlapping = overlap is not None and overlap > _OVERLAP
    if name == 'smote_tomek' and not overlapping:
        return (
            f'is for classes that overlap, and only {overlap or 0.0:.0%} of '
            f'the rare rows are mostly near other classes')
    return None


def _skewed_share(continuous: pd.DataFrame, binaries: int) -> float:
    """Returns the share of the features whose distribution is not normal.

    Args:
        continuous: the numeric features that are not binary.
        binaries: number of binary features, which are never normal.

    Returns:
        The share of the features (continuous and binary) that are binary or
            materially skewed or heavy-tailed. 0 if there are none.

    """
    total = continuous.shape[1] + binaries
    if total == 0:
        return 0.0
    varied = continuous.loc[
        :, (continuous.count() > 3) & (continuous.std() > 0)]  # noqa: PLR2004
    skewed = (
        (varied.skew().abs() > _SKEW) | (varied.kurt().abs() > _KURTOSIS))
    return (int(skewed.sum()) + binaries) / total


def _splits(item: base.Dataset, folds: int) -> int:
    """Returns the number of folds to cross-validate with.

    Args:
        item: the dataset.
        folds: the number of folds asked for.

    Returns:
        `folds`, or fewer if the rarest class (for classification) or the
            training rows have fewer rows.

    """
    rows = _real_rows(item)
    most = len(rows)
    if item.task == 'classify':
        counts = item.y.loc[rows].value_counts()
        most = int(counts.min()) if len(counts) else 0
    return min(folds, most)


def _suits(name: str, found: Profile) -> str:
    """Returns why a model suits the data.

    Args:
        name: name of the model.
        found: the profile of the dataset.

    Returns:
        The reason.

    """
    traits = _traits_of(name)
    if traits is None:
        return 'was named in "choices"'
    if traits.counts == 'equidispersed':
        return (
            f'the label is counts that vary about as a Poisson model assumes '
            f'(dispersion {found.dispersion or 0.0:.2f})')
    if traits.counts == 'overdispersed':
        return (
            f'the label is counts that vary more than a Poisson model '
            f'assumes (dispersion {found.dispersion or 0.0:.2f})')
    if traits.counts == 'zero_inflated':
        return 'the label is counts with more zeros than a Poisson model expects'
    if traits.ordinal:
        return f'the {found.classes} classes are ordered'
    if traits.outliers:
        return 'the errors have outliers, which it weighs less'
    if traits.normal_errors:
        return 'the errors are about normal and the features are not collinear'
    if traits.independent and not traits.normal_features:
        return 'the features are not collinear'
    if traits.max_classes is not None or traits.min_classes:
        return f'the label has {found.classes} classes'
    if traits.missing and found.missing:
        return 'uses missing values directly'
    if traits.categories and found.categories:
        return 'uses categorical features directly'
    return 'its needs and assumptions are met'


def _traits_of(name: str) -> _Traits | None:
    """Returns what a model needs from the data and assumes about it.

    Args:
        name: name of a model.

    Returns:
        Its traits in `_MODELS` or `_OTHER_MODELS`, or `None` if it is in
            neither.

    """
    return _MODELS.get(name) or _OTHER_MODELS.get(name)


def _unscaled(continuous: pd.DataFrame) -> bool:
    """Returns whether numeric features are on very different scales.

    Args:
        continuous: the numeric features that are not binary.

    Returns:
        Whether the largest standard deviation is more than `_SCALES` times
            the smallest (of the features that vary).

    """
    spreads = continuous.std()
    spreads = spreads[spreads > 0]
    if len(spreads) < 2:  # noqa: PLR2004
        return False
    return bool(spreads.max() / spreads.min() > _SCALES)
