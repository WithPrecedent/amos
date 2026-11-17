"""Assumptions and needs of techniques, and how amos checks them.

A technique lists by name the `assumptions` it makes about the data (such as
"normal_errors", which a linear regression's standard errors rely on) and
the `needs` it has of the data (such as "no_missing_values"). A model that
assumes something can still be fitted when the data does not meet it, but
what it finds may be misleading. A model that needs something cannot be
fitted without it, or fits poorly. Each name is a subclass of `Assumption`
or `Need`, named in snake case as techniques are (`NormalErrors` is
"normal_errors"), so you can add your own by subclassing either one and
naming it in the `assumptions` or `needs` of your technique:

```py
class Stationarity(amos.conditions.Assumption):
    description = "the values do not drift over time"
```

Each condition names the pretest that checks it before a technique is
applied (`pretested_by`, such as "check_normality"), the audit that checks
it after a model is fitted (`audited_by`, such as "audit_residuals"), or
both. A condition without either is still listed (and documented), but it
cannot be tested from the data. The pretests are run when a technique with
`assumptions` or `needs` is applied (see `check_before`), and the dataset's
`assumption_violations` decides what amos does if one fails (see
`enforce`).

Contents:
    Verdict: the result of checking one assumption or need.
    Condition: base class for assumptions and needs.
    Assumption: base class for what a technique assumes about the data.
    Need: base class for what a technique needs from the data.
    borrow: returns the assumption or need with a name.
    catalog: returns every assumption and need, by name.
    conditions_of: returns the names of the assumptions and needs of a
        technique.
    name_of: returns the name of a technique, from an instance or a class.
    judge_before: checks the conditions of techniques with their pretests.
    judge_after: checks the conditions of a model with its audits.
    enforce: does what the dataset's `assumption_violations` says about the
        conditions that the data does not meet.
    check_before: checks a technique's conditions before it is applied.
    record: adds verdicts to a dataset's tables and history (as an audit
        does).
    tabulate: returns verdicts as a table.
    _CONDITIONS: every assumption and need, by name.
    _COLUMNS: the columns of a table of verdicts.

"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, ClassVar

import camina
import pandas as pd

from . import base

# Every assumption and need, by its name. Each subclass of `Assumption` and
# `Need` adds itself.
_CONDITIONS: dict[str, type[Condition]] = {}
# The columns of a table of verdicts, in order.
_COLUMNS: tuple[str, ...] = (
    'technique', 'condition', 'kind', 'check', 'passed', 'statistic',
    'p_value', 'threshold', 'detail')


@dataclasses.dataclass(frozen = True)
class Verdict:
    """The result of checking one assumption or need of a technique.

    Args:
        condition: name of the assumption or need.
        kind: "assumption" or "need".
        check: name of the pretest or audit that checked it.
        technique: name of the technique that makes the assumption or has
            the need, or `None` if it was checked for no technique in
            particular. Defaults to `None`.
        passed: whether the data meets it, or `None` if it could not be
            tested. Defaults to `None`.
        statistic: the statistic of the test, if it has one. Defaults to
            `None`.
        p_value: the p-value of the test, if it has one. Defaults to `None`.
        threshold: the value of the statistic beyond which the data does not
            meet it, if there is one. Defaults to `None`.
        detail: what was found, in words. For a technique that fails, it is
            phrased to follow the name of the technique (such as "assumes
            normal errors, but they are not ..."). Defaults to an empty
            `str`.
        hard: whether the technique cannot be applied without it (see
            `Need.hard`). Defaults to `False`.

    """

    condition: str
    kind: str
    check: str
    technique: str | None = None
    passed: bool | None = None
    statistic: float | None = None
    p_value: float | None = None
    threshold: float | None = None
    detail: str = ''
    hard: bool = False

    """ Properties """

    @property
    def violated(self) -> bool:
        """Returns whether a technique's condition is not met by the data.

        Returns:
            Whether the check failed for a technique. A check made for no
                technique in particular describes the data, so it is never
                a violation.

        """
        return self.passed is False and self.technique is not None

    """ Public Methods """

    def describe(self) -> str:
        """Returns the verdict as a sentence about its technique.

        Returns:
            The name of the technique followed by `detail`, or `detail`
                alone if there is no technique.

        """
        if self.technique is None:
            return self.detail
        return f'{self.technique!r} {self.detail}'

    def to_dict(self) -> dict[str, Any]:
        """Returns the verdict as plain Python values, for the history.

        Returns:
            The fields of the verdict, with floats rounded to 4 places.

        """
        return {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in dataclasses.asdict(self).items()}


@dataclasses.dataclass(frozen = True)
class Condition:
    """Base class for the assumptions and needs of techniques.

    A subclass is added to the conditions that amos knows under its `name`
    (the snake case name of the class, unless it sets one), so that a
    technique can name it in its `assumptions` or `needs`. Pass
    `register = False` in the class statement to leave a base class out.

    A condition that can be tested before a technique is applied names its
    pretest in `pretested_by` and overrides `pretest`. One that can only be
    tested after a model is fitted (from its errors, for example) names its
    audit in `audited_by` and overrides `audit`. One that cannot be tested
    from the data names neither.

    """

    # "assumption" or "need".
    kind: ClassVar[str] = ''
    # The name that techniques use for the condition.
    name: ClassVar[str] = ''
    # What the condition is, in a sentence that completes "The technique
    # assumes (or needs) that ...".
    description: ClassVar[str] = ''
    # Name of the pretest that checks it before a technique is applied, or
    # an empty `str` if none does.
    pretested_by: ClassVar[str] = ''
    # Name of the audit that checks it after a model is fitted, or an empty
    # `str` if none does.
    audited_by: ClassVar[str] = ''
    # What a violation harms: "inference" (standard errors, p-values, and
    # intervals), "prediction", or "both".
    affects: ClassVar[str] = 'both'
    # Whether the technique cannot be applied without it.
    hard: ClassVar[bool] = False

    """ Initialization Methods """

    def __init_subclass__(
        cls,
        *,
        register: bool = True,
        **kwargs: Any) -> None:
        """Adds the subclass to the conditions that amos knows.

        Args:
            register: whether to add it. Defaults to `True`.
            **kwargs: passed to the `__init_subclass__` of the parent class.

        """
        super().__init_subclass__(**kwargs)
        if not register:
            return
        cls.name = cls.__dict__.get('name') or str(
            camina.snakify(cls.__name__))
        _CONDITIONS[cls.name] = cls

    """ Public Methods """

    def audit(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any) -> Verdict | None:  # noqa: ARG002
        """Checks the condition from a fitted model.

        Subclasses that name an audit in `audited_by` override this.

        Args:
            item: the dataset, with a fitted model and its predictions.
            technique: the model's technique (a class or an instance), or
                `None` to check it for no technique in particular.

        Returns:
            The verdict, or `None` if the condition has no audit.

        """
        return None

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,  # noqa: ARG002
        found: Any,  # noqa: ARG002
        cache: dict[str, Any]) -> Verdict | None:  # noqa: ARG002
        """Checks the condition before a technique is applied.

        Subclasses that name a pretest in `pretested_by` override this.

        Args:
            item: the dataset that the technique is about to work on.
            technique: the technique (a class or an instance), or `None` to
                check it for no technique in particular.
            found: the profile of the dataset's real training rows (see
                `auto.profile`).
            cache: results that are slow to find, which every condition
                checked at the same time shares, by name.

        Returns:
            The verdict, or `None` if the condition has no pretest.

        """
        return None

    """ Private Methods """

    def _verdict(
        self,
        technique: Any,
        check: str,
        passed: bool | None,
        detail: str,
        **values: float | None) -> Verdict:
        """Returns a verdict about this condition.

        Args:
            technique: the technique (a class or an instance), or `None`.
            check: name of the pretest or audit.
            passed: whether the data meets the condition, or `None` if it
                could not be tested.
            detail: what was found, in words.
            **values: the "statistic", "p_value", and "threshold", if the
                test has them.

        Returns:
            The verdict.

        """
        return Verdict(
            condition = self.name,
            kind = self.kind,
            check = check,
            technique = None if technique is None else name_of(technique),
            passed = passed,
            detail = detail,
            hard = self.hard,
            **{k: None if v is None else float(v) for k, v in values.items()})


@dataclasses.dataclass(frozen = True)
class Assumption(Condition, register = False):
    """Base class for what a technique assumes about the data.

    A technique can be applied when the data does not meet one of its
    assumptions, but what it finds may be misleading (such as standard
    errors that are too small). Name an assumption in a technique's
    `assumptions`.

    """

    kind: ClassVar[str] = 'assumption'


@dataclasses.dataclass(frozen = True)
class Need(Condition, register = False):
    """Base class for what a technique needs from the data.

    A need that is `hard` (most are) is one that the technique cannot be
    applied without, such as a model that cannot use missing values, so the
    technique is never applied when the data does not meet it, whatever the
    dataset's `assumption_violations`: an `AssumptionError` is raised, or
    the branch of an experiment is left out. A need that is not hard (such
    as the most rows that a model is fast enough for) is treated as an
    assumption. Name a need in a technique's `needs`.

    """

    kind: ClassVar[str] = 'need'
    hard: ClassVar[bool] = True


""" Public Functions """


def borrow(name: str) -> type[Condition]:
    """Returns the assumption or need with a name.

    Args:
        name: the name of the condition (such as "normal_errors").

    Raises:
        KeyError: if amos knows no condition with that name.

    Returns:
        The subclass of `Assumption` or `Need`.

    """
    try:
        return _CONDITIONS[name]
    except KeyError:
        message = (
            f'{name!r} is not an assumption or need that amos knows: '
            f'subclass amos.conditions.Assumption or Need to add it')
        raise KeyError(message) from None


def catalog() -> dict[str, type[Condition]]:
    """Returns every assumption and need, by name.

    Returns:
        A copy of the conditions that amos knows, sorted by name.

    """
    return dict(sorted(_CONDITIONS.items()))


def check_before(
    item: base.Dataset,
    technique: Any,
    parameters: Mapping[str, Any] | None = None) -> list[str]:
    """Checks a technique's assumptions and needs before it is applied.

    The pretests of the technique's conditions are run on the real
    training rows. Their verdicts are added to the dataset's "assumptions"
    table, and then the dataset's `assumption_violations` decides what is
    done about those that fail (see `enforce`).

    Columns that the technique is given apart from the features (those that
    its `column_parameters`, such as the "groups" of `gee`, name in
    `parameters`) are not features for it, so they are left out of the
    features that are checked.

    Args:
        item: the dataset that the technique is about to work on.
        technique: the technique (an instance).
        parameters: the parameters that it is applied with. Defaults to
            `None`.

    Returns:
        The violations (as `Verdict.describe` writes them), which the
            technique adds to its entry in the dataset's `history` once it
            is applied.

    """
    view = _without_extra_columns(item, technique, parameters or {})
    verdicts = judge_before(view, [technique])
    if not verdicts:
        return []
    _add_rows(item, verdicts, 'assumptions')
    enforce(item, verdicts)
    return [verdict.describe() for verdict in verdicts if verdict.violated]


def conditions_of(technique: Any) -> list[str]:
    """Returns the names of the assumptions and needs of a technique.

    Args:
        technique: a technique (a class or an instance).

    Returns:
        The names in its `assumptions` and then its `needs`, without
            duplicates. A technique without them has none.

    """
    names: list[str] = []
    for name in (
        *getattr(technique, 'assumptions', ()),
        *getattr(technique, 'needs', ())):
        if name not in names:
            names.append(str(name))
    return names


def enforce(item: base.Dataset, verdicts: Sequence[Verdict]) -> None:
    """Does what the dataset says to do about the conditions that fail.

    A need that is `hard` is never let pass: in a branch of an experiment,
    the branch is left out, and elsewhere an error is raised. For the other
    conditions, the dataset's `assumption_violations` decides: "raise" an
    error, "warn" (with an `AmosWarning`), "ignore" them (they are still in
    the dataset's tables and history), or "skip" the branch of an experiment
    (and warn outside of one).

    Args:
        item: the dataset.
        verdicts: the verdicts of the checks.

    Raises:
        AssumptionError: if a hard need fails outside of an experiment, or
            a condition fails and the policy is "raise".
        SkippedBranchError: if a hard need fails in a branch of an
            experiment, or a condition fails there and the policy is
            "skip".

    """
    failed = [verdict for verdict in verdicts if verdict.violated]
    if not failed:
        return
    experiment = base._IN_EXPERIMENT.get()
    hard = [verdict for verdict in failed if verdict.hard]
    if hard:
        message = '; '.join(verdict.describe() for verdict in hard)
        if experiment:
            raise base.SkippedBranchError(message)
        raise base.AssumptionError(message)
    policy = item.assumption_violations
    message = '; '.join(verdict.describe() for verdict in failed)
    if policy == 'raise':
        raise base.AssumptionError(message)
    if policy == 'skip' and experiment:
        raise base.SkippedBranchError(message)
    if policy == 'ignore':
        return
    for verdict in failed:
        warnings.warn(
            verdict.describe(), category = base.AmosWarning, stacklevel = 5)


def judge_after(
    item: base.Dataset,
    technique: Any,
    *,
    check: str | None = None) -> list[Verdict]:
    """Checks the conditions of a fitted model with their audits.

    Args:
        item: the dataset, with a fitted model and its predictions.
        technique: the model's technique (a class or an instance), or
            `None` if it is not known.
        check: name of an audit, to check only the conditions that it
            audits (each of them, whether or not the model makes it), or
            `None` to check every condition of the model that has an audit.

    Returns:
        A verdict for each condition checked. A condition that the model
            does not make is checked for no technique in particular, so it
            is never a violation.

    """
    made = conditions_of(technique) if technique is not None else []
    if check is None:
        names = made
    else:
        names = [
            name for name, kind in _CONDITIONS.items()
            if kind.audited_by == check]
    verdicts = []
    for name in names:
        condition = borrow(name)()
        if not condition.audited_by:
            continue
        owner = technique if name in made else None
        verdict = _safely(
            condition, owner, condition.audited_by,
            functools.partial(condition.audit, item, owner))
        if verdict is not None:
            verdicts.append(verdict)
    return verdicts


def judge_before(
    item: base.Dataset,
    techniques: Iterable[Any],
    *,
    check: str | None = None,
    found: Any = None,
    cache: dict[str, Any] | None = None) -> list[Verdict]:
    """Checks the conditions of techniques with their pretests.

    The real training rows are profiled once (see `auto.profile`), and
    every pretest uses the profile.

    Args:
        item: the dataset that the techniques are about to work on.
        techniques: the techniques (classes or instances) whose conditions
            to check. If there are none, every condition that has a pretest
            (or that `check` checks) is checked for no technique in
            particular, which describes the data.
        check: name of a pretest, to check only the conditions that it
            checks, or `None` to check every condition that has a pretest.
        found: the profile of the dataset, if it was already made. Defaults
            to `None`, which makes it.
        cache: results that are slow to find, which every condition checked
            shares, if some were already found. Defaults to `None`.

    Returns:
        A verdict for each condition checked, in the order of the
            techniques and of each one's conditions.

    """
    pairs: list[tuple[Any, str]] = []
    listed = list(techniques)
    if listed:
        pairs = [
            (technique, name)
            for technique in listed for name in conditions_of(technique)]
    else:
        pairs = [(None, name) for name in _CONDITIONS]
    chosen = []
    for technique, name in pairs:
        condition = borrow(name)()
        if not condition.pretested_by:
            continue
        if check is not None and condition.pretested_by != check:
            continue
        chosen.append((technique, condition))
    if not chosen:
        return []
    if found is None:
        # `auto` imports this module.
        from . import auto  # noqa: PLC0415
        found = auto.profile(item)
    cache = {} if cache is None else cache
    verdicts = []
    for technique, condition in chosen:
        verdict = _safely(
            condition, technique, condition.pretested_by,
            functools.partial(condition.pretest, item, technique, found, cache))
        if verdict is not None:
            verdicts.append(verdict)
    return verdicts


def name_of(technique: Any) -> str:
    """Returns the name of a technique.

    Args:
        technique: a technique (a class or an instance) or its name.

    Returns:
        The name of an instance (its `name`) or of a class (the snake case
            name of the class, as the library knows it).

    """
    if isinstance(technique, str):
        return technique
    if isinstance(technique, type):
        return str(camina.snakify(technique.__name__))
    return str(getattr(technique, 'name', '') or camina.snakify(
        type(technique).__name__))


def record(
    item: base.Dataset,
    verdicts: Sequence[Verdict],
    *,
    technique: str,
    role: str,
    table: str = 'assumptions') -> None:
    """Adds verdicts to the tables and history of a dataset.

    Args:
        item: the dataset.
        verdicts: the verdicts.
        technique: the name to record them under in `history`: the
            technique that was checked, or the pretest or audit that checked
            them.
        role: "pretests" or "audits", for the entry in `history`.
        table: the name of the table to add them to. Defaults to
            "assumptions", the table of every check that amos made before
            applying a technique and after fitting a model.

    """
    _add_rows(item, verdicts, table)
    violations = [
        verdict.describe() for verdict in verdicts if verdict.violated]
    details: dict[str, Any] = {'violations': violations} if violations else {}
    item.record(
        technique,
        role = role,
        verdicts = [verdict.to_dict() for verdict in verdicts],
        policy = item.assumption_violations,
        **details)


def tabulate(verdicts: Sequence[Verdict]) -> pd.DataFrame:
    """Returns verdicts as a table.

    Args:
        verdicts: the verdicts.

    Returns:
        A row for each verdict, with its technique, condition, kind, check,
            whether it passed, its statistic, p-value, and threshold, and
            what was found.

    """
    rows = [
        {column: getattr(verdict, column) for column in _COLUMNS}
        for verdict in verdicts]
    return pd.DataFrame(rows, columns = list(_COLUMNS))


""" Private Functions """


def _add_rows(
    item: base.Dataset,
    verdicts: Sequence[Verdict],
    table: str) -> None:
    """Adds verdicts to a table of a dataset, after any rows it has.

    Args:
        item: the dataset.
        verdicts: the verdicts.
        table: the name of the table.

    """
    rows = tabulate(verdicts)
    existing = item.tables.get(table)
    item.tables[table] = (
        rows if existing is None
        else pd.concat([existing, rows], ignore_index = True))


def _without_extra_columns(
    item: base.Dataset,
    technique: Any,
    parameters: Mapping[str, Any]) -> base.Dataset:
    """Returns `item` without the technique's extra columns as features.

    Args:
        item: the dataset.
        technique: the technique, whose `column_parameters` (if it has
            any) name the parameters that give it columns apart from the
            features.
        parameters: the parameters that the technique is applied with.

    Returns:
        `item` itself if the technique is given no such columns, or else a
            shallow copy of it (with the same data) in which they are groups,
            which are not features.

    """
    extras = [
        str(name)
        for parameter in getattr(technique, 'column_parameters', ())
        for name in camina.listify(parameters.get(parameter))
        if name in item.data.columns and name not in item.groups
        and name not in (item.label, item.order)]
    if not extras:
        return item
    return dataclasses.replace(item, groups = [*item.groups, *extras])


def _safely(
    condition: Condition,
    technique: Any,
    check: str,
    test: Callable[[], Verdict | None]) -> Verdict | None:
    """Returns the verdict of a test, or an untested one if it fails.

    Args:
        condition: the condition being tested.
        technique: the technique (a class or an instance), or `None`.
        check: name of the pretest or audit.
        test: the test.

    Returns:
        The verdict of the test, or a verdict that it could not be tested,
            with the error, if the test raised one.

    """
    try:
        return test()
    # A test that fails (on data it does not suit, for example) leaves the
    # condition untested rather than stopping the workflow.
    except Exception as error:  # noqa: BLE001
        return condition._verdict(
            technique, check, None, f'could not be tested: {error}')
