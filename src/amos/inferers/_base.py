"""The base class of inferers, and what they share.

Contents:
    Inferer: base class for techniques that infer causal effects.
    Inference: what an inferer found, which `Inferer.apply` adds to the dataset.

"""

from __future__ import annotations

import abc
import contextlib
import dataclasses
import importlib
import io
import logging
import warnings
from collections.abc import Callable, Generator, Hashable, Mapping, Sequence
from typing import Any, ClassVar

import chrisjen
import numpy as np
import pandas as pd

from .. import base, models, utilities

# Packages whose messages (printed or logged) the inferers keep quiet.
_LOGGERS: tuple[str, ...] = ('causalml', 'dowhy', 'tigramite')


# The number of made-up rows that a learner is fitted to on the GPU, to check
# that it can be before an inferer fits it.
_PROBE_ROWS: int = 64


# The value of the standard normal distribution below which 97.5% of it
# falls, which turns a 95% confidence interval into a standard error.
_Z_95: float = 1.959963984540054


@dataclasses.dataclass
class Inferer(base.Operation, base.Visualizer, abc.ABC):
    """Base class for techniques that infer causal effects.

    Most inferers share these parameters (set in the settings or passed to
    `apply`); each technique lists the others that it takes:

    | Parameter | Meaning |
    | --- | --- |
    | `treatment` | Name of the column with the treatment. Required, except by tigramite's methods of causal discovery. |
    | `outcome_model` | Name of an `amos` model for the label (by default, `random_forest`). DoubleML and causalml. |
    | `treatment_model` | Name of an `amos` model for the treatment, and for any instrument or selection (by default, the outcome model). DoubleML and causalml. |
    | `n_folds` | Number of folds for cross-fitting (5 by default). DoubleML and causalml. |

    If the dataset's `use_gpu` is `True` and an NVIDIA GPU is found, the
    models of DoubleML and causalml that can be fitted on a GPU (`catboost`,
    `lightgbm`, `tabpfn`, and `xgboost`) are fitted on the first one. Each is
    first fitted to a few rows on the GPU, and if that fails, it warns and
    the inferer fits it on the CPU.

    An inferer uses only the real rows: while it is applied, the rows that a
    sampler added to the training rows (the dataset's `synthetic` rows) are
    left out of the data, and the data is then restored.

    Pass `visualize = True` to `apply` (or set "visualize_critics" in the
    "general" section of the settings) to draw the estimates as a figure in
    the dataset's `figures`, under the technique's name: a `coefficient_plot`
    of the effects, or a `causal_graph` of the links that tigramite's
    methods of causal discovery find (see `base.Visualizer`).

    The techniques of each package are a genre within this one: `Doubleml`,
    `Dowhy`, `Causalml`, and `Tigramite`, which write `infer`. To add a
    technique of one of those packages, subclass its genre; to add one of
    another kind, subclass `Inferer` and write `infer`, which returns an
    `Inference` that `apply` adds to the dataset.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, in which case it is the snake case name of the
            class.
        contents: the estimator that the technique wraps, as an import path.
            Defaults to `None`.
        parameters: keyword arguments for `apply`. Defaults to an empty
            `dict`.

    """

    stage_setting: ClassVar[str] = 'visualize_critics'
    visual: ClassVar[str] = 'coefficient_plot'
    model_specific_setting: ClassVar[str] = 'inferers'

    """ Required Methods """

    @abc.abstractmethod
    def infer(self, item: base.Dataset, **kwargs: Any) -> Inference:
        """Returns what the technique infers from `item`.

        Args:
            item: the dataset, without the rows that a sampler made up.
            **kwargs: parameters for the inference.

        Returns:
            The estimates (or links), the fitted estimator, any other tables,
                and what to record in `history`.

        """

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        *,
        visualize: bool = False,
        **kwargs: Any) -> base.Dataset:
        """Adds what `infer` finds to `item`.

        The table of estimates (or links) is added to the dataset's `tables`
        under the technique's name, and any other tables under the
        technique's name and their own (such as "{name}_effects"). The
        fitted estimator is kept in `fitted`, and the entry in `history`
        has the tool, the parameters, and (for an effect) the first
        estimate as "effect".

        Args:
            item: the dataset.
            visualize: whether to draw the table as a figure (with the plot
                named by `visual`), which is added to the figures of `item`
                under the technique's name. The dataset's
                `visualize_critics` (the setting of the "general" section)
                replaces it, if it is set. Defaults to `False`.
            **kwargs: parameters for `infer`.

        Returns:
            The dataset, with the new tables (and figure).

        """
        found = self.infer(item, **kwargs)
        item.tables[self.name] = found.table
        for suffix, table in found.tables.items():
            item.tables[f'{self.name}_{suffix}'] = table
        item.fitted[self.name] = found.fitted
        details = {
            'tool': utilities.describe_tool(type(found.fitted)),
            **found.details}
        if found.effect:
            details['effect'] = (
                float(found.table['coefficient'].iloc[0])
                if len(found.table) else None)
        item.record(self.name, **details)
        self._visualize(item, visualize)
        return item

    """ Private Methods """

    def _call_apply(
        self,
        method: Callable[..., Any],
        item: Any,
        /,
        **kwargs: Any) -> base.Dataset:
        """Calls `apply` with the rows that a sampler added left out.

        Args:
            method: the `apply` method, before it was wrapped.
            item: a `Dataset` or anything that `Dataset.create` accepts.
            **kwargs: parameters that take precedence over `parameters`.

        Returns:
            The changed `Dataset`, with all of its rows.

        """
        dataset = base.Dataset.create(item)
        with _dropping_synthetic_rows(dataset):
            return super()._call_apply(method, dataset, **kwargs)


@dataclasses.dataclass(frozen = True)
class Inference:
    """What an inferer found, which `Inferer.apply` adds to the dataset.

    Args:
        table: the estimates (one row for each effect, with a "coefficient"
            column), or the links that causal discovery found.
        fitted: the fitted estimator (or the results of causal discovery),
            which is kept in the dataset's `fitted`.
        tables: other tables, by the name that follows the technique's in
            the dataset's `tables` (such as "effects" or "refutations").
            Defaults to an empty `dict`.
        details: what to record in `history`, such as the parameters. Unless
            it has a "tool", the type of `fitted` is recorded as the tool.
            Defaults to an empty `dict`.
        effect: whether `table` holds effects, so that the first estimate
            is recorded as the "effect". Defaults to `True`.

    """

    table: pd.DataFrame
    fitted: Any
    tables: Mapping[str, pd.DataFrame] = dataclasses.field(
        default_factory = dict)
    details: Mapping[str, Any] = dataclasses.field(default_factory = dict)
    effect: bool = True


@dataclasses.dataclass(frozen = True)
class _Learners:
    """Builds the models that an inferer fits, named as `amos` models.

    Args:
        outcome: name of the model of the label.
        treatment: name of the model of the treatment (and of any instrument
            or selection).
        seed: seed for the models.
        on_gpu: names of the models that are fitted on the GPU. Defaults to
            an empty `frozenset`, which fits every model on the CPU.

    """

    outcome: str
    treatment: str
    seed: int | None
    on_gpu: frozenset[str] = frozenset()

    @classmethod
    def create(
        cls,
        item: base.Dataset,
        outcome: str,
        treatment: str,
        inferer: str) -> _Learners:
        """Returns the learners of an inferer, on the GPU if they can be.

        A model is fitted on the GPU if the dataset's `use_gpu` is `True`,
        the model can be fitted on a GPU (see `Model._device_parameters`), an
        NVIDIA GPU is found, and the model is fitted on it to a few rows
        without an error (see `_fits_on_gpu`).

        Args:
            item: the dataset.
            outcome: name of the model of the label.
            treatment: name of the model of the treatment.
            inferer: name of the inferer, for warnings.

        Returns:
            The learners. If no GPU is found, it warns that the models are
                fitted on the CPU.

        """
        capable = []
        if item.use_gpu:
            capable = [
                n for n in dict.fromkeys([outcome, treatment])
                if _technique(n)._device_parameters(gpu = True) is not None]
        if not capable:
            return cls(outcome, treatment, item.seed)
        if not utilities.gpus():
            warnings.warn(
                'use_gpu is true, but no NVIDIA GPU was found, so models are '
                'fitted on the CPU',
                category = base.AmosWarning,
                stacklevel = 3)
            return cls(outcome, treatment, item.seed)
        # Whether a package can use the GPU does not depend on the task, so
        # each model is checked once, with the task that its role usually
        # has (the outcome's, if one model has both roles), so that tabpfn,
        # for example, does not download the weights of a model it will not
        # use.
        tasks = {treatment: 'classify', outcome: 'regress'}
        on_gpu = set()
        for name in capable:
            if _fits_on_gpu(name, tasks[name], item.seed, inferer):
                on_gpu.add(name)
        return cls(outcome, treatment, item.seed, frozenset(on_gpu))

    def of_outcome(self, task: str) -> Any:
        """Returns an unfitted model of the label.

        Args:
            task: "classify" or "regress".

        Returns:
            The model.

        """
        return _learner(
            self.outcome, task, self.seed, gpu = self.outcome in self.on_gpu)

    def of_treatment(self, task: str) -> Any:
        """Returns an unfitted model of the treatment.

        Args:
            task: "classify" or "regress".

        Returns:
            The model.

        """
        return _learner(
            self.treatment, task, self.seed,
            gpu = self.treatment in self.on_gpu)


@dataclasses.dataclass(frozen = True)
class _Setup:
    """What the estimator of an inferer is built from.

    Args:
        item: the dataset.
        frame: the label, the treatment, any other columns with roles in the
            model, and the controls, as numbers.
        label: name of the label.
        treatment: name of the treatment.
        controls: names of the controls (the other features).
        learners: builds the models of the label and treatment.
        n_folds: number of folds for cross-fitting.
        options: the technique's other parameters.

    """

    item: base.Dataset
    frame: pd.DataFrame
    label: str
    treatment: str
    controls: list[str]
    learners: _Learners
    n_folds: int
    options: dict[str, Any]


""" Private Functions """


def _estimates(
    index: Sequence[Hashable],
    coefficients: Sequence[float] | np.ndarray,
    errors: Sequence[float] | np.ndarray,
    *,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    name: str | None = None) -> pd.DataFrame:
    """Returns a table of estimates, with their test statistics and p-values.

    Wraps:
        [`scipy.stats.norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
        from SciPy, for the p-values.

    Args:
        index: the label of each estimate.
        coefficients: the estimates.
        errors: their standard errors.
        lower: the lower ends of their 95% confidence intervals.
        upper: the upper ends of their 95% confidence intervals.
        name: name of the index. Defaults to `None`.

    Returns:
        The table, with the columns of `_COLUMNS`. The statistic is the
            estimate divided by its standard error, and the p-value is
            two-sided, from the normal distribution. Both are missing if the
            standard error is 0 or missing.

    """
    stats = importlib.import_module('scipy.stats')
    coefficients = np.asarray(coefficients, dtype = float)
    errors = np.asarray(errors, dtype = float)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        statistic = np.where(errors > 0, coefficients / errors, np.nan)
    return pd.DataFrame(
        {
            'coefficient': coefficients,
            'standard_error': errors,
            'statistic': statistic,
            'p_value': 2 * stats.norm.sf(np.abs(statistic)),
            'ci_lower': np.asarray(lower, dtype = float),
            'ci_upper': np.asarray(upper, dtype = float)},
        index = pd.Index(list(index), name = name))


def _fits_on_gpu(name: str, task: str, seed: int | None, inferer: str) -> bool:
    """Returns whether a learner can be fitted on the GPU, by fitting it.

    Args:
        name: name of an `amos` model that can be fitted on a GPU.
        task: the task to check it with, "classify" or "regress" (or the
            other one, if the model cannot do it).
        seed: seed for the model.
        inferer: name of the inferer, for the warning.

    Returns:
        Whether the learner was fitted on the GPU and made predictions. If
            not, it warns that the inferer fits it on the CPU.

    """
    technique = _technique(name)
    if technique.contents is None and task not in technique.tools:
        task = 'regress' if task == 'classify' else 'classify'
    rows = np.arange(_PROBE_ROWS)
    x = np.column_stack([rows / _PROBE_ROWS, rows % 2])
    y = rows % 2 if task == 'classify' else rows / _PROBE_ROWS
    # DoubleML and causalml fit the learners themselves, many times over, so
    # unlike a model (see `Model.apply`), a learner that fails on the GPU
    # cannot be fitted again on the CPU partway through an inferer. Instead,
    # it is fitted once to a few made-up rows before the inferer starts (and
    # makes predictions, which is when tabpfn first runs on the GPU), and
    # anything that stops it means that the inferer fits it on the CPU. That
    # catches what usually keeps a package off the GPU (being built without
    # GPU support, or a GPU too old for it), but not running out of memory on
    # the full data, whose error is raised.
    try:
        with _quiet():
            learner = _learner(name, task, seed, gpu = True)
            learner.fit(x, y)
            learner.predict(x)
    except Exception as error:  # noqa: BLE001
        warnings.warn(
            f'{name!r} could not be fitted on the GPU, so {inferer!r} fits it '
            f'on the CPU instead: {error}',
            category = base.AmosWarning,
            stacklevel = 4)
        return False
    return True


def _learner(
    name: str,
    task: str,
    seed: int | None,
    *,
    gpu: bool = False) -> Any:
    """Returns an unfitted model, chosen by the name of an `amos` model.

    Args:
        name: name of an `amos` model (such as "random_forest").
        task: "classify" or "regress".
        seed: seed for the model, if it takes one.
        gpu: whether to build the model for the GPU, if it can be fitted on
            one. Defaults to `False`, which builds it for the CPU.

    Raises:
        TypeError: if `name` is not the name of a model.
        ValueError: if the model cannot do `task`.

    Returns:
        The model, built with the `amos` model's default parameters and the
            parameters of its device (see `Model._device_parameters`).

    """
    technique = _technique(name)
    tool = technique.contents or technique.tools.get(task)
    if tool is None:
        message = f'the model {name!r} cannot {task}'
        raise ValueError(message)
    built = utilities.import_tool(tool)
    # A model's defaults (such as the kind of a statsmodels model) can depend
    # on the task, so they are chosen for an empty dataset of the task.
    parameters = technique._prepare(
        base.Dataset(task = task),
        {**technique._keywords(), 'random_state': seed})
    # As in `Operation._make_tool`, the device's parameters are passed even if
    # the tool only takes them through `**kwargs` (as lightgbm's models do),
    # and the model's own parameters take precedence.
    device = technique._device_parameters(gpu = gpu) or {}
    return built(**{
        **device, **utilities.accepted_parameters(built, parameters)})


def _levels(values: pd.Series) -> list[Any]:
    """Returns the distinct values of a column, sorted if they can be.

    Args:
        values: a column.

    Returns:
        Its distinct values that are not missing.

    """
    levels = list(values.dropna().unique().tolist())
    try:
        return sorted(levels)
    except TypeError:
        return levels


def _listed(item: Any) -> list[Any]:
    """Returns a setting as a `list`.

    Args:
        item: `None`, a `str` (which may list items separated by commas), or
            a sequence.

    Returns:
        The items (an empty `list` for `None`).

    """
    if item is None:
        return []
    if isinstance(item, str):
        return [i.strip() for i in item.split(',') if i.strip()]
    if isinstance(item, Sequence):
        return list(item)
    return [item]


def _numbers(values: pd.Series) -> pd.Series:
    """Returns `values` as numbers.

    Args:
        values: a column.

    Returns:
        The column as floats. A column with two values that are not numbers
            becomes 1 for the last value (in sorted order) and 0 for the
            other.

    """
    if pd.api.types.is_numeric_dtype(values.dtype):
        return values.astype(float)
    levels = sorted(values.dropna().unique().tolist())
    numbers: pd.Series = (values == levels[-1]).astype(float)
    return numbers


def _prepare(
    item: base.Dataset,
    columns: Sequence[str],
    technique: str) -> tuple[pd.DataFrame, str, list[str]]:
    """Returns the data of an inferer: the label, `columns`, and the controls.

    The controls are the features that are not among `columns`.

    Args:
        item: the dataset. It must have a label.
        columns: names of the treatment and of any other columns with roles
            in the model (such as an instrument).
        technique: name of the technique, for messages.

    Raises:
        KeyError: if a column is not in the data.
        ValueError: if a control is not a number or boolean.

    Returns:
        The data (as numbers, see `_numbers`), the name of the label, and the
            names of the controls.

    """
    label = item.require_label()
    missing = [c for c in columns if c not in item.data.columns]
    if missing:
        message = f'the columns {missing} are not in the data'
        raise KeyError(message)
    controls = [c for c in item.features if c not in columns]
    unusable = [
        c for c in controls
        if base._kind_of(item.data[c]) not in {'numerics', 'booleans'}]
    if unusable:
        message = (
            f'{technique!r} needs numbers, but {unusable} are not: encode or '
            f'remove them first'
        )
        raise ValueError(message)
    frame = item.data[controls].astype(float)
    frame[label] = _numbers(item.y)
    for column in columns:
        frame[column] = _numbers(item.data[column])
    return frame, label, controls


@contextlib.contextmanager
def _quiet() -> Generator[None, None, None]:
    """Keeps the inferers' packages from printing, logging, and warning.

    causalml prints a message when it is imported, and the packages log
    their progress and warn about the data.

    Yields:
        None: control, to the code in the context.

    """
    loggers = [logging.getLogger(n) for n in _LOGGERS]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            warnings.catch_warnings()):
            warnings.simplefilter('ignore')
            yield
    finally:
        for logger, level in zip(loggers, levels, strict = True):
            logger.setLevel(level)


def _require_treatment(treatment: str | None, technique: str) -> str:
    """Returns the name of the treatment, which is required.

    Args:
        treatment: name of the treatment, or `None`.
        technique: name of the technique, for the message.

    Raises:
        ValueError: if `treatment` is `None`.

    Returns:
        The name of the treatment.

    """
    if treatment is None:
        message = f'{technique!r} needs the name of a "treatment" column'
        raise ValueError(message)
    return treatment


def _require_values(values: pd.Series, count: int, technique: str) -> None:
    """Raises an error if a column does not have `count` values.

    Args:
        values: a column.
        count: the number of values that it must have.
        technique: name of the technique, for the message.

    Raises:
        ValueError: if `values` does not have `count` distinct values.

    """
    found = values.nunique(dropna = True)
    if found != count:
        message = (
            f'{technique!r} needs {values.name!r} to have {count} values, '
            f'but it has {found}'
        )
        raise ValueError(message)


@contextlib.contextmanager
def _dropping_synthetic_rows(item: base.Dataset) -> Generator[None, None, None]:
    """Leaves the rows that a sampler added out of the data in the context.

    A sampler such as `random_over` or `smote` adds copies of training rows
    or made-up rows (the dataset's `synthetic` rows) so that a model learns
    as much from each class. An inferer that used them would count some
    rows more than once, which changes its estimates and makes its standard
    errors too small.

    Args:
        item: the dataset. Its `data` is put back as it was afterwards.

    Yields:
        None: control, to the code in the context.

    """
    if not len(item.synthetic):
        yield
        return
    data = item.data
    item.data = data.drop(index = item.synthetic)
    try:
        yield
    finally:
        item.data = data


@contextlib.contextmanager
def _seeded(seed: int | None) -> Generator[None, None, None]:
    """Seeds `numpy`'s global random numbers in the context.

    DoubleML (for inner folds), DoWhy (for bootstrapping), and causalml draw
    from `numpy`'s global random numbers, which are put back as they were
    afterwards.

    Args:
        seed: the seed.

    Yields:
        None: control, to the code in the context.

    """
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(seed)  # noqa: NPY002
    try:
        yield
    finally:
        np.random.set_state(state)  # noqa: NPY002


def _task(values: pd.Series) -> str:
    """Returns the task of a model of a column.

    Args:
        values: a column.

    Returns:
        "classify" if the column has two values, and otherwise "regress".

    """
    two = values.nunique(dropna = True) == 2  # noqa: PLR2004
    return 'classify' if two else 'regress'


def _technique(name: str) -> models.Model:
    """Returns the `amos` model named `name`, with its default parameters.

    Args:
        name: name of an `amos` model (such as "random_forest").

    Raises:
        TypeError: if `name` is not the name of a model.

    Returns:
        The model.

    """
    technique = chrisjen.library.borrow(name, genre = 'model')()
    if not isinstance(technique, models.Model):
        message = f'{name!r} is not a model'
        raise TypeError(message)
    return technique
