"""Base classes for amos.

Contents:
    Dataset: data and everything learned from it in a workflow.
    Branch: one branch of an experiment and its result.
    Operation: base class for amos techniques, which work on a `Dataset`.

"""

from __future__ import annotations

import abc
import dataclasses
import pathlib
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from typing import Any, TypeAlias

import camina
import chrisjen
import numpy as np
import pandas as pd

from . import options, utilities

# Parameters of a technique, as `chrisjen` stores them.
GenericDict: TypeAlias = MutableMapping[Hashable, Any]
# Functions that load each kind of file, by its extension, for
# `Dataset.create`.
_READERS: dict[str, Callable[..., pd.DataFrame]] = {
    '.csv': pd.read_csv,
    '.dta': pd.read_stata,
    '.feather': pd.read_feather,
    '.json': pd.read_json,
    '.parquet': pd.read_parquet,
    '.pickle': pd.read_pickle,
    '.pkl': pd.read_pickle,
    '.sav': pd.read_spss,
    '.tsv': lambda path: pd.read_csv(path, sep = '\t'),
    '.xls': pd.read_excel,
    '.xlsx': pd.read_excel}
# Tasks that a `Dataset` can have.
_TASKS: tuple[str, ...] = ('classify', 'regress')


# `eq` is `False` so that comparing datasets does not compare their data
# frames, which `pandas` does not allow.
@dataclasses.dataclass(eq = False)
class Dataset:
    """Data and everything learned from it in a workflow.

    A `Dataset` is the item that an `amos` workflow works on. It holds the
    data in a single `pandas.DataFrame`, the name of the column that models
    predict (the `label`), and the rows in the training and test sets. As a
    workflow runs, each technique adds what it learns: the fitted `model`, its
    `predictions`, `metrics`, `tables`, `figures`, and a record of every step
    in `history`.

    The data stays in one `DataFrame` even after it is split. Transformers
    learn from the training rows and then change every row, so nothing is
    learned from the test rows, and every row keeps its original label in the
    index.

    Args:
        data: the data, with one row per observation and one column per
            variable. Defaults to an empty `DataFrame`.
        label: name of the column with the outcome (the dependent variable)
            that models predict. Defaults to `None`.
        task: "classify" or "regress". Defaults to `None`, in which case it is
            inferred from `label` (see `infer_task`).
        seed: seed for every random process, which makes a workflow
            reproducible. It is passed as `random_state` to every tool that
            accepts one. Defaults to `None`.
        groups: names of columns that identify groups of rows (such as race,
            court, or judge). They stay in the data but are not features, so
            transformers and models do not use them directly. Fairness metrics
            compare predictions across them, `fixest` can use them as fixed
            effects or clusters, and `group_split` can keep each group in one
            set. Defaults to an empty `list`.
        train: index labels of the rows in the training set. Defaults to
            `None`, which means that every row is used for training.
        test: index labels of the rows in the test set. Defaults to `None`,
            which means that the data has not been split.
        synthetic: index labels of the training rows that a sampler made up
            (synthetic rows and extra copies of a row), which are not real
            observations. Validators do not score a model on them. Defaults
            to an empty `Index`.
        model: the fitted model. Defaults to `None`.
        predictions: the model's predictions for the test rows (or for every
            row, if the data has not been split). Defaults to `None`.
        probabilities: the model's predicted probability of each class (one
            column for each class) for the same rows as `predictions`.
            Defaults to `None`.
        metrics: scores of the model, by the name of each metric. Defaults to
            an empty `dict`.
        tables: tables that describe the data or the model, by name. Defaults
            to an empty `dict`.
        figures: `matplotlib` figures, by name. Defaults to an empty `dict`.
        fitted: tools that were fitted to the data (such as a scaler), by the
            name of the technique that fitted them. Defaults to an empty
            `dict`.
        history: a record of each technique that was applied, in order, with
            the tool and parameters it used. Defaults to an empty `list`.
        branches: the branches of the most recent `experiment` that the
            dataset came from, with the result of each. A `scorecard`
            compares them. Defaults to an empty `list`.

    """

    data: pd.DataFrame = dataclasses.field(default_factory = pd.DataFrame)
    label: str | None = None
    task: str | None = None
    seed: int | None = None
    groups: list[str] = dataclasses.field(default_factory = list)
    train: pd.Index | None = None
    test: pd.Index | None = None
    synthetic: pd.Index = dataclasses.field(
        default_factory = lambda: pd.Index([]))
    model: Any = None
    predictions: pd.Series | None = None
    probabilities: pd.DataFrame | None = None
    metrics: dict[str, Any] = dataclasses.field(default_factory = dict)
    tables: dict[str, pd.DataFrame] = dataclasses.field(default_factory = dict)
    figures: dict[str, Any] = dataclasses.field(default_factory = dict)
    fitted: dict[str, Any] = dataclasses.field(default_factory = dict)
    history: list[dict[str, Any]] = dataclasses.field(default_factory = list)
    branches: list[Branch] = dataclasses.field(
        default_factory = list, repr = False)

    """ Initialization Methods """

    def __post_init__(self) -> None:
        """Validates `data`, `label`, `groups`, and `task`."""
        if not isinstance(self.data, pd.DataFrame):
            self.data = pd.DataFrame(self.data)
        if self.label is not None and self.label not in self.data.columns:
            message = f'the label {self.label!r} is not a column of the data'
            raise KeyError(message)
        self.groups = _validate_groups(self.groups, self.data, self.label)
        if self.task is None and self.label is not None:
            self.task = self.infer_task()
        elif self.task is not None and self.task not in _TASKS:
            message = f'task must be one of {_TASKS}, not {self.task!r}'
            raise ValueError(message)

    """ Class Methods """

    @classmethod
    def create(
        cls,
        item: Any,
        *,
        label: str | None = None,
        task: str | None = None,
        seed: int | None = None,
        groups: Sequence[str] | str | None = None) -> Dataset:
        """Returns a `Dataset` made from `item`.

        Args:
            item: a `Dataset`, a `pandas.DataFrame` or `Series`, a Polars
                `DataFrame` or `LazyFrame`, a `numpy` array, a `dict` of
                columns, the path to a data file (csv, tsv, Excel, parquet,
                feather, json, Stata, SPSS, or pickle), or an object with a
                `frame` attribute (such as a scikit-learn dataset loaded with
                `as_frame = True`).
            label: name of the label column. Defaults to `None`. If `item` has
                a `target` (as a scikit-learn dataset does), its name is used.
            task: "classify" or "regress". Defaults to `None`, in which case
                it is inferred from the label.
            seed: seed for every random process. Defaults to `None`.
            groups: names of the columns that identify groups of rows.
                Defaults to `None`.

        Raises:
            TypeError: if `item` cannot be made into a `Dataset`.

        Returns:
            A `Dataset`. If `item` is already a `Dataset`, it is returned
                with any of `label`, `task`, `seed`, and `groups` that it
                lacks.

        """
        if isinstance(item, Dataset):
            if item.label is None and label is not None:
                if label not in item.data.columns:
                    message = f'the label {label!r} is not a column of the data'
                    raise KeyError(message)
                item.label = label
                item.task = task or item.infer_task()
            if item.task is None and task is not None:
                item.task = task
            if item.seed is None:
                item.seed = seed
            if not item.groups and groups:
                item.groups = _validate_groups(groups, item.data, item.label)
            return item
        if isinstance(item, str | pathlib.Path):
            data = _read(pathlib.Path(item))
        elif type(item).__module__.startswith('polars'):
            # Polars is optional, so its frames are recognized by their module.
            # A `LazyFrame` is collected first.
            frame = item.collect() if hasattr(item, 'collect') else item
            data = frame.to_pandas()
        elif isinstance(item, pd.DataFrame):
            data = item
        elif isinstance(item, pd.Series):
            data = item.to_frame()
        # A scikit-learn dataset is a `dict`, so it is checked before other
        # mappings.
        elif isinstance(getattr(item, 'frame', None), pd.DataFrame):
            data = item.frame
            target = getattr(item, 'target', None)
            if label is None and isinstance(target, pd.Series):
                label = str(target.name)
        elif isinstance(item, np.ndarray | Mapping):
            data = pd.DataFrame(item)
        else:
            message = f'a Dataset cannot be made from {type(item).__name__}'
            raise TypeError(message)
        return cls(
            data = data,
            label = label,
            task = task,
            seed = seed,
            groups = _listify(groups))

    """ Properties """

    @property
    def booleans(self) -> list[str]:
        """Returns the names of the features that are booleans."""
        return self._kind('booleans')

    @property
    def categoricals(self) -> list[str]:
        """Returns the names of the features that are categories or text."""
        return self._kind('categoricals')

    @property
    def classes(self) -> list[Any]:
        """Returns the classes of the label, sorted if they can be.

        For a binary label, the last class is the "positive" one that metrics
        such as `precision` and `roc_auc` are about (`1` or `True`, for
        example).

        """
        # `tolist` turns `numpy` scalars into Python values.
        values = list(self.y.dropna().unique().tolist())
        try:
            return sorted(values)
        except TypeError:
            return values

    @property
    def dates(self) -> list[str]:
        """Returns the names of the features that are dates or times."""
        return self._kind('dates')

    @property
    def features(self) -> list[str]:
        """Returns the names of every column except the label and groups."""
        return [
            c for c in self.data.columns
            if c != self.label and c not in self.groups]

    @property
    def is_split(self) -> bool:
        """Returns whether the data has been split into training and test."""
        return self.test is not None

    @property
    def numerics(self) -> list[str]:
        """Returns the names of the features that are numbers (not booleans)."""
        return self._kind('numerics')

    @property
    def x(self) -> pd.DataFrame:
        """Returns the features of every row."""
        return self.data[self.features]

    @property
    def x_test(self) -> pd.DataFrame:
        """Returns the features of the test rows."""
        return self.data.loc[self._test_rows(), self.features]

    @property
    def x_train(self) -> pd.DataFrame:
        """Returns the features of the training rows."""
        return self.data.loc[self._train_rows(), self.features]

    @property
    def y(self) -> pd.Series:
        """Returns the label of every row."""
        return self.data[self._require_label()]

    @property
    def y_test(self) -> pd.Series:
        """Returns the label of the test rows."""
        labels: pd.Series = self.y.loc[self._test_rows()]
        return labels

    @property
    def y_train(self) -> pd.Series:
        """Returns the label of the training rows."""
        labels: pd.Series = self.y.loc[self._train_rows()]
        return labels

    """ Public Methods """

    def infer_task(self) -> str:
        """Returns the task implied by the type of the label.

        A label that is boolean, categorical, or text is classified. So is an
        integer label with `options._CLASSIFY_THRESHOLD` unique values or
        fewer (such as 0 and 1), or a label of whole numbers stored as floats
        with only two unique values. Any other label is regressed. Pass `task`
        to override this.

        Returns:
            "classify" or "regress".

        """
        label = self.y
        unique = label.nunique(dropna = True)
        if not pd.api.types.is_numeric_dtype(label.dtype):
            return 'classify'
        if pd.api.types.is_bool_dtype(label.dtype):
            return 'classify'
        if pd.api.types.is_integer_dtype(label.dtype):
            if unique <= options._CLASSIFY_THRESHOLD:
                return 'classify'
            return 'regress'
        values = label.dropna()
        if unique <= 2 and bool((values == values.round()).all()):  # noqa: PLR2004
            return 'classify'
        return 'regress'

    def record(self, technique: str, **details: Any) -> None:
        """Adds an entry to `history`.

        Args:
            technique: name of the technique that was applied.
            **details: anything else to record, such as the tool and its
                parameters.

        """
        self.history.append({'technique': technique, **details})

    def replace(self, data: pd.DataFrame) -> None:
        """Replaces `data`, keeping `train` and `test` to the rows that remain.

        Args:
            data: the new data.

        Raises:
            KeyError: if the label or a group is not a column of `data`.

        """
        if self.label is not None and self.label not in data.columns:
            message = f'the label {self.label!r} is not a column of the data'
            raise KeyError(message)
        missing = [g for g in self.groups if g not in data.columns]
        if missing:
            message = f'the groups {missing} are not columns of the data'
            raise KeyError(message)
        self.data = data
        if self.train is not None:
            self.train = self.train[self.train.isin(data.index)]
        if self.test is not None:
            self.test = self.test[self.test.isin(data.index)]
        self.synthetic = self.synthetic[self.synthetic.isin(data.index)]

    def resample(
        self,
        x: pd.DataFrame,
        y: Sequence[Any],
        origins: Sequence[int] | np.ndarray | None = None) -> None:
        """Replaces the training rows with `x` and `y`.

        Samplers add or remove training rows. The test rows (and their index
        labels) are not changed. The new training rows are given new index
        labels: integers after the largest current label if the index is made
        of integers, and otherwise "resampled_0", "resampled_1", and so on.
        The first copy of each real row is real. Synthetic rows and any other
        copies are added to `synthetic`.

        Args:
            x: features of the new training rows.
            y: labels of the new training rows.
            origins: for each new row, the position (in the training rows) of
                the row it copies, which gives it that row's groups, or -1 for
                a synthetic row, which has no groups. Defaults to `None`, in
                which case no new row has groups and every new row is real.

        """
        label = self._require_label()
        made_up = self._train_rows().isin(self.synthetic)
        rows = x.copy()
        rows[label] = np.asarray(y)
        if self.groups:
            training = self.data.loc[self._train_rows(), self.groups]
            positions = np.full(len(rows), -1) if origins is None else (
                np.asarray(origins))
            copied = positions >= 0
            for group in self.groups:
                values = pd.Series(
                    pd.NA, index = rows.index, dtype = training[group].dtype)
                values[copied] = training[group].to_numpy()[positions[copied]]
                rows[group] = values
        rows.index = self._new_labels(len(rows))
        rows = rows[self.data.columns]
        test = self.data.loc[self._test_rows()] if self.is_split else None
        self.data = pd.concat([rows, test]) if test is not None else rows
        self.train = rows.index
        self.synthetic = rows.index[_made_up(origins, made_up, len(rows))]

    def split(self, train: Iterable[Any], test: Iterable[Any]) -> None:
        """Sets the training and test rows.

        Args:
            train: index labels of the training rows.
            test: index labels of the test rows.

        Raises:
            ValueError: if a row is in both sets or a label is not in the data.

        """
        training, testing = pd.Index(train), pd.Index(test)
        if len(training.intersection(testing)) > 0:
            message = 'a row cannot be in both the training and test sets'
            raise ValueError(message)
        missing = training.append(testing).difference(self.data.index)
        if len(missing) > 0:
            message = f'{len(missing)} rows to split are not in the data'
            raise ValueError(message)
        self.train, self.test = training, testing

    def update_features(
        self,
        columns: Sequence[str],
        values: pd.DataFrame) -> None:
        """Replaces the feature `columns` with `values`.

        Transformers use this to swap the columns they were given for the
        columns they produce, which may have different names or a different
        number of columns (one-hot encoding, for example). The new columns
        take the place of the first column they replace, so the order of the
        other columns does not change.

        Args:
            columns: names of the columns to remove.
            values: the new columns, with the same index as `data`.

        Raises:
            ValueError: if `values` has a different index from `data`, or if a
                new column has the same name as a column that is kept.

        """
        if not values.index.equals(self.data.index):
            message = 'the new columns must have the same index as the data'
            raise ValueError(message)
        removed = set(columns)
        kept = [c for c in self.data.columns if c not in removed]
        clashes = [c for c in values.columns if c in kept]
        if clashes:
            message = (
                f'the new columns {clashes} have the same names as columns '
                f'that are kept'
            )
            raise ValueError(message)
        place = min(
            (i for i, c in enumerate(self.data.columns) if c in removed),
            default = len(self.data.columns))
        before = [c for c in self.data.columns[:place] if c not in removed]
        after = [c for c in self.data.columns[place:] if c not in removed]
        self.data = pd.concat(
            [self.data[before], values, self.data[after]], axis = 1)

    """ Private Methods """

    def _kind(self, kind: str) -> list[str]:
        """Returns the names of the features of one kind.

        Args:
            kind: "booleans", "numerics", or "categoricals". Dates and times
                are not in any kind.

        Returns:
            Names of the features of that kind, in order.

        """
        return [c for c in self.features if _kind_of(self.data[c]) == kind]

    def _new_labels(self, count: int) -> pd.Index:
        """Returns `count` index labels that are not used by the test rows.

        Args:
            count: number of labels.

        Returns:
            New index labels.

        """
        index = self.data.index
        if len(index) == 0 or pd.api.types.is_integer_dtype(index.dtype):
            start = int(index.max()) + 1 if len(index) > 0 else 0
            return pd.RangeIndex(start, start + count)
        return pd.Index([f'resampled_{i}' for i in range(count)])

    def _require_label(self) -> str:
        """Returns `label`.

        Raises:
            ValueError: if there is no label.

        Returns:
            The name of the label column.

        """
        if self.label is None:
            message = (
                'the dataset has no label: pass one when it is created or set '
                '"label" in the "general" section of the settings'
            )
            raise ValueError(message)
        return self.label

    def _test_rows(self) -> pd.Index:
        """Returns the index labels of the test rows.

        Raises:
            ValueError: if the data has not been split.

        Returns:
            The test rows.

        """
        if self.test is None:
            message = 'the data has not been split: apply a splitter first'
            raise ValueError(message)
        return self.test

    def _train_rows(self) -> pd.Index:
        """Returns the index labels of the training rows.

        Returns:
            The training rows, or every row if the data has not been split.

        """
        return self.data.index if self.train is None else self.train

    """ Dunder Methods """

    def __repr__(self) -> str:
        """Returns a short description of the dataset.

        Returns:
            The size of the data, its label, and how it has been split.

        """
        rows, columns = self.data.shape
        parts = [f'rows={rows}', f'columns={columns}']
        if self.label is not None:
            parts.append(f'label={self.label!r}')
            parts.append(f'task={self.task!r}')
        if self.groups:
            parts.append(f'groups={self.groups!r}')
        if self.is_split:
            parts.append(f'train={len(self._train_rows())}')
            parts.append(f'test={len(self._test_rows())}')
        if self.model is not None:
            parts.append(f'model={type(self.model).__name__}')
        return f'Dataset({", ".join(parts)})'


@dataclasses.dataclass
class Branch:
    """One branch of an experiment: a combination of techniques and its result.

    An `experiment` tries every combination of one technique from each step.
    Each combination is a branch. The winning dataset keeps a `Branch` for
    every one of them, so that a `scorecard` can compare them after the
    experiment.

    Args:
        label: the names of the nodes in the branch, joined by " > ".
        steps: the technique used at each step, by the name of the step.
        result: the outcome of the branch: a `Dataset` with only the label
            column, and the branch's predictions, probabilities, and metrics.
        score: the branch's score from the experiment's criteria. Higher is
            better. Defaults to `None`.
        criterion: the name of the experiment's criteria. Defaults to `None`.

    """

    label: str
    steps: dict[str, str]
    result: Dataset
    score: float | None = None
    criterion: str | None = None


@dataclasses.dataclass
class Operation(chrisjen.Technique, abc.ABC):
    """Base class for amos techniques, which work on a `Dataset`.

    An operation is a `chrisjen.Technique`, so it is stored in the
    `chrisjen.library` under its snake case name and can be named in the
    settings of any worker or step. Its `apply` method makes the item into a
    `Dataset` (if it is not one already), passes it to `implement` with the
    technique's parameters, and returns it. The dataset is changed in place.
    Every operation adds an entry to the dataset's `history`.

    `Operation` is a genre: its abstract subclasses (`Loader`, `Cleaner`,
    `Munger`, `Describer`, `Splitter`, `Transformer`, `Sampler`, `Model`,
    `Validator`, `Metric`, `Evaluator`, `Inference`, and `Plot`) are genres
    within it. To add a technique, subclass the genre that fits and write the
    method it requires, or set `contents` to the tool to wrap. To add a new
    kind of technique, subclass `Operation` and write `implement`.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, in which case it is the snake case name of the
            class.
        contents: the tool that the technique wraps: a callable, or the import
            path of one. A path is only imported when the technique is used.
            Defaults to `None`.
        parameters: keyword arguments for the tool. Defaults to an empty
            `dict`.

    """

    # An empty name is replaced with the snake case name of the class when the
    # technique is created (by `holden.Labeled`), so the name is never `None`.
    name: str = ''

    """ Public Methods """

    def apply(self, item: Any, **kwargs: Any) -> Dataset:
        """Applies the operation to `item`.

        Args:
            item: a `Dataset` or anything that `Dataset.create` accepts.
            **kwargs: keyword arguments for `implement` that take precedence
                over `parameters`.

        Returns:
            The changed `Dataset`.

        """
        dataset = Dataset.create(item)
        count = len(dataset.history)
        result = self.implement(dataset, **{**self._keywords(), **kwargs})
        # Records the operation if `implement` did not record it in more
        # detail.
        if len(result.history) == count:
            result.record(self.name, tool = utilities.describe_tool(
                self.contents))
        return result

    @abc.abstractmethod
    def implement(self, item: Dataset, **kwargs: Any) -> Dataset:
        """Applies the operation to `item`.

        Args:
            item: the dataset to work on.
            **kwargs: parameters for the operation.

        Returns:
            The changed dataset.

        """

    """ Private Methods """

    def _make_tool(
        self,
        item: Dataset,
        parameters: Mapping[str, Any],
        tool: str | Callable[..., Any] | None = None) -> Any:
        """Returns the wrapped tool, built with the parameters it accepts.

        Parameters are first passed to `_prepare`, which subclasses use to add
        defaults that depend on the data. If the dataset has a `seed` and the
        tool accepts a `random_state` that was not set, the seed is used.

        Args:
            item: the dataset the tool will be used on.
            parameters: parameters for the tool. Only those it accepts are
                passed to it.
            tool: the tool (a class or other callable, or its import path).
                Defaults to `None`, in which case `contents` is used.

        Raises:
            NotImplementedError: if there is no tool.

        Returns:
            The built tool.

        """
        source = self.contents if tool is None else tool
        if source is None:
            message = (
                f'technique {self.name!r} has no tool: pass a callable or '
                f'import path as contents or override implement'
            )
            raise NotImplementedError(message)
        built = utilities.import_tool(source)
        parameters = self._prepare(item, dict(parameters))
        accepted = utilities.accepted_parameters(built, parameters)
        if (
            item.seed is not None
            and 'random_state' not in accepted
            and utilities.accepted_parameters(built, {'random_state': None})):
            accepted['random_state'] = item.seed
        return built(**accepted)

    def _keywords(self) -> dict[str, Any]:
        """Returns `parameters` as keyword arguments.

        Returns:
            A copy of `parameters` with `str` keys.

        """
        return {str(k): v for k, v in self.parameters.items()}

    def _namify(self) -> str:
        """Returns the snake case name of the class.

        `holden` calls this to name an instance that was created without a
        name (such as the criteria built for a contest).

        Returns:
            The default name of the technique.

        """
        return str(camina.snakify(type(self).__name__))

    def _prepare(
        self,
        item: Dataset,  # noqa: ARG002
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Returns `parameters` with any defaults that depend on `item`.

        Subclasses override this to choose a default (such as the scoring
        function of a feature selector) based on the data or task.

        Args:
            item: the dataset the tool will be used on.
            parameters: parameters for the tool.

        Returns:
            The parameters to build the tool with.

        """
        return parameters


""" Private Functions """


def _listify(item: Sequence[str] | str | None) -> list[str]:
    """Returns column names as a `list`.

    Args:
        item: a name, a sequence of names, or `None`.

    Returns:
        The names (an empty `list` for `None`).

    """
    if item is None:
        return []
    if isinstance(item, str):
        return [item]
    return list(item)


def _made_up(
    origins: Sequence[int] | np.ndarray | None,
    made_up: np.ndarray,
    count: int) -> np.ndarray:
    """Returns which resampled rows are not real observations.

    Args:
        origins: for each resampled row, the position of the training row it
            copies, or -1 for a synthetic row. `None` means that every row is
            real.
        made_up: for each training row before resampling, whether it was
            made up by an earlier sampler.
        count: number of resampled rows.

    Returns:
        A boolean array that is `True` for each synthetic row, each copy of a
            made-up row, and each copy of a row after its first.

    """
    if origins is None:
        return np.zeros(count, dtype = bool)
    flags = np.ones(count, dtype = bool)
    seen: set[int] = set()
    for position, origin in enumerate(np.asarray(origins)):
        if origin >= 0 and not made_up[origin] and origin not in seen:
            seen.add(int(origin))
            flags[position] = False
    return flags


def _validate_groups(
    groups: Sequence[str] | str | None,
    data: pd.DataFrame,
    label: str | None) -> list[str]:
    """Returns the names of the group columns, checking them.

    Args:
        groups: a name, a sequence of names, or `None`.
        data: the data.
        label: name of the label, which cannot be a group.

    Raises:
        KeyError: if a group is not a column of `data`.
        ValueError: if the label is a group.

    Returns:
        The names of the group columns.

    """
    names = _listify(groups)
    missing = [g for g in names if g not in data.columns]
    if missing:
        message = f'the groups {missing} are not columns of the data'
        raise KeyError(message)
    if label is not None and label in names:
        message = f'the label {label!r} cannot also be a group'
        raise ValueError(message)
    return names


def _kind_of(column: pd.Series) -> str:
    """Returns the kind of values in `column`.

    Args:
        column: a column of data.

    Returns:
        "booleans", "numerics", "dates" (for dates, times, and periods), or
            "categoricals" (for categories, text, and anything else).

    """
    dtype = column.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return 'booleans'
    if pd.api.types.is_numeric_dtype(dtype):
        return 'numerics'
    if (
        pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
        or isinstance(dtype, pd.PeriodDtype)):
        return 'dates'
    return 'categoricals'


def _read(path: pathlib.Path) -> pd.DataFrame:
    """Loads the data file at `path`.

    Args:
        path: path to a data file.

    Raises:
        FileNotFoundError: if there is no file at `path`.
        ValueError: if the type of file is not supported.

    Returns:
        The data.

    """
    if not path.is_file():
        message = f'there is no data file at {path}'
        raise FileNotFoundError(message)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        message = (
            f'{path.suffix!r} files are not supported: use one of '
            f'{sorted(_READERS)}'
        )
        raise ValueError(message)
    return reader(path)
