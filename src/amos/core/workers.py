"""Worker designs for data science workflows.

`amos` uses the designs of `chrisjen` (`flow`, `benchmark`, `contest`, and
`survey`) and adds one of its own.

Contents:
    Experiment: tries every combination of techniques, keeps the best, and
        records how every combination did.

"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import weakref
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import chrisjen
import numpy as np
import pandas as pd

from . import base, metrics, utilities


@dataclasses.dataclass
class Experiment(chrisjen.Contest):
    """Tries every combination of techniques and records how each one did.

    Like a `contest`, an experiment applies every combination of one technique
    from each step (every path through its graph) to its own copy of the
    dataset, and keeps the dataset with the best score from its `criteria`.
    It also adds a comparison table to the tables of the winning dataset,
    named "{name}_comparison", with one row for each combination: its score,
    its rank, and every metric it computed. And it keeps a `Branch` for each
    combination in the winning dataset's `branches`, so that a `scorecard` can
    compare every combination on every metric. This is what a paper reports
    to show that a result does not depend on one arbitrary choice of
    preprocessing or model.

    If the criteria is a `Metric`, its score is also stored in the `metrics`
    of each combination's dataset, so the table shows the real value of the
    metric (not the negated value used to rank metrics for which lower is
    better).

    The whole dataset of every combination is kept in `results`, so that a
    `regression_table` can estimate the effects of each one's model. If
    `conserve_memory` is true (in the worker's section of the settings, or in
    the "general" section for every worker), only the best dataset so far is
    kept while the combinations are tried, so no more than two are in memory
    at once. The comparison table and the branches are the same, but only
    the winner's branch has its whole `dataset`.

    Each path is a branch of an experiment (see `base.in_experiment`). A
    technique whose assumptions or needs the data does not meet can leave
    its branch out (see `base.SkippedBranchError`): the branch is not
    scored, and its row of the comparison table has no score and says why
    it was left out. The comparison table also has a "violations" column,
    with the assumptions that each branch did not meet, if any did.

    If `n_jobs` is more than 1 (in the worker's section of the settings),
    that many paths are applied at once, each in its own process, and the
    results are compared in the order of the paths, as they are one at a
    time. Each process holds a copy of the dataset and of what its path
    learns, so memory grows with `n_jobs`. A path that fits its model on a
    GPU shares the GPU with the others.

    Args:
        name: name used to refer to the worker in a workflow.
        contents: the graph of nodes. Defaults to an empty `dict`.
        parameters: keyword arguments for the worker to use. Defaults to an
            empty `dict`.
        criteria: the criteria used to score each result. Higher scores are
            better. Defaults to `None`.
        conserve_memory: whether to keep only the winner's whole dataset in
            `results`. Defaults to `False`.
        n_jobs: how many paths to apply at once (-1 for one for each
            processor). Defaults to 1, which applies one path at a time.

    Attributes:
        results: the result of each path in the most recent call, by the
            names of the nodes in the path (joined by " > "). If
            `conserve_memory` is true, only the winner's.
        scores: the score of each path in the most recent call.
        outcomes: the outcome of each path that made a `Dataset` in the most
            recent call: its label and group columns, predictions,
            probabilities, and metrics.
        excluded: why each path that was left out in the most recent call
            was left out, by its label.
        violations: the assumptions and needs that each path did not meet in
            the most recent call (as `conditions.Verdict.describe` writes
            them), for each path that did not meet some.
        skipped: the entries of `history` of the techniques that each path
            skipped in the most recent call, for each path that skipped any.
            They are also in the experiment's entry in the winner's `history`
            ("skipped_techniques"), since the datasets of the other paths may
            not be kept.
        winner: label of the path with the best result in the most recent
            call.

    """

    outcomes: dict[str, base.Dataset] = dataclasses.field(
        default_factory = dict)
    skipped: dict[str, list[dict[str, Any]]] = dataclasses.field(
        default_factory = dict)
    excluded: dict[str, str] = dataclasses.field(default_factory = dict)
    violations: dict[str, list[str]] = dataclasses.field(
        default_factory = dict)
    n_jobs: int | None = 1
    # How many entries of `history` the item had before any path was tried.
    _inherited: int = dataclasses.field(
        default = 0, init = False, repr = False)

    """ Public Methods """

    def apply(self, item: Any, **kwargs: Any) -> Any:
        """Returns the result of the path with the best score.

        Args:
            item: data or object to change.
            **kwargs: keyword arguments passed to the nodes.

        Raises:
            AssumptionError: if every path was left out because the data did
                not meet an assumption or need of one of its techniques.
            ValueError: if there are no paths or no `criteria`.

        Returns:
            The best result. Ties go to the first path.

        """
        self.outcomes = {}
        self.skipped = {}
        self.excluded = {}
        self.violations = {}
        self._inherited = len(getattr(item, 'history', ()))
        with base.skipping_incompatible_techniques():
            best = super().apply(item, **kwargs)
        if best is None and self.excluded:
            reasons = '; '.join(
                f'{label}: {reason}' for label, reason in self.excluded.items())
            message = f'{self.name!r} left out every path: {reasons}'
            raise base.AssumptionError(message)
        if isinstance(best, base.Dataset):
            best.branches = self._branches()
            best.tables[f'{self.name}_comparison'] = self.tabulate()
            details: dict[str, Any] = (
                {'skipped_techniques': self.skipped} if self.skipped else {})
            if self.excluded:
                details['excluded_paths'] = dict(self.excluded)
            if self.violations:
                details['path_violations'] = {
                    label: list(found)
                    for label, found in self.violations.items()}
            best.record(
                str(self.name),
                winner = self.winner,
                criterion = getattr(self.criteria, 'name', None),
                paths = len(self.scores),
                **details)
        return best

    def each_path(
        self,
        item: Any,
        **kwargs: Any) -> Iterator[tuple[str, Any]]:
        """Applies each path to its own copy of `item`, as a branch.

        With `n_jobs` of 1, the paths are applied one at a time, each only
        when its result is asked for (as in `chrisjen`'s `each_path`), so
        that a result can be dropped before the next is made. Otherwise,
        they are applied in other processes, at most `n_jobs` at once, and
        their results are given back in the order of the paths. A path that
        is left out (see `base.SkippedBranchError`) gives no result, and why
        it was left out is kept in `excluded`.

        Args:
            item: data or object to change. It is not changed itself.
            **kwargs: keyword arguments passed to the nodes.

        Raises:
            ValueError: if there are no paths.

        Yields:
            tuple[str, Any]: the label of each path and its result.

        """
        paths = self.walk()
        if not paths:
            message = f'{self.name!r} has no paths to compare'
            raise ValueError(message)
        state = dataclasses.replace(
            base.workflow_state(), in_experiment = True)
        results: Iterable[Any]
        if self.n_jobs == 1:
            results = (
                _apply_path(path, item, state, kwargs) for path in paths)
        else:
            joblib = importlib.import_module('joblib')
            parallel = joblib.Parallel(
                n_jobs = self.n_jobs, return_as = 'generator')
            results = parallel(
                joblib.delayed(_apply_path)(path, item, state, kwargs)
                for path in paths)
        for path, result in zip(paths, results, strict = True):
            label = chrisjen.utilities.label_path(path)
            if isinstance(result, _LeftOut):
                self.excluded[label] = result.reason
                continue
            yield label, result

    def tabulate(self) -> pd.DataFrame:
        """Returns a table comparing every path of the most recent call.

        Returns:
            One row for each path (labeled with the names of its nodes),
                sorted from best to worst, with "rank" and "score" columns and
                a column for each metric that the paths computed. If a path
                did not meet an assumption or need of one of its techniques,
                a "violations" column lists them, and a path that was left
                out (last, with no rank or score) says why.

        """
        rows: dict[str, dict[str, Any]] = {}
        for label, score in self.scores.items():
            outcome = self.outcomes.get(label)
            values = {} if outcome is None else dict(outcome.metrics)
            rows[label] = {'score': score, **values}
        table = pd.DataFrame.from_dict(rows, orient = 'index')
        table.index.name = 'path'
        table = table.sort_values('score', ascending = False, kind = 'stable')
        table.insert(0, 'rank', range(1, len(table) + 1))
        if self.excluded:
            left_out = pd.DataFrame(
                {'score': np.nan}, index = pd.Index(list(self.excluded)))
            table = pd.concat([table, left_out])
            table['rank'] = table['rank'].astype('Int64')
            table.index.name = 'path'
        if self.violations or self.excluded:
            table['violations'] = [
                f'left out: {self.excluded[label]}' if label in self.excluded
                else '; '.join(self.violations.get(label, []))
                for label in table.index]
        return table

    """ Private Methods """

    def _branches(self) -> list[base.Branch]:
        """Returns a record of every path that made a `Dataset`.

        Returns:
            A `Branch` for each path, in the order of the paths.

        """
        paths = self.walk()
        columns = _step_names(paths)
        criterion = getattr(self.criteria, 'name', None)
        branches = []
        for path in paths:
            label = chrisjen.utilities.label_path(path)
            outcome = self.outcomes.get(label)
            if outcome is None:
                continue
            steps = {
                columns[position]: _technique_of(node)
                for position, node in enumerate(path) if position in columns}
            # Only the datasets that were kept (every one, unless memory is
            # conserved) can be found again through the branch.
            dataset = self.results.get(label)
            branches.append(base.Branch(
                label = label,
                steps = steps,
                result = outcome,
                score = self.scores.get(label),
                criterion = criterion,
                source = None if dataset is None else weakref.ref(dataset)))
        return branches

    def _observe(self, label: str, result: Any) -> None:
        """Keeps the outcome of each path before its dataset may be dropped.

        If the criteria is a `Metric`, its real value is first stored in the
        metrics of `result` (it is negated in the score if lower values are
        better).

        Args:
            label: the label of the path.
            result: the result of the path.

        """
        if not isinstance(result, base.Dataset):
            return
        criteria = self.criteria
        if isinstance(criteria, metrics.Metric):
            sign = 1 if criteria.greater_is_better else -1
            result.metrics[criteria.name] = sign * self.scores[label]
        self.outcomes[label] = _outcome(result)
        entries = result.history[self._inherited:]
        skipped = [entry for entry in entries if entry.get('skipped')]
        if skipped:
            self.skipped[label] = skipped
        violations = [
            violation for entry in entries
            for violation in entry.get('violations', [])]
        if violations:
            self.violations[label] = violations


""" Private Functions """


@dataclasses.dataclass(frozen = True)
class _LeftOut:
    """The result of a path that was left out of an experiment.

    Args:
        reason: why it was left out.

    """

    reason: str


def _apply_path(
    path: Sequence[Any],
    item: Any,
    state: base.WorkflowState,
    kwargs: Mapping[str, Any]) -> Any:
    """Applies copies of the nodes of a path to a copy of `item`.

    This may run in another process (or thread), so it first restores the
    settings of the workflow that the experiment is in, with the label of
    the path as the label of the branch (after the label of the branch that
    the experiment is in, if it is in one).

    Args:
        path: the nodes of the path.
        item: data or object to change. It is not changed itself.
        state: the settings of the workflow (see `base.workflow_state`).
        kwargs: keyword arguments passed to the nodes.

    Returns:
        The result of the last node, or a `_LeftOut` with the reason if a
            technique left the path out (see `base.SkippedBranchError`).

    """
    result = copy.deepcopy(item)
    label = chrisjen.utilities.label_path(path)
    branch = f'{state.branch} | {label}' if state.branch else label
    state = dataclasses.replace(state, branch = branch)
    with base.restored_workflow_state(state):
        try:
            for node in path:
                result = copy.deepcopy(node).apply(result, **kwargs)
        except base.SkippedBranchError as error:
            return _LeftOut(reason = str(error))
    return result


def _outcome(item: base.Dataset) -> base.Dataset:
    """Returns the outcome of a branch: its label, predictions, and metrics.

    Keeping only the label and group columns (not every feature) lets the
    winning dataset remember every branch without keeping a copy of the data
    for each. The groups let fairness metrics score each branch.

    Args:
        item: the dataset that a branch made.

    Returns:
        A `Dataset` with only the label and group columns, and the
            predictions, probabilities, and metrics of `item`.

    """
    columns = ([] if item.label is None else [item.label]) + item.groups
    return base.Dataset(
        data = item.data[columns].copy(),
        label = item.label,
        groups = item.groups,
        task = item.task,
        seed = item.seed,
        use_gpu = item.use_gpu,
        train = item.train,
        test = item.test,
        predictions = item.predictions,
        probabilities = item.probabilities,
        metrics = dict(item.metrics))


def _step_names(paths: list[list[Any]]) -> dict[int, str]:
    """Returns a name for each position in the paths that holds alternatives.

    A position is named for its step (as in the settings) if it holds steps,
    or else for the genre of the techniques in it (such as "scaler"), or else
    "step_{n}". Positions that hold workers, which are part of every path
    rather than alternatives, are not named.

    Args:
        paths: the paths of a worker, as lists of nodes.

    Returns:
        The name of each position, by its index in a path.

    """
    names: dict[int, str] = {}
    for position in range(max((len(p) for p in paths), default = 0)):
        nodes = [p[position] for p in paths if position < len(p)]
        if all(isinstance(n, chrisjen.Worker) for n in nodes):
            continue
        name = None
        for node in nodes:
            parts = utilities.technique_and_step(node)
            if parts is not None:
                name = parts[1]
                break
            if isinstance(node, base.Operation):
                name = chrisjen.library.classify(type(node))
                break
        name = name or f'step_{position + 1}'
        while name in names.values():
            name = f'{name}_{position + 1}'
        names[position] = name
    return names


def _technique_of(node: Any) -> str:
    """Returns the name of the technique that a node in a path applies.

    Args:
        node: a node in a path: a step or a technique.

    Returns:
        The name of the technique that a step wraps, or the node's name.

    """
    parts = utilities.technique_and_step(node)
    return str(node.name) if parts is None else parts[0]
