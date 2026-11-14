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
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import chrisjen
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
            probabilities, and metrics (see `_outcome`).
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
            ValueError: if there are no paths or no `criteria`.

        Returns:
            The best result. Ties go to the first path.

        """
        self.outcomes = {}
        self.skipped = {}
        self._inherited = len(getattr(item, 'history', ()))
        with base.skipping_incompatible_techniques():
            best = super().apply(item, **kwargs)
        if isinstance(best, base.Dataset):
            best.branches = self._branches()
            best.tables[f'{self.name}_comparison'] = self.tabulate()
            details = (
                {'skipped_techniques': self.skipped} if self.skipped else {})
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
        """Applies each path to its own copy of `item`.

        With `n_jobs` of 1, this is `chrisjen`'s `each_path`. Otherwise, the
        paths are applied in other processes, at most `n_jobs` at once, and
        their results are given back in the order of the paths.

        Args:
            item: data or object to change. It is not changed itself.
            **kwargs: keyword arguments passed to the nodes.

        Raises:
            ValueError: if there are no paths.

        Yields:
            tuple[str, Any]: the label of each path and its result.

        """
        if self.n_jobs == 1:
            yield from super().each_path(item, **kwargs)
            return
        paths = self.walk()
        if not paths:
            message = f'{self.name!r} has no paths to compare'
            raise ValueError(message)
        joblib = importlib.import_module('joblib')
        state = base.workflow_state()
        parallel = joblib.Parallel(
            n_jobs = self.n_jobs, return_as = 'generator')
        results = parallel(
            joblib.delayed(_apply_path)(path, item, state, kwargs)
            for path in paths)
        for path, result in zip(paths, results, strict = True):
            yield chrisjen.utilities.label_path(path), result

    def tabulate(self) -> pd.DataFrame:
        """Returns a table comparing every path of the most recent call.

        Returns:
            One row for each path (labeled with the names of its nodes),
                sorted from best to worst, with "rank" and "score" columns and
                a column for each metric that the paths computed.

        """
        rows = {}
        for label, score in self.scores.items():
            outcome = self.outcomes.get(label)
            values = {} if outcome is None else dict(outcome.metrics)
            rows[label] = {'score': score, **values}
        table = pd.DataFrame.from_dict(rows, orient = 'index')
        table.index.name = 'path'
        table = table.sort_values('score', ascending = False, kind = 'stable')
        table.insert(0, 'rank', range(1, len(table) + 1))
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
        skipped = [
            entry for entry in result.history[self._inherited:]
            if entry.get('skipped')]
        if skipped:
            self.skipped[label] = skipped


""" Private Functions """


def _apply_path(
    path: Sequence[Any],
    item: Any,
    state: base.WorkflowState,
    kwargs: Mapping[str, Any]) -> Any:
    """Applies copies of the nodes of a path to a copy of `item`.

    This runs in another process (or thread), so it first restores the
    settings of the workflow that the experiment is in.

    Args:
        path: the nodes of the path.
        item: data or object to change. It is not changed itself.
        state: the settings of the workflow (see `base.workflow_state`).
        kwargs: keyword arguments passed to the nodes.

    Returns:
        The result of the last node.

    """
    result = copy.deepcopy(item)
    with base.restored_workflow_state(state):
        for node in path:
            result = copy.deepcopy(node).apply(result, **kwargs)
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
