"""Worker designs for data science workflows.

`amos` uses the designs of `chrisjen` (`flow`, `benchmark`, `contest`, and
`survey`) and adds one of its own.

Contents:
    Experiment: tries every combination of techniques, keeps the best, and
        records how every combination did.

"""

from __future__ import annotations

import dataclasses
from typing import Any

import chrisjen
import pandas as pd

from . import base, metrics


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

    Args:
        name: name used to refer to the worker in a workflow.
        contents: the graph of nodes. Defaults to an empty `dict`.
        parameters: keyword arguments for the worker to use. Defaults to an
            empty `dict`.
        criteria: the criteria used to score each result. Higher scores are
            better. Defaults to `None`.

    Attributes:
        results: the result of each path in the most recent call, by the
            names of the nodes in the path (joined by " > ").
        scores: the score of each path in the most recent call.
        winner: label of the path with the best result in the most recent
            call.

    """

    scores: dict[str, Any] = dataclasses.field(default_factory = dict)

    """ Public Methods """

    def implement(self, item: Any, **kwargs: Any) -> Any:
        """Returns the result of the path with the best score.

        Args:
            item: data or object to change.
            **kwargs: keyword arguments passed to the nodes.

        Raises:
            ValueError: if there are no paths or no `criteria`.

        Returns:
            The best result. Ties go to the first path.

        """
        results = self.try_paths(item, **kwargs)
        self.scores = dict(zip(
            results, self.compare(list(results.values())), strict = True))
        criteria = self.criteria
        if isinstance(criteria, metrics.Metric):
            # Stores the real value of the metric, which is negated in the
            # score if lower values are better.
            sign = 1 if criteria.greater_is_better else -1
            for label, result in results.items():
                if isinstance(result, base.Dataset):
                    result.metrics[criteria.name] = sign * self.scores[label]
        self.winner = max(self.scores, key = self.scores.__getitem__)
        best = results[self.winner]
        if isinstance(best, base.Dataset):
            best.branches = self._branches(results)
            best.tables[f'{self.name}_comparison'] = self.tabulate()
            best.record(
                str(self.name),
                winner = self.winner,
                criterion = getattr(self.criteria, 'name', None),
                paths = len(results))
        return best

    def tabulate(self) -> pd.DataFrame:
        """Returns a table comparing every path of the most recent call.

        Returns:
            One row for each path (labeled with the names of its nodes),
                sorted from best to worst, with "rank" and "score" columns and
                a column for each metric that the paths computed.

        """
        rows = {}
        for label, score in self.scores.items():
            result = self.results[label]
            values = dict(getattr(result, 'metrics', {}))
            rows[label] = {'score': score, **values}
        table = pd.DataFrame.from_dict(rows, orient = 'index')
        table.index.name = 'path'
        table = table.sort_values('score', ascending = False, kind = 'stable')
        table.insert(0, 'rank', range(1, len(table) + 1))
        return table

    """ Private Methods """

    def _branches(self, results: dict[str, Any]) -> list[base.Branch]:
        """Returns a record of every path that made a `Dataset`.

        Args:
            results: the result of each path, by its label.

        Returns:
            A `Branch` for each path, in the order of the paths.

        """
        paths = self.walk()
        columns = _step_names(paths)
        criterion = getattr(self.criteria, 'name', None)
        branches = []
        for path in paths:
            label = ' > '.join(str(getattr(n, 'name', n)) for n in path)
            result = results.get(label)
            if not isinstance(result, base.Dataset):
                continue
            steps = {
                columns[position]: _technique_of(node)
                for position, node in enumerate(path) if position in columns}
            branches.append(base.Branch(
                label = label,
                steps = steps,
                result = _outcome(result),
                score = self.scores.get(label),
                criterion = criterion))
        return branches


""" Private Functions """


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
            if isinstance(node, chrisjen.Step) and node.contents is not None:
                name = str(node.name).removeprefix(f'{node.contents.name}_')
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
    if isinstance(node, chrisjen.Step) and node.contents is not None:
        return str(node.contents.name)
    return str(node.name)
