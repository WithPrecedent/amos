"""Reports that describe a project after it is applied.

Contents:
    Findings: a plain-text account of what a project did and found.

"""

from __future__ import annotations

import dataclasses
from typing import Any

import chrisjen

from . import base


@dataclasses.dataclass
class Findings(chrisjen.Report):
    """A plain-text account of what a project did and found.

    This is the default report of an `amos.Project`. It describes the data,
    how it was split, the path taken through the workflow, the winner of each
    `experiment` or `contest`, the model, its metrics, and the tables and
    figures that were made. It ends with the assumptions and needs of the
    techniques that the data did not meet (see the `conditions` module):
    those of the result's own techniques, then, for each experiment, those
    of each of its paths and the paths that were left out.

    Args:
        contents: the text of the most recent report. Defaults to `None`.

    """

    def generate(self, project: chrisjen.Project) -> str:
        """Generates the report and stores it in `contents`.

        Args:
            project: the project to report on, after it has been applied.

        Returns:
            The text of the report.

        """
        result = project.result
        workflow = project.workflow
        paths = [] if workflow is None else [
            chrisjen.utilities.label_path(path)
            for path in workflow.walk()]
        lines = [
            f'project: {project.name}',
            f'id: {project.id}',
            f'workflow: {"; ".join(paths) or "none"}']
        if isinstance(result, base.Dataset):
            lines.extend(_describe(result))
        else:
            lines.append(f'result: {result!r}')
        comparators = _comparators(workflow)
        for worker in comparators:
            if worker.winner is not None:
                # A worker that conserves memory keeps only the winner's
                # result, but it scores every path it compares.
                compared = getattr(worker, 'scores', None) or worker.results
                lines.append(
                    f'{worker.name}: the best of {len(compared)} paths '
                    f'was {worker.winner}')
        lines.extend(_assumptions(result, comparators))
        text = '\n'.join(lines)
        self.contents = text
        return text


""" Private Functions """


def _assumptions(result: Any, comparators: list[Any]) -> list[str]:
    """Returns lines about the assumptions and needs that were not met.

    Args:
        result: the result of the project.
        comparators: the workers of the workflow that compare paths.

    Returns:
        A heading and a line for each violation of the result's techniques,
            then for each path of each experiment that did not meet some or
            was left out, or no lines if there were none.

    """
    lines: list[str] = []
    if isinstance(result, base.Dataset):
        found = [
            violation for entry in result.history
            for violation in entry.get('violations', [])
            if isinstance(violation, str)]
        lines.extend(f'  {violation}' for violation in dict.fromkeys(found))
    for worker in comparators:
        for label, violations in getattr(worker, 'violations', {}).items():
            lines.append(f'  {worker.name} path {label}:')
            lines.extend(f'    {violation}' for violation in violations)
        for label, reason in getattr(worker, 'excluded', {}).items():
            lines.append(f'  {worker.name} left out path {label}: {reason}')
    if not lines:
        return []
    policy = getattr(result, 'assumption_violations', 'warn')
    return [
        f'assumptions not met (assumption_violations = {policy}):', *lines]


def _comparators(worker: Any) -> list[Any]:
    """Returns every worker in `worker` (and itself) that compares paths.

    Args:
        worker: a `chrisjen.Worker`.

    Returns:
        The `contest` and `experiment` workers (and others that keep a
            winner), outermost first.

    """
    found = []
    if isinstance(worker, chrisjen.Comparator) and hasattr(worker, 'winner'):
        found.append(worker)
    if isinstance(worker, chrisjen.Worker):
        for node in worker:
            found.extend(_comparators(node))
    elif isinstance(getattr(worker, 'contents', None), chrisjen.Worker):
        # A step that wraps a worker.
        found.extend(_comparators(worker.contents))
    return found


def _describe(item: base.Dataset) -> list[str]:
    """Returns lines that describe a dataset.

    Args:
        item: the dataset.

    Returns:
        Lines about its data, split, model, metrics, tables, and figures.

    """
    rows, columns = item.data.shape
    lines = [f'data: {rows} rows and {columns} columns']
    if item.label is not None:
        lines.append(f'label: {item.label} ({item.task})')
    if item.seed is not None:
        lines.append(f'seed: {item.seed}')
    if item.is_split:
        lines.append(
            f'split: {len(item.x_train)} training rows and '
            f'{len(item.x_test)} test rows')
    if item.model is not None:
        lines.append(f'model: {type(item.model).__name__}')
    if item.metrics:
        values = ', '.join(
            f'{name} {value:.3f}' for name, value in item.metrics.items())
        lines.append(f'metrics: {values}')
    if item.tables:
        lines.append(f'tables: {", ".join(item.tables)}')
    if item.figures:
        lines.append(f'figures: {", ".join(item.figures)}')
    return lines
