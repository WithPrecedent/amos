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
    figures that were made.

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
            ' > '.join(str(getattr(node, 'name', node)) for node in path)
            for path in workflow.walk()]
        lines = [
            f'project: {project.name}',
            f'id: {project.id}',
            f'workflow: {"; ".join(paths) or "none"}']
        if isinstance(result, base.Dataset):
            lines.extend(_describe(result))
        else:
            lines.append(f'result: {result!r}')
        for worker in _comparators(workflow):
            if worker.winner is not None:
                lines.append(
                    f'{worker.name}: the best of {len(worker.results)} paths '
                    f'was {worker.winner}')
        text = '\n'.join(lines)
        self.contents = text
        return text


""" Private Functions """


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
