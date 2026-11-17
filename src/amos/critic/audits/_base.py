"""The base class of audits, and what they share.

Contents:
    Audit: base class for techniques that check the assumptions of a fitted
        model from its errors.

"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any

import camina
import chrisjen

from ...core import base, conditions


@dataclasses.dataclass
class Audit(base.Operation, abc.ABC):
    """Base class for techniques that check a fitted model's assumptions.

    Some assumptions (such as errors of constant variance) can only be
    checked from the errors of a fitted model. An audit checks the
    assumptions that it is named for (each condition's `audited_by`) from
    the errors of the dataset's model on the rows that it predicted (the
    test rows, or every row if the data is not split). It adds a table to
    the dataset's `tables` under its name, with a row for each assumption:
    whether the data meets it, the statistic of its test, and what was
    found. The rows are also added to the "assumptions" table, with those of
    the pretests.

    An assumption that the model makes (one in its `assumptions`) and that
    the data does not meet is a violation, which the dataset's
    `assumption_violations` decides what to do about: "raise" an
    `AssumptionError`, "warn" (the default), or "ignore" it. With "skip", a
    branch of an experiment that audits its own model is left out, but a
    critic after an experiment (which sees only the winning branch) warns.
    An audit also tests an assumption that the model does not make, but
    that is never a violation.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used. Defaults to `None`.
        parameters: keyword arguments for `apply`. Defaults to an empty
            `dict`.

    """

    """ Public Methods """

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Checks the assumptions of the dataset's model from its errors.

        Args:
            item: the dataset, with a fitted model and its predictions.
            **kwargs: not used.

        Raises:
            AssumptionError: if the model makes an assumption that the data
                does not meet, and the dataset's `assumption_violations` is
                "raise".
            SkippedBranchError: if it does so in a branch of an experiment,
                and `assumption_violations` is "skip".
            ValueError: if there is no fitted model with predictions.

        Returns:
            The dataset, with the table of verdicts.

        """
        if item.model is None or item.predictions is None:
            message = (
                f'{self.name!r} needs a fitted model and its predictions: '
                f'apply a model first')
            raise ValueError(message)
        technique = _model_technique(item)
        verdicts = conditions.judge_after(
            item, technique, check = self._check())
        item.tables[self.name] = conditions.tabulate(verdicts)
        conditions.record(
            item, verdicts, technique = self.name, role = 'audits')
        conditions.enforce(item, verdicts)
        return item

    """ Private Methods """

    def _check(self) -> str | None:
        """Returns the name of the audit whose assumptions this checks.

        Returns:
            The snake case name of the class (such as "audit_residuals"),
                whatever the technique is named in a workflow, or `None` to
                check every assumption of the model that has an audit.

        """
        return str(camina.snakify(type(self).__name__))


""" Private Functions """


def _model_technique(item: base.Dataset) -> type[Any] | None:
    """Returns the technique that fitted the dataset's model.

    Args:
        item: the dataset.

    Returns:
        The class of the model in the library, or `None` if the model's
            technique is not known (or not in the library).

    """
    name = item.model_name
    if name is None:
        return None
    try:
        kind: type[Any] = chrisjen.library.borrow(name)
    except (KeyError, ValueError):
        return None
    return kind
