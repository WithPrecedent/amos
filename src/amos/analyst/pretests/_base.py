"""The base class of pretests, and what they share.

Contents:
    Pretest: base class for techniques that check the assumptions and needs
        of techniques before they are applied.

"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Sequence
from typing import Any

import camina
import chrisjen

from ...core import base, conditions


@dataclasses.dataclass
class Pretest(base.Operation, abc.ABC):
    """Base class for techniques that check assumptions before they matter.

    A pretest checks the assumptions and needs (see the `conditions`
    module) that it is named for (each condition's `pretested_by`) on the
    real training rows. It checks them for the `techniques` it is given, or
    else for the techniques of the workflow that is being applied (in a
    `Project`), or else for no technique in particular, which describes the
    data. It adds a table to the dataset's `tables` under its name, with a
    row for each condition of each technique: whether the data meets it,
    the statistic of its test, and what was found.

    A pretest only reports. amos acts on a technique's conditions when the
    technique itself is applied (when the data may have been scaled,
    encoded, or split since the pretest), as the dataset's
    `assumption_violations` says (see `base.Operation._check_conditions`).
    Put a pretest in the "analyst" stage, after the steps that change the
    features (such as "scale" and "encode") and before the models, so that
    it sees the data that they will see.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used. Defaults to `None`.
        parameters: keyword arguments for `apply`. Defaults to an empty
            `dict`.

    """

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        techniques: Sequence[str] | str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Checks the assumptions and needs of techniques on `item`.

        Args:
            item: the dataset that the techniques will be applied to.
            techniques: names of the techniques whose conditions to check.
                Defaults to `None`, which checks those of the techniques of
                the workflow that is being applied, or (if there is none)
                every condition that this pretest checks, for no technique
                in particular.
            **kwargs: not used.

        Raises:
            KeyError: if a technique is not in the library.

        Returns:
            The dataset, with the table of verdicts.

        """
        kinds = _techniques(techniques)
        verdicts = conditions.judge_before(item, kinds, check = self._check())
        item.tables[self.name] = conditions.tabulate(verdicts)
        failed = [
            verdict.describe() for verdict in verdicts if verdict.violated]
        details = {'failed': failed} if failed else {}
        item.record(
            self.name,
            table = self.name,
            techniques = [conditions.name_of(kind) for kind in kinds],
            **details)
        return item

    """ Private Methods """

    def _check(self) -> str | None:
        """Returns the name of the pretest whose conditions this checks.

        Returns:
            The snake case name of the class (such as "check_normality"),
                whatever the technique is named in a workflow, or `None` to
                check every condition that has a pretest.

        """
        return str(camina.snakify(type(self).__name__))


""" Private Functions """


def _techniques(names: Sequence[str] | str | None) -> list[type[Any]]:
    """Returns the techniques whose conditions a pretest checks.

    Args:
        names: names of techniques in the library, or `None` for the
            techniques of the workflow that is being applied.

    Raises:
        KeyError: if a name is not in the library.

    Returns:
        The classes of the techniques that have assumptions or needs,
            without duplicates.

    """
    if names is None:
        found = [kind for _, kind in base._WORKFLOW_TECHNIQUES.get()]
    else:
        found = [chrisjen.library.borrow(n) for n in camina.listify(names)]
    kinds: list[type[Any]] = []
    for kind in found:
        if conditions.conditions_of(kind) and kind not in kinds:
            kinds.append(kind)
    return kinds
