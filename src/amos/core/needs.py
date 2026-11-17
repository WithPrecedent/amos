"""What techniques need from the data, and the tests of each need.

Each need is a subclass of `conditions.Need`, named in snake case, which a
technique names in its `needs`. Most needs are `hard`: the technique cannot
be applied without them (see `conditions.enforce`). The needs for a number
of rows or features (`min_rows`, `max_rows`, and `max_features`) are not
hard, and read their limit from the attribute of the same name of the
technique (such as `max_rows = 10_000`).

Contents:
    NoMissingValues: no feature has a missing value.
    NumericFeatures: every feature is a number or a boolean.
    MoreRowsThanFeatures: there are more training rows than features.
    MinRows: there are at least the technique's `min_rows` training rows.
    MaxRows: there are at most the technique's `max_rows` training rows.
    MaxFeatures: there are at most the technique's `max_features` features.
    BinaryLabel: the label has two classes.
    MulticlassLabel: the label has three or more classes.
    OrdinalLabel: the label has three or more ordered classes.
    CountLabel: the label is counts.

"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import pandas as pd

from . import base, conditions

""" Needs of the Features """


@dataclasses.dataclass(frozen = True)
class NoMissingValues(conditions.Need):
    """No feature has a missing value (impute them first)."""

    description: ClassVar[str] = 'no feature has a missing value'
    pretested_by: ClassVar[str] = 'check_features'

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether any feature has a missing value.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        passed = not found.missing
        detail = (
            'has no missing values in the features' if passed else
            'cannot use missing values (impute them first)')
        return self._verdict(technique, self.pretested_by, passed, detail)


@dataclasses.dataclass(frozen = True)
class NumericFeatures(conditions.Need):
    """Every feature is a number or a boolean (encode the others first).

    Categorical features (the `pandas` "category" type), text, and dates
    are not.

    """

    description: ClassVar[str] = 'every feature is a number or a boolean'
    pretested_by: ClassVar[str] = 'check_features'

    def pretest(
        self,
        item: base.Dataset,
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether any feature is categorical, text, or a date.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        if found.text:
            columns = _unusable(item, categories = False)
            return self._verdict(
                technique, self.pretested_by, passed = False,
                detail = (
                    f'cannot use the text or dates in {columns}: encode them '
                    f'(with an encoder such as one_hot) or remove them (with '
                    f'drop_columns) first'))
        if found.categories:
            columns = _unusable(item, categories = True)
            return self._verdict(
                technique, self.pretested_by, passed = False,
                detail = (
                    f'cannot use the categorical features {columns} (encode '
                    f'them first)'))
        return self._verdict(
            technique, self.pretested_by, passed = True,
            detail = 'has only numeric and boolean features')


@dataclasses.dataclass(frozen = True)
class MoreRowsThanFeatures(conditions.Need):
    """There are more training rows than features."""

    description: ClassVar[str] = 'there are more training rows than features'
    pretested_by: ClassVar[str] = 'check_sample_size'

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether there are more training rows than features.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        passed = not found.wide
        counts = f'{found.rows} rows and {found.features} features'
        detail = (
            f'has more rows than features ({counts})' if passed else
            f'needs more rows than features ({counts})')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = found.rows, threshold = found.features)


@dataclasses.dataclass(frozen = True)
class MinRows(conditions.Need):
    """There are at least the technique's `min_rows` training rows.

    A technique with fewer rows than this learns poorly from them.

    """

    description: ClassVar[str] = (
        'there are at least as many training rows as it learns well from')
    pretested_by: ClassVar[str] = 'check_sample_size'
    affects: ClassVar[str] = 'prediction'
    hard: ClassVar[bool] = False

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether there are enough training rows for the technique.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        limit = getattr(technique, 'min_rows', None)
        if not limit:
            return self._verdict(
                technique, self.pretested_by, None,
                'needs a technique with "min_rows" to test')
        passed = found.rows >= limit
        detail = (
            f'has at least {limit} rows ({found.rows})' if passed else
            f'needs at least {limit} rows, not {found.rows}')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = found.rows, threshold = limit)


@dataclasses.dataclass(frozen = True)
class MaxRows(conditions.Need):
    """There are at most the technique's `max_rows` training rows.

    A technique with more rows than this is too slow to fit.

    """

    description: ClassVar[str] = (
        'there are no more training rows than it is fast enough for')
    pretested_by: ClassVar[str] = 'check_sample_size'
    affects: ClassVar[str] = 'prediction'
    hard: ClassVar[bool] = False

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether there are too many training rows for the technique.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        limit = getattr(technique, 'max_rows', None)
        if limit is None:
            return self._verdict(
                technique, self.pretested_by, None,
                'needs a technique with "max_rows" to test')
        passed = found.rows <= limit
        detail = (
            f'has at most {limit} rows ({found.rows})' if passed else
            f'is too slow for more than {limit} rows')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = found.rows, threshold = limit)


@dataclasses.dataclass(frozen = True)
class MaxFeatures(conditions.Need):
    """There are at most the technique's `max_features` features.

    A technique with more features than this works poorly (as distances do,
    for example, in many dimensions).

    """

    description: ClassVar[str] = (
        'there are no more features than it works well with')
    pretested_by: ClassVar[str] = 'check_sample_size'
    affects: ClassVar[str] = 'prediction'
    hard: ClassVar[bool] = False

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether there are too many features for the technique.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        limit = getattr(technique, 'max_features', None)
        if limit is None:
            return self._verdict(
                technique, self.pretested_by, None,
                'needs a technique with "max_features" to test')
        passed = found.features <= limit
        detail = (
            f'has at most {limit} features ({found.features})' if passed else
            f'works poorly with more than {limit} features')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = found.features, threshold = limit)


""" Needs of the Label """


@dataclasses.dataclass(frozen = True)
class BinaryLabel(conditions.Need):
    """The label has two classes."""

    description: ClassVar[str] = 'the label has two classes'
    pretested_by: ClassVar[str] = 'check_label'

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether a classified label has two classes.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        if found.task != 'classify':
            return self._verdict(
                technique, self.pretested_by, None,
                'is only tested for a classified label')
        classes = found.classes or 0
        passed = classes <= 2  # noqa: PLR2004
        detail = (
            f'has a label of {classes} classes' if passed else
            f'is for 2 classes, not {classes}')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = classes, threshold = 2)


@dataclasses.dataclass(frozen = True)
class MulticlassLabel(conditions.Need):
    """The label has three or more classes."""

    description: ClassVar[str] = 'the label has three or more classes'
    pretested_by: ClassVar[str] = 'check_label'
    hard: ClassVar[bool] = False

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether a classified label has three or more classes.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        if found.task != 'classify':
            return self._verdict(
                technique, self.pretested_by, None,
                'is only tested for a classified label')
        classes = found.classes or 0
        passed = classes >= 3  # noqa: PLR2004
        detail = (
            f'has a label of {classes} classes' if passed else
            f'is for 3 or more classes, not {classes}')
        return self._verdict(
            technique, self.pretested_by, passed, detail,
            statistic = classes, threshold = 3)


@dataclasses.dataclass(frozen = True)
class OrdinalLabel(conditions.Need):
    """The label has three or more ordered classes.

    The classes are ordered if they are whole numbers or an ordered
    `pandas` "category" type.

    """

    description: ClassVar[str] = 'the label has three or more ordered classes'
    pretested_by: ClassVar[str] = 'check_label'
    hard: ClassVar[bool] = False

    def pretest(
        self,
        item: base.Dataset,
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether the classes of the label are ordered.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        if found.task != 'classify':
            return self._verdict(
                technique, self.pretested_by, None,
                'is only tested for a classified label')
        dtype = item.y.dtype
        ordered = isinstance(dtype, pd.CategoricalDtype) and bool(
            dtype.ordered)
        passed = bool(found.ordinal) or (
            ordered and (found.classes or 0) >= 3)  # noqa: PLR2004
        detail = (
            f'has {found.classes} ordered classes' if passed else
            'is for three or more ordered classes')
        return self._verdict(technique, self.pretested_by, passed, detail)


@dataclasses.dataclass(frozen = True)
class CountLabel(conditions.Need):
    """The label is counts: whole numbers of 0 or more."""

    description: ClassVar[str] = 'the label is counts'
    pretested_by: ClassVar[str] = 'check_label'

    def pretest(
        self,
        item: base.Dataset,  # noqa: ARG002
        technique: Any,
        found: Any,
        cache: dict[str, Any]) -> conditions.Verdict:  # noqa: ARG002
        """Tests whether a regressed label is counts.

        Args:
            item: the dataset.
            technique: the technique, or `None`.
            found: the profile of the dataset.
            cache: results shared by the conditions checked together.

        Returns:
            The verdict.

        """
        if found.task != 'regress':
            return self._verdict(
                technique, self.pretested_by, None,
                'is only tested for a regressed label')
        passed = bool(found.counts)
        detail = (
            'has a label of counts' if passed else
            'needs counts (whole numbers of 0 or more) as its label')
        return self._verdict(technique, self.pretested_by, passed, detail)


""" Private Functions """


def _unusable(item: base.Dataset, *, categories: bool) -> list[str]:
    """Returns the features that a model that needs numbers cannot use.

    Args:
        item: the dataset.
        categories: whether to return the categorical features (the
            `pandas` "category" type), or else the text and dates.

    Returns:
        The names of the features.

    """
    names = []
    for column in item.features:
        values = item.data[column]
        kind = base._kind_of(values)
        category = isinstance(values.dtype, pd.CategoricalDtype)
        text = kind == 'dates' or (kind == 'categoricals' and not category)
        if (category if categories else text):
            names.append(column)
    return names
