"""A selector that keeps or drops named groups of features.

Contents:
    _SECTION: name of the section of the settings with the feature groups.
    _ALWAYS: name in the feature groups of the features every group keeps.
    _GROUP_TECHNIQUE: names of `FeatureGroup` and its numbered copies.
    FeatureGroup: keeps the features that the settings list.
    register_feature_groups: adds the feature groups of the settings to the
        library.

"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, cast

import camina
import chrisjen
import pandas as pd

from ...core import base
from ._base import Selector

# Name of the section of the settings that lists the features of each group.
# It is not a worker, so it is one of `chrisjen`'s special sections.
_SECTION: str = 'feature_groups'
chrisjen.options.add_special_setting(_SECTION)


# Name in the feature groups of the features that every group keeps. It is
# not a group, so it is not a technique.
_ALWAYS: str = 'always_use'
# Names of `FeatureGroup` (and its numbered copies) in the settings, whose
# parameters can list groups (see `register_feature_groups`).
_GROUP_TECHNIQUE: re.Pattern[str] = re.compile(r'feature_group\d*')


@dataclasses.dataclass
class FeatureGroup(Selector):
    """Keeps the features that the settings list, and removes the others.

    The features are "columns": by default, those that the "feature_groups"
    section of the settings lists for the technique's name (see
    `register_feature_groups`). If that section lists features as "always_use",
    a group built for a project keeps them too, whatever its other features are,
    so every group shares them. A listed name that is not a feature keeps the
    features made from it by an encoder or other transformer, whose names start
    with it and an underscore (so "region" keeps "region_north" and
    "region_south" after `one_hot`). The label and groups are not features, so
    they are always kept and cannot be listed.

    Unlike other selectors, it learns nothing from the data, and it chooses
    from features of every kind.

    """

    """ Class Methods """

    @classmethod
    def build(
        cls,
        name: str,
        project: chrisjen.Project,
        parameters: base.GenericDict | None = None,
        **kwargs: Any) -> FeatureGroup:
        """Builds the technique, with the features of its group.

        Args:
            name: name of the technique, which is the name of its group in
                the "feature_groups" section of the settings, if it has one.
            project: the project the technique belongs to.
            parameters: more parameters, which take precedence over those in
                the settings. Defaults to `None`.
            **kwargs: other arguments for the constructor.

        Returns:
            The built technique.

        """
        technique = cast('FeatureGroup', super().build(
            name, project, parameters = parameters, **kwargs))
        groups = project.idea.get(_SECTION, {})
        if name in groups and name != _ALWAYS:
            technique.parameters.setdefault('columns', groups[name])
        columns = technique.parameters.get('columns')
        if columns is not None and _ALWAYS in groups:
            listed = [str(c) for c in camina.listify(columns)]
            always = [str(c) for c in camina.listify(groups[_ALWAYS])]
            technique.parameters['columns'] = [
                *listed, *(c for c in always if c not in listed)]
        return technique

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        columns: Sequence[str] | str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Keeps the features in `columns`, and removes the others.

        Args:
            item: the dataset to change.
            columns: names of the features to keep. Defaults to `None`, which
                raises an error.
            **kwargs: not used.

        Raises:
            KeyError: if a name in `columns` is not a feature and no feature
                was made from it.
            ValueError: if `columns` is not given.

        Returns:
            The dataset, with only the chosen features.

        """
        if columns is None:
            message = (
                f'{self.name!r} has no features: list them for it in the '
                f'"{_SECTION}" section of the settings, or pass "columns"'
            )
            raise ValueError(message)
        names = [str(c) for c in camina.listify(columns)]
        kept = _match_features(item.features, names)
        missing = [n for n in names if not kept[n]]
        if missing:
            message = (
                f'{self.name!r} lists {missing}, which are not features of '
                f'the data (the label and groups are not features)'
            )
            raise KeyError(message)
        chosen = {f for found in kept.values() for f in found}
        removed = [c for c in item.features if c not in chosen]
        item.update_features(removed, pd.DataFrame(index = item.data.index))
        # A group is often only in a project's own library, so it records its
        # genre for the scorecard (which looks names up in chrisjen.library).
        item.record(
            self.name,
            genre = 'selector',
            listed = names,
            kept = [c for c in item.features if c in chosen],
            removed = removed)
        return item


""" Public Functions """


def register_feature_groups(
    idea: MutableMapping[str, Any],
    library: chrisjen.Library | None = None) -> None:
    """Adds the feature groups of the settings to the library.

    Each name in the "feature_groups" section of the settings is stored in the
    library as `FeatureGroup`, so it can be named as a technique (usually of a
    step such as "select"), and that technique keeps the features listed for
    it. The `Project` calls this when it builds its workflow. "always_use" is
    not a group: its features are kept by every group (see `FeatureGroup`),
    so it is checked but not added.

    Groups can also be listed in the parameters of `feature_group` (or of a
    numbered copy, such as `feature_group2`), as in a "feature_group_parameters"
    section with "raw = a, b" and "refined = c, d". Each step that names that
    technique then names each of its groups in its place, so an experiment
    compares them as branches, and the groups are added to the
    "feature_groups" section. "always_use" in those parameters is kept by
    each of those groups.

    Args:
        idea: the settings of a project.
        library: the library to add the groups to. Defaults to `None`, which
            uses `chrisjen.library`.

    Raises:
        TypeError: if the "feature_groups" section is not a section, or a
            group is not a name or a list of names.
        ValueError: if a group lists no features, has the name of another
            class in the library, or is listed with other features in two
            places.

    """
    library = chrisjen.library if library is None else library
    _expand_parameter_groups(idea)
    if _SECTION not in idea:
        return
    groups = idea[_SECTION]
    if not isinstance(groups, Mapping):
        message = (
            f'the "{_SECTION}" settings must be a section that names each '
            f'group, not {groups!r}'
        )
        raise TypeError(message)
    for name, columns in groups.items():
        if not isinstance(columns, str | Sequence) or not all(
            isinstance(c, str) for c in camina.listify(columns)):
            message = (
                f'the feature group {name!r} must be a column name or a list '
                f'of them, not {columns!r}'
            )
            raise TypeError(message)
        if not camina.listify(columns):
            message = f'the feature group {name!r} lists no features'
            raise ValueError(message)
        if name == _ALWAYS:
            continue
        existing = library.all.get(name)
        if existing is FeatureGroup:
            continue
        if existing is not None:
            message = (
                f'the feature group {name!r} has the name of the '
                f'{existing.__name__} class in the library: name it something '
                f'else'
            )
            raise ValueError(message)
        library.add(FeatureGroup, name = name)


""" Private Functions """


def _add_group(
    idea: MutableMapping[str, Any],
    group: str,
    columns: list[str]) -> None:
    """Adds `group` to the "feature_groups" section of `idea`.

    Args:
        idea: the settings of a project, which are changed in place.
        group: name of the group.
        columns: the features of the group.

    Raises:
        ValueError: if the section already has `group` with other features.

    """
    # `bobbie.Settings.get` raises a `KeyError` for a missing section, so the
    # section is looked for first.
    section = dict(idea[_SECTION] or {}) if _SECTION in idea else {}
    existing = section.get(group)
    if existing is not None and [str(c) for c in camina.listify(existing)] != (
        columns):
        message = (
            f'the feature group {group!r} is listed with other features in '
            f'the "{_SECTION}" section'
        )
        raise ValueError(message)
    section[group] = columns
    idea[_SECTION] = section


def _expand_parameter_groups(idea: MutableMapping[str, Any]) -> None:
    """Names the groups in the parameters of `feature_group` in its place.

    Each setting of techniques (a worker's "techniques" or a step's
    "{step}_techniques") that names `feature_group` (or a numbered copy)
    whose parameters list groups names each group instead, and each group,
    with the "always_use" features of those parameters, is added to the
    "feature_groups" section. Settings that are already expanded are left as
    they are.

    Args:
        idea: the settings of a project, which are changed in place.

    Raises:
        ValueError: if the parameters of a technique list both "columns" and
            groups, or a group is listed with other features elsewhere.

    """
    for section_name in list(idea):
        section = idea[section_name]
        if not isinstance(section, Mapping):
            continue
        changes: dict[str, list[str]] = {}
        for key, value in section.items():
            if key != 'techniques' and not str(key).endswith('_techniques'):
                continue
            names = [str(n) for n in camina.listify(value)]
            expanded: list[str] = []
            for name in names:
                groups = _parameter_groups(idea, name)
                if groups is None:
                    expanded.append(name)
                    continue
                for group, columns in groups.items():
                    _add_group(idea, group, columns)
                    expanded.append(group)
            if expanded != names:
                changes[key] = expanded
        if changes:
            idea[section_name] = {**section, **changes}


def _match_features(
    features: Sequence[str],
    names: Sequence[str]) -> dict[str, list[str]]:
    """Returns the features that each listed name keeps.

    A name keeps the feature with its name, or, if there is none, every
    feature whose name starts with it and an underscore (the features that an
    encoder or other transformer made from it).

    Args:
        features: names of the features of a dataset.
        names: names listed in a feature group.

    Returns:
        The features that each name keeps, in the order of `features`. A
            name that keeps none has an empty `list`.

    """
    matched: dict[str, list[str]] = {}
    for name in names:
        if name in features:
            matched[name] = [name]
        else:
            matched[name] = [f for f in features if f.startswith(f'{name}_')]
    return matched


def _parameter_groups(
    idea: Mapping[str, Any],
    name: str) -> dict[str, list[str]] | None:
    """Returns the groups listed in the parameters of the technique `name`.

    Args:
        idea: the settings of a project.
        name: name of a technique.

    Raises:
        ValueError: if the parameters list both "columns" and groups.

    Returns:
        The features of each group (with the "always_use" features of the
            parameters), by its name, or `None` if `name` is not
            `feature_group` (or a numbered copy) or its parameters list no
            groups.

    """
    if not _GROUP_TECHNIQUE.fullmatch(name):
        return None
    section = f'{name}_parameters'
    # `bobbie.Settings.get` raises a `KeyError` for a missing section.
    parameters = idea[section] if section in idea else None  # noqa: SIM401
    if not isinstance(parameters, Mapping):
        return None
    listed = {
        str(k): [str(c) for c in camina.listify(v)]
        for k, v in parameters.items() if k != 'columns'}
    always = listed.pop(_ALWAYS, [])
    if not listed:
        return None
    if 'columns' in parameters:
        message = (
            f'the parameters of {name!r} list both "columns" and groups '
            f'{sorted(listed)}: list one or the other'
        )
        raise ValueError(message)
    return {
        group: [*columns, *(c for c in always if c not in columns)]
        for group, columns in listed.items()}
