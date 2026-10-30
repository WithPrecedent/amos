"""Techniques that choose which features a model uses.

These are techniques of the "analyst" stage, usually in a step of their own
(such as "select") after the features are encoded and scaled. Each one keeps
some of the features and removes the rest. Most are transformers (see the
`transformers` module), so they learn which features to keep from the
training rows only, choose from the numeric and boolean features by default,
and can be limited to some `columns`.

`feature_group` instead keeps the features that the settings list, so an
experiment can compare groups of features that you chose. Name each group
in a "feature_groups" section of the settings, and then name the groups as
the techniques of a step:

```ini
[analyst]
steps = split, select, model
select_techniques = demographics, case_facts
model_techniques = logit

[feature_groups]
demographics = age, sex, race
case_facts = court, year, claims
```

Contents:
    Selector: genre of techniques that keep some of the features.
    ExhaustiveSelection: keeps the best of every set of 1 to 3 features.
    FeatureGroup: keeps the features that the settings list.
    GenericUnivariate: keeps the features most related to the label, by a
        mode that the parameters choose.
    KBest: keeps the k features most related to the label.
    RecursiveElimination: removes the least important feature of a model,
        again and again.
    RecursiveEliminationCV: removes features as `recursive_elimination`
        does and keeps the number that scores best.
    SelectFdr: keeps the features related to the label, controlling the
        false discovery rate.
    SelectFpr: keeps the features related to the label, controlling the
        false positive rate.
    SelectFromModel: keeps the features that a model finds most important.
    SelectFwe: keeps the features related to the label, controlling the
        family-wise error rate.
    SelectPercentile: keeps a share of the features most related to the
        label.
    SequentialSelection: adds (or removes) features with mlxtend while the
        score improves.
    SequentialSelectionSK: adds (or removes) features with scikit-learn
        while the score improves.
    VarianceThreshold: removes features whose variance is at or below a
        threshold.
    register_feature_groups: adds the feature groups of the settings to the
        library.
    _SECTION: name of the section of the settings with the feature groups.

"""

from __future__ import annotations

import abc
import dataclasses
import importlib
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

import camina
import chrisjen
import numpy as np
import pandas as pd

from . import base, models, transformers, utilities

# Name of the section of the settings that lists the features of each group.
# It is not a worker, so it is one of `chrisjen`'s special sections.
_SECTION: str = 'feature_groups'
if _SECTION not in chrisjen.options._SPECIAL_SETTINGS:
    chrisjen.options._SPECIAL_SETTINGS.append(_SECTION)


@dataclasses.dataclass
class Selector(transformers.Transformer, abc.ABC):
    """Genre of techniques that keep some of the features.

    A selector chooses from the numeric and boolean features by default, and
    the features it does not choose from are kept. A selector that would keep
    none of the features raises an error instead.

    """

    kinds: ClassVar[tuple[str, ...]] = ('numerics', 'booleans')

    """ Private Methods """

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Fits the selector, checking that it keeps a feature.

        Args:
            tool: the built selector.
            x: the training rows of the columns to choose from.
            y: the training labels, or `None` if there is no label.

        Raises:
            ValueError: if the selector keeps none of the features.

        """
        super()._fit(tool, x, y)
        support = getattr(tool, 'get_support', None)
        if callable(support) and not np.asarray(support()).any():
            message = (
                f'{self.name!r} kept none of the {x.shape[1]} features: '
                f'loosen its parameters (such as "alpha" or "threshold")'
            )
            raise ValueError(message)


class _ModelBased:
    """Makes a selector one that chooses features with a model.

    The model is "estimator": the name of a model in the library (such as
    "random_forest"), which is built for the task with its default
    parameters, or an estimator, its class, or its import path. By default,
    it is a logistic regression (to classify) or a linear regression, so
    scale the features first. The model must report the importance of each
    feature (as `coef_` or `feature_importances_`) unless the selector
    scores sets of features instead.

    It is a plain class rather than a `Selector`, so that the library does
    not store it as a technique.

    """

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the model.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with "estimator".

        """
        return {
            **parameters,
            'estimator': _build_estimator(item, parameters.get('estimator'))}


class _ModelSearch(_ModelBased):
    """Makes a selector one that scores sets of features by cross-validation.

    The selector fits a model (see `_ModelBased`) to sets of the features, in
    cross-validation on the training rows, and keeps the set with the best
    score. The score is "scoring" (the name of a scikit-learn scorer), or by
    default accuracy or R². The folds ("cv", 5 by default) keep the rows of
    each of the dataset's first groups together, as a model's search does
    (see `models.search_folds`).

    """

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the model, scorer, and folds of the search.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with "estimator", "scoring", and "cv".

        """
        scoring = parameters.get('scoring') or (
            'accuracy' if item.task == 'classify' else 'r2')
        cv = models.search_folds(
            item, item.x_train, item.y_train, parameters.get('cv', 5))
        return {
            **super()._prepare(item, parameters),
            'scoring': scoring,
            'cv': cv}


class _MlxtendSearch(_ModelSearch):
    """Makes a selector one of mlxtend's searches for the best features.

    mlxtend's selectors return an array rather than a `DataFrame`, and name
    the features they keep in their own attributes.

    """

    """ Private Methods """

    def _to_frame(
        self,
        values: Any,
        tool: Any,
        columns: list[str],  # noqa: ARG002
        index: pd.Index) -> pd.DataFrame:
        """Returns the selected features as a `DataFrame`.

        Args:
            values: the output of the selector's `transform` method.
            tool: the fitted selector.
            columns: not used.
            index: index of the data.

        Returns:
            The selected features, with `index` and their names.

        """
        names = getattr(tool, 'k_feature_names_', None)
        if names is None:
            names = tool.best_feature_names_
        # mlxtend returns an array, which is of objects if the features were
        # of several types, so each column is given back its type.
        return pd.DataFrame(
            np.asarray(values), index = index,
            columns = [str(n) for n in names]).infer_objects()


@dataclasses.dataclass
class ExhaustiveSelection(_MlxtendSearch, Selector):
    """Keeps the best of every set of 1 to 3 features (by cross-validation).

    mlxtend's exhaustive search tries every set of `min_features` to
    `max_features` features (1 to 3 by default), so it is slow for many
    features. See `_ModelBased` and `_ModelSearch` for the model, score, and
    folds. The score of each set is a table in the dataset's `tables` as
    "{name}_search".

    Wraps:
        [`mlxtend.feature_selection.ExhaustiveFeatureSelector`](https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/)
        from mlxtend.

    """

    contents: str = 'mlxtend.feature_selection.ExhaustiveFeatureSelector'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {
            'min_features': 1, 'max_features': 3, 'print_progress': False})

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> base.Dataset:
        """Keeps the best set of features, and adds the score of each set.

        Args:
            item: the dataset to transform.
            columns: names of the columns to choose from. Defaults to `None`,
                which uses `select`.
            **kwargs: parameters for the selector.

        Returns:
            The transformed dataset, with the table of the search.

        """
        super().apply(item, columns = columns, **kwargs)
        _add_search(item, self.name)
        return item

    """ Private Methods """

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Tries no more features than there are.

        Args:
            tool: the built selector.
            x: the training rows of the columns to choose from.
            y: the training labels.

        """
        most = min(tool.max_features, x.shape[1])
        tool.set_params(
            max_features = most, min_features = min(tool.min_features, most))
        tool.fit(x, y)


@dataclasses.dataclass
class FeatureGroup(Selector):
    """Keeps the features that the settings list, and removes the others.

    The features are "columns": by default, those that the "feature_groups"
    section of the settings lists for the technique's name (see
    `register_feature_groups`). A listed name that is not a feature keeps the
    features made from it by an encoder or other transformer, whose names
    start with it and an underscore (so "court" keeps "court_federal" and
    "court_state" after `one_hot`). The label and groups are not features, so
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
        if name in groups:
            technique.parameters.setdefault('columns', groups[name])
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
        item.record(
            self.name,
            listed = names,
            kept = [c for c in item.features if c in chosen],
            removed = removed)
        return item


@dataclasses.dataclass
class GenericUnivariate(Selector):
    """Keeps the features most related to the label, by a mode it is given.

    "mode" is "percentile" (the default), "k_best", "fpr", "fdr", or "fwe",
    and "param" is the setting of that mode (50 by default, so it keeps half
    of the features). This lets an experiment search the mode as a parameter.
    The default test is an F test suited to the task (`f_classif` or
    `f_regression`). Set "score_func" to another function or its import path.

    Wraps:
        [`sklearn.feature_selection.GenericUnivariateSelect`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.GenericUnivariateSelect'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'mode': 'percentile', 'param': 50})

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class KBest(Selector):
    """Keeps the k features (10 by default) most related to the label.

    The default test is an F test suited to the task (`f_classif` or
    `f_regression`). Set "score_func" to another function or its import path.

    Wraps:
        [`sklearn.feature_selection.SelectKBest`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SelectKBest'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'k': 10})

    """ Private Methods """

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Keeps every feature if there are no more than k.

        Args:
            tool: the built selector.
            x: the training rows of the columns to choose from.
            y: the training labels, or `None` if there is no label.

        """
        if isinstance(tool.k, int) and tool.k > x.shape[1]:
            tool.set_params(k = 'all')
        super()._fit(tool, x, y)

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class RecursiveElimination(_ModelBased, Selector):
    """Removes the least important feature of a model, again and again.

    It fits the model to the features, removes the "step" least important
    (1 by default), and fits it again, until `n_features_to_select` are left
    (half of the features by default). See `_ModelBased` for the model.

    Wraps:
        [`sklearn.feature_selection.RFE`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.RFE'

    """ Private Methods """

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Keeps every feature if there are no more than it would keep.

        Args:
            tool: the built selector.
            x: the training rows of the columns to choose from.
            y: the training labels.

        """
        wanted = tool.n_features_to_select
        if isinstance(wanted, int) and wanted > x.shape[1]:
            tool.set_params(n_features_to_select = x.shape[1])
        super()._fit(tool, x, y)


@dataclasses.dataclass
class RecursiveEliminationCV(_ModelSearch, Selector):
    """Removes features as `recursive_elimination` does, keeping the best.

    It removes the least important feature of the model again and again, in
    cross-validation on the training rows, and keeps the number of features
    that scored best (at least `min_features_to_select`, 1 by default). See
    `_ModelBased` and `_ModelSearch` for the model, score, and folds. The
    score of each number of features is a table in the dataset's `tables` as
    "{name}_search", with a row for each number ("size") and the mean and
    standard deviation of its scores ("score" and "std"), best first.

    Wraps:
        [`sklearn.feature_selection.RFECV`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.RFECV'

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> base.Dataset:
        """Keeps the best number of features, and adds the score of each.

        Args:
            item: the dataset to transform.
            columns: names of the columns to choose from. Defaults to `None`,
                which uses `select`.
            **kwargs: parameters for the selector.

        Returns:
            The transformed dataset, with the table of the search.

        """
        super().apply(item, columns = columns, **kwargs)
        tool = item.fitted.get(self.name)
        if tool is not None:
            results = tool.cv_results_
            item.tables[f'{self.name}_search'] = pd.DataFrame({
                'size': np.asarray(results['n_features'], dtype = int),
                'score': np.asarray(results['mean_test_score'], dtype = float),
                'std': np.asarray(results['std_test_score'], dtype = float),
                }).sort_values(
                    'score', ascending = False, kind = 'stable',
                    ignore_index = True)
        return item


@dataclasses.dataclass
class SelectFdr(Selector):
    """Keeps the features related to the label, controlling false discovery.

    It keeps the features whose p-values pass the Benjamini-Hochberg
    procedure for a false discovery rate of "alpha" (0.05 by default). The
    default test is an F test suited to the task (`f_classif` or
    `f_regression`). Set "score_func" to another function or its import path.

    Wraps:
        [`sklearn.feature_selection.SelectFdr`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SelectFdr'

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class SelectFpr(Selector):
    """Keeps the features related to the label, controlling false positives.

    It keeps the features whose p-values are below "alpha" (0.05 by
    default). The default test is an F test suited to the task
    (`f_classif` or `f_regression`). Set "score_func" to another function or
    its import path.

    Wraps:
        [`sklearn.feature_selection.SelectFpr`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SelectFpr'

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class SelectFromModel(_ModelBased, Selector):
    """Keeps the features that a model finds most important.

    It fits the model and keeps the features whose importance (the size of
    their coefficients, or `feature_importances_`) is at least "threshold"
    (by default, the mean importance). See `_ModelBased` for the model.

    Wraps:
        [`sklearn.feature_selection.SelectFromModel`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SelectFromModel'


@dataclasses.dataclass
class SelectFwe(Selector):
    """Keeps the features related to the label, controlling family-wise error.

    It keeps the features whose p-values pass a Bonferroni correction for a
    family-wise error rate of "alpha" (0.05 by default). The default test is
    an F test suited to the task (`f_classif` or `f_regression`). Set
    "score_func" to another function or its import path.

    Wraps:
        [`sklearn.feature_selection.SelectFwe`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SelectFwe'

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class SelectPercentile(Selector):
    """Keeps the features (50% by default) most related to the label.

    The default test is an F test suited to the task (`f_classif` or
    `f_regression`). Set "score_func" to another function or its import path.

    Wraps:
        [`sklearn.feature_selection.SelectPercentile`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SelectPercentile'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'percentile': 50})

    """ Private Methods """

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the scoring function for the task.

        Args:
            item: the dataset to transform.
            parameters: parameters for the selector.

        Returns:
            The parameters, with a "score_func".

        """
        return _with_score_function(item, parameters)


@dataclasses.dataclass
class SequentialSelection(_MlxtendSearch, Selector):
    """Adds (or removes) features one at a time while the score improves.

    mlxtend's sequential search starts with no features and adds the one
    that most improves the score of the model, again and again (or, if
    "forward" is false, starts with every feature and removes them). If
    "floating" is true, it also tries to remove (or add back) a feature after
    each step. It keeps `k_features` features, or by default ("best") the
    number with the best score. See `_ModelBased` and `_ModelSearch` for the
    model, score, and folds. The score of the set of each size is a table in
    the dataset's `tables` as "{name}_search". `sequential_selection_sk`
    does much the same without mlxtend.

    Wraps:
        [`mlxtend.feature_selection.SequentialFeatureSelector`](https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/)
        from mlxtend.

    """

    contents: str = 'mlxtend.feature_selection.SequentialFeatureSelector'
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'k_features': 'best'})

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        columns: Sequence[str] | None = None,
        **kwargs: Any) -> base.Dataset:
        """Keeps the features that were chosen, and adds the score of each set.

        Args:
            item: the dataset to transform.
            columns: names of the columns to choose from. Defaults to `None`,
                which uses `select`.
            **kwargs: parameters for the selector.

        Returns:
            The transformed dataset, with the table of the search.

        """
        super().apply(item, columns = columns, **kwargs)
        _add_search(item, self.name)
        return item

    """ Private Methods """

    def _fit(
        self,
        tool: Any,
        x: pd.DataFrame,
        y: pd.Series | None) -> None:
        """Keeps every feature if there are no more than `k_features`.

        Args:
            tool: the built selector.
            x: the training rows of the columns to choose from.
            y: the training labels.

        """
        if isinstance(tool.k_features, int) and tool.k_features > x.shape[1]:
            tool.set_params(k_features = x.shape[1])
        tool.fit(x, y)


@dataclasses.dataclass
class SequentialSelectionSK(_ModelSearch, Selector):
    """Adds (or removes) features one at a time until it keeps enough.

    scikit-learn's sequential search adds the feature that most improves the
    score of the model, again and again (or, if "direction" is "backward",
    removes the feature whose removal least lowers it). It keeps
    `n_features_to_select` features, or by default ("auto") half of them, or
    if "tol" is set, stops when the score improves by less than "tol". See
    `_ModelBased` and `_ModelSearch` for the model, score, and folds.
    Unlike `sequential_selection`, it needs no optional package, but it does
    not report the score of each set.

    Wraps:
        [`sklearn.feature_selection.SequentialFeatureSelector`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.SequentialFeatureSelector'


@dataclasses.dataclass
class VarianceThreshold(Selector):
    """Removes features whose variance is at or below a threshold (0).

    Wraps:
        [`sklearn.feature_selection.VarianceThreshold`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)
        from scikit-learn.

    """

    contents: str = 'sklearn.feature_selection.VarianceThreshold'


""" Public Functions """


def register_feature_groups(
    idea: Mapping[str, Any],
    library: chrisjen.Library | None = None) -> None:
    """Adds the feature groups of the settings to the library.

    Each name in the "feature_groups" section of the settings is stored in the
    library as `FeatureGroup`, so it can be named as a technique (usually of a
    step such as "select"), and that technique keeps the features listed for
    it. The `Project` calls this when it builds its workflow.

    Args:
        idea: the settings of a project.
        library: the library to add the groups to. Defaults to `None`, which
            uses `chrisjen.library`.

    Raises:
        TypeError: if the "feature_groups" section is not a section, or a
            group is not a name or a list of names.
        ValueError: if a group lists no features, or has the name of another
            class in the library.

    """
    library = chrisjen.library if library is None else library
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


def _add_search(item: base.Dataset, name: str) -> None:
    """Adds the score of each set of features that an mlxtend search tried.

    The table is "{name}_search" in the dataset's `tables`, with a row for
    each set, best first: its features (joined by commas), their number
    ("size"), and the mean and standard deviation of its scores in
    cross-validation ("score" and "std").

    Args:
        item: the dataset with the fitted selector.
        name: name of the selector.

    """
    tool = item.fitted.get(name)
    if tool is None:
        return
    rows = []
    for subset in tool.get_metric_dict().values():
        names = [str(n) for n in subset['feature_names']]
        rows.append({
            'features': ', '.join(names),
            'size': len(names),
            'score': float(subset['avg_score']),
            'std': float(subset['std_dev'])})
    item.tables[f'{name}_search'] = pd.DataFrame(rows).sort_values(
        'score', ascending = False, kind = 'stable', ignore_index = True)


def _build_estimator(item: base.Dataset, estimator: Any) -> Any:
    """Returns the model that a selector chooses features with.

    Args:
        item: the dataset to transform.
        estimator: the name of a model in the library, an estimator, its
            class, its import path, or `None` for a logistic regression (to
            classify) or a linear regression.

    Raises:
        ValueError: if `estimator` names a model in the library that cannot
            do the dataset's task.

    Returns:
        The built (and unfitted) estimator, with the dataset's seed as its
            `random_state` if it has one that is not set.

    """
    if estimator is None:
        linear_model = importlib.import_module('sklearn.linear_model')
        estimator = (
            linear_model.LogisticRegression(max_iter = 1000)
            if item.task == 'classify'
            else linear_model.LinearRegression())
    elif isinstance(estimator, str) and _is_model(estimator):
        kind = cast(
            'type[models.Model]',
            chrisjen.library.borrow(estimator, genre = 'model'))
        technique = kind()
        task = technique._check_task(item)
        estimator = technique._make_tool(
            item, technique._keywords(), tool = technique._choose_tool(task))
    else:
        estimator = utilities.import_tool(estimator)
        if isinstance(estimator, type):
            estimator = estimator()
    if item.seed is not None and hasattr(estimator, 'get_params') and (
        estimator.get_params().get('random_state', False) is None):
        estimator.set_params(random_state = item.seed)
    return estimator


def _is_model(name: str) -> bool:
    """Returns whether `name` is the name of a model in the library.

    Args:
        name: a name, or an import path.

    Returns:
        Whether a class in the "model" genre of the library has the name.

    """
    try:
        chrisjen.library.borrow(name, genre = 'model')
    except KeyError:
        return False
    return True


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


def _with_score_function(
    item: base.Dataset,
    parameters: dict[str, Any]) -> dict[str, Any]:
    """Returns `parameters` with a scoring function for feature selection.

    Wraps:
        - [`sklearn.feature_selection.f_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)
          from scikit-learn, by default, to classify.
        - [`sklearn.feature_selection.f_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)
          from scikit-learn, by default, to regress.

    Args:
        item: the dataset.
        parameters: parameters for a feature selector.

    Returns:
        The parameters, with "score_func" imported if it was an import path,
            or chosen for the task if it was not given.

    """
    score = parameters.get('score_func')
    if score is None:
        name = 'f_classif' if item.task == 'classify' else 'f_regression'
        score = f'sklearn.feature_selection.{name}'
    return {**parameters, 'score_func': utilities.import_tool(score)}
