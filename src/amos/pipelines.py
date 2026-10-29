"""Named pipelines: an analyst, a critic, and an artist for a kind of study.

Name a pipeline in the "general" section (or in the project's section), and
the settings of the project's analyst, critic, and artist are filled in for
you, so a whole study can be a few lines:

```ini
[general]
label = target
seed = 43
pipeline = classification
```

Every setting that you write yourself wins, so a pipeline is a starting
point: change one setting (such as "model_techniques" in the "analyst"
section, or a technique's parameters) and keep the rest. If the project's
section does not list its workers, they are the analyst, critic, and artist.
A worker's own section can name another pipeline for that worker. The
settings are filled in when the project is created, so `Project.idea` (and
the "settings.json" that `Project.export` saves) has every one of them.

The steps that depend on the data use the choosers of the `auto` module, so
one pipeline adapts to the data when it is applied: `auto_imputer` fills
missing values (if there are any), `auto_encoder` encodes each categorical
feature by how many levels it has, `auto_sampler` balances rare classes (if
they are rare), `auto_model` chooses among models whose assumptions the data
meets, `auto_evaluator` applies the evaluators that suit the model, and
`auto_plot` draws the figures that suit it. Each records why it chose what
it did in a table of the dataset.

| Pipeline | Task | For |
| --- | --- | --- |
| `classification` | classify | Predicting classes: an experiment that compares `logit`, `random_forest`, `gradient_boosting`, and `baseline` by balanced accuracy, after balancing rare classes. |
| `regression` | regress | Predicting a number: an experiment that compares `ridge`, `random_forest`, `gradient_boosting`, and `baseline` by RMSE. |
| `count_regression` | regress | Predicting counts: an experiment that compares `poisson`, `negative_binomial`, and a random forest and gradient boosting fitted for counts, with `baseline`, by RMSE. |
| `classification_inference` | classify | The effects of the features on classes: a `logit_sm`, `mnlogit`, or `ordinal_regression` model (whichever suits the classes), with its coefficients in a regression table. |
| `regression_inference` | regress | The effects of the features on a number: `ols`, or `robust_regression` if the errors have outliers. |
| `count_inference` | regress | The effects of the features on counts: `poisson`, `negative_binomial` (for overdispersed counts), or `zero_inflated_poisson` (for excess zeros). |
| `panel_classification` | classify | Panel data of classes: a `fixest` logit model with the groups as fixed effects, clustered by the first group. |
| `panel_regression` | regress | Panel data of numbers: a `fixest` model with the groups as fixed effects, clustered by the first group. |

The pipelines that predict scale the numeric features and resample rare
classes, which helps models predict. The pipelines for inference do
neither: they keep the coefficients in the units of the features, keep the
real share of each class (resampling would distort the coefficients and
their standard errors), and encode each categorical feature against a
reference level (its first), so its coefficients are differences from that
level. The panel pipelines need "groups" in the "general" section.

Contents:
    Pipeline: the settings of a named pipeline.
    PIPELINES: the named pipelines, by name. Add a `Pipeline` to it to name
        your own.
    expand: fills in the settings of the pipelines named in a project's
        settings.

"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import chrisjen

# The workers that a pipeline fills in, in the order they are applied.
_STAGES: tuple[str, ...] = ('analyst', 'critic', 'artist')
# Settings of a project's section that list the nodes of its workflow, with
# or without the project's name before them (such as "cancer_workers").
_LISTS: tuple[str, ...] = ('workers', 'steps', 'techniques')


@dataclasses.dataclass(frozen = True)
class Pipeline:
    """The settings of a named pipeline.

    Each stage (`analyst`, `critic`, and `artist`) holds the settings of the
    section of the worker with its name, as they would be written in a
    settings file (with lists as lists).

    Args:
        name: name used to refer to the pipeline in settings.
        task: "classify" or "regress": the task of the label that the
            pipeline is for.
        description: what the pipeline is for.
        analyst: settings of the "analyst" section.
        critic: settings of the "critic" section.
        artist: settings of the "artist" section. Defaults to drawing the
            figures that suit the model with `auto_plot`.
        parameters: parameters of techniques, by the name of the technique
            (or "{technique}_{step}"), as in a "{name}_parameters" section.
            Defaults to an empty `dict`.
        fixed_effects: whether the dataset's groups are the fixed effects of
            `fixest`, with the first group as its cluster. If `True`, the
            "general" section must have "groups". Defaults to `False`.

    """

    name: str
    task: str
    description: str
    analyst: Mapping[str, Any]
    critic: Mapping[str, Any]
    artist: Mapping[str, Any] = dataclasses.field(
        default_factory = lambda: {'techniques': ['auto_plot']})
    parameters: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory = dict)
    fixed_effects: bool = False

    """ Properties """

    @property
    def stages(self) -> dict[str, Mapping[str, Any]]:
        """Returns the settings of each worker, by the name of its section."""
        return {
            'analyst': self.analyst,
            'critic': self.critic,
            'artist': self.artist}

    """ Public Methods """

    def parameters_for(
        self,
        groups: Sequence[str] | str | None) -> dict[str, dict[str, Any]]:
        """Returns the parameters of techniques for a project.

        Args:
            groups: the "groups" of the "general" section of the settings.

        Raises:
            ValueError: if the pipeline uses the groups as fixed effects and
                there are none.

        Returns:
            A copy of `parameters`, with the fixed effects and cluster of
                `fixest` if the pipeline uses the groups as fixed effects.

        """
        parameters = {
            name: dict(copy.deepcopy(values))
            for name, values in self.parameters.items()}
        if self.fixed_effects:
            names = chrisjen.utilities._listify_names(groups)
            if not names:
                message = (
                    f'the {self.name!r} pipeline uses the groups as fixed '
                    f'effects: set "groups" in the "general" section'
                )
                raise ValueError(message)
            fixest = parameters.setdefault('fixest', {})
            fixest.setdefault('fixed_effects', names)
            fixest.setdefault('cluster', names[0])
        return parameters


""" Pipelines """


def _predicting_analyst(
    split: str,
    criterion: str,
    models: Sequence[str],
    *,
    sample: bool = False) -> dict[str, Any]:
    """Returns the settings of an analyst that compares models to predict.

    The analyst is an experiment that fills missing values, scales the
    numeric features, encodes the categorical features, resamples rare
    classes (if `sample` is true), and tries each model, choosing the
    techniques of every step but the model with "auto". Scaling comes before
    encoding, so dummy variables stay 0 or 1 (and `auto_sampler` does not
    make values between them).

    Args:
        split: name of the splitter.
        criterion: name of the metric that compares the models.
        models: names of the models to compare.
        sample: whether to resample rare classes. Defaults to `False`.

    Returns:
        The settings of the "analyst" section.

    """
    steps = ['split', 'impute', 'scale', 'encode', 'model']
    if sample:
        steps.insert(-1, 'sample')
    analyst: dict[str, Any] = {
        'design': 'experiment',
        'criterion': criterion,
        'steps': steps,
        'split_techniques': [split],
        'impute_techniques': ['auto'],
        'scale_techniques': ['standard'],
        'encode_techniques': ['auto'],
        'model_techniques': list(models)}
    if sample:
        analyst['sample_techniques'] = ['auto']
    return analyst


def _explaining_analyst(split: str, model: str) -> dict[str, Any]:
    """Returns the settings of an analyst that fits a model for inference.

    The analyst fills missing values, encodes the categorical features, and
    fits one model, without scaling or resampling (see the module's
    docstring).

    Args:
        split: name of the splitter.
        model: name of the model, or "auto" to choose it with `auto_model`.

    Returns:
        The settings of the "analyst" section.

    """
    return {
        'steps': ['split', 'impute', 'encode', 'model'],
        'split_techniques': [split],
        'impute_techniques': ['auto'],
        'encode_techniques': ['auto'],
        'model_techniques': [model]}


def _critic(validator: str) -> dict[str, Any]:
    """Returns the settings of a critic.

    Args:
        validator: name of the validator that cross-validates the model on
            the training rows.

    Returns:
        The settings of the "critic" section: the validator, and then the
            evaluators that suit the model.

    """
    return {'techniques': [validator, 'auto_evaluator']}


# Parameters for the encoders of the pipelines for inference: a categorical
# feature is encoded against its first level, which keeps its dummy
# variables from adding up to the intercept.
_REFERENCE_LEVEL: dict[str, dict[str, Any]] = {
    'auto_encode': {'drop': 'first'}}
# Trees in a random forest. More than scikit-learn's 100 keeps the forest's
# predictions and importances from depending on the seed.
_FOREST: dict[str, Any] = {'n_estimators': 300}

PIPELINES: dict[str, Pipeline] = {
    pipeline.name: pipeline for pipeline in (
        Pipeline(
            name = 'classification',
            task = 'classify',
            description = (
                'Predicts classes, comparing a logit model, a random forest, '
                'and gradient boosting by balanced accuracy.'),
            analyst = _predicting_analyst(
                'stratified',
                'balanced_accuracy',
                ['logit', 'random_forest', 'gradient_boosting', 'baseline'],
                sample = True),
            critic = _critic('stratified_k_fold'),
            parameters = {'random_forest': _FOREST}),
        Pipeline(
            name = 'regression',
            task = 'regress',
            description = (
                'Predicts a number, comparing ridge regression, a random '
                'forest, and gradient boosting by RMSE.'),
            analyst = _predicting_analyst(
                'train_test',
                'rmse',
                ['ridge', 'random_forest', 'gradient_boosting', 'baseline']),
            critic = _critic('k_fold'),
            parameters = {'random_forest': _FOREST}),
        Pipeline(
            name = 'count_regression',
            task = 'regress',
            description = (
                'Predicts counts, comparing Poisson and negative binomial '
                'regression with a random forest and gradient boosting '
                'fitted for counts, by RMSE.'),
            analyst = _predicting_analyst(
                'train_test',
                'rmse',
                ['poisson', 'negative_binomial', 'random_forest',
                 'gradient_boosting', 'baseline']),
            critic = _critic('k_fold'),
            parameters = {
                'random_forest': {**_FOREST, 'criterion': 'poisson'},
                'gradient_boosting': {'loss': 'poisson'}}),
        Pipeline(
            name = 'classification_inference',
            task = 'classify',
            description = (
                'Estimates the effects of the features on classes with a '
                'logit, multinomial logit, or ordinal model.'),
            analyst = _explaining_analyst('stratified', 'auto'),
            critic = _critic('stratified_k_fold'),
            parameters = {
                **_REFERENCE_LEVEL,
                'auto_model': {
                    'choices': ['ordinal_regression', 'logit_sm', 'mnlogit']}}),
        Pipeline(
            name = 'regression_inference',
            task = 'regress',
            description = (
                'Estimates the effects of the features on a number with '
                'ordinary least squares, or robust regression for errors '
                'with outliers.'),
            analyst = _explaining_analyst('train_test', 'auto'),
            critic = _critic('k_fold'),
            parameters = {
                **_REFERENCE_LEVEL,
                'auto_model': {'choices': ['robust_regression', 'ols']}}),
        Pipeline(
            name = 'count_inference',
            task = 'regress',
            description = (
                'Estimates the effects of the features on counts with the '
                'Poisson, negative binomial, or zero-inflated Poisson model '
                'that suits their dispersion.'),
            analyst = _explaining_analyst('train_test', 'auto'),
            critic = _critic('k_fold'),
            parameters = {
                **_REFERENCE_LEVEL,
                'auto_model': {
                    'choices': [
                        'poisson', 'negative_binomial',
                        'zero_inflated_poisson']}}),
        Pipeline(
            name = 'panel_classification',
            task = 'classify',
            description = (
                'Estimates the effects of the features on classes in panel '
                'data with a fixed effects logit model and standard errors '
                'clustered by group.'),
            analyst = _explaining_analyst('stratified', 'fixest'),
            critic = _critic('stratified_k_fold'),
            parameters = _REFERENCE_LEVEL,
            fixed_effects = True),
        Pipeline(
            name = 'panel_regression',
            task = 'regress',
            description = (
                'Estimates the effects of the features on a number in panel '
                'data with a fixed effects model and standard errors '
                'clustered by group.'),
            analyst = _explaining_analyst('train_test', 'fixest'),
            critic = _critic('k_fold'),
            parameters = _REFERENCE_LEVEL,
            fixed_effects = True))}


""" Public Functions """


def expand(idea: MutableMapping[str, Any], name: str | None = None) -> None:
    """Fills in the settings of the pipelines named in `idea`.

    A pipeline named in the "general" section or the project's section fills
    in the analyst, critic, and artist that the project lists (all three, if
    it lists no workers, and a project section is added if there is none). A
    pipeline named in the section of one of those workers fills in that
    worker instead. Every setting already in `idea` is kept, so calling this
    again changes nothing. The "task" of the "general" section is set to the
    task of the pipelines, if it is not set.

    Sections are replaced rather than changed in place, so a section that
    `idea` shares with other settings is not changed.

    Args:
        idea: the settings of a project, which are changed.
        name: the name of the project. Defaults to `None`, in which case it
            is found as `chrisjen` finds it: from the first section whose
            name ends in "_project", or the first other section that is not
            special, a worker's, or parameters.

    Raises:
        ValueError: if a pipeline is not named in `PIPELINES`, is named in
            the section of a worker other than the analyst, critic, or
            artist, is for another task than the "task" of the "general"
            section or the other pipelines named, or uses the groups as
            fixed effects and the "general" section has none.

    """
    general = dict(idea.get('general', {}))
    project = _project_section(idea, name)
    chosen = _named(idea[project], project) if project is not None else None
    chosen = chosen or _named(general, 'general')
    used: dict[str, Pipeline] = {}
    if chosen is not None:
        if project is None:
            project = f'{name or chosen.name}_project'
            idea[project] = {}
        for stage in _workers(idea, project):
            if stage in _STAGES:
                used[stage] = chosen
    for key, section in list(idea.items()):
        if key in {'general', project} or not isinstance(section, Mapping):
            continue
        own = _named(section, key)
        if own is None:
            continue
        if key not in _STAGES:
            message = (
                f'a pipeline is named in the {key!r} section, but only the '
                f'"general" section, the project\'s section, and the '
                f'sections of the {", ".join(_STAGES)} can name one'
            )
            raise ValueError(message)
        used[key] = own
    if not used:
        return
    for stage, pipeline in used.items():
        idea[stage] = {
            **copy.deepcopy(dict(pipeline.stages[stage])),
            **idea.get(stage, {})}
    pipelines = list({p.name: p for p in used.values()}.values())
    for pipeline in pipelines:
        for technique, values in pipeline.parameters_for(
                general.get('groups')).items():
            key = f'{technique}_parameters'
            idea[key] = {**values, **idea.get(key, {})}
    general['task'] = _task_of(pipelines, general.get('task'))
    idea['general'] = general


""" Private Functions """


def _named(section: Mapping[str, Any], key: str) -> Pipeline | None:
    """Returns the pipeline named in a section of the settings, if any.

    Args:
        section: the section.
        key: the name of the section, for an error message.

    Raises:
        ValueError: if the name is not in `PIPELINES`.

    Returns:
        The pipeline named by the section's "pipeline" setting, or `None` if
            it has none.

    """
    value = section.get('pipeline')
    if value is None:
        return None
    pipeline = PIPELINES.get(str(value).strip())
    if pipeline is None:
        message = (
            f'there is no pipeline named {value!r} (in the {key!r} section): '
            f'the pipelines are {", ".join(PIPELINES)}'
        )
        raise ValueError(message)
    return pipeline


def _project_section(
    idea: Mapping[str, Any],
    name: str | None) -> str | None:
    """Returns the name of the project's section of the settings.

    Args:
        idea: the settings of the project.
        name: the name of the project, or `None` to find it.

    Returns:
        "{name}_project" or `name`, if `name` is given and has a section.
            Otherwise, the first section whose name ends in "_project", or
            the first other section that is not special (such as "general"),
            a worker's that a pipeline fills in, or parameters. `None` if
            there is no such section.

    """
    sections = [
        key for key, value in idea.items() if isinstance(value, Mapping)]
    if name is not None:
        return next(
            (key for key in (f'{name}_project', name) if key in sections),
            None)
    for key in sections:
        if key.endswith('_project'):
            return key
    return next(
        (key for key in sections
         if key not in chrisjen.options._SPECIAL_SETTINGS
         and key not in _STAGES
         and not key.endswith('_parameters')),
        None)


def _task_of(pipelines: Sequence[Pipeline], task: str | None) -> str:
    """Returns the task of the pipelines, checked against the settings.

    Args:
        pipelines: the pipelines named in the settings.
        task: the "task" of the "general" section, or `None`.

    Raises:
        ValueError: if the pipelines are for different tasks, or for another
            task than `task`.

    Returns:
        The task of the pipelines.

    """
    tasks = {pipeline.task for pipeline in pipelines}
    if len(tasks) > 1:
        named = ', '.join(f'{p.name} ({p.task})' for p in pipelines)
        message = f'the pipelines named are for different tasks: {named}'
        raise ValueError(message)
    found = tasks.pop()
    if task is not None and task != found:
        message = (
            f'the {pipelines[0].name!r} pipeline is for a label to {found}, '
            f'but the "general" section sets "task" to {task!r}'
        )
        raise ValueError(message)
    return found


def _workers(idea: MutableMapping[str, Any], project: str) -> list[str]:
    """Returns the workers of the project, listing the stages if it has none.

    Args:
        idea: the settings of the project. If the project's section lists no
            workers, steps, or techniques, it is replaced by one that lists
            the stages as its workers.
        project: the name of the project's section.

    Returns:
        The names that the project's section lists.

    """
    prefix = project.removesuffix('_project')
    section = dict(idea[project])
    for key, value in section.items():
        if key.removeprefix(f'{prefix}_') in _LISTS:
            return chrisjen.utilities._listify_names(value)
    section[f'{prefix}_workers'] = list(_STAGES)
    idea[project] = section
    return list(_STAGES)
