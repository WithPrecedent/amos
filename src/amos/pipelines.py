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

Every model also has a pipeline of its own for each task that it can do,
named for the model and the task, such as `xgboost_classification`,
`xgboost_regression`, or `ols_regression` (for a model whose name ends in
"_regression", "_inference" takes its place, as in `quantile_inference`).
`fixest` has the panel pipelines. Each prepares the data as its model
needs, and no more:

* Features are scaled only for a model that needs it: standardized for a
  penalized linear model (whose penalty depends on the scale of each
  feature) and for a support vector machine, nearest neighbors, or a neural
  network (which measure distances or learn weights), and made more normal
  for naive Bayes (which assumes that they are). Trees and models for
  inference are not scaled.
* Missing values are left for the models that handle them (gradient
  boosting, LightGBM, XGBoost, CatBoost, explainable boosting, and TabPFN),
  and filled for the others.
* Categories are encoded against a reference level for a model with an
  intercept and no penalty, so that their dummy variables are not
  collinear.
* Rare classes are weighted by the models that can weigh them (such as a
  random forest or a logit model), resampled for those that cannot (such as
  nearest neighbors or XGBoost), and kept as they are for models for
  inference, explainable boosting, and TabPFN.
* A model that overfits without tuning has a hyperparameter search (scored
  by balanced accuracy or RMSE), such as a decision tree's depth or the
  penalty of a lasso. Nearest neighbors keep only the 20 features most
  related to the label, and a neural network stops early.

The critic and artist of each apply the evaluators and draw the plots that
suit its model's family (with `auto_evaluator` and `auto_plot`, which leave
out those that do not suit the fitted model, such as SHAP values without
shap installed): SHAP values and partial dependence for trees, a regression
table and the plots that check a model's assumptions (normal errors,
influential rows, calibration) for models for inference, shape functions for
explainable boosting, a learning curve for a neural network, and survival
curves for a Cox model. Models of groups (`gee`, `mixedlm`, and
`binomial_bayes_mixedglm`) need "groups" in the "general" section, and are
tested on groups that they have not seen. The `cox_regression` pipeline
needs "event" in the "cox_parameters" section, and `wls_regression` needs
"weights" in the "wls_parameters" section. See the [technique
catalog](https://WithPrecedent.github.io/amos/catalog/#pipelines) for every
pipeline.

Contents:
    Pipeline: the settings of a named pipeline.
    PIPELINES: the named pipelines, by name. Add a `Pipeline` to it to name
        your own.
    expand: fills in the settings of the pipelines named in a project's
        settings.
    _STAGES: the workers that a pipeline fills in.
    _LISTS: the settings of a project's section that list its workflow.
    _REFERENCE_LEVEL: the parameters that encode categories against a
        reference level.
    _FOREST: the parameters of a random forest of the pipelines.
    _SEARCH_SCORING: the scorer of a hyperparameter search, by task.
    _OUTCOME_EVALUATORS: the evaluators of how well any model predicts.
    _OUTCOME_PLOTS: the plots of how well any model predicts.
    _TREES, _MISSING_AWARE_TREES, _PENALIZED, _LEAST_SQUARES,
        _SCALE_SENSITIVE, _GAUSSIAN, _ADDITIVE, _PRETRAINED, _STATISTICAL,
        _SURVIVAL, _BASELINE: how each family of models is prepared,
        criticized, and drawn.
    _MODEL_PIPELINES: the pipeline of each model for each task.

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
        grouped: whether the model needs the dataset's groups (such as a
            mixed model's random intercepts), so that the "general" section
            must have "groups". Defaults to `False`.
        required: parameters that the settings must give, by the technique
            whose parameters section must have them (such as `{"cox":
            ["event"]}`), because they name columns that only the user
            knows. Defaults to an empty `dict`.
        shared_with: techniques that are given the `required` parameters of
            another technique, by the name of that technique (such as
            `{"cox": ["survival_curves"]}`). Defaults to an empty `dict`.

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
    grouped: bool = False
    required: Mapping[str, Sequence[str]] = dataclasses.field(
        default_factory = dict)
    shared_with: Mapping[str, Sequence[str]] = dataclasses.field(
        default_factory = dict)

    """ Properties """

    @property
    def models(self) -> list[str]:
        """Returns the models that the pipeline fits or chooses among."""
        names = chrisjen.utilities._listify_names(
            self.analyst.get('model_techniques'))
        if names == ['auto']:
            return chrisjen.utilities._listify_names(
                self.parameters.get('auto_model', {}).get('choices'))
        return names

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
            ValueError: if the pipeline uses the groups as fixed effects (or
                its model needs groups) and there are none.

        Returns:
            A copy of `parameters`, with the fixed effects and cluster of
                `fixest` if the pipeline uses the groups as fixed effects.

        """
        parameters = {
            name: dict(copy.deepcopy(values))
            for name, values in self.parameters.items()}
        names = chrisjen.utilities._listify_names(groups)
        if (self.fixed_effects or self.grouped) and not names:
            use = (
                'uses the groups as fixed effects' if self.fixed_effects
                else 'models the rows of each group as correlated')
            message = (
                f'the {self.name!r} pipeline {use}: set "groups" in the '
                f'"general" section'
            )
            raise ValueError(message)
        if self.fixed_effects:
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


""" Pipelines for Each Model """


@dataclasses.dataclass(frozen = True)
class _Family:
    """How the models of a family are prepared, criticized, and drawn.

    Args:
        evaluators: the evaluators that `auto_evaluator` chooses among (those
            that suit the data and the fitted model are applied).
        plots: the plots that `auto_plot` chooses among (those that suit the
            data and the fitted model are drawn).
        impute: whether to fill missing values, for a model that cannot use
            them. Defaults to `True`.
        scale: name of the scaler, for a model that needs features on
            similar scales (or of a similar shape), or `None` for one that
            does not. Defaults to `None`.
        reference: whether to encode each categorical feature against a
            reference level (its first), for a model whose intercept would
            make a full set of dummy variables collinear. Defaults to
            `False`.

    """

    evaluators: tuple[str, ...]
    plots: tuple[str, ...]
    impute: bool = True
    scale: str | None = None
    reference: bool = False


# The scorer of a hyperparameter search, by task. Balanced accuracy is not
# fooled by a model that predicts only the common class, and RMSE is the
# criterion of the other pipelines for regression.
_SEARCH_SCORING: dict[str, str] = {
    'classify': 'balanced_accuracy',
    'regress': 'neg_root_mean_squared_error'}
# The evaluators and plots of how well any model predicts (those that do not
# suit the task are left out by the choosers).
_OUTCOME_EVALUATORS: tuple[str, ...] = (
    'scorecard', 'classification_report', 'confusion')
_OUTCOME_PLOTS: tuple[str, ...] = (
    'confusion_heatmap', 'roc_curve', 'precision_recall_curve',
    'calibration_curve', 'actual_vs_predicted', 'residual_plot')
# Each family of models, with the evaluators and plots that explain it best.
# Trees need no scaling and have their own importances and SHAP values.
_TREES = _Family(
    evaluators = (
        *_OUTCOME_EVALUATORS, 'feature_importance', 'permutation_importance',
        'shap_importance', 'fairness'),
    plots = (
        *_OUTCOME_PLOTS, 'importance_plot', 'partial_dependence',
        'shap_beeswarm', 'tree_plot', 'search_plot', 'fairness_plot'))
# Boosted trees that learn which way to send a missing value at each split,
# so missing values are left for them (filling them would hide whether a
# value was missing, which can itself predict the label).
_MISSING_AWARE_TREES = dataclasses.replace(_TREES, impute = False)
# Linear models with a penalty, which shrinks a coefficient by the scale of
# its feature, so the features are standardized. Their coefficients are their
# importances, and a Q-Q plot checks the normal errors of a regression.
_PENALIZED = _Family(
    evaluators = (
        *_OUTCOME_EVALUATORS, 'feature_importance', 'permutation_importance',
        'fairness'),
    plots = (
        *_OUTCOME_PLOTS, 'qq_plot', 'importance_plot', 'search_plot',
        'fairness_plot'),
    scale = 'standard')
# Least squares without a penalty needs no scaling, but its intercept makes a
# full set of dummy variables collinear.
_LEAST_SQUARES = dataclasses.replace(_PENALIZED, scale = None, reference = True)
# Models of distances, kernels, or weights that are not linear, which need
# standardized features and are explained by shuffling the features and by
# partial dependence.
_SCALE_SENSITIVE = _Family(
    evaluators = (*_OUTCOME_EVALUATORS, 'permutation_importance', 'fairness'),
    plots = (
        *_OUTCOME_PLOTS, 'importance_plot', 'partial_dependence',
        'search_plot', 'fairness_plot'),
    scale = 'standard')
# Gaussian naive Bayes assumes that each feature is normal within each class,
# so the features are made more normal (Yeo-Johnson). Its probabilities are
# often poorly calibrated, which the calibration curve shows.
_GAUSSIAN = _Family(
    evaluators = _SCALE_SENSITIVE.evaluators,
    plots = (*_OUTCOME_PLOTS, 'importance_plot', 'fairness_plot'),
    scale = 'gauss')
# Explainable boosting machines handle missing values and need no scaling,
# and their shape functions show the effect of each feature exactly.
_ADDITIVE = _Family(
    evaluators = (
        *_OUTCOME_EVALUATORS, 'feature_importance', 'permutation_importance',
        'fairness'),
    plots = (
        *_OUTCOME_PLOTS, 'importance_plot', 'shape_functions',
        'fairness_plot'),
    impute = False)
# TabPFN fills, scales, and weighs the features itself, and refitting it
# (as shuffling the features does) is slow.
_PRETRAINED = _Family(
    evaluators = (*_OUTCOME_EVALUATORS, 'fairness'),
    plots = (*_OUTCOME_PLOTS, 'fairness_plot'),
    impute = False)
# Models for inference keep the features in their units and the real share
# of each class (see the module's docstring). Their coefficients go in a
# regression table, and their diagnostics check their assumptions: normal
# errors (Q-Q plot), influential rows, and calibrated probabilities.
_STATISTICAL = _Family(
    evaluators = (*_OUTCOME_EVALUATORS, 'regression_table', 'fairness'),
    plots = (
        *_OUTCOME_PLOTS, 'qq_plot', 'influence_plot', 'coefficient_plot',
        'fairness_plot'),
    reference = True)
# A Cox model's predictions are times that censored rows never reached, so
# its coefficients (hazard ratios) and survival curves describe it instead.
_SURVIVAL = _Family(
    evaluators = ('scorecard', 'regression_table'),
    plots = ('coefficient_plot',),
    reference = True)
# The baseline ignores the features, so only how well it predicts matters.
_BASELINE = _Family(
    evaluators = _OUTCOME_EVALUATORS, plots = _OUTCOME_PLOTS, impute = False)


def _pipeline_name(model: str, task: str) -> str:
    """Returns the name of the pipeline of a model for a task.

    Args:
        model: name of the model.
        task: "classify" or "regress".

    Returns:
        The name of the model followed by "_classification" or
            "_regression", or, for a model whose name ends in "_regression"
            (such as "quantile_regression"), with "_inference" in its place
            (such as "quantile_inference"). A pipeline is never named as a
            technique is, since the project that it names would be.

    """
    if model.endswith('_regression'):
        return f'{model.removesuffix("_regression")}_inference'
    return f'{model}_{"classification" if task == "classify" else "regression"}'


def _model_pipeline(
    model: str,
    task: str,
    description: str,
    family: _Family,
    *,
    parameters: Mapping[str, Mapping[str, Any]] | None = None,
    select: str | None = None,
    sample: bool = False,
    plots: Sequence[str] = (),
    grouped: bool = False,
    required: Mapping[str, Sequence[str]] | None = None,
    shared_with: Mapping[str, Sequence[str]] | None = None) -> Pipeline:
    """Returns the pipeline of one model for one task.

    The analyst splits the data, fills missing values (if the family cannot
    use them), scales the numeric features (if the family needs it), encodes
    the categorical features, selects features (if `select` is given),
    resamples rare classes (if `sample` is true and the label is
    classified), and fits the model. The critic cross-validates the model and
    applies the family's evaluators, and the artist draws the family's plots
    and `plots`. A model with groups is split, and cross-validated, with the
    rows of each group kept together.

    Args:
        model: name of the model.
        task: "classify" or "regress".
        description: what the pipeline is for.
        family: how the model is prepared, criticized, and drawn.
        parameters: parameters of techniques, by the name of the technique.
            If the model's parameters have a "search", its "scoring" is
            balanced accuracy or RMSE (for the task), unless they set one.
            Defaults to `None`.
        select: name of the selector, or `None` to keep every feature.
            Defaults to `None`.
        sample: whether to resample rare classes (with `auto_sampler`), for
            a classifier that cannot weigh the classes itself. Defaults to
            `False`.
        plots: plots to draw after those that `auto_plot` chooses. Defaults
            to an empty tuple.
        grouped: whether the model needs the dataset's groups. Defaults to
            `False`.
        required: as for `Pipeline`. Defaults to `None`.
        shared_with: as for `Pipeline`. Defaults to `None`.

    Returns:
        The pipeline.

    """
    classify = task == 'classify'
    if grouped:
        split = 'group_split'
        validator = 'stratified_group_k_fold' if classify else 'group_k_fold'
    else:
        split = 'stratified' if classify else 'train_test'
        validator = 'stratified_k_fold' if classify else 'k_fold'
    steps = ['split']
    analyst: dict[str, Any] = {'split_techniques': [split]}
    if family.impute:
        steps.append('impute')
        analyst['impute_techniques'] = ['auto']
    if family.scale is not None:
        steps.append('scale')
        analyst['scale_techniques'] = [family.scale]
    steps.append('encode')
    analyst['encode_techniques'] = ['auto']
    if select is not None:
        steps.append('select')
        analyst['select_techniques'] = [select]
    if sample and classify:
        steps.append('sample')
        analyst['sample_techniques'] = ['auto']
    steps.append('model')
    analyst['model_techniques'] = [model]
    chosen: dict[str, dict[str, Any]] = {
        name: dict(values) for name, values in (parameters or {}).items()}
    if 'search' in chosen.get(model, {}):
        chosen[model].setdefault('scoring', _SEARCH_SCORING[task])
    if family.reference:
        chosen.update(copy.deepcopy(_REFERENCE_LEVEL))
    chosen['auto_evaluator'] = {'choices': list(family.evaluators)}
    chosen['auto_plot'] = {'choices': list(family.plots)}
    return Pipeline(
        name = _pipeline_name(model, task),
        task = task,
        description = description,
        analyst = {'steps': steps, **analyst},
        critic = _critic(validator),
        artist = {'techniques': ['auto_plot', *plots]},
        parameters = chosen,
        grouped = grouped,
        required = dict(required or {}),
        shared_with = dict(shared_with or {}))


def _model_pipelines(
    model: str,
    family: _Family,
    descriptions: Mapping[str, str],
    *,
    parameters: Mapping[str, Mapping[str, Any]] | None = None,
    class_weights: Mapping[str, Any] | None = None,
    **options: Any) -> tuple[Pipeline, ...]:
    """Returns a pipeline of a model for each task that it is described for.

    Args:
        model: name of the model.
        family: how the model is prepared, criticized, and drawn.
        descriptions: what each pipeline is for, by its task ("classify" or
            "regress").
        parameters: parameters of techniques, by the name of the technique,
            for every task. Defaults to `None`.
        class_weights: parameters of the model that weigh each class by how
            rare it is, which are added to its parameters to classify.
            Defaults to `None`.
        **options: other arguments of `_model_pipeline` (such as "sample").

    Returns:
        The pipelines, in the order of `descriptions`.

    """
    pipelines = []
    for task, description in descriptions.items():
        chosen = {
            name: dict(values) for name, values in (parameters or {}).items()}
        if task == 'classify' and class_weights:
            chosen[model] = {**chosen.get(model, {}), **class_weights}
        pipelines.append(_model_pipeline(
            model, task, description, family, parameters = chosen, **options))
    return tuple(pipelines)


# A pipeline for each model (and each task it can do). `fixest` has the
# panel pipelines instead.
_MODEL_PIPELINES: tuple[Pipeline, ...] = (
    *_model_pipelines(
        'adaboost', _TREES,
        {'classify': (
            'Classifies with AdaBoost, resampling rare classes. AdaBoost '
            'chases rows that it gets wrong, so noise and outliers sway it: '
            'a grid search tunes its learning rate, which tempers that.'),
         'regress': (
            'Regresses with AdaBoost, which chases rows that it gets wrong '
            '(so outliers sway it), with a grid search of its learning rate '
            'and number of trees.')},
        parameters = {'adaboost': {
            'search': 'grid', 'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.5, 1.0]}},
        sample = True),
    *_model_pipelines(
        'baseline', _BASELINE,
        {'classify': (
            'Predicts the most common class for every row: the score that '
            'every other model must beat, drawn beside the share of each '
            'class.'),
         'regress': (
            'Predicts the mean label for every row: the score that every '
            'other model must beat, drawn beside the distribution of the '
            'label.')},
        plots = ('label_plot',)),
    *_model_pipelines(
        'binomial_bayes_mixedglm', _STATISTICAL,
        {'classify': (
            'Estimates the effects of the features on two classes with a '
            'Bayesian logit model with a random intercept for each group, '
            'tested on groups that it has not seen.')},
        grouped = True),
    *_model_pipelines(
        'catboost', _MISSING_AWARE_TREES,
        {'classify': (
            'Classifies with CatBoost, which handles missing values itself '
            'and whose ordered boosting resists overfitting without a '
            'search, with the classes weighted by how rare they are.'),
         'regress': (
            'Regresses with CatBoost, which handles missing values itself '
            'and whose ordered boosting resists overfitting without a '
            'search.')},
        class_weights = {'auto_class_weights': 'Balanced'}),
    *_model_pipelines(
        'cox', _SURVIVAL,
        {'regress': (
            'Estimates the hazard ratios of the features for the time until '
            'an event with a Cox model, with Kaplan-Meier survival curves. '
            'Set "event" in the "cox_parameters" section.')},
        plots = ('survival_curves',),
        required = {'cox': ['event']},
        shared_with = {'cox': ['survival_curves']}),
    *_model_pipelines(
        'decision_tree', _TREES,
        {'classify': (
            'Classifies with one decision tree, drawn split by split. A '
            'single tree overfits, so a grid search limits its depth and '
            'the size of its leaves; the classes are weighted by rarity.'),
         'regress': (
            'Regresses with one decision tree, drawn split by split. A '
            'single tree overfits, so a grid search limits its depth and '
            'the size of its leaves.')},
        parameters = {'decision_tree': {
            'search': 'grid', 'max_depth': [2, 3, 4, 6, 8],
            'min_samples_leaf': [1, 5, 10, 20]}},
        class_weights = {'class_weight': 'balanced'}),
    *_model_pipelines(
        'elastic_net', _PENALIZED,
        {'regress': (
            'Regresses with both lasso and ridge penalties on standardized '
            'features, which suits correlated features, with a grid search '
            'of the strength and mix of the penalties.')},
        parameters = {'elastic_net': {
            'search': 'grid', 'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9], 'max_iter': 10_000}}),
    *_model_pipelines(
        'explainable_boosting', _ADDITIVE,
        {'classify': (
            'Classifies with an explainable boosting machine, which handles '
            'missing values and needs no scaling, with the shape function '
            'of each feature. Classes are not resampled, so the shapes '
            'show the real odds.'),
         'regress': (
            'Regresses with an explainable boosting machine, which handles '
            'missing values and needs no scaling, with the shape function '
            'of each feature.')}),
    *_model_pipelines(
        'extra_trees', _TREES,
        {'classify': (
            'Classifies with 300 extremely randomized trees, which need no '
            'scaling, weighting the classes by how rare they are.'),
         'regress': (
            'Regresses with 300 extremely randomized trees, which need no '
            'scaling and vary less than a random forest.')},
        parameters = {'extra_trees': {**_FOREST, 'n_jobs': -1}},
        class_weights = {'class_weight': 'balanced'}),
    *_model_pipelines(
        'gee', _STATISTICAL,
        {'classify': (
            'Estimates the average effects of the features on two classes '
            'across groups with generalized estimating equations, which '
            'allow for exchangeable correlation within each group.'),
         'regress': (
            'Estimates the average effects of the features on a number '
            'across groups with generalized estimating equations, which '
            'allow for exchangeable correlation within each group.')},
        parameters = {'gee': {'covariance': 'exchangeable'}},
        grouped = True),
    *_model_pipelines(
        'generalized_poisson', _STATISTICAL,
        {'regress': (
            'Estimates the effects of the features on counts that vary more '
            '(or less) than a Poisson distribution allows, with generalized '
            'Poisson regression.')}),
    *_model_pipelines(
        'glm', _STATISTICAL,
        {'classify': (
            'Estimates the effects of the features on two classes with a '
            'binomial generalized linear model, checking its calibration '
            'and influential rows.'),
         'regress': (
            'Estimates the effects of the features on a number with a '
            'gaussian generalized linear model (set "family" for another), '
            'checking its errors and influential rows.')}),
    *_model_pipelines(
        'gradient_boosting', _MISSING_AWARE_TREES,
        {'classify': (
            'Classifies with histogram gradient boosting, which handles '
            'missing values itself, with a random search of its learning '
            'rate, tree size, and penalty, weighting the classes by '
            'rarity.'),
         'regress': (
            'Regresses with histogram gradient boosting, which handles '
            'missing values itself, with a random search of its learning '
            'rate, tree size, and penalty.')},
        parameters = {'gradient_boosting': {
            'search': 'random', 'n_iter': 10,
            'learning_rate': [0.03, 0.3], 'max_leaf_nodes': [15, 63],
            'min_samples_leaf': [5, 40], 'l2_regularization': [0.0, 1.0]}},
        class_weights = {'class_weight': 'balanced'}),
    *_model_pipelines(
        'knn', _SCALE_SENSITIVE,
        {'classify': (
            'Classifies by the nearest training rows on standardized '
            'features, keeping the 20 most related features (distances '
            'mean little in many dimensions) and resampling rare classes, '
            'with a grid search of the neighbors and their weights.'),
         'regress': (
            'Regresses by the nearest training rows on standardized '
            'features, keeping the 20 most related features (distances '
            'mean little in many dimensions), with a grid search of the '
            'neighbors and their weights.')},
        parameters = {
            'knn': {
                'search': 'grid', 'n_neighbors': [3, 5, 7, 11, 15, 21],
                'weights': ['uniform', 'distance']},
            'k_best': {'k': 20}},
        select = 'k_best',
        sample = True),
    *_model_pipelines(
        'lasso', _PENALIZED,
        {'regress': (
            'Regresses with a lasso penalty on standardized features, which '
            'drops weak features, with a grid search of the penalty.')},
        parameters = {'lasso': {
            'search': 'grid', 'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'max_iter': 10_000}}),
    *_model_pipelines(
        'lightgbm', _MISSING_AWARE_TREES,
        {'classify': (
            'Classifies with LightGBM, which handles missing values itself, '
            'with a random search of its learning rate, leaves, and '
            'penalties, weighting the classes by how rare they are.'),
         'regress': (
            'Regresses with LightGBM, which handles missing values itself, '
            'with a random search of its learning rate, leaves, and '
            'penalties.')},
        parameters = {'lightgbm': {
            'search': 'random', 'n_iter': 10, 'n_estimators': 300,
            'learning_rate': [0.03, 0.3], 'num_leaves': [15, 63],
            'min_child_samples': [5, 40], 'colsample_bytree': [0.6, 1.0],
            'reg_lambda': [0.0, 1.0]}},
        class_weights = {'class_weight': 'balanced'}),
    *_model_pipelines(
        'linear', _LEAST_SQUARES,
        {'regress': (
            'Regresses by ordinary least squares (scikit-learn), encoding '
            'categories against a reference level so the dummy variables '
            'are not collinear, and checking its errors with a Q-Q plot.')}),
    *_model_pipelines(
        'logit', _PENALIZED,
        {'classify': (
            'Classifies with a penalized logit model on standardized '
            'features, weighting the classes by how rare they are, with a '
            'grid search of the penalty.')},
        parameters = {'logit': {
            'search': 'grid', 'C': [0.01, 0.1, 1.0, 10.0, 100.0]}},
        class_weights = {'class_weight': 'balanced'}),
    *_model_pipelines(
        'logit_sm', _STATISTICAL,
        {'classify': (
            'Estimates the effects of the features on two classes with a '
            'logit model, checking that its probabilities are calibrated.')}),
    *_model_pipelines(
        'mixedlm', _STATISTICAL,
        {'regress': (
            'Estimates the effects of the features on a number with a '
            'linear mixed model, with a random intercept for each group, '
            'tested on groups that it has not seen.')},
        grouped = True),
    *_model_pipelines(
        'mnlogit', _STATISTICAL,
        {'classify': (
            'Estimates the effects of the features on three or more classes '
            'with a multinomial logit model, each class compared with the '
            'reference class.')}),
    *_model_pipelines(
        'naive_bayes', _GAUSSIAN,
        {'classify': (
            'Classifies with Gaussian naive Bayes, after making the features '
            'more normal (as it assumes) and resampling rare classes, '
            'checking its probabilities, which are often poorly '
            'calibrated.')},
        sample = True),
    *_model_pipelines(
        'negative_binomial', _STATISTICAL,
        {'regress': (
            'Estimates the effects of the features on counts that vary more '
            'than a Poisson distribution allows (overdispersion), with '
            'negative binomial regression.')}),
    *_model_pipelines(
        'neural_network', _SCALE_SENSITIVE,
        {'classify': (
            'Classifies with a neural network on standardized features, '
            'stopping early to avoid overfitting and resampling rare '
            'classes, with a grid search of its penalty and learning rate '
            'and a learning curve.'),
         'regress': (
            'Regresses with a neural network on standardized features, '
            'stopping early to avoid overfitting, with a grid search of its '
            'penalty and learning rate and a learning curve.')},
        parameters = {'neural_network': {
            'search': 'grid', 'early_stopping': True,
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01]}},
        sample = True,
        plots = ('learning_curve',)),
    *_model_pipelines(
        'ols', _STATISTICAL,
        {'regress': (
            'Estimates the effects of the features on a number by ordinary '
            'least squares, checking its normal errors (Q-Q plot), equal '
            'variance (residuals), and influential rows.')}),
    *_model_pipelines(
        'ordinal_regression', _STATISTICAL,
        {'classify': (
            'Estimates the effects of the features on ordered classes with '
            'an ordered logit model.')}),
    *_model_pipelines(
        'poisson', _STATISTICAL,
        {'regress': (
            'Estimates the effects of the features on counts with Poisson '
            'regression, which assumes that their variance equals their '
            'mean (the residuals show if it does not).')}),
    *_model_pipelines(
        'probit', _STATISTICAL,
        {'classify': (
            'Estimates the effects of the features on two classes with a '
            'probit model, checking that its probabilities are '
            'calibrated.')}),
    *_model_pipelines(
        'quantile_regression', _STATISTICAL,
        {'regress': (
            'Estimates the effects of the features on the median of the '
            'label, which outliers and skewed errors sway less than the '
            'mean.')}),
    *_model_pipelines(
        'random_forest', _TREES,
        {'classify': (
            'Classifies with a random forest of 300 trees, which needs no '
            'scaling, weighting the classes in the sample of each tree by '
            'how rare they are.'),
         'regress': (
            'Regresses with a random forest of 300 trees, which needs no '
            'scaling.')},
        parameters = {'random_forest': {**_FOREST, 'n_jobs': -1}},
        class_weights = {'class_weight': 'balanced_subsample'}),
    *_model_pipelines(
        'ridge', _PENALIZED,
        {'regress': (
            'Regresses with a ridge penalty on standardized features, which '
            'steadies the coefficients of correlated features, with a grid '
            'search of the penalty.')},
        parameters = {'ridge': {
            'search': 'grid', 'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}}),
    *_model_pipelines(
        'robust_regression', _STATISTICAL,
        {'regress': (
            'Estimates the effects of the features on a number with robust '
            'regression, which gives less weight to rows with outlying '
            'errors.')}),
    *_model_pipelines(
        'svm', _SCALE_SENSITIVE,
        {'classify': (
            'Classifies with a support vector machine on standardized '
            'features, weighting the classes by how rare they are, with a '
            'grid search of its penalty and kernel width. It is slow with '
            'more than about 10,000 rows.'),
         'regress': (
            'Regresses with a support vector machine on standardized '
            'features, with a grid search of its penalty and kernel width. '
            'It is slow with more than about 10,000 rows.')},
        parameters = {'svm': {
            'search': 'grid', 'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 0.01, 0.1]}},
        class_weights = {'class_weight': 'balanced'}),
    *_model_pipelines(
        'tabpfn', _PRETRAINED,
        {'classify': (
            'Classifies with TabPFN, which fills, scales, and weighs the '
            'features itself. It suits up to about 10,000 rows and 500 '
            'features.'),
         'regress': (
            'Regresses with TabPFN, which fills, scales, and weighs the '
            'features itself. It suits up to about 10,000 rows and 500 '
            'features.')}),
    *_model_pipelines(
        'wls', _STATISTICAL,
        {'regress': (
            'Estimates the effects of the features on a number by weighted '
            'least squares, for errors whose variance differs from row to '
            'row. Set "weights" in the "wls_parameters" section.')},
        required = {'wls': ['weights']}),
    *_model_pipelines(
        'xgboost', _MISSING_AWARE_TREES,
        {'classify': (
            'Classifies with XGBoost, which handles missing values itself, '
            'resampling rare classes, with a random search of its learning '
            'rate, depth, and sampling of rows and features.'),
         'regress': (
            'Regresses with XGBoost, which handles missing values itself, '
            'with a random search of its learning rate, depth, and sampling '
            'of rows and features.')},
        parameters = {'xgboost': {
            'search': 'random', 'n_iter': 10, 'n_estimators': 300,
            'learning_rate': [0.03, 0.3], 'max_depth': [3, 8],
            'subsample': [0.6, 1.0], 'colsample_bytree': [0.6, 1.0],
            'min_child_weight': [1, 10]}},
        sample = True),
    *_model_pipelines(
        'zero_inflated_poisson', _STATISTICAL,
        {'regress': (
            'Estimates the effects of the features on counts with more '
            'zeros than a Poisson distribution has, with zero-inflated '
            'Poisson regression.')}))

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
            fixed_effects = True),
        *_MODEL_PIPELINES)}


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
        _share_required(idea, pipeline)
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


def _share_required(idea: MutableMapping[str, Any], pipeline: Pipeline) -> None:
    """Checks the required parameters of a pipeline, and shares them.

    Args:
        idea: the settings of a project. The parameters sections of the
            techniques in `pipeline.shared_with` are replaced by ones with the
            required parameters (unless they set them).
        pipeline: a pipeline named in the settings.

    Raises:
        ValueError: if the settings do not give a required parameter.

    """
    for technique, names in pipeline.required.items():
        key = f'{technique}_parameters'
        values = idea.get(key, {})
        missing = [name for name in names if values.get(name) is None]
        if missing:
            message = (
                f'the {pipeline.name!r} pipeline needs '
                f'{", ".join(repr(name) for name in missing)} in the '
                f'{key!r} section'
            )
            raise ValueError(message)
        for other in pipeline.shared_with.get(technique, ()):
            target = f'{other}_parameters'
            idea[target] = {
                **{name: values[name] for name in names},
                **idea.get(target, {})}


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
