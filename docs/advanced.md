# Advanced User Guide

The [tutorial](tutorial.md) shows how to build a study. This guide describes
how `amos` works, every setting, and how to add your own techniques. The
[technique catalog](catalog.md) lists every technique. Every example on this
page is run by the `amos` unit tests.

## How the pieces fit together

```text
settings ──create──▶ Idea ──draft──▶ workflow ──apply──▶ Dataset ──▶ report, export
(file or dict)     (chrisjen      (graph of workers,   (your data and
                    settings)      steps, techniques)   all it taught)
```

`amos` is built on [chrisjen](https://WithPrecedent.github.io/chrisjen), which
reads settings and builds a workflow from them. Everything that `chrisjen`
does works in `amos`, so its [advanced
guide](https://WithPrecedent.github.io/chrisjen/advanced/) applies here too.
`amos` adds:

| Class | Description |
| --- | --- |
| `Project` | A `chrisjen.Project` that makes your data into a `Dataset`, applies the workflow to a copy of it, and can `export` the results. |
| `Dataset` | The item that flows through the workflow: the data, the label, the split, the model, and everything learned. |
| `Operation` | Base class for every `amos` technique (a `chrisjen.Technique` that works on a `Dataset`). |
| `Cleaner`, `Describer`, `Splitter`, `Transformer`, `Sampler`, `Model`, `Metric`, `Evaluator`, `Plot`, `Effect` | The genres of techniques. `Transformer` has the genres `Imputer`, `Scaler`, `Encoder`, `Mixer`, and `Reducer`, and `Metric` has the genre `GroupMetric` (fairness metrics). |
| `Experiment` | A design that compares every combination of techniques and keeps a table of how each did. |
| `Findings` | The default report. |

## The library

Every technique is a class, stored in the `chrisjen` library (also available as
`amos.library`) under its snake case name as soon as it is defined. That is
how names in settings are found. The genres are nested layers of the library:

```python
import dataclasses

import numpy as np
import pandas as pd
import sklearn.datasets

import amos

print(sorted(amos.library["vertex"]["operation"]))
# ['cleaner', 'describer', 'effect', 'evaluator', 'metric', 'model', 'plot', 'sampler', 'splitter', 'transformer']
print(amos.library.classify("smote"), amos.library.classify("one_hot"))
# sampler encoder
print(amos.library.all["random_forest"])
# <class 'amos.models.RandomForest'>
```

Names must be unique across the library, including the names of `chrisjen`
classes such as "none", "flow", and "summary". The [technique
catalog](catalog.md) lists every technique by genre.

## Settings reference

A project's settings are sections of settings. `amos` reads two special
sections. Every other section is read by `chrisjen`.

| Section | Setting | Meaning |
| --- | --- | --- |
| `general` | `label` | The column that models predict. |
| | `task` | "classify" or "regress", if it should not be inferred from the label. |
| | `seed` | Seed for every random process (passed as `random_state`). |
| | `groups` | Columns that identify groups of rows (see "Groups and fairness" below). |
| `files` | `root_folder` | Folder for the project's files. Defaults to the current folder. |
| | `input_folder` | Folder (in the root folder) where data files named by a relative path are looked for. |
| | `output_folder` | Folder (in the root folder) in which `export` makes a folder for each run. |
| | `export_results` | Whether to export the results after every run. Defaults to `False`. |

The `chrisjen` settings, in brief:

| Setting | Meaning |
| --- | --- |
| A section named `{name}_project` | The project. Its `{name}_workers` setting lists the workers, in order. |
| A section named for a worker | The worker's settings, below. |
| `steps` or `{name}_steps` | The steps of the worker (or of `name`). |
| `techniques` or `{name}_techniques` | The techniques of the worker (or of the step `name`). In an `experiment`, the techniques of a step are alternatives. |
| `design` or `{name}_design` | How the worker applies its nodes. Defaults to `flow`. |
| `criterion` or `{name}_criterion` | What an `experiment` or `contest` compares: a metric (such as `roc_auc`) or another `chrisjen.Criteria`. |
| A section named `{name}_parameters` | Parameters for the technique, step, or worker `name`. |
| A section named `{technique}_{step}_parameters` | Parameters for one technique in one step. |

The designs:

| Design | What it does |
| --- | --- |
| `flow` | Applies its nodes one after another. The default. |
| `experiment` | Tries every combination of one technique from each step, keeps the best, and adds a comparison table. |
| `contest` | Like `experiment`, without the table. |
| `survey` | Tries every combination and averages the results. The results must be numbers, so it is rarely used with a `Dataset`. |
| `benchmark` | Repeats its nodes until its criterion is met. |

## The dataset

A `Dataset` has these attributes:

| Attribute | Meaning |
| --- | --- |
| `data` | The data, a `pandas.DataFrame`. It stays one `DataFrame` after the data is split. |
| `label` | The name of the column that models predict. |
| `task` | "classify" or "regress". |
| `seed` | Seed for every random process. |
| `groups` | Columns that identify groups of rows. They stay in the data but are not features. |
| `train`, `test` | Index labels of the training and test rows. `test` is `None` until the data is split. |
| `model` | The fitted model. |
| `predictions` | Predictions for the test rows (or every row, if the data is not split). |
| `probabilities` | Predicted probabilities of each class, for the same rows. |
| `metrics` | Scores, by the name of each metric. |
| `tables`, `figures` | Tables and figures, by the name of the technique that made them. |
| `fitted` | Fitted tools (such as a scaler), by the name of the technique that fitted them. |
| `history` | A record of each technique applied, with its tool and parameters. |
| `branches` | A record of every branch of the most recent experiment (for the scorecard). |

Its properties select parts of the data: `x` and `y` (the features and the
label), `x_train`, `y_train`, `x_test`, and `y_test`, and the names of the
features of each kind:

```python
data = pd.DataFrame({
    "age": [31, 45, 52, 38],
    "court": ["state", "federal", "state", "state"],
    "appealed": [True, False, True, False],
    "decided": pd.to_datetime(["2020-01-02", "2021-03-04", "2022-05-06", "2023-07-08"]),
    "reversed": ["yes", "no", "no", "yes"],
})
dataset = amos.Dataset(data, label = "reversed")
print(dataset.numerics, dataset.categoricals, dataset.booleans, dataset.dates)
# ['age'] ['court'] ['appealed'] ['decided']
print(dataset.task, dataset.classes)
# classify ['no', 'yes']
```

Models cannot use dates or text, so encode them (dates with `date_parts`, and
text with an encoder such as `one_hot` or `tfidf`) or remove them before the
model step.

`Dataset.create` makes a dataset from a `DataFrame`, a Polars `DataFrame` or
`LazyFrame`, a `numpy` array, a `dict` of columns, the path to a data file
(csv, tsv, Excel, parquet, feather, json, Stata, SPSS, or pickle), or a
scikit-learn dataset loaded with `as_frame = True`. A `Project` does this for
you.

## How techniques work

### Parameters

A technique receives its parameters from the settings and from the workers
that contain it, as in `chrisjen`: a technique's own parameters are overridden
by those of its step, then by those of its workers, and then by any passed to
`Project.apply`. A technique that wraps a tool passes the tool only the
parameters it accepts, so a parameter set for a whole worker does not break
the techniques that do not use it.

If the dataset has a `seed` and a tool accepts a `random_state` that was not
set, the seed is used. Every random process in a project is reproducible.

### Columns

Each genre of transformer changes the features of some kinds by default:
imputers change the numeric features with missing values (`mode_impute`
changes features of every kind), scalers and mixers change the numeric
features, encoders change the categorical and text features, and reducers
change the numeric and boolean features. Pass `columns` to choose others:

```python
cancer = sklearn.datasets.load_breast_cancer(as_frame = True)
dataset = amos.Dataset.create(cancer, seed = 43)
amos.transformers.MinMax().apply(dataset, columns = ["mean radius"])
print(dataset.history[-1]["columns"])
# ['mean radius']
print(dataset.data["mean radius"].max())
# 1.0
```

The columns that a transformer makes take the place of the columns it
changed, so the order of the other columns does not change. Transformers are
fitted to the training rows (all rows, before the data is split) and then
change every row. The fitted tool is stored in `fitted`:

```python
print(type(dataset.fitted["min_max"]).__name__)
# MinMaxScaler
```

### Labels and the positive class

Metrics, encoders, and models handle labels of any type. For a binary label,
the positive class is the last of `Dataset.classes` (such as 1, `True`, or
"yes"), and metrics such as `precision` and `roc_auc` are about that class.
For more than two classes, `precision`, `recall`, and `f1` are averaged over
the classes ("macro") and `roc_auc` compares each class with the rest. Pass
`average`, `pos_label`, or the other parameters of the scikit-learn metric to
change this. Encoders that learn from the label (such as `target`) are given
the position of each class when the classes are text.

### Hyperparameter search

Every model accepts these parameters:

| Parameter | Meaning |
| --- | --- |
| `search` | "grid", "random", or "optuna". Without it, lists are passed to the model as they are. |
| `cv` | Number of cross-validation folds. Defaults to 5. |
| `n_iter` | Number of combinations to try in a random or Optuna search. Defaults to 10. |
| `scoring` | The scikit-learn scorer to compare with, such as "roc_auc". Defaults to the model's own score. |

With a search, every parameter given as a list is searched. In a random or
Optuna search, a list of two numbers is a range (whole numbers are drawn from
it, inclusive; other numbers are drawn from between them). An Optuna search
chooses each combination to try from the results of the ones before, so it
usually finds good values in fewer tries, and it searches a range that spans
two or more orders of magnitude (such as a learning rate from 0.001 to 0.1) on
a log scale. The search uses the training rows only, and the results of every
combination are stored in `tables` as "{name}_search".

```python
amos.splitters.Stratified().apply(dataset)
amos.models.RandomForest().apply(
    dataset, search = "random", n_estimators = [20, 60], max_depth = [2, 8],
    n_iter = 3, cv = 3)
search = dataset.tables["random_forest_search"]
print(len(search), search["param_n_estimators"].between(20, 60).all())
# 3 True
amos.models.RandomForest().apply(
    dataset, search = "optuna", n_estimators = [20, 60], max_depth = [2, 8],
    n_iter = 3, cv = 3)
print(list(dataset.tables["random_forest_search"].columns[:2]))
# ['param_max_depth', 'param_n_estimators']
```

### Statistical inference

Machine learning models predict. Research usually also needs inference: which
features matter, by how much, and how surely. `ols` and `glm` wrap statsmodels
and store a table of coefficients with their standard errors, test
statistics, p-values, and 95% confidence intervals:

```python
diabetes = sklearn.datasets.load_diabetes(as_frame = True)
dataset = amos.Dataset.create(diabetes, seed = 43)
amos.models.OLS().apply(dataset)
table = dataset.tables["ols_coefficients"]
print(list(table.columns))
# ['coefficient', 'standard_error', 'statistic', 'p_value', 'ci_lower', 'ci_upper']
print(table.loc["bmi", "p_value"] < 0.001)
# True
```

`glm` is a logistic regression (without a penalty) for classification and a
linear regression for regression. Set its "family" parameter for other
generalized linear models, such as "poisson" for counts.

`fixest` wraps pyfixest for regressions with fixed effects (an intercept for
each level of a column, such as each judge or year) and standard errors
clustered by a column. The columns named by its "fixed_effects" and "cluster"
parameters are usually the dataset's `groups`, and they are not used as
features. It is a least squares regression, or a logit model for
classification; set "family" to "poisson" for counts or "probit":

```python
clinics = diabetes.frame.copy()
clinics["clinic"] = [f"c{i % 10}" for i in range(len(clinics))]
dataset = amos.Dataset(clinics, label = "target", seed = 43, groups = ["clinic"])
amos.models.Fixest().apply(dataset, fixed_effects = "clinic", cluster = "clinic")
print(list(dataset.tables["fixest_coefficients"].index[:3]))
# ['age', 'sex', 'bmi']
```

## Groups and fairness

Research about people often needs to know whether a model treats groups
differently. Name the columns that identify groups (such as race, court, or
judge) as the dataset's `groups`, or in the "groups" setting of the "general"
section. Groups stay in the data but are not features, so transformers and
models do not use them. Then:

* The fairness metrics (`demographic_parity`, `equalized_odds`, and
  `equal_opportunity`, with a "_ratio" version of each) compare the model's
  predictions across the dataset's first group, or the column named by their
  "group" parameter. They wrap fairlearn and need a label with two classes.
* `fairness` makes a table of the count, selection rate, accuracy, and true
  and false positive rates of each group, with the largest difference and
  smallest ratio between groups.
* The `scorecard` adds `demographic_parity` and `equalized_odds` for every
  branch.
* `fixest` can use groups as fixed effects and clusters, and `group_split`
  keeps each row of the dataset's first group in the same set.

```python
data = cancer.frame.copy()
data["clinic"] = np.where(np.arange(len(data)) % 3 == 0, "north", "south")
dataset = amos.Dataset(data, label = "target", seed = 43, groups = ["clinic"])
print("clinic" in dataset.features)
# False
amos.splitters.Stratified().apply(dataset)
amos.transformers.Standard().apply(dataset)
amos.models.Logit().apply(dataset)
amos.evaluators.Fairness().apply(dataset)
print(list(dataset.tables["fairness"].index))
# ['north', 'south', 'difference', 'ratio']
amos.metrics.EqualizedOdds().apply(dataset)
print(0 <= dataset.metrics["equalized_odds"] <= 1)
# True
```

Rows that a sampler copies keep their groups. Synthetic rows (made by
`smote`, for example) have no groups.

## Uncertainty

`conformal` turns any model's predictions into intervals (for regression) or
sets of classes (for classification) that contain the true value for at least
a chosen share of rows, with MAPIE's cross-conformal methods. The share of
test rows covered and the width of the intervals (or the size of the sets)
are stored in `metrics`:

```python
dataset = amos.Dataset.create(diabetes, seed = 43)
amos.splitters.TrainTest().apply(dataset)
amos.models.Linear().apply(dataset)
amos.evaluators.Conformal().apply(dataset, confidence = 0.9)
print(list(dataset.tables["conformal"].columns))
# ['actual', 'prediction', 'lower', 'upper', 'covered']
print(dataset.metrics["coverage"] > 0.8)
# True
```

## Causal effects

A model predicts the label. An *effect* estimates how much a treatment (such
as a program or a ruling) changes the label, after accounting for the other
features. The effects in `amos.effects` use double machine learning, from
DoubleML: models of the label and of the treatment (named with `amos` model
names) are fitted to some folds of the rows and applied to the others, so
flexible models can control for the features without biasing the estimate.
An effect uses every row, so it needs no split.

| Effect | Estimates |
| --- | --- |
| `partially_linear` | The effect of a treatment that changes the label by the same amount for every row. |
| `interactive_regression` | The average effect of a treatment with two values, which may differ from row to row. |

```python
rng = np.random.default_rng(1)
age = rng.normal(40, 10, 500)
program = (rng.random(500) < 1 / (1 + np.exp(-(age - 40) / 5))).astype(int)
outcome = 2 * program + 0.1 * age + rng.normal(0, 1, 500)
treated = pd.DataFrame({"age": age, "program": program, "outcome": outcome})
dataset = amos.Dataset(treated, label = "outcome", seed = 43)
amos.effects.PartiallyLinear().apply(
    dataset, treatment = "program", outcome_model = "linear",
    treatment_model = "logit")
effect = dataset.tables["partially_linear"].loc["program"]
print(effect["ci_lower"] < 2 < effect["ci_upper"])
# True
```

The estimates are causal only if every feature that affects both the
treatment and the label is among the features.

## Survival analysis

For the time until an event (such as rearrest or the end of a case), the
label is the time, and a column says whether the event happened (1) or the
row stopped being observed first (0, censored). `kaplan_meier` and
`survival_curves` describe and draw the share of rows without the event over
time, overall or by group (with a table of the number of rows at risk, if
"at_risk" is set). `cox` is a Cox proportional hazards regression,
with hazard ratios in its coefficients table, and `concordance` scores its
predictions. They wrap lifelines:

```python
rng = np.random.default_rng(2)
prior = rng.poisson(2, 400).astype(float)
times = pd.DataFrame({
    "prior": prior,
    "arrested": (rng.random(400) < 0.7).astype(int),
    "days": rng.exponential(400 / (1 + prior)).round() + 1,
})
dataset = amos.Dataset(times, label = "days", seed = 43, groups = ["arrested"])
amos.describers.KaplanMeier().apply(dataset, event = "arrested")
amos.splitters.TrainTest().apply(dataset)
amos.models.Cox().apply(dataset, event = "arrested")
amos.metrics.Concordance().apply(dataset)
print(dataset.tables["cox_coefficients"].loc["prior", "hazard_ratio"] > 1)
# True
print(dataset.metrics["concordance"] > 0.5)
# True
```

## Figures

The artist's plots draw `matplotlib` figures, which are stored in the
dataset's `figures` and saved by `Project.export`. Every plot takes a "width"
and "height" (in inches) and a "title". They fall into a few families:

| Family | Plots |
| --- | --- |
| The data | `histograms`, `box_plots`, `count_plots`, `pair_plot`, `correlation_heatmap`, `missing_heatmap`, `label_plot` |
| Classifiers | `roc_curve`, `precision_recall_curve`, `det_curve`, `calibration_curve`, `confusion_heatmap` |
| Regressions | `actual_vs_predicted`, `residual_plot`, `qq_plot`, `influence_plot`, `coefficient_plot` |
| Models | `importance_plot`, `partial_dependence`, `learning_curve`, `validation_curve`, `search_plot`, `prediction_intervals`, `tree_plot`, `shape_functions` |
| SHAP values | `shap_bar`, `shap_beeswarm`, `shap_violin`, `shap_heatmap`, `shap_decision`, `shap_embedding`, `shap_waterfall`, `shap_force`, `shap_scatter`, `shap_partial_dependence`, `shap_group_difference` |
| Groups and time | `fairness_plot`, `survival_curves` |

Plots of the data are split or colored by the label's classes (or by the
dataset's first group, or by the column "by"). Plots of a table, such as
`coefficient_plot` (of the coefficients of `ols`, `glm`, `fixest`, or `cox`,
or of an effect), `search_plot`, `fairness_plot`, and `prediction_intervals`,
draw the table that another technique made, and the last two make it if it
is missing. `learning_curve` and `validation_curve` fit copies of a
scikit-learn model again, in cross-validation on the training rows.

The SHAP plots draw the SHAP values that `shap_importance` found (or find
them) for the class "category", with shap's own plotting functions. shap
draws with `matplotlib.pyplot`, so `amos` lends it each figure while it
draws, without opening a window, and seeds the random numbers that shap uses
to jitter dots, so that a figure is the same every time.

```python
dataset = amos.Dataset.create(diabetes, seed = 43)
amos.splitters.TrainTest().apply(dataset)
amos.models.OLS().apply(dataset)
for plot in (amos.plots.CoefficientPlot(), amos.plots.QqPlot()):
    plot.apply(dataset)
amos.plots.InfluencePlot().apply(dataset, title = "Influence of each row")
print(sorted(dataset.figures))
# ['coefficient_plot', 'influence_plot', 'qq_plot']
```

## Writing your own techniques

Subclass the genre that fits and write the method it requires. The class is
in the library under its snake case name as soon as it is defined, so it can
be named in settings right away.

| Genre | Write | Returns |
| --- | --- | --- |
| `Cleaner` | `clean(self, data, **kwargs)` | The cleaned `DataFrame`. |
| `Describer` | `describe(self, item, **kwargs)` | A table, stored in `tables`. |
| `Splitter` | `divide(self, item, test_size, **kwargs)` | The training and test index labels. |
| `Evaluator` | `evaluate(self, item, **kwargs)` | A table, stored in `tables`. |
| `Plot` | `draw(self, item, figure, **kwargs)` | Nothing: it draws on `figure`. |
| `Transformer`, `Sampler`, `Model`, `Metric` | Set `contents` to a tool. | |
| `Operation` | `implement(self, item, **kwargs)` | The changed `Dataset`. |

Keyword parameters of these methods are filled from the settings. For
example, a cleaner that limits extreme values (winsorizes):

```python
@dataclasses.dataclass
class Winsorize(amos.Cleaner):
    """Limits each numeric column to its 1st and 99th percentiles."""

    def clean(self, data, lower = 0.01, upper = 0.99, **kwargs):
        numbers = data.select_dtypes("number")
        limits = numbers.quantile([lower, upper])
        data[numbers.columns] = numbers.clip(
            limits.iloc[0], limits.iloc[1], axis = 1)
        return data


dataset = amos.Dataset.create(cancer, seed = 43)
Winsorize().apply(dataset, upper = 0.95)
print(dataset.history[-1])
# {'technique': 'winsorize', 'rows': [569, 569], 'columns': [31, 31]}
```

That cleaner looks at every row, so it belongs before the split. A technique
that learns from the data should be a `Transformer`, so that it learns only
from the training rows. Most transformers, samplers, models, and metrics need
no code at all: set `contents` to the import path of a scikit-learn-style
tool. Its parameters (and `columns`, for a transformer) come from the settings:

```python
@dataclasses.dataclass
class Spline(amos.Mixer):
    contents: str = "sklearn.preprocessing.SplineTransformer"


@dataclasses.dataclass
class BayesRidge(amos.Model):
    contents: str = "sklearn.linear_model.BayesianRidge"


@dataclasses.dataclass
class MedianError(amos.Metric):
    contents: str = "sklearn.metrics.median_absolute_error"
    greater_is_better = False
    tasks = ("regress",)


settings = {
    "general": {"label": "target", "seed": 43},
    "diabetes_project": {"techniques": "train_test, spline, bayes_ridge, median_error"},
    "spline_parameters": {"columns": ["bmi"], "n_knots": 4},
}
project = amos.Project.create(settings, item = diabetes)
print(type(project.result.model).__name__, "median_error" in project.result.metrics)
# BayesianRidge True
print(project.result.features[2:5])
# ['bmi_sp_0', 'bmi_sp_1', 'bmi_sp_2']
```

A model whose tool differs by task sets `tools` (a `dict` mapping "classify"
and "regress" to tools) instead of `contents`. A metric sets
`greater_is_better`, `tasks`, and `uses_probabilities` as class attributes.
Because every metric is also a `chrisjen.Criteria`, a new metric can be the
criterion of an experiment.

## Experiments

An `experiment` is a `chrisjen` `contest` that also records how every
combination did. You can build one in code, as any `chrisjen` worker. Its
`populate` method takes the nodes in order. A list of nodes is a set of
alternatives:

```python
dataset = amos.Dataset.create(cancer, seed = 43)
amos.splitters.Stratified().apply(dataset)
experiment = amos.Experiment(name = "models", criteria = amos.metrics.F1())
experiment.populate([
    [amos.transformers.Standard(), amos.transformers.Robust()],
    [amos.models.Baseline(), amos.models.Logit()],
])
result = experiment.apply(dataset)
print(experiment.winner.endswith("logit"))
# True
print(result.tables["models_comparison"].shape)
# (4, 3)
```

`scores` has the score of each combination, `results` the dataset that each
made, and `winner` the label of the best. The criterion of an experiment can be
a metric for which lower is better (such as `log_loss`). It is negated to rank
the combinations, but the table shows its real value. The winning dataset also
keeps a `Branch` for every combination in its `branches`: the technique used at
each step, its score, and its predictions, which is what a scorecard compares.

## Scorecards

The `scorecard` is the critic's summary of an analysis. It has one row for
each branch of the most recent experiment (each combination of techniques),
ranked by the experiment's criterion:

| Column | Meaning |
| --- | --- |
| `rank` | 1 for the best branch. |
| One for each step | The technique used at that step. Steps are named as in the settings (such as "scale"), or by their genre (such as "scaler") in an experiment built in code. |
| One for each metric | The branch's score, computed from its own predictions on the test rows: the criterion first, then the other standard metrics for the task, then any other metrics the branches computed. |

A dataset that did not come from an experiment gets one row for its final
model, with the techniques that made it. As a technique in the critic,
`scorecard` stores its table in the dataset's `tables` and the final model's
scores in its `metrics`. Its "metrics" parameter chooses the metrics, and the
`scorecard_digits` and `scorecard_title` settings (in the critic's section) set
the digits shown and the title.

In Python, use `Project.scorecard` (which returns the critic's scorecard, or
makes one) or `Scorecard.create` with a dataset or a project:

```python
scorecard = amos.evaluators.Scorecard.create(result)
print(scorecard.heading)
# 4 branches ranked by f1
print(list(scorecard.table.columns[:4]))
# ['rank', 'scaler', 'model', 'f1']
print(scorecard.to_markdown().splitlines()[1])
# | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
```

A scorecard can be saved in several formats:

| Method | Saves | Needs |
| --- | --- | --- |
| `to_csv(path)` | A csv file, with scores at full precision. It also returns the text. | |
| `to_markdown(path)` | A Markdown table, with numbers aligned to the right. It also returns the text. | |
| `to_latex(path)` | A LaTeX table (with the booktabs package). It also returns the text. | |
| `to_html(path)` | An HTML table. It also returns the HTML. | great_tables (`amos[tables]`) |
| `to_word(path)` | A Word document with a heading and a table. | python-docx (`amos[word]`) |
| `to_image(path)` | An image of the table, in the format of the file's extension (such as png, svg, or pdf). `to_figure` returns the `matplotlib` figure instead. | matplotlib (`amos[plots]`) |
| `export(folder, formats)` | Several formats, named "scorecard" (csv, md, docx, and png by default; also tex, html, svg, and pdf). | |

Scores in the Markdown table, Word document, and image are rounded to `digits`
(3 by default). The title of the Word document and image is `title`, or a
description of the branches if there is no title.

## Reports and exports

`Findings`, the default report, describes the data, the split, the workflow,
the winner of each experiment or contest, the model, the metrics, and the
tables and figures. To use another report, pass a `chrisjen.Report` as
`report` to `Project.create`.

`Project.export` saves these files with the project's `clerk` (a
`nagata.FileManager`), in a folder named for the run in the output folder:

| File | Contents |
| --- | --- |
| report.txt | The text of the report. |
| settings.json | The settings of the project. |
| environment.json | The versions of Python and of the packages used. |
| history.json | Each technique applied, with its tool and parameters. |
| metrics.csv | The model's metrics. |
| predictions.csv | The labels, predictions, and probabilities of the predicted rows. |
| tables/{name}.csv | Each table. |
| figures/{name}.png | Each figure. |
| scorecard.csv, .md, .docx, .png | The scorecard in each format. The Word document and image are saved if python-docx and matplotlib are installed. |
| data.csv | The data after the workflow, if `export(data = True)`. |

## Errors you may see

| Error | Cause |
| --- | --- |
| `ValueError: the dataset has no label` | A technique needs a label. Set "label" in the "general" section or pass it to `Dataset`. |
| `ValueError: the data has not been split` | A technique asked for the test rows before a splitter was applied. |
| `ValueError: '...' cannot use the dates or text in [...]` | A model was given text or dates. Encode them (with an encoder such as `one_hot`) or remove them (with `drop_columns`) first. |
| `ValueError: '...' can classify, but the label is set to regress` | The model does only one task. Choose another model or set "task". |
| `ValueError: '...' needs predictions: apply a model first` | A metric, evaluator, or plot came before the model. |
| `ValueError: '...' needs predicted probabilities, which the model does not make` | The metric (such as `roc_auc`) needs probabilities. |
| `ImportError: ... install it with "pip install amos[...]"` | The technique wraps an optional package that is not installed. |
| `ValueError: '...' needs a group` | A fairness metric or `fairness` was used without groups. Set "groups" in the "general" section or pass "group". |
| `KeyError: the fixed_effects column ... is not in the data` | A column named by a model's parameter (such as "fixed_effects", "cluster", or "event") is missing. |
| `ValueError: '...' needs the name of a "treatment" column` | An effect was used without a "treatment" parameter. |
| `KeyError: no class named ... is in the library` | A name in the settings is not a technique. Check the spelling against the [technique catalog](catalog.md). |
| `ValueError: ... needs criteria to compare results` | An `experiment` or `contest` has no "criterion" setting. |
