# Idea File

A project's settings are its idea (a `chrisjen.Idea`). You can pass them to
`Project.create` as a `dict`, but a settings file keeps the design of a study
apart from its code, so it is easy to read, change, and share with the data.
This page shows how to write one and describes every setting of its "general"
section. The [advanced user guide](advanced.md#settings-reference) lists the
settings of the other sections. Every example on this page is run by the
`amos` unit tests.

## Writing the file

A settings file is made of sections, and each section is a set of named
settings. This one describes a project named "cancer" with two workers:

<!-- file: study.ini -->
```ini
[general]
label = target
seed = 43

[cancer_project]
cancer_workers = explorer, analyst

[explorer]
techniques = summarize

[analyst]
steps = split, scale, model
split_techniques = stratified
scale_techniques = standard
model_techniques = logit
```

Pass the path to the file to `Project.create`:

```python
import numpy as np
import pandas as pd
import sklearn.datasets

import amos

data = sklearn.datasets.load_breast_cancer(as_frame = True).frame
project = amos.Project.create("study.ini", item = data)
dataset = project.result
print(project.name, dataset.label, dataset.task, dataset.seed)
# cancer target classify 43
print(project.idea["analyst"]["steps"])
# ['split', 'scale', 'model']
```

The name of the project is the part before "_project" of the first section
whose name ends in "_project". If there is none, it is the name of the first
section other than "general" and "files". Pass `name` to `Project.create` to
choose it yourself.

### Formats

The format of the file is found from its extension:

| Format | Extension | Values |
| --- | --- | --- |
| ini | `.ini` | Text, which is converted for you (see below). |
| TOML | `.toml` | TOML's own types. |
| YAML | `.yaml` or `.yml` | YAML's own types. Reading YAML needs the `pyyaml` package. |
| JSON | `.json` | JSON's own types, and text is converted as in an ini file. |
| Python | `.py` | The file must define a `dict` named `settings`. Its values are used as they are. |

Here are the same settings in TOML:

<!-- file: study.toml -->
```toml
[general]
label = "target"
seed = 43

[cancer_project]
cancer_workers = ["explorer", "analyst"]

[explorer]
techniques = "summarize"

[analyst]
steps = ["split", "scale", "model"]
split_techniques = "stratified"
scale_techniques = "standard"
model_techniques = "logit"
```

in YAML:

<!-- file: study.yaml -->
```yaml
general:
  label: target
  seed: 43
cancer_project:
  cancer_workers: [explorer, analyst]
explorer:
  techniques: summarize
analyst:
  steps: [split, scale, model]
  split_techniques: stratified
  scale_techniques: standard
  model_techniques: logit
```

in JSON:

<!-- file: study.json -->
```json
{
  "general": {"label": "target", "seed": 43},
  "cancer_project": {"cancer_workers": ["explorer", "analyst"]},
  "explorer": {"techniques": "summarize"},
  "analyst": {
    "steps": ["split", "scale", "model"],
    "split_techniques": "stratified",
    "scale_techniques": "standard",
    "model_techniques": "logit"
  }
}
```

and in Python:

<!-- file: study.py -->
```python
settings = {
    "general": {"label": "target", "seed": 43},
    "cancer_project": {"cancer_workers": ["explorer", "analyst"]},
    "explorer": {"techniques": "summarize"},
    "analyst": {
        "steps": ["split", "scale", "model"],
        "split_techniques": "stratified",
        "scale_techniques": "standard",
        "model_techniques": "logit",
    },
}
```

Each gives the same project:

```python
paths = ["study.toml", "study.yaml", "study.json", "study.py"]
ideas = [amos.Project.create(path, item = data).idea for path in paths]
print(all(dict(idea) == dict(project.idea) for idea in ideas))
# True
```

### How ini values are converted

Every value in an ini file is text, so it is converted for you (and so is
the text in a JSON file):

| Value | Becomes | Example |
| --- | --- | --- |
| A whole number | An `int` | `seed = 43` |
| Any other number | A `float` | `test_size = 0.25` |
| "true" or "yes" (in any case) | `True` | `use_gpu = true` |
| "false" or "no" (in any case) | `False` | `use_gpu = false` |
| Items separated by a comma and a space | A `list`, with each item converted | `steps = split, scale, model` |
| Anything else | Text | `label = target` |

A few things follow from these rules:

* A list needs a comma and a space between its items. "split,scale" is a
  single piece of text.
* A list of one item is written as the item alone. Every setting that takes
  a list (such as "groups" or "techniques") also takes one name.
* Text that looks like a number, true, or false is converted, so an ini or
  JSON file cannot name a column "yes" or "2020". Use a TOML, YAML (with
  quotes), or Python file for those.
* Names of settings are case sensitive, and `%` has no special meaning.
  Lines that start with `#` or `;` are comments.

### Sections

| Section | What it holds |
| --- | --- |
| `general` | Settings for the data and the whole project, described below. |
| `files` | Where the project keeps its files, and whether it exports its results after every run (see the [settings reference](advanced.md#settings-reference)). |
| `{name}_project` | The project. Its `{name}_workers` setting lists the workers, in order. |
| A section named for a worker | The worker's steps, techniques, and design (see the [settings reference](advanced.md#settings-reference)). |
| `{name}_parameters` | Parameters for the technique, step, or worker `name` (see [Set parameters](tutorial.md#6-set-parameters)). |

## The general section

The "general" section describes the data and how every technique treats it.
Every setting in it is optional.

| Setting | Value | Default | Meaning |
| --- | --- | --- | --- |
| `label` | A column name | None | The column that models predict. |
| `task` | "classify" or "regress" | Inferred from the label | Whether models classify or regress the label. |
| `seed` | A whole number | None | Seed for every random process, which makes the results reproducible. |
| `groups` | A column name, or a list of them | None | Columns that identify groups of rows, which are not used as features. |
| `use_gpu` | true or false | false | Whether to fit the models that can use a GPU on an NVIDIA GPU. |
| `conserve_memory` | true or false | false | Whether every experiment keeps only the best combination's whole dataset while it tries them. |
| `pipeline` | The name of a pipeline | None | Fills in the settings of the analyst, critic, and artist for a kind of study. |

The settings (except `conserve_memory` and `pipeline`, which are for the
workers) are given to the `Dataset` that the project makes from your
data:

* If you pass data (such as a `DataFrame` or the path to a data file), the
  dataset is made with these settings.
* If you pass a `Dataset`, it keeps its own label, task, seed, and groups,
  and the settings fill in only those it lacks. "use_gpu" replaces the
  dataset's own if it is set.
* If the workflow loads its own data with a loader (such as `load_file`),
  "label", "task", and "groups" are the loader's default parameters, and
  "seed" is the dataset's seed. Parameters in the loader's own section take
  precedence.

```python
settings = {
    "general": {"seed": 43},
    "cancer_project": {"cancer_workers": ["explorer"]},
    "explorer": {"techniques": "summarize"},
}
mine = amos.Dataset(data, label = "target", seed = 7)
project = amos.Project.create(settings, item = mine)
print(project.result.label, project.result.seed)
# target 7
```

### label

The name of the column that models predict. Splitters that stratify, models,
metrics, and evaluators need it, so a project without a label can only
describe and change the data. They raise an error that asks you to set it.
The label stays in the data but is not a feature.

A loader's data may have a label of its own: set "label" in the loader's
parameters to use another one for that loader.

<!-- file: cases.csv -->
```csv
age,court,reversed
31,state,yes
45,federal,no
52,state,no
38,state,yes
60,federal,no
```

```python
settings = {
    "general": {"label": "reversed", "seed": 43},
    "cases_project": {"cases_workers": ["loader"]},
    "loader": {"techniques": "load_file"},
    "load_file_parameters": {"source": "cases.csv"},
}
project = amos.Project.create(settings)
print(project.result)
# Dataset(rows=5, columns=3, label='reversed', task='classify')
```

The values of the label can be anything, including "yes" and "no". Only the
settings file is converted, not the data.

### task

"classify" or "regress". Without it, the task is inferred from the label:

* A label of text, booleans, or categories is classified.
* A label of whole numbers with 10 or fewer different values (such as 0 and
  1, or ratings from 1 to 5) is classified.
* A label of decimal numbers that are all whole and take only two values
  (such as 0.0 and 1.0) is classified.
* Any other label is regressed.

Set "task" when the inference is wrong for your study, such as ratings that
should be regressed:

```python
rng = np.random.default_rng(0)
ratings = pd.DataFrame({"hours": rng.uniform(0, 10, 200)})
ratings["stars"] = np.clip(ratings["hours"] // 2 + 1, 1, 5).astype(int)
print(amos.Dataset(ratings, label = "stars").task)
# classify
settings = {
    "general": {"label": "stars", "task": "regress", "seed": 43},
    "ratings_project": {"ratings_workers": ["analyst"]},
    "analyst": {
        "steps": ["split", "model"],
        "split_techniques": "train_test",
        "model_techniques": "linear",
    },
}
project = amos.Project.create(settings, item = ratings)
print(project.result.task)
# regress
```

The task chooses the version of each technique (such as
`RandomForestClassifier` or `RandomForestRegressor`) and the metrics that the
scorecard reports. A model that can only do one task raises an error if the
task is the other one.

### seed

A whole number that seeds every random process: splitting the data,
sampling, models, searches, and so on. It is passed as `random_state` to
every tool that accepts one, unless the tool's parameters set
`random_state` themselves. With the same settings, data, and seed, a project
gives the same results every time:

```python
settings = {
    "general": {"label": "target", "seed": 43},
    "cancer_project": {"cancer_workers": ["analyst"]},
    "analyst": {
        "steps": ["split", "model"],
        "split_techniques": "stratified",
        "model_techniques": "random_forest",
    },
}
first = amos.Project.create(settings, item = data)
second = amos.Project.create(settings, item = data)
print(first.result.predictions.equals(second.result.predictions))
# True
print(first.result.history[-1]["parameters"]["random_state"])
# 43
```

Without a seed, random processes differ from run to run. Set one for any
study you will report. (Models fitted on a GPU can still differ slightly from
the same models fitted on a CPU; see
[Fitting models on a GPU](advanced.md#fitting-models-on-a-gpu).)

### groups

The names of columns that identify groups of rows, such as race, court, or
judge. Groups stay in the data, but they are not features, so transformers
and models do not use them. They are used by:

* the fairness metrics (`demographic_parity`, `equalized_odds`, and
  `equal_opportunity`, and their "_ratio" versions), which compare the
  model's predictions across the first group,
* `fairness`, which makes a table of how the model does for each group,
* the `scorecard`, which adds fairness metrics for every branch,
* `fixest`, which can use them as fixed effects and clusters, and
* `group_split`, which keeps the rows of each value of the first group in
  the same set.

```python
clinics = data.copy()
clinics["clinic"] = np.where(np.arange(len(clinics)) % 3 == 0, "north", "south")
settings = {
    "general": {"label": "target", "seed": 43, "groups": "clinic"},
    "cancer_project": {"cancer_workers": ["analyst", "critic"]},
    "analyst": {
        "steps": ["split", "scale", "model"],
        "split_techniques": "stratified",
        "scale_techniques": "standard",
        "model_techniques": "logit",
    },
    "critic": {"techniques": "fairness"},
}
project = amos.Project.create(settings, item = clinics)
print(project.result.groups, "clinic" in project.result.features)
# ['clinic'] False
print(list(project.result.tables["fairness"].index))
# ['north', 'south', 'difference', 'ratio']
```

Every group must be a column of the data, or an error names the missing
ones. See [Groups and fairness](advanced.md#groups-and-fairness) for more.

### use_gpu

Whether to fit the models that can use a GPU (`xgboost`, `catboost`,
`lightgbm`, and `tabpfn`, and the same models when an inferer fits them) on
an NVIDIA GPU. It is false unless the settings set it to true, so a project
runs on the CPU unless it asks for a GPU:

```ini
[general]
label = target
seed = 43
use_gpu = true
```

If there is no GPU, or a model cannot be fitted on it, `amos` warns and fits
the model on the CPU, so the same settings work on any computer. Every other
technique runs on the CPU either way. It must be true or false, or an error
is raised. See [Fitting models on a GPU](advanced.md#fitting-models-on-a-gpu)
for the details, including how to keep one model on the CPU and which GPUs
need older packages.

### conserve_memory

Whether every `experiment` (and `contest` or `survey`) keeps only what it
still needs while it tries its combinations of techniques. An experiment
normally keeps every combination's whole dataset, with its features and
fitted model, until the last one is done, so that a `regression_table` can
show every combination's model. With many combinations or a large dataset,
that may not fit in memory. If it is true, an experiment keeps only the best
dataset so far, so no more than two are in memory at once; its comparison
table, scores, and branches (and so the scorecard) are the same, but a
`regression_table` has only the winner's model.

```ini
[general]
label = target
conserve_memory = true
```

A worker's own setting wins over the general one, so one experiment can keep
every dataset while the others conserve memory:

```ini
[analyst]
design = experiment
conserve_memory = false
```

### pipeline

The name of a pipeline: the settings of an analyst, a critic, and an artist
for a kind of study, chosen so that you do not have to write them. A whole
study can be four lines:

<!-- file: pipeline.ini -->
```ini
[general]
label = target
seed = 43
pipeline = classification
```

```python
project = amos.Project.create("pipeline.ini", item = data)
print(project.name, project.idea["classification_project"])
# classification {'classification_workers': ['analyst', 'critic', 'artist']}
print(project.idea["analyst"]["steps"])
# ['split', 'impute', 'scale', 'encode', 'sample', 'model']
print(project.idea["analyst"]["model_techniques"])
# ['logit', 'random_forest', 'gradient_boosting', 'baseline']
print(project.idea["critic"], project.idea["artist"])
# {'techniques': ['stratified_k_fold', 'auto_evaluator']} {'techniques': ['auto_plot']}
```

Without a project section, one named for the pipeline is added, with the
analyst, critic, and artist as its workers. A project section that lists
its own workers keeps them, and the pipeline fills in only the analyst,
critic, and artist among them. The pipeline also sets "task" (if it is not
set) and the parameters of some techniques, and `project.idea` (and the
exported "settings.json") has every setting that it filled in, so the
methods are still on the record.

The steps whose techniques depend on the data are "auto" (see [Choosing
techniques automatically](advanced.md#choosing-techniques-automatically)),
so one pipeline adapts to the data when it runs. Missing values are filled
(if there are any), categorical features are encoded by how many levels
they have (dummy variables for a few, the mean label for many), and rare
classes are balanced (if they are rare). Each choice and its reason is in a
table:

```python
result = project.result
print(result.tables["auto_imputer"].loc["none", "reason"])
# no feature has missing values
print(result.tables["auto_sampler"].loc["none", "reason"])
# the classes are balanced: the rarest has 60% of the rows of the most common
print(result.tables["analyst_comparison"].index[0])
# stratified_split > auto_impute > standard_scale > auto_encode > auto_sample > logit_model
print(result.tables["auto_evaluator"].loc["regression_table", "reason"])
# logit has no coefficients with standard errors
```

These are the pipelines:

| Pipeline | Task | What its analyst does |
| --- | --- | --- |
| `classification` | classify | Compares `logit`, `random_forest`, `gradient_boosting`, and `baseline` in an experiment, by balanced accuracy, after balancing rare classes. |
| `regression` | regress | Compares `ridge`, `random_forest`, `gradient_boosting`, and `baseline` in an experiment, by RMSE. |
| `count_regression` | regress | Compares `poisson`, `negative_binomial`, a random forest and gradient boosting fitted for counts, and `baseline` in an experiment, by RMSE. |
| `classification_inference` | classify | Fits `logit_sm` for two classes, `ordinal_regression` for ordered classes, or `mnlogit` for others, with `auto_model`. |
| `regression_inference` | regress | Fits `ols`, or `robust_regression` if the errors have outliers, with `auto_model`. |
| `count_inference` | regress | Fits `poisson`, `negative_binomial` for overdispersed counts, or `zero_inflated_poisson` for excess zeros, with `auto_model`. |
| `panel_classification` | classify | Fits a `fixest` logit model with the groups as fixed effects, clustered by the first group. |
| `panel_regression` | regress | Fits a `fixest` model with the groups as fixed effects, clustered by the first group. |

In each, the critic cross-validates the model (`stratified_k_fold` or
`k_fold`) and applies the evaluators that suit it with `auto_evaluator`
(such as `confusion` for classes, `regression_table` for a model with
coefficients and standard errors, `shap_importance` for a model of trees,
and `fairness` for two classes with groups), and the artist draws the
figures that suit it with `auto_plot`.

The pipelines that predict scale the numeric features and balance rare
classes. The pipelines for inference ("_inference" and "panel_") do
neither: they keep the coefficients in the units of the features and the
real share of each class (balancing would distort the coefficients and
their standard errors), and they encode each categorical feature against
its first level, so its coefficients are differences from that level. The
panel pipelines need "groups".

Every setting that you write yourself wins, so a pipeline is a starting
point. Change what you need and keep the rest:

```python
settings = {
    "general": {"label": "target", "seed": 43, "pipeline": "classification"},
    "analyst": {"model_techniques": ["logit", "random_forest"]},
    "random_forest_parameters": {"max_depth": 4},
}
project = amos.Project.create(settings, item = data)
print(project.idea["random_forest_parameters"])
# {'n_estimators': 300, 'max_depth': 4}
print(len(project.result.tables["analyst_comparison"]))
# 2
```

A worker's own section can also name a pipeline, which fills in only that
worker (and its parameters), such as `pipeline = classification_inference`
in the "critic" section. To see what a pipeline fills in, look at it in
`amos.pipelines.PIPELINES`. Add a `Pipeline` to it to name one of your own:

```python
print(amos.pipelines.PIPELINES["regression_inference"].analyst["steps"])
# ['split', 'impute', 'encode', 'model']
amos.pipelines.PIPELINES["quick"] = amos.Pipeline(
    name = "quick",
    task = "classify",
    description = "A logit model and its confusion matrix.",
    analyst = {
        "steps": ["split", "model"],
        "split_techniques": ["stratified"],
        "model_techniques": ["logit"],
    },
    critic = {"techniques": ["confusion"]},
    artist = {"techniques": ["confusion_heatmap"]},
)
settings = {"general": {"label": "target", "seed": 43, "pipeline": "quick"}}
quick = amos.Project.create(settings, item = data)
print(sorted(quick.result.figures))
# ['confusion_heatmap']
del amos.pipelines.PIPELINES["quick"]
```
