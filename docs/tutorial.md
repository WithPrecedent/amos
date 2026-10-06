# Tutorial

This tutorial builds a study of the Wisconsin breast cancer data in stages. It
starts by applying a few techniques by hand and ends with a project that
compares alternatives, reads its settings from a file, and exports everything
needed to reproduce it. Every example on this page is run by the `amos` unit
tests, so you can copy and paste them.

## The vocabulary

| Part | What it is | In code |
| --- | --- | --- |
| Dataset | Your data and everything learned from it. | `amos.Dataset` |
| Technique | One method, such as "standard" scaling or a "logit" model. | Subclasses of `amos.Operation` |
| Worker | A stage of a project, such as "analyst". | A `chrisjen` worker |
| Step | A part of a worker with alternative techniques, such as "scale". | A step node in a worker |
| Project | The settings, the workflow built from them, and the result. | `amos.Project` |

`amos` is built on [chrisjen](https://WithPrecedent.github.io/chrisjen), which
turns settings into workflows. `amos` adds the `Dataset` that flows through
them and the data science techniques that work on it.

## 1. Load your data

A `Dataset` holds your data in a `pandas.DataFrame`. Its `label` is the column
that models predict, and its `seed` makes every random process reproducible:

```python
import sklearn.datasets

import amos

cancer = sklearn.datasets.load_breast_cancer(as_frame = True)
data = cancer.frame
dataset = amos.Dataset(data, label = "target", seed = 43)
print(dataset)
# Dataset(rows=569, columns=31, label='target', task='classify')
```

The task, "classify" or "regress", is inferred from the label: a label of
text, booleans, or a few whole numbers is classified, and other numbers are
regressed. You can also pass `task`. A dataset sorts its features into kinds,
which techniques use to choose the columns they change:

```python
print(dataset.classes)
# [0, 1]
print(len(dataset.numerics), dataset.categoricals, dataset.booleans)
# 30 [] []
```

## 2. Apply techniques by hand

Every technique has an `apply` method that changes the dataset and returns
it. The techniques are grouped in modules by what they do:

```python
from amos import describers, metrics, models, splitters, transformers

describers.Summarize().apply(dataset)
splitters.Stratified().apply(dataset)
transformers.Standard().apply(dataset)
models.Logit().apply(dataset)
metrics.RocAuc().apply(dataset)
print(len(dataset.train), len(dataset.test))
# 426 143
print(dataset.metrics["roc_auc"] > 0.95)
# True
```

`summarize` added a table of summary statistics, `stratified` split the rows
into training and test sets with the same share of each class, `standard`
rescaled the numeric features, `logit` fitted a logistic regression to the
training rows and predicted the test rows, and `roc_auc` scored the
predictions. The scaler learned the means and standard deviations from the
training rows only, so nothing about the test rows leaked into the model.

Each technique recorded what it did:

```python
print([entry["technique"] for entry in dataset.history])
# ['summarize', 'stratified', 'standard', 'logit', 'roc_auc']
print(dataset.history[2]["tool"])
# sklearn.preprocessing.StandardScaler
print(dataset.history[1]["seed"], dataset.history[1]["test"])
# 43 143
```

## 3. Describe a project

Applying techniques by hand is fine for exploring. For a study, describe the
whole project in settings. A project needs a section whose name ends in
`_project`, and its `{name}_workers` setting lists the workers, in order. Each
worker has a section that lists its techniques. The "general" section holds
the label and the seed:

```python
settings = {
    "general": {"label": "target", "seed": 43},
    "cancer_project": {"cancer_workers": "explorer, analyst, critic"},
    "explorer": {"techniques": "summarize, label_balance"},
    "analyst": {"techniques": "stratified, standard, logit"},
    "critic": {"techniques": "roc_auc, f1, confusion"},
}
project = amos.Project.create(settings, item = data)
print(project.result.metrics["roc_auc"] == dataset.metrics["roc_auc"])
# True
print(list(project.result.tables))
# ['summarize', 'label_balance', 'confusion']
```

The project did the same work as the techniques applied by hand, and got the
same result, because both used the same seed. `Project.create` worked on a
copy of `data`, so the data you passed is never changed.

## 4. Compare alternatives

There is rarely one obvious way to prepare data or one obvious model. An
`experiment` tries several. Give the analyst **steps**, each with a list of
alternative techniques, and a **criterion** to compare them with:

```python
settings["analyst"] = {
    "design": "experiment",
    "criterion": "roc_auc",
    "steps": "split, scale, model",
    "split_techniques": "stratified",
    "scale_techniques": "standard, robust, none",
    "model_techniques": "logit, random_forest",
}
project = amos.Project.create(settings, item = data)
analyst = {node.name: node for node in project.workflow}["analyst"]
print(len(analyst.results))
# 6
```

The experiment tried every combination of one technique from each step (1
splitter, 3 scalers, and 2 models make 6), each on its own copy of the data.
It kept the combination with the highest ROC AUC on the test rows and passed it
on to the critic. `none` is a technique that does nothing, so it tests whether
scaling helps at all.

Every combination is compared in a table, named for the worker. Each
combination is labeled with its steps, each named "{technique}_{step}":

```python
table = project.result.tables["analyst_comparison"]
print(list(table.columns))
# ['rank', 'score', 'roc_auc']
print(table.index[0] == analyst.winner)
# True
print(table.index[0].startswith("stratified_split > "))
# True
```

The combinations are always tried in the same order, and ties go to the
first, so the comparison is the same every time. A `contest` (from
`chrisjen`) does the same thing without the table.

## 5. Use a settings file

Settings can live in an ini, toml, json, yaml, or Python file. In an ini file,
numbers, `True` and `False`, and comma-separated lists are converted for you:

<!-- file: study.ini -->
```ini
[general]
label = target
seed = 43

[cancer_project]
cancer_workers = explorer, analyst, critic

[explorer]
techniques = summarize, label_balance

[analyst]
design = experiment
criterion = roc_auc
steps = split, scale, model
split_techniques = stratified
scale_techniques = standard, robust, none
model_techniques = logit, random_forest

[critic]
techniques = scorecard, confusion

[random_forest_parameters]
n_estimators = 200
max_depth = 5
```

Pass the path to the file instead of a `dict`:

```python
project = amos.Project.create("study.ini", item = data)
print(sorted(project.result.metrics)[:4])
# ['accuracy', 'balanced_accuracy', 'f1', 'log_loss']
```

The `scorecard` compared every branch of the experiment on every standard
metric for the task, and stored the final model's scores in `metrics`. It is
ready to publish:

```python
scorecard = project.scorecard
print(len(scorecard.table), scorecard.heading)
# 6 6 branches ranked by roc_auc
print(scorecard.to_markdown().splitlines()[0])
# | rank | split | scale | model | roc_auc | accuracy | balanced_accuracy | precision | recall | f1 | log_loss |
```

Save it with `to_csv`, `to_markdown`, `to_word` (a Word document), or
`to_image` (such as a png file), or in all four formats with `export`.

## 6. Set parameters

A section named `{technique}_parameters` holds parameters for that technique,
as `random_forest_parameters` does above. They are passed to the tool that the
technique wraps (here, scikit-learn's `RandomForestClassifier`), and a tool
only receives the parameters it accepts. A section for a worker (such as
`analyst_parameters`) passes its parameters to every technique in the worker.

Any model can search for the best value of parameters given as lists. Add
"search" ("grid" or "random") and list the values:

```python
settings["analyst"]["model_techniques"] = "random_forest"
settings["random_forest_parameters"] = {
    "search": "grid",
    "max_depth": [3, 6],
    "n_estimators": [50, 100],
    "cv": 3,
}
project = amos.Project.create(settings, item = data)
search = project.result.tables["random_forest_search"]
print(len(search), list(search.columns[-3:]))
# 4 ['mean_test_score', 'std_test_score', 'rank_test_score']
```

The search used 3-fold cross-validation on the training rows only and refitted
the best combination to all of them. For a random search, a list of two
numbers is a range to draw from, and "n_iter" sets the number of draws.

## 7. Evaluate and draw

The critic evaluates the model, and the artist draws it. Tables go in
`tables` and figures in `figures`:

```python
settings["critic"] = {"techniques": "scorecard, confusion, permutation_importance"}
settings["artist"] = {"techniques": "roc_curve, importance_plot"}
settings["cancer_project"]["cancer_workers"] = "explorer, analyst, critic, artist"
project = amos.Project.create(settings, item = data)
print(project.result.tables["confusion"].to_numpy().sum())
# 143
figure = project.result.figures["roc_curve"]
print(type(figure).__name__)
# Figure
```

Figures are `matplotlib` figures. In a notebook, a figure is shown when it is
the last line of a cell. Save one with `figure.savefig("roc.png")`.

## 8. Report and export

A project's `report` describes what it did:

```python
lines = project.report.contents.splitlines()
print(lines[0])
# project: cancer
print(lines[1].startswith("id: cancer_"))
# True
print(lines[2])
# workflow: explorer > analyst > critic > artist
```

Each run has an `id`, which is the name of the project and the date and time
unless you pass `id` to `Project.create`. `export` saves the
report, the settings, the versions of Python and every package that did the
work, the history of every technique and its parameters, the metrics, the
predictions, and every table and figure:

```python
folder = project.export()
print(sorted(path.name for path in (folder / "tables").iterdir()))
# ['analyst_comparison.csv', 'confusion.csv', 'label_balance.csv', 'permutation_importance.csv', 'random_forest_search.csv', 'scorecard.csv', 'summarize.csv']
```

Anyone with the settings file, the data, and "environment.json" can reproduce
these results exactly.

## Next steps

* The [advanced user guide](advanced.md) lists every technique, describes the
  settings, and shows how to write your own techniques.
* The [recipes](recipes.md) show complete studies: regression with statistical
  inference, imbalanced classes, categorical data, and more.
