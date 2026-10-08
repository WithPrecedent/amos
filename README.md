# amos

| | |
| --- | --- |
| Version | [![PyPI Latest Release](https://img.shields.io/pypi/v/amos.svg?style=flat-square&color=cornflowerblue&label=PyPI&logo=PyPI&logoColor=yellow)](https://pypi.org/project/amos/) [![GitHub Latest Release](https://img.shields.io/github/v/tag/WithPrecedent/amos?style=flat-square&color=forestgreen&label=GitHub&logo=github)](https://github.com/WithPrecedent/amos/releases) |
| Status | [![Build Status](https://img.shields.io/github/actions/workflow/status/WithPrecedent/amos/ci.yml?branch=main&style=flat-square&color=cadetblue&label=Tests&logo=pytest)](https://github.com/WithPrecedent/amos/actions/workflows/ci.yml?query=branch%3Amain) [![Development Status](https://img.shields.io/badge/Development-Active-seagreen?style=flat-square&logo=git)](https://www.repostatus.org/#active) [![Project Stability](https://img.shields.io/pypi/status/amos?style=flat-square&logo=pypi&label=Stability&logoColor=yellow)](https://pypi.org/project/amos/) |
| Documentation | [![Hosted By](https://img.shields.io/badge/Hosted_by-Github_Pages-blue?style=flat-square&color=forestgreen&logo=github)](https://WithPrecedent.github.io/amos) |
| Tools | [![Documentation](https://img.shields.io/badge/MkDocs-magenta?style=flat-square&color=deepskyblue&logo=markdown&labelColor=gray)](https://squidfunk.github.io/mkdocs-material/) [![Linter](https://img.shields.io/endpoint?style=flat-square&url=https://raw.githubusercontent.com/charliermarsh/Ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/Ruff) [![Dependency Manager](https://img.shields.io/badge/uv-mediumpurple?style=flat-square&logo=uv&labelColor=gray)](https://docs.astral.sh/uv/) [![Pre-commit](https://img.shields.io/badge/pre--commit-darkolivegreen?style=flat-square&logo=pre-commit&logoColor=white&labelColor=gray)](https://github.com/TezRomacH/python-package-template/blob/master/.pre-commit-config.yaml) [![CI](https://img.shields.io/badge/GitHub_Actions-forestgreen?style=flat-square&logo=githubactions&labelColor=gray&logoColor=white)](https://github.com/features/actions) [![Editor Settings](https://img.shields.io/badge/Editor_Config-paleturquoise?style=flat-square&logo=editorconfig&labelColor=gray)](https://editorconfig.org/) [![Repository Template](https://img.shields.io/badge/snickerdoodle-bisque?style=flat-square&logo=cookiecutter&labelColor=gray)](https://www.github.com/WithPrecedent/snickerdoodle) [![Dependency Maintainer](https://img.shields.io/badge/dependabot-forestgreen?style=flat-square&logo=dependabot&logoColor=white&labelColor=gray)](https://github.com/dependabot) |
| Compatibility | [![Compatible Python Versions](https://img.shields.io/pypi/pyversions/amos?style=flat-square&color=cornflowerblue&label=Python&logo=python&logoColor=yellow)](https://pypi.python.org/pypi/amos/) [![Linux](https://img.shields.io/badge/Linux-lightseagreen?style=flat-square&logo=linux&labelColor=gray&logoColor=white)](https://www.linux.org/) [![MacOS](https://img.shields.io/badge/MacOS-antiquewhite?style=flat-square&logo=apple&labelColor=gray)](https://www.apple.com/macos/)  [![Windows](https://img.shields.io/badge/Windows-blue?style=flat-square)](https://www.microsoft.com/en-us/windows?r=1) |
| Stats | [![PyPI Download Rate (per month)](https://img.shields.io/pypi/dm/amos?style=flat-square&color=cornflowerblue&label=Downloads%20💾&logo=pypi&logoColor=yellow)](https://pypi.org/project/amos) [![GitHub Stars](https://img.shields.io/github/stars/WithPrecedent/amos?style=flat-square&color=forestgreen&label=Stars%20⭐&logo=github)](https://github.com/WithPrecedent/amos/stargazers) [![GitHub Contributors](https://img.shields.io/github/contributors/WithPrecedent/amos?style=flat-square&color=forestgreen&label=Contributors%20🙋&logo=github)](https://github.com/WithPrecedent/amos/graphs/contributors) [![GitHub Issues](https://img.shields.io/github/issues/WithPrecedent/amos?style=flat-square&color=forestgreen&label=Issues%20📘&logo=github)](https://github.com/WithPrecedent/amos/graphs/contributors) [![GitHub Forks](https://img.shields.io/github/forks/WithPrecedent/amos?style=flat-square&color=forestgreen&label=Forks%20🍴&logo=github)](https://github.com/WithPrecedent/amos/forks) |
| | |


## What is amos?

<p align="center">
<img src="https://media.giphy.com/media/rSi5EIResK2kW2B1NB/giphy.gif" height="300"/>
</p>

**A**utomated **M**odeling for **O**pen **S**cholarship.
Like the Rocinante's
mechanic in *The Expanse*, `amos` handles the dirty jobs of a data science
project: cleaning, describing, splitting, preprocessing, modeling, evaluating,
and drawing. It is designed for academic research, where a result has to be
explained and reproduced.

You describe a study in a plain settings file (or a Python `dict`). `amos`
builds the workflow with [chrisjen](https://github.com/WithPrecedent/chrisjen),
runs it on your data, compares every combination of the methods you list, and
records everything needed to reproduce the result.

```python
import sklearn.datasets

import amos

cancer = sklearn.datasets.load_breast_cancer(as_frame = True)
settings = {
    "general": {"seed": 43},
    "cancer_project": {"cancer_workers": "analyst, critic"},
    "analyst": {
        "design": "experiment",
        "criterion": "roc_auc",
        "steps": "split, scale, model",
        "split_techniques": "stratified",
        "scale_techniques": "standard, min_max",
        "model_techniques": "baseline, logit",
    },
    "critic": {"techniques": "scorecard, confusion"},
}
project = amos.Project.create(settings, item = cancer)
print(project.result)
# Dataset(rows=569, columns=31, label='target', task='classify', train=426, test=143, model=LogisticRegression)
```

That project split the data, tried four combinations of scaling and models,
kept the one with the best ROC AUC on the test rows, and scored it. The
comparison of all four is stored in a table:

```python
table = project.result.tables["analyst_comparison"]
print(list(table.columns))
# ['rank', 'score', 'roc_auc']
print(table["roc_auc"].iloc[-1])
# 0.5
```

## Why use amos?

<p align="center">
<img src="https://media.giphy.com/media/AvsNp0GaQ8GkOPZ5hy/giphy.gif" height="300"/>
</p>

### Intuitive

`amos` is accessible to researchers at all levels. It uses one vocabulary for every stage of a
project. A project is made of **workers** (its stages), which use
**techniques** (the actual methods), sometimes organized in **steps**. The
settings file only names things, so it doubles as a readable description of
your methods:

| Stage | What it does | Techniques |
| --- | --- | --- |
| wrangler | Cleans the data. | `drop_duplicates`, `drop_missing`, `filter_rows`, `auto_categorize`, ... |
| explorer | Describes the data in tables. | `summarize`, `frequencies`, `correlations`, `missing_values`, `kaplan_meier`, ... |
| analyst | Splits, preprocesses, and models the data, or estimates causal effects. | `stratified`, `median_impute`, `standard`, `one_hot`, `date_parts`, `smote`, `logit`, `catboost`, `fixest`, `cox`, `partially_linear`, ... |
| critic | Evaluates the model. | `scorecard`, `roc_auc`, `fairness`, `conformal`, `permutation_importance`, `shap_importance`, ... |
| artist | Draws figures. | `roc_curve`, `calibration_curve`, `pair_plot`, `coefficient_plot`, `partial_dependence`, `shap_beeswarm`, `shap_waterfall`, `survival_curves`, ... |

The stages are only a convention: any technique can be used in any worker, and
you can name your workers whatever you like.

### Reproducible

<p align="center">
<img src="https://media.giphy.com/media/lIz5wEPUomv6bvok31/giphy.gif" height="300"/>
</p>

Open scholarship means that others can check your work. `amos` helps by:

* Passing one seed to every tool that takes a `random_state`, so a projectngives the same result every time.
* Applying a project to a copy of your data, so running it twice gives thensame answer and your data is never changed.
* Keeping a history of every technique that was applied, with the exact tool and parameters it used.
* Isolating training and testing data. Every scaler, encoder, imputer, and sampler is fitted to the training rows and then applied to the test rows, so nothing leaks from the test set.
* All settings and parameters can be implemented through a single settings file that will produce the same results on any computer.

### Universal

`amos` wraps the major Python data science packages behind one interface, so you can compare their methods side by side without learning each package's quirks:

| Package | What `amos` uses it for |
| --- | --- |
| [scikit-learn](https://scikit-learn.org) | Splitting, imputing, scaling, feature selection, models, metrics, and permutation importance. |
| [category_encoders](https://contrib.scikit-learn.org/category_encoders/) and [skrub](https://skrub-data.org) | Target, weight of evidence, and a dozen other data encoders. |
| [imbalanced-learn](https://imbalanced-learn.org) | SMOTE and other ways to balance the classes of the training rows. |
| [xgboost](https://xgboost.readthedocs.io), [lightgbm](https://lightgbm.readthedocs.io), and [catboost](https://catboost.ai) | Gradient boosting. |
| [InterpretML](https://interpret.ml) | Explainable boosting machines: accurate models whose every effect can be shown. |
| [TabPFN](https://github.com/PriorLabs/TabPFN) | A pretrained model that is often the most accurate on small data. |
| [Optuna](https://optuna.org) | Hyperparameter searches that learn from each try. |
| [statsmodels](https://www.statsmodels.org) and [pyfixest](https://py-econometrics.github.io/pyfixest/) | Regressions with standard errors, p-values, and confidence intervals, including fixed effects and clustered standard errors. Survival analysis: Kaplan-Meier curves and Cox regression of the time until an event. |
| [DoubleML](https://docs.doubleml.org) | Causal effects of a treatment, estimated with double machine learning. |
| [fairlearn](https://fairlearn.org) | Fairness metrics that compare a model across groups. |
| [MAPIE](https://mapie.readthedocs.io) | Conformal prediction: intervals and sets with a known rate of coverage. |
| [shap](https://shap.readthedocs.io) and [eli5](https://eli5.readthedocs.io) | Explaining models with SHAP values (and shap's plots of them) and weights. |
| [matplotlib](https://matplotlib.org) and [seaborn](https://seaborn.pydata.org) | Figures and other visualizations. |
| [great_tables](https://posit-dev.github.io/great-tables/) and [python-docx](https://python-docx.readthedocs.io) | Scorecards as HTML tables and Word documents. |



### Robust and Transparent

A result that depends on one arbitrary choice of preprocessing or model is
fragile. The `experiment` design tries **every combination** of the techniques
you list for each step and reports how each one did, so readers can see that
your conclusions are robust (or not). The critic's `scorecard` puts every
combination side by side, on every metric, in a table ready for a paper (as
csv, Markdown, LaTeX, HTML, Word, or an image).

Name the columns that identify groups of people or places as `groups`, and `amos` keeps them out of the model's
features while it uses them to check fairness across groups, to add fixed
effects and cluster standard errors, and to keep each group in one set when
it splits the data. Any step can include `none`, which tests
whether a technique helps at all.

### Extensible

Every technique is a small Python class that is added to a library as soon as
it is defined, so it can be named in settings right away. To wrap a tool that
`amos` does not include, name its import path:

```python
transformer = amos.Transformer(
    name = "yeo_johnson", contents = "sklearn.preprocessing.PowerTransformer")
print(transformer.name)
# yeo_johnson
```

## Getting started

### Requirements

`amos` requires Python 3.11 or later. It runs on Linux, macOS, and Windows. It
is built on [chrisjen](https://github.com/WithPrecedent/chrisjen), `numpy`,
`pandas`, and `scikit-learn`, which are installed automatically.

### Installation

To install `amos`, use `pip`:

```sh
pip install amos
```

The other packages that `amos` wraps are optional. A technique imports its
package only when it is used, and tells you which extra to install if the
package is missing. Install them all (except TabPFN) with `pip install
amos[all]`, or choose:

| Extra | Installs | For |
| --- | --- | --- |
| `boosting` | xgboost, lightgbm, catboost | Gradient boosting models. |
| `causal` | DoubleML | Causal effects (`partially_linear` and `interactive_regression`). |
| `encoders` | category_encoders, skrub | Target encoders, and encoders for text and dates. |
| `explain` | shap, eli5, InterpretML | `shap_importance`, the SHAP plots (such as `shap_beeswarm`), `explain_weights`, and `explainable_boosting`. |
| `fairness` | fairlearn | Fairness metrics and the `fairness` table. |
| `plots` | matplotlib, seaborn | The artist's figures, and scorecards as images. |
| `polars` | Polars, pyarrow | Reading Polars data frames. |
| `sampling` | imbalanced-learn | Samplers such as `smote`. |
| `statistics` | statsmodels, pyfixest | `ols`, `glm`, and `fixest` models with inference. |
| `survival` | statsmodels | `cox`, `kaplan_meier`, `survival_curves`, and `concordance`. |
| `tables` | great_tables | Scorecards as HTML tables. |
| `tuning` | Optuna | `search = "optuna"` for any model. |
| `uncertainty` | MAPIE | `conformal` intervals and sets. |
| `word` | python-docx | Scorecards as Word documents. |
| `tabpfn` | TabPFN | The `tabpfn` model. It needs PyTorch, which is large, so it is not part of `all`. |

### Usage

<p align="center">
<img src="https://media.giphy.com/media/3ornk6qNeKHAevVcRi/giphy.gif" height="300"/>
</p>

#### Describe your study

Most users describe a study in a settings file. This is the Wisconsin breast
cancer study from above, with every stage, as an ini file:

<!-- file: cancer.ini -->
```ini
[general]
seed = 43
label = target

[cancer_project]
cancer_workers = wrangler, explorer, analyst, critic, artist

[wrangler]
techniques = drop_duplicates, drop_constant

[explorer]
techniques = summarize, missing_values, label_balance

[analyst]
design = experiment
criterion = roc_auc
steps = split, scale, sample, model
split_techniques = stratified
scale_techniques = standard, min_max
sample_techniques = none, smote
model_techniques = logit, random_forest

[critic]
techniques = scorecard, confusion, permutation_importance

[artist]
techniques = roc_curve, importance_plot

[random_forest_parameters]
n_estimators = 100

[roc_curve_parameters]
title = How well the best model separates the classes

[importance_plot_parameters]
title = The ten features that matter most
limit = 10
```

The "general" section names the label (the column that models predict) and
the seed. The project section lists the workers. In the "analyst", each
`{step}_techniques` setting lists alternatives, and the `experiment` design
tries every combination: 2 scalers, 2 samplers, and 2 models make 8
combinations. Parameters for any technique go in a `{technique}_parameters`
section.

#### Run it

Pass the settings and the data (a `pandas.DataFrame`, a path to a data file,
or a scikit-learn dataset) to `Project.create`:

```python
project = amos.Project.create("cancer.ini", item = cancer, id = "first_run")
result = project.result
print(result.metrics["roc_auc"] > 0.95)
# True
print(sorted(result.figures))
# ['importance_plot', 'roc_curve']
```

The `artist` drew two figures, titled by the `roc_curve_parameters` and
`importance_plot_parameters` sections. The ROC curve shows how well the best
combination tells the two kinds of tumors apart on the test rows it never
learned from. The importance plot shows how much its score drops when each
feature is shuffled, which works for any model:

<p align="center">
<img src="https://raw.githubusercontent.com/WithPrecedent/amos/main/docs/img/roc_curve.png" alt="The ROC curve of the best model, with an area under the curve of 0.98" height="320"/>
<img src="https://raw.githubusercontent.com/WithPrecedent/amos/main/docs/img/importance_plot.png" alt="The ten features whose shuffling lowers the model's score the most" height="320"/>
</p>

The result includes a `Dataset`, the fitted `model`, its `predictions`,
`metrics`, `tables`, `figures`, and the `history` of every technique.


#### Compare every branch

The critic's `scorecard` compares every branch of the analysis: each
combination of techniques that the experiment tried, ranked by the criterion,
with every standard metric for the task computed from that branch's own
predictions on the test rows. It can be saved as a csv file, a Markdown table,
a Word document, or an image:

```python
scorecard = project.scorecard
print(len(scorecard.table), list(scorecard.table.columns[:5]))
# 8 ['rank', 'split', 'scale', 'sample', 'model']
scorecard.to_csv("scorecard.csv")
scorecard.to_markdown("scorecard.md")
scorecard.to_word("scorecard.docx")
scorecard.to_image("scorecard.png")
```

This is the scorecard of the study, saved as an image. The best branch is
shaded:

<p align="center">
<img src="https://raw.githubusercontent.com/WithPrecedent/amos/main/docs/img/scorecard.png" alt="A table of the eight branches of the study, ranked by ROC AUC, with the scaler, sampler, and model of each and seven metrics" width="100%"/>
</p>

#### Export the results

`export` saves everything needed to report and reproduce the results in a
folder named for the run, including the scorecard in all four formats:

```python
folder = project.export()
print(sorted(path.name for path in folder.iterdir()))
# ['environment.json', 'figures', 'history.json', 'metrics.csv', 'predictions.csv', 'report.txt', 'scorecard.csv', 'scorecard.docx', 'scorecard.md', 'scorecard.png', 'settings.json', 'tables']
```

There is much more to `amos`, including hyperparameter searches, statistical
inference with `statsmodels`, every technique and its parameters, and how to
write your own. See the [documentation](https://WithPrecedent.github.io/amos),
especially the [tutorial](https://WithPrecedent.github.io/amos/tutorial/), the
[advanced user guide](https://WithPrecedent.github.io/amos/advanced/), the
[technique catalog](https://WithPrecedent.github.io/amos/catalog/), and the
[recipes](https://WithPrecedent.github.io/amos/recipes/).

## Contributing

Contributors are always welcome. Feel free to grab an
[issue](https://www.github.com/WithPrecedent/amos/issues) to work on or make a
suggested improvement. If you wish to contribute, please read the
[Contribution
Guide](https://www.github.com/WithPrecedent/amos/blob/main/CONTRIBUTING.md) and
[Code of
Conduct](https://www.github.com/WithPrecedent/amos/blob/main/CODE_OF_CONDUCT.md).

## Similar Projects

<p align="center">
<img src="https://media.giphy.com/media/l44QEvSn731HrwyGY/giphy.gif" height="300"/>
</p>

* [scikit-learn](https://scikit-learn.org): its `Pipeline` and
  `GridSearchCV` combine preprocessing and models and search over
  hyperparameters. `amos` uses scikit-learn throughout, and adds named
  columns, supervised encoders, samplers, comparisons of whole combinations
  of methods, and settings files.
* [PyCaret](https://pycaret.org): a low-code library that compares many
  models with a few function calls. It automates more choices for you, while
  `amos` asks you to list your methods so that they can be reported.
* [Kedro](https://kedro.org): a framework for reproducible data pipelines,
  with more structure for large, long-running projects.
* [MLflow](https://mlflow.org) and [DVC](https://dvc.org): track experiments,
  data, and models over time. They work well alongside `amos`.

## License

Use of this repository is authorized under the [Apache Software License
2.0](https://www.github.com/WithPrecedent/amos/blob/main/LICENSE).
