# Changelog

All notable changes to this project will be documented in this file.

<!-- insertion marker -->

## 0.2.0

* Rebuilt `amos` as **A**utomated **M**odeling for **O**pen **S**cholarship: a
  package for reproducible data science in academic research, built on
  `chrisjen` 0.2.0. A study is described in a settings file (or a `dict`),
  which `chrisjen` turns into a workflow of workers, steps, and techniques.
* Added `Dataset`, the item that flows through an `amos` workflow. It keeps
  the data in one `pandas.DataFrame` with the label, the training and test
  rows, the fitted model, its predictions and probabilities, metrics, tables,
  figures, fitted tools, and a `history` of every technique applied.
* Added `Operation`, the base class of every `amos` technique, and 116
  techniques in nine genres, each stored in the `chrisjen` library under its
  snake case name: cleaners (10), describers (6), splitters (4), transformers
  (imputers, scalers, encoders, mixers, and reducers: 37), samplers (9),
  models (19), metrics (17), evaluators (6), and plots (8). They wrap
  scikit-learn, category_encoders, imbalanced-learn, xgboost, lightgbm,
  statsmodels, shap, matplotlib, and seaborn. Optional packages are imported
  only when they are used, and are installed with extras (such as
  `amos[all]`).
* Transformers and samplers learn from the training rows only, keep named
  columns, can be limited to some `columns`, and are given the label when
  they are fitted. Models do both tasks where they can, search for the best
  hyperparameters when asked, and (with `ols` and `glm`) report coefficients
  with standard errors, p-values, and confidence intervals. Every metric is
  also a `chrisjen.Criteria`, so it can be an experiment's criterion.
* Added the `Experiment` design, which tries every combination of
  techniques, keeps the best, and adds a table comparing every combination.
* Added `amos.Project`, which makes data into a `Dataset`, reads "label",
  "task", and "seed" from the "general" section of the settings, applies the
  workflow to a copy of the data, passes the seed to every tool that takes a
  `random_state`, and exports the report, settings, environment, history,
  predictions, tables, and figures.
* Added the `Findings` report.
* Removed the general-purpose containers, converters, factories, and
  registries of 0.1.x. They now live in `bunches`, `camina`, `holden`, and
  `wonka`.
* Moved to the `snickerdoodle` template, based on `uv` and `hatchling`, with
  ruff, mypy, pre-commit, mkdocs, and GitHub Actions. The README and
  documentation examples are run by the unit tests.

## 0.1.10

* The last release of the earlier, general-purpose `amos`, a toolkit of
  containers and converters for Python projects.
