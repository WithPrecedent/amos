# Changelog

All notable changes to this project will be documented in this file.

<!-- insertion marker -->

## 0.2.3

* Renamed scikit-learn's logistic regression from `logit` to `sk_logit`, and
  the scikit-learn PCA reducer from `pca` to `pca_reduce`, so that `logit`
  and `pca` are the statsmodels techniques below. Settings that name `logit`
  or `pca` must use the new names to keep their old behavior.
* Replaced lifelines with statsmodels for the survival tools (`cox`,
  `kaplan_meier`, `concordance`, and `survival_curves`), so the `survival`
  extra installs statsmodels. amos now requires pandas 3.
* Added loaders, a new genre for the wrangler: `load_file`, `download`, and
  `openml` load data through the project's clerk (its `nagata.FileManager`).
  Files are found, and downloads are saved, in the clerk's input folder, and
  a project with a loader needs no `item`. Loaders read every row and read
  text as UTF-8.
* Added validators, a new genre for the analyst: `k_fold`,
  `stratified_k_fold`, `group_k_fold`, `stratified_group_k_fold`,
  `repeated_k_fold`, `repeated_stratified_k_fold`, `shuffle_split`,
  `stratified_shuffle_split`, `group_shuffle_split`, `leave_one_group_out`,
  `leave_one_row_out`, and `time_series_split`. After the model, a validator
  refits copies of it on folds of the training rows and scores each fold
  with the critic's metrics. Their means are stored in `metrics` as
  "cv_{metric}", so scorecards and an experiment's comparison show them.
* Datasets record the training rows that a sampler made up (`synthetic`).
  Validators never score them, and apply the samplers again in each fold.
* Added models from statsmodels, each with a table of coefficients: `logit`,
  `probit`, `mnlogit`, `ordinal_regression`, `wls`, `quantile_regression`,
  `robust_regression`, `mixedlm`, `gee`, `binomial_bayes_mixedglm`,
  `poisson`, `negative_binomial`, `generalized_poisson`, and
  `zero_inflated_poisson`. `mixedlm`, `gee`, and `binomial_bayes_mixedglm`
  use the dataset's first group unless "groups" names a column.
* The statsmodels models that cannot estimate a feature that is a
  combination of others (collinear) leave it out, the same way on every
  computer: the model's `dropped_` lists it, and its row in the table of
  coefficients is empty.
* A `glm` or `gee` of a family that regresses refuses a classified label,
  and the models of counts explain how to regress a label of few whole
  numbers. Effects can use the statsmodels models as learners.
* Added `pca` and `factor_analysis` to the critic: statsmodels' analyses of
  the features that the model learned from.
* `influence_plot` now draws a `glm`.
* The README's table of stages lists every technique, divided into the steps
  of each stage, with a link to each one's API documentation. The advanced
  user guide has new sections on loading data and cross-validation.

## 0.2.2

* Added eleven plots from shap to the artist: `shap_bar`, `shap_beeswarm`,
  `shap_violin`, `shap_heatmap`, `shap_decision`, `shap_embedding`,
  `shap_waterfall`, `shap_force`, `shap_scatter` (a dependence plot),
  `shap_partial_dependence`, and `shap_group_difference`. They draw the SHAP
  values that `shap_importance` found (or find them), for the class chosen
  with "category" (by default, the last class). shap draws with
  `matplotlib.pyplot`, so these plots lend it their figure while it draws,
  without opening a window, and leave pyplot as it was. The random numbers
  that shap uses to jitter dots and to bootstrap are seeded with the
  dataset's seed, so the figures are reproducible.
* Added plots from the other packages:
  * scikit-learn: `calibration_curve`, `det_curve`, `partial_dependence`,
    `learning_curve`, `validation_curve`, and `tree_plot`.
  * seaborn: `box_plots`, `count_plots`, `pair_plot`, `label_plot`, and
    `missing_heatmap`, split or colored by the label's classes (or by a
    group, or by "by").
  * statsmodels: `qq_plot` and `influence_plot`.
  * InterpretML: `shape_functions`, the shape functions of an
    `explainable_boosting` model.
  * Plots of the tables made with other packages: `coefficient_plot` (of
    the coefficients of `ols`, `glm`, `fixest`, or `cox`, or of an effect from
    DoubleML), `fairness_plot` (fairlearn), `prediction_intervals` (MAPIE),
    and `search_plot` (of any search, including Optuna's).
* `survival_curves` can add a table of the number of rows at risk
  (`at_risk`).
* A plot can have its own usual size (`Plot.size`), which "width" and
  "height" still change, and a plot whose tool makes its own figure (as
  shap's force plot does) can return that figure from `draw`.
* Moved the technique catalog out of the advanced user guide into its own
  page of the user guide, "Technique Catalog". Its tables are written from
  the techniques' docstrings by `docs/scripts/technique_catalog.py`, and a
  test checks that they are up to date.

## 0.2.1

* Replaced the `scorecard` technique of the critic stage with `Scorecard`,
  which compares every branch of an analysis: one row for each combination of
  techniques that the most recent `experiment` tried, ranked by its criterion,
  with the technique used at each step and every standard metric for the
  task, computed from each branch's own predictions. It still stores the final
  model's scores in the dataset's `metrics`. Its table now has one row for
  each branch (instead of one row for each metric).
* A scorecard can be saved as a csv file (`to_csv`), a Markdown table
  (`to_markdown`), a Word document (`to_word`), or an image (`to_image`), or
  in all four formats at once (`export`). `Scorecard.create` makes one from a
  dataset or an applied project, and `Project.scorecard` returns the
  project's scorecard. `Project.export` saves it in every format.
* An `experiment` now keeps a `Branch` (the technique used at each step, the
  score, and the predictions) for every combination in the winning dataset's
  `branches`.
* Added the `word` extra, which installs python-docx for Word documents.
* Added the scorecard of the README's breast cancer study to the README.
* Added `groups` to `Dataset` (and a "groups" setting in the "general"
  section): columns that identify groups of people or places, which stay in
  the data but are not features. Cleaners and samplers keep them (synthetic
  rows have none), and `group_split` uses the first group by default.
* Added support for 13 more packages, for 139 techniques in ten genres:
  * fairlearn: six fairness metrics (`demographic_parity`, `equalized_odds`,
    and `equal_opportunity`, with a "_ratio" version of each) in the new
    `GroupMetric` genre, and the `fairness` evaluator, which compares the
    model across groups. The `scorecard` adds `demographic_parity` and
    `equalized_odds` for every branch when a dataset has groups and a label
    with two classes.
  * MAPIE: the `conformal` evaluator, which makes prediction intervals (or
    sets of classes) with a known rate of coverage.
  * eli5: the `explain_weights` evaluator.
  * InterpretML: the `explainable_boosting` model, whose term importances
    `feature_importance` reports.
  * CatBoost: the `catboost` model, which uses categorical columns directly.
  * TabPFN: the `tabpfn` model (in its own `tabpfn` extra, which is not part
    of `all` because it needs PyTorch).
  * pyfixest: the `fixest` model, a regression (or logit) with fixed effects
    and clustered standard errors, whose coefficients are stored in a table.
  * lifelines: the `cox` model, the `concordance` metric, the `kaplan_meier`
    describer, and the `survival_curves` plot.
  * DoubleML: the new `Effect` genre, with `partially_linear` and
    `interactive_regression`, which estimate the causal effect of a
    treatment with double machine learning.
  * skrub: the `date_parts`, `gap`, `min_hash`, and `tfidf` encoders.
  * Optuna: `search = "optuna"` for any model.
  * Polars: `Dataset.create` accepts Polars data frames (lazy or not).
  * great_tables: scorecards as HTML tables (`to_html`). Scorecards can also
    be saved as LaTeX tables (`to_latex`), and `export` saves both.
* Added the `causal`, `fairness`, `polars`, `survival`, `tables`, `tuning`,
  `uncertainty`, and `tabpfn` extras, and added packages to the `boosting`,
  `encoders`, `explain`, and `statistics` extras.
* Added `Dataset.dates`, the names of the date and time columns.
* Code that calls pyfixest no longer changes the level or handlers of the root
  logger.
* Added recipes for fairness and for fixed effects, and sections of the
  advanced user guide on groups and fairness, uncertainty, causal effects,
  and survival analysis.
* Left private classes, methods, and functions out of the API documentation.

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
* Added an ROC curve and an importance plot from the README's breast cancer
  study to the README, with `docs/scripts/readme_figures.py` to draw them
  again.
* Added tests that compare `amos` with each package that it wraps, using the
  same data, parameters, and seed.
* Removed the general-purpose containers, converters, factories, and
  registries of 0.1.x. They now live in `bunches`, `camina`, `holden`, and
  `wonka`.
* Moved to the `snickerdoodle` template, based on `uv` and `hatchling`, with
  ruff, mypy, pre-commit, mkdocs, and GitHub Actions. The README and
  documentation examples are run by the unit tests.

## 0.1.10

* The last release of the earlier, general-purpose `amos`, a toolkit of
  containers and converters for Python projects.
