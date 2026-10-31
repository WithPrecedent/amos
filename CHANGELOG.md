# Changelog

All notable changes to this project will be documented in this file.

<!-- insertion marker -->

## Unreleased

* Added selectors, a new genre of analyst techniques that keep some of the
  features and remove the rest (the new `selectors` module and `Selector`
  class), usually in a step of their own such as "select". `k_best`,
  `select_percentile`, `variance_threshold`, `sequential_selection`, and
  `exhaustive_selection` moved there from the reducers (their names in
  settings are unchanged, but their classes are now in `amos.selectors`, and
  the `Reducer` genre has only `pca_reduce`). New selectors wrap the rest
  of scikit-learn's `feature_selection` module: `select_fpr`, `select_fdr`,
  `select_fwe`, `generic_univariate`, `select_from_model`,
  `recursive_elimination`, `recursive_elimination_cv` (which adds a
  "{name}_search" table of the score of each number of features), and
  `sequential_selection_sk` (which needs no optional package). The
  selectors that fit a model take the name of a model in the library as
  their "estimator" (such as `random_forest`, which is built for the task),
  and a selector that would keep no features raises an error.
* Added feature groups: each group named in a new "feature_groups" section of
  the settings (such as `demographics = age, sex, race`) is a technique that
  keeps only its features, so an experiment can compare groups of features
  that you chose (`select_techniques = demographics, case_facts`). A listed
  column that has been encoded keeps the columns made from it (so "court"
  keeps "court_federal" and "court_state" after `one_hot`). The
  `feature_group` technique does the same with the features passed as its
  "columns".
* The scorecard of a dataset that did not come from an experiment lists its
  selectors as a step.

## 0.6.0

* Added `Clerk` (in the new `clerks` module), a `nagata.FileManager` that is
  now the `clerk` of every `Project` and loader. It keeps data and results
  outside a project's code, in the "workspace" that holds the "code"
  folder: data in its "data" folder (or in a subfolder of it named by the
  new "data_subfolder" setting of the "files" section, such as
  "crime/nibrs"), and each run's results in its "studies" folder. The
  workspace is the folder that holds the nearest "code" folder above the
  current folder (or the current folder, if there is none), unless the
  `AMOS_WORKSPACE` environment variable or `options._DEFAULT_ROOT` names
  another. Before, a project kept its files in the current folder, so a
  project that relied on that should set "root_folder", "input_folder", or
  "output_folder" in its "files" section.

## 0.5.2

* Added named pipelines (the new `pipelines` module and `Pipeline` class).
  `pipeline = classification` in the "general" section (or the project's
  section) fills in the settings of the analyst, critic, and artist, so a
  study can be a label, a seed, and a pipeline. The pipelines are
  `classification`, `regression`, and `count_regression`, which compare
  models in an experiment, and `classification_inference`,
  `regression_inference`, `count_inference`, `panel_classification`, and
  `panel_regression`, which fit one model for its coefficients (without
  scaling or resampling, and with each category encoded against a
  reference level). Settings that you write win over the pipeline's, a
  worker's own section can name a pipeline for that worker, the pipeline
  sets "task" if it is not set, and the filled-in settings are in
  `Project.idea` and the exported "settings.json". Add a `Pipeline` to
  `pipelines.PIPELINES` to name your own.
* Added three choosers, which the pipelines use and "auto" now resolves to
  in steps named "impute", "encode", and "evaluate" (and in a worker named
  "critic"): `auto_imputer` fills missing values (with `iterative_impute`
  when many numeric values are missing, `median_impute` otherwise, and
  `mode_impute` for categories); `auto_encoder` encodes each categorical
  feature by its number of levels (`one_hot` for 15 or fewer, `target` for
  more with a label of two classes or a regressed one, `count` for a label
  of more classes, `tfidf` for free text, and `date_parts` for dates); and
  `auto_evaluator` applies the evaluators that suit the task and the model.
* `auto_model` knows the needs of `ols`, `logit_sm`, `probit`, and
  `mnlogit` (such as the number of classes), which it considers when they
  are named in "choices".
* Fixed `smote_tomek` and `smote_enn` marking every training row as
  synthetic, because they remove some of the original rows after adding
  rows. Validators then had no real rows to cross-validate, and choosers
  profiled none. Each row they keep that equals an original row is now
  real.
* Fixed `fixest` predicting missing values (and so a `scorecard` failing
  with "Input contains NaN") for rows whose level of a fixed effect has no
  estimate: a level that was not in the training rows, or one that
  pyfixest dropped (a singleton, or a level separated by having rows of
  only one class). Such a row is now predicted with that fixed effect at
  the average of its estimated levels. The estimates of the levels are in
  the model's `effect_levels_`. A `fixest` model can therefore be
  validated by `group_k_fold` (each held-out group is predicted with the
  average fixed effect), which used to fail with "could not predict".
* Fixed `permutation_importance`, `shap_partial_dependence`, and
  `decision_regions` failing for a model that needs columns besides its
  features, such as the fixed effects and cluster of a `fixest` model.
  `permutation_importance` and `shap_partial_dependence` hold those columns
  at each row's own values (as `shap_importance` does), and
  `decision_regions` holds them at their most common values in the real
  training rows (as it holds the other features at their medians).
  `permutation_importance` also works for models that are not scikit-learn
  estimators, which scikit-learn's scorers rejected.
* Fixed the report of a project saying that an experiment that conserves
  memory chose "the best of 1 paths". It now gives the number of paths
  compared.
* Inferers (such as `dr_learner` and `partially_linear`) now leave out the
  rows that a sampler (such as `random_over` or `smote`) added to the
  training rows, which counted some rows more than once and biased the
  estimates and their standard errors. The data is put back afterwards.

## 0.5.1

* Added a `conserve_memory` setting (from `chrisjen`'s new option for
  comparing workers), for an experiment's section or for every experiment in
  the "general" section. An experiment that conserves memory scores each
  combination as soon as it is tried and keeps only the best whole dataset
  so far, rather than every combination's dataset until the last one is
  done, which ran out of memory with many combinations of a large dataset.
  Its comparison table, scores, and branches are the same, but only the
  winner's branch has its whole `dataset`, so a `regression_table` has only
  the winner's model. `Experiment.outcomes` has the outcome of each
  combination (its label and group columns, predictions, probabilities, and
  metrics). Requires `chrisjen` 0.4.0 or later.

## 0.5.0

* A validator resamples only the model's features when it applies the
  analysis's samplers to each fold, and gives each copied row the other
  columns that the model needs (such as the fixed effects and cluster of a
  `fixest` model) from the row it copies, as a sampler does for the groups.
  Before, `smote` and the other samplers that interpolate numbers failed in
  a validator of a `fixest` model whose fixed effects were text.
* `shap_importance` (and the SHAP plots, which use it) explains a model that
  needs columns besides its features, such as a `fixest` model, by
  explaining each row with those columns held at its own values, so its
  SHAP values are the effects of its features given its fixed effects.
  Before, it failed because the explained rows had no fixed effects.
* `feature_importance` (and `importance_plot`) leaves out the features that
  a model dropped, such as the collinear features that pyfixest drops,
  rather than failing because there were fewer coefficients than features.

## 0.4.1

* `regression_table` takes "sets" of features (such as the dummy variables
  of one category, as "family, partner, stranger; under_12, teen" in
  settings), whose members' average marginal effects are each the change
  from the set's reference (every member 0) to that member alone, rather
  than from moving each dummy on its own, which made rows that cannot
  happen.
* Added techniques that wrap mlxtend, which the new `mlxtend` extra (and
  `all`) installs: `sequential_selection` and `exhaustive_selection`,
  reducers that search for the set of features with which a model (a
  logistic or linear regression, or "estimator") scores best in
  cross-validation, folding by the dataset's first group as a model's search
  does, and store the score of each set as "{name}_search";
  `association_rules`, a describer of which values (of the categorical and
  boolean columns, and of a classified label) tend to appear in the same
  rows; `bias_variance`, an evaluator that splits the model's expected loss
  on the test rows into its bias and variance; and `decision_regions`, a plot
  of the class that a classifier predicts across two features.
* Added dtreeviz's plots of decision trees, which the `explain` extra now
  installs: `tree_leaf_sizes`, `tree_leaf_distributions`, `tree_leaf_purity`,
  `tree_feature_space`, and `tree_prediction_path`. They draw a
  `decision_tree` or one tree of a `random_forest`, `extra_trees`, or
  `adaboost` model (and `tree_leaf_sizes` and `tree_leaf_distributions` one
  tree of an `xgboost` or `lightgbm` model), in the style of the other
  figures. dtreeviz's drawing of a whole tree is not wrapped, since it needs
  Graphviz and makes only SVG files; `tree_plot` draws one.
* `models.search_folds` returns the folds of a hyperparameter search (which
  `Model._folds` now calls), so that other techniques can fold by groups as
  a search does.
* Plots that draw with `matplotlib.pyplot` (shap's, and now mlxtend's and
  dtreeviz's) are lent their figure by `plots._lent`, which also keeps
  `pyplot.show` from opening a window while they draw.
* A model's hyperparameter search ("grid", "random", or "optuna") keeps the
  rows of each group of the dataset's first group column in one fold, as the
  group validators do, when the dataset has groups: with
  `StratifiedGroupKFold` for a classifier and `GroupKFold` for a regression.
  Before, its folds divided the rows at random, so rows of one group (such as
  the votes of the judges of one case) could be on both sides of a fold and
  the search scored a model on rows it had partly seen. A row with a missing
  group is a group of its own.
* Brought in the CI fixes of `snickerdoodle` 0.4.3: GitHub releases are made
  with the GitHub CLI (in place of the deprecated `gh-action-auto-release`
  Action), with the released version's section of CHANGELOG.md as their
  notes, only from `main` and after the tests and linters pass. The test
  matrix now runs every Python version it names, instead of recreating the
  environment with Python 3.11 at `uv run`.

## 0.4.0

* A technique now writes `apply` instead of `implement`, following
  `chrisjen` 0.3.0 (now the oldest version that `amos` accepts), whose nodes
  all do. Calling `apply` works as before: it accepts
  anything that `Dataset.create` accepts, fills in the technique's
  parameters, and records the technique in the dataset's `history`. That
  work moved to `Operation._call_apply`, which `chrisjen` calls around each
  `apply` (but not around a `super().apply` call from a subclass). A
  technique that wrote `implement` must rename it to `apply`.
* Added the `auto` module, which lets a step's techniques be "auto" in the
  settings: `amos` then chooses techniques that suit the data when the
  workflow reaches the step. "auto" works in a step that samples, models, or
  plots (found from the name of the step, or of a worker such as "artist"),
  and `auto_sampler`, `auto_model`, and `auto_plot` can be named in any
  step. `auto_sampler` resamples only imbalanced classes, with a sampler that
  suits their size, overlap, and features. `auto_model` rules out the models
  that cannot use the features or whose assumptions the data does not meet
  (normal errors, collinear features, the dispersion of counts, and so on),
  cross-validates the rest, and fits the simplest model within one standard
  error of the best. `auto_plot` draws the data before a model is fitted,
  and the plots that suit the task and the model after. Each records a table
  of the techniques it considered and why each was or was not chosen, and
  `amos.auto.profile` returns what it found about the data.
* Fixed validators, `conformal`, `explain_weights`, `feature_importance`, and
  the marginal effects of `regression_table` for `lightgbm` models whose
  feature names have spaces. lightgbm replaces the spaces with underscores,
  so the names it reports were not columns of the data: validators failed,
  and the other techniques named features that the data did not have (and
  `regression_table` left them out). `utilities.feature_names` now gives
  back the columns that a model was fitted to.
* `lightgbm` is now quiet, as its default "verbose" of -1 meant it to be.
  The setting never reached lightgbm, which takes it only through
  `**kwargs`, so lightgbm printed a warning for each tree it could not split
  further. A model's `keyword_parameters` now names such parameters.
* Added the `regression_table` evaluator, which reports every model the way
  Stata, R, and statsmodels report regressions: a column for each branch of
  the experiment (or for the final model), each feature's estimate with stars
  and its standard error in parentheses, and the kind of estimates, the
  number of observations, and cross-validated metrics below them. A model
  that reports its coefficients (the statsmodels and pyfixest models) shows
  them. Any other model shows each feature's average marginal effect, with
  standard errors from refitting the model to bootstrap samples of the
  training rows ("simulations", 100 by default). Set "marginal" to true to
  show marginal effects for every model. `RegressionTable.create` makes one
  from a dataset, an applied project, or several datasets, and it is saved
  in the same formats as a scorecard. Word documents, images, and `to_text`
  look like Stata's output of `regress` (a block for each model, with its
  statistics above a table of "Coef." or "dy/dx", "Std. Err.", "t" or "z",
  the p-value, and the 95% confidence interval), and csv, Markdown, LaTeX,
  and HTML put the models side by side. Set "style" to "columns" to put them
  side by side in Word documents and images too. `Project.export` saves each
  regression table in the workflow. The README shows the tables of the best
  `logit` and `xgboost` branches of its study.
* Each `Branch` of an experiment now keeps a weak reference to the whole
  dataset that it made (`Branch.dataset`), so the fitted model of every
  branch can be reached after the experiment. Copies and pickles of a branch
  do not keep it.
* The scorecard's exports are now shared with the regression table. LaTeX
  tables now escape "<" and ">".

* A path to a data file passed as a project's `item`, or to
  `Dataset.create`, is now read with `nagata`'s formats, just as `load_file`
  reads it, so it may be in any format that `nagata` knows (such as json
  lines, fixed-width, HDF, ORC, XML, or SAS) or compressed (such as
  "cases.csv.gz"). A project reads it with its `clerk`. Before, it was read
  with `amos`'s own list of 11 `pandas` readers. A format added to `nagata`
  (a subclass of `nagata.base.FileFormat`) is now read by an item too.
* The error for a file whose format is not known from its name now lists the
  extensions that `nagata` knows.

## 0.3.1

* Added the `use_gpu` setting to the "general" section, which fits the
  models that can use a GPU (`xgboost`, `catboost`, `lightgbm`, and
  `tabpfn`) on an NVIDIA GPU. It is false unless the settings set it to
  true, and it is kept in the new `Dataset.use_gpu`. `amos` finds GPUs by
  asking NVIDIA's driver, so it needs no other package. If there is no GPU,
  or a model cannot be fitted on it, the model warns and is fitted on the
  CPU. `export` records the GPUs in "environment.json".
* DoubleML's and causalml's inferers fit their `outcome_model` and
  `treatment_model` on the GPU too when `use_gpu` is true. Those packages
  fit the models themselves, so each model that can use a GPU is first
  fitted to a few rows on it, and if that fails, it warns and the inferer
  fits that model on the CPU.
* `tabpfn` is now fitted on the CPU unless `use_gpu` is true, as a model or
  as an inferer's model. Before, it used a GPU whenever PyTorch found one.
* Models now record their device in their history's parameters (such as
  xgboost's "device" or catboost's "task_type").

## 0.3.0

* Fixed the `stratified` and `train_test` splitters failing ("only integer
  scalar arrays can be converted to a scalar index") when the index of the
  data was text and pyarrow was installed. `pandas` stores text with
  pyarrow when it is installed, and scikit-learn cannot select rows from an
  index that is stored that way. The splitters now give scikit-learn the
  positions of the rows to divide, as `group_split` already did, and look
  up their index labels afterward. Every split has the same rows as before.
* Added a `CITATION.cff` file and a citation section to the README
* Changed the development status to Beta
* Made the README badges larger

## 0.2.7

* Updated to the `snickerdoodle` 0.4.0 template
* Made `ruff` prefer single quotes
* Made `ruff` ignore boolean default positional arguments (`FBT002`)
* Excluded `uv.lock` from the large file check

## 0.2.6

* Added mergers, a new genre for the wrangler, for data from tables that
  are not organized the same way. A merger matches each row of the data to
  a row of another table (its "source": a data file, which is loaded with
  the project's clerk, or a table) and adds that row's columns.
  `merge_keys` matches the row with the same keys, `merge_nearest` the row
  with the nearest date or number, and `merge_ranges` the row whose range
  has a date or number in it. `merge_summary` adds summaries of every row
  with the same keys. A merger never adds, removes, or reorders rows, and
  records how many rows it matched in the dataset's history. Every merger
  takes "columns", "prefix", and "indicator".
* Added shapers, a new genre for the wrangler, which change what a row is.
  `wide_to_long` stacks columns into rows and `long_to_wide` spreads rows
  into columns, with "stubs" for sets of columns such as "income_2019" and
  "income_2020". `lists_to_rows` makes a row for each item of a list (such
  as each judge of a case), and `group_rows` makes one row for each group
  of rows, with summaries of its columns. A shaper comes before the data is
  split, and takes "label", "task", and "groups" for the reshaped data.
* In Python, the genres are `amos.Merger` (whose subclasses write `match`)
  and `amos.Shaper` (whose subclasses write `shape`), in `amos.mergers` and
  `amos.shapers`. The advanced user guide has new sections on merging and
  reshaping data.
* Fixed `Project.export` failing ("keys must be str, int, float, bool or
  None") when the dataset's history had keys that a json file cannot
  store. A sampler recorded the number of rows of each class by `numpy`
  values when the label was `pandas` booleans that allow missing values
  (the "boolean" type). Samplers now record the classes as plain Python
  values, and `export` saves any other such key as its plain value or its
  `str`, as it already saved values.

## 0.2.5

* Renamed effects to inferers and moved them from the analyst to the
  critic, since they judge what the data can say about causes. In Python,
  `amos.effects` is now `amos.inferers` and `amos.Effect` is now
  `amos.Inferer`, and the genre is "inferer" in the library. The names of
  `partially_linear` and `interactive_regression` in settings are
  unchanged. The inferers of each package are a genre of their own
  (`Doubleml`, `Dowhy`, `Causalml`, and `Tigramite`).
* Added the rest of DoubleML's models: `partially_linear_iv`,
  `partially_logistic`, `partially_linear_panel`, `interactive_iv`,
  `potential_outcomes`, `quantile_effects`, `difference_in_differences`,
  `regression_discontinuity`, and `sample_selection`.
  `difference_in_differences` handles two or more periods, units first
  treated in different periods, panels (with "unit"), and repeated
  cross-sections, and stores the effect by the number of periods since the
  treatment as "{name}_periods". `regression_discontinuity` is sharp or
  fuzzy, as the data show, and needs rdrobust.
* Added DoWhy's estimators: `regression_adjustment`, `glm_adjustment`,
  `doubly_robust`, `propensity_matching`, `propensity_stratification`,
  `propensity_weighting`, `distance_matching`, and `instrumental_variable`,
  with bootstrapped confidence intervals ("simulations") and DoWhy's
  refuters ("refuters"), whose results are stored as "{name}_refutations".
* Added causalml's meta-learners: `s_learner`, `t_learner`, `x_learner`,
  `dr_learner`, and `tmle`, which compare each value of the treatment with
  a "control" and store the effect for each row as "{name}_effects".
  causalml's R-learner, causal forest, and doubly robust learner with an
  instrument are left out, since they do not work in causalml 0.17.
* Added tigramite's causal discovery in time series: `pcmci`, `pcmci_plus`,
  and `lpcmci`, which store the causal links between the variables that
  they find, and `time_series_effect`, which estimates the effect of one
  series on the label at a lag.
* The `causal` extra installs DoubleML, tigramite, rdrobust, and networkx,
  and causalml on Python 3.11 and 3.12 or DoWhy on Python 3.13 and later.
  causalml has no version for Python 3.13 and later, and on Python 3.11 and
  3.12 the versions of DoWhy that work with pandas 3 cannot be installed
  with it. causalml's and DoWhy's inferers explain where they cannot be
  used.
* Every class and function that wraps a tool from another package names the
  tool and its package in a "Wraps:" section of its docstring, with a link to
  its documentation, so the API reference shows which tool each technique
  uses.
* Using causalml no longer changes matplotlib's style (causalml sets it to
  "fivethirtyeight" when its meta-learners are imported).
* The advanced user guide's section on causal inference covers every
  inferer, with a table of which to use for each design of study.

## 0.2.4

* Renamed scikit-learn's logistic regression from `sk_logit` to `logit`
  (`amos.models.Logit`), and statsmodels' logistic regression from `logit`
  to `logit_sm` (`amos.models.LogitSM`), whose table of coefficients is now
  "logit_sm_coefficients". Settings that name `sk_logit` must use `logit`,
  and settings that name `logit` for statsmodels' model must use
  `logit_sm`.
* Added mungers, a new genre for the wrangler. A munger changes what is in
  columns, or makes new columns from them, without adding or removing rows,
  a whole column at a time with the vectorized methods of pandas.
  `auto_categorize`, `convert_types`, and `strip_text` moved to the mungers
  from the cleaners (in Python, they are now in `amos.mungers`; their names
  in settings are unchanged). The new mungers are `flag_patterns`,
  `count_patterns`, `map_patterns`, `extract_pattern`, `extract_all`,
  `split_text`, `replace_text`, `normalize_text`, `parse_numbers`,
  `parse_dates`, `parse_booleans`, `map_values`, `coalesce`,
  `combine_flags`, and `derive_columns`. A munger records the columns that
  it changed and created in the dataset's history.
* Figures share one style. By default, it is SciencePlots' "science" and
  "nature" styles (one column of Nature, 3.3 inches wide, with 7-point
  text) and its "bright" color cycle, which people with color blindness
  can tell apart. Every plot takes "style" (which can be "xkcd"), "colors",
  and "latex" (off unless asked for, although "science" turns LaTeX on).
  The defaults are `options._PLOT_STYLE` and `options._PLOT_COLORS`, and
  SciencePlots is in the `plots` extra.
* A plot's usual size is scaled to the width of its style. The SHAP plots
  and `influence_plot`, whose packages fix the sizes of their text, keep
  their usual sizes, and the style's text is enlarged to match.
* `Project.export` saves figures at 300 dots per inch
  (`options._FIGURE_DPI`).
* `shap_bar` and `shap_waterfall` use the first two colors of the style's
  cycle, `shap_waterfall` writes each value beside its arrow, and both no
  longer fail with styles that show minor ticks. The plots that
  scikit-learn and statsmodels draw (`actual_vs_predicted`,
  `residual_plot`, `qq_plot`, and `partial_dependence`) use the colors of
  the style's cycle, and axes of categories and heatmaps have no minor
  ticks.
* The README's study adds xgboost and draws a confusion heatmap and a SHAP
  waterfall. The advanced user guide has new sections on munging data and
  the styles of figures.

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
