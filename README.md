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

## Why use amos?

<p align="center">
<img src="https://media.giphy.com/media/AvsNp0GaQ8GkOPZ5hy/giphy.gif" height="300"/>
</p>

### Accessible

`amos` is accessible to researchers at all levels. It uses one vocabulary for every stage of a
project. A project is made of **workers** (its stages), which use
**techniques** (the actual methods), sometimes organized in **steps**. The
settings file only names things, so it doubles as a readable description of
your methods:

<table>
<thead>
<tr><th>Stage</th><th>Steps</th><th>Techniques</th></tr>
</thead>
<tbody>
<tr><td rowspan="3" valign="middle">wrangler</td><td>Loaders</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/loaders/#amos.loaders.Download"><code>download</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/loaders/#amos.loaders.LoadFile"><code>load_file</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/loaders/#amos.loaders.Openml"><code>openml</code></a></td></tr>
<tr><td>Cleaners</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/cleaners/#amos.cleaners.DropColumns"><code>drop_columns</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/cleaners/#amos.cleaners.DropConstant"><code>drop_constant</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/cleaners/#amos.cleaners.DropDuplicates"><code>drop_duplicates</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/cleaners/#amos.cleaners.DropMissing"><code>drop_missing</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/cleaners/#amos.cleaners.FilterRows"><code>filter_rows</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/cleaners/#amos.cleaners.KeepColumns"><code>keep_columns</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/cleaners/#amos.cleaners.RenameColumns"><code>rename_columns</code></a></td></tr>
<tr><td>Mungers</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.AutoCategorize"><code>auto_categorize</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.Coalesce"><code>coalesce</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.CombineFlags"><code>combine_flags</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.ConvertTypes"><code>convert_types</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.CountPatterns"><code>count_patterns</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.DeriveColumns"><code>derive_columns</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.ExtractAll"><code>extract_all</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.ExtractPattern"><code>extract_pattern</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.FlagPatterns"><code>flag_patterns</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.MapPatterns"><code>map_patterns</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.MapValues"><code>map_values</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.NormalizeText"><code>normalize_text</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.ParseBooleans"><code>parse_booleans</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.ParseDates"><code>parse_dates</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.ParseNumbers"><code>parse_numbers</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.ReplaceText"><code>replace_text</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.SplitText"><code>split_text</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/mungers/#amos.mungers.StripText"><code>strip_text</code></a></td></tr>
<tr><td>explorer</td><td>Describers</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/describers/#amos.describers.Correlations"><code>correlations</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/describers/#amos.describers.Describe"><code>describe</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/describers/#amos.describers.Frequencies"><code>frequencies</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/describers/#amos.describers.KaplanMeier"><code>kaplan_meier</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/describers/#amos.describers.LabelBalance"><code>label_balance</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/describers/#amos.describers.MissingValues"><code>missing_values</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/describers/#amos.describers.Summarize"><code>summarize</code></a></td></tr>
<tr><td rowspan="10" valign="middle">analyst</td><td>Splitters</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/splitters/#amos.splitters.GroupSplit"><code>group_split</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/splitters/#amos.splitters.Stratified"><code>stratified</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/splitters/#amos.splitters.TimeSplit"><code>time_split</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/splitters/#amos.splitters.TrainTest"><code>train_test</code></a></td></tr>
<tr><td>Imputers</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.IterativeImpute"><code>iterative_impute</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.KnnImpute"><code>knn_impute</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.MeanImpute"><code>mean_impute</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.MedianImpute"><code>median_impute</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.ModeImpute"><code>mode_impute</code></a></td></tr>
<tr><td>Scalers</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Binarize"><code>binarize</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Bins"><code>bins</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Gauss"><code>gauss</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.MaxAbs"><code>max_abs</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.MinMax"><code>min_max</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Normalize"><code>normalize</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Quantile"><code>quantile</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Robust"><code>robust</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Standard"><code>standard</code></a></td></tr>
<tr><td>Encoders</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.BackwardDifference"><code>backward_difference</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.BaseN"><code>base_n</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Binary"><code>binary</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.CatBoost"><code>cat_boost</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Count"><code>count</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.DateParts"><code>date_parts</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Gap"><code>gap</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Hashing"><code>hashing</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Helmert"><code>helmert</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.JamesStein"><code>james_stein</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.LeaveOneOut"><code>leave_one_out</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.MEstimate"><code>m_estimate</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.MinHash"><code>min_hash</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.OneHot"><code>one_hot</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Ordinal"><code>ordinal</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.PolynomialCoding"><code>polynomial_coding</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.SumCoding"><code>sum_coding</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Target"><code>target</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Tfidf"><code>tfidf</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.WeightOfEvidence"><code>weight_of_evidence</code></a></td></tr>
<tr><td>Mixers</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Interactions"><code>interactions</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Polynomial"><code>polynomial</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.Splines"><code>splines</code></a></td></tr>
<tr><td>Reducers</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.KBest"><code>k_best</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.PCAReduce"><code>pca_reduce</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.SelectPercentile"><code>select_percentile</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/transformers/#amos.transformers.VarianceThreshold"><code>variance_threshold</code></a></td></tr>
<tr><td>Samplers</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/samplers/#amos.samplers.Adasyn"><code>adasyn</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/samplers/#amos.samplers.BorderlineSmote"><code>borderline_smote</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/samplers/#amos.samplers.NearMiss"><code>near_miss</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/samplers/#amos.samplers.RandomOver"><code>random_over</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/samplers/#amos.samplers.RandomUnder"><code>random_under</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/samplers/#amos.samplers.Smote"><code>smote</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/samplers/#amos.samplers.SmoteEnn"><code>smote_enn</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/samplers/#amos.samplers.SmoteTomek"><code>smote_tomek</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/samplers/#amos.samplers.TomekLinks"><code>tomek_links</code></a></td></tr>
<tr><td>Models</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Adaboost"><code>adaboost</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Baseline"><code>baseline</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.BinomialBayesMixedglm"><code>binomial_bayes_mixedglm</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Catboost"><code>catboost</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Cox"><code>cox</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.DecisionTree"><code>decision_tree</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.ElasticNet"><code>elastic_net</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.ExplainableBoosting"><code>explainable_boosting</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.ExtraTrees"><code>extra_trees</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Fixest"><code>fixest</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.GEE"><code>gee</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.GeneralizedPoisson"><code>generalized_poisson</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.GLM"><code>glm</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.GradientBoosting"><code>gradient_boosting</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.KNN"><code>knn</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Lasso"><code>lasso</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Lightgbm"><code>lightgbm</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Linear"><code>linear</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Logit"><code>logit</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Mixedlm"><code>mixedlm</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Mnlogit"><code>mnlogit</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.NaiveBayes"><code>naive_bayes</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.NegativeBinomial"><code>negative_binomial</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.NeuralNetwork"><code>neural_network</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.OLS"><code>ols</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.OrdinalRegression"><code>ordinal_regression</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Poisson"><code>poisson</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Probit"><code>probit</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.QuantileRegression"><code>quantile_regression</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.RandomForest"><code>random_forest</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Ridge"><code>ridge</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.RobustRegression"><code>robust_regression</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.SkLogit"><code>sk_logit</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.SVM"><code>svm</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Tabpfn"><code>tabpfn</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.WLS"><code>wls</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.Xgboost"><code>xgboost</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/models/#amos.models.ZeroInflatedPoisson"><code>zero_inflated_poisson</code></a></td></tr>
<tr><td>Validators</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/validators/#amos.validators.GroupKFold"><code>group_k_fold</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/validators/#amos.validators.GroupShuffleSplit"><code>group_shuffle_split</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/validators/#amos.validators.KFold"><code>k_fold</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/validators/#amos.validators.LeaveOneGroupOut"><code>leave_one_group_out</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/validators/#amos.validators.LeaveOneRowOut"><code>leave_one_row_out</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/validators/#amos.validators.RepeatedKFold"><code>repeated_k_fold</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/validators/#amos.validators.RepeatedStratifiedKFold"><code>repeated_stratified_k_fold</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/validators/#amos.validators.ShuffleSplit"><code>shuffle_split</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/validators/#amos.validators.StratifiedGroupKFold"><code>stratified_group_k_fold</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/validators/#amos.validators.StratifiedKFold"><code>stratified_k_fold</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/validators/#amos.validators.StratifiedShuffleSplit"><code>stratified_shuffle_split</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/validators/#amos.validators.TimeSeriesSplit"><code>time_series_split</code></a></td></tr>
<tr><td>Effects</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/effects/#amos.effects.InteractiveRegression"><code>interactive_regression</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/effects/#amos.effects.PartiallyLinear"><code>partially_linear</code></a></td></tr>
<tr><td rowspan="2" valign="middle">critic</td><td>Metrics</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.Accuracy"><code>accuracy</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.AveragePrecision"><code>average_precision</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.BalancedAccuracy"><code>balanced_accuracy</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.Brier"><code>brier</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.CohenKappa"><code>cohen_kappa</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.Concordance"><code>concordance</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.DemographicParity"><code>demographic_parity</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.DemographicParityRatio"><code>demographic_parity_ratio</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.EqualOpportunity"><code>equal_opportunity</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.EqualOpportunityRatio"><code>equal_opportunity_ratio</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.EqualizedOdds"><code>equalized_odds</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.EqualizedOddsRatio"><code>equalized_odds_ratio</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.ExplainedVariance"><code>explained_variance</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.F1"><code>f1</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.LogLoss"><code>log_loss</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.MAE"><code>mae</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.MAPE"><code>mape</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.Matthews"><code>matthews</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.MSE"><code>mse</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.Precision"><code>precision</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.R2"><code>r2</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.Recall"><code>recall</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.RMSE"><code>rmse</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/metrics/#amos.metrics.RocAuc"><code>roc_auc</code></a></td></tr>
<tr><td>Evaluators</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/evaluators/#amos.evaluators.ClassificationReport"><code>classification_report</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/evaluators/#amos.evaluators.Conformal"><code>conformal</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/evaluators/#amos.evaluators.Confusion"><code>confusion</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/evaluators/#amos.evaluators.ExplainWeights"><code>explain_weights</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/evaluators/#amos.evaluators.FactorAnalysis"><code>factor_analysis</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/evaluators/#amos.evaluators.Fairness"><code>fairness</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/evaluators/#amos.evaluators.FeatureImportance"><code>feature_importance</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/evaluators/#amos.evaluators.PCA"><code>pca</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/evaluators/#amos.evaluators.PermutationImportance"><code>permutation_importance</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/evaluators/#amos.evaluators.Scorecard"><code>scorecard</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/evaluators/#amos.evaluators.ShapImportance"><code>shap_importance</code></a></td></tr>
<tr><td>artist</td><td>Plots</td><td><a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ActualVsPredicted"><code>actual_vs_predicted</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.BoxPlots"><code>box_plots</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.CalibrationCurve"><code>calibration_curve</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.CoefficientPlot"><code>coefficient_plot</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ConfusionHeatmap"><code>confusion_heatmap</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.CorrelationHeatmap"><code>correlation_heatmap</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.CountPlots"><code>count_plots</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.DetCurve"><code>det_curve</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.FairnessPlot"><code>fairness_plot</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.Histograms"><code>histograms</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ImportancePlot"><code>importance_plot</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.InfluencePlot"><code>influence_plot</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.LabelPlot"><code>label_plot</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.LearningCurve"><code>learning_curve</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.MissingHeatmap"><code>missing_heatmap</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.PairPlot"><code>pair_plot</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.PartialDependence"><code>partial_dependence</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.PrecisionRecallCurve"><code>precision_recall_curve</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.PredictionIntervals"><code>prediction_intervals</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.QqPlot"><code>qq_plot</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ResidualPlot"><code>residual_plot</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.RocCurve"><code>roc_curve</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.SearchPlot"><code>search_plot</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ShapBar"><code>shap_bar</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ShapBeeswarm"><code>shap_beeswarm</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ShapDecision"><code>shap_decision</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ShapEmbedding"><code>shap_embedding</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ShapForce"><code>shap_force</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ShapGroupDifference"><code>shap_group_difference</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ShapHeatmap"><code>shap_heatmap</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ShapPartialDependence"><code>shap_partial_dependence</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ShapScatter"><code>shap_scatter</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ShapViolin"><code>shap_violin</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ShapWaterfall"><code>shap_waterfall</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ShapeFunctions"><code>shape_functions</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.SurvivalCurves"><code>survival_curves</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.TreePlot"><code>tree_plot</code></a>, <a href="https://WithPrecedent.github.io/amos/reference/amos/plots/#amos.plots.ValidationCurve"><code>validation_curve</code></a></td></tr>
</tbody>
</table>

The stages are only a convention: any technique can be used in any worker, and
you can name your workers whatever you like. And, any step can include `none`, which tests
whether a technique changes the result versus doing nothing at all.

### Reproducible

<p align="center">
<img src="https://media.giphy.com/media/lIz5wEPUomv6bvok31/giphy.gif" height="300"/>
</p>

Open scholarship means that others can check your work. `amos` helps by:

* Passing one seed to every tool that takes a `random_state`, so a project yields the same result every time.
* Applying a project to a copy of your data, so running it twice gives the same answer and your data is never changed.
* Keeping a history of every technique that was applied, with the exact tool and parameters it used.
* Isolating training and testing data. Every scaler, encoder, imputer, and sampler is fitted to the training rows and then applied to the test rows, so nothing leaks from the test set.
* All settings and parameters can be implemented through a single settings file that will produce the same results on any computer.

### Flexible

`amos` wraps the major Python data science packages behind one interface, so you can compare their methods side by side without learning each package's quirks:

| Package | What `amos` uses it for |
| --- | --- |
| [scikit-learn](https://scikit-learn.org) | Splitting, imputing, scaling, feature selection, models, cross-validation, metrics, and permutation importance. |
| [category_encoders](https://contrib.scikit-learn.org/category_encoders/) and [skrub](https://skrub-data.org) | Target, weight of evidence, and a dozen other data encoders. |
| [imbalanced-learn](https://imbalanced-learn.org) | SMOTE and other ways to balance the classes of the training rows. |
| [xgboost](https://xgboost.readthedocs.io), [lightgbm](https://lightgbm.readthedocs.io), and [catboost](https://catboost.ai) | Gradient boosting. |
| [InterpretML](https://interpret.ml) | Explainable boosting machines: accurate models whose every effect can be shown. |
| [TabPFN](https://github.com/PriorLabs/TabPFN) | A pretrained model that is most accurate on small data. |
| [Optuna](https://optuna.org) | Hyperparameter searches that learn from each try. |
| [statsmodels](https://www.statsmodels.org) and [pyfixest](https://py-econometrics.github.io/pyfixest/) | Regressions with standard errors, p-values, and confidence intervals: least squares, generalized linear, quantile, robust, mixed, and GEE models; models of counts, of two or more classes, and of ordered classes; fixed effects and clustered standard errors; and principal component and factor analysis of the features. Survival analysis: Kaplan-Meier curves and Cox regression of the time until an event. |
| [DoubleML](https://docs.doubleml.org) | Causal effects of a treatment, estimated with double machine learning. |
| [fairlearn](https://fairlearn.org) | Fairness metrics that compare a model across groups. |
| [MAPIE](https://mapie.readthedocs.io) | Conformal prediction: intervals and sets with a known rate of coverage. |
| [shap](https://shap.readthedocs.io) and [eli5](https://eli5.readthedocs.io) | Explaining models with SHAP values (and shap's plots of them) and weights. |
| [matplotlib](https://matplotlib.org), [seaborn](https://seaborn.pydata.org), and [SciencePlots](https://github.com/garrettj403/SciencePlots) | Figures and other visualizations, in the style of a scientific journal (Nature's, by default) and with colors that people with color blindness can tell apart. |
| [great_tables](https://posit-dev.github.io/great-tables/) and [python-docx](https://python-docx.readthedocs.io) | Scorecards as HTML tables and Word documents. |



### Robust and Transparent

A result that depends on one arbitrary choice of preprocessing or model is
fragile. The `experiment` design tries **every combination** of the techniques
you list for each step and reports how each one did, so readers can see that
your conclusions are robust (or not). The critic's `scorecard` puts every
combination side by side, on every metric, in a table (as
csv, Markdown, LaTeX, HTML, Word, or an image).

Name the columns that identify groups as `groups`, and `amos` keeps them out of the model's
features while it uses them to check fairness across groups, to add fixed
effects and cluster standard errors, and to keep each group in one set when
it splits the data.

### Extensible

Every technique is a small Python class that is added to a library as soon as
it is defined, so it can be named in settings right away. To wrap a tool that
`amos` does not include, name its import path:

```python
import amos

transformer = amos.Transformer(
    name = "yeo_johnson", contents = "sklearn.preprocessing.PowerTransformer")
print(transformer.name)
# yeo_johnson
```

You can use any naming convention you like, but, by default, `amos` uses snakecase
of a class name as the default key for accessing it from the stored technique library.

## Getting started

### Requirements

`amos` requires Python 3.11 or later. It runs on Linux, macOS, and Windows. It
is built on [`chrisjen`](https://github.com/WithPrecedent/chrisjen), `numpy`,
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
| `plots` | matplotlib, seaborn, SciencePlots | The artist's figures, and scorecards as images. |
| `polars` | Polars, pyarrow | Reading Polars data frames. |
| `sampling` | imbalanced-learn | Samplers such as `smote`. |
| `statistics` | statsmodels, pyfixest | `ols`, `glm`, `logit`, `fixest`, and the other models with inference (such as `poisson`, `probit`, and `mixedlm`). |
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

Most users describe a study in a settings file. This is a study of the
Wisconsin breast cancer data that uses every stage, as an ini file:

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
model_techniques = sk_logit, random_forest, xgboost

[critic]
techniques = scorecard, confusion, permutation_importance

[artist]
techniques = roc_curve, importance_plot, confusion_heatmap, shap_waterfall

[random_forest_parameters]
n_estimators = 100

[roc_curve_parameters]
title = How well the best model separates the classes

[importance_plot_parameters]
title = The ten features that matter most
limit = 10

[confusion_heatmap_parameters]
title = The test rows by their actual and predicted classes

[shap_waterfall_parameters]
title = How each feature moved the prediction for one tumor
```

The "general" section names the label (the column that models predict) and
the seed. The project section lists the workers. In the "analyst", each
`{step}_techniques` setting lists alternatives, and the `experiment` design
tries every combination: 2 scalers, 2 samplers, and 3 models (scikit-learn's
logistic regression, a random forest, and xgboost's gradient boosting) make
12 combinations. Parameters for any technique go in a `{technique}_parameters`
section. The study uses optional packages (imbalanced-learn for `smote`,
xgboost, shap for `shap_waterfall`, and matplotlib, seaborn, and SciencePlots
for the figures), which `pip install amos[all]` installs.

#### Run it

Pass the settings and the data (a `pandas.DataFrame`, a path to a data file,
or a scikit-learn dataset) to `Project.create`:

```python
import sklearn.datasets

import amos

cancer = sklearn.datasets.load_breast_cancer(as_frame = True)
project = amos.Project.create("cancer.ini", item = cancer, id = "first_run")
result = project.result
print(result.metrics["roc_auc"] > 0.95)
# True
print(sorted(result.figures))
# ['confusion_heatmap', 'importance_plot', 'roc_curve', 'shap_waterfall']
```

The `artist` drew four figures, titled by their `{technique}_parameters`
sections. The ROC curve shows how well the best combination tells the two
kinds of tumors apart on the test rows it never learned from, and the
confusion heatmap counts the test rows that it classified rightly and
wrongly. The importance plot shows how much its score drops when each feature
is shuffled, which works for any model, and the SHAP waterfall shows how each
feature (after scaling) moved its prediction for one tumor:

<p align="center">
<img src="https://raw.githubusercontent.com/WithPrecedent/amos/main/docs/img/roc_curve.png" alt="The ROC curve of the best model, with an area under the curve of 0.98" height="320"/>
<img src="https://raw.githubusercontent.com/WithPrecedent/amos/main/docs/img/confusion_heatmap.png" alt="The test rows by their actual and predicted classes: 138 of the 143 are classified correctly" height="320"/>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/WithPrecedent/amos/main/docs/img/importance_plot.png" alt="The ten features whose shuffling lowers the model's score the most" height="320"/>
<img src="https://raw.githubusercontent.com/WithPrecedent/amos/main/docs/img/shap_waterfall.png" alt="How each feature moved the best model's prediction for one tumor, from the largest change to the smallest" height="320"/>
</p>

The result includes a `Dataset`, the fitted `model`, its `predictions`,
`metrics`, `tables`, `figures`, and the `history` of every technique.

The settings can also say where the data comes from. A loader in the
wrangler (`load_file`, `download`, or `openml`) loads a data file, or
downloads one first, with the project's file manager, and then the project
needs no `item`. See [loading
data](https://WithPrecedent.github.io/amos/advanced/#loading-data).


#### Compare every branch

The critic's `scorecard` compares every branch of the analysis: each
combination of techniques that the experiment tried, ranked by the criterion,
with every standard metric for the task computed from that branch's own
predictions on the test rows. It can be saved as a csv file, a Markdown table,
a Word document, or an image:

```python
scorecard = project.scorecard
print(len(scorecard.table), list(scorecard.table.columns[:5]))
# 12 ['rank', 'split', 'scale', 'sample', 'model']
scorecard.to_csv("scorecard.csv")
scorecard.to_markdown("scorecard.md")
scorecard.to_word("scorecard.docx")
scorecard.to_image("scorecard.png")
```

This is the scorecard of the study, saved as an image. The best branch is
shaded:

<p align="center">
<img src="https://raw.githubusercontent.com/WithPrecedent/amos/main/docs/img/scorecard.png" alt="A table of the twelve branches of the study, ranked by ROC AUC, with the scaler, sampler, and model of each and seven metrics" width="100%"/>
</p>

#### Export the results

`export` saves everything needed to report and reproduce the results in a
folder named for the run.

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
