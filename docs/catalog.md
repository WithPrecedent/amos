---
# The tables are wide, so the page leaves out the navigation on its left.
hide:
- navigation
---

# Technique Catalog

Every technique in `amos`, by the stage (worker) that usually uses it. A
technique is named in settings by the name in the first column, and its class
is in the module of its step (for example, `random_forest` is
`amos.models.RandomForest`). Each name links to its class in the [API
reference](reference/amos/index.md). The [advanced user guide](advanced.md)
describes how techniques work, their parameters, and how to write your own.

Each technique is listed with the tool it wraps. Optional packages are only
imported when a technique that needs them is used (see the extras in the
[overview](index.md)). Most models do both tasks. The `cat_boost` encoder
(CatBoost-style target encoding, from category_encoders) is not the
`catboost` model.

[Metrics](#metrics) are listed after the techniques. They are not
techniques, so they are not steps of a workflow: name one as a worker's
`criterion`, or in the `metrics` of a technique that uses them (such as
`scorecard` or a validator).

[Pipelines](#pipelines) are listed last. A pipeline is not a technique
either: it is a set of settings for the analyst, critic, and artist of a
study, which you name with `pipeline` (see [pipeline](idea.md#pipeline)).
Every model has a pipeline of its own for each task that it can do. The
[pipeline chart](pipelines.md) shows the steps of every pipeline side by side.

## Loaders (wrangler)

| Name | Description |
| --- | --- |
| [`download`][amos.loaders.Download] | Downloads a file from a URL, or a dataset from OpenML, and loads it. |
| [`load_chunks`][amos.loaders.LoadChunks] | Loads chosen rows and columns of a huge CSV file, a chunk at a time. |
| [`load_file`][amos.loaders.LoadFile] | Loads a data file in any format that the clerk knows. |
| [`load_files`][amos.loaders.LoadFiles] | Loads a batch of files into one table. |

## Parsers (wrangler)

| Name | Description |
| --- | --- |
| [`parse_all`][amos.parsers.ParseAll] | Finds information in text with a rulebook. |
| [`parse_categories`][amos.parsers.ParseCategories] | Lists the value of every pattern that text matches, in order. |
| [`parse_category`][amos.parsers.ParseCategory] | Gives each text the value of the first pattern that it matches. |
| [`parse_counts`][amos.parsers.ParseCounts] | Counts the matches of a pattern in text. |
| [`parse_fields`][amos.parsers.ParseFields] | Makes a column for each labeled section of text. |
| [`parse_flags`][amos.parsers.ParseFlags] | Flags whether text matches a pattern. |
| [`parse_match`][amos.parsers.ParseMatch] | Keeps the first match of a pattern in text (or the groups of a match). |
| [`parse_matches`][amos.parsers.ParseMatches] | Keeps every match of a pattern in text (or the groups of each match). |
| [`parse_names`][amos.parsers.ParseNames] | Finds the names in text. |
| [`parse_replace`][amos.parsers.ParseReplace] | Replaces the matches of a pattern in text. |
| [`parse_split`][amos.parsers.ParseSplit] | Splits text into columns where a pattern matches. |

## Cleaners (wrangler)

| Name | Description |
| --- | --- |
| [`clean_booleans`][amos.cleaners.CleanBooleans] | Turns text such as "yes" and "no" into booleans. |
| [`clean_dates`][amos.cleaners.CleanDates] | Turns text into dates. |
| [`clean_numbers`][amos.cleaners.CleanNumbers] | Turns text such as "$1,234.50" into numbers. |
| [`drop_columns`][amos.cleaners.DropColumns] | Removes columns. |
| [`drop_constant`][amos.cleaners.DropConstant] | Removes columns that have only one value (and so tell you nothing). |
| [`drop_duplicates`][amos.cleaners.DropDuplicates] | Removes rows that duplicate an earlier row. |
| [`drop_missing`][amos.cleaners.DropMissing] | Removes rows with missing values. |
| [`filter_rows`][amos.cleaners.FilterRows] | Keeps the rows that match a query, or a condition for each column. |
| [`keep_columns`][amos.cleaners.KeepColumns] | Keeps only some columns. The label and groups are always kept. |
| [`rename_columns`][amos.cleaners.RenameColumns] | Renames columns, including the label and groups of the dataset. |

## Mungers (wrangler)

| Name | Description |
| --- | --- |
| [`auto_categorize`][amos.mungers.AutoCategorize] | Makes columns with few unique values categorical. |
| [`coalesce`][amos.mungers.Coalesce] | Takes the first value that is not missing from several columns. |
| [`combine_flags`][amos.mungers.CombineFlags] | Combines flags into one: whether any or all are true, or how many are. |
| [`convert_types`][amos.mungers.ConvertTypes] | Changes the data types of columns. |
| [`derive_columns`][amos.mungers.DeriveColumns] | Makes columns from expressions of other columns, such as "a / b". |
| [`html_to_text`][amos.mungers.HtmlToText] | Converts columns of HTML or XML to plain text. |
| [`map_values`][amos.mungers.MapValues] | Replaces values with others, such as "N/A" with a missing value. |
| [`normalize_text`][amos.mungers.NormalizeText] | Normalizes the spaces, case, accents, and punctuation of text. |
| [`set_kinds`][amos.mungers.SetKinds] | Sets the kind of each column, with the smallest type that holds it. |
| [`strip_text`][amos.mungers.StripText] | Trims spaces from text, and optionally makes it lowercase. |

## Mergers (wrangler)

| Name | Description |
| --- | --- |
| [`merge_keys`][amos.mergers.MergeKeys] | Adds columns from another table's row that has the same keys. |
| [`merge_nearest`][amos.mergers.MergeNearest] | Adds columns from another table's row with the nearest date or number. |
| [`merge_ranges`][amos.mergers.MergeRanges] | Adds columns from another table's row whose range has a date or number. |
| [`merge_summary`][amos.mergers.MergeSummary] | Adds summaries of the rows of another table that have the same keys. |

## Shapers (wrangler)

| Name | Description |
| --- | --- |
| [`group_rows`][amos.shapers.GroupRows] | Makes one row for each group of rows, with summaries of its columns. |
| [`lists_to_rows`][amos.shapers.ListsToRows] | Makes a row for each item of a list, such as each author of a book. |
| [`long_to_wide`][amos.shapers.LongToWide] | Spreads the rows of each group into columns, such as one for each year. |
| [`wide_to_long`][amos.shapers.WideToLong] | Stacks columns into rows, such as a row for each year. |

## Describers (explorer)

| Name | Description |
| --- | --- |
| [`association_rules`][amos.describers.AssociationRules] | Finds rules of which values tend to appear in the same rows. |
| [`correlations`][amos.describers.Correlations] | Tabulates the correlations between the numeric columns and the label. |
| [`describe`][amos.describers.Describe] | Tabulates `pandas`' description of every column, one row per column. |
| [`distributions`][amos.describers.Distributions] | Finds the distribution that fits each column best, and its skew. |
| [`frequencies`][amos.describers.Frequencies] | Counts each value of the categorical columns, with its share. |
| [`is_bernoulli`][amos.describers.IsBernoulli] | Tests whether each column is a Bernoulli distribution (0s and 1s). |
| [`is_beta`][amos.describers.IsBeta] | Tests whether each column fits a beta distribution. |
| [`is_bimodal`][amos.describers.IsBimodal] | Tests whether each column has more than one peak (Hartigan's dip test). |
| [`is_binomial`][amos.describers.IsBinomial] | Tests whether each column fits a binomial distribution. |
| [`is_exponential`][amos.describers.IsExponential] | Tests whether each column fits an exponential distribution. |
| [`is_left_skewed`][amos.describers.IsLeftSkewed] | Tests whether each column is skewed to the left. |
| [`is_log_normal`][amos.describers.IsLogNormal] | Tests whether each column fits a log-normal distribution. |
| [`is_normal`][amos.describers.IsNormal] | Tests whether each column fits a normal distribution. |
| [`is_poisson`][amos.describers.IsPoisson] | Tests whether each column fits a Poisson distribution. |
| [`is_right_skewed`][amos.describers.IsRightSkewed] | Tests whether each column is skewed to the right. |
| [`is_student`][amos.describers.IsStudent] | Tests whether each column fits a Student's t distribution. |
| [`is_uniform`][amos.describers.IsUniform] | Tests whether each column fits a uniform distribution. |
| [`kaplan_meier`][amos.describers.KaplanMeier] | Estimates the share of rows still without an event over time. |
| [`label_balance`][amos.describers.LabelBalance] | Counts each value of the label, with its share. |
| [`missing_values`][amos.describers.MissingValues] | Counts the missing values in each column, with their share. |
| [`summarize`][amos.describers.Summarize] | Tabulates the summary statistics of numeric columns that papers report. |

## Splitters (analyst)

| Name | Description |
| --- | --- |
| [`group_split`][amos.splitters.GroupSplit] | Keeps all of the rows of each group in the same set. |
| [`stratified`][amos.splitters.Stratified] | Splits at random, keeping the share of each class in both sets. |
| [`time_split`][amos.splitters.TimeSplit] | Uses the latest rows as the test set. |
| [`train_test`][amos.splitters.TrainTest] | Splits the rows at random. |

## Imputers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| [`iterative_impute`][amos.imputers.IterativeImpute] | <code>sklearn.<wbr>impute.<wbr>IterativeImputer</code> | Fills missing values by modeling each feature from the others. |
| [`knn_impute`][amos.imputers.KnnImpute] | <code>sklearn.<wbr>impute.<wbr>KNNImputer</code> | Fills missing values with the average of the most similar rows. |
| [`mean_impute`][amos.imputers.MeanImpute] | <code>sklearn.<wbr>impute.<wbr>SimpleImputer</code> | Fills missing values with the mean of the training rows. |
| [`median_impute`][amos.imputers.MedianImpute] | <code>sklearn.<wbr>impute.<wbr>SimpleImputer</code> | Fills missing values with the median of the training rows. |
| [`mode_impute`][amos.imputers.ModeImpute] | <code>sklearn.<wbr>impute.<wbr>SimpleImputer</code> | Fills missing values with the most common value of the training rows. |

## Scalers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| [`binarize`][amos.scalers.Binarize] | <code>sklearn.<wbr>preprocessing.<wbr>Binarizer</code> | Makes each feature 1 if it is above a threshold (0 by default). |
| [`bins`][amos.scalers.Bins] | <code>sklearn.<wbr>preprocessing.<wbr>KBinsDiscretizer</code> | Sorts each feature into bins with about the same number of rows. |
| [`gauss`][amos.scalers.Gauss] | <code>sklearn.<wbr>preprocessing.<wbr>PowerTransformer</code> | Makes each feature more like a normal distribution (Yeo-Johnson). |
| [`max_abs`][amos.scalers.MaxAbs] | <code>sklearn.<wbr>preprocessing.<wbr>MaxAbsScaler</code> | Divides each feature by its largest absolute value. |
| [`min_max`][amos.scalers.MinMax] | <code>sklearn.<wbr>preprocessing.<wbr>MinMaxScaler</code> | Rescales each feature to the range from 0 to 1. |
| [`normalize`][amos.scalers.Normalize] | <code>sklearn.<wbr>preprocessing.<wbr>Normalizer</code> | Rescales each row to a length of 1. |
| [`quantile`][amos.scalers.Quantile] | <code>sklearn.<wbr>preprocessing.<wbr>QuantileTransformer</code> | Replaces each value with its quantile in the training rows. |
| [`robust`][amos.scalers.Robust] | <code>sklearn.<wbr>preprocessing.<wbr>RobustScaler</code> | Centers on the median and scales by the interquartile range. |
| [`standard`][amos.scalers.Standard] | <code>sklearn.<wbr>preprocessing.<wbr>StandardScaler</code> | Centers each feature on 0 with a standard deviation of 1. |

## Encoders (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| [`backward_difference`][amos.encoders.BackwardDifference] | <code>category_encoders.<wbr>BackwardDifferenceEncoder</code> | Compares each category to the one before it. |
| [`base_n`][amos.encoders.BaseN] | <code>category_encoders.<wbr>BaseNEncoder</code> | Writes the number of each category in base N (4 by default). |
| [`binary`][amos.encoders.Binary] | <code>category_encoders.<wbr>BinaryEncoder</code> | Writes the number of each category in binary digits. |
| [`cat_boost`][amos.encoders.CatBoost] | <code>category_encoders.<wbr>CatBoostEncoder</code> | Target encoding in the manner of CatBoost, which limits leakage. |
| [`count`][amos.encoders.Count] | <code>category_encoders.<wbr>CountEncoder</code> | Replaces each category with how often it is in the training rows. |
| [`date_parts`][amos.encoders.DateParts] | <code>skrub.<wbr>DatetimeEncoder</code> | Splits dates and times into parts: year, month, day, and so on. |
| [`gap`][amos.encoders.Gap] | <code>skrub.<wbr>GapEncoder</code> | Encodes messy text as a mix of topics of its substrings. |
| [`hashing`][amos.encoders.Hashing] | <code>category_encoders.<wbr>HashingEncoder</code> | Hashes the categories into a fixed number of columns. |
| [`helmert`][amos.encoders.Helmert] | <code>category_encoders.<wbr>HelmertEncoder</code> | Compares each category to the mean of the categories before it. |
| [`james_stein`][amos.encoders.JamesStein] | <code>category_encoders.<wbr>JamesSteinEncoder</code> | Target encoding shrunk toward the overall mean (James-Stein). |
| [`leave_one_out`][amos.encoders.LeaveOneOut] | <code>category_encoders.<wbr>LeaveOneOutEncoder</code> | Target encoding that leaves out each row's own label. |
| [`m_estimate`][amos.encoders.MEstimate] | <code>category_encoders.<wbr>MEstimateEncoder</code> | Target encoding shrunk toward the overall mean by m rows. |
| [`min_hash`][amos.encoders.MinHash] | <code>skrub.<wbr>MinHashEncoder</code> | Encodes messy text by hashing its substrings, which is fast and robust. |
| [`one_hot`][amos.encoders.OneHot] | <code>sklearn.<wbr>preprocessing.<wbr>OneHotEncoder</code> | Makes a column of 0s and 1s for each category (dummy variables). |
| [`ordinal`][amos.encoders.Ordinal] | <code>sklearn.<wbr>preprocessing.<wbr>OrdinalEncoder</code> | Numbers the categories (0, 1, 2, ...) in sorted order. |
| [`polynomial_coding`][amos.encoders.PolynomialCoding] | <code>category_encoders.<wbr>PolynomialEncoder</code> | Contrasts categories as an ordered (polynomial) sequence. |
| [`sum_coding`][amos.encoders.SumCoding] | <code>category_encoders.<wbr>SumEncoder</code> | Compares each category to the mean of all categories (effect coding). |
| [`target`][amos.encoders.Target] | <code>category_encoders.<wbr>TargetEncoder</code> | Replaces each category with the mean label of its training rows. |
| [`tfidf`][amos.encoders.Tfidf] | <code>skrub.<wbr>StringEncoder</code> | Encodes text by the TF-IDF of its substrings, reduced with SVD. |
| [`weight_of_evidence`][amos.encoders.WeightOfEvidence] | <code>category_encoders.<wbr>WOEEncoder</code> | Replaces each category with its weight of evidence (binary labels). |

## Mixers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| [`interactions`][amos.mixers.Interactions] | <code>sklearn.<wbr>preprocessing.<wbr>PolynomialFeatures</code> | Adds the product of each pair of features, or of the terms of formulas. |
| [`polynomial`][amos.mixers.Polynomial] | <code>sklearn.<wbr>preprocessing.<wbr>PolynomialFeatures</code> | Adds the squares and products of the features (degree 2 by default). |
| [`splines`][amos.mixers.Splines] | <code>sklearn.<wbr>preprocessing.<wbr>SplineTransformer</code> | Replaces each feature with a set of spline curves. |

## Selectors (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| [`exhaustive_selection`][amos.selectors.ExhaustiveSelection] | <code>mlxtend.<wbr>feature_selection.<wbr>ExhaustiveFeatureSelector</code> | Keeps the best of every set of 1 to 3 features (by cross-validation). |
| [`feature_group`][amos.selectors.FeatureGroup] |  | Keeps the features that the settings list, and removes the others. |
| [`generic_univariate`][amos.selectors.GenericUnivariate] | <code>sklearn.<wbr>feature_selection.<wbr>GenericUnivariateSelect</code> | Keeps the features most related to the label, by a mode it is given. |
| [`k_best`][amos.selectors.KBest] | <code>sklearn.<wbr>feature_selection.<wbr>SelectKBest</code> | Keeps the k features (10 by default) most related to the label. |
| [`recursive_elimination`][amos.selectors.RecursiveElimination] | <code>sklearn.<wbr>feature_selection.<wbr>RFE</code> | Removes the least important feature of a model, again and again. |
| [`recursive_elimination_cv`][amos.selectors.RecursiveEliminationCV] | <code>sklearn.<wbr>feature_selection.<wbr>RFECV</code> | Removes features as `recursive_elimination` does, keeping the best. |
| [`select_fdr`][amos.selectors.SelectFdr] | <code>sklearn.<wbr>feature_selection.<wbr>SelectFdr</code> | Keeps the features related to the label, controlling false discovery. |
| [`select_fpr`][amos.selectors.SelectFpr] | <code>sklearn.<wbr>feature_selection.<wbr>SelectFpr</code> | Keeps the features related to the label, controlling false positives. |
| [`select_from_model`][amos.selectors.SelectFromModel] | <code>sklearn.<wbr>feature_selection.<wbr>SelectFromModel</code> | Keeps the features that a model finds most important. |
| [`select_fwe`][amos.selectors.SelectFwe] | <code>sklearn.<wbr>feature_selection.<wbr>SelectFwe</code> | Keeps the features related to the label, controlling family-wise error. |
| [`select_percentile`][amos.selectors.SelectPercentile] | <code>sklearn.<wbr>feature_selection.<wbr>SelectPercentile</code> | Keeps the features (50% by default) most related to the label. |
| [`sequential_selection`][amos.selectors.SequentialSelection] | <code>mlxtend.<wbr>feature_selection.<wbr>SequentialFeatureSelector</code> | Adds (or removes) features one at a time while the score improves. |
| [`sequential_selection_sk`][amos.selectors.SequentialSelectionSK] | <code>sklearn.<wbr>feature_selection.<wbr>SequentialFeatureSelector</code> | Adds (or removes) features one at a time until it keeps enough. |
| [`variance_threshold`][amos.selectors.VarianceThreshold] | <code>sklearn.<wbr>feature_selection.<wbr>VarianceThreshold</code> | Removes features whose variance is at or below a threshold (0). |

## Reducers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| [`pca_reduce`][amos.reducers.PCAReduce] | <code>sklearn.<wbr>decomposition.<wbr>PCA</code> | Replaces the features with their principal components. |

## Samplers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| [`adasyn`][amos.samplers.Adasyn] | <code>imblearn.<wbr>over_sampling.<wbr>ADASYN</code> | Adds synthetic rows where the rare class is hardest to learn. |
| [`borderline_smote`][amos.samplers.BorderlineSmote] | <code>imblearn.<wbr>over_sampling.<wbr>BorderlineSMOTE</code> | Adds synthetic rows near the border between the classes. |
| [`near_miss`][amos.samplers.NearMiss] | <code>imblearn.<wbr>under_sampling.<wbr>NearMiss</code> | Removes rows of the common classes that are far from the rare rows. |
| [`random_over`][amos.samplers.RandomOver] | <code>imblearn.<wbr>over_sampling.<wbr>RandomOverSampler</code> | Duplicates random rows of the rare classes. |
| [`random_under`][amos.samplers.RandomUnder] | <code>imblearn.<wbr>under_sampling.<wbr>RandomUnderSampler</code> | Removes random rows of the common classes. |
| [`smote`][amos.samplers.Smote] | <code>imblearn.<wbr>over_sampling.<wbr>SMOTE</code> | Adds synthetic rows between rare rows and their nearest neighbors. |
| [`smote_enn`][amos.samplers.SmoteEnn] | <code>imblearn.<wbr>combine.<wbr>SMOTEENN</code> | Applies `smote` and then removes rows that their neighbors misclassify. |
| [`smote_tomek`][amos.samplers.SmoteTomek] | <code>imblearn.<wbr>combine.<wbr>SMOTETomek</code> | Applies `smote` and then removes Tomek links. |
| [`tomek_links`][amos.samplers.TomekLinks] | <code>imblearn.<wbr>under_sampling.<wbr>TomekLinks</code> | Removes rows of the common class that are paired with rare rows. |

## Pretests (analyst)

| Name | Description |
| --- | --- |
| [`check_assumptions`][amos.pretests.CheckAssumptions] | Checks every assumption and need of the techniques that has a pretest. |
| [`check_continuity`][amos.pretests.CheckContinuity] | Checks whether rows bunch on one side of a cutoff. |
| [`check_dispersion`][amos.pretests.CheckDispersion] | Checks whether counts vary as a Poisson model assumes. |
| [`check_features`][amos.pretests.CheckFeatures] | Checks whether the features have missing values or are not numbers. |
| [`check_instrument`][amos.pretests.CheckInstrument] | Checks whether an instrument is strong enough. |
| [`check_label`][amos.pretests.CheckLabel] | Checks whether the label is the kind that techniques are for. |
| [`check_linearity`][amos.pretests.CheckLinearity] | Checks whether the label (or its log-odds) is linear in the features. |
| [`check_missingness`][amos.pretests.CheckMissingness] | Checks whether values are missing completely at random. |
| [`check_multicollinearity`][amos.pretests.CheckMulticollinearity] | Checks whether any feature is nearly a linear function of the others. |
| [`check_normality`][amos.pretests.CheckNormality] | Checks whether the errors of a regression and the features are normal. |
| [`check_overlap`][amos.pretests.CheckOverlap] | Checks whether every row could have been treated or not. |
| [`check_parallel_trends`][amos.pretests.CheckParallelTrends] | Checks whether treated units trended differently before treatment. |
| [`check_proportional_odds`][amos.pretests.CheckProportionalOdds] | Checks whether features have the same effect at every cutoff. |
| [`check_sample_size`][amos.pretests.CheckSampleSize] | Checks whether there are enough (or too many) rows and features. |
| [`check_scales`][amos.pretests.CheckScales] | Checks whether the features are on similar scales. |
| [`check_stationarity`][amos.pretests.CheckStationarity] | Checks whether each time series is stationary. |

## Models (analyst)

| Name | Classify | Regress | Description |
| --- | --- | --- | --- |
| [`adaboost`][amos.models.Adaboost] | <code>sklearn.<wbr>ensemble.<wbr>AdaBoostClassifier</code> | <code>sklearn.<wbr>ensemble.<wbr>AdaBoostRegressor</code> | Fits AdaBoost: small trees in sequence, each correcting the last. |
| [`baseline`][amos.models.Baseline] | <code>sklearn.<wbr>dummy.<wbr>DummyClassifier</code> | <code>sklearn.<wbr>dummy.<wbr>DummyRegressor</code> | Predicts the most common class (or the mean) for every row. |
| [`binomial_bayes_mixedglm`][amos.models.BinomialBayesMixedglm] | <code>amos.<wbr>models.<wbr>Statsmodel</code> |  | Fits a Bayesian logistic regression with a random intercept per group. |
| [`catboost`][amos.models.Catboost] | <code>catboost.<wbr>CatBoostClassifier</code> | <code>catboost.<wbr>CatBoostRegressor</code> | Fits CatBoost gradient boosting, which handles categorical features. |
| [`cox`][amos.models.Cox] |  | <code>amos.<wbr>models.<wbr>ProportionalHazards</code> | Fits a Cox proportional hazards regression of the time until an event. |
| [`decision_tree`][amos.models.DecisionTree] | <code>sklearn.<wbr>tree.<wbr>DecisionTreeClassifier</code> | <code>sklearn.<wbr>tree.<wbr>DecisionTreeRegressor</code> | Fits a single decision tree. |
| [`elastic_net`][amos.models.ElasticNet] |  | <code>sklearn.<wbr>linear_model.<wbr>ElasticNet</code> | Fits a linear regression with both lasso and ridge penalties. |
| [`explainable_boosting`][amos.models.ExplainableBoosting] | <code>interpret.<wbr>glassbox.<wbr>ExplainableBoostingClassifier</code> | <code>interpret.<wbr>glassbox.<wbr>ExplainableBoostingRegressor</code> | Fits an Explainable Boosting Machine from InterpretML. |
| [`extra_trees`][amos.models.ExtraTrees] | <code>sklearn.<wbr>ensemble.<wbr>ExtraTreesClassifier</code> | <code>sklearn.<wbr>ensemble.<wbr>ExtraTreesRegressor</code> | Fits an ensemble of extremely randomized trees. |
| [`fixest`][amos.models.Fixest] | <code>amos.<wbr>models.<wbr>FixedEffects</code> | <code>amos.<wbr>models.<wbr>FixedEffects</code> | Fits a regression with fixed effects and clustered standard errors. |
| [`gee`][amos.models.GEE] | <code>amos.<wbr>models.<wbr>Statsmodel</code> | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Fits generalized estimating equations from statsmodels, with inference. |
| [`generalized_poisson`][amos.models.GeneralizedPoisson] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Fits a generalized Poisson regression of counts, with inference. |
| [`glm`][amos.models.GLM] | <code>amos.<wbr>models.<wbr>Statsmodel</code> | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Fits a generalized linear model from statsmodels, with inference. |
| [`gradient_boosting`][amos.models.GradientBoosting] | <code>sklearn.<wbr>ensemble.<wbr>HistGradientBoostingClassifier</code> | <code>sklearn.<wbr>ensemble.<wbr>HistGradientBoostingRegressor</code> | Fits histogram-based gradient boosting from scikit-learn. |
| [`knn`][amos.models.KNN] | <code>sklearn.<wbr>neighbors.<wbr>KNeighborsClassifier</code> | <code>sklearn.<wbr>neighbors.<wbr>KNeighborsRegressor</code> | Predicts from the k nearest training rows (5 by default). |
| [`lasso`][amos.models.Lasso] |  | <code>sklearn.<wbr>linear_model.<wbr>Lasso</code> | Fits a linear regression with a lasso (L1) penalty. |
| [`lightgbm`][amos.models.Lightgbm] | <code>lightgbm.<wbr>LGBMClassifier</code> | <code>lightgbm.<wbr>LGBMRegressor</code> | Fits LightGBM gradient boosting. |
| [`linear`][amos.models.Linear] |  | <code>sklearn.<wbr>linear_model.<wbr>LinearRegression</code> | Fits an ordinary least squares regression from scikit-learn. |
| [`logit`][amos.models.Logit] | <code>sklearn.<wbr>linear_model.<wbr>LogisticRegression</code> |  | Fits a logistic regression from scikit-learn. |
| [`logit_sm`][amos.models.LogitSM] | <code>amos.<wbr>models.<wbr>Statsmodel</code> |  | Fits a logistic regression from statsmodels, with inference. |
| [`mixedlm`][amos.models.Mixedlm] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Fits a linear mixed model from statsmodels, with inference. |
| [`mnlogit`][amos.models.Mnlogit] | <code>amos.<wbr>models.<wbr>Statsmodel</code> |  | Fits a multinomial logistic regression from statsmodels, with inference. |
| [`naive_bayes`][amos.models.NaiveBayes] | <code>sklearn.<wbr>naive_bayes.<wbr>GaussianNB</code> |  | Fits a Gaussian naive Bayes classifier. |
| [`negative_binomial`][amos.models.NegativeBinomial] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Fits a negative binomial regression of counts, with inference. |
| [`neural_network`][amos.models.NeuralNetwork] | <code>sklearn.<wbr>neural_network.<wbr>MLPClassifier</code> | <code>sklearn.<wbr>neural_network.<wbr>MLPRegressor</code> | Fits a multi-layer perceptron (a simple neural network). |
| [`ols`][amos.models.OLS] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Fits ordinary least squares from statsmodels, with inference. |
| [`ordinal_regression`][amos.models.OrdinalRegression] | <code>amos.<wbr>models.<wbr>Statsmodel</code> |  | Fits an ordinal regression (ordered logit or probit) from statsmodels. |
| [`poisson`][amos.models.Poisson] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Fits a Poisson regression of counts (such as the number of visits). |
| [`probit`][amos.models.Probit] | <code>amos.<wbr>models.<wbr>Statsmodel</code> |  | Fits a probit regression of two classes from statsmodels, with inference. |
| [`quantile_regression`][amos.models.QuantileRegression] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Fits a quantile regression from statsmodels, with inference. |
| [`random_forest`][amos.models.RandomForest] | <code>sklearn.<wbr>ensemble.<wbr>RandomForestClassifier</code> | <code>sklearn.<wbr>ensemble.<wbr>RandomForestRegressor</code> | Fits a random forest. |
| [`ridge`][amos.models.Ridge] |  | <code>sklearn.<wbr>linear_model.<wbr>Ridge</code> | Fits a linear regression with a ridge (L2) penalty. |
| [`robust_regression`][amos.models.RobustRegression] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Fits a robust linear regression from statsmodels, with inference. |
| [`svm`][amos.models.SVM] | <code>sklearn.<wbr>svm.<wbr>SVC</code> | <code>sklearn.<wbr>svm.<wbr>SVR</code> | Fits a support vector machine (with probabilities for classification). |
| [`tabpfn`][amos.models.Tabpfn] | <code>tabpfn.<wbr>TabPFNClassifier</code> | <code>tabpfn.<wbr>TabPFNRegressor</code> | Fits TabPFN, a pretrained model that is often best on small data. |
| [`wls`][amos.models.WLS] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Fits weighted least squares from statsmodels, with inference. |
| [`xgboost`][amos.models.Xgboost] | <code>xgboost.<wbr>XGBClassifier</code> | <code>xgboost.<wbr>XGBRegressor</code> | Fits XGBoost gradient boosting. |
| [`zero_inflated_poisson`][amos.models.ZeroInflatedPoisson] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Fits a zero-inflated Poisson regression of counts, with inference. |

## Validators (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| [`group_k_fold`][amos.validators.GroupKFold] | <code>sklearn.<wbr>model_selection.<wbr>GroupKFold</code> | Folds that keep all of the rows of each group in the same fold. |
| [`group_shuffle_split`][amos.validators.GroupShuffleSplit] | <code>sklearn.<wbr>model_selection.<wbr>GroupShuffleSplit</code> | Repeated random splits of the groups (not the rows) into two sets. |
| [`k_fold`][amos.validators.KFold] | <code>sklearn.<wbr>model_selection.<wbr>KFold</code> | Divides the rows into folds ("n_splits", 5 by default) at random. |
| [`leave_one_group_out`][amos.validators.LeaveOneGroupOut] | <code>sklearn.<wbr>model_selection.<wbr>LeaveOneGroupOut</code> | Scores each group with a copy of the model fitted to the other groups. |
| [`leave_one_row_out`][amos.validators.LeaveOneRowOut] | <code>sklearn.<wbr>model_selection.<wbr>LeaveOneOut</code> | Predicts each row with a copy of the model fitted to every other row. |
| [`repeated_k_fold`][amos.validators.RepeatedKFold] | <code>sklearn.<wbr>model_selection.<wbr>RepeatedKFold</code> | `k_fold` repeated with different random folds ("n_repeats" times). |
| [`repeated_stratified_k_fold`][amos.validators.RepeatedStratifiedKFold] | <code>sklearn.<wbr>model_selection.<wbr>RepeatedStratifiedKFold</code> | `stratified_k_fold` repeated with different random folds. |
| [`shuffle_split`][amos.validators.ShuffleSplit] | <code>sklearn.<wbr>model_selection.<wbr>ShuffleSplit</code> | Repeated random splits of the rows into two sets (Monte Carlo). |
| [`stratified_group_k_fold`][amos.validators.StratifiedGroupKFold] | <code>sklearn.<wbr>model_selection.<wbr>StratifiedGroupKFold</code> | Folds that keep each group together and the classes in proportion. |
| [`stratified_k_fold`][amos.validators.StratifiedKFold] | <code>sklearn.<wbr>model_selection.<wbr>StratifiedKFold</code> | Folds that keep the share of each class the same in every fold. |
| [`stratified_shuffle_split`][amos.validators.StratifiedShuffleSplit] | <code>sklearn.<wbr>model_selection.<wbr>StratifiedShuffleSplit</code> | Repeated random splits into two sets, with the classes in proportion. |
| [`time_series_split`][amos.validators.TimeSeriesSplit] | <code>sklearn.<wbr>model_selection.<wbr>TimeSeriesSplit</code> | Scores later rows with copies of the model fitted to earlier rows. |

## Evaluators (critic)

| Name | Description |
| --- | --- |
| [`bias_variance`][amos.evaluators.BiasVariance] | Splits the model's expected test error into bias and variance. |
| [`classification_report`][amos.evaluators.ClassificationReport] | Tabulates the precision, recall, f1, and number of rows of each class. |
| [`conformal`][amos.evaluators.Conformal] | Makes prediction intervals (or sets) with a known rate of coverage. |
| [`confusion`][amos.evaluators.Confusion] | Counts how many rows of each class were predicted to be each class. |
| [`explain_weights`][amos.evaluators.ExplainWeights] | Tabulates eli5's explanation of the weights of the model's features. |
| [`factor_analysis`][amos.evaluators.FactorAnalysis] | Finds the hidden factors that explain the correlations of features. |
| [`fairness`][amos.evaluators.Fairness] | Compares how the model does for each group, with the gaps between them. |
| [`feature_importance`][amos.evaluators.FeatureImportance] | Tabulates the importance that the model itself gives each feature. |
| [`pca`][amos.evaluators.PCA] | Measures how much of the features' variance each principal component has. |
| [`permutation_importance`][amos.evaluators.PermutationImportance] | Measures how much the model's score drops when a feature is shuffled. |
| [`regression_table`][amos.evaluators.RegressionTable] | Puts the estimates of every model side by side, ready to publish. |
| [`scorecard`][amos.evaluators.Scorecard] | Tabulates the results of every branch of an analysis, ready to publish. |
| [`shap_importance`][amos.evaluators.ShapImportance] | Tabulates the mean absolute SHAP value of each feature. |

## Audits (critic)

| Name | Description |
| --- | --- |
| [`audit_assumptions`][amos.audits.AuditAssumptions] | Checks every assumption of the model that has an audit. |
| [`audit_errors`][amos.audits.AuditErrors] | Checks whether the model's errors have constant variance. |
| [`audit_independence`][amos.audits.AuditIndependence] | Checks whether the model's errors are independent. |
| [`audit_proportional_hazards`][amos.audits.AuditProportionalHazards] | Checks whether each feature's effect on a Cox model's hazard is constant. |
| [`audit_residuals`][amos.audits.AuditResiduals] | Checks whether the model's errors are normal. |

## Inferers (critic)

| Name | Tool | Description |
| --- | --- | --- |
| [`difference_in_differences`][amos.inferers.DifferenceInDifferences] | <code>doubleml.<wbr>did.<wbr>DoubleMLDIDMulti</code> | Estimates a treatment's effect over time by difference-in-differences. |
| [`distance_matching`][amos.inferers.DistanceMatching] | <code>dowhy.<wbr>causal_estimators.<wbr>distance_matching_estimator.<wbr>DistanceMatchingEstimator</code> | Estimates an effect by matching treated rows with similar untreated ones. |
| [`doubly_robust`][amos.inferers.DoublyRobust] | <code>dowhy.<wbr>causal_estimators.<wbr>doubly_robust_estimator.<wbr>DoublyRobustEstimator</code> | Estimates an effect with a doubly robust estimator (augmented IPW). |
| [`dr_learner`][amos.inferers.DRLearner] | <code>causalml.<wbr>inference.<wbr>meta.<wbr>BaseDRRegressor</code> | Estimates a treatment's effect with a doubly robust learner. |
| [`glm_adjustment`][amos.inferers.GLMAdjustment] | <code>dowhy.<wbr>causal_estimators.<wbr>generalized_linear_model_estimator.<wbr>GeneralizedLinearModelEstimator</code> | Estimates an effect with a generalized linear model of the label. |
| [`instrumental_variable`][amos.inferers.InstrumentalVariable] | <code>dowhy.<wbr>causal_estimators.<wbr>instrumental_variable_estimator.<wbr>InstrumentalVariableEstimator</code> | Estimates an effect through an instrumental variable. |
| [`interactive_iv`][amos.inferers.InteractiveIV] | <code>doubleml.<wbr>DoubleMLIIVM</code> | Estimates a treatment's effect on compliers, with an instrument (LATE). |
| [`interactive_regression`][amos.inferers.InteractiveRegression] | <code>doubleml.<wbr>DoubleMLIRM</code> | Estimates the average effect of a treatment with two values. |
| [`lpcmci`][amos.inferers.LPCMCI] | <code>tigramite.<wbr>lpcmci.<wbr>LPCMCI</code> | Finds causes in time series, allowing for hidden causes (LPCMCI). |
| [`partially_linear`][amos.inferers.PartiallyLinear] | <code>doubleml.<wbr>DoubleMLPLR</code> | Estimates a treatment effect that is the same for every row. |
| [`partially_linear_iv`][amos.inferers.PartiallyLinearIV] | <code>doubleml.<wbr>DoubleMLPLIV</code> | Estimates a constant treatment effect with an instrument. |
| [`partially_linear_panel`][amos.inferers.PartiallyLinearPanel] | <code>doubleml.<wbr>DoubleMLPLPR</code> | Estimates a constant treatment effect in a panel: units seen over time. |
| [`partially_logistic`][amos.inferers.PartiallyLogistic] | <code>doubleml.<wbr>DoubleMLLPLR</code> | Estimates a treatment's effect on a label with two classes. |
| [`pcmci`][amos.inferers.PCMCI] | <code>tigramite.<wbr>pcmci.<wbr>PCMCI</code> | Finds which variables cause which others at later times (PCMCI). |
| [`pcmci_plus`][amos.inferers.PCMCIPlus] | <code>tigramite.<wbr>pcmci.<wbr>PCMCI</code> | Finds causes at later times and at the same time (PCMCI+). |
| [`potential_outcomes`][amos.inferers.PotentialOutcomes] | <code>doubleml.<wbr>DoubleMLAPOS</code> | Estimates the effect of each level of a treatment, against one of them. |
| [`propensity_matching`][amos.inferers.PropensityMatching] | <code>dowhy.<wbr>causal_estimators.<wbr>propensity_score_matching_estimator.<wbr>PropensityScoreMatchingEstimator</code> | Estimates an effect by matching rows on their propensity scores. |
| [`propensity_stratification`][amos.inferers.PropensityStratification] | <code>dowhy.<wbr>causal_estimators.<wbr>propensity_score_stratification_estimator.<wbr>PropensityScoreStratificationEstimator</code> | Estimates an effect by comparing rows within strata of propensity. |
| [`propensity_weighting`][amos.inferers.PropensityWeighting] | <code>dowhy.<wbr>causal_estimators.<wbr>propensity_score_weighting_estimator.<wbr>PropensityScoreWeightingEstimator</code> | Estimates an effect by inverse propensity weighting. |
| [`quantile_effects`][amos.inferers.QuantileEffects] | <code>doubleml.<wbr>DoubleMLQTE</code> | Estimates a treatment's effects on quantiles of the label. |
| [`regression_adjustment`][amos.inferers.RegressionAdjustment] | <code>dowhy.<wbr>causal_estimators.<wbr>linear_regression_estimator.<wbr>LinearRegressionEstimator</code> | Estimates an effect with a linear regression adjusted for the features. |
| [`regression_discontinuity`][amos.inferers.RegressionDiscontinuity] | <code>doubleml.<wbr>rdd.<wbr>RDFlex</code> | Estimates a treatment's effect at a cutoff (regression discontinuity). |
| [`s_learner`][amos.inferers.SLearner] | <code>causalml.<wbr>inference.<wbr>meta.<wbr>BaseSRegressor</code> | Estimates effects with one model that includes the treatment (S-learner). |
| [`sample_selection`][amos.inferers.SampleSelection] | <code>doubleml.<wbr>DoubleMLSSM</code> | Estimates a treatment's effect when the label is only sometimes seen. |
| [`t_learner`][amos.inferers.TLearner] | <code>causalml.<wbr>inference.<wbr>meta.<wbr>BaseTRegressor</code> | Estimates effects with a separate model for each group (T-learner). |
| [`time_series_effect`][amos.inferers.TimeSeriesEffect] | <code>tigramite.<wbr>models.<wbr>LinearMediation</code> | Estimates the causal effect of one series (the treatment) on the label. |
| [`tmle`][amos.inferers.TMLE] | <code>causalml.<wbr>inference.<wbr>meta.<wbr>TMLELearner</code> | Estimates an average effect by targeted maximum likelihood (TMLE). |
| [`x_learner`][amos.inferers.XLearner] | <code>causalml.<wbr>inference.<wbr>meta.<wbr>BaseXRegressor</code> | Estimates effects with an X-learner, which suits a rare treatment. |

## Plots (artist)

| Name | Description |
| --- | --- |
| [`actual_vs_predicted`][amos.plots.ActualVsPredicted] | Plots the label against the model's predictions (regression). |
| [`association_rules_plot`][amos.plots.AssociationRulesPlot] | Plots each association rule's support and confidence, colored by lift. |
| [`bias_variance_plot`][amos.plots.BiasVariancePlot] | Plots the model's expected loss beside its bias and variance, as bars. |
| [`box_plots`][amos.plots.BoxPlots] | Draws each numeric feature as a box, split by class. |
| [`calibration_curve`][amos.plots.CalibrationCurve] | Plots the share of rows in each class against its predicted probability. |
| [`causal_graph`][amos.plots.CausalGraph] | Draws the causal links that tigramite found, as a graph of the variables. |
| [`classification_report_heatmap`][amos.plots.ClassificationReportHeatmap] | Draws the precision, recall, and f1 of each class as a heatmap. |
| [`coefficient_plot`][amos.plots.CoefficientPlot] | Plots coefficients with their confidence intervals (a forest plot). |
| [`confusion_heatmap`][amos.plots.ConfusionHeatmap] | Draws the confusion matrix as a heatmap (classification). |
| [`correlation_heatmap`][amos.plots.CorrelationHeatmap] | Draws the correlations between the numeric columns as a heatmap. |
| [`count_plots`][amos.plots.CountPlots] | Plots the count of each value of each categorical feature, by class. |
| [`decision_regions`][amos.plots.DecisionRegions] | Shows the class that the model predicts across two features. |
| [`det_curve`][amos.plots.DetCurve] | Plots the detection error tradeoff (DET) curve (classification). |
| [`distribution_fits`][amos.plots.DistributionFits] | Draws each column's histogram with a fitted distribution over it. |
| [`distribution_plots`][amos.plots.DistributionPlots] | Plots the distribution of every column, in a grid. |
| [`estimates_plot`][amos.plots.EstimatesPlot] | Plots every model's estimates with their confidence intervals. |
| [`fairness_plot`][amos.plots.FairnessPlot] | Plots the model's fairness metrics for each group, as bars. |
| [`frequency_plots`][amos.plots.FrequencyPlots] | Plots the share of the rows with each value of each column, in a grid. |
| [`histograms`][amos.plots.Histograms] | Plots the distribution of each numeric feature, in a grid. |
| [`importance_plot`][amos.plots.ImportancePlot] | Plots the most important features, as horizontal bars. |
| [`influence_plot`][amos.plots.InfluencePlot] | Plots the influence of each training row on a statsmodels regression. |
| [`label_plot`][amos.plots.LabelPlot] | Plots the label's distribution: a bar for each class, or a histogram. |
| [`learning_curve`][amos.plots.LearningCurve] | Plots the model's score as it learns from more of the training rows. |
| [`loadings_heatmap`][amos.plots.LoadingsHeatmap] | Shows how strongly each feature loads on each factor, as a heatmap. |
| [`missing_bars`][amos.plots.MissingBars] | Plots the share of each column's values that are missing, as bars. |
| [`missing_heatmap`][amos.plots.MissingHeatmap] | Shows where values are missing, as a heatmap of the rows and columns. |
| [`pair_plot`][amos.plots.PairPlot] | Plots each pair of numeric features against each other, by class. |
| [`partial_dependence`][amos.plots.PartialDependence] | Plots the model's average prediction as features change. |
| [`precision_recall_curve`][amos.plots.PrecisionRecallCurve] | Plots precision against recall at every threshold (classification). |
| [`prediction_intervals`][amos.plots.PredictionIntervals] | Plots conformal prediction intervals of the test rows (or set sizes). |
| [`qq_plot`][amos.plots.QqPlot] | Plots the quantiles of the residuals against a normal distribution's. |
| [`residual_plot`][amos.plots.ResidualPlot] | Plots the model's errors against its predictions (regression). |
| [`roc_curve`][amos.plots.RocCurve] | Plots the receiver operating characteristic (ROC) curve (classification). |
| [`scorecard_plot`][amos.plots.ScorecardPlot] | Plots each branch's score on each metric of a scorecard, as dots. |
| [`scree_plot`][amos.plots.ScreePlot] | Plots the share of the variance that each principal component explains. |
| [`search_plot`][amos.plots.SearchPlot] | Plots the cross-validated score of each try in a hyperparameter search. |
| [`shap_bar`][amos.plots.ShapBar] | Plots the mean absolute SHAP value of the most important features. |
| [`shap_beeswarm`][amos.plots.ShapBeeswarm] | Plots the SHAP value of each row for the most important features. |
| [`shap_decision`][amos.plots.ShapDecision] | Plots how the features move each row's prediction, as lines. |
| [`shap_embedding`][amos.plots.ShapEmbedding] | Plots the rows by their SHAP values, colored by one feature's. |
| [`shap_force`][amos.plots.ShapForce] | Shows how each feature pushes one row's prediction up or down. |
| [`shap_group_difference`][amos.plots.ShapGroupDifference] | Plots how the SHAP values of the features differ between two groups. |
| [`shap_heatmap`][amos.plots.ShapHeatmap] | Draws the SHAP values of every explained row as a heatmap. |
| [`shap_partial_dependence`][amos.plots.ShapPartialDependence] | Plots the model's prediction as one feature changes, with each row's. |
| [`shap_scatter`][amos.plots.ShapScatter] | Plots a feature's SHAP values against its values (a dependence plot). |
| [`shap_violin`][amos.plots.ShapViolin] | Plots the distribution of the SHAP values of the most important features. |
| [`shap_waterfall`][amos.plots.ShapWaterfall] | Shows how each feature moves one row's prediction (a waterfall plot). |
| [`shape_functions`][amos.plots.ShapeFunctions] | Plots an explainable boosting model's contribution from each feature. |
| [`survival_curves`][amos.plots.SurvivalCurves] | Plots the share of rows without an event over time (Kaplan-Meier). |
| [`tree_feature_space`][amos.plots.TreeFeatureSpace] | Shows how a decision tree divides one or two features. |
| [`tree_leaf_distributions`][amos.plots.TreeLeafDistributions] | Plots the label of the training rows in each leaf of a decision tree. |
| [`tree_leaf_purity`][amos.plots.TreeLeafPurity] | Plots the impurity of each leaf of a decision tree. |
| [`tree_leaf_sizes`][amos.plots.TreeLeafSizes] | Plots the number of training rows in each leaf of a decision tree. |
| [`tree_plot`][amos.plots.TreePlot] | Draws the splits of a decision tree (or of one tree of a forest). |
| [`tree_prediction_path`][amos.plots.TreePredictionPath] | Shows how much each feature decides one row's path down a decision tree. |
| [`validation_curve`][amos.plots.ValidationCurve] | Plots the model's score as one of its parameters changes. |
| [`weights_plot`][amos.plots.WeightsPlot] | Plots the weights of the features that eli5 explains, as bars. |

## Choosers (any stage)

| Name | Description |
| --- | --- |
| [`auto`][amos.core.auto.Auto] | Chooses the techniques of the step that it is in. |
| [`auto_encoder`][amos.core.auto.AutoEncoder] | Encodes each categorical feature in the way that suits it. |
| [`auto_evaluator`][amos.core.auto.AutoEvaluator] | Applies the evaluators that suit the task and the model. |
| [`auto_imputer`][amos.core.auto.AutoImputer] | Fills missing values in the ways that suit the features. |
| [`auto_model`][amos.core.auto.AutoModel] | Fits the best of the models whose needs and assumptions the data meets. |
| [`auto_plot`][amos.core.auto.AutoPlot] | Draws the figures that suit the data and the model. |
| [`auto_sampler`][amos.core.auto.AutoSampler] | Chooses a sampler that suits the classes and the features, or none. |

## Metrics

| Name | Tool | Task | Better | Description |
| --- | --- | --- | --- | --- |
| [`accuracy`][amos.core.metrics.Accuracy] | <code>sklearn.<wbr>metrics.<wbr>accuracy_score</code> | classify | higher | The share of rows classified correctly. |
| [`average_precision`][amos.core.metrics.AveragePrecision] | <code>sklearn.<wbr>metrics.<wbr>average_precision_score</code> | classify | higher | The area under the precision-recall curve. |
| [`balanced_accuracy`][amos.core.metrics.BalancedAccuracy] | <code>sklearn.<wbr>metrics.<wbr>balanced_accuracy_score</code> | classify | higher | The average share of each class classified correctly. |
| [`brier`][amos.core.metrics.Brier] | <code>sklearn.<wbr>metrics.<wbr>brier_score_loss</code> | classify | lower | The mean squared error of the predicted probabilities (lower is better). |
| [`cohen_kappa`][amos.core.metrics.CohenKappa] | <code>sklearn.<wbr>metrics.<wbr>cohen_kappa_score</code> | classify | higher | Agreement between the predictions and labels beyond chance. |
| [`concordance`][amos.core.metrics.Concordance] | <code>amos.<wbr>metrics.<wbr>concordance_index</code> | regress | higher | How often the model orders pairs of times correctly (Harrell's C). |
| [`demographic_parity`][amos.core.metrics.DemographicParity] | <code>fairlearn.<wbr>metrics.<wbr>demographic_parity_difference</code> | classify | lower | The largest gap between groups in the share predicted to be positive. |
| [`demographic_parity_ratio`][amos.core.metrics.DemographicParityRatio] | <code>fairlearn.<wbr>metrics.<wbr>demographic_parity_ratio</code> | classify | higher | The smallest group's share predicted positive over the largest's. |
| [`equal_opportunity`][amos.core.metrics.EqualOpportunity] | <code>fairlearn.<wbr>metrics.<wbr>equal_opportunity_difference</code> | classify | lower | The largest gap between groups in the true positive rate. |
| [`equal_opportunity_ratio`][amos.core.metrics.EqualOpportunityRatio] | <code>fairlearn.<wbr>metrics.<wbr>equal_opportunity_ratio</code> | classify | higher | The smallest group's true positive rate over the largest's. |
| [`equalized_odds`][amos.core.metrics.EqualizedOdds] | <code>fairlearn.<wbr>metrics.<wbr>equalized_odds_difference</code> | classify | lower | The larger gap between groups in true or false positive rates. |
| [`equalized_odds_ratio`][amos.core.metrics.EqualizedOddsRatio] | <code>fairlearn.<wbr>metrics.<wbr>equalized_odds_ratio</code> | classify | higher | The smaller ratio between groups of true or false positive rates. |
| [`explained_variance`][amos.core.metrics.ExplainedVariance] | <code>sklearn.<wbr>metrics.<wbr>explained_variance_score</code> | regress | higher | The share of the variance of the label that the model explains. |
| [`f1`][amos.core.metrics.F1] | <code>sklearn.<wbr>metrics.<wbr>f1_score</code> | classify | higher | The harmonic mean of precision and recall. |
| [`log_loss`][amos.core.metrics.LogLoss] | <code>sklearn.<wbr>metrics.<wbr>log_loss</code> | classify | lower | The negative log-likelihood of the true labels (lower is better). |
| [`mae`][amos.core.metrics.MAE] | <code>sklearn.<wbr>metrics.<wbr>mean_absolute_error</code> | regress | lower | The mean absolute error (lower is better). |
| [`mape`][amos.core.metrics.MAPE] | <code>sklearn.<wbr>metrics.<wbr>mean_absolute_percentage_error</code> | regress | lower | The mean absolute percentage error (lower is better). |
| [`matthews`][amos.core.metrics.Matthews] | <code>sklearn.<wbr>metrics.<wbr>matthews_corrcoef</code> | classify | higher | The Matthews correlation coefficient (phi for two classes). |
| [`mse`][amos.core.metrics.MSE] | <code>sklearn.<wbr>metrics.<wbr>mean_squared_error</code> | regress | lower | The mean squared error (lower is better). |
| [`precision`][amos.core.metrics.Precision] | <code>sklearn.<wbr>metrics.<wbr>precision_score</code> | classify | higher | The share of rows predicted to be positive that are positive. |
| [`r2`][amos.core.metrics.R2] | <code>sklearn.<wbr>metrics.<wbr>r2_score</code> | regress | higher | The coefficient of determination (R squared). |
| [`recall`][amos.core.metrics.Recall] | <code>sklearn.<wbr>metrics.<wbr>recall_score</code> | classify | higher | The share of positive rows that are predicted to be positive. |
| [`rmse`][amos.core.metrics.RMSE] | <code>sklearn.<wbr>metrics.<wbr>root_mean_squared_error</code> | regress | lower | The root mean squared error (lower is better). |
| [`roc_auc`][amos.core.metrics.RocAuc] | <code>sklearn.<wbr>metrics.<wbr>roc_auc_score</code> | classify | higher | The area under the receiver operating characteristic (ROC) curve. |

## Pipelines

| Name | Task | Models | Analyst | Description |
| --- | --- | --- | --- | --- |
| `adaboost_classification` | classify | `adaboost` | split: stratified → impute: auto → encode: auto → sample: auto → model: adaboost | Classifies with AdaBoost, resampling rare classes. AdaBoost chases rows that it gets wrong, so noise and outliers sway it: a grid search tunes its learning rate, which tempers that. |
| `adaboost_regression` | regress | `adaboost` | split: train_test → impute: auto → encode: auto → model: adaboost | Regresses with AdaBoost, which chases rows that it gets wrong (so outliers sway it), with a grid search of its learning rate and number of trees. |
| `baseline_classification` | classify | `baseline` | split: stratified → encode: auto → model: baseline | Predicts the most common class for every row: the score that every other model must beat, drawn beside the share of each class. |
| `baseline_regression` | regress | `baseline` | split: train_test → encode: auto → model: baseline | Predicts the mean label for every row: the score that every other model must beat, drawn beside the distribution of the label. |
| `binomial_bayes_mixedglm_classification` | classify | `binomial_bayes_mixedglm` | split: group_split → impute: auto → encode: auto → model: binomial_bayes_mixedglm | Estimates the effects of the features on two classes with a Bayesian logit model with a random intercept for each group, tested on groups that it has not seen. |
| `catboost_classification` | classify | `catboost` | split: stratified → encode: auto → model: catboost | Classifies with CatBoost, which handles missing values itself and whose ordered boosting resists overfitting without a search, with the classes weighted by how rare they are. |
| `catboost_regression` | regress | `catboost` | split: train_test → encode: auto → model: catboost | Regresses with CatBoost, which handles missing values itself and whose ordered boosting resists overfitting without a search. |
| `classification` | classify | `logit`, `random_forest`, `gradient_boosting`, `baseline` | split: stratified → impute: auto → scale: standard → encode: auto → sample: auto → model: logit, random_forest, gradient_boosting, baseline | Predicts classes, comparing a logit model, a random forest, and gradient boosting by balanced accuracy. |
| `classification_inference` | classify | `ordinal_regression`, `logit_sm`, `mnlogit` | split: stratified → impute: auto → encode: auto → model: auto | Estimates the effects of the features on classes with a logit, multinomial logit, or ordinal model. |
| `count_inference` | regress | `poisson`, `negative_binomial`, `zero_inflated_poisson` | split: train_test → impute: auto → encode: auto → model: auto | Estimates the effects of the features on counts with the Poisson, negative binomial, or zero-inflated Poisson model that suits their dispersion. |
| `count_regression` | regress | `poisson`, `negative_binomial`, `random_forest`, `gradient_boosting`, `baseline` | split: train_test → impute: auto → scale: standard → encode: auto → model: poisson, negative_binomial, random_forest, gradient_boosting, baseline | Predicts counts, comparing Poisson and negative binomial regression with a random forest and gradient boosting fitted for counts, by RMSE. |
| `cox_regression` | regress | `cox` | split: train_test → impute: auto → encode: auto → model: cox | Estimates the hazard ratios of the features for the time until an event with a Cox model, with Kaplan-Meier survival curves. Set "event" in the "cox_parameters" section. |
| `decision_tree_classification` | classify | `decision_tree` | split: stratified → impute: auto → encode: auto → model: decision_tree | Classifies with one decision tree, drawn split by split. A single tree overfits, so a grid search limits its depth and the size of its leaves; the classes are weighted by rarity. |
| `decision_tree_regression` | regress | `decision_tree` | split: train_test → impute: auto → encode: auto → model: decision_tree | Regresses with one decision tree, drawn split by split. A single tree overfits, so a grid search limits its depth and the size of its leaves. |
| `elastic_net_regression` | regress | `elastic_net` | split: train_test → impute: auto → scale: standard → encode: auto → model: elastic_net | Regresses with both lasso and ridge penalties on standardized features, which suits correlated features, with a grid search of the strength and mix of the penalties. |
| `explainable_boosting_classification` | classify | `explainable_boosting` | split: stratified → encode: auto → model: explainable_boosting | Classifies with an explainable boosting machine, which handles missing values and needs no scaling, with the shape function of each feature. Classes are not resampled, so the shapes show the real odds. |
| `explainable_boosting_regression` | regress | `explainable_boosting` | split: train_test → encode: auto → model: explainable_boosting | Regresses with an explainable boosting machine, which handles missing values and needs no scaling, with the shape function of each feature. |
| `extra_trees_classification` | classify | `extra_trees` | split: stratified → impute: auto → encode: auto → model: extra_trees | Classifies with 300 extremely randomized trees, which need no scaling, weighting the classes by how rare they are. |
| `extra_trees_regression` | regress | `extra_trees` | split: train_test → impute: auto → encode: auto → model: extra_trees | Regresses with 300 extremely randomized trees, which need no scaling and vary less than a random forest. |
| `gee_classification` | classify | `gee` | split: group_split → impute: auto → encode: auto → model: gee | Estimates the average effects of the features on two classes across groups with generalized estimating equations, which allow for exchangeable correlation within each group. |
| `gee_regression` | regress | `gee` | split: group_split → impute: auto → encode: auto → model: gee | Estimates the average effects of the features on a number across groups with generalized estimating equations, which allow for exchangeable correlation within each group. |
| `generalized_poisson_regression` | regress | `generalized_poisson` | split: train_test → impute: auto → encode: auto → model: generalized_poisson | Estimates the effects of the features on counts that vary more (or less) than a Poisson distribution allows, with generalized Poisson regression. |
| `glm_classification` | classify | `glm` | split: stratified → impute: auto → encode: auto → model: glm | Estimates the effects of the features on two classes with a binomial generalized linear model, checking its calibration and influential rows. |
| `glm_regression` | regress | `glm` | split: train_test → impute: auto → encode: auto → model: glm | Estimates the effects of the features on a number with a gaussian generalized linear model (set "family" for another), checking its errors and influential rows. |
| `gradient_boosting_classification` | classify | `gradient_boosting` | split: stratified → encode: auto → model: gradient_boosting | Classifies with histogram gradient boosting, which handles missing values itself, with a random search of its learning rate, tree size, and penalty, weighting the classes by rarity. |
| `gradient_boosting_regression` | regress | `gradient_boosting` | split: train_test → encode: auto → model: gradient_boosting | Regresses with histogram gradient boosting, which handles missing values itself, with a random search of its learning rate, tree size, and penalty. |
| `knn_classification` | classify | `knn` | split: stratified → impute: auto → scale: standard → encode: auto → select: k_best → sample: auto → model: knn | Classifies by the nearest training rows on standardized features, keeping the 20 most related features (distances mean little in many dimensions) and resampling rare classes, with a grid search of the neighbors and their weights. |
| `knn_regression` | regress | `knn` | split: train_test → impute: auto → scale: standard → encode: auto → select: k_best → model: knn | Regresses by the nearest training rows on standardized features, keeping the 20 most related features (distances mean little in many dimensions), with a grid search of the neighbors and their weights. |
| `lasso_regression` | regress | `lasso` | split: train_test → impute: auto → scale: standard → encode: auto → model: lasso | Regresses with a lasso penalty on standardized features, which drops weak features, with a grid search of the penalty. |
| `lightgbm_classification` | classify | `lightgbm` | split: stratified → encode: auto → model: lightgbm | Classifies with LightGBM, which handles missing values itself, with a random search of its learning rate, leaves, and penalties, weighting the classes by how rare they are. |
| `lightgbm_regression` | regress | `lightgbm` | split: train_test → encode: auto → model: lightgbm | Regresses with LightGBM, which handles missing values itself, with a random search of its learning rate, leaves, and penalties. |
| `linear_regression` | regress | `linear` | split: train_test → impute: auto → encode: auto → model: linear | Regresses by ordinary least squares (scikit-learn), encoding categories against a reference level so the dummy variables are not collinear, and checking its errors with a Q-Q plot. |
| `logit_classification` | classify | `logit` | split: stratified → impute: auto → scale: standard → encode: auto → model: logit | Classifies with a penalized logit model on standardized features, weighting the classes by how rare they are, with a grid search of the penalty. |
| `logit_sm_classification` | classify | `logit_sm` | split: stratified → impute: auto → encode: auto → model: logit_sm | Estimates the effects of the features on two classes with a logit model, checking that its probabilities are calibrated. |
| `mixedlm_regression` | regress | `mixedlm` | split: group_split → impute: auto → encode: auto → model: mixedlm | Estimates the effects of the features on a number with a linear mixed model, with a random intercept for each group, tested on groups that it has not seen. |
| `mnlogit_classification` | classify | `mnlogit` | split: stratified → impute: auto → encode: auto → model: mnlogit | Estimates the effects of the features on three or more classes with a multinomial logit model, each class compared with the reference class. |
| `naive_bayes_classification` | classify | `naive_bayes` | split: stratified → impute: auto → scale: gauss → encode: auto → sample: auto → model: naive_bayes | Classifies with Gaussian naive Bayes, after making the features more normal (as it assumes) and resampling rare classes, checking its probabilities, which are often poorly calibrated. |
| `negative_binomial_regression` | regress | `negative_binomial` | split: train_test → impute: auto → encode: auto → model: negative_binomial | Estimates the effects of the features on counts that vary more than a Poisson distribution allows (overdispersion), with negative binomial regression. |
| `neural_network_classification` | classify | `neural_network` | split: stratified → impute: auto → scale: standard → encode: auto → sample: auto → model: neural_network | Classifies with a neural network on standardized features, stopping early to avoid overfitting and resampling rare classes, with a grid search of its penalty and learning rate and a learning curve. |
| `neural_network_regression` | regress | `neural_network` | split: train_test → impute: auto → scale: standard → encode: auto → model: neural_network | Regresses with a neural network on standardized features, stopping early to avoid overfitting, with a grid search of its penalty and learning rate and a learning curve. |
| `ols_regression` | regress | `ols` | split: train_test → impute: auto → encode: auto → model: ols | Estimates the effects of the features on a number by ordinary least squares, checking its normal errors (Q-Q plot), equal variance (residuals), and influential rows. |
| `ordinal_inference` | classify | `ordinal_regression` | split: stratified → impute: auto → encode: auto → model: ordinal_regression | Estimates the effects of the features on ordered classes with an ordered logit model. |
| `panel_classification` | classify | `fixest` | split: stratified → impute: auto → encode: auto → model: fixest | Estimates the effects of the features on classes in panel data with a fixed effects logit model and standard errors clustered by group. |
| `panel_regression` | regress | `fixest` | split: train_test → impute: auto → encode: auto → model: fixest | Estimates the effects of the features on a number in panel data with a fixed effects model and standard errors clustered by group. |
| `poisson_regression` | regress | `poisson` | split: train_test → impute: auto → encode: auto → model: poisson | Estimates the effects of the features on counts with Poisson regression, which assumes that their variance equals their mean (the residuals show if it does not). |
| `probit_classification` | classify | `probit` | split: stratified → impute: auto → encode: auto → model: probit | Estimates the effects of the features on two classes with a probit model, checking that its probabilities are calibrated. |
| `quantile_inference` | regress | `quantile_regression` | split: train_test → impute: auto → encode: auto → model: quantile_regression | Estimates the effects of the features on the median of the label, which outliers and skewed errors sway less than the mean. |
| `random_forest_classification` | classify | `random_forest` | split: stratified → impute: auto → encode: auto → model: random_forest | Classifies with a random forest of 300 trees, which needs no scaling, weighting the classes in the sample of each tree by how rare they are. |
| `random_forest_regression` | regress | `random_forest` | split: train_test → impute: auto → encode: auto → model: random_forest | Regresses with a random forest of 300 trees, which needs no scaling. |
| `regression` | regress | `ridge`, `random_forest`, `gradient_boosting`, `baseline` | split: train_test → impute: auto → scale: standard → encode: auto → model: ridge, random_forest, gradient_boosting, baseline | Predicts a number, comparing ridge regression, a random forest, and gradient boosting by RMSE. |
| `regression_inference` | regress | `robust_regression`, `ols` | split: train_test → impute: auto → encode: auto → model: auto | Estimates the effects of the features on a number with ordinary least squares, or robust regression for errors with outliers. |
| `ridge_regression` | regress | `ridge` | split: train_test → impute: auto → scale: standard → encode: auto → model: ridge | Regresses with a ridge penalty on standardized features, which steadies the coefficients of correlated features, with a grid search of the penalty. |
| `robust_inference` | regress | `robust_regression` | split: train_test → impute: auto → encode: auto → model: robust_regression | Estimates the effects of the features on a number with robust regression, which gives less weight to rows with outlying errors. |
| `svm_classification` | classify | `svm` | split: stratified → impute: auto → scale: standard → encode: auto → model: svm | Classifies with a support vector machine on standardized features, weighting the classes by how rare they are, with a grid search of its penalty and kernel width. It is slow with more than about 10,000 rows. |
| `svm_regression` | regress | `svm` | split: train_test → impute: auto → scale: standard → encode: auto → model: svm | Regresses with a support vector machine on standardized features, with a grid search of its penalty and kernel width. It is slow with more than about 10,000 rows. |
| `tabpfn_classification` | classify | `tabpfn` | split: stratified → encode: auto → model: tabpfn | Classifies with TabPFN, which fills, scales, and weighs the features itself. It suits up to about 10,000 rows and 500 features. |
| `tabpfn_regression` | regress | `tabpfn` | split: train_test → encode: auto → model: tabpfn | Regresses with TabPFN, which fills, scales, and weighs the features itself. It suits up to about 10,000 rows and 500 features. |
| `wls_regression` | regress | `wls` | split: train_test → impute: auto → encode: auto → model: wls | Estimates the effects of the features on a number by weighted least squares, for errors whose variance differs from row to row. Set "weights" in the "wls_parameters" section. |
| `xgboost_classification` | classify | `xgboost` | split: stratified → encode: auto → sample: auto → model: xgboost | Classifies with XGBoost, which handles missing values itself, resampling rare classes, with a random search of its learning rate, depth, and sampling of rows and features. |
| `xgboost_regression` | regress | `xgboost` | split: train_test → encode: auto → model: xgboost | Regresses with XGBoost, which handles missing values itself, with a random search of its learning rate, depth, and sampling of rows and features. |
| `zero_inflated_poisson_regression` | regress | `zero_inflated_poisson` | split: train_test → impute: auto → encode: auto → model: zero_inflated_poisson | Estimates the effects of the features on counts with more zeros than a Poisson distribution has, with zero-inflated Poisson regression. |
