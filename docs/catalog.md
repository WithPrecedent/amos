---
# The tables are wide, so the page leaves out the navigation on its left.
hide:
- navigation
---

# Technique Catalog

Every technique in `amos`, by the stage (worker) that usually uses it. A
technique is named in settings by the name in the first column, and its class
is in the module of its stage (for example, `random_forest` is
`amos.models.RandomForest`). Each name links to its class in the [API
reference](reference/amos/index.md). The [advanced user guide](advanced.md)
describes how techniques work, their parameters, and how to write your own.

Each technique is listed with the tool it wraps. Optional packages are only
imported when a technique that needs them is used (see the extras in the
[overview](index.md)). Most models do both tasks. The `cat_boost` encoder
(CatBoost-style target encoding, from category_encoders) is not the
`catboost` model.

## Loaders (wrangler)

| Name | Tool | Description |
| --- | --- | --- |
| [`download`][amos.loaders.Download] |  | Downloads a file from a URL and loads it. |
| [`load_file`][amos.loaders.LoadFile] |  | Loads a data file in any format that the clerk knows. |
| [`openml`][amos.loaders.Openml] | <code>sklearn.<wbr>datasets.<wbr>fetch_openml</code> | Downloads a dataset from OpenML (by its name or id) and loads it. |

## Cleaners (wrangler)

| Name | Description |
| --- | --- |
| [`drop_columns`][amos.cleaners.DropColumns] | Removes columns. |
| [`drop_constant`][amos.cleaners.DropConstant] | Removes columns that have only one value (and so tell you nothing). |
| [`drop_duplicates`][amos.cleaners.DropDuplicates] | Removes rows that duplicate an earlier row. |
| [`drop_missing`][amos.cleaners.DropMissing] | Removes rows with missing values. |
| [`filter_rows`][amos.cleaners.FilterRows] | Keeps the rows that match a query. |
| [`keep_columns`][amos.cleaners.KeepColumns] | Keeps only some columns. The label and groups are always kept. |
| [`rename_columns`][amos.cleaners.RenameColumns] | Renames columns, including the label and groups of the dataset. |

## Mungers (wrangler)

| Name | Description |
| --- | --- |
| [`auto_categorize`][amos.mungers.AutoCategorize] | Makes columns with few unique values categorical. |
| [`coalesce`][amos.mungers.Coalesce] | Takes the first value that is not missing from several columns. |
| [`combine_flags`][amos.mungers.CombineFlags] | Combines flags into one: whether any or all are true, or how many are. |
| [`convert_types`][amos.mungers.ConvertTypes] | Changes the data types of columns. |
| [`count_patterns`][amos.mungers.CountPatterns] | Counts the matches of patterns in text. |
| [`derive_columns`][amos.mungers.DeriveColumns] | Makes columns from expressions of other columns, such as "a / b". |
| [`extract_all`][amos.mungers.ExtractAll] | Keeps every match of a pattern in text. |
| [`extract_pattern`][amos.mungers.ExtractPattern] | Keeps the first match of a pattern in text (or the groups of a match). |
| [`flag_patterns`][amos.mungers.FlagPatterns] | Makes columns that say whether text matches patterns. |
| [`map_patterns`][amos.mungers.MapPatterns] | Turns text into values by the first pattern that it matches. |
| [`map_values`][amos.mungers.MapValues] | Replaces values with others, such as "N/A" with a missing value. |
| [`normalize_text`][amos.mungers.NormalizeText] | Normalizes the spaces, case, accents, and punctuation of text. |
| [`parse_booleans`][amos.mungers.ParseBooleans] | Turns text such as "yes" and "no" into booleans. |
| [`parse_dates`][amos.mungers.ParseDates] | Turns text into dates. |
| [`parse_numbers`][amos.mungers.ParseNumbers] | Turns text such as "$1,234.50" into numbers. |
| [`replace_text`][amos.mungers.ReplaceText] | Replaces the matches of patterns in text. |
| [`split_text`][amos.mungers.SplitText] | Splits text into columns where a pattern matches. |
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
| [`lists_to_rows`][amos.shapers.ListsToRows] | Makes a row for each item of a list, such as each judge of a case. |
| [`long_to_wide`][amos.shapers.LongToWide] | Spreads the rows of each group into columns, such as one for each year. |
| [`wide_to_long`][amos.shapers.WideToLong] | Stacks columns into rows, such as a row for each year. |

## Describers (explorer)

| Name | Description |
| --- | --- |
| [`association_rules`][amos.describers.AssociationRules] | Rules of which values tend to appear in the same rows. |
| [`correlations`][amos.describers.Correlations] | Correlations between the numeric columns (including the label). |
| [`describe`][amos.describers.Describe] | The `pandas` description of every column, one row per column. |
| [`frequencies`][amos.describers.Frequencies] | The count and share of each value of the categorical columns. |
| [`kaplan_meier`][amos.describers.KaplanMeier] | The share of rows without an event over time (a survival curve). |
| [`label_balance`][amos.describers.LabelBalance] | The count and share of each value of the label. |
| [`missing_values`][amos.describers.MissingValues] | The count and share of missing values in each column. |
| [`summarize`][amos.describers.Summarize] | Summary statistics of the numeric columns, as reported in papers. |

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
| [`iterative_impute`][amos.transformers.IterativeImpute] | <code>sklearn.<wbr>impute.<wbr>IterativeImputer</code> | Fills missing values by modeling each feature from the others. |
| [`knn_impute`][amos.transformers.KnnImpute] | <code>sklearn.<wbr>impute.<wbr>KNNImputer</code> | Fills missing values with the average of the most similar rows. |
| [`mean_impute`][amos.transformers.MeanImpute] | <code>sklearn.<wbr>impute.<wbr>SimpleImputer</code> | Fills missing values with the mean of the training rows. |
| [`median_impute`][amos.transformers.MedianImpute] | <code>sklearn.<wbr>impute.<wbr>SimpleImputer</code> | Fills missing values with the median of the training rows. |
| [`mode_impute`][amos.transformers.ModeImpute] | <code>sklearn.<wbr>impute.<wbr>SimpleImputer</code> | Fills missing values with the most common value of the training rows. |

## Scalers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| [`binarize`][amos.transformers.Binarize] | <code>sklearn.<wbr>preprocessing.<wbr>Binarizer</code> | Makes each feature 1 if it is above a threshold (0 by default). |
| [`bins`][amos.transformers.Bins] | <code>sklearn.<wbr>preprocessing.<wbr>KBinsDiscretizer</code> | Sorts each feature into bins with about the same number of rows. |
| [`gauss`][amos.transformers.Gauss] | <code>sklearn.<wbr>preprocessing.<wbr>PowerTransformer</code> | Makes each feature more like a normal distribution (Yeo-Johnson). |
| [`max_abs`][amos.transformers.MaxAbs] | <code>sklearn.<wbr>preprocessing.<wbr>MaxAbsScaler</code> | Divides each feature by its largest absolute value. |
| [`min_max`][amos.transformers.MinMax] | <code>sklearn.<wbr>preprocessing.<wbr>MinMaxScaler</code> | Rescales each feature to the range from 0 to 1. |
| [`normalize`][amos.transformers.Normalize] | <code>sklearn.<wbr>preprocessing.<wbr>Normalizer</code> | Rescales each row to a length of 1. |
| [`quantile`][amos.transformers.Quantile] | <code>sklearn.<wbr>preprocessing.<wbr>QuantileTransformer</code> | Replaces each value with its quantile in the training rows. |
| [`robust`][amos.transformers.Robust] | <code>sklearn.<wbr>preprocessing.<wbr>RobustScaler</code> | Centers on the median and scales by the interquartile range. |
| [`standard`][amos.transformers.Standard] | <code>sklearn.<wbr>preprocessing.<wbr>StandardScaler</code> | Centers each feature on 0 with a standard deviation of 1. |

## Encoders (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| [`backward_difference`][amos.transformers.BackwardDifference] | <code>category_encoders.<wbr>BackwardDifferenceEncoder</code> | Compares each category to the one before it. |
| [`base_n`][amos.transformers.BaseN] | <code>category_encoders.<wbr>BaseNEncoder</code> | Writes the number of each category in base N (4 by default). |
| [`binary`][amos.transformers.Binary] | <code>category_encoders.<wbr>BinaryEncoder</code> | Writes the number of each category in binary digits. |
| [`cat_boost`][amos.transformers.CatBoost] | <code>category_encoders.<wbr>CatBoostEncoder</code> | Target encoding in the manner of CatBoost, which limits leakage. |
| [`count`][amos.transformers.Count] | <code>category_encoders.<wbr>CountEncoder</code> | Replaces each category with how often it is in the training rows. |
| [`date_parts`][amos.transformers.DateParts] | <code>skrub.<wbr>DatetimeEncoder</code> | Splits dates and times into parts: year, month, day, and so on. |
| [`gap`][amos.transformers.Gap] | <code>skrub.<wbr>GapEncoder</code> | Encodes messy text as a mix of topics of its substrings. |
| [`hashing`][amos.transformers.Hashing] | <code>category_encoders.<wbr>HashingEncoder</code> | Hashes the categories into a fixed number of columns. |
| [`helmert`][amos.transformers.Helmert] | <code>category_encoders.<wbr>HelmertEncoder</code> | Compares each category to the mean of the categories before it. |
| [`james_stein`][amos.transformers.JamesStein] | <code>category_encoders.<wbr>JamesSteinEncoder</code> | Target encoding shrunk toward the overall mean (James-Stein). |
| [`leave_one_out`][amos.transformers.LeaveOneOut] | <code>category_encoders.<wbr>LeaveOneOutEncoder</code> | Target encoding that leaves out each row's own label. |
| [`m_estimate`][amos.transformers.MEstimate] | <code>category_encoders.<wbr>MEstimateEncoder</code> | Target encoding shrunk toward the overall mean by m rows. |
| [`min_hash`][amos.transformers.MinHash] | <code>skrub.<wbr>MinHashEncoder</code> | Encodes messy text by hashing its substrings, which is fast and robust. |
| [`one_hot`][amos.transformers.OneHot] | <code>sklearn.<wbr>preprocessing.<wbr>OneHotEncoder</code> | Makes a column of 0s and 1s for each category (dummy variables). |
| [`ordinal`][amos.transformers.Ordinal] | <code>sklearn.<wbr>preprocessing.<wbr>OrdinalEncoder</code> | Numbers the categories (0, 1, 2, ...) in sorted order. |
| [`polynomial_coding`][amos.transformers.PolynomialCoding] | <code>category_encoders.<wbr>PolynomialEncoder</code> | Contrasts categories as an ordered (polynomial) sequence. |
| [`sum_coding`][amos.transformers.SumCoding] | <code>category_encoders.<wbr>SumEncoder</code> | Compares each category to the mean of all categories (effect coding). |
| [`target`][amos.transformers.Target] | <code>category_encoders.<wbr>TargetEncoder</code> | Replaces each category with the mean label of its training rows. |
| [`tfidf`][amos.transformers.Tfidf] | <code>skrub.<wbr>StringEncoder</code> | Encodes text by the TF-IDF of its substrings, reduced with SVD. |
| [`weight_of_evidence`][amos.transformers.WeightOfEvidence] | <code>category_encoders.<wbr>WOEEncoder</code> | Replaces each category with its weight of evidence (binary labels). |

## Mixers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| [`interactions`][amos.transformers.Interactions] | <code>sklearn.<wbr>preprocessing.<wbr>PolynomialFeatures</code> | Adds the product of each pair of features. |
| [`polynomial`][amos.transformers.Polynomial] | <code>sklearn.<wbr>preprocessing.<wbr>PolynomialFeatures</code> | Adds the squares and products of the features (degree 2 by default). |
| [`splines`][amos.transformers.Splines] | <code>sklearn.<wbr>preprocessing.<wbr>SplineTransformer</code> | Replaces each feature with a set of spline curves. |

## Reducers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| [`exhaustive_selection`][amos.transformers.ExhaustiveSelection] | <code>mlxtend.<wbr>feature_selection.<wbr>ExhaustiveFeatureSelector</code> | Keeps the best of every set of 1 to 3 features (by cross-validation). |
| [`k_best`][amos.transformers.KBest] | <code>sklearn.<wbr>feature_selection.<wbr>SelectKBest</code> | Keeps the k features (10 by default) most related to the label. |
| [`pca_reduce`][amos.transformers.PCAReduce] | <code>sklearn.<wbr>decomposition.<wbr>PCA</code> | Replaces the features with their principal components. |
| [`select_percentile`][amos.transformers.SelectPercentile] | <code>sklearn.<wbr>feature_selection.<wbr>SelectPercentile</code> | Keeps the features (50% by default) most related to the label. |
| [`sequential_selection`][amos.transformers.SequentialSelection] | <code>mlxtend.<wbr>feature_selection.<wbr>SequentialFeatureSelector</code> | Adds (or removes) features one at a time while the score improves. |
| [`variance_threshold`][amos.transformers.VarianceThreshold] | <code>sklearn.<wbr>feature_selection.<wbr>VarianceThreshold</code> | Removes features whose variance is at or below a threshold (0). |

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

## Models (analyst)

| Name | Classify | Regress | Description |
| --- | --- | --- | --- |
| [`adaboost`][amos.models.Adaboost] | <code>sklearn.<wbr>ensemble.<wbr>AdaBoostClassifier</code> | <code>sklearn.<wbr>ensemble.<wbr>AdaBoostRegressor</code> | AdaBoost: a sequence of small trees, each fixing the last one's errors. |
| [`baseline`][amos.models.Baseline] | <code>sklearn.<wbr>dummy.<wbr>DummyClassifier</code> | <code>sklearn.<wbr>dummy.<wbr>DummyRegressor</code> | Predicts the most common class (or the mean) for every row. |
| [`binomial_bayes_mixedglm`][amos.models.BinomialBayesMixedglm] | <code>amos.<wbr>models.<wbr>Statsmodel</code> |  | A Bayesian logistic regression with a random intercept for each group. |
| [`catboost`][amos.models.Catboost] | <code>catboost.<wbr>CatBoostClassifier</code> | <code>catboost.<wbr>CatBoostRegressor</code> | CatBoost gradient boosting, which uses categorical features directly. |
| [`cox`][amos.models.Cox] |  | <code>amos.<wbr>models.<wbr>ProportionalHazards</code> | Cox proportional hazards regression of the time until an event. |
| [`decision_tree`][amos.models.DecisionTree] | <code>sklearn.<wbr>tree.<wbr>DecisionTreeClassifier</code> | <code>sklearn.<wbr>tree.<wbr>DecisionTreeRegressor</code> | A single decision tree. |
| [`elastic_net`][amos.models.ElasticNet] |  | <code>sklearn.<wbr>linear_model.<wbr>ElasticNet</code> | Linear regression with both lasso and ridge penalties. |
| [`explainable_boosting`][amos.models.ExplainableBoosting] | <code>interpret.<wbr>glassbox.<wbr>ExplainableBoostingClassifier</code> | <code>interpret.<wbr>glassbox.<wbr>ExplainableBoostingRegressor</code> | An Explainable Boosting Machine from InterpretML. |
| [`extra_trees`][amos.models.ExtraTrees] | <code>sklearn.<wbr>ensemble.<wbr>ExtraTreesClassifier</code> | <code>sklearn.<wbr>ensemble.<wbr>ExtraTreesRegressor</code> | An ensemble of extremely randomized trees. |
| [`fixest`][amos.models.Fixest] | <code>amos.<wbr>models.<wbr>FixedEffects</code> | <code>amos.<wbr>models.<wbr>FixedEffects</code> | Regression with fixed effects and clustered standard errors (pyfixest). |
| [`gee`][amos.models.GEE] | <code>amos.<wbr>models.<wbr>Statsmodel</code> | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Generalized estimating equations from statsmodels, with inference. |
| [`generalized_poisson`][amos.models.GeneralizedPoisson] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Generalized Poisson regression of counts, with inference. |
| [`glm`][amos.models.GLM] | <code>amos.<wbr>models.<wbr>Statsmodel</code> | <code>amos.<wbr>models.<wbr>Statsmodel</code> | A generalized linear model from statsmodels, with inference. |
| [`gradient_boosting`][amos.models.GradientBoosting] | <code>sklearn.<wbr>ensemble.<wbr>HistGradientBoostingClassifier</code> | <code>sklearn.<wbr>ensemble.<wbr>HistGradientBoostingRegressor</code> | Histogram-based gradient boosting from scikit-learn. |
| [`knn`][amos.models.KNN] | <code>sklearn.<wbr>neighbors.<wbr>KNeighborsClassifier</code> | <code>sklearn.<wbr>neighbors.<wbr>KNeighborsRegressor</code> | Predicts from the k nearest training rows (5 by default). |
| [`lasso`][amos.models.Lasso] |  | <code>sklearn.<wbr>linear_model.<wbr>Lasso</code> | Linear regression with a lasso (L1) penalty. |
| [`lightgbm`][amos.models.Lightgbm] | <code>lightgbm.<wbr>LGBMClassifier</code> | <code>lightgbm.<wbr>LGBMRegressor</code> | LightGBM gradient boosting. |
| [`linear`][amos.models.Linear] |  | <code>sklearn.<wbr>linear_model.<wbr>LinearRegression</code> | Ordinary least squares regression from scikit-learn. |
| [`logit`][amos.models.Logit] | <code>sklearn.<wbr>linear_model.<wbr>LogisticRegression</code> |  | Logistic regression from scikit-learn. |
| [`logit_sm`][amos.models.LogitSM] | <code>amos.<wbr>models.<wbr>Statsmodel</code> |  | Logistic regression from statsmodels, with inference. |
| [`mixedlm`][amos.models.Mixedlm] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | A linear mixed model from statsmodels, with inference. |
| [`mnlogit`][amos.models.Mnlogit] | <code>amos.<wbr>models.<wbr>Statsmodel</code> |  | Multinomial logistic regression from statsmodels, with inference. |
| [`naive_bayes`][amos.models.NaiveBayes] | <code>sklearn.<wbr>naive_bayes.<wbr>GaussianNB</code> |  | Gaussian naive Bayes. |
| [`negative_binomial`][amos.models.NegativeBinomial] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Negative binomial regression of counts, with inference. |
| [`neural_network`][amos.models.NeuralNetwork] | <code>sklearn.<wbr>neural_network.<wbr>MLPClassifier</code> | <code>sklearn.<wbr>neural_network.<wbr>MLPRegressor</code> | A multi-layer perceptron (a simple neural network). |
| [`ols`][amos.models.OLS] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Ordinary least squares regression from statsmodels, with inference. |
| [`ordinal_regression`][amos.models.OrdinalRegression] | <code>amos.<wbr>models.<wbr>Statsmodel</code> |  | Ordinal regression (ordered logit or probit) from statsmodels. |
| [`poisson`][amos.models.Poisson] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Poisson regression of counts (such as the number of arrests). |
| [`probit`][amos.models.Probit] | <code>amos.<wbr>models.<wbr>Statsmodel</code> |  | Probit regression of two classes from statsmodels, with inference. |
| [`quantile_regression`][amos.models.QuantileRegression] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Quantile regression from statsmodels, with inference. |
| [`random_forest`][amos.models.RandomForest] | <code>sklearn.<wbr>ensemble.<wbr>RandomForestClassifier</code> | <code>sklearn.<wbr>ensemble.<wbr>RandomForestRegressor</code> | A random forest. |
| [`ridge`][amos.models.Ridge] |  | <code>sklearn.<wbr>linear_model.<wbr>Ridge</code> | Linear regression with a ridge (L2) penalty. |
| [`robust_regression`][amos.models.RobustRegression] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Robust linear regression from statsmodels, with inference. |
| [`svm`][amos.models.SVM] | <code>sklearn.<wbr>svm.<wbr>SVC</code> | <code>sklearn.<wbr>svm.<wbr>SVR</code> | A support vector machine (with probabilities for classification). |
| [`tabpfn`][amos.models.Tabpfn] | <code>tabpfn.<wbr>TabPFNClassifier</code> | <code>tabpfn.<wbr>TabPFNRegressor</code> | TabPFN, a pretrained model that is often the most accurate on small data. |
| [`wls`][amos.models.WLS] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Weighted least squares regression from statsmodels, with inference. |
| [`xgboost`][amos.models.Xgboost] | <code>xgboost.<wbr>XGBClassifier</code> | <code>xgboost.<wbr>XGBRegressor</code> | XGBoost gradient boosting. |
| [`zero_inflated_poisson`][amos.models.ZeroInflatedPoisson] |  | <code>amos.<wbr>models.<wbr>Statsmodel</code> | Zero-inflated Poisson regression of counts, with inference. |

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

## Metrics (critic)

| Name | Tool | Task | Better | Description |
| --- | --- | --- | --- | --- |
| [`accuracy`][amos.metrics.Accuracy] | <code>sklearn.<wbr>metrics.<wbr>accuracy_score</code> | classify | higher | The share of rows classified correctly. |
| [`average_precision`][amos.metrics.AveragePrecision] | <code>sklearn.<wbr>metrics.<wbr>average_precision_score</code> | classify | higher | The area under the precision-recall curve. |
| [`balanced_accuracy`][amos.metrics.BalancedAccuracy] | <code>sklearn.<wbr>metrics.<wbr>balanced_accuracy_score</code> | classify | higher | The average share of each class classified correctly. |
| [`brier`][amos.metrics.Brier] | <code>sklearn.<wbr>metrics.<wbr>brier_score_loss</code> | classify | lower | The mean squared error of the predicted probabilities (lower is better). |
| [`cohen_kappa`][amos.metrics.CohenKappa] | <code>sklearn.<wbr>metrics.<wbr>cohen_kappa_score</code> | classify | higher | Agreement between the predictions and labels beyond chance. |
| [`concordance`][amos.metrics.Concordance] | <code>amos.<wbr>metrics.<wbr>concordance_index</code> | regress | higher | How often the model orders pairs of times correctly (Harrell's C). |
| [`demographic_parity`][amos.metrics.DemographicParity] | <code>fairlearn.<wbr>metrics.<wbr>demographic_parity_difference</code> | classify | lower | The largest gap between groups in the share predicted to be positive. |
| [`demographic_parity_ratio`][amos.metrics.DemographicParityRatio] | <code>fairlearn.<wbr>metrics.<wbr>demographic_parity_ratio</code> | classify | higher | The smallest group's share predicted positive over the largest's. |
| [`equal_opportunity`][amos.metrics.EqualOpportunity] | <code>fairlearn.<wbr>metrics.<wbr>equal_opportunity_difference</code> | classify | lower | The largest gap between groups in the true positive rate. |
| [`equal_opportunity_ratio`][amos.metrics.EqualOpportunityRatio] | <code>fairlearn.<wbr>metrics.<wbr>equal_opportunity_ratio</code> | classify | higher | The smallest group's true positive rate over the largest's. |
| [`equalized_odds`][amos.metrics.EqualizedOdds] | <code>fairlearn.<wbr>metrics.<wbr>equalized_odds_difference</code> | classify | lower | The larger gap between groups in true or false positive rates. |
| [`equalized_odds_ratio`][amos.metrics.EqualizedOddsRatio] | <code>fairlearn.<wbr>metrics.<wbr>equalized_odds_ratio</code> | classify | higher | The smaller ratio between groups of true or false positive rates. |
| [`explained_variance`][amos.metrics.ExplainedVariance] | <code>sklearn.<wbr>metrics.<wbr>explained_variance_score</code> | regress | higher | The share of the variance of the label that the model explains. |
| [`f1`][amos.metrics.F1] | <code>sklearn.<wbr>metrics.<wbr>f1_score</code> | classify | higher | The harmonic mean of precision and recall. |
| [`log_loss`][amos.metrics.LogLoss] | <code>sklearn.<wbr>metrics.<wbr>log_loss</code> | classify | lower | The negative log-likelihood of the true labels (lower is better). |
| [`mae`][amos.metrics.MAE] | <code>sklearn.<wbr>metrics.<wbr>mean_absolute_error</code> | regress | lower | The mean absolute error (lower is better). |
| [`mape`][amos.metrics.MAPE] | <code>sklearn.<wbr>metrics.<wbr>mean_absolute_percentage_error</code> | regress | lower | The mean absolute percentage error (lower is better). |
| [`matthews`][amos.metrics.Matthews] | <code>sklearn.<wbr>metrics.<wbr>matthews_corrcoef</code> | classify | higher | The Matthews correlation coefficient (phi for two classes). |
| [`mse`][amos.metrics.MSE] | <code>sklearn.<wbr>metrics.<wbr>mean_squared_error</code> | regress | lower | The mean squared error (lower is better). |
| [`precision`][amos.metrics.Precision] | <code>sklearn.<wbr>metrics.<wbr>precision_score</code> | classify | higher | The share of rows predicted to be positive that are positive. |
| [`r2`][amos.metrics.R2] | <code>sklearn.<wbr>metrics.<wbr>r2_score</code> | regress | higher | The coefficient of determination (R squared). |
| [`recall`][amos.metrics.Recall] | <code>sklearn.<wbr>metrics.<wbr>recall_score</code> | classify | higher | The share of positive rows that are predicted to be positive. |
| [`rmse`][amos.metrics.RMSE] | <code>sklearn.<wbr>metrics.<wbr>root_mean_squared_error</code> | regress | lower | The root mean squared error (lower is better). |
| [`roc_auc`][amos.metrics.RocAuc] | <code>sklearn.<wbr>metrics.<wbr>roc_auc_score</code> | classify | higher | The area under the receiver operating characteristic (ROC) curve. |

## Evaluators (critic)

| Name | Description |
| --- | --- |
| [`bias_variance`][amos.evaluators.BiasVariance] | The model's expected error on the test rows, as its bias and variance. |
| [`classification_report`][amos.evaluators.ClassificationReport] | Precision, recall, f1, and the number of rows of each class. |
| [`conformal`][amos.evaluators.Conformal] | Prediction intervals (or sets) with a known rate of coverage. |
| [`confusion`][amos.evaluators.Confusion] | How many rows of each class were predicted to be each class. |
| [`explain_weights`][amos.evaluators.ExplainWeights] | eli5's explanation of the weights of the model's features. |
| [`factor_analysis`][amos.evaluators.FactorAnalysis] | The hidden factors that explain the correlations of the features. |
| [`fairness`][amos.evaluators.Fairness] | How the model does for each group, and the gaps between groups. |
| [`feature_importance`][amos.evaluators.FeatureImportance] | The importance that the model itself gives each feature. |
| [`pca`][amos.evaluators.PCA] | How much of the variance of the features each principal component has. |
| [`permutation_importance`][amos.evaluators.PermutationImportance] | How much the model's score drops when each feature is shuffled. |
| [`regression_table`][amos.evaluators.RegressionTable] | The estimates of every model side by side, ready to publish. |
| [`scorecard`][amos.evaluators.Scorecard] | The results of every branch of an analysis, ready to publish. |
| [`shap_importance`][amos.evaluators.ShapImportance] | The mean absolute SHAP value of each feature. |

## Inferers (critic)

| Name | Tool | Description |
| --- | --- | --- |
| [`difference_in_differences`][amos.inferers.DifferenceInDifferences] | <code>doubleml.<wbr>did.<wbr>DoubleMLDIDMulti</code> | DoubleML's difference-in-differences: a treatment's effect over time. |
| [`distance_matching`][amos.inferers.DistanceMatching] | <code>dowhy.<wbr>causal_estimators.<wbr>distance_matching_estimator.<wbr>DistanceMatchingEstimator</code> | DoWhy's matching of treated rows with the most similar untreated rows. |
| [`doubly_robust`][amos.inferers.DoublyRobust] | <code>dowhy.<wbr>causal_estimators.<wbr>doubly_robust_estimator.<wbr>DoublyRobustEstimator</code> | DoWhy's doubly robust estimator (augmented inverse propensity weighting). |
| [`dr_learner`][amos.inferers.DRLearner] | <code>causalml.<wbr>inference.<wbr>meta.<wbr>BaseDRRegressor</code> | causalml's doubly robust learner of the effect of a treatment. |
| [`glm_adjustment`][amos.inferers.GLMAdjustment] | <code>dowhy.<wbr>causal_estimators.<wbr>generalized_linear_model_estimator.<wbr>GeneralizedLinearModelEstimator</code> | DoWhy's generalized linear model of the label, adjusted for the features. |
| [`instrumental_variable`][amos.inferers.InstrumentalVariable] | <code>dowhy.<wbr>causal_estimators.<wbr>instrumental_variable_estimator.<wbr>InstrumentalVariableEstimator</code> | DoWhy's estimate of an effect through an instrument. |
| [`interactive_iv`][amos.inferers.InteractiveIV] | <code>doubleml.<wbr>DoubleMLIIVM</code> | DoubleML's effect for those whom an instrument moves to take a treatment. |
| [`interactive_regression`][amos.inferers.InteractiveRegression] | <code>doubleml.<wbr>DoubleMLIRM</code> | DoubleML's average effect of a treatment with two values. |
| [`lpcmci`][amos.inferers.LPCMCI] | <code>tigramite.<wbr>lpcmci.<wbr>LPCMCI</code> | tigramite's LPCMCI: causal discovery in time series with hidden causes. |
| [`partially_linear`][amos.inferers.PartiallyLinear] | <code>doubleml.<wbr>DoubleMLPLR</code> | DoubleML's effect of a treatment that adds the same amount for every row. |
| [`partially_linear_iv`][amos.inferers.PartiallyLinearIV] | <code>doubleml.<wbr>DoubleMLPLIV</code> | DoubleML's partially linear model of a treatment with an instrument. |
| [`partially_linear_panel`][amos.inferers.PartiallyLinearPanel] | <code>doubleml.<wbr>DoubleMLPLPR</code> | DoubleML's partially linear model of a panel: units seen over time. |
| [`partially_logistic`][amos.inferers.PartiallyLogistic] | <code>doubleml.<wbr>DoubleMLLPLR</code> | DoubleML's partially linear model of a label with two classes. |
| [`pcmci`][amos.inferers.PCMCI] | <code>tigramite.<wbr>pcmci.<wbr>PCMCI</code> | tigramite's PCMCI: which variables cause which others at later times. |
| [`pcmci_plus`][amos.inferers.PCMCIPlus] | <code>tigramite.<wbr>pcmci.<wbr>PCMCI</code> | tigramite's PCMCI+: causes at later times and at the same time. |
| [`potential_outcomes`][amos.inferers.PotentialOutcomes] | <code>doubleml.<wbr>DoubleMLAPOS</code> | DoubleML's effects of each level of a treatment, compared to one of them. |
| [`propensity_matching`][amos.inferers.PropensityMatching] | <code>dowhy.<wbr>causal_estimators.<wbr>propensity_score_matching_estimator.<wbr>PropensityScoreMatchingEstimator</code> | DoWhy's matching of treated and untreated rows by propensity score. |
| [`propensity_stratification`][amos.inferers.PropensityStratification] | <code>dowhy.<wbr>causal_estimators.<wbr>propensity_score_stratification_estimator.<wbr>PropensityScoreStratificationEstimator</code> | DoWhy's comparison of the treated and untreated in strata of propensity. |
| [`propensity_weighting`][amos.inferers.PropensityWeighting] | <code>dowhy.<wbr>causal_estimators.<wbr>propensity_score_weighting_estimator.<wbr>PropensityScoreWeightingEstimator</code> | DoWhy's inverse propensity weighting. |
| [`quantile_effects`][amos.inferers.QuantileEffects] | <code>doubleml.<wbr>DoubleMLQTE</code> | DoubleML's effects of a treatment on quantiles of the label. |
| [`regression_adjustment`][amos.inferers.RegressionAdjustment] | <code>dowhy.<wbr>causal_estimators.<wbr>linear_regression_estimator.<wbr>LinearRegressionEstimator</code> | DoWhy's linear regression of the label, which adjusts for the features. |
| [`regression_discontinuity`][amos.inferers.RegressionDiscontinuity] | <code>doubleml.<wbr>rdd.<wbr>RDFlex</code> | DoubleML's regression discontinuity: the effect at a cutoff. |
| [`s_learner`][amos.inferers.SLearner] | <code>causalml.<wbr>inference.<wbr>meta.<wbr>BaseSRegressor</code> | causalml's S-learner: one model of the label, with the treatment in it. |
| [`sample_selection`][amos.inferers.SampleSelection] | <code>doubleml.<wbr>DoubleMLSSM</code> | DoubleML's effect of a treatment when the label is only sometimes seen. |
| [`t_learner`][amos.inferers.TLearner] | <code>causalml.<wbr>inference.<wbr>meta.<wbr>BaseTRegressor</code> | causalml's T-learner: one model of the label for each group. |
| [`time_series_effect`][amos.inferers.TimeSeriesEffect] | <code>tigramite.<wbr>models.<wbr>LinearMediation</code> | tigramite's causal effect of one series (the treatment) on the label. |
| [`tmle`][amos.inferers.TMLE] | <code>causalml.<wbr>inference.<wbr>meta.<wbr>TMLELearner</code> | causalml's targeted maximum likelihood estimation of an average effect. |
| [`x_learner`][amos.inferers.XLearner] | <code>causalml.<wbr>inference.<wbr>meta.<wbr>BaseXRegressor</code> | causalml's X-learner, which suits a treatment that few rows have. |

## Plots (artist)

| Name | Description |
| --- | --- |
| [`actual_vs_predicted`][amos.plots.ActualVsPredicted] | The label against the model's predictions (regression). |
| [`box_plots`][amos.plots.BoxPlots] | Each numeric feature as a box, split by class. |
| [`calibration_curve`][amos.plots.CalibrationCurve] | The share of rows in each class against its predicted probability. |
| [`coefficient_plot`][amos.plots.CoefficientPlot] | Coefficients with their confidence intervals (a forest plot). |
| [`confusion_heatmap`][amos.plots.ConfusionHeatmap] | The confusion matrix as a heatmap (classification). |
| [`correlation_heatmap`][amos.plots.CorrelationHeatmap] | The correlations between the numeric columns as a heatmap. |
| [`count_plots`][amos.plots.CountPlots] | The count of each value of each categorical feature, split by class. |
| [`decision_regions`][amos.plots.DecisionRegions] | The class that the model predicts across two features (classification). |
| [`det_curve`][amos.plots.DetCurve] | The detection error tradeoff (DET) curve (classification). |
| [`fairness_plot`][amos.plots.FairnessPlot] | The model's fairness metrics for each group, as bars. |
| [`histograms`][amos.plots.Histograms] | The distribution of each numeric feature, in a grid. |
| [`importance_plot`][amos.plots.ImportancePlot] | The most important features, as horizontal bars. |
| [`influence_plot`][amos.plots.InfluencePlot] | The influence of each training row on a statsmodels regression. |
| [`label_plot`][amos.plots.LabelPlot] | The distribution of the label: a bar for each class, or a histogram. |
| [`learning_curve`][amos.plots.LearningCurve] | The model's score as it learns from more of the training rows. |
| [`missing_heatmap`][amos.plots.MissingHeatmap] | Where values are missing, as a heatmap of the rows and columns. |
| [`pair_plot`][amos.plots.PairPlot] | Each pair of numeric features against each other, colored by class. |
| [`partial_dependence`][amos.plots.PartialDependence] | The model's average prediction as features change (partial dependence). |
| [`precision_recall_curve`][amos.plots.PrecisionRecallCurve] | Precision against recall at every threshold (classification). |
| [`prediction_intervals`][amos.plots.PredictionIntervals] | Conformal prediction intervals of the test rows (or the set sizes). |
| [`qq_plot`][amos.plots.QqPlot] | The quantiles of the residuals against a normal distribution's. |
| [`residual_plot`][amos.plots.ResidualPlot] | The model's errors against its predictions (regression). |
| [`roc_curve`][amos.plots.RocCurve] | The receiver operating characteristic (ROC) curve (classification). |
| [`search_plot`][amos.plots.SearchPlot] | The cross-validated score of each try in a hyperparameter search. |
| [`shap_bar`][amos.plots.ShapBar] | The mean absolute SHAP value of the most important features. |
| [`shap_beeswarm`][amos.plots.ShapBeeswarm] | The SHAP value of each row for the most important features. |
| [`shap_decision`][amos.plots.ShapDecision] | How the features move each row's prediction, as lines. |
| [`shap_embedding`][amos.plots.ShapEmbedding] | The rows placed by their SHAP values, colored by one feature's. |
| [`shap_force`][amos.plots.ShapForce] | How each feature pushes one row's prediction up or down. |
| [`shap_group_difference`][amos.plots.ShapGroupDifference] | How the SHAP values of the features differ between two groups. |
| [`shap_heatmap`][amos.plots.ShapHeatmap] | The SHAP values of every explained row, as a heatmap. |
| [`shap_partial_dependence`][amos.plots.ShapPartialDependence] | The model's prediction as one feature changes, with each row's. |
| [`shap_scatter`][amos.plots.ShapScatter] | A feature's SHAP values against its values (a dependence plot). |
| [`shap_violin`][amos.plots.ShapViolin] | The distribution of the SHAP values of the most important features. |
| [`shap_waterfall`][amos.plots.ShapWaterfall] | How each feature moves one row's prediction (a waterfall plot). |
| [`shape_functions`][amos.plots.ShapeFunctions] | An explainable boosting model's contribution from each feature. |
| [`survival_curves`][amos.plots.SurvivalCurves] | The share of rows without an event over time (Kaplan-Meier curves). |
| [`tree_feature_space`][amos.plots.TreeFeatureSpace] | How a decision tree divides one or two features. |
| [`tree_leaf_distributions`][amos.plots.TreeLeafDistributions] | The label of the training rows in each leaf of a decision tree. |
| [`tree_leaf_purity`][amos.plots.TreeLeafPurity] | The impurity of each leaf of a decision tree. |
| [`tree_leaf_sizes`][amos.plots.TreeLeafSizes] | The number of training rows in each leaf of a decision tree. |
| [`tree_plot`][amos.plots.TreePlot] | The splits of a decision tree (or of one tree of a forest). |
| [`tree_prediction_path`][amos.plots.TreePredictionPath] | How much each feature decides one row's path down a decision tree. |
| [`validation_curve`][amos.plots.ValidationCurve] | The model's score as one of its parameters changes. |

## Choosers (any stage)

| Name | Description |
| --- | --- |
| [`auto`][amos.auto.Auto] | Chooses the techniques of the step that it is in. |
| [`auto_model`][amos.auto.AutoModel] | Fits the best of the models whose needs and assumptions the data meets. |
| [`auto_plot`][amos.auto.AutoPlot] | Draws the figures that suit the data and the model. |
| [`auto_sampler`][amos.auto.AutoSampler] | Chooses a sampler that suits the classes and the features, or none. |
