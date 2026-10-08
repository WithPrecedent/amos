# Technique Catalog

Every technique in `amos`, by the stage (worker) that usually uses it. A
technique is named in settings by the name in the first column, and its class
is in the module of its stage (for example, `random_forest` is
`amos.models.RandomForest`). The [advanced user guide](advanced.md) describes
how techniques work, their parameters, and how to write your own.

Each technique is listed with the tool it wraps. Optional packages are only
imported when a technique that needs them is used (see the extras in the
[overview](index.md)). Most models do both tasks. The `cat_boost` encoder
(CatBoost-style target encoding, from category_encoders) is not the
`catboost` model.

## Loaders (wrangler)

| Name | Tool | Description |
| --- | --- | --- |
| `download` |  | Downloads a file from a URL and loads it. |
| `load_file` |  | Loads a data file in any format that the clerk knows. |
| `openml` | `sklearn.datasets.fetch_openml` | Downloads a dataset from OpenML (by its name or id) and loads it. |

## Cleaners (wrangler)

| Name | Description |
| --- | --- |
| `drop_columns` | Removes columns. |
| `drop_constant` | Removes columns that have only one value (and so tell you nothing). |
| `drop_duplicates` | Removes rows that duplicate an earlier row. |
| `drop_missing` | Removes rows with missing values. |
| `filter_rows` | Keeps the rows that match a query. |
| `keep_columns` | Keeps only some columns. The label and groups are always kept. |
| `rename_columns` | Renames columns, including the label and groups of the dataset. |

## Mungers (wrangler)

| Name | Description |
| --- | --- |
| `auto_categorize` | Makes columns with few unique values categorical. |
| `coalesce` | Takes the first value that is not missing from several columns. |
| `combine_flags` | Combines flags into one: whether any or all are true, or how many are. |
| `convert_types` | Changes the data types of columns. |
| `count_patterns` | Counts the matches of patterns in text. |
| `derive_columns` | Makes columns from expressions of other columns, such as "a / b". |
| `extract_all` | Keeps every match of a pattern in text. |
| `extract_pattern` | Keeps the first match of a pattern in text (or the groups of a match). |
| `flag_patterns` | Makes columns that say whether text matches patterns. |
| `map_patterns` | Turns text into values by the first pattern that it matches. |
| `map_values` | Replaces values with others, such as "N/A" with a missing value. |
| `normalize_text` | Normalizes the spaces, case, accents, and punctuation of text. |
| `parse_booleans` | Turns text such as "yes" and "no" into booleans. |
| `parse_dates` | Turns text into dates. |
| `parse_numbers` | Turns text such as "$1,234.50" into numbers. |
| `replace_text` | Replaces the matches of patterns in text. |
| `split_text` | Splits text into columns where a pattern matches. |
| `strip_text` | Trims spaces from text, and optionally makes it lowercase. |

## Describers (explorer)

| Name | Description |
| --- | --- |
| `correlations` | Correlations between the numeric columns (including the label). |
| `describe` | The `pandas` description of every column, one row per column. |
| `frequencies` | The count and share of each value of the categorical columns. |
| `kaplan_meier` | The share of rows without an event over time (a survival curve). |
| `label_balance` | The count and share of each value of the label. |
| `missing_values` | The count and share of missing values in each column. |
| `summarize` | Summary statistics of the numeric columns, as reported in papers. |

## Splitters (analyst)

| Name | Description |
| --- | --- |
| `group_split` | Keeps all of the rows of each group in the same set. |
| `stratified` | Splits at random, keeping the share of each class in both sets. |
| `time_split` | Uses the latest rows as the test set. |
| `train_test` | Splits the rows at random. |

## Imputers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| `iterative_impute` | `sklearn.impute.IterativeImputer` | Fills missing values by modeling each feature from the others. |
| `knn_impute` | `sklearn.impute.KNNImputer` | Fills missing values with the average of the most similar rows. |
| `mean_impute` | `sklearn.impute.SimpleImputer` | Fills missing values with the mean of the training rows. |
| `median_impute` | `sklearn.impute.SimpleImputer` | Fills missing values with the median of the training rows. |
| `mode_impute` | `sklearn.impute.SimpleImputer` | Fills missing values with the most common value of the training rows. |

## Scalers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| `binarize` | `sklearn.preprocessing.Binarizer` | Makes each feature 1 if it is above a threshold (0 by default). |
| `bins` | `sklearn.preprocessing.KBinsDiscretizer` | Sorts each feature into bins with about the same number of rows. |
| `gauss` | `sklearn.preprocessing.PowerTransformer` | Makes each feature more like a normal distribution (Yeo-Johnson). |
| `max_abs` | `sklearn.preprocessing.MaxAbsScaler` | Divides each feature by its largest absolute value. |
| `min_max` | `sklearn.preprocessing.MinMaxScaler` | Rescales each feature to the range from 0 to 1. |
| `normalize` | `sklearn.preprocessing.Normalizer` | Rescales each row to a length of 1. |
| `quantile` | `sklearn.preprocessing.QuantileTransformer` | Replaces each value with its quantile in the training rows. |
| `robust` | `sklearn.preprocessing.RobustScaler` | Centers on the median and scales by the interquartile range. |
| `standard` | `sklearn.preprocessing.StandardScaler` | Centers each feature on 0 with a standard deviation of 1. |

## Encoders (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| `backward_difference` | `category_encoders.BackwardDifferenceEncoder` | Compares each category to the one before it. |
| `base_n` | `category_encoders.BaseNEncoder` | Writes the number of each category in base N (4 by default). |
| `binary` | `category_encoders.BinaryEncoder` | Writes the number of each category in binary digits. |
| `cat_boost` | `category_encoders.CatBoostEncoder` | Target encoding in the manner of CatBoost, which limits leakage. |
| `count` | `category_encoders.CountEncoder` | Replaces each category with how often it is in the training rows. |
| `date_parts` | `skrub.DatetimeEncoder` | Splits dates and times into parts: year, month, day, and so on. |
| `gap` | `skrub.GapEncoder` | Encodes messy text as a mix of topics of its substrings. |
| `hashing` | `category_encoders.HashingEncoder` | Hashes the categories into a fixed number of columns. |
| `helmert` | `category_encoders.HelmertEncoder` | Compares each category to the mean of the categories before it. |
| `james_stein` | `category_encoders.JamesSteinEncoder` | Target encoding shrunk toward the overall mean (James-Stein). |
| `leave_one_out` | `category_encoders.LeaveOneOutEncoder` | Target encoding that leaves out each row's own label. |
| `m_estimate` | `category_encoders.MEstimateEncoder` | Target encoding shrunk toward the overall mean by m rows. |
| `min_hash` | `skrub.MinHashEncoder` | Encodes messy text by hashing its substrings, which is fast and robust. |
| `one_hot` | `sklearn.preprocessing.OneHotEncoder` | Makes a column of 0s and 1s for each category (dummy variables). |
| `ordinal` | `sklearn.preprocessing.OrdinalEncoder` | Numbers the categories (0, 1, 2, ...) in sorted order. |
| `polynomial_coding` | `category_encoders.PolynomialEncoder` | Contrasts categories as an ordered (polynomial) sequence. |
| `sum_coding` | `category_encoders.SumEncoder` | Compares each category to the mean of all categories (effect coding). |
| `target` | `category_encoders.TargetEncoder` | Replaces each category with the mean label of its training rows. |
| `tfidf` | `skrub.StringEncoder` | Encodes text by the TF-IDF of its substrings, reduced with SVD. |
| `weight_of_evidence` | `category_encoders.WOEEncoder` | Replaces each category with its weight of evidence (binary labels). |

## Mixers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| `interactions` | `sklearn.preprocessing.PolynomialFeatures` | Adds the product of each pair of features. |
| `polynomial` | `sklearn.preprocessing.PolynomialFeatures` | Adds the squares and products of the features (degree 2 by default). |
| `splines` | `sklearn.preprocessing.SplineTransformer` | Replaces each feature with a set of spline curves. |

## Reducers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| `k_best` | `sklearn.feature_selection.SelectKBest` | Keeps the k features (10 by default) most related to the label. |
| `pca_reduce` | `sklearn.decomposition.PCA` | Replaces the features with their principal components. |
| `select_percentile` | `sklearn.feature_selection.SelectPercentile` | Keeps the features (50% by default) most related to the label. |
| `variance_threshold` | `sklearn.feature_selection.VarianceThreshold` | Removes features whose variance is at or below a threshold (0). |

## Samplers (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| `adasyn` | `imblearn.over_sampling.ADASYN` | Adds synthetic rows where the rare class is hardest to learn. |
| `borderline_smote` | `imblearn.over_sampling.BorderlineSMOTE` | Adds synthetic rows near the border between the classes. |
| `near_miss` | `imblearn.under_sampling.NearMiss` | Removes rows of the common classes that are far from the rare rows. |
| `random_over` | `imblearn.over_sampling.RandomOverSampler` | Duplicates random rows of the rare classes. |
| `random_under` | `imblearn.under_sampling.RandomUnderSampler` | Removes random rows of the common classes. |
| `smote` | `imblearn.over_sampling.SMOTE` | Adds synthetic rows between rare rows and their nearest neighbors. |
| `smote_enn` | `imblearn.combine.SMOTEENN` | Applies `smote` and then removes rows that their neighbors misclassify. |
| `smote_tomek` | `imblearn.combine.SMOTETomek` | Applies `smote` and then removes Tomek links. |
| `tomek_links` | `imblearn.under_sampling.TomekLinks` | Removes rows of the common class that are paired with rare rows. |

## Models (analyst)

| Name | Classify | Regress | Description |
| --- | --- | --- | --- |
| `adaboost` | `sklearn.ensemble.AdaBoostClassifier` | `sklearn.ensemble.AdaBoostRegressor` | AdaBoost: a sequence of small trees, each fixing the last one's errors. |
| `baseline` | `sklearn.dummy.DummyClassifier` | `sklearn.dummy.DummyRegressor` | Predicts the most common class (or the mean) for every row. |
| `binomial_bayes_mixedglm` | `amos.models.Statsmodel` |  | A Bayesian logistic regression with a random intercept for each group. |
| `catboost` | `catboost.CatBoostClassifier` | `catboost.CatBoostRegressor` | CatBoost gradient boosting, which uses categorical features directly. |
| `cox` |  | `amos.models.ProportionalHazards` | Cox proportional hazards regression of the time until an event. |
| `decision_tree` | `sklearn.tree.DecisionTreeClassifier` | `sklearn.tree.DecisionTreeRegressor` | A single decision tree. |
| `elastic_net` |  | `sklearn.linear_model.ElasticNet` | Linear regression with both lasso and ridge penalties. |
| `explainable_boosting` | `interpret.glassbox.ExplainableBoostingClassifier` | `interpret.glassbox.ExplainableBoostingRegressor` | An Explainable Boosting Machine from InterpretML. |
| `extra_trees` | `sklearn.ensemble.ExtraTreesClassifier` | `sklearn.ensemble.ExtraTreesRegressor` | An ensemble of extremely randomized trees. |
| `fixest` | `amos.models.FixedEffects` | `amos.models.FixedEffects` | Regression with fixed effects and clustered standard errors (pyfixest). |
| `gee` | `amos.models.Statsmodel` | `amos.models.Statsmodel` | Generalized estimating equations from statsmodels, with inference. |
| `generalized_poisson` |  | `amos.models.Statsmodel` | Generalized Poisson regression of counts, with inference. |
| `glm` | `amos.models.Statsmodel` | `amos.models.Statsmodel` | A generalized linear model from statsmodels, with inference. |
| `gradient_boosting` | `sklearn.ensemble.HistGradientBoostingClassifier` | `sklearn.ensemble.HistGradientBoostingRegressor` | Histogram-based gradient boosting from scikit-learn. |
| `knn` | `sklearn.neighbors.KNeighborsClassifier` | `sklearn.neighbors.KNeighborsRegressor` | Predicts from the k nearest training rows (5 by default). |
| `lasso` |  | `sklearn.linear_model.Lasso` | Linear regression with a lasso (L1) penalty. |
| `lightgbm` | `lightgbm.LGBMClassifier` | `lightgbm.LGBMRegressor` | LightGBM gradient boosting. |
| `linear` |  | `sklearn.linear_model.LinearRegression` | Ordinary least squares regression from scikit-learn. |
| `logit` | `amos.models.Statsmodel` |  | Logistic regression from statsmodels, with inference. |
| `mixedlm` |  | `amos.models.Statsmodel` | A linear mixed model from statsmodels, with inference. |
| `mnlogit` | `amos.models.Statsmodel` |  | Multinomial logistic regression from statsmodels, with inference. |
| `naive_bayes` | `sklearn.naive_bayes.GaussianNB` |  | Gaussian naive Bayes. |
| `negative_binomial` |  | `amos.models.Statsmodel` | Negative binomial regression of counts, with inference. |
| `neural_network` | `sklearn.neural_network.MLPClassifier` | `sklearn.neural_network.MLPRegressor` | A multi-layer perceptron (a simple neural network). |
| `ols` |  | `amos.models.Statsmodel` | Ordinary least squares regression from statsmodels, with inference. |
| `ordinal_regression` | `amos.models.Statsmodel` |  | Ordinal regression (ordered logit or probit) from statsmodels. |
| `poisson` |  | `amos.models.Statsmodel` | Poisson regression of counts (such as the number of arrests). |
| `probit` | `amos.models.Statsmodel` |  | Probit regression of two classes from statsmodels, with inference. |
| `quantile_regression` |  | `amos.models.Statsmodel` | Quantile regression from statsmodels, with inference. |
| `random_forest` | `sklearn.ensemble.RandomForestClassifier` | `sklearn.ensemble.RandomForestRegressor` | A random forest. |
| `ridge` |  | `sklearn.linear_model.Ridge` | Linear regression with a ridge (L2) penalty. |
| `robust_regression` |  | `amos.models.Statsmodel` | Robust linear regression from statsmodels, with inference. |
| `sk_logit` | `sklearn.linear_model.LogisticRegression` |  | Logistic regression from scikit-learn. |
| `svm` | `sklearn.svm.SVC` | `sklearn.svm.SVR` | A support vector machine (with probabilities for classification). |
| `tabpfn` | `tabpfn.TabPFNClassifier` | `tabpfn.TabPFNRegressor` | TabPFN, a pretrained model that is often the most accurate on small data. |
| `wls` |  | `amos.models.Statsmodel` | Weighted least squares regression from statsmodels, with inference. |
| `xgboost` | `xgboost.XGBClassifier` | `xgboost.XGBRegressor` | XGBoost gradient boosting. |
| `zero_inflated_poisson` |  | `amos.models.Statsmodel` | Zero-inflated Poisson regression of counts, with inference. |

## Validators (analyst)

| Name | Tool | Description |
| --- | --- | --- |
| `group_k_fold` | `sklearn.model_selection.GroupKFold` | Folds that keep all of the rows of each group in the same fold. |
| `group_shuffle_split` | `sklearn.model_selection.GroupShuffleSplit` | Repeated random splits of the groups (not the rows) into two sets. |
| `k_fold` | `sklearn.model_selection.KFold` | Divides the rows into folds ("n_splits", 5 by default) at random. |
| `leave_one_group_out` | `sklearn.model_selection.LeaveOneGroupOut` | Scores each group with a copy of the model fitted to the other groups. |
| `leave_one_row_out` | `sklearn.model_selection.LeaveOneOut` | Predicts each row with a copy of the model fitted to every other row. |
| `repeated_k_fold` | `sklearn.model_selection.RepeatedKFold` | `k_fold` repeated with different random folds ("n_repeats" times). |
| `repeated_stratified_k_fold` | `sklearn.model_selection.RepeatedStratifiedKFold` | `stratified_k_fold` repeated with different random folds. |
| `shuffle_split` | `sklearn.model_selection.ShuffleSplit` | Repeated random splits of the rows into two sets (Monte Carlo). |
| `stratified_group_k_fold` | `sklearn.model_selection.StratifiedGroupKFold` | Folds that keep each group together and the classes in proportion. |
| `stratified_k_fold` | `sklearn.model_selection.StratifiedKFold` | Folds that keep the share of each class the same in every fold. |
| `stratified_shuffle_split` | `sklearn.model_selection.StratifiedShuffleSplit` | Repeated random splits into two sets, with the classes in proportion. |
| `time_series_split` | `sklearn.model_selection.TimeSeriesSplit` | Scores later rows with copies of the model fitted to earlier rows. |

## Effects (analyst)

| Name | Description |
| --- | --- |
| `interactive_regression` | The average effect of a treatment that has two values (such as 0 and 1). |
| `partially_linear` | The effect of a treatment that adds to the label in the same way for all. |

## Metrics (critic)

| Name | Tool | Task | Better | Description |
| --- | --- | --- | --- | --- |
| `accuracy` | `sklearn.metrics.accuracy_score` | classify | higher | The share of rows classified correctly. |
| `average_precision` | `sklearn.metrics.average_precision_score` | classify | higher | The area under the precision-recall curve. |
| `balanced_accuracy` | `sklearn.metrics.balanced_accuracy_score` | classify | higher | The average share of each class classified correctly. |
| `brier` | `sklearn.metrics.brier_score_loss` | classify | lower | The mean squared error of the predicted probabilities (lower is better). |
| `cohen_kappa` | `sklearn.metrics.cohen_kappa_score` | classify | higher | Agreement between the predictions and labels beyond chance. |
| `concordance` | `amos.metrics.concordance_index` | regress | higher | How often the model orders pairs of times correctly (Harrell's C). |
| `demographic_parity` | `fairlearn.metrics.demographic_parity_difference` | classify | lower | The largest gap between groups in the share predicted to be positive. |
| `demographic_parity_ratio` | `fairlearn.metrics.demographic_parity_ratio` | classify | higher | The smallest group's share predicted positive over the largest's. |
| `equal_opportunity` | `fairlearn.metrics.equal_opportunity_difference` | classify | lower | The largest gap between groups in the true positive rate. |
| `equal_opportunity_ratio` | `fairlearn.metrics.equal_opportunity_ratio` | classify | higher | The smallest group's true positive rate over the largest's. |
| `equalized_odds` | `fairlearn.metrics.equalized_odds_difference` | classify | lower | The larger gap between groups in true or false positive rates. |
| `equalized_odds_ratio` | `fairlearn.metrics.equalized_odds_ratio` | classify | higher | The smaller ratio between groups of true or false positive rates. |
| `explained_variance` | `sklearn.metrics.explained_variance_score` | regress | higher | The share of the variance of the label that the model explains. |
| `f1` | `sklearn.metrics.f1_score` | classify | higher | The harmonic mean of precision and recall. |
| `log_loss` | `sklearn.metrics.log_loss` | classify | lower | The negative log-likelihood of the true labels (lower is better). |
| `mae` | `sklearn.metrics.mean_absolute_error` | regress | lower | The mean absolute error (lower is better). |
| `mape` | `sklearn.metrics.mean_absolute_percentage_error` | regress | lower | The mean absolute percentage error (lower is better). |
| `matthews` | `sklearn.metrics.matthews_corrcoef` | classify | higher | The Matthews correlation coefficient (phi for two classes). |
| `mse` | `sklearn.metrics.mean_squared_error` | regress | lower | The mean squared error (lower is better). |
| `precision` | `sklearn.metrics.precision_score` | classify | higher | The share of rows predicted to be positive that are positive. |
| `r2` | `sklearn.metrics.r2_score` | regress | higher | The coefficient of determination (R squared). |
| `recall` | `sklearn.metrics.recall_score` | classify | higher | The share of positive rows that are predicted to be positive. |
| `rmse` | `sklearn.metrics.root_mean_squared_error` | regress | lower | The root mean squared error (lower is better). |
| `roc_auc` | `sklearn.metrics.roc_auc_score` | classify | higher | The area under the receiver operating characteristic (ROC) curve. |

## Evaluators (critic)

| Name | Description |
| --- | --- |
| `classification_report` | Precision, recall, f1, and the number of rows of each class. |
| `conformal` | Prediction intervals (or sets) with a known rate of coverage. |
| `confusion` | How many rows of each class were predicted to be each class. |
| `explain_weights` | eli5's explanation of the weights of the model's features. |
| `factor_analysis` | The hidden factors that explain the correlations of the features. |
| `fairness` | How the model does for each group, and the gaps between groups. |
| `feature_importance` | The importance that the model itself gives each feature. |
| `pca` | How much of the variance of the features each principal component has. |
| `permutation_importance` | How much the model's score drops when each feature is shuffled. |
| `scorecard` | The results of every branch of an analysis, ready to publish. |
| `shap_importance` | The mean absolute SHAP value of each feature. |

## Plots (artist)

| Name | Description |
| --- | --- |
| `actual_vs_predicted` | The label against the model's predictions (regression). |
| `box_plots` | Each numeric feature as a box, split by class. |
| `calibration_curve` | The share of rows in each class against its predicted probability. |
| `coefficient_plot` | Coefficients with their confidence intervals (a forest plot). |
| `confusion_heatmap` | The confusion matrix as a heatmap (classification). |
| `correlation_heatmap` | The correlations between the numeric columns as a heatmap. |
| `count_plots` | The count of each value of each categorical feature, split by class. |
| `det_curve` | The detection error tradeoff (DET) curve (classification). |
| `fairness_plot` | The model's fairness metrics for each group, as bars. |
| `histograms` | The distribution of each numeric feature, in a grid. |
| `importance_plot` | The most important features, as horizontal bars. |
| `influence_plot` | The influence of each training row on a statsmodels regression. |
| `label_plot` | The distribution of the label: a bar for each class, or a histogram. |
| `learning_curve` | The model's score as it learns from more of the training rows. |
| `missing_heatmap` | Where values are missing, as a heatmap of the rows and columns. |
| `pair_plot` | Each pair of numeric features against each other, colored by class. |
| `partial_dependence` | The model's average prediction as features change (partial dependence). |
| `precision_recall_curve` | Precision against recall at every threshold (classification). |
| `prediction_intervals` | Conformal prediction intervals of the test rows (or the set sizes). |
| `qq_plot` | The quantiles of the residuals against a normal distribution's. |
| `residual_plot` | The model's errors against its predictions (regression). |
| `roc_curve` | The receiver operating characteristic (ROC) curve (classification). |
| `search_plot` | The cross-validated score of each try in a hyperparameter search. |
| `shap_bar` | The mean absolute SHAP value of the most important features. |
| `shap_beeswarm` | The SHAP value of each row for the most important features. |
| `shap_decision` | How the features move each row's prediction, as lines. |
| `shap_embedding` | The rows placed by their SHAP values, colored by one feature's. |
| `shap_force` | How each feature pushes one row's prediction up or down. |
| `shap_group_difference` | How the SHAP values of the features differ between two groups. |
| `shap_heatmap` | The SHAP values of every explained row, as a heatmap. |
| `shap_partial_dependence` | The model's prediction as one feature changes, with each row's. |
| `shap_scatter` | A feature's SHAP values against its values (a dependence plot). |
| `shap_violin` | The distribution of the SHAP values of the most important features. |
| `shap_waterfall` | How each feature moves one row's prediction (a waterfall plot). |
| `shape_functions` | An explainable boosting model's contribution from each feature. |
| `survival_curves` | The share of rows without an event over time (Kaplan-Meier curves). |
| `tree_plot` | The splits of a decision tree (or of one tree of a forest). |
| `validation_curve` | The model's score as one of its parameters changes. |
