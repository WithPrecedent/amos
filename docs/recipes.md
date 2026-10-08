# Recipes

Short, complete studies for common jobs. They are run by the `amos` unit
tests. See the [tutorial](tutorial.md) for the basics, the [advanced user
guide](advanced.md) for every setting, and the [technique
catalog](catalog.md) for every technique.

## Classify more than two classes

The wine data has three classes. Metrics such as `f1` are averaged over the
classes, and the confusion matrix shows which classes are confused:

```python
import numpy as np
import pandas as pd
import sklearn.datasets

import amos

wine = sklearn.datasets.load_wine(as_frame = True)
settings = {
    "general": {"seed": 43},
    "wine_project": {"wine_workers": "analyst, critic"},
    "analyst": {
        "design": "experiment",
        "criterion": "f1",
        "steps": "split, scale, model",
        "split_techniques": "stratified",
        "scale_techniques": "standard",
        "model_techniques": "sk_logit, knn, naive_bayes",
    },
    "critic": {"techniques": "classification_report, confusion"},
}
project = amos.Project.create(settings, item = wine)
print(project.result.classes)
# [0, 1, 2]
print(project.result.tables["confusion"].shape)
# (3, 3)
print(project.result.probabilities.shape)
# (45, 3)
```

## Regression with statistical inference

A paper usually reports the coefficients of a regression, with their standard
errors and p-values. The `ols` model wraps statsmodels and stores them in a
table. Here, an experiment also checks whether a random forest predicts the
held-out rows better than the regression:

```python
diabetes = sklearn.datasets.load_diabetes(as_frame = True)
settings = {
    "general": {"seed": 43},
    "diabetes_project": {"diabetes_workers": "analyst, critic"},
    "analyst": {
        "design": "experiment",
        "criterion": "rmse",
        "steps": "split, model",
        "split_techniques": "train_test",
        "model_techniques": "ols, random_forest, baseline",
    },
    "critic": {"techniques": "scorecard"},
}
project = amos.Project.create(settings, item = diabetes)
comparison = project.result.tables["analyst_comparison"]
print(comparison.index[-1])
# train_test_split > baseline_model
print((comparison["rmse"] > 0).all())
# True
```

`rmse` is better when it is lower, so it is negated to rank the combinations,
but the table shows its real value. To report the regression itself, fit it to
every row (without a split):

```python
dataset = amos.Dataset.create(diabetes)
amos.models.OLS().apply(dataset)
coefficients = dataset.tables["ols_coefficients"]
significant = coefficients[coefficients["p_value"] < 0.05].index
print(list(significant))
# ['const', 'sex', 'bmi', 'bp', 's5']
```

## Imbalanced classes

When one class is rare, accuracy is misleading (always predicting the common
class is accurate). Compare ways of balancing the training rows with a metric
that weighs both classes, and let the critic's `scorecard` compare them on
every metric:

```python
features, label = sklearn.datasets.make_classification(
    n_samples = 1000, n_features = 8, weights = [0.95], random_state = 0)
data = pd.DataFrame(features, columns = [f"x{i}" for i in range(8)])
data["rare"] = label
settings = {
    "general": {"label": "rare", "seed": 43},
    "rare_project": {"rare_workers": "explorer, analyst, critic"},
    "explorer": {"techniques": "label_balance"},
    "analyst": {
        "design": "experiment",
        "criterion": "balanced_accuracy",
        "steps": "split, sample, model",
        "split_techniques": "stratified",
        "sample_techniques": "none, smote, random_under",
        "model_techniques": "sk_logit",
    },
    "critic": {"techniques": "scorecard"},
}
project = amos.Project.create(settings, item = data)
print(project.result.tables["label_balance"]["count"].tolist())
# [944, 56]
scorecard = project.scorecard
print(sorted(scorecard.table["sample"]))
# ['none', 'random_under', 'smote']
print(list(scorecard.table.columns[4:7]))
# ['balanced_accuracy', 'accuracy', 'precision']
```

Only the training rows are resampled. The test rows keep their real balance,
so the scores show how each model would do on new data.

## Categorical data

Text and categories must become numbers before a model can use them. This
recipe makes a small set of court cases, cleans it, and compares three
encoders. The wrangler trims the text, removes the date, and makes columns
with few values categorical:

```python
rng = np.random.default_rng(0)
cases = pd.DataFrame({
    "court": rng.choice([" district ", "Appeals", "district", "supreme"], 400),
    "claim": rng.choice(["contract", "tort", "property", "civil rights"], 400),
    "judges": rng.integers(1, 4, 400),
    "filed": pd.date_range("2015-01-01", periods = 400, freq = "W"),
    "damages": rng.lognormal(10, 1, 400),
})
cases["won"] = (
    (cases["claim"] == "tort") ^ (rng.random(400) < 0.3)).map(
        {True: "plaintiff", False: "defendant"})
settings = {
    "general": {"label": "won", "seed": 43},
    "cases_project": {"cases_workers": "wrangler, analyst"},
    "wrangler": {"techniques": "strip_text, drop_columns, auto_categorize"},
    "strip_text_parameters": {"lowercase": True},
    "drop_columns_parameters": {"columns": ["filed"]},
    "auto_categorize_parameters": {"columns": ["court", "claim"]},
    "analyst": {
        "design": "experiment",
        "criterion": "roc_auc",
        "steps": "split, encode, scale, model",
        "split_techniques": "stratified",
        "encode_techniques": "one_hot, target, count",
        "scale_techniques": "standard",
        "model_techniques": "sk_logit",
    },
}
project = amos.Project.create(settings, item = cases)
result = project.result
print(result.categoricals, len(result.tables["analyst_comparison"]))
# [] 3
```

Any technique can also be applied by hand, to check what it does:

```python
cleaned = amos.cleaners.StripText().apply(cases.copy(), lowercase = True)
print(sorted(cleaned.data["court"].unique()))
# ['appeals', 'district', 'supreme']
```

The label is text, and the positive class is the last in sorted order
("plaintiff"), so `roc_auc` measures how well each model finds the cases that
plaintiffs won.

## Does feature selection matter?

A robustness check: compare keeping every feature with two ways of reducing
them, for two models. With `none` in a step, the table shows whether the step
helps at all:

```python
cancer = sklearn.datasets.load_breast_cancer(as_frame = True)
settings = {
    "general": {"seed": 43},
    "cancer_project": {
        "design": "experiment",
        "criterion": "roc_auc",
        "steps": "split, scale, reduce, model",
        "split_techniques": "stratified",
        "scale_techniques": "standard",
        "reduce_techniques": "none, k_best, pca_reduce",
        "model_techniques": "sk_logit, random_forest",
    },
    "k_best_parameters": {"k": 5},
    "random_forest_parameters": {"n_estimators": 50},
}
project = amos.Project.create(settings, item = cancer)
comparison = project.result.tables["cancer_comparison"]
print(len(comparison))
# 6
print(all(comparison["roc_auc"] > 0.9))
# True
```

The project section is itself a worker, so a small study can put its steps
there instead of in an "analyst" section.

## Explain a model

SHAP values show how much each feature moves each prediction. `shap_importance`
stores the mean absolute value for each feature, and the artist draws the
values with shap's own plots: `shap_beeswarm` shows each row's SHAP values for
the most important features, and `shap_waterfall` shows how the features move
one row's prediction:

```python
settings = {
    "general": {"seed": 43},
    "cancer_project": {
        "techniques": "stratified, gradient_boosting, shap_importance, shap_beeswarm, shap_waterfall",
    },
    "shap_importance_parameters": {"rows": 50},
}
project = amos.Project.create(settings, item = cancer)
importance = project.result.tables["shap_importance"]
print(importance.index.name, len(importance))
# feature 30
print(sorted(project.result.figures))
# ['shap_beeswarm', 'shap_waterfall']
```

The other SHAP plots draw the same values in other ways: `shap_bar`,
`shap_violin`, and `shap_heatmap` for every feature; `shap_decision` and
`shap_embedding` for every row; `shap_force` for one row (like the waterfall);
`shap_scatter` and `shap_partial_dependence` for one feature; and
`shap_group_difference` for the difference between two groups. They draw the
SHAP values that `shap_importance` found, or find them if it was not applied.
Set "row" (an index label) in `shap_waterfall_parameters` to explain a
particular row, and "category" to explain a class other than the last. `importance_plot` draws any
table of importances as plain bars. `permutation_importance` works with any
model, and `feature_importance` reports a model's own importances (or the size
of its coefficients).

## Check fairness across groups

Name the columns that identify groups in the "groups" setting of the
"general" section. They are kept out of the features, and the `scorecard`
adds fairness metrics for every branch when the label has two classes. The
`fairness` table compares the final model across the groups:

```python
data = cancer.frame.copy()
data["clinic"] = np.where(np.arange(len(data)) % 3 == 0, "north", "south")
settings = {
    "general": {"label": "target", "seed": 43, "groups": "clinic"},
    "cancer_project": {"cancer_workers": "analyst, critic"},
    "analyst": {
        "design": "experiment",
        "criterion": "roc_auc",
        "steps": "split, scale, model",
        "split_techniques": "stratified",
        "scale_techniques": "standard",
        "model_techniques": "sk_logit, random_forest",
    },
    "critic": {"techniques": "scorecard, fairness"},
}
project = amos.Project.create(settings, item = data)
print(list(project.scorecard.table.columns[-2:]))
# ['demographic_parity', 'equalized_odds']
print(list(project.result.tables["fairness"].index))
# ['north', 'south', 'difference', 'ratio']
```

An artist with `fairness_plot` draws the table, and `shap_group_difference`
shows which features explain the difference between the groups.

## Fixed effects and clustered standard errors

Panel data, such as cases decided by many judges, often calls for fixed
effects and standard errors clustered by group. `fixest` wraps pyfixest:

```python
rng = np.random.default_rng(7)
judges = rng.integers(0, 20, 1000)
severity = rng.normal(0, 1, 1000)
represented = rng.integers(0, 2, 1000)
sentence = (
    12 + 3 * severity - 2 * represented + judges / 4 + rng.normal(0, 2, 1000))
cases = pd.DataFrame({
    "severity": severity,
    "represented": represented,
    "judge": [f"judge {j}" for j in judges],
    "sentence": sentence,
})
settings = {
    "general": {"label": "sentence", "seed": 43, "groups": "judge"},
    "sentencing_project": {"techniques": "train_test, fixest"},
    "fixest_parameters": {"fixed_effects": "judge", "cluster": "judge"},
}
project = amos.Project.create(settings, item = cases)
coefficients = project.result.tables["fixest_coefficients"]
print(coefficients["coefficient"].round().to_dict())
# {'severity': 3.0, 'represented': -2.0}
```

The table also has the standard errors, p-values, and confidence intervals.
For a label with two classes, `fixest` fits a logit model.
