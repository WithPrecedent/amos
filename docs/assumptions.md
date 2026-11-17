---
# The tables are wide, so the page leaves out the navigation on its left.
hide:
- navigation
---

# Assumptions

Every statistical technique assumes something about the data. A linear
regression assumes that the label is a linear function of the features and
that its errors are normal, independent, and of constant variance; a nearest
neighbors model assumes that the features are on similar scales. A technique
can still be applied when the data does not meet one of its assumptions, but
what it finds may be misleading: standard errors that are too small, or a
model that is beaten by a simpler one. Some techniques also **need**
something of the data, without which they cannot be applied at all (such as
a model that cannot use missing values).

Each technique lists its assumptions and needs by name, in its
`assumptions` and `needs`:

```python
import amos

print(amos.models.Linear.assumptions)
# ('linearity', 'normal_errors', 'constant_variance', 'independent_errors', 'no_multicollinearity')
print(amos.models.Linear.needs)
# ('numeric_features', 'no_missing_values', 'more_rows_than_features')
```

## How amos checks them

amos tests each assumption and need that it can, at the time when it can:

* **Pretests** (in the analyst stage) test an assumption before a technique
  is applied, on the real training rows. `check_normality`,
  `check_multicollinearity`, `check_linearity`, `check_scales`,
  `check_dispersion`, `check_features`, `check_label`, and
  `check_sample_size` each test some of them, and `check_assumptions` runs
  every pretest that the techniques of the workflow need.
* **Audits** (in the critic stage) test an assumption that can only be
  tested from the errors of a fitted model. `audit_residuals` tests whether
  the errors are normal, `audit_errors` whether they have constant variance,
  and `audit_independence` whether they are independent (in the dataset's
  `order`, and within its groups). `audit_assumptions` runs every audit that
  the dataset's model needs, and `auto_evaluator` applies it to a regression
  whose model makes such assumptions.

Whether or not a workflow names a pretest, amos runs the pretests of a
technique just before it is applied, so that they test the data that the
technique sees (after it was scaled or encoded, for example). The results
are in the dataset's "assumptions" table and its `history`. `auto_model`
uses the same pretests to rule out the models whose assumptions the data
does not meet.

Some assumptions cannot be tested from the data (such as that every
confounder is measured). They are listed below, but amos does not check
them.

With many rows, a test rejects even trivial departures from an assumption,
so an assumption fails only if its test rejects it (at the 5% level) and
the departure is large enough to matter: for example, errors that are
materially skewed or heavy-tailed, or nonlinear terms that explain more
than 2% of what a linear fit leaves unexplained.

## What amos does when one fails

The `assumption_violations` setting of the "general" section decides:

| Setting | What amos does |
| --- | --- |
| `warn` (the default) | Warns (with an `amos.AmosWarning`) and goes on. The violations are in the report, and the comparison table of an experiment has a "violations" column. |
| `raise` | Raises an `amos.AssumptionError`. |
| `ignore` | Goes on without a warning. The results of the tests are still recorded. |
| `skip` | Leaves the branch of an experiment that uses the technique out of the comparison (and warns outside of an experiment). |

A **hard** need is never let pass, whatever the setting: in an experiment,
the branch is left out, and elsewhere an `amos.AssumptionError` is raised.

<!-- file: assumptions.ini -->
```ini
[general]
label = target
order = date
assumption_violations = skip

[analyst]
design = experiment
criterion = rmse
steps = split, scale, check, model
split_techniques = time_split
scale_techniques = standard
check_techniques = check_assumptions
model_techniques = linear, ridge, random_forest

[critic]
techniques = audit_assumptions, scorecard
```

## Writing your own

An assumption is a subclass of `amos.Assumption` (and a need, of
`amos.Need`), named in snake case as techniques are. Name the pretest (or
audit) that tests it and write its `pretest` (or `audit`) method, then name
it in the `assumptions` of a technique:

```python
import dataclasses

import amos


@dataclasses.dataclass(frozen = True)
class PositiveLabel(amos.Assumption):
    """The label is greater than 0."""

    description = 'the label is greater than 0'
    pretested_by = 'check_label'

    def pretest(self, item, technique, found, cache):
        passed = bool((item.y > 0).all())
        detail = (
            'has a positive label' if passed else
            'assumes a positive label, but some values are 0 or less')
        return self._verdict(technique, self.pretested_by, passed, detail)


@dataclasses.dataclass
class LogLinear(amos.models.Linear):
    """Fits a linear regression of the log of the label."""

    assumptions = (*amos.models.Linear.assumptions, 'positive_label')


print(amos.conditions.borrow('positive_label').description)
# the label is greater than 0
```

The tables below list every assumption and need that amos knows, and then
the assumptions and needs of each technique, by genre (genres whose
techniques have none are left out).

## Every assumption

| Name | The technique assumes that | Pretest | Audit | Harms |
| --- | --- | --- | --- | --- |
| [`causal_sufficiency`][amos.core.assumptions.CausalSufficiency] | every common cause of the variables is measured |  |  | inference |
| [`constant_variance`][amos.core.assumptions.ConstantVariance] | the errors have the same variance for every row |  | [`audit_errors`][amos.audits.AuditErrors] | inference |
| [`continuity`][amos.core.assumptions.Continuity] | the outcome would change smoothly at the cutoff without the treatment |  |  | inference |
| [`equidispersion`][amos.core.assumptions.Equidispersion] | the variance of counts equals their mean | [`check_dispersion`][amos.pretests.CheckDispersion] |  | inference |
| [`independent_errors`][amos.core.assumptions.IndependentErrors] | the errors of different rows are independent |  | [`audit_independence`][amos.audits.AuditIndependence] | inference |
| [`linearity`][amos.core.assumptions.Linearity] | the label (or its log-odds) is a linear function of the features | [`check_linearity`][amos.pretests.CheckLinearity] |  | both |
| [`missing_at_random`][amos.core.assumptions.MissingAtRandom] | whether a value is missing depends only on values that are observed |  |  | both |
| [`missing_completely_at_random`][amos.core.assumptions.MissingCompletelyAtRandom] | whether a value is missing has nothing to do with any value |  |  | both |
| [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | no feature is nearly a linear function of the others | [`check_multicollinearity`][amos.pretests.CheckMulticollinearity] |  | both |
| [`normal_errors`][amos.core.assumptions.NormalErrors] | the errors of a regression are normal | [`check_normality`][amos.pretests.CheckNormality] | [`audit_residuals`][amos.audits.AuditResiduals] | inference |
| [`normal_features`][amos.core.assumptions.NormalFeatures] | the features are normal | [`check_normality`][amos.pretests.CheckNormality] |  | both |
| [`overlap`][amos.core.assumptions.Overlap] | every row could have been treated or not |  |  | inference |
| [`parallel_trends`][amos.core.assumptions.ParallelTrends] | without the treatment, the treated and other rows would have changed alike |  |  | inference |
| [`proportional_hazards`][amos.core.assumptions.ProportionalHazards] | each feature multiplies the hazard by the same amount at every time |  |  | both |
| [`proportional_odds`][amos.core.assumptions.ProportionalOdds] | each feature has the same effect on the odds of being above every class |  |  | both |
| [`similar_scales`][amos.core.assumptions.SimilarScales] | the features are on similar scales | [`check_scales`][amos.pretests.CheckScales] |  | prediction |
| [`stationarity`][amos.core.assumptions.Stationarity] | the distribution of each series does not change over time |  |  | both |
| [`unconfoundedness`][amos.core.assumptions.Unconfoundedness] | every common cause of the treatment and the outcome is a feature |  |  | inference |
| [`valid_instrument`][amos.core.assumptions.ValidInstrument] | the instrument changes the treatment and affects the outcome only through it |  |  | inference |

## Every need

| Name | The technique needs | Pretest | Hard |
| --- | --- | --- | --- |
| [`binary_label`][amos.core.needs.BinaryLabel] | the label has two classes | [`check_label`][amos.pretests.CheckLabel] | yes |
| [`count_label`][amos.core.needs.CountLabel] | the label is counts | [`check_label`][amos.pretests.CheckLabel] | yes |
| [`max_features`][amos.core.needs.MaxFeatures] | there are no more features than it works well with | [`check_sample_size`][amos.pretests.CheckSampleSize] | no |
| [`max_rows`][amos.core.needs.MaxRows] | there are no more training rows than it is fast enough for | [`check_sample_size`][amos.pretests.CheckSampleSize] | no |
| [`min_rows`][amos.core.needs.MinRows] | there are at least as many training rows as it learns well from | [`check_sample_size`][amos.pretests.CheckSampleSize] | no |
| [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] | there are more training rows than features | [`check_sample_size`][amos.pretests.CheckSampleSize] | yes |
| [`multiclass_label`][amos.core.needs.MulticlassLabel] | the label has three or more classes | [`check_label`][amos.pretests.CheckLabel] | no |
| [`no_missing_values`][amos.core.needs.NoMissingValues] | no feature has a missing value | [`check_features`][amos.pretests.CheckFeatures] | yes |
| [`numeric_features`][amos.core.needs.NumericFeatures] | every feature is a number or a boolean | [`check_features`][amos.pretests.CheckFeatures] | yes |
| [`ordinal_label`][amos.core.needs.OrdinalLabel] | the label has three or more ordered classes | [`check_label`][amos.pretests.CheckLabel] | no |

## Imputers (analyst)

| Name | Assumptions | Needs |
| --- | --- | --- |
| [`iterative_impute`][amos.imputers.IterativeImpute] | [`missing_at_random`][amos.core.assumptions.MissingAtRandom] | none |
| [`knn_impute`][amos.imputers.KnnImpute] | [`missing_at_random`][amos.core.assumptions.MissingAtRandom] | none |
| [`mean_impute`][amos.imputers.MeanImpute] | [`missing_completely_at_random`][amos.core.assumptions.MissingCompletelyAtRandom] | none |
| [`median_impute`][amos.imputers.MedianImpute] | [`missing_completely_at_random`][amos.core.assumptions.MissingCompletelyAtRandom] | none |
| [`mode_impute`][amos.imputers.ModeImpute] | [`missing_completely_at_random`][amos.core.assumptions.MissingCompletelyAtRandom] | none |

## Samplers (analyst)

| Name | Assumptions | Needs |
| --- | --- | --- |
| [`adasyn`][amos.samplers.Adasyn] | [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`borderline_smote`][amos.samplers.BorderlineSmote] | [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`near_miss`][amos.samplers.NearMiss] | [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`random_over`][amos.samplers.RandomOver] | none | none |
| [`random_under`][amos.samplers.RandomUnder] | none | none |
| [`smote`][amos.samplers.Smote] | [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`smote_enn`][amos.samplers.SmoteEnn] | [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`smote_tomek`][amos.samplers.SmoteTomek] | [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`tomek_links`][amos.samplers.TomekLinks] | [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |

## Models (analyst)

| Name | Assumptions | Needs |
| --- | --- | --- |
| [`adaboost`][amos.models.Adaboost] | none | [`numeric_features`][amos.core.needs.NumericFeatures] |
| [`baseline`][amos.models.Baseline] | none | none |
| [`binomial_bayes_mixedglm`][amos.models.BinomialBayesMixedglm] | [`linearity`][amos.core.assumptions.Linearity], [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`binary_label`][amos.core.needs.BinaryLabel], [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`catboost`][amos.models.Catboost] | none | none |
| [`cox`][amos.models.Cox] | [`proportional_hazards`][amos.core.assumptions.ProportionalHazards], [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`decision_tree`][amos.models.DecisionTree] | none | [`numeric_features`][amos.core.needs.NumericFeatures] |
| [`elastic_net`][amos.models.ElasticNet] | [`linearity`][amos.core.assumptions.Linearity], [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`explainable_boosting`][amos.models.ExplainableBoosting] | none | none |
| [`extra_trees`][amos.models.ExtraTrees] | none | [`numeric_features`][amos.core.needs.NumericFeatures] |
| [`fixest`][amos.models.Fixest] | [`linearity`][amos.core.assumptions.Linearity], [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`gee`][amos.models.GEE] | [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`generalized_poisson`][amos.models.GeneralizedPoisson] | [`independent_errors`][amos.core.assumptions.IndependentErrors] | [`count_label`][amos.core.needs.CountLabel], [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`glm`][amos.models.GLM] | [`independent_errors`][amos.core.assumptions.IndependentErrors], [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`gradient_boosting`][amos.models.GradientBoosting] | none | none |
| [`knn`][amos.models.KNN] | [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`max_rows`][amos.core.needs.MaxRows], [`max_features`][amos.core.needs.MaxFeatures] |
| [`lasso`][amos.models.Lasso] | [`linearity`][amos.core.assumptions.Linearity], [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`lightgbm`][amos.models.Lightgbm] | none | none |
| [`linear`][amos.models.Linear] | [`linearity`][amos.core.assumptions.Linearity], [`normal_errors`][amos.core.assumptions.NormalErrors], [`constant_variance`][amos.core.assumptions.ConstantVariance], [`independent_errors`][amos.core.assumptions.IndependentErrors], [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`logit`][amos.models.Logit] | [`linearity`][amos.core.assumptions.Linearity], [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`logit_sm`][amos.models.LogitSM] | [`linearity`][amos.core.assumptions.Linearity], [`independent_errors`][amos.core.assumptions.IndependentErrors], [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`binary_label`][amos.core.needs.BinaryLabel], [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`mixedlm`][amos.models.Mixedlm] | [`linearity`][amos.core.assumptions.Linearity], [`normal_errors`][amos.core.assumptions.NormalErrors], [`constant_variance`][amos.core.assumptions.ConstantVariance], [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`mnlogit`][amos.models.Mnlogit] | [`linearity`][amos.core.assumptions.Linearity], [`independent_errors`][amos.core.assumptions.IndependentErrors], [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`multiclass_label`][amos.core.needs.MulticlassLabel], [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`naive_bayes`][amos.models.NaiveBayes] | [`normal_features`][amos.core.assumptions.NormalFeatures], [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`negative_binomial`][amos.models.NegativeBinomial] | [`independent_errors`][amos.core.assumptions.IndependentErrors] | [`count_label`][amos.core.needs.CountLabel], [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`neural_network`][amos.models.NeuralNetwork] | [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`min_rows`][amos.core.needs.MinRows] |
| [`ols`][amos.models.OLS] | [`linearity`][amos.core.assumptions.Linearity], [`constant_variance`][amos.core.assumptions.ConstantVariance], [`independent_errors`][amos.core.assumptions.IndependentErrors], [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`ordinal_regression`][amos.models.OrdinalRegression] | [`proportional_odds`][amos.core.assumptions.ProportionalOdds], [`independent_errors`][amos.core.assumptions.IndependentErrors] | [`ordinal_label`][amos.core.needs.OrdinalLabel], [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`poisson`][amos.models.Poisson] | [`equidispersion`][amos.core.assumptions.Equidispersion], [`independent_errors`][amos.core.assumptions.IndependentErrors] | [`count_label`][amos.core.needs.CountLabel], [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`probit`][amos.models.Probit] | [`linearity`][amos.core.assumptions.Linearity], [`independent_errors`][amos.core.assumptions.IndependentErrors], [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`binary_label`][amos.core.needs.BinaryLabel], [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`quantile_regression`][amos.models.QuantileRegression] | [`linearity`][amos.core.assumptions.Linearity], [`independent_errors`][amos.core.assumptions.IndependentErrors] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`random_forest`][amos.models.RandomForest] | none | [`numeric_features`][amos.core.needs.NumericFeatures] |
| [`ridge`][amos.models.Ridge] | [`linearity`][amos.core.assumptions.Linearity], [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues] |
| [`robust_regression`][amos.models.RobustRegression] | [`linearity`][amos.core.assumptions.Linearity], [`independent_errors`][amos.core.assumptions.IndependentErrors] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`svm`][amos.models.SVM] | [`similar_scales`][amos.core.assumptions.SimilarScales] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`max_rows`][amos.core.needs.MaxRows] |
| [`tabpfn`][amos.models.Tabpfn] | none | none |
| [`wls`][amos.models.WLS] | [`linearity`][amos.core.assumptions.Linearity], [`independent_errors`][amos.core.assumptions.IndependentErrors], [`no_multicollinearity`][amos.core.assumptions.NoMulticollinearity] | [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |
| [`xgboost`][amos.models.Xgboost] | none | [`numeric_features`][amos.core.needs.NumericFeatures] |
| [`zero_inflated_poisson`][amos.models.ZeroInflatedPoisson] | [`independent_errors`][amos.core.assumptions.IndependentErrors] | [`count_label`][amos.core.needs.CountLabel], [`numeric_features`][amos.core.needs.NumericFeatures], [`no_missing_values`][amos.core.needs.NoMissingValues], [`more_rows_than_features`][amos.core.needs.MoreRowsThanFeatures] |

## Inferers (critic)

| Name | Assumptions | Needs |
| --- | --- | --- |
| [`difference_in_differences`][amos.inferers.DifferenceInDifferences] | [`parallel_trends`][amos.core.assumptions.ParallelTrends], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`distance_matching`][amos.inferers.DistanceMatching] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`doubly_robust`][amos.inferers.DoublyRobust] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`dr_learner`][amos.inferers.DRLearner] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`glm_adjustment`][amos.inferers.GLMAdjustment] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`instrumental_variable`][amos.inferers.InstrumentalVariable] | [`valid_instrument`][amos.core.assumptions.ValidInstrument] | none |
| [`interactive_iv`][amos.inferers.InteractiveIV] | [`valid_instrument`][amos.core.assumptions.ValidInstrument] | none |
| [`interactive_regression`][amos.inferers.InteractiveRegression] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`lpcmci`][amos.inferers.LPCMCI] | [`stationarity`][amos.core.assumptions.Stationarity] | none |
| [`partially_linear`][amos.inferers.PartiallyLinear] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness] | none |
| [`partially_linear_iv`][amos.inferers.PartiallyLinearIV] | [`valid_instrument`][amos.core.assumptions.ValidInstrument] | none |
| [`partially_linear_panel`][amos.inferers.PartiallyLinearPanel] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness] | none |
| [`partially_logistic`][amos.inferers.PartiallyLogistic] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness] | none |
| [`pcmci`][amos.inferers.PCMCI] | [`stationarity`][amos.core.assumptions.Stationarity], [`causal_sufficiency`][amos.core.assumptions.CausalSufficiency] | none |
| [`pcmci_plus`][amos.inferers.PCMCIPlus] | [`stationarity`][amos.core.assumptions.Stationarity], [`causal_sufficiency`][amos.core.assumptions.CausalSufficiency] | none |
| [`potential_outcomes`][amos.inferers.PotentialOutcomes] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`propensity_matching`][amos.inferers.PropensityMatching] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`propensity_stratification`][amos.inferers.PropensityStratification] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`propensity_weighting`][amos.inferers.PropensityWeighting] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`quantile_effects`][amos.inferers.QuantileEffects] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`regression_adjustment`][amos.inferers.RegressionAdjustment] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`regression_discontinuity`][amos.inferers.RegressionDiscontinuity] | [`continuity`][amos.core.assumptions.Continuity] | none |
| [`s_learner`][amos.inferers.SLearner] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`sample_selection`][amos.inferers.SampleSelection] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness] | none |
| [`t_learner`][amos.inferers.TLearner] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`time_series_effect`][amos.inferers.TimeSeriesEffect] | [`stationarity`][amos.core.assumptions.Stationarity], [`causal_sufficiency`][amos.core.assumptions.CausalSufficiency] | none |
| [`tmle`][amos.inferers.TMLE] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
| [`x_learner`][amos.inferers.XLearner] | [`unconfoundedness`][amos.core.assumptions.Unconfoundedness], [`overlap`][amos.core.assumptions.Overlap] | none |
