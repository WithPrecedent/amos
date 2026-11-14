"""Techniques that fit models and make predictions.

These are the techniques of the last step of the "analyst" stage. Each one
wraps a model from scikit-learn, xgboost, lightgbm, catboost, InterpretML,
TabPFN, statsmodels, or pyfixest. Most work for both tasks: `random_forest`,
for example, is a random forest classifier when the label is classified and a
random forest regressor when it is regressed. A model is
fitted to the training rows and predicts the test rows (or every row, if the
data has not been split). The fitted model, its predictions, and (for
classifiers) its predicted probabilities are stored in the dataset.

Every model accepts the parameters of the tool it wraps. It also accepts
"search" ("grid", "random", or "optuna"), which searches for the best values
of any parameter that is given as a list, using cross-validation on the
training rows ("cv" folds, 5 by default, and "n_iter" draws for a random or
Optuna search). If the dataset has groups, the folds of a search keep the
rows of each group of its first group column together, as the group
validators do.

If "use_gpu" is true in the "general" section of the settings (see
`Dataset.use_gpu`) and an NVIDIA GPU is found, the models that can be fitted
on a GPU (`catboost`, `lightgbm`, `tabpfn`, and `xgboost`) are fitted on the
first one. If a model cannot use the GPU (because its package was built
without GPU support, for example), it warns and is fitted on the CPU instead.
Every other model is fitted on the CPU, as every model is when "use_gpu" is
false (the default).

Contents:
    ModelTraits: what a model needs from the data, and what it assumes
        about it.
    Model: base class for techniques that fit models.
    LabelCoded: lets a classifier that needs classes numbered 0 to n - 1 use
        any classes.
    Statsmodel: adapts statsmodels regressions to the scikit-learn interface.
    FixedEffects: adapts pyfixest regressions to the scikit-learn interface.
    ProportionalHazards: adapts statsmodels' Cox model to the scikit-learn
        interface.
    Adaboost, Baseline, BinomialBayesMixedglm, Catboost, Cox, DecisionTree,
        ElasticNet, ExplainableBoosting, ExtraTrees, Fixest, GEE,
        GeneralizedPoisson, GLM, GradientBoosting, KNN, Lasso, Lightgbm,
        Linear, Logit, LogitSM, Mixedlm, Mnlogit, NaiveBayes,
        NegativeBinomial, NeuralNetwork, OLS, OrdinalRegression, Poisson,
        Probit, QuantileRegression, RandomForest, Ridge, RobustRegression,
        SVM, Tabpfn, WLS, Xgboost, ZeroInflatedPoisson: models.
    search_folds: the folds of a search, which keep the rows of each group
        together.

"""

from __future__ import annotations

__all__: list[str] = [
    'GEE',
    'GLM',
    'KNN',
    'OLS',
    'SVM',
    'WLS',
    'Adaboost',
    'Baseline',
    'BinomialBayesMixedglm',
    'Catboost',
    'Cox',
    'DecisionTree',
    'ElasticNet',
    'ExplainableBoosting',
    'ExtraTrees',
    'FixedEffects',
    'Fixest',
    'GeneralizedPoisson',
    'GradientBoosting',
    'LabelCoded',
    'Lasso',
    'Lightgbm',
    'Linear',
    'Logit',
    'LogitSM',
    'Mixedlm',
    'Mnlogit',
    'Model',
    'ModelTraits',
    'NaiveBayes',
    'NegativeBinomial',
    'NeuralNetwork',
    'OrdinalRegression',
    'Poisson',
    'Probit',
    'ProportionalHazards',
    'QuantileRegression',
    'RandomForest',
    'Ridge',
    'RobustRegression',
    'Statsmodel',
    'Tabpfn',
    'Xgboost',
    'ZeroInflatedPoisson',
    'search_folds',
]

from ._adapters import FixedEffects, ProportionalHazards, Statsmodel
from ._base import LabelCoded, Model, ModelTraits, search_folds
from ._techniques import (
    GEE,
    GLM,
    KNN,
    OLS,
    SVM,
    WLS,
    Adaboost,
    Baseline,
    BinomialBayesMixedglm,
    Catboost,
    Cox,
    DecisionTree,
    ElasticNet,
    ExplainableBoosting,
    ExtraTrees,
    Fixest,
    GeneralizedPoisson,
    GradientBoosting,
    Lasso,
    Lightgbm,
    Linear,
    Logit,
    LogitSM,
    Mixedlm,
    Mnlogit,
    NaiveBayes,
    NegativeBinomial,
    NeuralNetwork,
    OrdinalRegression,
    Poisson,
    Probit,
    QuantileRegression,
    RandomForest,
    Ridge,
    RobustRegression,
    Tabpfn,
    Xgboost,
    ZeroInflatedPoisson,
)

# The public names say that they are in this package (rather than in
# the private module that defines them), where they are imported
# from, so that records, pickles, and the documentation name them as
# `amos.models.Name`.
for _name in __all__:
    globals()[_name].__module__ = __name__
