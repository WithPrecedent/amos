"""The models that amos wraps.

Contents:
    Adaboost, Baseline, BinomialBayesMixedglm, Catboost, Cox, DecisionTree,
        ElasticNet, ExplainableBoosting, ExtraTrees, Fixest, GEE,
        GeneralizedPoisson, GLM, GradientBoosting, KNN, Lasso, Lightgbm, Linear,
        Logit, LogitSM, Mixedlm, Mnlogit, NaiveBayes, NegativeBinomial,
        NeuralNetwork, OLS, OrdinalRegression, Poisson, Probit,
        QuantileRegression, RandomForest, Ridge, RobustRegression, SVM, Tabpfn,
        WLS, Xgboost, ZeroInflatedPoisson: models.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, ClassVar

from .. import base
from ._adapters import FixedEffects, ProportionalHazards, Statsmodel
from ._base import Model, ModelTraits

""" Models """


class _Statsmodels:
    """Makes a model one kind of `Statsmodel` (see its table of kinds).

    It is a plain class rather than a `Model`, so that the library does not
    store it as a technique.

    """

    # The kind of `Statsmodel`.
    kind: ClassVar[str] = 'ols'

    def _prepare(
        self,
        item: base.Dataset,  # noqa: ARG002
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Makes the statsmodels adapter the model's kind.

        Args:
            item: the dataset to model.
            parameters: parameters for the model.

        Returns:
            The parameters, with "kind".

        """
        return {**parameters, 'kind': self.kind}


@dataclasses.dataclass
class Adaboost(Model):
    """AdaBoost: a sequence of small trees, each fixing the last one's errors.

    Wraps:
        - [`sklearn.ensemble.AdaBoostClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
          from scikit-learn, to classify.
        - [`sklearn.ensemble.AdaBoostRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
          from scikit-learn, to regress.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.ensemble.AdaBoostClassifier',
        'regress': 'sklearn.ensemble.AdaBoostRegressor'}


@dataclasses.dataclass
class Baseline(Model):
    """Predicts the most common class (or the mean) for every row.

    Every model should beat this. It is the standard comparison for showing
    that a model learned something from the features.

    Wraps:
        - [`sklearn.dummy.DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
          from scikit-learn, to classify.
        - [`sklearn.dummy.DummyRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
          from scikit-learn, to regress.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.dummy.DummyClassifier',
        'regress': 'sklearn.dummy.DummyRegressor'}


@dataclasses.dataclass
class BinomialBayesMixedglm(_Statsmodels, Model):
    """A Bayesian logistic regression with a random intercept for each group.

    statsmodels' `BinomialBayesMixedGLM`, fitted by variational Bayes, for a
    label of two classes and rows that are correlated within groups (such as
    the decisions of each judge). Set "groups" to the column of groups (by
    default, the dataset's first group). Its table of coefficients has the
    mean and standard deviation of each coefficient's posterior, and its 95%
    credible interval. Predictions use the fixed effects only, so they suit
    groups the model has not seen.

    Wraps:
        [`statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM`](https://www.statsmodels.org/stable/generated/statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.html)
        from statsmodels.

    """

    tools: ClassVar[Mapping[str, Any]] = {'classify': Statsmodel}
    column_parameters: ClassVar[tuple[str, ...]] = ('groups',)
    kind: ClassVar[str] = 'binomial_bayes_mixedglm'

    def apply(
        self,
        item: base.Dataset,
        search: str | None = None,
        cv: int = 5,
        n_iter: int = 10,
        scoring: str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Fits the model, with the dataset's first group as "groups".

        Args:
            item: the dataset to model.
            search: as for `Model.apply`.
            cv: as for `Model.apply`.
            n_iter: as for `Model.apply`.
            scoring: as for `Model.apply`.
            **kwargs: parameters for the model.

        Returns:
            The dataset, with the fitted model and its predictions.

        """
        return super().apply(
            item,
            search = search,
            cv = cv,
            n_iter = n_iter,
            scoring = scoring,
            **_with_groups(item, kwargs))


@dataclasses.dataclass
class Catboost(Model):
    """CatBoost gradient boosting, which uses categorical features directly.

    Categorical features (the `pandas` "category" type) are passed to it as
    categories, so they need no encoder. With "use_gpu", it is fitted on the
    GPU, where catboost's results can vary slightly from one run to the next.

    Wraps:
        - [`catboost.CatBoostClassifier`](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier)
          from catboost, to classify.
        - [`catboost.CatBoostRegressor`](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor)
          from catboost, to regress.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'catboost',
        complexity = 3,
        missing = True,
        categories = True)

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'catboost.CatBoostClassifier',
        'regress': 'catboost.CatBoostRegressor'}
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {
            'verbose': 0,
            # Otherwise, catboost writes files to the current folder.
            'allow_writing_files': False})

    def _device_parameters(self, *, gpu: bool) -> dict[str, Any] | None:
        """Returns the parameters that fit catboost on a GPU or the CPU.

        Args:
            gpu: whether to fit the model on the first GPU (or on the CPU).

        Returns:
            The "task_type" and, for a GPU, the "devices" to use.

        """
        if gpu:
            return {'task_type': 'GPU', 'devices': '0'}
        return {'task_type': 'CPU'}

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Names the categorical features for catboost.

        Args:
            item: the dataset to model.
            parameters: parameters for the model.

        Returns:
            The parameters, with "cat_features" if there are categorical
                features.

        """
        if item.categoricals:
            parameters.setdefault('cat_features', item.categoricals)
        return parameters


@dataclasses.dataclass
class Cox(Model):
    """Cox proportional hazards regression of the time until an event.

    The label is the time until the event (such as rearrest) or until the row
    stopped being observed. Set "event" to the column that is 1 if the event
    happened and 0 if the row was censored (it is not used as a feature).
    The coefficients, hazard ratios, standard errors, p-values, and
    confidence intervals are stored in the dataset's `tables` as
    "{name}_coefficients". The predictions are the expected times until the
    event.

    Wraps:
        [`statsmodels.duration.hazard_regression.PHReg`](https://www.statsmodels.org/stable/generated/statsmodels.duration.hazard_regression.PHReg.html)
        from statsmodels.

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': ProportionalHazards}
    column_parameters: ClassVar[tuple[str, ...]] = ('event',)


@dataclasses.dataclass
class DecisionTree(Model):
    """A single decision tree.

    Wraps:
        - [`sklearn.tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
          from scikit-learn, to classify.
        - [`sklearn.tree.DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
          from scikit-learn, to regress.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.tree.DecisionTreeClassifier',
        'regress': 'sklearn.tree.DecisionTreeRegressor'}


@dataclasses.dataclass
class ElasticNet(Model):
    """Linear regression with both lasso and ridge penalties.

    Wraps:
        [`sklearn.linear_model.ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
        from scikit-learn.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'regress': 'sklearn.linear_model.ElasticNet'}


@dataclasses.dataclass
class ExplainableBoosting(Model):
    """An Explainable Boosting Machine from InterpretML.

    It is nearly as accurate as gradient boosting, but it is a sum of one
    function of each feature (and of a few pairs), so the effect of every
    feature can be shown exactly. `feature_importance` reports the importance
    of each term.

    Wraps:
        - [`interpret.glassbox.ExplainableBoostingClassifier`](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html)
          from InterpretML, to classify.
        - [`interpret.glassbox.ExplainableBoostingRegressor`](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html)
          from InterpretML, to regress.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'interpret.glassbox.ExplainableBoostingClassifier',
        'regress': 'interpret.glassbox.ExplainableBoostingRegressor'}


@dataclasses.dataclass
class ExtraTrees(Model):
    """An ensemble of extremely randomized trees.

    Wraps:
        - [`sklearn.ensemble.ExtraTreesClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
          from scikit-learn, to classify.
        - [`sklearn.ensemble.ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
          from scikit-learn, to regress.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.ensemble.ExtraTreesClassifier',
        'regress': 'sklearn.ensemble.ExtraTreesRegressor'}


@dataclasses.dataclass
class Fixest(Model):
    """Regression with fixed effects and clustered standard errors (pyfixest).

    Set "fixed_effects" to the columns whose levels each get their own
    intercept (such as "judge" or "year") and "cluster" to the column to
    cluster the standard errors by (such as "court"). Those columns are
    usually the dataset's `groups`; they are not used as features. The label
    is regressed by least squares, or by a logit model if it is classified.
    Set "family" to "poisson" for counts or "probit" for a probit model. The
    coefficients, standard errors, p-values, and confidence intervals are
    stored in the dataset's `tables` as "{name}_coefficients".

    Wraps:
        - [`pyfixest.feols`](https://py-econometrics.github.io/pyfixest/reference/estimation.api.feols.feols.html)
          from pyfixest, for the "gaussian" family.
        - [`pyfixest.fepois`](https://py-econometrics.github.io/pyfixest/reference/estimation.api.fepois.fepois.html)
          from pyfixest, for the "poisson" family.
        - [`pyfixest.feglm`](https://py-econometrics.github.io/pyfixest/reference/estimation.api.feglm.feglm.html)
          from pyfixest, for the "logit" and "probit" families.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': FixedEffects,
        'regress': FixedEffects}
    column_parameters: ClassVar[tuple[str, ...]] = ('fixed_effects', 'cluster')

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the family for the task.

        Args:
            item: the dataset to model.
            parameters: parameters for the model.

        Returns:
            The parameters, with a "family".

        """
        family = 'logit' if item.task == 'classify' else 'gaussian'
        return {'family': family, **parameters}


@dataclasses.dataclass
class GEE(Model):
    """Generalized estimating equations from statsmodels, with inference.

    A `glm` for rows that are correlated within groups (such as several
    cases of one court): the coefficients are averages over the groups, and
    their standard errors allow for the correlation. Set "groups" to the
    column of groups (by default, the dataset's first group), "family" as
    for a `glm`, and "covariance" to "exchangeable" to model the correlation
    (by default, "independence").

    Wraps:
        [`statsmodels.genmod.generalized_estimating_equations.GEE`](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_estimating_equations.GEE.html)
        from statsmodels.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': Statsmodel,
        'regress': Statsmodel}
    column_parameters: ClassVar[tuple[str, ...]] = ('groups',)

    def apply(
        self,
        item: base.Dataset,
        search: str | None = None,
        cv: int = 5,
        n_iter: int = 10,
        scoring: str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Fits the model, with the dataset's first group as "groups".

        Args:
            item: the dataset to model.
            search: as for `Model.apply`.
            cv: as for `Model.apply`.
            n_iter: as for `Model.apply`.
            scoring: as for `Model.apply`.
            **kwargs: parameters for the model.

        Returns:
            The dataset, with the fitted model and its predictions.

        """
        return super().apply(
            item,
            search = search,
            cv = cv,
            n_iter = n_iter,
            scoring = scoring,
            **_with_groups(item, kwargs))

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the family for the task.

        Args:
            item: the dataset to model.
            parameters: parameters for the model.

        Returns:
            The parameters, with "kind" and "family".

        """
        family = _family(self.name, item, parameters)
        return {**parameters, 'family': family, 'kind': 'gee'}


@dataclasses.dataclass
class GeneralizedPoisson(_Statsmodels, Model):
    """Generalized Poisson regression of counts, with inference.

    Like `poisson`, for counts that vary more (or less) than a Poisson
    distribution's.

    Wraps:
        [`statsmodels.discrete.discrete_model.GeneralizedPoisson`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.GeneralizedPoisson.html)
        from statsmodels.

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'generalized_poisson'
    counts: ClassVar[bool] = True


@dataclasses.dataclass
class GLM(Model):
    """A generalized linear model from statsmodels, with inference.

    By default, the family is "binomial" (logistic regression without a
    penalty) for classification and "gaussian" for regression. Set "family"
    to use another (such as "poisson" for counts). The coefficients, standard
    errors, p-values, and confidence intervals are stored in the dataset's
    `tables` as "{name}_coefficients".

    Wraps:
        [`statsmodels.genmod.generalized_linear_model.GLM`](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html)
        from statsmodels.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': Statsmodel,
        'regress': Statsmodel}

    def _prepare(
        self,
        item: base.Dataset,
        parameters: dict[str, Any]) -> dict[str, Any]:
        """Chooses the family for the task.

        Args:
            item: the dataset to model.
            parameters: parameters for the model.

        Returns:
            The parameters, with "kind" and "family".

        """
        family = _family(self.name, item, parameters)
        return {**parameters, 'family': family, 'kind': 'glm'}


@dataclasses.dataclass
class GradientBoosting(Model):
    """Histogram-based gradient boosting from scikit-learn.

    It is fast on large data and handles missing values itself.

    Wraps:
        - [`sklearn.ensemble.HistGradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
          from scikit-learn, to classify.
        - [`sklearn.ensemble.HistGradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
          from scikit-learn, to regress.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'sklearn',
        complexity = 3,
        missing = True,
        categories = True)

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.ensemble.HistGradientBoostingClassifier',
        'regress': 'sklearn.ensemble.HistGradientBoostingRegressor'}


@dataclasses.dataclass
class KNN(Model):
    """Predicts from the k nearest training rows (5 by default).

    Wraps:
        - [`sklearn.neighbors.KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
          from scikit-learn, to classify.
        - [`sklearn.neighbors.KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
          from scikit-learn, to regress.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'sklearn',
        complexity = 2,
        scaled = True,
        max_rows = 50_000,
        max_features = 30)

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.neighbors.KNeighborsClassifier',
        'regress': 'sklearn.neighbors.KNeighborsRegressor'}


@dataclasses.dataclass
class Lasso(Model):
    """Linear regression with a lasso (L1) penalty.

    Wraps:
        [`sklearn.linear_model.Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
        from scikit-learn.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'sklearn',
        complexity = 1,
        scaled = True)

    tools: ClassVar[Mapping[str, Any]] = {
        'regress': 'sklearn.linear_model.Lasso'}


@dataclasses.dataclass
class Lightgbm(Model):
    """LightGBM gradient boosting.

    With "use_gpu", it is fitted on the GPU with OpenCL, which lightgbm's
    packages for Linux support (its packages for Windows and macOS do not, so
    it is fitted on the CPU there).

    Wraps:
        - [`lightgbm.LGBMClassifier`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)
          from lightgbm, to classify.
        - [`lightgbm.LGBMRegressor`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
          from lightgbm, to regress.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'lightgbm',
        complexity = 3,
        missing = True,
        categories = True)

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'lightgbm.LGBMClassifier',
        'regress': 'lightgbm.LGBMRegressor'}
    # lightgbm's models take "verbose" through `**kwargs`. It is -1 by
    # default, so lightgbm does not print a warning for each tree it cannot
    # split further (which small data has many of).
    keyword_parameters: ClassVar[tuple[str, ...]] = ('verbose',)
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'verbose': -1})

    def _device_parameters(self, *, gpu: bool) -> dict[str, Any] | None:
        """Returns the parameters that fit lightgbm on a GPU or the CPU.

        Args:
            gpu: whether to fit the model on the GPU (or on the CPU).

        Returns:
            The "device_type": "gpu" (lightgbm's OpenCL version, which its
                packages include) or "cpu".

        """
        return {'device_type': 'gpu' if gpu else 'cpu'}


@dataclasses.dataclass
class Linear(Model):
    """Ordinary least squares regression from scikit-learn.

    Wraps:
        [`sklearn.linear_model.LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
        from scikit-learn.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'sklearn',
        complexity = 1,
        normal_errors = True,
        independent = True,
        wide = False)

    tools: ClassVar[Mapping[str, Any]] = {
        'regress': 'sklearn.linear_model.LinearRegression'}


@dataclasses.dataclass
class Logit(Model):
    """Logistic regression from scikit-learn.

    scikit-learn adds a ridge (L2) penalty by default. Use `logit_sm` for
    statsmodels' logistic regression, without a penalty and with p-values.

    Wraps:
        [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
        from scikit-learn.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'sklearn',
        complexity = 1,
        scaled = True)

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.linear_model.LogisticRegression'}
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'max_iter': 1000})


@dataclasses.dataclass
class LogitSM(_Statsmodels, Model):
    """Logistic regression from statsmodels, with inference.

    Unlike `logit` (scikit-learn's logistic regression), it has no penalty,
    so its coefficients have standard errors, p-values, and confidence
    intervals (as does `glm` for a label of two classes).

    Wraps:
        [`statsmodels.discrete.discrete_model.Logit`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html)
        from statsmodels.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'statsmodels',
        complexity = 1,
        wide = False,
        max_classes = 2)

    tools: ClassVar[Mapping[str, Any]] = {'classify': Statsmodel}
    kind: ClassVar[str] = 'logit'


@dataclasses.dataclass
class Mixedlm(_Statsmodels, Model):
    """A linear mixed model from statsmodels, with inference.

    A linear regression with a random intercept for each group (such as each
    judge), for rows that are correlated within groups. Set "groups" to the
    column of groups (by default, the dataset's first group). Predictions use
    the fixed effects only, so they suit groups the model has not seen.

    Wraps:
        [`statsmodels.regression.mixed_linear_model.MixedLM`](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html)
        from statsmodels.

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    column_parameters: ClassVar[tuple[str, ...]] = ('groups',)
    kind: ClassVar[str] = 'mixedlm'

    def apply(
        self,
        item: base.Dataset,
        search: str | None = None,
        cv: int = 5,
        n_iter: int = 10,
        scoring: str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Fits the model, with the dataset's first group as "groups".

        Args:
            item: the dataset to model.
            search: as for `Model.apply`.
            cv: as for `Model.apply`.
            n_iter: as for `Model.apply`.
            scoring: as for `Model.apply`.
            **kwargs: parameters for the model.

        Returns:
            The dataset, with the fitted model and its predictions.

        """
        return super().apply(
            item,
            search = search,
            cv = cv,
            n_iter = n_iter,
            scoring = scoring,
            **_with_groups(item, kwargs))


@dataclasses.dataclass
class Mnlogit(_Statsmodels, Model):
    """Multinomial logistic regression from statsmodels, with inference.

    For a label of any number of classes. Each class other than the
    reference has its own coefficients, compared with the reference. Set
    "reference" to the class to compare with (by default, the first class,
    sorted).

    Wraps:
        [`statsmodels.discrete.discrete_model.MNLogit`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.MNLogit.html)
        from statsmodels.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'statsmodels',
        complexity = 2,
        wide = False,
        min_classes = 3)

    tools: ClassVar[Mapping[str, Any]] = {'classify': Statsmodel}
    kind: ClassVar[str] = 'mnlogit'


@dataclasses.dataclass
class NaiveBayes(Model):
    """Gaussian naive Bayes.

    Wraps:
        [`sklearn.naive_bayes.GaussianNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
        from scikit-learn.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'sklearn',
        complexity = 1,
        normal_features = True,
        independent = True)

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.naive_bayes.GaussianNB'}


@dataclasses.dataclass
class NegativeBinomial(_Statsmodels, Model):
    """Negative binomial regression of counts, with inference.

    Like `poisson`, for counts that vary more than a Poisson distribution's
    (which is common). The extra variance ("alpha") is estimated.

    Wraps:
        [`statsmodels.discrete.discrete_model.NegativeBinomial`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html)
        from statsmodels.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'statsmodels',
        complexity = 1,
        wide = False,
        counts = 'overdispersed')

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'negative_binomial'
    counts: ClassVar[bool] = True


@dataclasses.dataclass
class NeuralNetwork(Model):
    """A multi-layer perceptron (a simple neural network).

    Wraps:
        - [`sklearn.neural_network.MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
          from scikit-learn, to classify.
        - [`sklearn.neural_network.MLPRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
          from scikit-learn, to regress.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'sklearn',
        complexity = 4,
        scaled = True,
        min_rows = 1000)

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.neural_network.MLPClassifier',
        'regress': 'sklearn.neural_network.MLPRegressor'}
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'max_iter': 1000})


@dataclasses.dataclass
class OLS(_Statsmodels, Model):
    """Ordinary least squares regression from statsmodels, with inference.

    The coefficients, standard errors, p-values, and confidence intervals are
    stored in the dataset's `tables` as "{name}_coefficients".

    Wraps:
        [`statsmodels.regression.linear_model.OLS`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)
        from statsmodels.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'statsmodels',
        complexity = 1,
        independent = True,
        wide = False)

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'ols'


@dataclasses.dataclass
class OrdinalRegression(_Statsmodels, Model):
    """Ordinal regression (ordered logit or probit) from statsmodels.

    For a label of ordered classes (such as a sentence that is lenient,
    typical, or harsh). The classes are ordered as the categories of a label
    that is an ordered categorical, and otherwise sorted. Set "distribution"
    to "probit" for an ordered probit.

    Wraps:
        [`statsmodels.miscmodels.ordinal_model.OrderedModel`](https://www.statsmodels.org/stable/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html)
        from statsmodels.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'statsmodels',
        complexity = 1,
        wide = False,
        ordinal = True)

    tools: ClassVar[Mapping[str, Any]] = {'classify': Statsmodel}
    kind: ClassVar[str] = 'ordinal_regression'


@dataclasses.dataclass
class Poisson(_Statsmodels, Model):
    """Poisson regression of counts (such as the number of arrests).

    The label must be whole numbers of 0 or more. amos classifies whole
    numbers with few values, so set "task" to "regress" for such counts.

    Wraps:
        [`statsmodels.discrete.discrete_model.Poisson`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Poisson.html)
        from statsmodels.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'statsmodels',
        complexity = 1,
        wide = False,
        counts = 'equidispersed')

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'poisson'
    counts: ClassVar[bool] = True


@dataclasses.dataclass
class Probit(_Statsmodels, Model):
    """Probit regression of two classes from statsmodels, with inference.

    Wraps:
        [`statsmodels.discrete.discrete_model.Probit`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Probit.html)
        from statsmodels.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'statsmodels',
        complexity = 1,
        wide = False,
        max_classes = 2)

    tools: ClassVar[Mapping[str, Any]] = {'classify': Statsmodel}
    kind: ClassVar[str] = 'probit'


@dataclasses.dataclass
class QuantileRegression(_Statsmodels, Model):
    """Quantile regression from statsmodels, with inference.

    Predicts a quantile of the label ("quantile", by default the median)
    rather than its mean, which outliers sway less.

    Wraps:
        [`statsmodels.regression.quantile_regression.QuantReg`](https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html)
        from statsmodels.

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'quantile_regression'


@dataclasses.dataclass
class RandomForest(Model):
    """A random forest.

    Wraps:
        - [`sklearn.ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
          from scikit-learn, to classify.
        - [`sklearn.ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
          from scikit-learn, to regress.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'sklearn',
        complexity = 3,
        missing = True)

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.ensemble.RandomForestClassifier',
        'regress': 'sklearn.ensemble.RandomForestRegressor'}


@dataclasses.dataclass
class Ridge(Model):
    """Linear regression with a ridge (L2) penalty.

    Wraps:
        [`sklearn.linear_model.Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
        from scikit-learn.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'sklearn',
        complexity = 1,
        scaled = True)

    tools: ClassVar[Mapping[str, Any]] = {
        'regress': 'sklearn.linear_model.Ridge'}


@dataclasses.dataclass
class RobustRegression(_Statsmodels, Model):
    """Robust linear regression from statsmodels, with inference.

    Gives less weight to rows with large errors (outliers), by "norm"
    (Huber's, by default).

    Wraps:
        [`statsmodels.robust.robust_linear_model.RLM`](https://www.statsmodels.org/stable/generated/statsmodels.robust.robust_linear_model.RLM.html)
        from statsmodels.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'statsmodels',
        complexity = 1,
        wide = False,
        outliers = True)

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'robust_regression'


@dataclasses.dataclass
class SVM(Model):
    """A support vector machine (with probabilities for classification).

    Wraps:
        - [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
          from scikit-learn, to classify.
        - [`sklearn.svm.SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
          from scikit-learn, to regress.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'sklearn',
        complexity = 2,
        scaled = True,
        max_rows = 10_000)

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'sklearn.svm.SVC',
        'regress': 'sklearn.svm.SVR'}
    parameters: base.GenericDict = dataclasses.field(
        default_factory = lambda: {'probability': True})


@dataclasses.dataclass
class Tabpfn(Model):
    """TabPFN, a pretrained model that is often the most accurate on small data.

    TabPFN is a neural network trained in advance on millions of synthetic
    tables, so it is not trained on your data in the usual way. It works best
    with up to about 10,000 rows and 500 features. It needs PyTorch (install
    it with "pip install amos[tabpfn]"), downloads its weights the first time
    it is used, and runs much faster with a GPU, which it uses if "use_gpu"
    is true. On a CPU, it is only practical for about 1,000 rows or fewer.
    Check the license of its weights before using it.

    Wraps:
        - [`tabpfn.TabPFNClassifier`](https://github.com/PriorLabs/TabPFN)
          from TabPFN, to classify.
        - [`tabpfn.TabPFNRegressor`](https://github.com/PriorLabs/TabPFN)
          from TabPFN, to regress.

    """

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'tabpfn.TabPFNClassifier',
        'regress': 'tabpfn.TabPFNRegressor'}

    def _device_parameters(self, *, gpu: bool) -> dict[str, Any] | None:
        """Returns the parameters that fit TabPFN on a GPU or the CPU.

        Args:
            gpu: whether to fit the model on the first GPU (or on the CPU).

        Returns:
            The "device": "cuda" or "cpu".

        """
        return {'device': 'cuda' if gpu else 'cpu'}


@dataclasses.dataclass
class WLS(_Statsmodels, Model):
    """Weighted least squares regression from statsmodels, with inference.

    Set "weights" to the column with the weight of each row (such as the
    inverse of its variance). The weights are not a feature.

    Wraps:
        [`statsmodels.regression.linear_model.WLS`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html)
        from statsmodels.

    """

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    column_parameters: ClassVar[tuple[str, ...]] = ('weights',)
    kind: ClassVar[str] = 'wls'


@dataclasses.dataclass
class Xgboost(Model):
    """XGBoost gradient boosting.

    With "use_gpu", it is fitted on the GPU.

    Wraps:
        - [`xgboost.XGBClassifier`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)
          from xgboost, to classify.
        - [`xgboost.XGBRegressor`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor)
          from xgboost, to regress.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'xgboost',
        complexity = 3,
        missing = True)

    tools: ClassVar[Mapping[str, Any]] = {
        'classify': 'xgboost.XGBClassifier',
        'regress': 'xgboost.XGBRegressor'}
    numbered_classes: ClassVar[bool] = True

    def _device_parameters(self, *, gpu: bool) -> dict[str, Any] | None:
        """Returns the parameters that fit xgboost on a GPU or the CPU.

        Args:
            gpu: whether to fit the model on the first GPU (or on the CPU).

        Returns:
            The "device": "cuda" or "cpu".

        """
        return {'device': 'cuda' if gpu else 'cpu'}


@dataclasses.dataclass
class ZeroInflatedPoisson(_Statsmodels, Model):
    """Zero-inflated Poisson regression of counts, with inference.

    Like `poisson`, for counts with more zeros than a Poisson distribution
    has (such as rows that could not have had any events).

    Wraps:
        [`statsmodels.discrete.count_model.ZeroInflatedPoisson`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.count_model.ZeroInflatedPoisson.html)
        from statsmodels.

    """

    traits: ClassVar[ModelTraits | None] = ModelTraits(
        package = 'statsmodels',
        complexity = 2,
        wide = False,
        counts = 'zero_inflated')

    tools: ClassVar[Mapping[str, Any]] = {'regress': Statsmodel}
    kind: ClassVar[str] = 'zero_inflated_poisson'
    counts: ClassVar[bool] = True


""" Private Functions """


def _family(name: str, item: base.Dataset, parameters: Mapping[str, Any]) -> str:
    """Returns the family of a `glm` or `gee`, checking that it suits the task.

    Args:
        name: name of the model, for the message of an error.
        item: the dataset to model.
        parameters: parameters for the model, which may name a "family".

    Raises:
        ValueError: if the label is classified and the family is not
            "binomial".

    Returns:
        The "family" in `parameters`, or else "binomial" for classification
            and "gaussian" for regression.

    """
    classify = item.task == 'classify'
    family = str(parameters.get(
        'family', 'binomial' if classify else 'gaussian'))
    if classify and family != 'binomial':
        message = (
            f'{name!r} with the {family!r} family regresses, but the label is '
            f'set to classify: use the "binomial" family, or set "task" to '
            f'"regress" in the "general" section'
        )
        raise ValueError(message)
    return family


def _with_groups(
    item: base.Dataset,
    parameters: dict[str, Any]) -> dict[str, Any]:
    """Returns `parameters` with the dataset's first group as "groups".

    Args:
        item: the dataset to model.
        parameters: parameters for the model.

    Returns:
        The parameters, with "groups" if they did not name it and the dataset
            has groups.

    """
    if parameters.get('groups') is None and item.groups:
        return {**parameters, 'groups': item.groups[0]}
    return parameters
