"""Techniques that draw figures of the data and the model.

These are the techniques of the "artist" stage. Each one draws a `matplotlib`
figure and adds it to the dataset's `figures` (under the technique's name).
Figures are made without `matplotlib.pyplot`, so they do not open windows or
depend on a display. (shap draws its plots with `matplotlib.pyplot`, so the
SHAP plots lend it their figure while shap draws, and take it back after.)
mlxtend's and dtreeviz's plots draw on the figure's axes.
`Project.export` saves them as image files, and in a notebook, a figure is
shown by making it the last line of a cell.

Every plot accepts "width" and "height" (in inches) and "title" parameters,
and "style", "colors", "latex", and "start_at_zero" parameters that set how
it looks (see `Plot.make_figure`). The axes start at 0 unless they show
something with no natural zero (such as the values of a feature, the label
of a regression, or the parameters of a model), and the panels of a plot
that measure the same thing share their ranges. By default, figures have
SciencePlots' style for scientific figures, with the size and fonts of a
figure in Nature, and a cycle of colors that people with color blindness can
tell apart. Explorers and critics draw their tables with these plots, in the
same style, when they are given `visualize` (see `base.Visualizer`).

Contents:
    Plot: base class for techniques that draw a figure.
    ActualVsPredicted: the label against the predictions (regression).
    AssociationRulesPlot: the support and confidence of each association
        rule, colored by lift.
    BiasVariancePlot: the model's expected loss, bias, and variance.
    BoxPlots: each numeric feature as a box, split by class.
    CalibrationCurve: the share in each class against its probability.
    CausalGraph: the causal links that tigramite found, as a graph.
    ClassificationReportHeatmap: the precision, recall, and f1 of each class.
    CoefficientPlot: coefficients with their confidence intervals.
    ConfusionHeatmap: the confusion matrix as a heatmap (classification).
    CorrelationHeatmap: the correlations between numeric columns.
    CountPlots: the count of each value of each categorical feature.
    DecisionRegions: the class predicted across two features
        (classification).
    DetCurve: the detection error tradeoff curve (classification).
    DistributionFits: each column's histogram with a fitted distribution
        laid over it.
    DistributionPlots: the distribution of every column.
    EstimatesPlot: every model's estimates in a regression table.
    FairnessPlot: the model's fairness metrics for each group.
    FrequencyPlots: the share of the rows with each value of each column.
    Histograms: the distribution of each numeric feature.
    ImportancePlot: the most important features, as bars.
    InfluencePlot: the influence of each row on a statsmodels regression.
    LabelPlot: the distribution of the label.
    LearningCurve: the model's score as it learns from more rows.
    LoadingsHeatmap: how strongly each feature loads on each factor.
    MissingBars: the share of each column's values that are missing.
    MissingHeatmap: where values are missing.
    PairPlot: each pair of numeric features against each other.
    PartialDependence: the average prediction as features change.
    PrecisionRecallCurve: precision against recall (classification).
    PredictionIntervals: conformal prediction intervals (or set sizes).
    QqPlot: the quantiles of the residuals against a normal distribution's.
    ResidualPlot: the errors against the predictions (regression).
    RocCurve: the ROC curve (classification).
    ScorecardPlot: each branch's score on each metric of a scorecard.
    ScreePlot: the share of the variance of each principal component.
    SearchPlot: the score of each try in a hyperparameter search.
    ShapBar: the mean absolute SHAP value of the most important features.
    ShapBeeswarm: the SHAP value of each row for the most important features.
    ShapDecision: how the features move each row's prediction, as lines.
    ShapEmbedding: the rows placed by their SHAP values.
    ShapForce: how each feature pushes one row's prediction up or down.
    ShapGroupDifference: how the SHAP values differ between two groups.
    ShapHeatmap: the SHAP values of every explained row, as a heatmap.
    ShapPartialDependence: the average prediction as a feature changes.
    ShapScatter: a feature's SHAP values against its values.
    ShapViolin: the distribution of the SHAP values of each feature.
    ShapWaterfall: how each feature moves one row's prediction.
    ShapeFunctions: an explainable boosting model's shape functions.
    SurvivalCurves: the share of rows without an event over time.
    TreeFeatureSpace: how a decision tree divides one or two features.
    TreeLeafDistributions: the label of the training rows in each leaf.
    TreeLeafPurity: the impurity of each leaf of a decision tree.
    TreeLeafSizes: the number of training rows in each leaf.
    TreePlot: the splits of a decision tree.
    TreePredictionPath: how much each feature decides one row's path down a
        decision tree.
    ValidationCurve: the model's score as one of its parameters changes.
    WeightsPlot: the weights of the features that eli5 explains.

"""

from __future__ import annotations

__all__: list[str] = [
    'ActualVsPredicted',
    'AssociationRulesPlot',
    'BiasVariancePlot',
    'BoxPlots',
    'CalibrationCurve',
    'CausalGraph',
    'ClassificationReportHeatmap',
    'CoefficientPlot',
    'ConfusionHeatmap',
    'CorrelationHeatmap',
    'CountPlots',
    'DecisionRegions',
    'DetCurve',
    'DistributionFits',
    'DistributionPlots',
    'EstimatesPlot',
    'FairnessPlot',
    'FrequencyPlots',
    'Histograms',
    'ImportancePlot',
    'InfluencePlot',
    'LabelPlot',
    'LearningCurve',
    'LoadingsHeatmap',
    'MissingBars',
    'MissingHeatmap',
    'PairPlot',
    'PartialDependence',
    'Plot',
    'PrecisionRecallCurve',
    'PredictionIntervals',
    'QqPlot',
    'ResidualPlot',
    'RocCurve',
    'ScorecardPlot',
    'ScreePlot',
    'SearchPlot',
    'ShapBar',
    'ShapBeeswarm',
    'ShapDecision',
    'ShapEmbedding',
    'ShapForce',
    'ShapGroupDifference',
    'ShapHeatmap',
    'ShapPartialDependence',
    'ShapScatter',
    'ShapViolin',
    'ShapWaterfall',
    'ShapeFunctions',
    'SurvivalCurves',
    'TreeFeatureSpace',
    'TreeLeafDistributions',
    'TreeLeafPurity',
    'TreeLeafSizes',
    'TreePlot',
    'TreePredictionPath',
    'ValidationCurve',
    'WeightsPlot',
]

from ._base import _IMPORTANCE_LABELS as _IMPORTANCE_LABELS
from ._base import Plot
from ._base import _lent as _lent
from ._base import _start_at_zero as _start_at_zero
from ._base import _unwrapped as _unwrapped
from ._data import (
    AssociationRulesPlot,
    BoxPlots,
    CorrelationHeatmap,
    CountPlots,
    DistributionFits,
    DistributionPlots,
    FrequencyPlots,
    Histograms,
    LabelPlot,
    MissingBars,
    MissingHeatmap,
    PairPlot,
    SurvivalCurves,
)
from ._model import _INFLUENCE as _INFLUENCE
from ._model import _INTERVAL_COLUMNS as _INTERVAL_COLUMNS
from ._model import (
    ActualVsPredicted,
    BiasVariancePlot,
    CalibrationCurve,
    CausalGraph,
    ClassificationReportHeatmap,
    CoefficientPlot,
    ConfusionHeatmap,
    DecisionRegions,
    DetCurve,
    EstimatesPlot,
    FairnessPlot,
    ImportancePlot,
    InfluencePlot,
    LearningCurve,
    LoadingsHeatmap,
    PartialDependence,
    PrecisionRecallCurve,
    PredictionIntervals,
    QqPlot,
    ResidualPlot,
    RocCurve,
    ScorecardPlot,
    ScreePlot,
    SearchPlot,
    ShapeFunctions,
    ValidationCurve,
    WeightsPlot,
)
from ._shap import (
    ShapBar,
    ShapBeeswarm,
    ShapDecision,
    ShapEmbedding,
    ShapForce,
    ShapGroupDifference,
    ShapHeatmap,
    ShapPartialDependence,
    ShapScatter,
    ShapViolin,
    ShapWaterfall,
)
from ._trees import (
    TreeFeatureSpace,
    TreeLeafDistributions,
    TreeLeafPurity,
    TreeLeafSizes,
    TreePlot,
    TreePredictionPath,
)

# The public names say that they are in this package (rather than in
# the private module that defines them), where they are imported
# from, so that records, pickles, and the documentation name them as
# `amos.plots.Name`.
for _name in __all__:
    globals()[_name].__module__ = __name__
