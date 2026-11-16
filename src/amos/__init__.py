"""Automated Modeling for Open Scholarship"""

from __future__ import annotations

__version__ = '0.8.1'

__author__: str = 'Corey Rayburn Yung'

__all__: list[str] = [
    'AmosWarning',
    'Chooser',
    'Cleaner',
    'Clerk',
    'Dataset',
    'Describer',
    'Encoder',
    'Evaluator',
    'Experiment',
    'Findings',
    'GroupMetric',
    'Imputer',
    'IncompatibleTechniqueError',
    'Inference',
    'Inferer',
    'Loader',
    'Merger',
    'Metric',
    'Mixer',
    'Model',
    'Munger',
    'Operation',
    'Parser',
    'PatternParser',
    'Pipeline',
    'Plot',
    'Project',
    'Reducer',
    'Sampler',
    'Scaler',
    'Selector',
    'Shaper',
    'Splitter',
    'Transformer',
    'Validator',
    'library',
    'register_rule_kind',
    'register_rulebook',
    'register_value_map',
]

# Importing each module adds its techniques to the `chrisjen` library, so
# they can be named in settings as soon as `amos` is imported.
import sys

from chrisjen import library

from .analyst import _transformers as transformers
from .analyst import (
    encoders,
    imputers,
    mixers,
    models,
    reducers,
    samplers,
    scalers,
    selectors,
    splitters,
    validators,
)
from .analyst._transformers import Transformer
from .analyst.encoders import Encoder
from .analyst.imputers import Imputer
from .analyst.mixers import Mixer
from .analyst.models import Model
from .analyst.reducers import Reducer
from .analyst.samplers import Sampler
from .analyst.scalers import Scaler
from .analyst.selectors import Selector
from .analyst.splitters import Splitter
from .analyst.validators import Validator
from .artist import plots
from .artist.plots import Plot
from .core import (
    auto,
    base,
    clerks,
    interface,
    metrics,
    options,
    pipelines,
    reports,
    utilities,
    workers,
)
from .core.auto import Chooser
from .core.base import (
    AmosWarning,
    Dataset,
    IncompatibleTechniqueError,
    Operation,
)
from .core.clerks import Clerk
from .core.interface import Project
from .core.metrics import GroupMetric, Metric
from .core.pipelines import Pipeline
from .core.reports import Findings
from .core.workers import Experiment
from .critic import evaluators, inferers
from .critic.evaluators import Evaluator
from .critic.inferers import Inference, Inferer
from .explorer import describers
from .explorer.describers import Describer
from .wrangler import cleaners, loaders, mergers, mungers, parsers, shapers
from .wrangler.cleaners import Cleaner
from .wrangler.loaders import Loader
from .wrangler.mergers import Merger
from .wrangler.mungers import Munger, register_value_map
from .wrangler.parsers import (
    Parser,
    PatternParser,
    register_rule_kind,
    register_rulebook,
)
from .wrangler.shapers import Shaper

# Each step and each module of the core can be imported from `amos` itself
# (such as `amos.cleaners` or `amos.base`), wherever it is in the folders of
# its stage. The techniques of each step are named by that path (see the
# `__init__` of each step), so pickles and records find them there.
for _module in (
        auto, base, clerks, interface, metrics, options, pipelines, reports,
        utilities, workers, cleaners, loaders, mergers, mungers, parsers,
        shapers, describers, encoders, imputers, mixers, models, reducers,
        samplers, scalers, selectors, splitters, transformers, validators,
        evaluators, inferers, plots):
    _short = _module.__name__.rpartition('.')[2].removeprefix('_')
    sys.modules.setdefault(f'{__name__}.{_short}', _module)
