"""Automated Modeling for Open Scholarship"""

from __future__ import annotations

__version__ = '0.2.2'

__author__: str = 'Corey Rayburn Yung'

__all__: list[str] = [
    'Cleaner',
    'Dataset',
    'Describer',
    'Effect',
    'Encoder',
    'Evaluator',
    'Experiment',
    'Findings',
    'GroupMetric',
    'Imputer',
    'Loader',
    'Metric',
    'Mixer',
    'Model',
    'Operation',
    'Plot',
    'Project',
    'Reducer',
    'Sampler',
    'Scaler',
    'Splitter',
    'Transformer',
    'Validator',
    'library',
]

# Importing each module adds its techniques to the `chrisjen` library, so
# they can be named in settings as soon as `amos` is imported.
from chrisjen import library

from .base import Dataset, Operation
from .cleaners import Cleaner
from .describers import Describer
from .effects import Effect
from .evaluators import Evaluator
from .interface import Project
from .loaders import Loader
from .metrics import GroupMetric, Metric
from .models import Model
from .plots import Plot
from .reports import Findings
from .samplers import Sampler
from .splitters import Splitter
from .transformers import Encoder, Imputer, Mixer, Reducer, Scaler, Transformer
from .validators import Validator
from .workers import Experiment
