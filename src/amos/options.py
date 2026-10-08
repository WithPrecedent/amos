"""Default settings for amos.

These are module-level constants so that a package or project built on `amos`
can change them before a workflow is built (for example,
`amos.options._DEFAULT_TEST_SIZE = 0.2`).

Contents:
    _CLASSIFY_THRESHOLD: most unique values that an integer label can have
        and still be treated as classes.
    _CATEGORY_THRESHOLD: most unique values that a column can have for
        `auto_categorize` to make it categorical.
    _DEFAULT_REPORT: name of the report that a `Project` generates.
    _DEFAULT_ROOT: root folder for a project's files.
    _DEFAULT_TEST_SIZE: share of the rows that a splitter puts in the test set.
    _DOWNLOAD_TIMEOUT: seconds that `download` waits for a server to answer.
    _EXTRAS: optional `amos` extras that install each optional package.
    _FILE_SETTINGS: values that loaders use in place of some of the shared
        settings of a `nagata` clerk.
    _RECORDED_PACKAGES: packages whose versions are recorded by
        `Project.export`.

"""

from __future__ import annotations

import pathlib
from typing import Any

# An integer label with this many unique values (or fewer) is treated as a set
# of classes, so `Dataset.task` is "classify". Above it, "regress".
_CLASSIFY_THRESHOLD: int = 10
# `auto_categorize` makes a column categorical if it has this many unique
# values (or fewer).
_CATEGORY_THRESHOLD: int = 10
# Name (in the library) of the report that an `amos.Project` generates.
_DEFAULT_REPORT: str = 'findings'
# Root folder for a project's files. Unlike `chrisjen`, which uses a folder
# beside the current one, `amos` keeps a project's files in the current
# folder unless the "files" section of the settings says otherwise.
_DEFAULT_ROOT: pathlib.Path = pathlib.Path()
# Share of the rows that a splitter puts in the test set.
_DEFAULT_TEST_SIZE: float = 0.25
# Seconds that `download` waits for a server to answer before giving up.
_DOWNLOAD_TIMEOUT: float = 60
# The optional `amos` extra that installs each optional package. Used to give
# a helpful message when a technique's package is missing.
_EXTRAS: dict[str, str] = {
    'catboost': 'boosting',
    'category_encoders': 'encoders',
    'docx': 'word',
    'doubleml': 'causal',
    'eli5': 'explain',
    'fairlearn': 'fairness',
    'great_tables': 'tables',
    'imblearn': 'sampling',
    'interpret': 'explain',
    'lightgbm': 'boosting',
    'mapie': 'uncertainty',
    'matplotlib': 'plots',
    'optuna': 'tuning',
    'optuna_integration': 'tuning',
    'polars': 'polars',
    'pyfixest': 'statistics',
    'seaborn': 'plots',
    'shap': 'explain',
    'skrub': 'encoders',
    'statsmodels': 'statistics',
    'tabpfn': 'tabpfn',
    'xgboost': 'boosting'}
# Values that loaders pass in place of the shared settings of a `nagata`
# clerk (`FileFramework.settings`), by the name of the setting. The clerk's own
# values suit quick tests (it reads only the first 1000 rows of a csv file,
# for example), so loaders use the defaults of `pandas` instead.
_FILE_SETTINGS: dict[str, Any] = {
    'file_encoding': 'utf-8',
    'index_column': None,
    'test_size': None}
# Packages whose versions are written to "environment.json" by
# `Project.export`, if they are installed.
_RECORDED_PACKAGES: tuple[str, ...] = (
    'amos',
    'chrisjen',
    'numpy',
    'pandas',
    'scikit-learn',
    'scipy',
    'catboost',
    'category-encoders',
    'doubleml',
    'eli5',
    'fairlearn',
    'great-tables',
    'imbalanced-learn',
    'interpret-core',
    'lightgbm',
    'mapie',
    'matplotlib',
    'optuna',
    'polars',
    'pyfixest',
    'python-docx',
    'seaborn',
    'shap',
    'skrub',
    'statsmodels',
    'tabpfn',
    'xgboost')
