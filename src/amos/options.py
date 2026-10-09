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
    _FIGURE_DPI: dots per inch of the figures that `Project.export` saves.
    _FILE_SETTINGS: values that loaders use in place of some of the shared
        settings of a `nagata` clerk.
    _PLOT_COLORS: the color cycle of figures.
    _PLOT_STYLE: the style of figures.
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
    'causalml': 'causal',
    'docx': 'word',
    'doubleml': 'causal',
    'dowhy': 'causal',
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
    'rdrobust': 'causal',
    'scienceplots': 'plots',
    'seaborn': 'plots',
    'shap': 'explain',
    'skrub': 'encoders',
    'statsmodels': 'statistics',
    'tabpfn': 'tabpfn',
    'tigramite': 'causal',
    'xgboost': 'boosting'}
# Dots per inch of the figures that `Project.export` saves. Figures in the
# default style are one column of a journal wide (3.3 inches), so they need
# more dots than the 100 that `matplotlib` uses to be sharp in print.
_FIGURE_DPI: int = 300
# Values that loaders pass in place of the shared settings of a `nagata`
# clerk (`FileFramework.settings`), by the name of the setting. The clerk's own
# values suit quick tests (it reads only the first 1000 rows of a csv file,
# for example), so loaders use the defaults of `pandas` instead.
_FILE_SETTINGS: dict[str, Any] = {
    'file_encoding': 'utf-8',
    'index_column': None,
    'test_size': None}
# The color cycle of figures: the name of a style that only sets the colors,
# such as SciencePlots' "bright" (Paul Tol's palette, which people with color
# blindness can tell apart), or `None` for the colors of the style.
_PLOT_COLORS: str | None = 'bright'
# The style of figures: names of `matplotlib` or SciencePlots styles, applied
# in order, or "xkcd". The default is SciencePlots' style for scientific
# figures, with the sizes and fonts of figures in Nature.
_PLOT_STYLE: tuple[str, ...] = ('science', 'nature')
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
    'causalml',
    'doubleml',
    'dowhy',
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
    'rdrobust',
    'scienceplots',
    'seaborn',
    'shap',
    'skrub',
    'statsmodels',
    'tabpfn',
    'tigramite',
    'xgboost')
