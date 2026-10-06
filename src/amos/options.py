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
    _EXTRAS: optional `amos` extras that install each optional package.
    _RECORDED_PACKAGES: packages whose versions are recorded by
        `Project.export`.

"""

from __future__ import annotations

import pathlib

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
# The optional `amos` extra that installs each optional package. Used to give
# a helpful message when a technique's package is missing.
_EXTRAS: dict[str, str] = {
    'category_encoders': 'encoders',
    'docx': 'word',
    'imblearn': 'sampling',
    'lightgbm': 'boosting',
    'matplotlib': 'plots',
    'seaborn': 'plots',
    'shap': 'explain',
    'statsmodels': 'statistics',
    'xgboost': 'boosting'}
# Packages whose versions are written to "environment.json" by
# `Project.export`, if they are installed.
_RECORDED_PACKAGES: tuple[str, ...] = (
    'amos',
    'chrisjen',
    'numpy',
    'pandas',
    'scikit-learn',
    'scipy',
    'category-encoders',
    'imbalanced-learn',
    'lightgbm',
    'matplotlib',
    'python-docx',
    'seaborn',
    'shap',
    'statsmodels',
    'xgboost')
