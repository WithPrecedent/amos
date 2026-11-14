"""Shared tools.

Contents:
    accepted_parameters: limits parameters to those that a tool accepts.
    add_special_setting: adds a section of settings that is not a worker's.
    class_names: returns the names of the classes that a node can be built
        with.
    describe_tool: returns a name for a tool that can be stored in a record.
    environment: describes the software used to run a project.
    feature_names: returns the columns of the data that a model was fitted
        to.
    flatten_dict: returns the values of a nested `dict` in one level.
    gpus: describes the NVIDIA GPUs that models can be fitted on.
    import_tool: imports a tool from its import path, with a helpful error.
    importable: returns whether a package can be imported.
    is_special_setting: returns whether a section of settings is not a
        worker's.
    label_path: returns the label of a path through a worker.
    listify_names: returns a `list` of names from a setting.
    parameters_of: returns the parameters of a built tool.
    preserved_logging: restores the root logger after code that changes it.
    quote_bare_values: puts quotes around the words in an expression that are
        not columns.
    search_space: divides parameters into fixed values and values to search.
    select_columns: returns the names of columns, checking that they are in
        the data.
    settings_file_extensions: returns the extensions of settings files.
    settings_file_type: returns the type of a settings file.
    technique_and_step: returns the technique that a step wraps and the
        step's name.

"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import importlib.metadata
import inspect
import keyword
import logging
import platform
import re
import sys
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import bobbie
import chrisjen
import chrisjen.options
import chrisjen.utilities
import chrisjen.workshop

from . import options

if TYPE_CHECKING:
    import pandas as pd

# Names of NVIDIA's CUDA driver library on Linux and on Windows.
_CUDA_DRIVERS: tuple[str, ...] = ('libcuda.so.1', 'libcuda.so', 'nvcuda.dll')
# Numbers that the CUDA driver uses for the major and minor parts of a GPU's
# compute capability (`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` and
# `_MINOR` in "cuda.h").
_COMPUTE_CAPABILITY: tuple[int, int] = (75, 76)
# The kinds of hyperparameter search that models support.
_SEARCHES: frozenset[str] = frozenset({'grid', 'optuna', 'random'})
# Text in quotes or backticks (which `quote_bare_values` leaves alone), or a
# word ("word") that is not after a letter, digit, ".", or "@" and is not
# followed by "(", ".", or a single "=" (so not part of a number, an
# attribute, a local variable, the name of a function, something with an
# attribute, or the target of an assignment).
_BARE_WORD: re.Pattern[str] = re.compile(
    r"'(?:[^'\\]|\\.)*'|" + r'"(?:[^"\\]|\\.)*"|`[^`]*`'
    r'|(?<![\w.@])(?P<word>[A-Za-z_]\w*)\b(?!\s*[(.])(?!\s*=(?!=))')
# The name of the package at the start of a requirement (such as
# "scikit-learn>=1.5") and the extra in its marker (such as
# "; extra == 'sampling'").
_REQUIREMENT_NAME: re.Pattern[str] = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
_REQUIREMENT_EXTRA: re.Pattern[str] = re.compile(r"extra == '([^']+)'")
# A column (a word or text in backticks) and "==" or "!=" at the end of text.
_COMPARED_COLUMN: re.Pattern[str] = re.compile(
    r'(?P<column>`[^`]*`|[A-Za-z_]\w*)\s*(?:==|!=)\s*$')


def accepted_parameters(
    tool: Callable[..., Any],
    parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Returns the `parameters` that `tool` accepts as keywords.

    A parameter set for a whole worker (in a "{worker}_parameters" section) is
    passed to every technique in that worker, so each tool only receives the
    parameters it accepts. Some tools, such as the models in `xgboost`, accept
    any keyword (`**kwargs`). For those, the names returned by the
    `get_params` method of a default instance (the scikit-learn convention)
    are used instead, if there is one. The names of a class are found once
    and kept, since building a default instance can be slow.

    Args:
        tool: callable (usually a class) to examine.
        parameters: keyword parameters to filter.

    Returns:
        A `dict` of the `parameters` that `tool` accepts.

    """
    try:
        signature = inspect.signature(tool).parameters
    except (TypeError, ValueError):
        return dict(parameters)
    takes_any = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.values())
    if takes_any and hasattr(tool, 'get_params'):
        names = (
            _default_parameter_names(tool) if isinstance(tool, type)
            else _built_parameter_names(tool))
        if names is None:
            # A tool that cannot be built without arguments falls back to its
            # signature, which accepts everything.
            return dict(parameters)
        return {k: v for k, v in parameters.items() if k in names}
    return chrisjen.utilities.accepted_arguments(tool, parameters)


def add_special_setting(name: str) -> None:
    """Adds a section of settings that `chrisjen` does not build as a worker.

    Args:
        name: name of the section (such as "feature_groups").

    """
    # chrisjen 0.4.2 and earlier have no `add_special_setting`.
    adder = getattr(chrisjen.options, 'add_special_setting', None)
    if adder is not None:
        adder(name)
    elif name not in chrisjen.options._SPECIAL_SETTINGS:
        chrisjen.options._SPECIAL_SETTINGS.append(name)


def class_names(name: str) -> str | list[str]:
    """Returns the names of the classes that a node can be built with.

    A name that ends in a number (such as "xgboost2") can be built with the
    class named without it, as `chrisjen` builds it.

    Args:
        name: name of a node in the settings.

    Returns:
        `name`, or `name` and `name` without its number.

    """
    builder: Callable[[str], str | list[str]] = _newer_or_older(
        chrisjen.workshop, 'class_names')
    return builder(name)


def describe_tool(tool: Any) -> str | None:
    """Returns a name for `tool` that can be stored in a record.

    Args:
        tool: an import path, a callable, an instance, or `None`.

    Returns:
        `tool` itself if it is a `str`, the module and qualified name of a
            callable or of the class of an instance, or `None` if `tool` is
            `None`.

    """
    if tool is None or isinstance(tool, str):
        return tool
    kind = tool if inspect.isclass(tool) or callable(tool) else type(tool)
    module = getattr(kind, '__module__', None)
    name = getattr(kind, '__qualname__', None) or repr(kind)
    return f'{module}.{name}' if module else name


def environment(*, gpu: bool = False) -> dict[str, Any]:
    """Describes the software (and any GPUs) used to run a project.

    Recording this alongside a project's results lets others reproduce them
    with the same versions of Python and of the packages that did the work.

    Args:
        gpu: whether to also describe the GPUs that models could be fitted on
            (see `gpus`). Defaults to `False`.

    Returns:
        A `dict` with the Python version, the platform, and the version of
            amos and of each installed package that it requires (including
            those of its extras), and, if `gpu` is `True`, "gpus".

    """
    packages = {}
    for package in ('amos', *dict.fromkeys(n for n, _ in _requirements())):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    described: dict[str, Any] = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': packages}
    if gpu:
        described['gpus'] = [dict(found) for found in gpus()]
    return described


def feature_names(model: Any, columns: Iterable[Any]) -> list[Any] | None:
    """Returns the columns of the data that a model was fitted to.

    A fitted scikit-learn model names its features in `feature_names_in_`,
    but some tools change names that they cannot store: lightgbm replaces
    the spaces in a name with underscores ("mean radius" becomes
    "mean_radius"). Those names are given back as the columns they came
    from, so that they can select the model's columns of the data.

    Args:
        model: a fitted model.
        columns: the columns of the data.

    Returns:
        The model's features, in its order, as names in `columns` (or as the
            model names them, if no column matches), or `None` if the model
            does not name its features.

    """
    names = getattr(model, 'feature_names_in_', None)
    if names is None:
        return None
    columns = list(columns)
    known = set(columns)
    renamed: dict[str, Any] = {}
    for column in columns:
        renamed.setdefault(str(column).replace(' ', '_'), column)
    return [
        name if name in known else renamed.get(str(name), name)
        for name in names]


def flatten_dict(item: Mapping[str, Any]) -> dict[str, Any]:
    """Returns the values of a nested `dict` that are not `dict`s, in one level.

    Args:
        item: nested `dict` to flatten, such as a genre of the library.

    Returns:
        The keys and values at every level whose values are not `dict`s.

    """
    flattener: Callable[[Any], dict[str, Any]] = _newer_or_older(
        chrisjen.utilities, 'flatten_dict')
    return flattener(item)


@functools.cache
def gpus() -> tuple[dict[str, Any], ...]:
    """Describes the NVIDIA GPUs that models can be fitted on.

    The GPUs are found by asking NVIDIA's CUDA driver directly, so no other
    package is needed. Like the packages that fit models, it leaves out any
    GPU hidden by the "CUDA_VISIBLE_DEVICES" environment variable and lists
    the others in the order that they number them. The answer is kept, so the
    driver is only asked once.

    Returns:
        For each GPU, its "name", its "compute_capability" (such as "6.1"),
            its "memory" in GiB, and the newest version of "cuda" that the
            driver supports. Empty if there is no NVIDIA driver or GPU.

    """
    driver = _cuda_driver()
    if driver is None or driver.cuInit(0) != 0:
        return ()
    count = ctypes.c_int()
    version = ctypes.c_int()
    if (
        driver.cuDeviceGetCount(ctypes.byref(count)) != 0
        or driver.cuDriverGetVersion(ctypes.byref(version)) != 0):
        return ()
    # The driver reports CUDA 12.4, for example, as 12040.
    cuda = f'{version.value // 1000}.{version.value % 1000 // 10}'
    found = []
    for number in range(count.value):
        device = ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        major, minor = ctypes.c_int(), ctypes.c_int()
        memory = ctypes.c_size_t()
        answers = (
            driver.cuDeviceGet(ctypes.byref(device), number),
            driver.cuDeviceGetName(name, len(name), device),
            driver.cuDeviceGetAttribute(
                ctypes.byref(major), _COMPUTE_CAPABILITY[0], device),
            driver.cuDeviceGetAttribute(
                ctypes.byref(minor), _COMPUTE_CAPABILITY[1], device),
            driver.cuDeviceTotalMem_v2(ctypes.byref(memory), device))
        if any(answer != 0 for answer in answers):
            continue
        found.append({
            'name': name.value.decode(errors = 'replace'),
            'compute_capability': f'{major.value}.{minor.value}',
            'memory': round(memory.value / 2**30, 1),
            'cuda': cuda})
    return tuple(found)


def import_tool(tool: str | Callable[..., Any]) -> Any:
    """Returns `tool`, importing it first if it is an import path.

    Args:
        tool: a callable, or the import path of one (such as
            "sklearn.preprocessing.MinMaxScaler").

    Raises:
        ImportError: if `tool` is an import path that cannot be imported. If
            the package is one that an `amos` extra installs, the message says
            which extra.

    Returns:
        The tool.

    """
    if not isinstance(tool, str):
        return tool
    try:
        return chrisjen.utilities.import_object(tool)
    except ImportError as error:
        package = tool.split('.')[0].split(':')[0]
        extra = _extra_of(package)
        if extra is None:
            raise
        message = (
            f'{error}. The {package!r} package is optional: install it with '
            f'"pip install amos[{extra}]" (or "pip install amos[all]")'
        )
        raise ImportError(message) from error


def importable(package: str) -> bool:
    """Returns whether `package` can be imported.

    Args:
        package: name of the package.

    Returns:
        Whether importing it succeeds. Some optional packages are installed
            but fail when they are imported (such as xgboost without
            OpenMP), so any error means that it cannot be.

    """
    try:
        importlib.import_module(package)
    except Exception:  # noqa: BLE001
        return False
    return True


def is_special_setting(name: str) -> bool:
    """Returns whether a section of settings is not a worker's.

    Args:
        name: name of the section.

    Returns:
        Whether `chrisjen` treats the section as special (such as "general",
            "files", or a section added with `add_special_setting`).

    """
    # chrisjen 0.4.2 and earlier have no `is_special_setting`.
    checker = getattr(chrisjen.options, 'is_special_setting', None)
    if checker is not None:
        return bool(checker(name))
    return name in chrisjen.options._SPECIAL_SETTINGS


def label_path(path: Sequence[Any]) -> str:
    """Returns the label of a path: the names of its nodes, joined by " > ".

    Args:
        path: nodes in a path through a worker.

    Returns:
        The label, as `chrisjen` labels the results of each path.

    """
    labeler: Callable[[Sequence[Any]], str] = _newer_or_older(
        chrisjen.utilities, 'label_path')
    return labeler(path)


def listify_names(item: Any) -> list[str]:
    """Returns a `list` of names from a setting.

    Args:
        item: a `str` of names separated by commas, a sequence of names, or
            `None`.

    Returns:
        The names, with whitespace removed and empty names dropped.

    """
    listifier: Callable[[Any], list[str]] = _newer_or_older(
        chrisjen.utilities, 'listify_names')
    return listifier(item)


def parameters_of(tool: Any) -> dict[str, Any]:
    """Returns the parameters of a built tool, if it reports them.

    Args:
        tool: a built tool, such as a fitted scikit-learn transformer.

    Returns:
        The result of `get_params(deep = False)` (the scikit-learn
            convention), or an empty `dict` if `tool` has no such method.

    """
    try:
        return dict(tool.get_params(deep = False))
    except (AttributeError, TypeError):
        return {}


@contextlib.contextmanager
def preserved_logging() -> Generator[None, None, None]:
    """Restores the root logger after code that changes it.

    Some packages (such as pyfixest) configure the root logger when they run,
    which would make every library print its informational messages. Code in
    this context leaves the root logger's level and handlers as they were.

    Yields:
        None: control, to the code in the context.

    """
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    try:
        yield
    finally:
        root.setLevel(level)
        for handler in list(root.handlers):
            if handler not in handlers:
                root.removeHandler(handler)
        # A package may also have removed handlers (with `force = True`).
        for handler in handlers:
            if handler not in root.handlers:
                root.addHandler(handler)


def quote_bare_values(expression: str, data: pd.DataFrame) -> str:
    """Puts quotes around the words in `expression` that are not columns.

    This lets a `pandas` expression compare a column to text without quotes,
    as in "offender_race == B", which becomes "offender_race == 'B'". A word
    is left as it is if it is a column of `data`, a Python keyword (such as
    "and", "in", or "True"), the name of a function (followed by "("), an
    attribute or method (after "."), something with an attribute (followed
    by "."), or a local variable (after "@"). Text in quotes or backticks is
    not changed.

    Args:
        expression: a `pandas` expression, as `DataFrame.eval` and
            `DataFrame.query` take.
        data: the data that the expression is about.

    Raises:
        ValueError: if a word after "==" or "!=" is a column, and the column
            before the operator has that word as one of its values, so it is
            not clear whether the word is a column or a value.

    Returns:
        `expression`, with quotes around the words that are values.

    """
    columns = {str(c) for c in data.columns}

    def quote(match: re.Match[str]) -> str:
        """Returns the text of `match`, in quotes if it is a bare value.

        Args:
            match: a match of `_BARE_WORD`.

        Raises:
            ValueError: if the word is both a column and a value of the
                column that it is compared to.

        Returns:
            The text, quoted if it is a word that is not a column or name.

        """
        word = match.group('word')
        if word is None or keyword.iskeyword(word):
            return match.group(0)
        if word not in columns:
            return repr(word)
        compared = _COMPARED_COLUMN.search(expression[:match.start()])
        if compared is not None:
            other = compared.group('column').strip('`')
            if other in columns and (
                data[other].astype(str) == word).to_numpy().any():
                message = (
                    f'{word!r} in {expression!r} is both a column and a '
                    f'value of {other!r}: put the value in quotes, or the '
                    f'column in backticks'
                )
                raise ValueError(message)
        return match.group(0)

    return _BARE_WORD.sub(quote, expression)


def search_space(
    parameters: Mapping[str, Any],
    search: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Divides `parameters` into fixed values and values to search.

    A parameter whose value is a `list` (as "n_estimators = 50, 200" is in an
    ini file) is searched. For a grid search, the values are the candidates.
    For a random or Optuna search, a list of exactly two numbers is a range:
    whole numbers are drawn from the range (inclusive) and other numbers from
    between them (on a log scale, in an Optuna search, if the range covers
    two or more orders of magnitude). Any other list is a set of candidates.

    Wraps:
        - [`scipy.stats.randint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html)
          from SciPy, for a random search of whole numbers.
        - [`scipy.stats.uniform`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html)
          from SciPy, for a random search of other numbers.
        - [`optuna.distributions.IntDistribution`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.distributions.IntDistribution.html)
          from Optuna, for an Optuna search of whole numbers.
        - [`optuna.distributions.FloatDistribution`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.distributions.FloatDistribution.html)
          from Optuna, for an Optuna search of other numbers.
        - [`optuna.distributions.CategoricalDistribution`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.distributions.CategoricalDistribution.html)
          from Optuna, for an Optuna search of candidates.

    Args:
        parameters: parameters for a tool.
        search: "grid", "random", or "optuna".

    Raises:
        ValueError: if `search` is not one of those.

    Returns:
        The fixed parameters and the parameters to search.

    """
    if search not in _SEARCHES:
        message = f'search must be one of {sorted(_SEARCHES)}, not {search!r}'
        raise ValueError(message)
    fixed: dict[str, Any] = {}
    space: dict[str, Any] = {}
    for name, value in parameters.items():
        if not isinstance(value, list):
            fixed[name] = value
        elif search == 'random' and _is_range(value):
            space[name] = _distribution(value[0], value[1])
        elif search == 'optuna':
            space[name] = _optuna_distribution(value)
        else:
            space[name] = value
    return fixed, space


def select_columns(
    data: pd.DataFrame,
    columns: Sequence[str] | str | None) -> list[str]:
    """Returns `columns` as a `list`, checking that they are in `data`.

    Args:
        data: the data.
        columns: a column name, a sequence of names, or `None` for every
            column.

    Raises:
        KeyError: if a column is not in `data`.

    Returns:
        The names of the columns.

    """
    if columns is None:
        return list(data.columns)
    names = [columns] if isinstance(columns, str) else list(columns)
    missing = [c for c in names if c not in data.columns]
    if missing:
        message = f'the columns {missing} are not in the data'
        raise KeyError(message)
    return names



def settings_file_extensions() -> list[str]:
    """Returns the extensions of the settings files that `bobbie` supports.

    Returns:
        The extensions, without a leading ".".

    """
    # bobbie 0.4.0 and earlier have no `file_extensions`.
    lister = getattr(bobbie.options, 'file_extensions', None)
    if lister is not None:
        return list(lister())
    return list(bobbie.options._FILE_EXTENSIONS)


def settings_file_type(extension: str) -> str | None:
    """Returns the type of a settings file that `bobbie` reads and writes.

    Args:
        extension: the file's extension, with or without a leading ".", in
            any case.

    Returns:
        The file type (such as "json" or "yaml"), or `None` if `bobbie` does
            not support the extension.

    """
    # bobbie 0.4.0 and earlier have no `file_type`.
    finder = getattr(bobbie.options, 'file_type', None)
    if finder is not None:
        found = finder(extension)
        return None if found is None else str(found)
    extensions: Mapping[str, str] = bobbie.options._FILE_EXTENSIONS
    return extensions.get(extension.lower().removeprefix('.'))


def technique_and_step(node: Any) -> tuple[str, str] | None:
    """Returns the technique that a step wraps and the step's own name.

    `chrisjen` names a step that wraps a technique "{technique}_{step}"
    (such as "logit_model").

    Args:
        node: a node in a workflow.

    Returns:
        The name of the technique and of the step (such as "logit" and
            "model"), or `None` if `node` is not a step that wraps a node.

    """
    contents = getattr(node, 'contents', None)
    if not isinstance(node, chrisjen.Step) or contents is None:
        return None
    technique = str(contents.name)
    return technique, str(node.name).removeprefix(f'{technique}_')

""" Private Functions """


def _extra_of(package: str) -> str | None:
    """Returns the optional `amos` extra that installs a package.

    Args:
        package: the name that the package is imported by (such as
            "imblearn").

    Returns:
        The extra (such as "sampling"), or `None` if no extra installs it.
            `options._EXTRAS` is checked first, and then the extras in
            amos's requirements (other than "all").

    """
    if package in options._EXTRAS:
        return options._EXTRAS[package]
    for name, extra in _requirements():
        imported = options._IMPORT_NAMES.get(name, name.replace('-', '_'))
        if extra not in {None, 'all'} and imported == package:
            return extra
    return None


@functools.cache
def _requirements() -> tuple[tuple[str, str | None], ...]:
    """Returns the packages that amos requires, and the extra of each.

    Returns:
        The name of each package in amos's requirements (as installed, such
            as "scikit-learn"), in order, with the extra that installs it,
            or `None` for one that is always installed. Empty if amos's
            metadata cannot be found (as when it is not installed).

    """
    try:
        lines = importlib.metadata.requires('amos') or []
    except importlib.metadata.PackageNotFoundError:
        return ()
    found = []
    for line in lines:
        name = _REQUIREMENT_NAME.match(line)
        extra = _REQUIREMENT_EXTRA.search(line)
        if name is not None:
            found.append((
                name.group(0).lower(),
                None if extra is None else extra.group(1)))
    return tuple(found)


def _built_parameter_names(tool: Callable[..., Any]) -> frozenset[str] | None:
    """Returns the names of the parameters of a default instance of `tool`.

    Args:
        tool: a callable with a `get_params` method (the scikit-learn
            convention).

    Returns:
        The names that `get_params` returns for `tool()`, or `None` if `tool`
            cannot be built without arguments.

    """
    try:
        return frozenset(tool().get_params())
    except Exception:  # noqa: BLE001
        return None


@functools.cache
def _default_parameter_names(tool: type) -> frozenset[str] | None:
    """Returns the names of the parameters of a class, finding them once.

    Args:
        tool: a class with a `get_params` method.

    Returns:
        The names (see `_built_parameter_names`), or `None` if the class
            cannot be built without arguments.

    """
    return _built_parameter_names(tool)


def _cuda_driver() -> Any:
    """Returns NVIDIA's CUDA driver library, if it is installed.

    Returns:
        The library, loaded with `ctypes`, or `None` if it cannot be loaded
            (as on a computer without an NVIDIA GPU, or on macOS).

    """
    for name in _CUDA_DRIVERS:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _distribution(low: float, high: float) -> Any:
    """Returns a `scipy` distribution between `low` and `high`.

    Wraps:
        - [`scipy.stats.randint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html)
          from SciPy, for whole numbers.
        - [`scipy.stats.uniform`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html)
          from SciPy, for other numbers.

    Args:
        low: lowest value.
        high: highest value.

    Returns:
        A uniform integer distribution (inclusive of `high`) if both values
            are integers, and otherwise a uniform continuous distribution.

    """
    stats = importlib.import_module('scipy.stats')
    low, high = min(low, high), max(low, high)
    if isinstance(low, int) and isinstance(high, int):
        return stats.randint(low, high + 1)
    return stats.uniform(low, high - low)


def _optuna_distribution(value: list[Any]) -> Any:
    """Returns an Optuna distribution for the values of a parameter.

    Wraps:
        - [`optuna.distributions.IntDistribution`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.distributions.IntDistribution.html)
          from Optuna, for a range of whole numbers.
        - [`optuna.distributions.FloatDistribution`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.distributions.FloatDistribution.html)
          from Optuna, for a range of other numbers.
        - [`optuna.distributions.CategoricalDistribution`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.distributions.CategoricalDistribution.html)
          from Optuna, for candidates.

    Args:
        value: a list of two numbers (a range) or of candidates.

    Returns:
        An integer or float distribution for a range, and otherwise a
            categorical distribution of the candidates.

    """
    distributions = import_tool('optuna.distributions')
    if not _is_range(value):
        return distributions.CategoricalDistribution(value)
    low, high = min(value), max(value)
    if isinstance(low, int) and isinstance(high, int):
        return distributions.IntDistribution(low, high)
    # A range that spans two or more orders of magnitude (such as a learning
    # rate from 0.001 to 0.1) is searched on a log scale.
    log = low > 0 and high / low >= 100  # noqa: PLR2004
    return distributions.FloatDistribution(low, high, log = log)


def _is_range(value: list[Any]) -> bool:
    """Returns whether `value` is a list of two numbers (but not booleans).

    Args:
        value: list to check.

    Returns:
        Whether `value` is a list of exactly two numbers.

    """
    return len(value) == 2 and all(  # noqa: PLR2004
        isinstance(v, int | float) and not isinstance(v, bool) for v in value)


def _newer_or_older(module: Any, name: str) -> Any:
    """Returns a helper of `chrisjen` by its public name or its older one.

    `chrisjen` made some helpers public after 0.4.2 (such as
    `utilities.listify_names`, which was `_listify_names`). This finds
    either, so that amos works with both.

    Args:
        module: the module of `chrisjen` that has the helper.
        name: the public name of the helper.

    Returns:
        The helper.

    """
    found = getattr(module, name, None)
    return found if found is not None else getattr(module, f'_{name}')
