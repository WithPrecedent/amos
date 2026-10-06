"""Shared tools.

Contents:
    accepted_parameters: limits parameters to those that a tool accepts.
    describe_tool: returns a name for a tool that can be stored in a record.
    environment: describes the software used to run a project.
    import_tool: imports a tool from its import path, with a helpful error.
    parameters_of: returns the parameters of a built tool.
    search_space: divides parameters into fixed values and values to search.

"""

from __future__ import annotations

import importlib.metadata
import inspect
import platform
import sys
from collections.abc import Callable, Mapping
from typing import Any

import chrisjen.utilities

from . import options


def accepted_parameters(
    tool: Callable[..., Any],
    parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Returns the `parameters` that `tool` accepts as keywords.

    A parameter set for a whole worker (in a "{worker}_parameters" section) is
    passed to every technique in that worker, so each tool only receives the
    parameters it accepts. Some tools, such as the models in `xgboost`, accept
    any keyword (`**kwargs`). For those, the names returned by the
    `get_params` method of a default instance (the scikit-learn convention)
    are used instead, if there is one.

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
        try:
            names = set(tool().get_params())
        except Exception:  # noqa: BLE001
            # A tool that cannot be built without arguments falls back to its
            # signature, which accepts everything.
            return dict(parameters)
        return {k: v for k, v in parameters.items() if k in names}
    return chrisjen.utilities.accepted_arguments(tool, parameters)


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


def environment() -> dict[str, Any]:
    """Describes the software used to run a project.

    Recording this alongside a project's results lets others reproduce them
    with the same versions of Python and of the packages that did the work.

    Returns:
        A `dict` with the Python version, the platform, and the version of each
            installed package in `options._RECORDED_PACKAGES`.

    """
    packages = {}
    for package in options._RECORDED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': packages}


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
        extra = options._EXTRAS.get(package)
        if extra is None:
            raise
        message = (
            f'{error}. The {package!r} package is optional: install it with '
            f'"pip install amos[{extra}]" (or "pip install amos[all]")'
        )
        raise ImportError(message) from error


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


def search_space(
    parameters: Mapping[str, Any],
    search: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Divides `parameters` into fixed values and values to search.

    A parameter whose value is a `list` (as "n_estimators = 50, 200" is in an
    ini file) is searched. For a grid search, the values are the candidates.
    For a random search, a list of exactly two numbers is a range: integers
    are drawn uniformly from the range (inclusive) and other numbers are drawn
    from a uniform distribution between them. Any other list is a set of
    candidates.

    Args:
        parameters: parameters for a tool.
        search: "grid" or "random".

    Raises:
        ValueError: if `search` is not "grid" or "random".

    Returns:
        The fixed parameters and the parameters to search.

    """
    if search not in {'grid', 'random'}:
        message = f'search must be "grid" or "random", not {search!r}'
        raise ValueError(message)
    fixed: dict[str, Any] = {}
    space: dict[str, Any] = {}
    for name, value in parameters.items():
        if not isinstance(value, list):
            fixed[name] = value
        elif search == 'random' and _is_range(value):
            space[name] = _distribution(value[0], value[1])
        else:
            space[name] = value
    return fixed, space


""" Private Functions """


def _distribution(low: float, high: float) -> Any:
    """Returns a `scipy` distribution between `low` and `high`.

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


def _is_range(value: list[Any]) -> bool:
    """Returns whether `value` is a list of two numbers (but not booleans).

    Args:
        value: list to check.

    Returns:
        Whether `value` is a list of exactly two numbers.

    """
    return len(value) == 2 and all(  # noqa: PLR2004
        isinstance(v, int | float) and not isinstance(v, bool) for v in value)
