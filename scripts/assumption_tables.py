"""Writes the tables of the assumptions page in `docs/assumptions.md`.

The page lists every assumption and need that amos knows, and then, for
each genre of techniques in which some technique has assumptions or needs,
every technique of the genre with its assumptions and needs. The
introduction of the page (everything before its first section) is written
by hand and kept as it is. Run the script again whenever a technique's
assumptions or needs change:

    uv run python docs/scripts/assumption_tables.py

"""

from __future__ import annotations

import importlib.util
import inspect
import io
import pathlib
import types
from typing import Any

import amos

ROOT = pathlib.Path(__file__).parent.parent.parent
PAGE = ROOT / 'docs' / 'assumptions.md'
# The heading of the first section of the page, after its introduction.
FIRST = '## Every assumption'
# The modules whose conditions the page lists (not those that a user or a
# test adds).
MODULES: tuple[str, ...] = ('amos.core.assumptions', 'amos.core.needs')
# What the "Tests Used" column says of an assumption that cannot be tested.
UNTESTED = 'none: it cannot be tested from the data'


def _catalog() -> types.ModuleType:
    """Returns the script that writes the technique catalog.

    Its genres and helpers are shared, so the two pages list techniques in
    the same order and link them the same way.

    Returns:
        The `technique_catalog` module.

    """
    path = pathlib.Path(__file__).with_name('technique_catalog.py')
    spec = importlib.util.spec_from_file_location('technique_catalog', path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CATALOG = _catalog()


def main() -> None:
    """Writes the tables of the page after its introduction."""
    text = PAGE.read_text(encoding = 'utf-8')
    introduction = text[:text.index(FIRST)]
    PAGE.write_text(introduction + tables(), encoding = 'utf-8')
    print(f'wrote {PAGE.relative_to(ROOT)}')  # noqa: T201


def tables() -> str:
    """Returns the tables of the page.

    Returns:
        The Markdown of the sections, ending with a new line.

    """
    known = {
        name: kind for name, kind in amos.conditions.catalog().items()
        if kind.__module__ in MODULES}
    out = io.StringIO()
    out.write(f'{FIRST}\n\n')
    out.write(
        '| Name | The technique assumes that | Pretest | Audit | '
        'Tests Used |\n')
    out.write('| --- | --- | --- | --- | --- |\n')
    for name, kind in known.items():
        if kind.kind != 'assumption':
            continue
        tests = kind.tests_used or UNTESTED
        out.write(
            f'| {_condition(name, kind)} | {kind.description} | '
            f'{_check(kind.pretested_by)} | {_check(kind.audited_by)} | '
            f'{tests} |\n')
    out.write('\n## Every need\n\n')
    out.write('| Name | The technique needs | Pretest | Hard |\n')
    out.write('| --- | --- | --- | --- |\n')
    for name, kind in known.items():
        if kind.kind != 'need':
            continue
        hard = 'yes' if kind.hard else 'no'
        out.write(
            f'| {_condition(name, kind)} | {kind.description} | '
            f'{_check(kind.pretested_by)} | {hard} |\n')
    for genre, title in CATALOG.GENRES:
        if genre == 'metric':
            continue
        kinds = CATALOG._techniques(genre)
        if not any(_conditions(kind) for _, kind in kinds):
            continue
        out.write(f'\n## {title}\n\n')
        out.write('| Name | Assumptions | Needs |\n')
        out.write('| --- | --- | --- |\n')
        for name, kind in kinds:
            if inspect.isabstract(kind):
                continue
            assumptions = _listed(getattr(kind, 'assumptions', ()), known)
            needs = _listed(getattr(kind, 'needs', ()), known)
            out.write(
                f'| {CATALOG._name(name, kind)} | {assumptions} | '
                f'{needs} |\n')
    return out.getvalue().rstrip('\n') + '\n'


""" Private Functions """


def _check(name: str) -> str:
    """Returns the name of a pretest or audit, linked to its class.

    Args:
        name: the name of the pretest or audit, or an empty `str`.

    Returns:
        The name as code, linked to the class's section in the API
            reference, or an empty `str` if there is none.

    """
    if not name:
        return ''
    kind = amos.library.borrow(name)
    return CATALOG._name(name, kind)


def _condition(name: str, kind: type[Any]) -> str:
    """Returns the name of a condition, linked to its class.

    Args:
        name: the name of the condition.
        kind: the condition's class.

    Returns:
        The name as code, linked to the class's section in the API reference.

    """
    return f'[`{name}`][{kind.__module__}.{kind.__qualname__}]'


def _conditions(kind: type[Any]) -> list[str]:
    """Returns the assumptions and needs of a technique.

    Args:
        kind: the technique.

    Returns:
        Their names.

    """
    return amos.conditions.conditions_of(kind)


def _listed(names: tuple[str, ...], known: dict[str, type[Any]]) -> str:
    """Returns the names of conditions, linked to their classes.

    Args:
        names: the names.
        known: the conditions that the page lists, by name.

    Returns:
        The linked names, joined by commas, or "none".

    """
    if not names:
        return 'none'
    return ', '.join(
        _condition(name, known[name]) if name in known else f'`{name}`'
        for name in names)


if __name__ == '__main__':
    main()
