"""Writes the tables of the technique catalog in `docs/catalog.md`.

Each technique is listed under its genre with the tool that it wraps and the
first line of its docstring. The introduction of the page (everything before
its first section) is written by hand and kept as it is. Run the script again
whenever a technique is added or its docstring changes:

    uv run python docs/scripts/technique_catalog.py

"""

from __future__ import annotations

import inspect
import io
import pathlib
from typing import Any

import amos

ROOT = pathlib.Path(__file__).parent.parent.parent
CATALOG = ROOT / 'docs' / 'catalog.md'
# The heading of the first section of the catalog, after its introduction.
FIRST = '## Loaders (wrangler)'
# The genres with a table of their own (in order), with their stages.
GENRES: tuple[tuple[str, str], ...] = (
    ('loader', 'Loaders (wrangler)'),
    ('cleaner', 'Cleaners (wrangler)'),
    ('munger', 'Mungers (wrangler)'),
    ('merger', 'Mergers (wrangler)'),
    ('shaper', 'Shapers (wrangler)'),
    ('describer', 'Describers (explorer)'),
    ('splitter', 'Splitters (analyst)'),
    ('imputer', 'Imputers (analyst)'),
    ('scaler', 'Scalers (analyst)'),
    ('encoder', 'Encoders (analyst)'),
    ('mixer', 'Mixers (analyst)'),
    ('reducer', 'Reducers (analyst)'),
    ('sampler', 'Samplers (analyst)'),
    ('model', 'Models (analyst)'),
    ('validator', 'Validators (analyst)'),
    ('metric', 'Metrics (critic)'),
    ('evaluator', 'Evaluators (critic)'),
    ('inferer', 'Inferers (critic)'),
    ('plot', 'Plots (artist)'))


def main() -> None:
    """Writes the tables of the catalog after its introduction."""
    text = CATALOG.read_text(encoding = 'utf-8')
    introduction = text[:text.index(FIRST)]
    CATALOG.write_text(introduction + tables(), encoding = 'utf-8')
    print(f'wrote {CATALOG.relative_to(ROOT)}')  # noqa: T201


def tables() -> str:
    """Returns the tables of the catalog, one section for each genre.

    Returns:
        The Markdown of the sections, ending with a new line.

    """
    out = io.StringIO()
    for genre, title in GENRES:
        out.write(f'## {title}\n\n')
        kinds = _techniques(genre)
        if genre == 'model':
            out.write('| Name | Classify | Regress | Description |\n')
            out.write('| --- | --- | --- | --- |\n')
            for name, kind in kinds:
                classify = _tool(kind.tools.get('classify'))
                regress = _tool(kind.tools.get('regress'))
                out.write(
                    f'| `{name}` | {classify} | {regress} | '
                    f'{_description(kind)} |\n')
        elif genre == 'metric':
            out.write('| Name | Tool | Task | Better | Description |\n')
            out.write('| --- | --- | --- | --- | --- |\n')
            for name, kind in kinds:
                better = 'higher' if kind.greater_is_better else 'lower'
                out.write(
                    f'| `{name}` | {_tool(kind.contents)} | '
                    f'{", ".join(kind.tasks)} | {better} | '
                    f'{_description(kind)} |\n')
        elif all(getattr(k, 'contents', None) is None for _, k in kinds):
            out.write('| Name | Description |\n')
            out.write('| --- | --- |\n')
            for name, kind in kinds:
                out.write(f'| `{name}` | {_description(kind)} |\n')
        else:
            out.write('| Name | Tool | Description |\n')
            out.write('| --- | --- | --- |\n')
            for name, kind in kinds:
                out.write(
                    f'| `{name}` | {_tool(kind.contents)} | '
                    f'{_description(kind)} |\n')
        out.write('\n')
    return out.getvalue().rstrip('\n') + '\n'


""" Private Functions """


def _description(kind: type) -> str:
    """Returns the first line of the docstring of `kind`, as a sentence.

    Args:
        kind: a technique.

    Returns:
        The first line, ending with a period.

    """
    docstring = inspect.getdoc(kind) or ''
    return docstring.splitlines()[0].rstrip('.') + '.'


def _techniques(genre: str) -> list[tuple[str, type]]:
    """Returns the name and class of every technique in a genre, sorted.

    Techniques in the genre's own genres (such as the group metrics in the
    metrics) are included.

    Args:
        genre: name of a genre in the library.

    Returns:
        The name and class of each technique.

    """
    found: list[tuple[str, type]] = []

    def collect(layer: dict[str, Any]) -> None:
        for name, value in layer.items():
            if isinstance(value, dict):
                collect(value)
            else:
                found.append((name, value))

    collect(amos.library.get_genre(genre))
    return sorted(found)


def _tool(tool: Any) -> str:
    """Returns the import path of a wrapped tool, as code.

    Args:
        tool: an import path, a class, or `None`.

    Returns:
        The import path in backticks, or an empty string for `None`.

    """
    if tool is None:
        return ''
    if isinstance(tool, str):
        return f'`{tool}`'
    return f'`{tool.__module__}.{tool.__qualname__}`'


if __name__ == '__main__':
    main()
