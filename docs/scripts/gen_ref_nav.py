"""Generate the code reference pages and navigation.

The pages follow the stages of a project: the core of `amos`, then each
stage with a page for each of its steps. A step is documented where it is
imported from (such as `amos.cleaners`), not in the folder of its stage
(`amos.wrangler.cleaners`), since its techniques say they are there.

This code is adapted from pawamoy's packages. See, e.g.,
https://github.com/pawamoy/duty/blob/main/scripts/gen_ref_nav.py.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

SRC = Path('src')
PACKAGE = SRC / 'amos'
# The stages, in the order of a project, with the steps of each (in the
# order of the stage). A step in this order comes first; any other is added
# after them, in alphabetical order.
STAGES: dict[str, tuple[str, ...]] = {
    'wrangler': (
        'loaders', 'parsers', 'cleaners', 'mungers', 'mergers', 'shapers'),
    'explorer': ('describers',),
    'analyst': (
        'splitters', 'transformers', 'imputers', 'scalers', 'encoders',
        'mixers', 'selectors', 'reducers', 'samplers', 'pretests',
        'models', 'validators'),
    'critic': ('evaluators', 'audits', 'inferers'),
    'artist': ('plots',)}

nav = mkdocs_gen_files.Nav()
mod_symbol = '<code class="doc-symbol doc-symbol-nav doc-symbol-module"></code>'


def _write(ident: str, nav_parts: tuple[str, ...], source: Path) -> None:
    """Writes the page of `ident` and adds it to the navigation.

    Args:
        ident: the dotted path of the module to document.
        nav_parts: the titles of the page in the navigation.
        source: the file that the page's "edit" link opens.

    """
    doc_path = Path(*ident.split('.')).with_suffix('.md')
    if ident == 'amos':
        doc_path = Path('amos', 'index.md')
    nav[tuple(f'{mod_symbol} {part}' for part in nav_parts)] = (
        doc_path.as_posix())
    full_doc_path = Path('reference', doc_path)
    with mkdocs_gen_files.open(full_doc_path, 'w') as fd:
        fd.write(f'::: {ident}')
    mkdocs_gen_files.set_edit_path(full_doc_path, '..' / source)


def _steps(stage: str) -> list[str]:
    """Returns the steps of `stage`, in order.

    Args:
        stage: the name of a stage.

    Returns:
        The names of the steps of `stage`.

    """
    folder = PACKAGE / stage
    found = {
        path.name for path in folder.iterdir()
        if (path / '__init__.py').exists()}
    if (folder / '_transformers.py').exists():
        found.add('transformers')
    ordered = [step for step in STAGES[stage] if step in found]
    return ordered + sorted(found - set(ordered))


_write('amos', ('amos',), PACKAGE / '__init__.py')
for path in sorted((PACKAGE / 'core').glob('*.py')):
    if path.stem.startswith('_'):
        continue
    _write(f'amos.core.{path.stem}', ('core', path.stem), path)
for stage in STAGES:
    for step in _steps(stage):
        if step == 'transformers':
            source = PACKAGE / stage / '_transformers.py'
        else:
            source = PACKAGE / stage / step / '__init__.py'
        _write(f'amos.{step}', (stage, step), source)

with mkdocs_gen_files.open('reference/SUMMARY.md', 'w') as nav_file:
    nav_file.writelines(nav.build_literate_nav())
