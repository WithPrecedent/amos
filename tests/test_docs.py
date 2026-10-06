"""Runs the code examples in the README and documentation.

Every ```python block in a file is executed, in order, in a namespace shared by
the file's blocks (so later examples can use earlier ones). A block that
follows an HTML comment like `<!-- file: settings.ini -->` and is not python is
written to that file name in a temporary folder before the examples run.

A python block that starts with `# skip` is not run. In other blocks, the
comment lines right after a `print(...)` statement are the expected output, and
the test fails if the block prints something else.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import re
import sys
import types

import pytest

import amos

ROOT = pathlib.Path(__file__).parent.parent
FILES = [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md'))]
FILE_BLOCK = re.compile(
    r'<!-- file: (?P<name>[\w./-]+) -->\s*```\w*\n(?P<text>.*?)```', re.DOTALL
)
PYTHON_BLOCK = re.compile(r'```python\n(?P<code>.*?)```', re.DOTALL)


def expected_output(code: str) -> list[str]:
    """Returns the lines in the comments that follow `print` statements."""
    expected: list[str] = []
    depth = 0
    in_print = False
    collecting = False
    for line in code.splitlines():
        stripped = line.strip()
        if in_print or stripped.startswith('print('):
            in_print = True
            collecting = False
            depth += stripped.count('(') - stripped.count(')')
            if depth <= 0:
                in_print = False
                depth = 0
                collecting = True
        elif collecting and stripped.startswith('#'):
            expected.append(stripped[1:].removeprefix(' ').rstrip())
        else:
            collecting = False
    return expected


def actual_output(text: str) -> list[str]:
    """Returns printed lines without trailing blank lines."""
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return lines


# The classes that the examples add to the library are removed after each file
# by the `restore_library` fixture in conftest.py.


@pytest.mark.parametrize('path', FILES, ids = lambda p: p.name)
def test_examples_run(
    path: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = path.read_text(encoding = 'utf-8')
    monkeypatch.chdir(tmp_path)
    for match in FILE_BLOCK.finditer(text):
        target = tmp_path / match['name']
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_text(match['text'], encoding = 'utf-8')
    # The examples run as a real module so that classes defined in them (such
    # as dataclasses) can find their module.
    module = types.ModuleType(f'docs_{path.stem}')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = module.__dict__
    for number, match in enumerate(PYTHON_BLOCK.finditer(text), start = 1):
        code = match['code']
        if code.startswith('# skip'):
            continue
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                exec(  # noqa: S102
                    compile(code, f'{path.name}[block {number}]', 'exec'),
                    namespace,
                )
        except Exception as error:  # noqa: BLE001
            message = (
                f'{path.name} block {number} failed with {error!r}:\n{code}'
            )
            raise AssertionError(message) from error
        expected = expected_output(code)
        actual = actual_output(printed.getvalue())
        if expected or actual:
            assert actual == expected, (
                f'{path.name} block {number} printed different output than '
                f'its comments say:\n{code}'
            )
    assert os.getcwd() == str(tmp_path)  # noqa: PTH109


def test_every_technique_is_in_the_catalog() -> None:
    """Checks that the technique catalog in the docs lists every technique."""
    text = (ROOT / 'docs' / 'catalog.md').read_text(encoding = 'utf-8')
    operations = amos.library.get_genre('operation')
    names = [
        name for name, kind in amos.library.all.items()
        if isinstance(kind, type) and issubclass(kind, amos.Operation)]
    assert operations
    missing = [name for name in names if f'`{name}`' not in text]
    assert missing == [], f'missing from docs/catalog.md: {missing}'


def test_the_catalog_is_up_to_date() -> None:
    """Checks that the catalog's tables match the techniques' docstrings."""
    path = ROOT / 'docs' / 'scripts' / 'technique_catalog.py'
    spec = importlib.util.spec_from_file_location('technique_catalog', path)
    assert spec is not None
    assert spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    text = script.CATALOG.read_text(encoding = 'utf-8')
    assert text[text.index(script.FIRST):] == script.tables(), (
        'docs/catalog.md is out of date: run '
        '"uv run python docs/scripts/technique_catalog.py"')
