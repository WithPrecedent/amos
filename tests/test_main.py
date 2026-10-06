"""Tests the package as a whole."""

from __future__ import annotations

import abc
import ast
import collections
import importlib.metadata
import inspect
import pathlib
import re
import sys
import tomllib

import amos
import chrisjen

ROOT = pathlib.Path(__file__).parent.parent


def _names(layer: dict) -> list[str]:
    """Returns every class name in a layer of the library, at every level."""
    names = []
    for name, value in layer.items():
        if isinstance(value, dict):
            names.extend(_names(value))
        else:
            names.append(name)
    return names


def _normalize(name: str) -> str:
    """Returns a distribution name normalized as in PEP 503."""
    return re.sub(r'[-_.]+', '-', name.strip()).lower()


def test_version_matches_the_package_settings_and_changelog() -> None:
    with (ROOT / 'pyproject.toml').open('rb') as file:
        project = tomllib.load(file)['project']
    assert project['version'] == amos.__version__
    changelog = (ROOT / 'CHANGELOG.md').read_text(encoding = 'utf-8')
    assert f'## {amos.__version__}\n' in changelog


def test_imported_packages_are_declared() -> None:
    with (ROOT / 'pyproject.toml').open('rb') as file:
        project = tomllib.load(file)['project']
    requirements = list(project['dependencies'])
    for extra in project['optional-dependencies'].values():
        requirements.extend(extra)
    declared = {
        _normalize(r.split('>')[0].split('=')[0].split('[')[0])
        for r in requirements}
    distributions = importlib.metadata.packages_distributions()
    imported = set()
    for path in (ROOT / 'src' / 'amos').glob('*.py'):
        text = path.read_text(encoding = 'utf-8')
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                imported.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add(node.module.split('.')[0])
            # Tools imported lazily by their import paths.
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                module = node.value.split('.')[0]
                if '.' in node.value and module in distributions:
                    imported.add(module)
    third_party = imported - set(sys.stdlib_module_names) - {'amos'}
    for module in third_party:
        names = {_normalize(d) for d in distributions.get(module, [module])}
        assert names & declared, f'{module} is imported but not declared'


def test_public_names_exist() -> None:
    for name in amos.__all__:
        assert hasattr(amos, name), name
    assert len(set(amos.__all__)) == len(amos.__all__)
    assert amos.library is chrisjen.library


def test_every_genre_is_in_the_operation_layer() -> None:
    operations = chrisjen.library['vertex']['operation']
    for genre in (
        'cleaner', 'describer', 'splitter', 'transformer', 'sampler', 'model',
        'metric', 'evaluator', 'plot'):
        assert isinstance(operations[genre], dict), genre
    for genre in ('imputer', 'scaler', 'encoder', 'mixer', 'reducer'):
        assert isinstance(operations['transformer'][genre], dict), genre


def test_every_name_in_the_library_is_unique() -> None:
    counts = collections.Counter(_names(chrisjen.library.contents))
    assert [n for n, c in counts.items() if c > 1] == []


def test_every_concrete_operation_is_in_the_library() -> None:
    stored = set(chrisjen.library.all.values())
    modules = (
        amos.cleaners, amos.describers, amos.splitters, amos.transformers,
        amos.samplers, amos.models, amos.metrics, amos.evaluators, amos.plots)
    for module in modules:
        for _, kind in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(kind, amos.Operation)
                and kind.__module__ == module.__name__
                and not inspect.isabstract(kind)
                and abc.ABC not in kind.__bases__):
                assert kind in stored, kind


def test_designs_and_report_are_available_by_name() -> None:
    assert chrisjen.library.all['experiment'] is amos.Experiment
    assert chrisjen.library.all['findings'] is amos.Findings
