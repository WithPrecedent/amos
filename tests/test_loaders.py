"""Tests the loaders module."""

from __future__ import annotations

import functools
import http.server
import os
import pathlib
import threading
import urllib.error
import zipfile
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
import sklearn.utils
from conftest import SEED, make_numeric, make_regression

import amos
import nagata
from amos import loaders


class _Server(http.server.ThreadingHTTPServer):
    """A local web server that counts the requests for each file."""

    requests: list[str]

    @property
    def url(self) -> str:
        """Returns the address of the server."""
        host, port = self.server_address[:2]
        return f'http://{host!s}:{port}'


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves files from a folder and records each request."""

    def do_GET(self) -> None:  # noqa: N802
        self.server.requests.append(self.path)  # type: ignore[attr-defined]
        super().do_GET()

    def log_message(self, *args: Any) -> None:
        """Keeps the test output quiet."""


@pytest.fixture(autouse = True)
def _work_in(tmp_path: pathlib.Path) -> None:
    """Runs each test in its own folder."""
    os.chdir(tmp_path)


@pytest.fixture
def cases() -> pd.DataFrame:
    """Returns data with more rows than a clerk reads by default."""
    data = make_numeric(rows = 1500)
    data['court'] = 'Cour d’appel'
    return data


@pytest.fixture
def server(tmp_path: pathlib.Path) -> Iterator[_Server]:
    """Serves the files in a "web" folder at a local address."""
    folder = tmp_path / 'web'
    folder.mkdir()
    handler = functools.partial(_Handler, directory = str(folder))
    served = _Server(('127.0.0.1', 0), handler)
    served.requests = []
    thread = threading.Thread(target = served.serve_forever, daemon = True)
    thread.start()
    yield served
    served.shutdown()
    served.server_close()


def test_load_file_reads_every_row_as_utf_8(cases: pd.DataFrame) -> None:
    cases.to_csv('cases.csv', index = False, encoding = 'utf-8')
    dataset = loaders.LoadFile().apply(
        None, source = 'cases.csv', label = 'target')
    assert len(dataset.data) == 1500
    assert dataset.data['court'].iloc[0] == 'Cour d’appel'
    assert dataset.label == 'target'
    assert dataset.task == 'classify'
    assert dataset.history == [{
        'technique': 'load_file',
        'source': 'cases.csv',
        'rows': 1500,
        'columns': 8}]


def test_load_file_looks_in_the_clerk_input_folder(
    cases: pd.DataFrame) -> None:
    clerk = nagata.FileManager(root_folder = '.', input_folder = 'data')
    cases.to_csv('data/cases.csv', index = False)
    dataset = loaders.LoadFile(clerk = clerk).apply(None, source = 'cases.csv')
    assert dataset.data.shape == (1500, 8)
    with pytest.raises(FileNotFoundError, match = 'no file'):
        loaders.LoadFile(clerk = clerk).apply(None, source = 'missing.csv')


def test_load_file_keeps_the_seed_label_and_history() -> None:
    make_numeric().to_csv('cases.csv', index = False)
    start = amos.Dataset(seed = SEED)
    start.record('earlier')
    dataset = loaders.LoadFile().apply(
        start, source = 'cases.csv', label = 'target', groups = 'x0')
    assert dataset.seed == SEED
    assert dataset.groups == ['x0']
    assert [entry['technique'] for entry in dataset.history] == [
        'earlier', 'load_file']


def test_load_file_infers_the_task_of_the_new_data() -> None:
    make_numeric().to_parquet('cases.parquet')
    start = amos.Dataset(make_regression(), label = 'target')
    assert start.task == 'regress'
    # The parquet reader passes on any parameter, so this also checks that
    # "label" and "task" are not passed to it.
    dataset = loaders.LoadFile().apply(start, source = 'cases.parquet')
    assert (dataset.label, dataset.task) == ('target', 'classify')
    dataset = loaders.LoadFile().apply(
        None, source = 'cases.parquet', label = 'target', task = 'regress')
    assert dataset.task == 'regress'


@pytest.mark.parametrize('name', ['cases.csv.gz', 'cases.tsv', 'cases.json'])
def test_load_file_finds_the_format_from_the_name(
    cases: pd.DataFrame,
    name: str) -> None:
    if name.endswith('.json'):
        cases.to_json(name)
    else:
        cases.to_csv(name, index = False, sep = ',' if 'csv' in name else '\t')
    dataset = loaders.LoadFile().apply(None, source = name)
    assert dataset.data.shape == (1500, 8)


def test_load_file_with_a_format_member_and_reader_parameters(
    cases: pd.DataFrame) -> None:
    cases.to_csv('cases.data', index = False, sep = ';')
    with pytest.raises(ValueError, match = 'file_format'):
        loaders.LoadFile().apply(None, source = 'cases.data')
    # "threshold" is not a parameter of the reader, so it is left out.
    dataset = loaders.LoadFile().apply(
        None,
        source = 'cases.data',
        file_format = 'csv',
        sep = ';',
        threshold = 3)
    assert dataset.data.shape == (1500, 8)
    with zipfile.ZipFile('archive.zip', 'w') as archive:
        archive.write('cases.data', 'inner/cases.csv')
    dataset = loaders.LoadFile().apply(
        None, source = 'archive.zip', member = 'inner/cases.csv', sep = ';')
    assert dataset.data.shape == (1500, 8)
    assert pathlib.Path('archive/inner/cases.csv').is_file()
    with pytest.raises(KeyError, match = 'is not in'):
        loaders.LoadFile().apply(None, source = 'archive.zip', member = 'x')
    with pytest.raises(ValueError, match = 'not a format'):
        loaders.LoadFile().apply(
            None, source = 'cases.data', file_format = 'spreadsheet')


def test_a_loader_needs_a_source() -> None:
    with pytest.raises(ValueError, match = 'nothing to load'):
        loaders.LoadFile().apply(None)


def test_download_saves_the_file_once(
    server: _Server,
    tmp_path: pathlib.Path,
    cases: pd.DataFrame) -> None:
    cases.to_csv(tmp_path / 'web' / 'cases.csv', index = False)
    clerk = nagata.FileManager(root_folder = '.', input_folder = 'data')
    loader = loaders.Download(clerk = clerk)
    url = f'{server.url}/cases.csv'
    dataset = loader.apply(None, source = url, label = 'target')
    assert dataset.data.shape == (1500, 8)
    assert pathlib.Path('data/cases.csv').is_file()
    assert dataset.history[-1]['source'] == url
    loader.apply(None, source = url)
    assert server.requests == ['/cases.csv']
    loader.apply(None, source = url, refresh = True)
    assert server.requests == ['/cases.csv', '/cases.csv']


def test_download_names_and_unpacks_the_file(
    server: _Server,
    tmp_path: pathlib.Path,
    cases: pd.DataFrame) -> None:
    with zipfile.ZipFile(tmp_path / 'web' / 'bundle.zip', 'w') as archive:
        archive.writestr('cases.csv', cases.to_csv(index = False))
    url = f'{server.url}/bundle.zip?version=2'
    dataset = loaders.Download().apply(
        None,
        source = url,
        file_name = 'raw/cases.zip',
        member = 'cases.csv')
    assert dataset.data.shape == (1500, 8)
    assert pathlib.Path('raw/cases.zip').is_file()
    assert pathlib.Path('raw/cases/cases.csv').is_file()


def test_download_errors_leave_no_file(server: _Server) -> None:
    with pytest.raises(urllib.error.HTTPError):
        loaders.Download().apply(None, source = f'{server.url}/missing.csv')
    assert list(pathlib.Path().glob('missing.csv*')) == []
    with pytest.raises(ValueError, match = 'http and https'):
        loaders.Download().apply(None, source = 'file:///etc/hosts')
    with pytest.raises(ValueError, match = 'file_name'):
        loaders.Download().apply(None, source = f'{server.url}/')


def test_openml_downloads_into_the_clerk_input_folder() -> None:
    calls = []

    # Takes the same parameters as `sklearn.datasets.fetch_openml` (and a few
    # fewer), so "extra" is left out.
    def fetch(
        name = None,
        *,
        version = 'active',
        data_id = None,
        data_home = None,
        as_frame = 'auto',
        return_X_y = False):  # noqa: N803
        calls.append({
            'name': name,
            'data_id': data_id,
            'data_home': data_home,
            'as_frame': as_frame,
            'return_X_y': return_X_y,
            'version': version})
        frame = make_numeric()
        return sklearn.utils.Bunch(frame = frame, target = frame['target'])

    clerk = nagata.FileManager(root_folder = '.', input_folder = 'data')
    loader = loaders.Openml(clerk = clerk, contents = fetch)
    dataset = loader.apply(None, source = 'credit-g', version = 1, extra = 1)
    assert dataset.label == 'target'
    assert calls[-1] == {
        'name': 'credit-g',
        'data_id': None,
        'data_home': str(pathlib.Path('data')),
        'as_frame': True,
        'return_X_y': False,
        'version': 1}
    loader.apply(None, source = '31')
    assert calls[-1]['data_id'] == 31
    assert calls[-1]['name'] is None


def _settings(**extra: Any) -> dict[str, Any]:
    """Returns settings for a project that loads its own data."""
    return {
        'general': {'label': 'target', 'seed': SEED},
        'files': {'input_folder': 'data'},
        'study_project': {'study_workers': 'wrangler, analyst'},
        'wrangler': {
            'steps': 'load, clean',
            'load_techniques': 'load_file',
            'clean_techniques': 'drop_duplicates'},
        'load_file_parameters': {'source': 'cases.csv'},
        'analyst': {'techniques': 'stratified, sk_logit, accuracy'},
        **extra}


def test_a_project_with_a_loader_needs_no_item() -> None:
    pathlib.Path('data').mkdir()
    make_numeric().to_csv('data/cases.csv', index = False)
    project = amos.Project.create(_settings(), id = 'run')
    result = project.result
    assert isinstance(result, amos.Dataset)
    assert (result.label, result.seed) == ('target', SEED)
    assert result.history[0]['technique'] == 'load_file'
    assert 'accuracy' in result.metrics
    nodes = amos.interface._nodes(project.workflow)
    loader = next(n for n in nodes if isinstance(n, amos.Loader))
    assert loader.clerk is project.clerk
    # Applying again loads the data again, into a new dataset.
    assert len(project.apply().history) == len(result.history)


def test_loader_parameters_take_precedence_over_general() -> None:
    pathlib.Path('data').mkdir()
    data = make_numeric().rename(columns = {'target': 'outcome'})
    data.to_csv('data/cases.csv', index = False)
    settings = _settings(load_file_parameters = {
        'source': 'cases.csv', 'label': 'outcome'})
    project = amos.Project.create(settings, id = 'run')
    assert project.result.label == 'outcome'


def test_a_project_without_data_or_a_loader_waits() -> None:
    settings = _settings(wrangler = {'techniques': 'drop_duplicates'})
    project = amos.Project.create(settings, id = 'run')
    assert project.result is None
    with pytest.raises(ValueError, match = 'loader'):
        project.apply()
