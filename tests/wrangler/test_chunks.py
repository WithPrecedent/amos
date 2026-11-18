"""Tests reading huge CSV files a chunk at a time."""

from __future__ import annotations

import bz2
import gzip
import logging
import lzma
import os
import pathlib

import pandas as pd
import pyarrow as pa
import pytest
from conftest import LocalServer

import amos
import nagata
from amos import loaders

# Written as PostgreSQL writes CSV files: a backslash before a quote in a
# value, and line breaks in quoted values.
ROWS = (
    'id,court_id,date_filed,text\n'
    '1,ca1,2009-05-01,"plain"\n'
    '2,ca2,2011-02-03,"He said \\"no\\""\n'
    '3,ca1,2012-07-08,"two\nlines"\n'
    '4,ca9,2013-01-01,\n'
    '5,ca1,2014-03-04,"last, with a comma"\n')
# Keeps the dialect of `ROWS`.
DIALECT = {'escape_char': '\\', 'double_quote': False}


def _write(path: pathlib.Path, text: str = ROWS) -> pathlib.Path:
    """Writes `text` to `path`, compressed as its extension says."""
    data = ('﻿' + text).encode('utf-8')
    suffix = path.suffix
    if suffix == '.bz2':
        data = bz2.compress(data)
    elif suffix == '.gz':
        data = gzip.compress(data)
    elif suffix == '.xz':
        data = lzma.compress(data)
    elif suffix == '.zst':
        with pa.output_stream(str(path), compression = 'zstd') as stream:
            stream.write(data)
        return path
    path.write_bytes(data)
    return path


@pytest.fixture(autouse = True)
def _work_in(tmp_path: pathlib.Path) -> None:
    """Runs each test in its own folder."""
    os.chdir(tmp_path)


@pytest.mark.parametrize(
    'name', ['rows.csv', 'rows.csv.bz2', 'rows.csv.gz', 'rows.csv.xz',
             'rows.csv.zst'])
def test_read_chunks_reads_every_compression(name: str) -> None:
    path = _write(pathlib.Path(name))
    chunks = list(loaders.read_chunks(path, processes = 2, **DIALECT))
    data = pd.concat(chunks)
    assert data.columns.tolist() == ['id', 'court_id', 'date_filed', 'text']
    assert data['text'].tolist() == [
        'plain', 'He said "no"', 'two\nlines', '', 'last, with a comma']
    assert data.index.tolist() == [0, 1, 2, 3, 4]
    assert all(pd.api.types.is_string_dtype(t) for t in data.dtypes)


def test_read_chunks_keeps_chosen_rows_and_columns() -> None:
    path = _write(pathlib.Path('rows.csv.bz2'))
    chunks = list(loaders.read_chunks(
        path,
        columns = ['text', 'id', 'absent'],
        keep = {'court_id': ['ca1', 'ca2'], 'id': [2, 3, 5]},
        where = "text != 'last, with a comma'",
        block_size = 64,
        **DIALECT))
    # A small block parses a few rows at a time.
    assert len(chunks) > 1
    data = pd.concat(chunks)
    assert data.columns.tolist() == ['text', 'id']
    assert data['id'].tolist() == ['2', '3']
    # Each row is labeled by its place in the file.
    assert data.index.tolist() == [1, 2]
    one = pd.concat(loaders.read_chunks(
        path, keep = {'court_id': 'ca9'}, empty_as_missing = True, **DIALECT))
    assert one['id'].tolist() == ['4']
    assert pd.isna(one['text'].iloc[0])


def test_read_chunks_filters_with_a_function() -> None:
    path = _write(pathlib.Path('rows.csv'))
    data = pd.concat(loaders.read_chunks(
        path,
        columns = ['id', 'date_filed'],
        where = lambda chunk: chunk['date_filed'] >= '2012-01-01',
        **DIALECT))
    assert data['id'].tolist() == ['3', '4', '5']
    assert list(loaders.read_chunks(
        path, keep = {'court_id': []}, **DIALECT)) == []


def test_read_chunks_reports_its_progress_and_problems(
    caplog: pytest.LogCaptureFixture) -> None:
    path = _write(pathlib.Path('rows.csv'))
    with caplog.at_level(logging.INFO, logger = 'amos'):
        list(loaders.read_chunks(
            path, columns = ['id', 'absent'], block_size = 48,
            report_every = 2, **DIALECT))
    messages = [record.getMessage() for record in caplog.records]
    assert "rows.csv has no column ['absent'], so it is left out" in messages
    assert any(m.endswith(' rows read') for m in messages)
    assert 'rows.csv: read all 5 rows' in messages
    with pytest.raises(KeyError, match = 'no column'):
        list(loaders.read_chunks(path, keep = {'absent': ['x']}))
    with pytest.raises(ValueError, match = 'block_size'):
        list(loaders.read_chunks(path, block_size = 20, **DIALECT))
    pathlib.Path('empty.csv').write_bytes(b'')
    with pytest.raises(ValueError, match = 'column names'):
        list(loaders.read_chunks('empty.csv'))


def test_read_chunks_takes_other_dialects() -> None:
    text = 'id;note\n1;\'a;b\'\n2;\'it\'\'s\'\n'
    pathlib.Path('rows.txt').write_text(text, encoding = 'cp1252')
    data = pd.concat(loaders.read_chunks(
        'rows.txt', delimiter = ';', quote_char = "'", encoding = 'cp1252'))
    assert data['note'].tolist() == ['a;b', "it's"]
    pathlib.Path('bare.tsv').write_text('a\tb\n"x\t1\n', encoding = 'utf-8')
    data = pd.concat(loaders.read_chunks(
        'bare.tsv', delimiter = '\t', quote_char = None))
    assert data['a'].tolist() == ['"x']


def test_read_chunks_streams_a_url(server: LocalServer) -> None:
    _write(pathlib.Path('web/rows.csv.bz2'))
    data = pd.concat(loaders.read_chunks(
        f'{server.url}/rows.csv.bz2', keep = {'court_id': 'ca1'}, **DIALECT))
    assert data['id'].tolist() == ['1', '3', '5']
    assert not pathlib.Path('rows.csv.bz2').exists()


def test_load_chunks_downloads_and_loads_chosen_rows(
    server: LocalServer) -> None:
    _write(pathlib.Path('web/rows.csv.bz2'))
    clerk = nagata.FileManager(root_folder = '.', input_folder = 'data')
    loader = loaders.LoadChunks(clerk = clerk)
    url = f'{server.url}/rows.csv.bz2'
    dataset = loader.apply(
        None,
        source = url,
        columns = ['id', 'court_id'],
        keep = {'court_id': ['ca1', 'ca9']},
        block_size = 64,
        **DIALECT)
    assert dataset.data.to_dict('list') == {
        'id': ['1', '3', '4', '5'], 'court_id': ['ca1', 'ca1', 'ca9', 'ca1']}
    assert dataset.data.index.tolist() == [0, 1, 2, 3]
    assert pathlib.Path('data/rows.csv.bz2').is_file()
    assert dataset.history[-1]['source'] == url
    # The file is kept, so it is not downloaded again, and "limit" stops
    # reading early.
    dataset = loader.apply(None, source = url, limit = 2, **DIALECT)
    assert dataset.data['id'].tolist() == ['1', '2']
    assert server.requests == ['/rows.csv.bz2']
    loader.apply(None, source = url, stream = True, **DIALECT)
    assert server.requests == ['/rows.csv.bz2', '/rows.csv.bz2']
    # A local file is found in the input folder.
    empty = loader.apply(
        None,
        source = 'rows.csv.bz2',
        columns = ['id'],
        keep = {'court_id': 'none'},
        **DIALECT)
    assert empty.data.columns.tolist() == ['id']
    assert empty.data.empty
    assert amos.library.classify('load_chunks') == 'loader'
