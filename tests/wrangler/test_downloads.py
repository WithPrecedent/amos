"""Tests the tools of the loaders that download files and read web pages."""

from __future__ import annotations

import io
import os
import pathlib
import zipfile
from typing import Any

import pandas as pd
import pytest
import requests
from conftest import LocalServer

import nagata
from amos import loaders

PAGE = """
<html><body>
  <p>Releases:
    <a href="files/old.zip">CSV  Organized by
      Supreme Court Citation</a>
    <a href="/files/new.zip">csv organized by supreme court citation</a>
    <a href="https://example.org/other.zip">Other</a>
    <a name="anchor">No address</a>
  </p>
</body></html>
"""


class _Reply:
    """A response like those of `requests`, from a fake session."""

    def __init__(
        self,
        content: bytes,
        status: int = 200,
        headers: dict[str, str] | None = None) -> None:
        self.content = content
        self.status_code = status
        self.headers = headers or {}
        self.closed = False

    @property
    def text(self) -> str:
        """Returns the content as text."""
        return self.content.decode('utf-8')

    def raise_for_status(self) -> None:
        """Raises an error for a response that failed."""
        if self.status_code >= 400:
            raise requests.HTTPError(f'status {self.status_code}')

    def close(self) -> None:
        """Records that the response was closed."""
        self.closed = True


class _Site:
    """A session that serves pages and files by their addresses."""

    def __init__(self, pages: dict[str, bytes]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _Reply:
        """Records the request and returns the page or file at an address."""
        self.calls.append((url, kwargs))
        if url not in self.pages:
            return _Reply(b'', 404)
        return _Reply(self.pages[url])


def _archive(*members: str) -> bytes:
    """Returns a zip archive with a small file for each name."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for member in members:
            archive.writestr(member, 'a,b\n1,2\n')
    return buffer.getvalue()


@pytest.fixture(autouse = True)
def _work_in(tmp_path: pathlib.Path) -> None:
    """Runs each test in its own folder."""
    os.chdir(tmp_path)


def test_download_file_saves_once_unless_refreshed() -> None:
    site = _Site({'https://example.org/data.csv': b'a,b\n1,2\n'})
    path = loaders.download_file(
        'https://example.org/data.csv',
        'raw/data.csv',
        session = site,
        timeout = 5)
    assert path == pathlib.Path('raw/data.csv')
    assert path.read_bytes() == b'a,b\n1,2\n'
    assert site.calls[0][1]['timeout'] == 5
    assert site.calls[0][1]['stream'] is True
    loaders.download_file(
        'https://example.org/data.csv', 'raw/data.csv', session = site)
    assert len(site.calls) == 1
    loaders.download_file(
        'https://example.org/data.csv',
        'raw/data.csv',
        session = site,
        refresh = True)
    assert len(site.calls) == 2
    assert list(pathlib.Path('raw').iterdir()) == [path]


def test_download_file_works_with_a_real_server(server: LocalServer) -> None:
    folder = pathlib.Path('web')
    (folder / 'data.csv').write_text('a,b\n1,2\n', encoding = 'utf-8')
    path = loaders.download_file(f'{server.url}/data.csv', 'saved.csv')
    assert path.read_text(encoding = 'utf-8') == 'a,b\n1,2\n'
    with pytest.raises(requests.HTTPError):
        loaders.download_file(f'{server.url}/missing.csv', 'missing.csv')
    assert list(pathlib.Path().glob('missing.csv*')) == []
    with pytest.raises(ValueError, match = 'http and https'):
        loaders.download_file('ftp://example.org/data.csv', 'data.csv')


def test_download_file_resumes_a_download_that_stopped() -> None:
    content = bytes(range(256)) * 40

    class Ranged:
        """A session that sends the part of a file that a request asks for."""

        def __init__(self, sent: int | None = None) -> None:
            self.sent = sent

        def get(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            **kwargs: Any) -> _Reply:
            start = int((headers or {}).get('Range', 'bytes=0-')[6:-1])
            part = content[start:]
            length = {'Content-Length': str(len(part))}
            if self.sent is not None:
                part = part[:self.sent]
            return _Reply(part, 206 if start else 200, length)

    url = 'https://example.org/big.bin'
    with pytest.raises(OSError, match = 'resume'):
        loaders.download_file(url, 'big.bin', session = Ranged(1000), resume = True)
    assert pathlib.Path('big.bin.part').stat().st_size == 1000
    assert not pathlib.Path('big.bin').exists()
    path = loaders.download_file(url, 'big.bin', session = Ranged(), resume = True)
    assert path.read_bytes() == content
    assert not pathlib.Path('big.bin.part').exists()
    # Without "resume", a download that stops leaves nothing behind.
    with pytest.raises(OSError, match = 'stopped after'):
        loaders.download_file(url, 'other.bin', session = Ranged(10))
    assert list(pathlib.Path().glob('other.bin*')) == []


def test_download_file_checks_the_members_of_an_archive() -> None:
    site = _Site({
        'https://example.org/cases.zip': _archive('SCDB_caseCentered.csv'),
        'https://example.org/page.html': b'<p>not an archive</p>'})
    path = loaders.download_file(
        'https://example.org/cases.zip',
        'cases.zip',
        session = site,
        required_member = '*caseCentered*')
    assert path.is_file()
    with pytest.raises(ValueError, match = 'holds'):
        loaders.download_file(
            'https://example.org/cases.zip',
            'votes.zip',
            session = site,
            required_member = '*justiceCentered*')
    with pytest.raises(ValueError, match = 'not a zip archive'):
        loaders.download_file(
            'https://example.org/page.html',
            'page.zip',
            session = site,
            required_member = '*')
    assert sorted(p.name for p in pathlib.Path().iterdir()) == ['cases.zip']


def test_page_links_and_find_links() -> None:
    site = _Site({'https://example.org/data/': PAGE.encode('utf-8')})
    links = loaders.page_links('https://example.org/data/', session = site)
    assert links == [
        ('https://example.org/data/files/old.zip',
         'CSV Organized by Supreme Court Citation'),
        ('https://example.org/files/new.zip',
         'csv organized by supreme court citation'),
        ('https://example.org/other.zip', 'Other')]
    found = loaders.find_links(
        'https://example.org/data/',
        'CSV organized by Supreme Court citation',
        session = site)
    assert found == [
        'https://example.org/data/files/old.zip',
        'https://example.org/files/new.zip']
    assert loaders.find_links(
        'https://example.org/data/', 'Missing', session = site) == []
    with pytest.raises(requests.HTTPError):
        loaders.page_links('https://example.org/gone/', session = site)


def test_download_linked_file_tries_each_link_until_one_passes() -> None:
    site = _Site({
        'https://example.org/data/': PAGE.encode('utf-8'),
        'https://example.org/data/files/old.zip': _archive('votes.csv'),
        'https://example.org/files/new.zip': _archive('cases_2026.csv')})
    path = loaders.download_linked_file(
        'https://example.org/data/',
        'CSV Organized by Supreme Court Citation',
        'downloads',
        required_member = 'cases_*.csv',
        session = site)
    assert path == pathlib.Path('downloads/new.zip')
    assert not pathlib.Path('downloads/old.zip').exists()
    # Without a check, the first link is used, and a named file is only
    # downloaded once.
    path = loaders.download_linked_file(
        'https://example.org/data/',
        'CSV Organized by Supreme Court Citation',
        'downloads',
        file_name = 'first.zip',
        session = site)
    assert zipfile.ZipFile(path).namelist() == ['votes.csv']
    calls = len(site.calls)
    loaders.download_linked_file(
        'https://example.org/data/',
        'CSV Organized by Supreme Court Citation',
        'downloads',
        file_name = 'first.zip',
        session = site)
    assert len(site.calls) == calls
    with pytest.raises(ValueError, match = 'holds'):
        loaders.download_linked_file(
            'https://example.org/data/',
            'CSV Organized by Supreme Court Citation',
            'downloads',
            required_member = 'nothing*',
            refresh = True,
            session = site)
    with pytest.raises(LookupError, match = 'no link'):
        loaders.download_linked_file(
            'https://example.org/data/', 'Missing', 'downloads', session = site)


def test_open_session_identifies_amos() -> None:
    site = _Site({})
    assert loaders.open_session(site) is site
    session = loaders.open_session()
    assert isinstance(session, requests.Session)
    assert session.headers['User-Agent'].startswith('amos')


def test_download_loader_finds_a_file_by_its_link(server: LocalServer) -> None:
    folder = pathlib.Path('web')
    (folder / 'files').mkdir()
    (folder / 'index.html').write_text(PAGE, encoding = 'utf-8')
    (folder / 'files' / 'old.zip').write_bytes(_archive('votes.csv'))
    (folder / 'files' / 'new.zip').write_bytes(_archive('cases.csv'))
    clerk = nagata.FileManager(root_folder = '.', input_folder = 'data')
    dataset = loaders.Download(clerk = clerk).apply(
        None,
        source = f'{server.url}/index.html',
        link = 'csv organized by supreme court citation',
        required_member = 'cases.csv',
        member = 'cases.csv')
    assert dataset.data.to_dict('list') == {'a': [1], 'b': [2]}
    assert pathlib.Path('data/new.zip').is_file()
    assert server.requests == [
        '/index.html', '/files/old.zip', '/files/new.zip']


def test_download_loader_takes_a_session() -> None:
    site = _Site({'https://example.org/data.csv': b'a,b\n1,2\n'})
    clerk = nagata.FileManager(root_folder = '.', input_folder = 'data')
    dataset = loaders.Download(clerk = clerk).apply(
        None,
        source = 'https://example.org/data.csv',
        session = site)
    pd.testing.assert_frame_equal(
        dataset.data, pd.DataFrame({'a': [1], 'b': [2]}))
    assert pathlib.Path('data/data.csv').is_file()
