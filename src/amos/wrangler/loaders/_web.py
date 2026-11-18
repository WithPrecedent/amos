"""Tools that download files and read the links of web pages.

They work outside of a project as well as in the `download` loader, and take
any session with a `get` method like a `requests.Session` (such as a fake
one in tests), so that a package built on `amos` can use its own.

Contents:
    download_file: downloads a file, unless it was downloaded before.
    download_linked_file: downloads the file that a link on a web page names.
    find_links: returns the addresses of the links on a web page with chosen
        words.
    open_session: returns a session that identifies `amos` to servers.
    page_links: returns the address and words of each link on a web page.
    logger: the module's logger, which reports the files that are
        downloaded.
    _CHUNK: bytes of a download that are written to its file at a time.
    _PARTIAL: added to the name of a file while it is being downloaded.
    _SCHEMES: schemes of the URLs that can be downloaded.
    _USER_AGENT: sent with each request, since some servers refuse requests
        without one.

"""

from __future__ import annotations

import fnmatch
import functools
import html.parser
import http
import logging
import pathlib
import urllib.parse
import zipfile
from typing import Any

import requests

from ...core import options

logger = logging.getLogger(__name__)

# Bytes of a download that are written to its file at a time.
_CHUNK: int = 8 * 1024 * 1024
# Added to the name of a file while it is being downloaded, so that a
# download that stops is never mistaken for the whole file.
_PARTIAL: str = '.part'
# Schemes of the URLs that can be downloaded.
_SCHEMES: frozenset[str] = frozenset({'http', 'https'})
# Sent with each request, since some servers refuse requests without one.
_USER_AGENT: str = 'amos (https://github.com/WithPrecedent/amos)'


class _LinkExtractor(html.parser.HTMLParser):
    """Collects the links of a web page, with the words of each.

    `html.parser.HTMLParser` reads the markup and calls the `handle_...`
    methods for each tag and piece of text, which this class records. Feed it
    markup with `feed`, call `close`, and read `links`.

    Attributes:
        links: the address ("href") and the words of each link found so far,
            in the order of the page. The words are a `list` of the pieces
            of text in the link.
        inside: whether the parser is inside a link, so that text belongs to
            the last of `links`.

    """

    """ Initialization Methods """

    # pydoclint wants `__init__` documented in the class's docstring.
    def __init__(self) -> None:
        super().__init__(convert_charrefs = True)
        self.links: list[tuple[str, list[str]]] = []
        self.inside: bool = False

    """ Public Methods """

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]]) -> None:
        """Starts a link at an "a" tag that has an address.

        Args:
            tag: name of the tag, in lower case.
            attrs: the tag's attributes, as pairs of names and values.

        """
        if tag == 'a':
            address = dict(attrs).get('href')
            self.inside = bool(address)
            if address:
                self.links.append((address, []))

    def handle_endtag(self, tag: str) -> None:
        """Ends a link at the closing "a" tag.

        Args:
            tag: name of the tag, in lower case.

        """
        if tag == 'a':
            self.inside = False

    def handle_data(self, data: str) -> None:
        """Records text that is inside a link.

        Args:
            data: text between tags.

        """
        if self.inside:
            self.links[-1][1].append(data)


""" Public Functions """


def download_file(
    url: str,
    path: pathlib.Path | str,
    *,
    refresh: bool = False,
    timeout: float | None = None,
    session: Any = None,
    resume: bool = False,
    required_member: str | None = None) -> pathlib.Path:
    """Downloads a file, unless it was downloaded before.

    The file is written beside `path` (with ".part" added to its name) as it
    arrives, and is renamed to `path` once it is whole, so a download that
    stops is never mistaken for the data. Large files (even of many
    gigabytes) are written a piece at a time, so they need little memory.

    Args:
        url: the address of the file (http or https).
        path: where to save it. Its folder is made if needed, and "~" is the
            user's home folder.
        refresh: whether to download the file even if there is one at
            `path`. Defaults to `False`.
        timeout: seconds to wait for the server to answer. Defaults to
            `None`, which uses `options._DOWNLOAD_TIMEOUT` (read when the
            file is downloaded).
        session: an object with a `get` method like a `requests.Session`.
            Defaults to `None`, in which case `open_session` makes one.
        resume: whether to continue a download that stopped (whose ".part"
            file is still there) instead of starting over, if the server
            sends parts of files. A download that fails keeps its ".part"
            file. Defaults to `False`, in which case a download that fails
            leaves no file behind.
        required_member: a pattern (with "*" and "?", as in `fnmatch`) that
            the name of a file in the downloaded zip archive must match
            before the archive is saved, such as "*caseCentered*.csv" to be
            sure that a link led to the right file. Defaults to `None`, in
            which case the download is not checked.

    Raises:
        OSError: if the download stopped before the whole file arrived.
        ValueError: if the scheme of `url` is not http or https, or the
            download is not a zip archive with a file that matches
            `required_member`.
        requests.HTTPError: if the server refuses the request (or whatever
            the `raise_for_status` method of `session`'s responses raises).

    Returns:
        The path of the file.

    """
    path = pathlib.Path(path).expanduser()
    if path.is_file() and not refresh:
        return path
    _check_scheme(url)
    path.parent.mkdir(parents = True, exist_ok = True)
    partial = path.with_name(f'{path.name}{_PARTIAL}')
    start = partial.stat().st_size if resume and partial.is_file() else 0
    headers = {'Range': f'bytes={start}-'} if start else {}
    seconds = options._DOWNLOAD_TIMEOUT if timeout is None else timeout
    response = open_session(session).get(
        url, headers = headers, stream = True, timeout = seconds)
    try:
        response.raise_for_status()
        # A server that does not send parts of files sends the whole file.
        partly = bool(start) and getattr(
            response, 'status_code', http.HTTPStatus.OK) == (
            http.HTTPStatus.PARTIAL_CONTENT)
        expected = _length(response)
        written = _write(response, partial, append = partly)
        _check_length(written, expected, url, resume = resume)
        if required_member is not None:
            _check_member(partial, required_member, url)
        partial.replace(path)
    except BaseException:
        if not resume:
            partial.unlink(missing_ok = True)
        raise
    finally:
        _close(response)
    logger.info('downloaded %s to %s', path.name, path.parent)
    return path


def download_linked_file(
    url: str,
    link: str,
    folder: pathlib.Path | str,
    *,
    file_name: str | None = None,
    refresh: bool = False,
    timeout: float | None = None,
    session: Any = None,
    required_member: str | None = None) -> pathlib.Path:
    """Downloads the file that a link on a web page names.

    Some sites link to the newest release of their data from a page, with
    the same words each time but a new address, so the file is found by the
    words of its link. If several links have those words, each one is tried
    in the order of the page until one leads to a zip archive with a file
    that matches `required_member`, or else the first one is used.

    Args:
        url: the address of the page.
        link: the words of the link, which are compared without regard to
            case or spacing (such as "CSV Organized by Supreme Court
            Citation").
        folder: the folder to save the file in. "~" is the user's home
            folder.
        file_name: name to save the file as, which may include folders.
            Defaults to `None`, in which case the last part of the path of
            the link's address is used.
        refresh: whether to download the file even if `folder` has it. With
            no `file_name`, the page is always read, to name the file.
            Defaults to `False`.
        timeout: seconds to wait for the server to answer. Defaults to
            `None`, which uses `options._DOWNLOAD_TIMEOUT`.
        session: an object with a `get` method like a `requests.Session`.
            Defaults to `None`, in which case `open_session` makes one.
        required_member: a pattern (with "*" and "?") that the name of a file
            in the downloaded zip archive must match (see `download_file`).
            Defaults to `None`.

    Raises:
        LookupError: if no link on the page has the words of `link`.
        ValueError: if no link with those words leads to a zip archive with
            a file that matches `required_member`.

    Returns:
        The path of the file.

    """
    folder = pathlib.Path(folder).expanduser()
    if file_name and (folder / file_name).is_file() and not refresh:
        return folder / file_name
    session = open_session(session)
    addresses = find_links(url, link, session = session, timeout = timeout)
    if not addresses:
        message = f'there is no link to {link!r} on {url}'
        raise LookupError(message)
    fetch = functools.partial(
        download_file,
        refresh = refresh,
        timeout = timeout,
        session = session,
        required_member = required_member)
    # Each address but the last is tried in turn, and the last one's error
    # (if every one fails) is the one raised.
    for address in addresses[:-1]:
        try:
            return fetch(address, folder / (file_name or _name_of(address)))
        except ValueError as error:
            logger.info('skipped %s: %s', address, error)
    last = addresses[-1]
    return fetch(last, folder / (file_name or _name_of(last)))


def find_links(
    url: str,
    words: str,
    *,
    session: Any = None,
    timeout: float | None = None) -> list[str]:
    """Returns the addresses of the links on a web page with chosen words.

    Args:
        url: the address of the page.
        words: the words of the links, which are compared without regard to
            case or spacing.
        session: an object with a `get` method like a `requests.Session`.
            Defaults to `None`, in which case `open_session` makes one.
        timeout: seconds to wait for the server to answer. Defaults to
            `None`, which uses `options._DOWNLOAD_TIMEOUT`.

    Returns:
        The full address of each link with those words, in the order of the
            page, without repeats.

    """
    wanted = ' '.join(words.split()).casefold()
    found = [
        address
        for address, text in page_links(url, session = session, timeout = timeout)
        if text.casefold() == wanted]
    return list(dict.fromkeys(found))


def open_session(session: Any = None) -> Any:
    """Returns a session that identifies `amos` to servers.

    Args:
        session: an object with a `get` method like a `requests.Session`.
            Defaults to `None`.

    Returns:
        `session`, or a new `requests.Session` that sends `_USER_AGENT` if it
            is `None`.

    """
    if session is None:
        session = requests.Session()
        session.headers.update({'User-Agent': _USER_AGENT})
    return session


def page_links(
    url: str,
    *,
    session: Any = None,
    timeout: float | None = None) -> list[tuple[str, str]]:
    """Returns the address and words of each link on a web page.

    Args:
        url: the address of the page.
        session: an object with a `get` method like a `requests.Session`.
            Defaults to `None`, in which case `open_session` makes one.
        timeout: seconds to wait for the server to answer. Defaults to
            `None`, which uses `options._DOWNLOAD_TIMEOUT`.

    Raises:
        ValueError: if the scheme of `url` is not http or https.

    Returns:
        The full address (relative ones are joined to `url`) and the words of
            each link, in the order of the page. The words are on one line,
            with single spaces.

    """
    _check_scheme(url)
    seconds = options._DOWNLOAD_TIMEOUT if timeout is None else timeout
    response = open_session(session).get(url, timeout = seconds)
    try:
        response.raise_for_status()
        markup = response.text
    finally:
        _close(response)
    extractor = _LinkExtractor()
    extractor.feed(markup)
    extractor.close()
    return [
        (urllib.parse.urljoin(url, address), ' '.join(''.join(words).split()))
        for address, words in extractor.links]


""" Private Functions """


def _check_length(
    written: int,
    expected: int | None,
    url: str,
    *,
    resume: bool) -> None:
    """Checks that a download wrote as many bytes as the server said.

    Args:
        written: the number of bytes written.
        expected: the number that the server said it would send, or `None`
            if it did not say.
        url: the address of the file, for the message.
        resume: whether the download can be resumed, for the message.

    Raises:
        OSError: if fewer (or more) bytes were written.

    """
    if expected is not None and written != expected:
        message = (
            f'the download of {url} stopped after {written} of {expected} '
            f'bytes')
        if resume:
            message += ': download it again to resume'
        raise OSError(message)


def _check_member(path: pathlib.Path, pattern: str, url: str) -> None:
    """Checks that a zip archive has a file whose name matches `pattern`.

    Args:
        path: path of the downloaded file.
        pattern: a pattern with "*" and "?", as in `fnmatch`.
        url: where the file came from, for the message.

    Raises:
        ValueError: if the file is not a zip archive, or none of its files
            match.

    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        names = None
    if names is None or not any(
        fnmatch.fnmatchcase(name, pattern) for name in names):
        holds = 'is not a zip archive' if names is None else f'holds {names}'
        message = (
            f'the file at {url} should be a zip archive with a file named '
            f'like {pattern!r}, but it {holds}')
        raise ValueError(message)


def _check_scheme(url: str) -> None:
    """Checks that `url` can be downloaded.

    Args:
        url: a URL.

    Raises:
        ValueError: if its scheme is not http or https.

    """
    if urllib.parse.urlparse(url).scheme not in _SCHEMES:
        message = f'only http and https URLs can be downloaded, not {url!r}'
        raise ValueError(message)


def _close(response: Any) -> None:
    """Closes a response, if it can be closed.

    Args:
        response: a response from a session.

    """
    close = getattr(response, 'close', None)
    if callable(close):
        close()


def _length(response: Any) -> int | None:
    """Returns the number of bytes that a response says it sends.

    Args:
        response: a response from a session.

    Returns:
        Its "Content-Length", or `None` if it does not say (or the content
            is compressed for the trip, so that the length is not that of
            the file).

    """
    headers = getattr(response, 'headers', None) or {}
    if headers.get('Content-Encoding'):
        return None
    try:
        return int(headers['Content-Length'])
    except (KeyError, TypeError, ValueError):
        return None


def _name_of(url: str) -> str:
    """Returns the name of the file at `url`.

    Args:
        url: a URL.

    Raises:
        ValueError: if the URL's path does not end in a name.

    Returns:
        The last part of the URL's path.

    """
    path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
    name = pathlib.PurePosixPath(path).name
    if not name:
        message = (
            f'{url!r} does not end in a file name: set "file_name" (such as '
            f'"data.csv")')
        raise ValueError(message)
    return name


def _write(response: Any, path: pathlib.Path, *, append: bool) -> int:
    """Writes the content of a response to a file.

    Args:
        response: a response from a session. Its content is read a piece at
            a time with `iter_content` if it has that method, and otherwise
            from `content`.
        path: the file to write.
        append: whether to add to the end of the file instead of replacing
            it.

    Returns:
        The number of bytes written.

    """
    pieces = (
        response.iter_content(chunk_size = _CHUNK)
        if hasattr(response, 'iter_content') else [response.content])
    written = 0
    with path.open('ab' if append else 'wb') as file:
        for piece in pieces:
            if piece:
                file.write(piece)
                written += len(piece)
    return written
