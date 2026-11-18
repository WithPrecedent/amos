"""Reading chosen rows and columns of huge CSV files, a chunk at a time.

Some data comes as CSV files far too large to load at once, such as the bulk
files of court records (up to 55 GB, compressed with bzip2), of which a study
needs only a few columns and the rows of a few courts or years. `read_chunks`
reads such a file a block of bytes at a time with pyarrow's CSV reader, which
parses each block on every core, and keeps only the rows and columns asked
for, so the text of the rows that are not wanted never becomes Python
objects. A file compressed with bzip2 is decompressed on every core too, with
indexed_bzip2. `load_chunks` is the loader that does the same in a workflow.

Contents:
    LoadChunks: loads chosen rows and columns of a huge CSV file, a chunk at
        a time.
    read_chunks: yields chosen rows and columns of a huge CSV file, a chunk at
        a time.
    logger: the module's logger, which reports the progress of reading.
    _DECOMPRESSORS: functions that decompress a stream (such as a download)
        of a file compressed in the format of each extension.

"""

from __future__ import annotations

import bz2
import contextlib
import csv
import dataclasses
import gzip
import io
import logging
import lzma
import os
import pathlib
import urllib.parse
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import IO, Any, cast

import indexed_bzip2  # type: ignore[import-not-found]
import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
from pyarrow import csv as arrow_csv

from ...core import options
from . import _web
from ._base import Loader, _find

logger = logging.getLogger(__name__)

# Functions that decompress a stream (such as a download) of a file
# compressed in the format of each extension. A file on disk compressed with
# bzip2 is decompressed with indexed_bzip2 instead, on every core. pyarrow's
# streams are buffered by Python, so that the line of column names can be
# read from them.
_DECOMPRESSORS: dict[str, Callable[..., Any]] = {
    '.bz2': bz2.open,
    '.gz': gzip.open,
    '.xz': lzma.open,
    '.zst': lambda stream: io.BufferedReader(
        pa.input_stream(stream, compression = 'zstd'))}


@dataclasses.dataclass
class LoadChunks(Loader):
    """Loads chosen rows and columns of a huge CSV file, a chunk at a time.

    The file is read as `read_chunks` reads it (and takes its parameters):
    a block of bytes at a time, keeping only the columns in "columns" and
    the rows whose values are in "keep" (such as `{"court_id": ["ca1",
    "ca2"]}`) and that match "where" (such as `"date_filed >=
    '2010-01-01'"`), so a file of many gigabytes becomes a table of the rows
    that a study needs. The source is the path of the file, or its URL, in
    which case it is downloaded to the clerk's input folder first (and kept,
    so that it is only downloaded once) unless "stream" is true. Every value
    is read as text, and the rows are numbered from 0.

    Wraps:
        [`pyarrow.csv.open_csv`](https://arrow.apache.org/docs/python/generated/pyarrow.csv.open_csv.html)
        from pyarrow, and
        [`indexed_bzip2`](https://github.com/mxmlnkn/indexed_bzip2) for
        files compressed with bzip2.

    """

    def load(
        self,
        source: Any,
        *,
        stream: bool = False,
        file_name: str | None = None,
        refresh: bool = False,
        resume: bool = True,
        timeout: float | None = None,
        session: Any = None,
        limit: int | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Loads the chosen rows and columns of the file at `source`.

        Args:
            source: the path of a CSV file (which may be compressed), or its
                URL (http or https). A relative path is looked for in the
                current folder and then in the clerk's input folder.
            stream: whether to read a URL's file as it arrives instead of
                downloading it first. A stream needs no disk space, but
                starts over if the connection fails, and a file compressed
                with bzip2 is then decompressed on one core. Defaults to
                `False`.
            file_name: name to save a URL's file as in the clerk's input
                folder. Defaults to `None`, in which case the last part of
                the URL's path is used.
            refresh: whether to download a URL's file again even if it was
                downloaded before. Defaults to `False`.
            resume: whether to continue a download that stopped instead of
                starting over. Defaults to `True`, since the files that need
                this loader take a long time to download.
            timeout: seconds to wait for the server of a URL to answer.
                Defaults to `None`, which uses `options._DOWNLOAD_TIMEOUT`.
            session: an object with a `get` method like a `requests.Session`.
                Defaults to `None`, in which case one that identifies `amos`
                is made.
            limit: most rows to load, such as to try out a workflow on a few.
                Reading stops once there are that many. Defaults to `None`,
                which loads every row that is kept.
            **kwargs: parameters for `read_chunks`, such as "columns",
                "keep", "where", the dialect of the file (such as
                "escape_char"), and "block_size".

        Returns:
            The rows that were kept, numbered from 0, with every value as
                text.

        """
        text = str(source).strip()
        clerk = self._get_clerk()
        folder = pathlib.Path(clerk.input_folder)
        target: pathlib.Path | str
        if urllib.parse.urlparse(text).scheme in _web._SCHEMES:
            if stream:
                target = text
            else:
                target = _web.download_file(
                    text,
                    folder / (file_name or _web._name_of(text)),
                    refresh = refresh,
                    timeout = timeout,
                    session = session,
                    resume = resume)
        else:
            target = _find(pathlib.Path(text).expanduser(), folder)
        frames: list[pd.DataFrame] = []
        count = 0
        chunks = read_chunks(
            target, session = session, timeout = timeout, **kwargs)
        with contextlib.closing(chunks):
            for frame in chunks:
                if limit is not None and count + len(frame) >= limit:
                    frames.append(frame.iloc[:limit - count])
                    break
                frames.append(frame)
                count += len(frame)
        if not frames:
            return pd.DataFrame(columns = list(kwargs.get('columns') or []))
        return pd.concat(frames, ignore_index = True)


""" Public Functions """


def read_chunks(
    source: pathlib.Path | str,
    *,
    columns: Sequence[str] | None = None,
    keep: Mapping[str, Any] | None = None,
    where: str | Callable[[pd.DataFrame], Any] | None = None,
    delimiter: str = ',',
    quote_char: str | None = '"',
    double_quote: bool = True,
    escape_char: str | None = None,
    newlines_in_values: bool = True,
    encoding: str = 'utf-8',
    empty_as_missing: bool = False,
    block_size: int | None = None,
    processes: int | None = None,
    session: Any = None,
    timeout: float | None = None,
    report_every: int | None = None) -> Generator[pd.DataFrame, None, None]:
    r"""Yields chosen rows and columns of a huge CSV file, a chunk at a time.

    The file is read a block of bytes at a time ("block_size"), and each
    block is parsed on every core by pyarrow. The rows whose values are in
    `keep` are found in each block before any of its text becomes Python
    objects, so a file of many gigabytes takes about as long to read as to
    decompress. A file compressed with bzip2 (".bz2") is decompressed on
    every core with indexed_bzip2; one compressed with gzip, xz, or
    Zstandard (".gz", ".xz", or ".zst") is decompressed as it is read. A URL
    is read as it arrives, without saving it (see `download_file` to save it
    first).

    Every value is read as text, so that ids and dates keep their exact
    form. Convert columns afterward (such as with `convert_types` or
    `clean_numbers`).

    Args:
        source: the path of a CSV file (which may be compressed) or its URL
            (http or https). Its first line names its columns. "~" is the
            user's home folder.
        columns: the columns to return, in that order. Those that the file
            does not have are left out (with a warning in the log), since
            the columns of a published file can change from one release to
            the next. Defaults to `None`, which returns every column.
        keep: the values to keep of each of some columns, such as
            `{"court_id": ["ca1", "ca2"]}`: a row is kept only if each of
            these columns has one of its values. Values are compared as text
            (so 5 matches "5"), and one value need not be in a list. The
            columns need not be in `columns`. Defaults to `None`, which keeps
            every row.
        where: which of the rows (after `keep`) to keep: an expression for
            `pandas.DataFrame.query`, such as `"date_filed >= '2010-01-01'"`,
            or a function that takes a chunk and returns a boolean value
            for each of its rows. It sees the columns in `columns`. Defaults
            to `None`, which keeps every row.
        delimiter: the character between values. Defaults to ",".
        quote_char: the character that quotes a value, or `None` if values
            are not quoted. Defaults to '"'.
        double_quote: whether two quote characters in a quoted value stand
            for one. Defaults to `True`.
        escape_char: the character that makes the next one stand for itself
            (such as "\\", which PostgreSQL writes before quotes in values),
            or `None` for none. Defaults to `None`.
        newlines_in_values: whether quoted values can hold line breaks.
            Defaults to `True`.
        encoding: the encoding of the file. Defaults to "utf-8".
        empty_as_missing: whether empty values are missing rather than empty
            text. Defaults to `False`.
        block_size: how many bytes are parsed at a time, which must be more
            than the longest row (a few rows of court records hold megabytes
            of text). Defaults to `None`, which uses `options._BLOCK_SIZE`.
        processes: how many cores decompress a file compressed with bzip2.
            Defaults to `None`, which uses every core.
        session: an object with a `get` method like a `requests.Session`,
            for a URL. Defaults to `None`, in which case one that identifies
            `amos` is made.
        timeout: seconds to wait for the server of a URL to answer. Defaults
            to `None`, which uses `options._DOWNLOAD_TIMEOUT`.
        report_every: how many rows are read between the reports of progress
            in the log (at the "info" level). Defaults to `None`, which uses
            `options._REPORT_EVERY`.

    Raises:
        KeyError: if a column of `keep` is not in the file.
        ValueError: if a row is longer than `block_size`, or the file has no
            line of column names.

    Yields:
        pd.DataFrame: the rows of each block that are kept (blocks without
            any are skipped), with every value as text. Each row's label is
            its place in the file (0 for the first row after the column
            names).

    """
    name = _name_of(source)
    every = options._REPORT_EVERY if report_every is None else report_every
    wanted = _values_to_keep(keep)
    with _open(
        source,
        processes = processes,
        session = session,
        timeout = timeout) as raw:
        header = _header(
            raw,
            delimiter = delimiter,
            quote_char = quote_char,
            double_quote = double_quote,
            escape_char = escape_char,
            encoding = encoding)
        missing = [column for column in wanted if column not in header]
        if missing:
            message = f'{name} has no column {missing}: it has {header}'
            raise KeyError(message)
        output = _columns_in(header, columns, name)
        read = list(dict.fromkeys([*output, *wanted]))
        reader = arrow_csv.open_csv(
            raw,
            read_options = arrow_csv.ReadOptions(
                column_names = header,
                block_size = block_size or options._BLOCK_SIZE,
                use_threads = True,
                encoding = encoding),
            parse_options = arrow_csv.ParseOptions(
                delimiter = delimiter,
                quote_char = quote_char or False,
                double_quote = double_quote,
                escape_char = escape_char or False,
                newlines_in_values = newlines_in_values),
            convert_options = arrow_csv.ConvertOptions(
                include_columns = read,
                column_types = dict.fromkeys(read, pa.large_string()),
                strings_can_be_null = empty_as_missing,
                quoted_strings_can_be_null = empty_as_missing))
        count = 0
        for batch in _batches(reader, name):
            start, count = count, count + batch.num_rows
            if every and start // every != count // every:
                logger.info('%s: %d rows read', name, count)
            frame = _chosen(batch, wanted, output, start)
            if where is not None and len(frame):
                frame = _where(frame, where)
            if len(frame):
                yield frame
        logger.info('%s: read all %d rows', name, count)


""" Private Functions """


def _batches(reader: Any, name: str) -> Iterator[Any]:
    """Yields the parts of a file that pyarrow reads, explaining its errors.

    Args:
        reader: pyarrow's reader of a CSV file.
        name: the name of the file, for messages.

    Raises:
        ValueError: if a row is longer than the size of a block.

    Yields:
        Any: each part of the file, as a `pyarrow.RecordBatch`.

    """
    while True:
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            return
        except pa.ArrowInvalid as error:
            if 'straddling' not in str(error):
                raise
            message = (
                f'a row of {name} is longer than a block: set "block_size" '
                f'to more bytes than the longest row ({error})')
            raise ValueError(message) from error
        yield batch


def _chosen(
    batch: Any,
    wanted: Mapping[str, Any],
    columns: Sequence[str],
    start: int) -> pd.DataFrame:
    """Returns the rows of a part of a file whose values are in `wanted`.

    Args:
        batch: the part of the file.
        wanted: the values to keep of some columns.
        columns: the columns to return.
        start: the place in the file of the first row of `batch`.

    Returns:
        The rows that are kept, labeled by their places in the file.

    """
    places: Any = None
    if wanted:
        mask = None
        for column, values in wanted.items():
            found = pc.is_in(batch.column(column), value_set = values)
            mask = found if mask is None else pc.and_(mask, found)
        places = pc.indices_nonzero(pc.fill_null(mask, fill_value = False))
        batch = batch.take(places)
    frame: pd.DataFrame = batch.select(list(columns)).to_pandas()
    positions = (
        np.arange(len(frame)) if places is None
        else places.to_numpy(zero_copy_only = False))
    frame.index = pd.Index(positions + start)
    return frame


def _columns_in(
    header: Sequence[str],
    columns: Sequence[str] | None,
    name: str) -> list[str]:
    """Returns the columns to return that the file has.

    Args:
        header: the columns of the file.
        columns: the columns asked for, or `None` for every column.
        name: the name of the file, for the log.

    Returns:
        The columns, in the order asked for.

    """
    if columns is None:
        return list(header)
    if isinstance(columns, str):
        columns = [columns]
    absent = [column for column in columns if column not in header]
    if absent:
        logger.warning('%s has no column %s, so it is left out', name, absent)
    return [column for column in dict.fromkeys(columns) if column in header]


def _header(
    raw: IO[bytes],
    *,
    delimiter: str,
    quote_char: str | None,
    double_quote: bool,
    escape_char: str | None,
    encoding: str) -> list[str]:
    """Reads the line of column names at the start of a file.

    Only that line is read, so the rest of the file is left for pyarrow.

    Args:
        raw: the decompressed file, at its start.
        delimiter: the character between values.
        quote_char: the character that quotes a value, or `None`.
        double_quote: whether two quote characters stand for one.
        escape_char: the escape character, or `None`.
        encoding: the encoding of the file.

    Raises:
        ValueError: if the file is empty.

    Returns:
        The names of the columns.

    """
    line = raw.readline().decode(encoding).removeprefix('﻿')
    names = next(csv.reader(
        [line.rstrip('\r\n')],
        delimiter = delimiter,
        quotechar = quote_char,
        quoting = csv.QUOTE_MINIMAL if quote_char else csv.QUOTE_NONE,
        doublequote = double_quote,
        escapechar = escape_char), [])
    if not names:
        message = 'the file has no line of column names'
        raise ValueError(message)
    return names


def _name_of(source: pathlib.Path | str) -> str:
    """Returns the name of a file, for the log and messages.

    Args:
        source: a path or URL.

    Returns:
        The last part of its path.

    """
    text = str(source)
    if urllib.parse.urlparse(text).scheme in _web._SCHEMES:
        text = urllib.parse.urlparse(text).path
    return pathlib.PurePosixPath(text.replace('\\', '/')).name or text


@contextlib.contextmanager
def _open(
    source: pathlib.Path | str,
    *,
    processes: int | None,
    session: Any,
    timeout: float | None) -> Iterator[IO[bytes]]:
    """Opens a file, or a URL's file, as decompressed bytes.

    Args:
        source: a path or a URL (http or https).
        processes: how many cores decompress a file compressed with bzip2,
            or `None` for every core.
        session: an object with a `get` method like a `requests.Session`,
            or `None`.
        timeout: seconds to wait for the server of a URL, or `None`.

    Yields:
        IO[bytes]: the decompressed file. It (and the connection to a server)
            is closed when the `with` statement ends.

    """
    text = str(source)
    if urllib.parse.urlparse(text).scheme in _web._SCHEMES:
        suffix = pathlib.PurePosixPath(
            urllib.parse.urlparse(text).path).suffix.lower()
        seconds = options._DOWNLOAD_TIMEOUT if timeout is None else timeout
        response = _web.open_session(session).get(
            text, stream = True, timeout = seconds)
        try:
            response.raise_for_status()
            raw = response.raw
            # A server may compress what it sends (apart from the file's
            # own compression), which `requests` undoes only if asked.
            if hasattr(raw, 'decode_content'):
                raw.decode_content = True
            opener = _DECOMPRESSORS.get(suffix)
            with contextlib.closing(
                opener(raw) if opener else raw) as opened:
                yield cast('IO[bytes]', opened)
        finally:
            _web._close(response)
        return
    path = pathlib.Path(text).expanduser()
    suffix = path.suffix.lower()
    if suffix == '.bz2':
        cores = processes or os.cpu_count() or 1
        file: Any = indexed_bzip2.open(str(path), parallelization = cores)
    elif suffix in _DECOMPRESSORS:
        file = _DECOMPRESSORS[suffix](str(path))
    else:
        file = path.open('rb')
    with contextlib.closing(file):
        yield cast('IO[bytes]', file)


def _values_to_keep(keep: Mapping[str, Any] | None) -> dict[str, Any]:
    """Returns the values to keep of each column, as pyarrow text.

    Args:
        keep: the values to keep of each of some columns. A value that is
            text (or not iterable) stands for itself.

    Returns:
        The values of each column, as an array of text without repeats.

    """
    wanted: dict[str, Any] = {}
    for column, given in (keep or {}).items():
        values = (
            [given] if isinstance(given, str | bytes)
            or not isinstance(given, Iterable) else given)
        texts = list(dict.fromkeys(str(value) for value in values))
        wanted[str(column)] = pa.array(texts, type = pa.large_string())
    return wanted


def _where(
    frame: pd.DataFrame,
    where: str | Callable[[pd.DataFrame], Any]) -> pd.DataFrame:
    """Returns the rows of a chunk that `where` keeps.

    Args:
        frame: the chunk.
        where: an expression for `DataFrame.query`, or a function that
            returns a boolean value for each row.

    Returns:
        The rows that are kept.

    """
    if isinstance(where, str):
        return frame.query(where)
    # A missing value (such as a comparison with missing text) keeps no row.
    found = pd.Series(where(frame)).fillna(value = False)
    return frame.loc[np.asarray(found, dtype = bool)]
