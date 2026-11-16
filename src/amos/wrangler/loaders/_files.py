"""Loaders that read data files.

Contents:
    _GLOB_CHARACTERS: characters that make a source of `load_files` a pattern of
        paths.
    LoadFile: loads a data file.
    LoadFiles: loads a batch of files into one table.

"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

if TYPE_CHECKING:
    pass
from ._base import Loader, _find

# Characters that make a source of `load_files` a pattern of paths.
_GLOB_CHARACTERS: frozenset[str] = frozenset('*?[')


@dataclasses.dataclass
class LoadFile(Loader):
    """Loads a data file in any format that the clerk knows.

    Wraps:
        [`nagata.FileManager`](https://github.com/WithPrecedent/nagata)
        from nagata, which reads each format with pandas (such as
        `pandas.read_csv`).

    """

    def load(self, source: Any, **kwargs: Any) -> Any:
        """Loads the file at `source`.

        Args:
            source: path of the file. A relative path is looked for in the
                current folder and then in the clerk's input folder.
            **kwargs: parameters for `read`, such as "file_format", "member",
                and parameters for the reader (such as "sep").

        Returns:
            The loaded data.

        """
        return self.read(source, **kwargs)


@dataclasses.dataclass
class LoadFiles(Loader):
    """Loads a batch of files into one table.

    The source is a folder, a pattern of paths (such as "reports/*.txt"), or
    a list of paths. Each file is read with the clerk, as `load_file` reads
    one, so its format comes from its name unless "file_format" names one for
    every file. A file of a table (such as a CSV file) adds its rows, and a
    file of text (a ".txt" file, or any file read with "file_format" set to
    "text") adds one row, with its text in the "text" column. Either way,
    each row has the path of its file in the "file" column, so a batch of
    documents becomes a table that parsers can work on all at once. The
    files are read at the same time, in threads, and their rows are kept in
    the order of their paths.

    Wraps:
        [`nagata.FileManager`](https://github.com/WithPrecedent/nagata)
        from nagata, which reads each format with pandas (such as
        `pandas.read_csv`).

    """

    def load(
        self,
        source: Any,
        *,
        pattern: str = '*',
        recursive: bool = True,
        file_column: str = 'file',
        text_column: str = 'text',
        limit: int | None = None,
        workers: int | None = None,
        **kwargs: Any) -> pd.DataFrame:
        """Loads the files at `source` into one table.

        Args:
            source: a folder, a pattern of paths (such as "reports/*.txt"),
                or a list of paths of files. A relative path or pattern is
                looked for in the current folder and then in the clerk's
                input folder.
            pattern: the pattern of the names of the files to load from a
                folder (such as "*.txt"). Defaults to "*", which loads every
                file. Not used for other sources.
            recursive: whether to load the files of a folder's subfolders
                too. Defaults to `True`. Files and folders whose names start
                with "." are left out.
            file_column: name of the column with the path of each row's file:
                relative to the folder (or to the folder in which a pattern
                was found), or else as given, with "/" between its parts.
                Defaults to "file".
            text_column: name of the column with the text of each file of
                text. Defaults to "text".
            limit: most files to load (the first, in the order of their
                paths), such as to try out parsers on a few. Defaults to
                `None`, which loads every file.
            workers: most threads that read files at once. Defaults to `None`,
                which lets Python choose (from the number of processors).
            **kwargs: parameters for `read`, such as "file_format" and
                parameters for the reader (such as "sep").

        Raises:
            FileNotFoundError: if there are no files to load.
            TypeError: if a file holds neither a table nor text.
            ValueError: if some files are tables and others are text.

        Returns:
            The rows of every file, numbered from 0, with `file_column`
                first.

        """
        clerk = self._get_clerk()
        found = _find_files(
            source,
            pathlib.Path(clerk.input_folder),
            pattern = pattern,
            recursive = recursive)
        if limit is not None:
            found = found[:limit]
        if not found:
            message = f'there are no files to load at {source!r}'
            raise FileNotFoundError(message)
        paths = [path for path, _ in found]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers = workers) as pool:
            loaded = list(pool.map(
                functools.partial(self.read, **kwargs), paths))
        return _combine(
            [name for _, name in found],
            loaded,
            file_column = file_column,
            text_column = text_column)


""" Private Functions """


def _combine(
    names: list[str],
    loaded: list[Any],
    *,
    file_column: str,
    text_column: str) -> pd.DataFrame:
    """Returns the contents of a batch of files as one table.

    Args:
        names: the name of each file, for its rows' `file_column`.
        loaded: the contents of each file, in the same order: a table or
            text.
        file_column: name of the column with each row's file.
        text_column: name of the column with the text of a file of text.

    Raises:
        TypeError: if a file holds neither a table nor text.
        ValueError: if some files are tables and others are text.

    Returns:
        The rows of every file, numbered from 0, with `file_column` first.

    """
    for name, contents in zip(names, loaded, strict = True):
        if not isinstance(contents, pd.DataFrame | str):
            message = (
                f'{name!r} holds {type(contents).__name__}, not a table or '
                f'text: set "file_format" (such as "text") to read it as one'
            )
            raise TypeError(message)
    texts = [isinstance(c, str) for c in loaded]
    if all(texts):
        return pd.DataFrame(
            {file_column: names, text_column: loaded}).astype('str')
    if any(texts):
        mixed = [n for n, t in zip(names, texts, strict = True) if t][:3]
        message = (
            f'some files are tables and others (such as {mixed}) are text: '
            f'load them separately, or set "pattern" to choose one kind'
        )
        raise ValueError(message)
    tables = [
        table.assign(**{file_column: name})
        for name, table in zip(names, loaded, strict = True)]
    data = pd.concat(tables, ignore_index = True)
    order = [file_column, *[c for c in data.columns if c != file_column]]
    return cast('pd.DataFrame', data[order])


def _find_files(
    source: Any,
    folder: pathlib.Path,
    *,
    pattern: str,
    recursive: bool) -> list[tuple[pathlib.Path, str]]:
    """Returns the files that `load_files` loads, in the order of their paths.

    Args:
        source: a folder, a pattern of paths, or a sequence of paths of
            files. A relative path or pattern is looked for in the current
            folder and then in `folder`.
        folder: the clerk's input folder.
        pattern: the pattern of the names of the files of a folder.
        recursive: whether to look in the subfolders of a folder too.

    Raises:
        FileNotFoundError: if a path in a sequence is not a file, or `source`
            is not a folder, a pattern, or a file.

    Returns:
        The path of each file, with its name for the table: its path
            relative to the folder of a folder, or to the folder in which a
            pattern was found, and otherwise its path as given, with "/"
            between its parts. Files and folders whose names start with "."
            are left out of a folder or pattern.

    """
    if isinstance(source, Sequence) and not isinstance(source, str):
        return [
            (_find(pathlib.Path(p), folder), pathlib.Path(p).as_posix())
            for p in source]
    text = str(source)
    if _GLOB_CHARACTERS & set(text):
        base = pathlib.Path()
        if not any(base.glob(text)):
            base = folder
        return [
            (p, p.relative_to(base).as_posix()) for p in sorted(base.glob(text))
            if p.is_file() and not _is_hidden(p.relative_to(base))]
    path = pathlib.Path(text)
    directory = path if path.is_dir() else folder / path
    if directory.is_dir():
        search = directory.rglob if recursive else directory.glob
        return [
            (p, p.relative_to(directory).as_posix())
            for p in sorted(search(pattern))
            if p.is_file() and not _is_hidden(p.relative_to(directory))]
    return [(_find(path, folder), path.as_posix())]


def _is_hidden(path: pathlib.Path) -> bool:
    """Returns whether a part of `path` starts with "." (and so is hidden).

    Args:
        path: a path, relative to the folder it was found in.

    Returns:
        Whether any part of the path other than "." and ".." starts with ".".

    """
    return any(
        part.startswith('.') and part not in {'.', '..'} for part in path.parts)
