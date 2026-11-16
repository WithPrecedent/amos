"""Techniques that load data, usually at the start of the "wrangler" stage.

A loader makes the data of a dataset from a source: a data file, a batch of
files, or a file (or OpenML dataset) that it downloads first. Loaders only
download and import: whatever a file holds, they bring it into the dataset
as it is, and parsers, cleaners, and mungers then work on it there. Every
loader works through the project's clerk (the `nagata.FileManager` that is
the `clerk` of a `Project`): files are looked for, and downloads are saved,
in the clerk's input folder, and the clerk loads every format that it knows.
A project whose workflow has a loader needs no `item`.

Contents:
    Loader: base class for techniques that load data.
    Download: downloads a file from a URL, or a dataset from OpenML, and
        loads it.
    LoadFile: loads a data file.
    LoadFiles: loads a batch of files into one table.

"""

from __future__ import annotations

import abc
import concurrent.futures
import dataclasses
import functools
import pathlib
import shutil
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import nagata
import pandas as pd

from . import base, clerks, options, utilities

if TYPE_CHECKING:
    import chrisjen

# Characters that make a source of `load_files` a pattern of paths.
_GLOB_CHARACTERS: frozenset[str] = frozenset('*?[')
# Extensions of compressed files, which `pandas` opens itself. They are
# skipped when the format of a file is found from its name.
_COMPRESSIONS: frozenset[str] = frozenset({
    'bz2', 'gz', 'tar', 'xz', 'zip', 'zst'})
# Formats of the clerk that are read as plain text, in UTF-8 unless an
# "encoding" is given.
_TEXT_FORMATS: frozenset[str] = frozenset({'text'})
# Settings in the "general" section that are the default parameters of a
# loader built from a project's settings.
_GENERAL: tuple[str, ...] = ('label', 'task', 'groups')
# The function that `download` uses to fetch OpenML datasets, unless its
# `contents` names another.
_OPENML: str = 'sklearn.datasets.fetch_openml'
# Schemes of the URLs that `download` accepts.
_SCHEMES: frozenset[str] = frozenset({'http', 'https'})
# Sent with each download, since some servers refuse requests without one.
_USER_AGENT: str = 'amos (https://github.com/WithPrecedent/amos)'


@dataclasses.dataclass
class Loader(base.Operation, abc.ABC):
    """Base class for techniques that load data.

    A subclass writes a `load` method, which takes a source (such as a path or
    a URL) and returns the data. Most loaders get a file and pass it to
    `read`, which loads it with the clerk. The source is the "source"
    parameter, usually set in the "{name}_parameters" section of the settings.

    A loader starts the dataset that the rest of the workflow works on. The
    new dataset keeps the seed and history of the dataset that the loader is
    applied to, and its label and groups unless the loader's parameters set
    them. Everything else, including the task (unless "task" is set), comes
    from the loaded data. A loader
    built from a project's settings uses the project's clerk, and the
    "label", "task", and "groups" of the "general" section are its default
    parameters.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: the tool that the loader wraps, if any. Defaults to `None`.
        parameters: keyword arguments for `load`, and "source", "label",
            "task", and "groups". Defaults to an empty `dict`.
        clerk: the file manager that finds, saves, and loads files. Defaults
            to `None`, in which case one for the current folder is made when
            it is needed.

    """

    clerk: nagata.FileManager | None = dataclasses.field(
        default = None, repr = False)

    """ Required Methods """

    @abc.abstractmethod
    def load(self, source: Any, **kwargs: Any) -> Any:
        """Returns the data at `source`.

        Args:
            source: where the data is, such as a path or a URL.
            **kwargs: parameters for loading.

        Returns:
            The data: a `pandas.DataFrame` or anything else that
                `Dataset.create` accepts.

        """

    """ Class Methods """

    @classmethod
    def build(
        cls,
        name: str,
        project: chrisjen.Project,
        parameters: base.GenericDict | None = None,
        **kwargs: Any) -> Loader:
        """Builds a loader with the clerk and "general" settings of `project`.

        Args:
            name: name of the loader.
            project: the project the loader belongs to.
            parameters: more parameters, which take precedence over those in
                the settings. Defaults to `None`.
            **kwargs: other arguments for the constructor.

        Returns:
            The built loader.

        """
        kwargs.setdefault('clerk', project.clerk)
        loader = super().build(name, project, parameters, **kwargs)
        general = project.idea.get('general', {})
        for key in _GENERAL:
            if key in general:
                loader.parameters.setdefault(key, general[key])
        return loader

    """ Public Methods """

    def apply(self, item: base.Dataset, **kwargs: Any) -> base.Dataset:
        """Loads the data into a dataset that continues `item`.

        Args:
            item: the dataset so far.
            **kwargs: "source", any "label", "task", and "groups" of the data,
                and parameters for `load`.

        Raises:
            ValueError: if there is no "source".

        Returns:
            The loaded data, with the seed, history, and switches (such as
                `use_gpu` and `visualize_critics`) of `item`.

        """
        source = kwargs.pop('source', None)
        if source is None:
            message = (
                f'{self.name!r} has nothing to load: set its "source" in the '
                f'"{self.name}_parameters" section of the settings'
            )
            raise ValueError(message)
        label = kwargs.pop('label', None) or item.label
        # The task of `item` is not kept, since it may not suit the new data:
        # unless "task" is set, it is inferred from the label.
        task = kwargs.pop('task', None)
        groups = kwargs.pop('groups', None) or item.groups or None
        loaded = base.Dataset.create(
            self.load(source, **kwargs),
            label = label,
            task = task,
            seed = item.seed,
            groups = groups,
            use_gpu = item.use_gpu,
            visualize_explorers = item.visualize_explorers,
            visualize_critics = item.visualize_critics)
        loaded.history = item.history
        rows, columns = loaded.data.shape
        loaded.record(
            self.name, source = str(source), rows = rows, columns = columns)
        return loaded

    def read(
        self,
        path: pathlib.Path | str,
        *,
        file_format: str | None = None,
        member: str | None = None,
        **kwargs: Any) -> Any:
        """Loads the file at `path` with the clerk.

        A relative path is looked for in the current folder and then in the
        clerk's input folder. The format comes from the file's extension
        (skipping an extension of compression, such as ".gz", which `pandas`
        opens itself) unless `file_format` names it. Parameters are passed to
        the reader (such as `pandas.read_csv`) if it accepts them. A file of
        plain text (the "text" format) is read in UTF-8 unless "encoding"
        names another (and "errors" can say what to do with bytes that are
        not in it, such as "replace").

        Args:
            path: path of the file.
            file_format: name of the clerk's format for the file (such as
                "csv", "excel", or "parquet"). Defaults to `None`, in which
                case it comes from the file's extension.
            member: the file to load from a zip archive, which is extracted
                to a folder named for the archive. Defaults to `None`, which
                loads `path` itself.
            **kwargs: parameters for the reader.

        Raises:
            ValueError: if the format of the file is not known.

        Returns:
            The loaded data.

        """
        clerk = self._get_clerk()
        found = _find(pathlib.Path(path), pathlib.Path(clerk.input_folder))
        if member is not None:
            found = _extract(found, str(member))
        name = file_format or _format_of(found, clerk)
        formats = clerk.framework.formats
        if name not in formats:
            message = (
                f'{name!r} is not a format that the clerk knows: use one of '
                f'{sorted(map(str, formats))}'
            )
            raise ValueError(message)
        kind = formats[name]
        parameters = utilities.accepted_parameters(
            _reader_of(kind), {**_settings_for(kind), **kwargs})
        if name in _TEXT_FORMATS:
            # The format reads with `open`, whose default encoding depends on
            # the computer, and takes no parameters from the clerk.
            parameters.setdefault('encoding', kwargs.get(
                'encoding', options._FILE_SETTINGS['file_encoding']))
            if 'errors' in kwargs:
                parameters.setdefault('errors', kwargs['errors'])
        return clerk.load(file_path = found, file_format = kind, **parameters)

    """ Private Methods """

    def _call_apply(
        self,
        method: Callable[..., Any],
        item: Any,
        /,
        **kwargs: Any) -> base.Dataset:
        """Calls `apply`, starting a new dataset if `item` is `None`.

        Args:
            method: the `apply` method, before it was wrapped.
            item: the dataset so far, anything that `Dataset.create` accepts,
                or `None`, which starts a new dataset.
            **kwargs: parameters that take precedence over `parameters`.

        Returns:
            The dataset with the loaded data.

        """
        start = base.Dataset() if item is None else item
        return super()._call_apply(method, start, **kwargs)

    def _get_clerk(self) -> nagata.FileManager:
        """Returns `clerk`, making a `Clerk` of the workspace if there is none.

        Returns:
            The file manager.

        """
        if self.clerk is None:
            self.clerk = clerks.Clerk()
        return self.clerk


@dataclasses.dataclass
class Download(Loader):
    """Downloads a file from a URL, or a dataset from OpenML, and loads it.

    A URL's file is saved in the clerk's input folder and loaded from there,
    so it is only downloaded again if it is missing or "refresh" is true.
    Keeping the file lets others check the exact data that a project used.

    Any other source is the name (such as "credit-g") or id (such as 31) of a
    dataset on OpenML (https://www.openml.org), which shares thousands of
    datasets for research. scikit-learn saves each one in an "openml" folder
    in the clerk's input folder, so it is only downloaded once, and the
    dataset's default target is the label unless "label" is set.

    Wraps:
        [`nagata.FileManager`](https://github.com/WithPrecedent/nagata)
        from nagata, which reads each format with pandas (such as
        `pandas.read_csv`), and
        [`sklearn.datasets.fetch_openml`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
        from scikit-learn.

    """

    def load(
        self,
        source: Any,
        *,
        file_name: str | None = None,
        refresh: bool = False,
        timeout: float | None = None,
        **kwargs: Any) -> Any:
        """Downloads the data at `source` (unless it was before) and loads it.

        Args:
            source: the URL of a file (http or https), or the name or id of
                an OpenML dataset.
            file_name: name to save a URL's file as in the clerk's input
                folder, which may include folders. Defaults to `None`, in
                which case the last part of the URL's path is used. Not used
                for OpenML.
            refresh: whether to download a URL's file again even if it was
                downloaded before. Defaults to `False`. Not used for OpenML.
            timeout: seconds to wait for the server of a URL to answer.
                Defaults to `None`, which uses `options._DOWNLOAD_TIMEOUT`
                (read when the file is downloaded). Not used for OpenML.
            **kwargs: for a URL, parameters for `read`, such as
                "file_format", "member", and parameters for the reader. For
                OpenML, parameters for `fetch_openml`, such as "version" or
                "target_column".

        Raises:
            ValueError: if `source` is a URL whose scheme is not http or
                https.

        Returns:
            The loaded data: the file's, or the OpenML dataset as
                scikit-learn returns it.

        """
        text = str(source).strip()
        scheme = urllib.parse.urlparse(text).scheme
        if not scheme:
            return self._fetch_openml(text, **kwargs)
        if scheme not in _SCHEMES:
            message = (
                f'only http and https URLs can be downloaded, not {text!r} '
                f'(any other source must be the name or id of an OpenML '
                f'dataset)'
            )
            raise ValueError(message)
        name = file_name or _name_of(text)
        path = pathlib.Path(self._get_clerk().input_folder) / name
        if refresh or not path.is_file():
            _download(
                text,
                path,
                options._DOWNLOAD_TIMEOUT if timeout is None else timeout)
        return self.read(path, **kwargs)

    """ Private Methods """

    def _fetch_openml(self, source: str, **kwargs: Any) -> Any:
        """Downloads (unless it was before) and loads an OpenML dataset.

        Args:
            source: the name of the dataset (such as "credit-g") or its id
                (such as "31").
            **kwargs: parameters for `fetch_openml`. Those it does not accept
                are left out.

        Returns:
            The dataset, as scikit-learn returns it.

        """
        fetch = utilities.import_tool(self.contents or _OPENML)
        if source.isdigit():
            fixed: dict[str, Any] = {'data_id': int(source)}
        else:
            fixed = {'name': source}
        fixed |= {
            'data_home': str(self._get_clerk().input_folder),
            'as_frame': True,
            'return_X_y': False}
        extra = utilities.accepted_parameters(fetch, kwargs)
        return fetch(**fixed, **{
            k: v for k, v in extra.items() if k not in fixed})


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

    The source is a folder, a pattern of paths (such as "opinions/*.txt"), or
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
            source: a folder, a pattern of paths (such as "opinions/*.txt"),
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


def _download(url: str, path: pathlib.Path, timeout: float) -> None:
    """Saves the file at `url` to `path`.

    The file is written beside `path` and then renamed, so a download that
    fails leaves no file behind to be mistaken for the data.

    Args:
        url: the URL of the file. Its scheme must be in `_SCHEMES`.
        path: where to save the file.
        timeout: seconds to wait for the server to answer.

    """
    path.parent.mkdir(parents = True, exist_ok = True)
    partial = path.with_name(f'{path.name}.part')
    # `Download.load` checks that the scheme of `url` is http or https.
    request = urllib.request.Request(  # noqa: S310
        url, headers = {'User-Agent': _USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout = timeout) as response,  # noqa: S310
            partial.open('wb') as file):
            shutil.copyfileobj(response, file)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok = True)


def _extract(archive: pathlib.Path, member: str) -> pathlib.Path:
    """Extracts `member` from a zip archive to a folder named for the archive.

    Args:
        archive: path of the zip archive.
        member: name of the file in the archive.

    Raises:
        KeyError: if `member` is not in the archive.
        ValueError: if `archive` is not a zip archive.

    Returns:
        The path of the extracted file.

    """
    if not zipfile.is_zipfile(archive):
        message = f'{archive.name!r} is not a zip archive with {member!r} in it'
        raise ValueError(message)
    with zipfile.ZipFile(archive) as opened:
        names = opened.namelist()
        if member not in names:
            message = f'{member!r} is not in {archive.name!r}, which has {names}'
            raise KeyError(message)
        return pathlib.Path(opened.extract(member, archive.with_suffix('')))


def _find(path: pathlib.Path, folder: pathlib.Path) -> pathlib.Path:
    """Returns `path` if it is a file, or else `path` in `folder`.

    Args:
        path: path of a file.
        folder: the folder to look in if `path` is not a file.

    Raises:
        FileNotFoundError: if neither is a file.

    Returns:
        The path of the file.

    """
    if path.is_file():
        return path
    if (folder / path).is_file():
        return folder / path
    message = f'there is no file at {path} or {folder / path}'
    raise FileNotFoundError(message)


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


def _format_of(path: pathlib.Path, clerk: nagata.FileManager) -> str:
    """Returns the name of the clerk's format for the file at `path`.

    Args:
        path: path of a file.
        clerk: the file manager, which knows the extensions of its formats.

    Raises:
        ValueError: if the format is not known from the file's name.

    Returns:
        The name of the format.

    """
    extensions = [s.lstrip('.').lower() for s in path.suffixes]
    while extensions and extensions[-1] in _COMPRESSIONS:
        extensions.pop()
    name = clerk.extensions.get(extensions[-1]) if extensions else None
    if name is None:
        message = (
            f'the format of {path.name!r} is not known from its name, which '
            f'should end in one of {sorted(clerk.extensions)}: or load it '
            f'with a loader (such as load_file) and set "file_format" (such '
            f'as "csv"), or "member" for a file in a zip archive'
        )
        raise ValueError(message)
    return name


def _is_hidden(path: pathlib.Path) -> bool:
    """Returns whether a part of `path` starts with "." (and so is hidden).

    Args:
        path: a path, relative to the folder it was found in.

    Returns:
        Whether any part of the path other than "." and ".." starts with ".".

    """
    return any(
        part.startswith('.') and part not in {'.', '..'} for part in path.parts)


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
            f'"data.csv")'
        )
        raise ValueError(message)
    return name


def _reader_of(kind: nagata.FileFormat) -> Callable[..., Any]:
    """Returns the function that reads files of a format.

    Args:
        kind: one of the clerk's formats.

    Returns:
        The `pandas` function that the format uses (such as `read_csv`), or
            else the format's own `load` method.

    """
    reader = getattr(kind, 'loader', None)
    if isinstance(reader, str):
        return cast('Callable[..., Any]', getattr(pd, reader))
    # Every format that `nagata` registers has a `load` method, though the
    # base class does not declare one.
    return cast('Callable[..., Any]', kind.load)  # type: ignore[attr-defined]


def _settings_for(kind: nagata.FileFormat) -> dict[str, Any]:
    """Returns parameters that replace the clerk's shared settings for a format.

    Args:
        kind: one of the clerk's formats.

    Returns:
        The value from `options._FILE_SETTINGS` for each parameter of the
            format's reader that the clerk would fill from its settings.

    """
    return {
        parameter: options._FILE_SETTINGS[setting]
        for parameter, setting in (kind.load_parameters or {}).items()
        if setting in options._FILE_SETTINGS}
