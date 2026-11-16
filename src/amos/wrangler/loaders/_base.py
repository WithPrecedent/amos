"""The base class of loaders, and what they share.

Contents:
    _COMPRESSIONS: extensions of compressed files, which `pandas` opens itself.
        They are skipped when the format of a file is found from its name.
    _TEXT_FORMATS: formats of the clerk that are read as plain text, in UTF-8
        unless an "encoding" is given.
    _GENERAL: settings in the "general" section that are the default parameters
        of a loader built from a project's settings.
    Loader: base class for techniques that load data.

"""

from __future__ import annotations

import abc
import dataclasses
import pathlib
import zipfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import nagata
import pandas as pd

from ...core import base, clerks, options, utilities

if TYPE_CHECKING:
    import chrisjen


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


""" Private Functions """


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
