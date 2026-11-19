"""Savers of a whole dataset, so that a workflow can go on from it later.

Contents:
    SaveData: saves the whole dataset, and its data as a table, so that a
        later run can go on from it.
    _DATASET_FORMAT: the clerk's format of the file of the whole dataset.
    _TABLE_FORMAT: the default format of the table of the data.

"""

from __future__ import annotations

import dataclasses
import pathlib
import warnings
from collections.abc import Callable
from typing import Any, cast

import nagata
import pandas as pd

from ...core import base, options, utilities
from ._base import Saver

# The clerk's format of the file of the whole dataset.
_DATASET_FORMAT: str = 'joblib'
# The default format of the table of the data.
_TABLE_FORMAT: str = 'parquet'


@dataclasses.dataclass
class SaveData(Saver):
    """Saves the whole dataset, so that a later run can go on from it.

    The whole `Dataset` (its data, label, task, seed, groups, split, fitted
    tools, model, predictions, metrics, tables, figures, and history) is
    saved with joblib, in a ".joblib" file. To go on from it later, pass
    the file as the `item` of a project whose workflow starts after this
    point, or load it with a loader (such as `load_file`), or with
    `Dataset.create`. The dataset is what it was when it was saved,
    including its history, so the history of the later run shows every
    technique from the start.

    The data is also saved as a table (in a parquet file, unless
    "table_format" names another format of the clerk, such as "csv"), with
    its index, so that it can be read by other tools. Set "table_format" to
    false to save only the dataset. If the data cannot be saved in the
    format (parquet needs each column to hold one type, for example), a
    warning says so, and the dataset is still saved.

    Loading a joblib file runs code that it holds, so load only files that
    you or someone that you trust saved.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used. Defaults to `None`.
        parameters: "folder", "file_name", "table_format", and "compress"
            (see `save`). Defaults to an empty `dict`.
        clerk: the file manager that saves files. Defaults to `None`, in
            which case one for the workspace is made when it is needed.
        project_name: name of the project, which starts the default file
            name. Defaults to an empty `str`.

    """

    """ Required Methods """

    def save(
        self,
        item: base.Dataset,
        folder: pathlib.Path,
        file_name: str,
        *,
        table_format: str | bool | None = _TABLE_FORMAT,
        compress: int | bool = 0,
        **kwargs: Any) -> list[pathlib.Path]:
        """Saves the whole dataset, and its data as a table.

        Args:
            item: the dataset to save.
            folder: the folder to save the files in.
            file_name: name of the files, without an extension.
            table_format: name of the clerk's format of the table of the
                data (such as "parquet", "csv", or "excel"). Defaults to
                "parquet". `False` or `None` saves no table.
            compress: how much joblib compresses the file of the dataset,
                from 0 (not at all, which is fastest) to 9, or `True` (3).
                Defaults to 0.
            **kwargs: parameters for the writer of the table (such as
                `pandas.DataFrame.to_parquet`), which it takes if it accepts
                them.

        Raises:
            ValueError: if `table_format` is not a format that the clerk
                knows.

        Returns:
            The path of the file of the dataset, and of the table, if it was
                saved.

        """
        clerk = self._get_clerk()
        kind = None
        if table_format:
            kind = _format_named(str(table_format), clerk)
        dataset_path = folder / f'{file_name}.{_DATASET_FORMAT}'
        clerk.save(
            item,
            file_path = dataset_path,
            file_format = _DATASET_FORMAT,
            compress = compress)
        paths = [dataset_path]
        if kind is not None:
            extension = _extension_of(kind)
            table_path = folder / f'{file_name}.{extension}'
            parameters = utilities.accepted_parameters(
                _writer_of(kind),
                {**_settings_for(kind), 'index': True, **kwargs})
            try:
                clerk.save(
                    item.data,
                    file_path = table_path,
                    file_format = kind,
                    **parameters)
            except (ImportError, TypeError, ValueError) as error:
                warnings.warn(
                    f'{self.name!r} saved the dataset in {dataset_path}, but '
                    f'not its data as a {table_format} table: {error}. Set '
                    f'its "table_format" to another format (such as "csv"), '
                    f'or to false to save no table',
                    category = base.AmosWarning,
                    stacklevel = 2)
            else:
                paths.append(table_path)
        return paths


""" Private Functions """


def _extension_of(kind: nagata.FileFormat) -> str:
    """Returns the extension of new files of a format.

    Args:
        kind: one of the clerk's formats.

    Returns:
        The first of the format's extensions.

    """
    extensions = kind.extensions
    if isinstance(extensions, str):
        return extensions
    return str(next(iter(extensions)))


def _format_named(name: str, clerk: nagata.FileManager) -> nagata.FileFormat:
    """Returns the clerk's format named `name`.

    Args:
        name: name of a format (such as "parquet" or "csv").
        clerk: the file manager, which knows the formats.

    Raises:
        ValueError: if the clerk does not know the format.

    Returns:
        The format.

    """
    formats = clerk.framework.formats
    if name not in formats:
        message = (
            f'{name!r} is not a format that the clerk knows: set '
            f'"table_format" to one of {sorted(map(str, formats))}, or to '
            f'false to save no table'
        )
        raise ValueError(message)
    return cast('nagata.FileFormat', formats[name])


def _settings_for(kind: nagata.FileFormat) -> dict[str, Any]:
    """Returns parameters that replace the clerk's shared settings for a format.

    The clerk's own shared settings suit quick tests (it writes csv files in
    Windows-1252, for example), so the values of `options._FILE_SETTINGS`
    are used instead, as loaders use them.

    Args:
        kind: one of the clerk's formats.

    Returns:
        The value from `options._FILE_SETTINGS` for each parameter of the
            format's writer that the clerk would fill from its settings.

    """
    return {
        parameter: options._FILE_SETTINGS[setting]
        for parameter, setting in (kind.save_parameters or {}).items()
        if setting in options._FILE_SETTINGS}


def _writer_of(kind: nagata.FileFormat) -> Callable[..., Any]:
    """Returns the function that writes files of a format.

    Args:
        kind: one of the clerk's formats.

    Returns:
        The method of `pandas.DataFrame` that the format uses (such as
            `to_parquet`), or else the format's own `save` method.

    """
    writer = getattr(kind, 'saver', None)
    if isinstance(writer, str):
        return cast('Callable[..., Any]', getattr(pd.DataFrame, writer))
    # Every format that `nagata` registers has a `save` method, though the
    # base class does not declare one.
    return cast('Callable[..., Any]', kind.save)  # type: ignore[attr-defined]
