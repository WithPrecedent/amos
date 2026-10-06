"""Interface for creating, running, and exporting a project.

Contents:
    Project: a `chrisjen.Project` for data science.

"""

from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
from collections.abc import MutableMapping
from typing import Any, cast

import chrisjen
import nagata
import pandas as pd

from . import base, options, utilities

# Arguments of a `nagata.FileManager` that can be set in the "files" section
# of the settings.
_FOLDERS: tuple[str, ...] = ('input_folder', 'interim_folder', 'output_folder')


@dataclasses.dataclass
class Project(chrisjen.Project):
    """A `chrisjen.Project` for data science.

    It works like a `chrisjen.Project` (see its documentation), with these
    differences:

    * The item is made into a `Dataset`. Pass a `pandas.DataFrame`, a path to
      a data file, or anything else that `Dataset.create` accepts. The
      "general" section of the settings can set the dataset's "label",
      "task", and "seed".
    * Each call to `apply` works on a copy of the item, so applying a project
      twice gives the same result, and the item is never changed. A project
      created without an item is drafted but not applied.
    * The default report is `Findings`.
    * Files are kept in the current folder unless the "files" section of the
      settings sets a "root_folder" (or other folders).
    * `export` saves everything needed to report and reproduce the results.
      If "export_results" is true in the "files" section, the results are
      exported after each call to `apply`.

    Args:
        name: name of the project.
        id: unique name for this run of the project.
        automatic: whether to apply the project as soon as it is created.
        library: library of the classes used to build the workflow.
        clerk: file manager for the project's files.
        idea: settings that describe the project.
        workflow: the workflow of the project. Defaults to `None`, in which
            case it is built by `draft`.
        report: report generated after the project is applied. Defaults to
            `None`, in which case `Findings` is used.
        item: the data that the workflow is applied to. Defaults to `None`.

    Attributes:
        result: the `Dataset` made by the last call to `apply`.

    """

    """ Initialization Methods """

    def __post_init__(self) -> None:
        """Validates the item and report, and then drafts the project.

        If there is no item, the project is drafted but not applied, even if
        `automatic` is `True`. Pass the data to `apply` instead.

        """
        if self.report is None:
            self.report = self.library.all[options._DEFAULT_REPORT]()
        if self.item is None:
            self.automatic = False
        else:
            self.item = self._validate_item(self.item)
        super().__post_init__()

    """ Class Methods """

    @classmethod
    def create(
        cls,
        idea: chrisjen.Idea | MutableMapping[str, Any] | pathlib.Path | str,
        *,
        clerk: nagata.FileManager | pathlib.Path | str | None = None,
        **kwargs: Any) -> Project:
        """Creates a project.

        Every argument after `idea` must be passed by keyword.

        Args:
            idea: settings that describe the project: an `Idea`, a `dict`, or
                the path to a settings file.
            clerk: file manager, or the root folder for one. Defaults to
                `None`, in which case the "root_folder" of the "files" section
                of the settings (or `options._DEFAULT_ROOT`) is used.
            **kwargs: other arguments for `chrisjen.Project.create`, such as
                `item`, `name`, `id`, and `automatic`.

        Returns:
            A `Project` instance.

        """
        if not isinstance(idea, chrisjen.Idea):
            idea = chrisjen.Idea.create(idea)
        if not isinstance(clerk, nagata.FileManager):
            clerk = _make_clerk(clerk, idea)
        project = super().create(idea, clerk = clerk, **kwargs)
        return cast('Project', project)

    """ Public Methods """

    def apply(self, item: Any = None, **kwargs: Any) -> Any:
        """Applies the workflow to a copy of the data and generates the report.

        Args:
            item: the data to apply the workflow to. Defaults to `None`, in
                which case the project's `item` is used.
            **kwargs: keyword arguments passed to the nodes in the workflow.

        Returns:
            The resulting `Dataset`, which is also stored in `result`.

        """
        dataset = self._validate_item(self.item if item is None else item)
        result = super().apply(copy.deepcopy(dataset), **kwargs)
        if self.idea.get('files', {}).get('export_results', False):
            self.export()
        return result

    def export(
        self,
        folder: pathlib.Path | str | None = None,
        *,
        data: bool = False) -> pathlib.Path:
        """Saves everything needed to report and reproduce the results.

        The files are saved by the project's `clerk`:

        | File | Contents |
        | --- | --- |
        | report.txt | The text of the report. |
        | settings.json | The settings of the project. |
        | environment.json | Versions of Python and the packages used. |
        | history.json | Each technique applied, with its parameters. |
        | metrics.csv | The model's metrics. |
        | predictions.csv | The labels, predictions, and probabilities. |
        | tables/{name}.csv | Each table. |
        | figures/{name}.png | Each figure. |
        | data.csv | The data after the workflow (if `data` is `True`). |

        Args:
            folder: folder to save the files in. Defaults to `None`, in which
                case a folder named for the project's `id` in the clerk's
                output folder is used.
            data: whether to also save the data. Defaults to `False`.

        Raises:
            TypeError: if the result is not a `Dataset`.
            ValueError: if the project has not been applied.

        Returns:
            The absolute path of the folder the files were saved in.

        """
        result = self.result
        if result is None:
            message = 'apply the project before exporting its results'
            raise ValueError(message)
        if not isinstance(result, base.Dataset):
            message = f'only a Dataset can be exported, not {type(result)}'
            raise TypeError(message)
        if folder is None:
            folder = pathlib.Path(self.clerk.output_folder) / self.id
        folder = pathlib.Path(folder).resolve()
        for path in (folder, folder / 'tables', folder / 'figures'):
            path.mkdir(parents = True, exist_ok = True)
        report = '' if self.report is None else self.report.contents
        self._save_text(report or '', folder, 'report.txt')
        self._save_json(dict(self.idea), folder, 'settings.json')
        self._save_json(utilities.environment(), folder, 'environment.json')
        self._save_json(result.history, folder, 'history.json')
        if result.metrics:
            metrics = pd.Series(result.metrics, name = 'value').to_frame()
            self._save_table(metrics, folder, 'metrics')
        if result.predictions is not None:
            self._save_table(_predictions(result), folder, 'predictions')
        for name, table in result.tables.items():
            self._save_table(table, folder / 'tables', name)
        for name, figure in result.figures.items():
            self.clerk.save(
                figure,
                folder = folder / 'figures',
                file_name = name,
                file_format = 'png')
        if data:
            self._save_table(result.data, folder, 'data')
        return folder

    """ Private Methods """

    def _save_json(self, item: Any, folder: pathlib.Path, name: str) -> None:
        """Saves `item` as a json file.

        Args:
            item: object to save. Anything that json cannot store is saved as
                its `str`.
            folder: folder to save it in.
            name: file name, with its extension.

        """
        text = json.dumps(item, indent = 2, default = str)
        self._save_text(text, folder, name)

    def _save_table(
        self,
        table: pd.DataFrame,
        folder: pathlib.Path,
        name: str) -> None:
        """Saves `table` as a csv file, with its index.

        Args:
            table: table to save.
            folder: folder to save it in.
            name: file name, without its extension.

        """
        self.clerk.save(
            table,
            folder = folder,
            file_name = name,
            file_format = 'csv',
            encoding = 'utf-8',
            header = True,
            index = True)

    def _save_text(self, text: str, folder: pathlib.Path, name: str) -> None:
        """Saves `text` as a text file.

        Args:
            text: text to save.
            folder: folder to save it in.
            name: file name, with its extension.

        """
        self.clerk.save(
            text,
            folder = folder,
            file_name = name,
            file_format = 'text',
            encoding = 'utf-8')

    def _validate_item(self, item: Any) -> base.Dataset:
        """Returns `item` as a `Dataset`.

        A relative path to a data file is looked for in the clerk's input
        folder if it is not found in the current folder.

        Args:
            item: the data, or anything that `Dataset.create` accepts.

        Raises:
            ValueError: if `item` is `None`.

        Returns:
            A `Dataset`, with the "label", "task", and "seed" from the
                "general" section of the settings if it did not have them.

        """
        if item is None:
            message = 'there is no data: pass it as item'
            raise ValueError(message)
        if isinstance(item, str | pathlib.Path):
            path = pathlib.Path(item)
            if not path.exists():
                path = pathlib.Path(self.clerk.input_folder) / path
            item = path
        general = self.idea.get('general', {})
        return base.Dataset.create(
            item,
            label = general.get('label'),
            task = general.get('task'),
            seed = general.get('seed'))


""" Private Functions """


def _make_clerk(
    root: pathlib.Path | str | None,
    idea: chrisjen.Idea) -> nagata.FileManager:
    """Returns a file manager for a project.

    Args:
        root: the root folder, or `None` to use the "root_folder" of the
            "files" section of `idea` (or `options._DEFAULT_ROOT`).
        idea: settings of the project.

    Returns:
        A file manager. Folders in the "files" section of `idea` (such as
            "output_folder") are passed to it.

    """
    files = dict(idea.get('files', {}))
    if root is None:
        root = files.get('root_folder', options._DEFAULT_ROOT)
    folders = {k: v for k, v in files.items() if k in _FOLDERS}
    return nagata.FileManager(root_folder = pathlib.Path(root), **folders)


def _predictions(item: base.Dataset) -> pd.DataFrame:
    """Returns the labels, predictions, and probabilities of the predicted rows.

    Args:
        item: a dataset with predictions.

    Raises:
        ValueError: if there are no predictions.

    Returns:
        One row for each predicted row, with "actual" and "predicted" columns
            and a "probability_{class}" column for each class.

    """
    predictions = item.predictions
    if predictions is None:
        message = 'there are no predictions to save'
        raise ValueError(message)
    table = pd.DataFrame({
        'actual': item.y.loc[predictions.index],
        'predicted': predictions})
    if item.probabilities is not None:
        for column in item.probabilities.columns:
            table[f'probability_{column}'] = item.probabilities[column]
    return table
