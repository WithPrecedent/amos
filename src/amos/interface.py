"""Interface for creating, running, and exporting a project.

Contents:
    Project: a `chrisjen.Project` for data science.

"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import pathlib
from collections.abc import Mapping, MutableMapping
from typing import Any, cast

import chrisjen
import nagata
import numpy as np
import pandas as pd

from . import (
    auto,
    base,
    evaluators,
    loaders,
    options,
    pipelines,
    selectors,
    utilities,
)
from .clerks import Clerk

# Arguments of a `Clerk` that can be set in the "files" section of the
# settings.
_FOLDERS: tuple[str, ...] = (
    'input_folder', 'interim_folder', 'output_folder', 'data_subfolder')
# Formats of the scorecard (and of regression tables) that need an optional
# package, and the package.
_SCORECARD_FORMATS: tuple[tuple[str, str], ...] = (
    ('docx', 'docx'),
    ('png', 'matplotlib'))
# Settings of a worker that only a design that compares paths uses, which the
# winning settings drop when they make a contest a flow.
_COMPARISON_SETTINGS: tuple[str, ...] = (
    'design', 'criterion', 'conserve_memory')
# Parameters of a model that only its hyperparameter search uses.
_SEARCH_PARAMETERS: tuple[str, ...] = ('search', 'cv', 'n_iter', 'scoring')


@dataclasses.dataclass
class Project(chrisjen.Project):
    """A `chrisjen.Project` for data science.

    It works like a `chrisjen.Project` (see its documentation), with these
    differences:

    * The item is made into a `Dataset`. Pass a `pandas.DataFrame`, a path to
      a data file, or anything else that `Dataset.create` accepts. The
      "general" section of the settings can set the dataset's "label",
      "task", "seed", "groups", and "use_gpu" (which is `False` unless the
      settings set it to true).
    * A "pipeline" named in the settings (such as `pipeline =
      classification` in the "general" section) fills in the settings of the
      analyst, critic, and artist (see the `pipelines` module).
    * Each group named in a "feature_groups" section of the settings is a
      technique that keeps the features listed for it (see the `selectors`
      module).
    * A workflow with a loader (see the `loaders` module) loads its own data
      with the project's `clerk`, so it needs no item.
    * Each call to `apply` works on a copy of the item, so applying a project
      twice gives the same result, and the item is never changed. A project
      created without an item or a loader is drafted but not applied.
    * The default report is `Findings`.
    * Its `clerk` is a `Clerk`, which keeps data and results outside the
      project's code: data in the "data" folder of the workspace (the
      folder that holds the "code" folder, see `clerks.workspace`), or a
      subfolder of it named by "data_subfolder" (such as "crime/nibrs"), and
      each run's results in its "studies" folder. The "files" section of the
      settings can set the "root_folder" (the workspace) and the other
      folders.
    * `export` saves everything needed to report and reproduce the results.
      If "export_results" is true in the "files" section, the results are
      exported after each call to `apply`.
    * `winning_settings` returns settings that repeat only the winning path
      of each `contest` or `experiment`, which `export` also saves.

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
        item: the data that the workflow is applied to. Defaults to `None`,
            in which case a loader in the workflow loads the data.

    Attributes:
        result: the `Dataset` made by the last call to `apply`.

    """

    """ Initialization Methods """

    def __post_init__(self) -> None:
        """Drafts the project and validates the item and report.

        If there is no item and no loader in the workflow, the project is
        drafted but not applied, even if `automatic` is `True`. Pass the data
        to `apply` instead.

        """
        if self.report is None:
            self.report = self.library.all[options._DEFAULT_REPORT]()
        if self.workflow is None:
            self.draft()
        if self.item is not None:
            self.item = self._validate_item(self.item)
        elif not self._has_loader():
            self.automatic = False
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
            clerk: file manager, or the root folder (the workspace) for a
                `Clerk`. Defaults to `None`, in which case the "root_folder"
                of the "files" section of the settings (or the workspace that
                `clerks.workspace` finds) is used.
            **kwargs: other arguments for `chrisjen.Project.create`, such as
                `item`, `name`, `id`, and `automatic`.

        Returns:
            A `Project` instance.

        """
        if isinstance(idea, Mapping) and not isinstance(idea, chrisjen.Idea):
            # A copy, so that filling in pipelines does not change the
            # caller's settings.
            idea = dict(idea)
        if not isinstance(idea, chrisjen.Idea):
            # bobbie annotates `create` as returning its base class, but it
            # returns an instance of the class it is called on.
            idea = cast('chrisjen.Idea', chrisjen.Idea.create(idea))
        pipelines.expand(idea, kwargs.get('name'))
        if not isinstance(clerk, nagata.FileManager):
            clerk = _make_clerk(clerk, idea)
        project = super().create(idea, clerk = clerk, **kwargs)
        return cast('Project', project)

    """ Public Methods """

    def draft(self) -> None:
        """Builds the `workflow` from the settings.

        The settings of any pipeline named in them are filled in first (see
        `pipelines.expand`), and the groups of any "feature_groups" section
        are added to the library as techniques (see
        `selectors.register_feature_groups`). Each "auto" technique in the
        workflow is told which kind of technique to choose, from the name of
        its step or worker (see `auto.resolve`).

        """
        pipelines.expand(self.idea, self.name)
        selectors.register_feature_groups(self.idea, self.library)
        super().draft()
        auto.resolve(self.workflow)

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
        with base.skipping_incompatible_techniques():
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
        | winning_settings.json | The settings of only the winning path of each contest (see `winning_settings`), if the workflow has one. |
        | environment.json | Versions of Python and the packages used, and the GPUs (if "use_gpu" is true). |
        | history.json | Each technique applied, with its parameters. |
        | metrics.csv | The model's metrics. |
        | predictions.csv | The labels, predictions, and probabilities. |
        | tables/{name}.csv | Each table. |
        | figures/{name}.png | Each figure (see `options._FIGURE_DPI`). |
        | scorecard.csv, .md | The scorecard of every branch (see `scorecard`). |
        | scorecard.docx | The scorecard as a Word document (with python-docx). |
        | scorecard.png | The scorecard as an image (with matplotlib). |
        | {name}.csv, .md, .docx, .png | Each `regression_table` in the workflow, in the formats of the scorecard. |
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
        winning = self.winning_settings()
        if winning is not None:
            self._save_json(winning, folder, 'winning_settings.json')
        self._save_json(
            utilities.environment(gpu = result.use_gpu),
            folder,
            'environment.json')
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
                file_format = 'png',
                dpi = options._FIGURE_DPI)
        if result.predictions is not None:
            # The Word and image versions are saved if their optional
            # packages are installed.
            formats = ['csv', 'md'] + [
                extension for extension, package in _SCORECARD_FORMATS
                if _importable(package)]
            self.scorecard.export(folder, name = 'scorecard', formats = formats)
            for node in _nodes(self.workflow):
                if (
                    isinstance(node, evaluators.RegressionTable)
                    and node.table is not None):
                    node.export(folder, formats = formats)
        if data:
            self._save_table(result.data, folder, 'data')
        return folder

    def winning_settings(self) -> dict[str, Any] | None:
        """Returns settings that repeat only the winning path of each contest.

        Another researcher can create a project from them to fit the winning
        techniques alone, without trying every combination again. Each
        `contest` (or `experiment`) in the workflow becomes a `flow`:

        * Each step that it compared keeps only the technique of the winning
          path (as does a contest without steps, whose techniques are the
          alternatives), and its "design", "criterion", and
          "conserve_memory" settings are dropped.
        * The parameters sections of the techniques that lost are dropped,
          unless another worker still uses the technique, and so are their
          own worker sections and feature groups.
        * A model that searched for its hyperparameters is given the values
          that the search chose rather than the lists it searched, without
          its "search", "cv", "n_iter", and "scoring" parameters.
        * The settings of pipelines are already filled in, so each "pipeline"
          setting is dropped (it would otherwise add the settings of the
          contest back).

        Raises:
            ValueError: if the project has not been applied.

        Returns:
            The settings, or `None` if no contest in the workflow has a
                winner.

        """
        if self.result is None:
            message = 'apply the project before finding its winning settings'
            raise ValueError(message)
        history = getattr(self.result, 'history', [])
        winners = {}
        for worker in _contests(self.workflow):
            # A contest inside another is applied only as a copy, so its
            # winner is found in the history (where an experiment records
            # it).
            winner = worker.winner or _recorded_winner(worker, history)
            if winner is not None:
                winners[worker] = winner
        if not winners:
            return None
        idea = copy.deepcopy(self.idea)
        losers = []
        for worker, winner in winners.items():
            losers.extend(_keep_winning_path(idea, worker, winner))
        _drop_losing_settings(idea, losers)
        _pin_searched_parameters(idea, self.workflow, history)
        for name, section in list(idea.contents.items()):
            if isinstance(section, Mapping) and 'pipeline' in section:
                idea.contents[name] = {
                    k: v for k, v in section.items() if k != 'pipeline'}
        return dict(idea)

    """ Properties """

    @property
    def scorecard(self) -> evaluators.Scorecard:
        """Returns the scorecard of every branch of the most recent run.

        If the workflow has a `scorecard` technique, it is returned with the
        scorecard it made (so its settings, such as "metrics" or "digits",
        are kept). Otherwise, a new scorecard of the result is made. Save it
        with its `to_csv`, `to_markdown`, `to_word`, `to_image`, or `export`
        methods.

        Raises:
            TypeError: if the result is not a `Dataset`.
            ValueError: if the project has not been applied.

        Returns:
            The scorecard.

        """
        if self.result is None:
            message = 'apply the project before making its scorecard'
            raise ValueError(message)
        if not isinstance(self.result, base.Dataset):
            message = f'only a Dataset has a scorecard, not {type(self.result)}'
            raise TypeError(message)
        made = [
            node for node in _nodes(self.workflow)
            if isinstance(node, evaluators.Scorecard)
            and node.table is not None]
        if made:
            return made[-1]
        return evaluators.Scorecard.create(self.result)

    """ Private Methods """

    def _has_loader(self) -> bool:
        """Returns whether the workflow has a loader, which loads the data.

        Returns:
            Whether any node in the workflow, at any level, is a `Loader`.

        """
        return any(
            isinstance(node, loaders.Loader) for node in _nodes(self.workflow))

    def _save_json(self, item: Any, folder: pathlib.Path, name: str) -> None:
        """Saves `item` as a json file.

        Args:
            item: object to save. Anything that json cannot store is saved as
                its `str`, including the keys of a `dict` (see `_storable`).
            folder: folder to save it in.
            name: file name, with its extension.

        """
        text = json.dumps(_storable(item), indent = 2, default = str)
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

        A path to a data file is read as `load_file` reads it, with the
        project's clerk: a relative path is looked for in the current folder
        and then in the clerk's input folder, and the file may be in any
        format that the clerk knows.

        Args:
            item: the data, or anything that `Dataset.create` accepts. If it
                is `None` and the workflow has a loader, the loader loads the
                data into an empty `Dataset`.

        Raises:
            FileNotFoundError: if `item` is a path and there is no file there.
            ValueError: if `item` is `None` and the workflow has no loader, or
                if it is a path to a file whose format is not known.

        Returns:
            A `Dataset`, with the "label", "task", "seed", and "groups" from
                the "general" section of the settings if it did not have them,
                and the "use_gpu" of the settings. If the settings do not set
                "use_gpu", a `Dataset` keeps its own (which is `False` unless
                it was made with `use_gpu = True`), and other data is `False`.

        """
        general = self.idea.get('general', {})
        use_gpu = general.get('use_gpu')
        if item is None:
            if not self._has_loader():
                message = (
                    'there is no data: pass it as item, or load it with a '
                    'loader (such as load_file) in the workflow'
                )
                raise ValueError(message)
            # The loader applies the label, task, and groups, which an empty
            # dataset cannot have.
            return base.Dataset(seed = general.get('seed'), use_gpu = use_gpu)
        if isinstance(item, str | pathlib.Path):
            item = loaders.LoadFile(clerk = self.clerk).read(item)
        return base.Dataset.create(
            item,
            label = general.get('label'),
            task = general.get('task'),
            seed = general.get('seed'),
            groups = general.get('groups'),
            use_gpu = use_gpu)


""" Private Functions """


def _make_clerk(
    root: pathlib.Path | str | None,
    idea: chrisjen.Idea) -> nagata.FileManager:
    """Returns a file manager for a project.

    Args:
        root: the root folder (the workspace), or `None` to use the
            "root_folder" of the "files" section of `idea`, or else the
            workspace that `clerks.workspace` finds.
        idea: settings of the project.

    Returns:
        A `Clerk`. Folders in the "files" section of `idea` (such as
            "output_folder" and "data_subfolder") are passed to it.

    """
    files = dict(idea.get('files', {}))
    if root is None:
        root = files.get('root_folder')
    folders = {k: v for k, v in files.items() if k in _FOLDERS}
    return Clerk(root_folder = root, **folders)


def _chosen_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Returns the parameters of a fitted model, as recorded in its history.

    Args:
        parameters: the "parameters" of a model's entry in `history`. A
            classifier wrapped in a `LabelCoded` records only its
            "estimator", whose own parameters are used.

    Returns:
        The parameters, by name.

    """
    chosen = {k: v for k, v in parameters.items() if k != 'estimator'}
    wrapped = parameters.get('estimator')
    if wrapped is not None:
        chosen = {**utilities.parameters_of(wrapped), **chosen}
    return chosen


def _contests(workflow: Any) -> list[chrisjen.Contest]:
    """Returns every contest (and experiment) in a workflow.

    Args:
        workflow: a `chrisjen.Worker`, or `None`.

    Returns:
        `workflow` itself, if it is a contest, and every contest in it, in
            order.

    """
    nodes = [workflow, *_nodes(workflow)]
    return [node for node in nodes if isinstance(node, chrisjen.Contest)]


def _drop_losing_settings(
    idea: chrisjen.Idea,
    losers: list[tuple[str, str]]) -> None:
    """Drops the settings of the techniques that lost a contest.

    The "{technique}_{step}_parameters" section of each loser is dropped.
    Its "{technique}_parameters" section, its own worker section, and its
    feature group (in the "feature_groups" section) are dropped only if no
    worker, step, or criterion still names it.

    Args:
        idea: the settings, which are changed.
        losers: each technique that lost and the step (or contest without
            steps) that it was an alternative in.

    """
    contents = idea.contents
    for technique, step in losers:
        contents.pop(f'{technique}_{step}_parameters', None)
    used = {
        name for names in [*idea.techniques.values(), *idea.steps.values()]
        for name in names}
    used.update(str(name) for name in idea.criteria.values())
    unused = {technique for technique, _ in losers} - used
    for technique in unused:
        contents.pop(f'{technique}_parameters', None)
        if technique in idea.workers:
            del contents[technique]
    groups = contents.get('feature_groups')
    if isinstance(groups, Mapping) and unused & set(groups):
        contents['feature_groups'] = {
            k: v for k, v in groups.items() if k not in unused}


def _importable(package: str) -> bool:
    """Returns whether `package` can be imported.

    Args:
        package: name of the package.

    Returns:
        Whether importing it succeeds.

    """
    try:
        importlib.import_module(package)
    except Exception:  # noqa: BLE001
        return False
    return True


def _keep_winning_path(
    idea: chrisjen.Idea,
    worker: chrisjen.Contest,
    winner: str) -> list[tuple[str, str]]:
    """Makes a contest a flow of only the techniques of its winning path.

    Args:
        idea: the settings, which are changed.
        worker: the contest.
        winner: the label of its winning path (the names of its nodes,
            joined by " > ").

    Returns:
        Each technique that lost and the step (or the contest, if it has no
            steps) that it was an alternative in.

    """
    name = str(worker.name)
    path = next(
        (p for p in worker.walk()
         if ' > '.join(str(getattr(n, 'name', n)) for n in p) == winner),
        None)
    if path is None:
        return []
    techniques = idea.techniques
    # A contest without steps has its techniques as its alternatives.
    own = [] if name in idea.steps else techniques.get(name, [])
    losers: list[tuple[str, str]] = []
    for node in path:
        label = str(getattr(node, 'name', node))
        if isinstance(node, chrisjen.Step) and node.contents is not None:
            technique = str(node.contents.name)
            step = label.removeprefix(f'{technique}_')
        elif label in own:
            technique, step = label, name
        else:
            # A worker that is a step of its own is in every path.
            continue
        choices = techniques.get(step, [])
        if technique not in choices:
            continue
        _set_settings(idea, step, 'techniques', technique)
        losers.extend((c, step) for c in choices if c != technique)
    for key in _COMPARISON_SETTINGS:
        _set_settings(idea, name, key, None)
    return losers


def _nodes(worker: Any) -> list[Any]:
    """Returns every node in a workflow, in order, at every level.

    Args:
        worker: a `chrisjen.Worker`, or another node.

    Returns:
        The nodes of `worker` and of the workers and steps inside it.

    """
    found = []
    if isinstance(worker, chrisjen.Worker):
        for node in worker:
            found.append(node)
            found.extend(_nodes(node))
    elif isinstance(getattr(worker, 'contents', None), chrisjen.Vertex):
        # A step that wraps a technique or a worker.
        found.append(worker.contents)
        found.extend(_nodes(worker.contents))
    return found


def _pin_searched_parameters(
    idea: chrisjen.Idea,
    workflow: Any,
    history: list[dict[str, Any]]) -> None:
    """Gives each searched model the values that its search chose.

    The parameters of a model can be in its "{technique}_parameters"
    section and, for each step that wraps it, the "{step}_parameters" and
    "{technique}_{step}_parameters" sections. In each, a list of values that
    was searched is replaced by the value chosen, and the parameters of the
    search (see `_SEARCH_PARAMETERS`) are dropped.

    Args:
        idea: the settings, which are changed.
        workflow: the workflow of the project.
        history: the history of the result.

    """
    contents = idea.contents
    for entry in history:
        if not entry.get('search') or not entry.get('parameters'):
            continue
        technique = str(entry['technique'])
        chosen = _chosen_parameters(entry['parameters'])
        names = [f'{technique}_parameters']
        for node in _nodes(workflow):
            if (
                isinstance(node, chrisjen.Step)
                and node.contents is not None
                and str(node.contents.name) == technique):
                step = str(node.name).removeprefix(f'{technique}_')
                names.extend([
                    f'{step}_parameters', f'{technique}_{step}_parameters'])
        for name in names:
            section = contents.get(name)
            if not isinstance(section, Mapping):
                continue
            contents[name] = {
                key: (
                    _plain(chosen[key])
                    if isinstance(value, list) and key in chosen else value)
                for key, value in section.items()
                if key not in _SEARCH_PARAMETERS}


def _plain(value: Any) -> Any:
    """Returns `value` as a plain Python value, if it is a `numpy` value.

    Args:
        value: a value, such as a parameter that a search chose.

    Returns:
        The Python value of a `numpy` value, or `value` itself.

    """
    return value.item() if isinstance(value, np.generic) else value


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


def _recorded_winner(
    worker: chrisjen.Contest,
    history: list[dict[str, Any]]) -> str | None:
    """Returns the winner of a contest, as its entry in `history` records it.

    Args:
        worker: the contest.
        history: the history of the result.

    Returns:
        The "winner" of the last entry of `worker` in `history`, or `None`
            if there is none (only an `Experiment` records its winner).

    """
    for entry in reversed(history):
        if entry.get('technique') == worker.name and entry.get('winner'):
            return str(entry['winner'])
    return None


def _set_settings(
    idea: chrisjen.Idea,
    name: str,
    key: str,
    value: Any) -> None:
    """Sets every setting `key` that belongs to `name`, or deletes it.

    A setting belongs to `name` as `chrisjen.Idea` finds it: it is `key` in
    the section of `name`, or "{name}_{key}" in any worker section.

    Args:
        idea: the settings, which are changed.
        name: the name of a worker or step.
        key: the setting, such as "techniques" or "design".
        value: the new value, or `None` to delete the setting.

    """
    for section_name, section in idea.workers.items():
        settings = [f'{name}_{key}']
        if section_name == name:
            settings.append(key)
        for setting in settings:
            if setting not in section:
                continue
            if value is None:
                del section[setting]
            else:
                section[setting] = value


def _storable(item: Any) -> Any:
    """Returns `item` with keys that a json file can store.

    json saves a value that it does not know as its `str` (see
    `Project._save_json`), but it refuses a key that is not text, a number, a
    boolean, or `None`. A history can have such keys: a technique may record
    something for each class of the label, for example, and the classes of
    some labels are `numpy` values.

    Args:
        item: the object to save.

    Returns:
        `item`, with every key of every `dict` in it (at any depth) that json
            cannot store replaced: by the plain Python value of a `numpy`
            value, and otherwise by its `str`.

    """
    if isinstance(item, Mapping):
        stored = {}
        for key, value in item.items():
            name = key.item() if isinstance(key, np.generic) else key
            if not isinstance(name, str | int | float | bool | None):
                name = str(name)
            stored[name] = _storable(value)
        return stored
    if isinstance(item, list | tuple):
        return [_storable(value) for value in item]
    return item
