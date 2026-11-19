"""The base class of savers, and what they share.

Contents:
    Saver: base class for techniques that save a dataset.
    _MAX_BRANCH_LENGTH: the most characters of the label of a branch that a
        file name keeps. A longer label is cut and ends with a short hash of
        the whole label, so that the names of branches still differ.

"""

from __future__ import annotations

import abc
import dataclasses
import hashlib
import pathlib
import re
from typing import TYPE_CHECKING, Any

import nagata

from ...core import base, clerks

if TYPE_CHECKING:
    import chrisjen


# The most characters of the label of a branch that a file name keeps. A
# longer label is cut and ends with a short hash of the whole label, so that
# the names of branches still differ.
_MAX_BRANCH_LENGTH: int = 100


@dataclasses.dataclass
class Saver(base.Operation, abc.ABC):
    """Base class for techniques that save a dataset.

    A subclass writes a `save` method, which saves the dataset in a folder
    under a file name (without an extension) and returns the paths of the
    files that it saved. A saver does not change the dataset: it records in
    `history` where the files are, and passes the dataset on, so a saver can
    come anywhere in a workflow.

    The files are saved in the clerk's interim folder (the input folder,
    unless the "files" section of the settings names another), unless the
    "folder" parameter names another: one of the clerk's folders ("input",
    "interim", or "output"), a folder in the clerk's root folder (the
    workspace), or an absolute path. The file name is the "file_name"
    parameter, or else the name of the project and of the saver (such as
    "cancer_save_data"). Each run saves over the files of the one before.

    In a branch of an experiment, which applies the saver once for each
    combination of techniques, the label of the branch is added to the file
    name (such as "cancer_save_data_standard_scale-logit_model"), so that
    every branch is saved. A saver built from a project's settings uses the
    project's clerk.

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used by most savers. Defaults to `None`.
        parameters: keyword arguments for `save`, and "folder" and
            "file_name". Defaults to an empty `dict`.
        clerk: the file manager that saves files. Defaults to `None`, in
            which case one for the workspace is made when it is needed.
        project_name: name of the project, which starts the default file
            name. Defaults to an empty `str`, in which case the file name is
            the saver's name.

    """

    clerk: nagata.FileManager | None = dataclasses.field(
        default = None, repr = False)
    project_name: str = ''

    """ Required Methods """

    @abc.abstractmethod
    def save(
        self,
        item: base.Dataset,
        folder: pathlib.Path,
        file_name: str,
        **kwargs: Any) -> list[pathlib.Path]:
        """Saves `item` in `folder` and returns the paths of its files.

        Args:
            item: the dataset to save. It must not be changed.
            folder: the folder to save the files in, which exists.
            file_name: name of the files, without an extension.
            **kwargs: parameters for saving.

        Returns:
            The paths of the files that were saved.

        """

    """ Class Methods """

    @classmethod
    def build(
        cls,
        name: str,
        project: chrisjen.Project,
        parameters: base.GenericDict | None = None,
        **kwargs: Any) -> Saver:
        """Builds a saver with the clerk and name of `project`.

        Args:
            name: name of the saver.
            project: the project the saver belongs to.
            parameters: more parameters, which take precedence over those in
                the settings. Defaults to `None`.
            **kwargs: other arguments for the constructor.

        Returns:
            The built saver.

        """
        kwargs.setdefault('clerk', project.clerk)
        kwargs.setdefault('project_name', str(project.name or ''))
        return super().build(name, project, parameters, **kwargs)

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        *,
        folder: pathlib.Path | str | None = None,
        file_name: str | None = None,
        **kwargs: Any) -> base.Dataset:
        """Saves `item` and records where its files are.

        The entry in `history` is made before the files are saved, so the
        saved dataset has it too, with the "folder" and "file_name" of the
        files. The dataset that the workflow goes on with also lists the
        "files" that were saved.

        Args:
            item: the dataset to save.
            folder: the folder to save the files in. Defaults to `None`,
                which is the clerk's interim folder.
            file_name: name of the files, without an extension. Defaults to
                `None`, which is the name of the project and of the saver.
            **kwargs: parameters for `save`.

        Returns:
            `item`, unchanged except for the new entry in its `history`.

        """
        target = self._folder(folder)
        target.mkdir(parents = True, exist_ok = True)
        name = self._file_name(file_name)
        rows, columns = item.data.shape
        item.record(
            self.name,
            folder = str(target),
            file_name = name,
            rows = rows,
            columns = columns)
        entry = item.history[-1]
        paths = self.save(item, target, name, **kwargs)
        entry['files'] = [str(path) for path in paths]
        return item

    """ Private Methods """

    def _file_name(self, file_name: str | None) -> str:
        """Returns the name of the files, with the label of any branch.

        Args:
            file_name: the name that the parameters give, if any.

        Returns:
            `file_name`, or else the name of the project and of the saver,
                followed (in a branch of an experiment) by the label of the
                branch, made safe for a file name.

        """
        name = file_name or '_'.join(
            part for part in (self.project_name, self.name) if part)
        branch = base._BRANCH.get()
        if branch:
            name = f'{name}_{_safe_label(branch)}'
        return name

    def _folder(self, folder: pathlib.Path | str | None) -> pathlib.Path:
        """Returns the absolute path of the folder to save the files in.

        Args:
            folder: the folder that the parameters name, if any: one of the
                clerk's folders ("input", "interim", or "output"), a folder
                in the clerk's root folder, or an absolute path.

        Returns:
            The path of the folder.

        """
        clerk = self._get_clerk()
        if folder is None:
            return pathlib.Path(clerk.interim_folder).resolve()
        if isinstance(folder, str):
            named = getattr(clerk, f'{folder}_folder', None)
            if isinstance(named, pathlib.Path | str):
                return pathlib.Path(named).resolve()
        path = pathlib.Path(folder).expanduser()
        if not path.is_absolute():
            path = pathlib.Path(clerk.root_folder) / path
        return path.resolve()

    def _get_clerk(self) -> nagata.FileManager:
        """Returns `clerk`, making a `Clerk` of the workspace if there is none.

        Returns:
            The file manager.

        """
        if self.clerk is None:
            self.clerk = clerks.Clerk()
        return self.clerk


""" Private Functions """


def _safe_label(label: str) -> str:
    """Returns the label of a branch, made safe for a file name.

    Args:
        label: the label of a branch (see `base._BRANCH`).

    Returns:
        The label with each run of characters other than letters, numbers,
            and underscores made a hyphen. A label longer than
            `_MAX_BRANCH_LENGTH` is cut, and ends with a short hash of the
            whole label.

    """
    safe = re.sub(r'[^0-9A-Za-z_]+', '-', label).strip('-')
    if len(safe) > _MAX_BRANCH_LENGTH:
        digest = hashlib.sha1(
            label.encode('utf-8'), usedforsecurity = False).hexdigest()[:8]
        safe = f'{safe[:_MAX_BRANCH_LENGTH - 9]}-{digest}'
    return safe
