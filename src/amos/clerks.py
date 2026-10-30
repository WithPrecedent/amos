"""The file manager of a project, which keeps data and results outside it.

A project's code is usually kept in a folder of a "code" folder (such as
"projects/code/my_study"), and amos keeps its data and results beside the
"code" folder, in the "workspace" that holds it:

    projects/                (the workspace)
        code/
            my_study/        (the project's code)
        data/
            crime/nibrs/     (a project's data, in its own subfolder)
        studies/
            my_study_2026-10-30_09-15-00/   (each run's results)

So data that many projects use is kept once, results are never mixed with
the code, and neither has to be left out of a repository by hand. The
workspace is `options._DEFAULT_ROOT`, if a package sets it; otherwise the
folder named by the environment variable in `options._WORKSPACE_VARIABLE`
("AMOS_WORKSPACE"), if it is set; otherwise the folder that holds the
nearest folder named `options._CODE_FOLDER` ("code") above the current
folder; and otherwise the current folder itself.

Contents:
    Clerk: the file manager of a project.
    workspace: returns the folder that holds the data and results.

"""

from __future__ import annotations

import dataclasses
import os
import pathlib

import nagata

from . import options


@dataclasses.dataclass
class Clerk(nagata.FileManager):
    """The file manager of a project, which keeps data and results outside it.

    It is the `clerk` of a `Project` (a `nagata.FileManager`): loaders look
    for data files and save downloads in its `input_folder`, intermediate
    files are kept in its `interim_folder`, and `Project.export` saves each
    run in a folder of its `output_folder` named for the run. By default,
    these are the "data" and "studies" folders of the workspace (see
    `workspace`), so every project keeps its data and results in the same
    places, outside its code. A project's own data can be kept in a
    subfolder of the data folder (such as "crime/nibrs"), named by
    `data_subfolder`.

    Each folder can be set in the "files" section of a project's settings
    (relative folders are in `root_folder`), and the folders are made if
    they do not exist.

    Args:
        root_folder: the workspace, which relative folders are in. Defaults
            to `None`, which uses `workspace()`.
        input_folder: the folder of the data. Defaults to
            `options._DATA_FOLDER` ("data").
        interim_folder: the folder of intermediate files. Defaults to "input",
            the input folder.
        output_folder: the folder in which each run's results are saved.
            Defaults to `options._STUDIES_FOLDER` ("studies").
        framework: the formats of files that the clerk knows (see
            `nagata.FileManager`).
        data_subfolder: the subfolder of `input_folder` that holds the
            project's data (such as "crime/nibrs"), which becomes the input
            folder. Defaults to `None`, which uses `input_folder` itself.

    """

    root_folder: pathlib.Path | str | None = None  # type: ignore[assignment]
    input_folder: pathlib.Path | str = options._DATA_FOLDER
    interim_folder: pathlib.Path | str = 'input'
    output_folder: pathlib.Path | str = options._STUDIES_FOLDER
    data_subfolder: pathlib.Path | str | None = None

    """ Initialization Methods """

    def __post_init__(self) -> None:
        """Finds the workspace and puts the data subfolder in the input folder.

        Then `nagata.FileManager` makes each folder (relative folders in the
        workspace).

        """
        root = workspace() if self.root_folder is None else self.root_folder
        self.root_folder = pathlib.Path(root).expanduser()
        self.root_folder.mkdir(parents = True, exist_ok = True)
        if self.data_subfolder:
            self.input_folder = (
                self.root_folder / self.input_folder / self.data_subfolder)
        super().__post_init__()


""" Public Functions """


def workspace(start: pathlib.Path | str | None = None) -> pathlib.Path:
    """Returns the folder that holds the data and results of projects.

    Args:
        start: the folder to look up from for a "code" folder. Defaults to
            `None`, which uses the current folder.

    Returns:
        `options._DEFAULT_ROOT`, if it is set; otherwise the folder named by
            the environment variable in `options._WORKSPACE_VARIABLE`, if it
            is set; otherwise the folder that holds the nearest folder named
            `options._CODE_FOLDER` at or above `start`; and otherwise
            `start`.

    """
    if options._DEFAULT_ROOT is not None:
        return pathlib.Path(options._DEFAULT_ROOT).expanduser()
    named = os.environ.get(options._WORKSPACE_VARIABLE)
    if named:
        return pathlib.Path(named).expanduser()
    folder = pathlib.Path(start or pathlib.Path.cwd()).resolve()
    for path in (folder, *folder.parents):
        if path.name == options._CODE_FOLDER:
            return path.parent
    return folder
