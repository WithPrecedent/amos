"""Techniques that save a dataset, so that a workflow can go on from it later.

A saver saves the dataset at its point in a workflow, and passes it on
unchanged (except for an entry in its `history`). A long workflow can save
its dataset after its slow steps (such as loading and parsing huge files),
and a later run can go on from there: pass the saved file as the `item` of a
project whose workflow starts after that point, or load it with a loader
(such as `load_file`). Savers are a step of the "wrangler" stage, but they
can come anywhere in a workflow, such as after a model is fitted.

Every saver works through the project's clerk (the `nagata.FileManager` that
is the `clerk` of a `Project`): its files are saved in the clerk's interim
folder, unless the "folder" parameter names another, and are named for the
project and the saver (such as "cancer_save_data"), unless the "file_name"
parameter names them. In a branch of an experiment, the label of the branch
is added to the file name, so that every branch is saved.

Contents:
    Saver: base class for techniques that save a dataset.
    SaveData: saves the whole dataset, and its data as a table, so that a
        later run can go on from it.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'SaveData',
    'Saver']

from ._base import Saver
from ._data import SaveData

# The public names say that they are in `amos.savers` (rather than in the
# private module that defines them, or in `amos.wrangler.savers`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.savers.Name`. `amos` makes `amos.savers` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.savers'
