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
    Download: downloads a file from a URL, or a dataset from OpenML, and loads
        it.
    LoadFile: loads a data file.
    LoadFiles: loads a batch of files into one table.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'Download',
    'LoadFile',
    'LoadFiles',
    'Loader']

from ._base import Loader
from ._download import Download
from ._files import LoadFile, LoadFiles

# The public names say that they are in `amos.loaders` (rather than in the
# private module that defines them, or in `amos.wrangler.loaders`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.loaders.Name`. `amos` makes `amos.loaders` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.loaders'
