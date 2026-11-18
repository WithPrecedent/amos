"""Techniques that load data, usually at the start of the "wrangler" stage.

A loader makes the data of a dataset from a source: a data file, a batch of
files, or a file (or OpenML dataset) that it downloads first. Loaders only
download and import: whatever a file holds, they bring it into the dataset
as it is, and parsers, cleaners, and mungers then work on it there. Every
loader works through the project's clerk (the `nagata.FileManager` that is
the `clerk` of a `Project`): files are looked for, and downloads are saved,
in the clerk's input folder, and the clerk loads every format that it knows.
A project whose workflow has a loader needs no `item`.

The tools that the loaders use are public too, so they can be used outside
of a project: `download_file` and `download_linked_file` download files,
`page_links` and `find_links` read the links of web pages, and
`read_chunks` reads chosen rows and columns of a huge CSV file, a chunk at a
time.

Contents:
    Loader: base class for techniques that load data.
    Download: downloads a file from a URL, or a dataset from OpenML, and loads
        it.
    LoadFile: loads a data file.
    LoadFiles: loads a batch of files into one table.
    LoadChunks: loads chosen rows and columns of a huge CSV file, a chunk at
        a time.
    download_file: downloads a file, unless it was downloaded before.
    download_linked_file: downloads the file that a link on a web page names.
    find_links: returns the addresses of the links on a web page with chosen
        words.
    open_session: returns a session that identifies `amos` to servers.
    page_links: returns the address and words of each link on a web page.
    read_chunks: yields chosen rows and columns of a huge CSV file, a chunk at
        a time.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'Download',
    'LoadChunks',
    'LoadFile',
    'LoadFiles',
    'Loader',
    'download_file',
    'download_linked_file',
    'find_links',
    'open_session',
    'page_links',
    'read_chunks']

from ._base import Loader
from ._chunks import LoadChunks, read_chunks
from ._download import Download
from ._files import LoadFile, LoadFiles
from ._web import (
    download_file,
    download_linked_file,
    find_links,
    open_session,
    page_links,
)

# The public names say that they are in `amos.loaders` (rather than in the
# private module that defines them, or in `amos.wrangler.loaders`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.loaders.Name`. `amos` makes `amos.loaders` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.loaders'
