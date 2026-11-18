"""A loader that downloads data.

Contents:
    _OPENML: the function that `download` uses to fetch OpenML datasets, unless
        its `contents` names another.
    Download: downloads a file from a URL, or a dataset from OpenML, and loads
        it.

"""

from __future__ import annotations

import dataclasses
import pathlib
import urllib.parse
from typing import Any

from ...core import utilities
from . import _web
from ._base import Loader

# The function that `download` uses to fetch OpenML datasets, unless its
# `contents` names another.
_OPENML: str = 'sklearn.datasets.fetch_openml'


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

    With "link", the source is instead a web page, and the file is the one
    that the link with those words leads to (see
    `amos.loaders.download_linked_file`), which suits sites that link to the
    newest release of their data from the same page.

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
        link: str | None = None,
        required_member: str | None = None,
        resume: bool = False,
        session: Any = None,
        **kwargs: Any) -> Any:
        """Downloads the data at `source` (unless it was before) and loads it.

        Args:
            source: the URL of a file (http or https), of a web page that
                links to the file (with `link`), or the name or id of an
                OpenML dataset.
            file_name: name to save a URL's file as in the clerk's input
                folder, which may include folders. Defaults to `None`, in
                which case the last part of the path of the file's URL is
                used. Not used for OpenML.
            refresh: whether to download a URL's file again even if it was
                downloaded before. Defaults to `False`. Not used for OpenML.
            timeout: seconds to wait for the server of a URL to answer.
                Defaults to `None`, which uses `options._DOWNLOAD_TIMEOUT`
                (read when the file is downloaded). Not used for OpenML.
            link: the words of the link on the page at `source` that leads
                to the file, compared without regard to case or spacing
                (such as "CSV Organized by Supreme Court Citation").
                Defaults to `None`, in which case `source` is the file.
            required_member: a pattern (with "*" and "?") that the name of a
                file in the downloaded zip archive must match before it is
                saved, such as "*caseCentered*". With `link`, each link with
                its words is tried until one passes. Defaults to `None`, in
                which case the download is not checked.
            resume: whether to continue a download of a URL's file that
                stopped, instead of starting over. Defaults to `False`.
            session: an object with a `get` method like a `requests.Session`.
                Defaults to `None`, in which case one that identifies `amos`
                is made.
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
        if scheme not in _web._SCHEMES:
            message = (
                f'only http and https URLs can be downloaded, not {text!r} '
                f'(any other source must be the name or id of an OpenML '
                f'dataset)'
            )
            raise ValueError(message)
        folder = pathlib.Path(self._get_clerk().input_folder)
        if link is not None:
            path = _web.download_linked_file(
                text,
                str(link),
                folder,
                file_name = file_name,
                refresh = refresh,
                timeout = timeout,
                session = session,
                required_member = required_member)
        else:
            path = _web.download_file(
                text,
                folder / (file_name or _web._name_of(text)),
                refresh = refresh,
                timeout = timeout,
                session = session,
                resume = resume,
                required_member = required_member)
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
