"""A loader that downloads data.

Contents:
    _OPENML: the function that `download` uses to fetch OpenML datasets, unless
        its `contents` names another.
    _SCHEMES: schemes of the URLs that `download` accepts.
    _USER_AGENT: sent with each download, since some servers refuse requests
        without one.
    Download: downloads a file from a URL, or a dataset from OpenML, and loads
        it.

"""

from __future__ import annotations

import dataclasses
import pathlib
import shutil
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any

from ...core import options, utilities

if TYPE_CHECKING:
    pass
from ._base import Loader

# The function that `download` uses to fetch OpenML datasets, unless its
# `contents` names another.
_OPENML: str = 'sklearn.datasets.fetch_openml'
# Schemes of the URLs that `download` accepts.
_SCHEMES: frozenset[str] = frozenset({'http', 'https'})
# Sent with each download, since some servers refuse requests without one.
_USER_AGENT: str = 'amos (https://github.com/WithPrecedent/amos)'


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
        **kwargs: Any) -> Any:
        """Downloads the data at `source` (unless it was before) and loads it.

        Args:
            source: the URL of a file (http or https), or the name or id of
                an OpenML dataset.
            file_name: name to save a URL's file as in the clerk's input
                folder, which may include folders. Defaults to `None`, in
                which case the last part of the URL's path is used. Not used
                for OpenML.
            refresh: whether to download a URL's file again even if it was
                downloaded before. Defaults to `False`. Not used for OpenML.
            timeout: seconds to wait for the server of a URL to answer.
                Defaults to `None`, which uses `options._DOWNLOAD_TIMEOUT`
                (read when the file is downloaded). Not used for OpenML.
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
        if scheme not in _SCHEMES:
            message = (
                f'only http and https URLs can be downloaded, not {text!r} '
                f'(any other source must be the name or id of an OpenML '
                f'dataset)'
            )
            raise ValueError(message)
        name = file_name or _name_of(text)
        path = pathlib.Path(self._get_clerk().input_folder) / name
        if refresh or not path.is_file():
            _download(
                text,
                path,
                options._DOWNLOAD_TIMEOUT if timeout is None else timeout)
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


""" Private Functions """


def _download(url: str, path: pathlib.Path, timeout: float) -> None:
    """Saves the file at `url` to `path`.

    The file is written beside `path` and then renamed, so a download that
    fails leaves no file behind to be mistaken for the data.

    Args:
        url: the URL of the file. Its scheme must be in `_SCHEMES`.
        path: where to save the file.
        timeout: seconds to wait for the server to answer.

    """
    path.parent.mkdir(parents = True, exist_ok = True)
    partial = path.with_name(f'{path.name}.part')
    # `Download.load` checks that the scheme of `url` is http or https.
    request = urllib.request.Request(  # noqa: S310
        url, headers = {'User-Agent': _USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout = timeout) as response,  # noqa: S310
            partial.open('wb') as file):
            shutil.copyfileobj(response, file)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok = True)


def _name_of(url: str) -> str:
    """Returns the name of the file at `url`.

    Args:
        url: a URL.

    Raises:
        ValueError: if the URL's path does not end in a name.

    Returns:
        The last part of the URL's path.

    """
    path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
    name = pathlib.PurePosixPath(path).name
    if not name:
        message = (
            f'{url!r} does not end in a file name: set "file_name" (such as '
            f'"data.csv")'
        )
        raise ValueError(message)
    return name
