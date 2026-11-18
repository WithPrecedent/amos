"""Converting HTML and XML to plain text.

Contents:
    HtmlToText: converts columns of HTML or XML to plain text.
    html_to_text: converts HTML or XML to plain text.
    _BLOCK_TAGS: tags that start a new line when markup is converted to text.
    _SKIPPED_CLASSES: values of an element's "class" attribute that mark its
        contents as page numbers inserted by a publisher ("star pagination"),
        so its text is left out.
    _SKIPPED_TAGS: tags whose contents are page numbers or code rather than
        words, so their text is left out.
    _VOID_TAGS: tags that never have a closing tag (such as "br").

"""

from __future__ import annotations

import dataclasses
import functools
import html
import html.parser
import importlib
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ...core import utilities
from .._patterns import text_columns as _text_columns
from .._patterns import text_of as _text
from ._base import Munger

# Tags that start a new line when markup is converted to text. Besides those
# of HTML, they include the tags of the XML of some publishers of court
# opinions (such as "opinion", "parties", and "footnote").
_BLOCK_TAGS: frozenset[str] = frozenset({
    'address', 'article', 'author', 'blockquote', 'br', 'caption', 'center',
    'dd', 'div', 'dl', 'dt', 'footnote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'header', 'hr', 'li', 'ol', 'opinion', 'p', 'parties', 'pre', 'section',
    'table', 'td', 'th', 'tr', 'ul'})
# Values of an element's "class" attribute that mark its contents as page
# numbers inserted by a publisher ("star pagination"). Other text rarely
# has these classes.
_SKIPPED_CLASSES: frozenset[str] = frozenset({'page-label', 'star-pagination'})
# Tags whose contents are page numbers or code rather than words.
_SKIPPED_TAGS: frozenset[str] = frozenset({'page-number', 'script', 'style'})
# Tags that never have a closing tag.
_VOID_TAGS: frozenset[str] = frozenset({
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
    'meta', 'source', 'track', 'wbr'})


class _TextExtractor(html.parser.HTMLParser):
    """Collects the text of HTML or XML, with line breaks between blocks.

    `html.parser.HTMLParser` reads the markup and calls the `handle_...`
    methods for each tag and piece of text, which this class records. Feed it
    markup with `feed`, call `close`, and join `parts` to get the text.
    Character references (such as "&sect;") are converted to characters by
    the parser (`convert_charrefs`).

    Attributes:
        parts: the pieces of text found so far, in order, with a line break
            for the start and end of each block.
        stack: the tags that are open (started but not yet closed), from the
            outermost to the innermost, each with whether its contents are
            skipped.
        block_tags: tags that start a new line.
        skipped_tags: tags whose text is left out.
        skipped_classes: classes of elements whose text is left out.

    """

    """ Initialization Methods """

    def __init__(
        self,
        *,
        block_tags: frozenset[str],
        skipped_tags: frozenset[str],
        skipped_classes: frozenset[str]) -> None:
        """Starts with no text.

        Args:
            block_tags: tags that start a new line, in lower case.
            skipped_tags: tags whose text is left out, in lower case.
            skipped_classes: classes of elements whose text is left out.

        """
        super().__init__(convert_charrefs = True)
        self.parts: list[str] = []
        self.stack: list[tuple[str, bool]] = []
        self.block_tags = block_tags
        self.skipped_tags = skipped_tags
        self.skipped_classes = skipped_classes

    """ Properties """

    @property
    def skipping(self) -> bool:
        """Returns whether the current element's text is being skipped.

        Returns:
            Whether any open tag (the current one or one that contains it) is
                skipped.

        """
        return any(skip for _, skip in self.stack)

    """ Public Methods """

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]]) -> None:
        """Records an opening tag.

        A block tag adds a line break. The tag is added to `stack` (unless it
        never has a closing tag), marked as skipped if it is one of
        `skipped_tags` or has a class in `skipped_classes`.

        Args:
            tag: the name of the tag, in lower case.
            attrs: the tag's attributes, as (name, value) pairs.

        """
        classes = set((dict(attrs).get('class') or '').split())
        skip = tag in self.skipped_tags or bool(classes & self.skipped_classes)
        if tag in self.block_tags:
            self.parts.append('\n')
        if tag not in _VOID_TAGS:
            self.stack.append((tag, skip))

    def handle_endtag(self, tag: str) -> None:
        """Records a closing tag, closing any tags left open inside it.

        A block tag adds a line break. The innermost open tag with the same
        name is removed from `stack`, with any tags opened after it (which the
        markup failed to close). A closing tag that was never opened is
        ignored.

        Args:
            tag: the name of the tag, in lower case.

        """
        if tag in self.block_tags:
            self.parts.append('\n')
        names = [name for name, _ in self.stack]
        if tag in names:
            position = len(names) - 1 - names[::-1].index(tag)
            del self.stack[position:]

    def handle_data(self, data: str) -> None:
        """Records text, unless it is inside a skipped element.

        Args:
            data: a piece of text between tags.

        """
        if not self.skipping:
            self.parts.append(data)


@dataclasses.dataclass
class HtmlToText(Munger):
    """Converts columns of HTML or XML to plain text.

    Each value is converted as `amos.mungers.html_to_text` converts it:
    paragraphs and other blocks are put on lines of their own, character
    references (such as "&amp;") become characters, and the text of scripts,
    styles, and page numbers inserted by publishers is left out. Text without
    tags is kept as it is (except for its character references).

    Wraps:
        [`html.parser.HTMLParser`](https://docs.python.org/3/library/html.parser.html)
        from Python.

    """

    def munge(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        skipped_tags: Iterable[str] | None = None,
        skipped_classes: Iterable[str] | None = None,
        block_tags: Iterable[str] | None = None,
        workers: int = 1,
        **kwargs: Any) -> pd.DataFrame:
        """Converts the markup in `columns` to plain text.

        Args:
            data: the data to change.
            columns: columns to change. Defaults to `None`, which changes every
                text column.
            skipped_tags: tags whose text is left out. Defaults to `None`,
                which leaves out "script", "style", and "page-number".
            skipped_classes: values of the "class" attribute of elements
                whose text is left out. Defaults to `None`, which leaves out
                the page numbers that publishers insert ("star-pagination"
                and "page-label").
            block_tags: tags that start a new line. Defaults to `None`, which
                uses the blocks of HTML (such as "p", "div", and "li") and of
                the XML of some publishers (such as "opinion").
            workers: how many processes convert the text at once, which
                helps with many long documents. -1 uses every core. Defaults
                to 1.
            **kwargs: not used.

        Returns:
            The data, with plain text. Empty text becomes a missing value,
                and missing values stay missing.

        """
        convert = functools.partial(
            html_to_text,
            skipped_tags = skipped_tags,
            skipped_classes = skipped_classes,
            block_tags = block_tags)
        names = _text_columns(data) if columns is None else (
            utilities.select_columns(data, columns))
        for column in names:
            text = _text(data, column, self.name)
            present = text.notna().to_numpy()
            values = text[present].tolist()
            if workers == 1 or len(values) <= 1:
                converted = [convert(value) for value in values]
            else:
                joblib = importlib.import_module('joblib')
                parts = np.array_split(
                    np.arange(len(values)),
                    min(len(values), joblib.effective_n_jobs(workers)))
                pieces = joblib.Parallel(n_jobs = workers)(
                    joblib.delayed(_convert_all)(
                        [values[i] for i in part], convert)
                    for part in parts)
                converted = [value for piece in pieces for value in piece]
            result = text.copy()
            result[present] = converted
            data[column] = result.mask(result == '')
        return data


""" Public Functions """


def html_to_text(
    markup: str,
    *,
    skipped_tags: Iterable[str] | None = None,
    skipped_classes: Iterable[str] | None = None,
    block_tags: Iterable[str] | None = None) -> str:
    """Converts HTML or XML to plain text.

    Paragraphs and other blocks are put on lines of their own, character
    references (such as "&sect;") become characters, and the text of
    scripts, styles, and page numbers inserted by publishers ("star
    pagination") is left out. Markup that is not well formed (such as a tag
    that is never closed) is read as a browser would.

    Args:
        markup: HTML or XML. Text without tags is returned unchanged, except
            that character references are converted.
        skipped_tags: tags whose text is left out (with the text of every
            element inside them). Defaults to `None`, which leaves out
            "script", "style", and "page-number".
        skipped_classes: values of the "class" attribute of elements whose
            text is left out. Defaults to `None`, which leaves out the page
            numbers that publishers insert ("star-pagination" and
            "page-label").
        block_tags: tags that start a new line. Defaults to `None`, which
            uses the blocks of HTML (such as "p", "div", and "li") and of the
            XML of some publishers (such as "opinion" and "footnote").

    Returns:
        The text, with blank lines removed and the spaces of each line
            collapsed and trimmed.

    """
    if '<' not in markup:
        return html.unescape(markup)
    extractor = _TextExtractor(
        block_tags = _lowered(block_tags, _BLOCK_TAGS),
        skipped_tags = _lowered(skipped_tags, _SKIPPED_TAGS),
        skipped_classes = (
            _SKIPPED_CLASSES if skipped_classes is None
            else frozenset(_names(skipped_classes))))
    extractor.feed(markup)
    extractor.close()
    lines = ''.join(extractor.parts).splitlines()
    return '\n'.join(
        ' '.join(line.split()) for line in lines if line.strip())


""" Private Functions """


def _convert_all(values: list[str], convert: Any) -> list[str]:
    """Converts a list of markup, in one process.

    Args:
        values: the markup.
        convert: the function that converts one value.

    Returns:
        The text of each value, in the same order.

    """
    return [convert(value) for value in values]


def _lowered(
    names: Iterable[str] | None,
    default: frozenset[str]) -> frozenset[str]:
    """Returns names of tags in lower case, as the parser reports them.

    Args:
        names: the names, or `None`.
        default: the names to use if `names` is `None`.

    Returns:
        The names in lower case.

    """
    if names is None:
        return default
    return frozenset(name.lower() for name in _names(names))


def _names(names: Iterable[str]) -> list[str]:
    """Returns names given as a list or as text separated by commas.

    Settings files often give a list as text, such as "script, style".

    Args:
        names: a list of names, or text with names separated by commas.

    Returns:
        The names, trimmed of spaces, without blank ones.

    """
    if isinstance(names, str):
        names = names.split(',')
    return [str(name).strip() for name in names if str(name).strip()]
