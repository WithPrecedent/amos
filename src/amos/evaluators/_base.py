"""The base class of evaluators, and what they share.

Contents:
    Evaluator: base class for techniques that evaluate a model with a table.

"""

from __future__ import annotations

import abc
import dataclasses
import functools
import pathlib
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .. import base, utilities

# Replacements for the characters that LaTeX treats specially. Each character
# is replaced once, so the braces in a replacement are not escaped again.
_LATEX_ESCAPES: dict[int, str] = str.maketrans({
    '\\': r'\textbackslash{}',
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
    '<': r'\textless{}',
    '>': r'\textgreater{}'})


@dataclasses.dataclass
class Evaluator(base.Operation, base.Visualizer, abc.ABC):
    """Base class for techniques that evaluate a model with a table.

    A subclass writes an `evaluate` method, which returns a `DataFrame`. It is
    stored in the dataset's `tables` under the technique's name. A subclass
    names the plot that draws its table as `visual`, which `apply` uses if
    it is given `visualize` (see `base.Visualizer`).

    Args:
        name: name used to refer to the technique in a workflow. Defaults to
            an empty `str`, which uses the snake case name of the class.
        contents: not used by most evaluators. Defaults to `None`.
        parameters: keyword arguments for `evaluate`. Defaults to an empty
            `dict`.

    """

    stage_setting: ClassVar[str] = 'visualize_critics'
    model_specific_setting: ClassVar[str] = 'evaluators'

    """ Required Methods """

    @abc.abstractmethod
    def evaluate(self, item: base.Dataset, **kwargs: Any) -> pd.DataFrame:
        """Returns a table that evaluates the model of `item`.

        Args:
            item: the dataset with a fitted model.
            **kwargs: parameters for the evaluation.

        Returns:
            The table.

        """

    """ Public Methods """

    def apply(
        self,
        item: base.Dataset,
        *,
        visualize: bool = False,
        **kwargs: Any) -> base.Dataset:
        """Adds the table from `evaluate` to the tables of `item`.

        Args:
            item: the dataset with a fitted model.
            visualize: whether to draw the table as a figure (with the plot
                named by `visual`), which is added to the figures of `item`
                under the technique's name. The dataset's
                `visualize_critics` (the setting of the "general" section)
                replaces it, if it is set. Defaults to `False`.
            **kwargs: parameters for `evaluate`.

        Returns:
            The dataset, with the new table (and figure).

        """
        item.tables[self.name] = self.evaluate(item, **kwargs)
        item.record(self.name, table = self.name)
        self._visualize(item, visualize, kwargs)
        return item


class _Published:
    """Saves a table as csv, Markdown, LaTeX, HTML, Word, or an image.

    A mixin for the evaluators that make a table to publish (`Scorecard` and
    `RegressionTable`). The class that uses it has a `table`, the `digits`
    shown after the decimal point, a `heading`, and a `name` (as every
    technique does), and it can
    override the hooks that shade a row (`_shaded`), draw a rule above rows
    (`_rules`), and add notes below the table (`_notes`).

    """

    digits: int
    table: pd.DataFrame | None
    # What the table is called in the message of an empty table.
    noun: ClassVar[str] = 'table'

    """ Properties """

    @property
    def heading(self) -> str:
        """Returns the title of the table.

        Raises:
            NotImplementedError: always, since each kind of table has its
                own title.

        Returns:
            The title.

        """
        raise NotImplementedError

    """ Public Methods """

    def export(
        self,
        folder: pathlib.Path | str,
        *,
        name: str | None = None,
        formats: Sequence[str] = ('csv', 'md', 'docx', 'png'),
        ) -> dict[str, pathlib.Path]:
        """Saves the table in several formats.

        Args:
            folder: folder to save the files in. It is created if needed.
            name: name of the files, without extensions. Defaults to `None`,
                in which case the name of the technique is used.
            formats: extensions of the files to save: "csv", "md", "tex",
                "html", "docx", and any image format that matplotlib saves
                (such as "png", "svg", or "pdf"). Defaults to csv, md, docx,
                and png.

        Returns:
            The path of each file, by its format.

        """
        folder = pathlib.Path(folder)
        folder.mkdir(parents = True, exist_ok = True)
        paths = {}
        for extension in formats:
            path = folder / f'{name or getattr(self, "name", self.noun)}.{extension}'
            if extension == 'csv':
                self.to_csv(path)
            elif extension == 'md':
                self.to_markdown(path)
            elif extension == 'tex':
                self.to_latex(path)
            elif extension == 'html':
                self.to_html(path)
            elif extension == 'docx':
                self.to_word(path)
            else:
                self.to_image(path)
            paths[extension] = path
        return paths

    def to_csv(self, path: pathlib.Path | str | None = None) -> str:
        """Returns the table as csv text, and saves it if `path` is given.

        Args:
            path: file to save the text in. Defaults to `None`.

        Returns:
            The csv text.

        """
        text = self._csv_table().to_csv(index = False, lineterminator = '\n')
        if path is not None:
            pathlib.Path(path).write_text(text, encoding = 'utf-8')
        return text

    def to_figure(self) -> Any:
        """Returns the table drawn in a `matplotlib` figure.

        The header is bold, and any shaded row (such as the best branch of a
        scorecard) is shaded. Notes are written below the table.

        Returns:
            A `matplotlib.figure.Figure`.

        """
        figures = utilities.import_tool('matplotlib.figure')
        text = self._text()
        notes = self._notes()
        shaded = self._shaded()
        rows = len(text)
        lengths = [
            max([len(str(c)), *(len(v) for v in text[c])]) + 2
            for c in text.columns]
        width = max(4.0, 0.085 * sum(lengths))
        extra = 0.3 if notes else 0.0
        height = 0.3 * (rows + 1) + 0.5 + extra
        figure = figures.Figure(figsize = (width, height))
        axes = figure.add_axes((0, 0, 1, 1))
        axes.set_axis_off()
        bottom = extra / height
        grid = axes.table(
            cellText = text.to_numpy(),
            colLabels = list(text.columns),
            colWidths = [length / sum(lengths) for length in lengths],
            cellLoc = 'center',
            bbox = (
                0,
                bottom,
                1,
                (1 - bottom) * (rows + 1) / (rows + 1 + 1.6)))
        grid.auto_set_font_size(value = False)
        grid.set_fontsize(9)
        for (row, _), cell in grid.get_celld().items():
            cell.set_edgecolor('#bbbbbb')
            if row == 0:
                cell.set_text_props(weight = 'bold')
                cell.set_facecolor('#e6e6e6')
            elif shaded is not None and row == shaded + 1:
                cell.set_facecolor('#e3f1e3')
        figure.text(
            0.5, 0.98, self.heading,
            ha = 'center', va = 'top', fontsize = 11, weight = 'bold')
        if notes:
            figure.text(
                0.0, bottom / 2, notes, ha = 'left', va = 'center', fontsize = 8)
        return figure

    def to_html(self, path: pathlib.Path | str | None = None) -> str:
        """Returns the table as HTML, and saves it if `path` is given.

        The table is made by great_tables, with the heading as its title,
        numbers aligned to the right, any shaded row shaded, and the notes
        as a source note.

        Args:
            path: file to save the table in. Defaults to `None`.

        Returns:
            The HTML.

        """
        great_tables = utilities.import_tool('great_tables')
        text = self._text()
        numeric = [c for c in text.columns if c in self._numeric_columns()]
        table = great_tables.GT(text).tab_header(title = self.heading)
        if numeric:
            table = table.cols_align(align = 'right', columns = numeric)
        shaded = self._shaded()
        if shaded is not None:
            table = table.tab_style(
                style = great_tables.style.fill(color = '#e3f1e3'),
                locations = great_tables.loc.body(rows = [shaded]))
        notes = self._notes()
        if notes:
            table = table.tab_source_note(source_note = notes)
        html = str(table.as_raw_html())
        if path is not None:
            pathlib.Path(path).write_text(html, encoding = 'utf-8')
        return html

    def to_image(
        self,
        path: pathlib.Path | str,
        dpi: int = 200) -> pathlib.Path:
        """Saves the table as an image.

        Args:
            path: file to save the image in. Its extension (such as ".png",
                ".svg", or ".pdf") sets the format.
            dpi: dots per inch of an image with pixels. Defaults to 200.

        Returns:
            The path of the image.

        """
        path = pathlib.Path(path)
        self.to_figure().savefig(path, dpi = dpi, bbox_inches = 'tight')
        return path

    def to_latex(self, path: pathlib.Path | str | None = None) -> str:
        r"""Returns the table as LaTeX, and saves it if `path` is given.

        The table uses the booktabs package, so add `\usepackage{booktabs}`
        to the preamble of the document. Its caption is the heading, and the
        notes are a row below the bottom rule.

        Args:
            path: file to save the table in. Defaults to `None`.

        Returns:
            The LaTeX table.

        """
        text = self._text()
        numeric = self._numeric_columns()
        rules = set(self._rules())
        alignment = ''.join('r' if c in numeric else 'l' for c in text.columns)
        lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            rf'\caption{{{_latex(self.heading)}}}',
            rf'\begin{{tabular}}{{{alignment}}}',
            r'\toprule',
            ' & '.join(_latex(str(c)) for c in text.columns) + r' \\',
            r'\midrule']
        for position, row in enumerate(text.itertuples(index = False)):
            if position in rules:
                lines.append(r'\midrule')
            lines.append(' & '.join(_latex(value) for value in row) + r' \\')
        lines.append(r'\bottomrule')
        notes = self._notes()
        if notes:
            lines.append(
                rf'\multicolumn{{{len(text.columns)}}}{{l}}'
                rf'{{\footnotesize {_latex(notes)}}} \\')
        lines.extend([r'\end{tabular}', r'\end{table}'])
        latex = '\n'.join(lines) + '\n'
        if path is not None:
            pathlib.Path(path).write_text(latex, encoding = 'utf-8')
        return latex

    def to_markdown(self, path: pathlib.Path | str | None = None) -> str:
        """Returns the table as Markdown, and saves it if `path` is given.

        Numbers are aligned to the right, and the notes are a paragraph
        below the table.

        Args:
            path: file to save the table in. Defaults to `None`.

        Returns:
            The Markdown table.

        """
        text = self._text()
        numeric = self._numeric_columns()
        lines = [
            _markdown_row(str(c) for c in text.columns),
            _markdown_row(
                '---:' if c in numeric else '---' for c in text.columns)]
        lines.extend(
            _markdown_row(row) for row in text.itertuples(index = False))
        notes = self._notes()
        if notes:
            lines.extend(['', notes])
        markdown = '\n'.join(lines) + '\n'
        if path is not None:
            pathlib.Path(path).write_text(markdown, encoding = 'utf-8')
        return markdown

    def to_word(self, path: pathlib.Path | str) -> pathlib.Path:
        """Saves the table in a Word document.

        The document has the heading, a table in the "Table Grid" style
        (with a bold header and numbers aligned to the right), and the notes
        below it.

        Args:
            path: file to save the document in (usually ending in ".docx").

        Returns:
            The path of the document.

        """
        docx = utilities.import_tool('docx')
        alignment = utilities.import_tool('docx.enum.text.WD_ALIGN_PARAGRAPH')
        text = self._text()
        numeric = self._numeric_columns()
        document = docx.Document()
        document.add_heading(self.heading, level = 2)
        grid = document.add_table(
            rows = len(text) + 1, cols = len(text.columns))
        grid.style = 'Table Grid'
        for column, label in enumerate(text.columns):
            cell = grid.cell(0, column)
            cell.text = str(label)
            for run in cell.paragraphs[0].runs:
                run.font.bold = True
        for row, values in enumerate(text.itertuples(index = False), start = 1):
            for column, value in enumerate(values):
                cell = grid.cell(row, column)
                cell.text = value
                if text.columns[column] in numeric:
                    cell.paragraphs[0].alignment = alignment.RIGHT
        notes = self._notes()
        if notes:
            document.add_paragraph(notes)
        path = pathlib.Path(path)
        document.save(str(path))
        return path

    """ Private Methods """

    def _csv_table(self) -> pd.DataFrame:
        """Returns the table that `to_csv` saves: `table` itself.

        Returns:
            The table.

        """
        return self._require_table()

    def _notes(self) -> str | None:
        """Returns notes to write below the table, if any.

        Returns:
            The notes, or `None` if there are none.

        """
        return None

    def _numeric_columns(self) -> set[str]:
        """Returns the names of the columns of `table` that hold numbers.

        Returns:
            The names of the columns, which are aligned to the right.

        """
        table = self._require_table()
        return {
            c for c in table.columns
            if pd.api.types.is_numeric_dtype(table[c].dtype)}

    def _require_table(self) -> pd.DataFrame:
        """Returns `table`.

        Raises:
            ValueError: if there is no table yet.

        Returns:
            The most recent table.

        """
        if self.table is None:
            message = (
                f'the {self.noun} is empty: make one with '
                f'{type(self).__name__}.create or apply it to a dataset first'
            )
            raise ValueError(message)
        return self.table

    def _rules(self) -> list[int]:
        """Returns the positions of the rows to draw a rule above.

        Returns:
            The positions of the rows (from 0), which may be empty.

        """
        return []

    def _shaded(self) -> int | None:
        """Returns the position of the row to shade, if any.

        Returns:
            The position of the row (from 0), or `None` to shade none.

        """
        return None

    def _text(self) -> pd.DataFrame:
        """Returns `table` with every value written as text.

        Returns:
            The table, with numbers rounded to `digits` and missing values
                shown as a dash.

        """
        return self._require_table().map(
            functools.partial(_written, digits = self.digits))


""" Private Functions """


def _latex(text: str) -> str:
    """Returns `text` with the characters that LaTeX treats specially escaped.

    Args:
        text: text for a LaTeX table.

    Returns:
        The escaped text. A dash for a missing value becomes "---".

    """
    if text == '—':
        return '---'
    return text.translate(_LATEX_ESCAPES)


def _markdown_row(values: Iterable[str]) -> str:
    """Returns a row of a Markdown table.

    Args:
        values: the text of each cell.

    Returns:
        The row, with any "|" in the text escaped.

    """
    cells = [str(value).replace('|', r'\|') for value in values]
    return f'| {" | ".join(cells)} |'


def _model(item: base.Dataset) -> Any:
    """Returns the fitted model of `item`.

    Args:
        item: the dataset.

    Raises:
        ValueError: if there is no model.

    Returns:
        The fitted model.

    """
    if item.model is None:
        message = 'there is no fitted model: apply a model first'
        raise ValueError(message)
    return item.model


def _written(value: Any, digits: int) -> str:
    """Returns `value` written as text for a published table.

    Args:
        value: a value of a table.
        digits: digits after the decimal point of a number.

    Returns:
        Text as it is, whole numbers as they are, other numbers rounded to
            `digits`, and a dash for a missing value.

    """
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return '—'
    if isinstance(value, bool | int | np.integer):
        return str(value)
    if isinstance(value, float | np.floating):
        return f'{value:.{digits}f}'
    return str(value)
