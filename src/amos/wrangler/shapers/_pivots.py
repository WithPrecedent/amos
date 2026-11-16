"""Shapers that turn rows into columns and back.

Contents:
    LongToWide: spreads the rows of each group into columns.
    WideToLong: stacks columns into rows.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, cast

import camina
import numpy as np
import pandas as pd

from ...core import utilities
from ._base import Shaper, _claim, _keep_index


@dataclasses.dataclass
class LongToWide(Shaper):
    """Spreads the rows of each group into columns, such as one for each year.

    For example, a table with a row for each state in each year (with
    "state", "year", and "income" columns) becomes a table with a row for
    each state and a column of income for each year ("income_2019",
    "income_2020", and so on). It is the opposite of `wide_to_long`.

    Wraps:
        [`pandas.DataFrame.unstack`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.unstack.html)
        from pandas.

    """

    def shape(
        self,
        data: pd.DataFrame,
        *,
        names: str | None = None,
        values: str | None = None,
        stubs: Sequence[str] | str | None = None,
        ids: Sequence[str] | str | None = None,
        separator: str = '_',
        **kwargs: Any) -> pd.DataFrame:
        """Makes a column for each value of `names`.

        Use `values` for one column of values, whose new columns are named
        for the values of `names` alone, or `stubs` for one or more, whose
        new columns begin with their own names.

        Args:
            data: the data to reshape.
            names: the column whose values name the new columns (such as
                "year"). Defaults to `None`, which changes nothing.
            values: the column whose values fill the new columns, which are
                named for the values of `names` (such as "2019" and "2020").
                Defaults to `None`.
            stubs: the columns whose values fill the new columns, which are
                named "{stub}{separator}{name}" (such as "income_2019").
                Defaults to `None`.
            ids: the columns that say which new row each row belongs to
                (such as "state"). Defaults to `None`, in which case they
                are every column other than `names` and those of `values`
                or `stubs`. Columns that are none of these are left out.
            separator: the text between a stub and a name. Defaults to "_".
            **kwargs: not used.

        Raises:
            KeyError: if a column is not in the data.
            ValueError: if neither or both of `values` and `stubs` are
                given, a row has no value in `names`, two rows would fill
                the same cell, or a new column would have the name of
                another column.

        Returns:
            The data with one row for each group of `ids`, in the order in
                which the groups first appear, and the new columns in the
                order of their names. A cell that no row fills is missing.

        """
        if names is None:
            return data
        if (values is None) == (stubs is None):
            message = (
                f'{self.name!r} needs the columns that fill the new columns: '
                f'set either "values" (for one column) or "stubs"'
            )
            raise ValueError(message)
        spread = utilities.select_columns(
            data, camina.listify(values if stubs is None else stubs))
        utilities.select_columns(data, [names])
        used = [names, *spread]
        keys = (
            [c for c in data.columns if c not in used] if ids is None
            else utilities.select_columns(data, ids))
        named: pd.Series = data[names]
        if isinstance(named.dtype, pd.CategoricalDtype):
            # Only the categories that the rows have become columns.
            named = named.astype(named.cat.categories.dtype)
        blank = int(named.isna().sum())
        if blank:
            message = (
                f'{blank} rows have no value in {names!r}, so there is no '
                f'column for them: remove them first (with drop_missing)'
            )
            raise ValueError(message)
        # The number of the new row that each row belongs to, in the order
        # in which the groups first appear.
        rows = (
            data.groupby(keys, sort = False, dropna = False).ngroup()
            if keys else pd.Series(0, index = data.index))
        cells = pd.MultiIndex.from_arrays([rows, named])
        if cells.has_duplicates:
            repeated = data.loc[cells.duplicated(), [*keys, names]].iloc[0]
            message = (
                f'{int(cells.duplicated().sum())} rows have the same ids and '
                f'{names!r} as an earlier row (such as '
                f'{repeated.to_dict()}), and two rows cannot fill one cell: '
                f'remove them (with drop_duplicates) or summarize them (with '
                f'group_rows) first'
            )
            raise ValueError(message)
        wide = data[spread].set_axis(cells).unstack()  # noqa: PD010
        # The column that fills each new column, and the name that it is for.
        made = cast('list[tuple[Any, Any]]', wide.columns.tolist())
        labels = [
            _written(label) if stubs is None
            else f'{stub}{separator}{_written(label)}'
            for stub, label in made]
        for position, label in enumerate(labels):
            _claim(label, [*keys, *labels[:position]], self.name)
        wide = wide.set_axis(labels, axis = 1).reset_index(drop = True)
        first = data.loc[~np.asarray(rows.duplicated()), keys]
        return pd.concat([first.reset_index(drop = True), wide], axis = 1)


@dataclasses.dataclass
class WideToLong(Shaper):
    """Stacks columns into rows, such as a row for each year.

    For example, a table with a row for each state and a column of income for
    each year ("income_2019", "income_2020", and so on) becomes a table with
    a row for each state in each year. The other columns are repeated in
    each of the rows. It is the opposite of `long_to_wide`.

    Wraps:
        [`pandas.concat`](https://pandas.pydata.org/docs/reference/api/pandas.concat.html)
        from pandas.

    """

    def shape(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str] | str | None = None,
        stubs: Sequence[str] | str | None = None,
        name: str = 'variable',
        value: str = 'value',
        separator: str = '_',
        **kwargs: Any) -> pd.DataFrame:
        """Makes a row for each of `columns`, or for each column of `stubs`.

        Use `columns` to stack columns into one column of values, or `stubs`
        to stack sets of columns that begin with the same names.

        Args:
            data: the data to reshape.
            columns: the columns to stack into the `value` column. The
                `name` column says which of them each value came from.
                Defaults to `None`.
            stubs: the beginnings of the names of the columns to stack. Each
                stub becomes a column of the values of the columns named
                "{stub}{separator}{name}", and the `name` column has the rest
                of their names. For example, the stubs "income" and "tax"
                stack "income_2019", "income_2020", "tax_2019", and
                "tax_2020" into "income" and "tax" columns, with 2019 and
                2020 in the `name` column. Defaults to `None`. Without
                `columns` or `stubs`, nothing changes.
            name: name of the column to make that says where each value came
                from. If every one is a number (such as a year), the column
                is numbers. Defaults to "variable".
            value: name of the column of values that `columns` makes. It is
                not used with `stubs`. Defaults to "value".
            separator: the text between a stub and the rest of a name.
                Defaults to "_".
            **kwargs: not used.

        Raises:
            KeyError: if one of `columns` is not in the data, or no column
                begins with a stub.
            ValueError: if both `columns` and `stubs` are given, or a new
                column would have the name of another column.

        Returns:
            The data with a row for each stacked column of each row, in the
                order of the rows and then of the columns. A value that a
                row lacks is missing.

        """
        if columns is None and stubs is None:
            return data
        if columns is not None and stubs is not None:
            message = (
                f'{self.name!r} stacks either "columns" or the columns of '
                f'"stubs", not both'
            )
            raise ValueError(message)
        data = _keep_index(data)
        # The columns to stack into each new column, by what `name` says of
        # them.
        sources: dict[str, dict[str, str]] = {}
        if columns is not None:
            stacked = utilities.select_columns(data, columns)
            sources[value] = {str(c): c for c in stacked}
        else:
            for stub in camina.listify(stubs):
                start = f'{stub}{separator}'
                found = {
                    str(c)[len(start):]: c for c in data.columns
                    if str(c).startswith(start) and len(str(c)) > len(start)}
                if not found:
                    message = f'no column of the data begins with {start!r}'
                    raise KeyError(message)
                sources[stub] = found
        used = {c for found in sources.values() for c in found.values()}
        kept = [c for c in data.columns if c not in used]
        for position, column in enumerate([name, *sources]):
            _claim(column, [*kept, *[name, *sources][:position]], self.name)
        labels = list(dict.fromkeys(
            label for found in sources.values() for label in found))
        if not labels:
            return data
        blocks = []
        for label in labels:
            block = data[kept].copy()
            block[name] = label
            for column, found in sources.items():
                block[column] = (
                    data[found[label]] if label in found
                    # A set of columns that lacks this one is missing here.
                    else data[next(iter(found.values()))].iloc[:0].reindex(
                        data.index))
            blocks.append(block)
        long = pd.concat(blocks, ignore_index = True)
        order = np.argsort(
            np.tile(np.arange(len(data)), len(labels)), kind = 'stable')
        long = long.iloc[order].reset_index(drop = True)
        if stubs is not None:
            numbers = pd.to_numeric(long[name], errors = 'coerce')
            if not numbers.isna().any():
                long[name] = numbers
        return long


""" Private Functions """


def _written(label: Any) -> str:
    """Returns a value as it is written in the name of a column.

    Args:
        label: a value of the column that names new columns.

    Returns:
        The value as text. A whole number that is stored as a float (such as
            2019.0) is written without its decimals.

    """
    if isinstance(label, float) and label.is_integer():
        return str(int(label))
    return str(label)
