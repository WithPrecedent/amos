"""tigramite's causal discovery in time series.

Contents:
    Tigramite: genres of the techniques of each package.
    LPCMCI, PCMCI, PCMCIPlus: tigramite's causal discovery in time series.
    TimeSeriesEffect: tigramite's effect of one series on the label.

"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .. import base, utilities
from ._base import (
    _Z_95,
    Inference,
    Inferer,
    _estimates,
    _quiet,
    _require_treatment,
)

# tigramite's tests of conditional independence, by name.
_TESTS: dict[str, str] = {
    'cmiknn': 'tigramite.independence_tests.cmiknn.CMIknn',
    'parcorr': 'tigramite.independence_tests.parcorr.ParCorr',
    'robust_parcorr':
        'tigramite.independence_tests.robust_parcorr.RobustParCorr'}


@dataclasses.dataclass
class Tigramite(Inferer, abc.ABC):
    """Genre of tigramite's methods of causal discovery in time series.

    The rows are a time series, in order, and the variables are the label
    and the features (which must be numbers, without missing values). The
    method tests which variables, at which lags (up to "max_lag", 5 by
    default), cause which others, with tigramite's test of conditional
    independence "test": "parcorr" (partial correlation, by default),
    "robust_parcorr" (for variables that are not normally distributed), or
    "cmiknn" (conditional mutual information, for relationships that are not
    linear). The table has a row for each link that it finds at the level
    "alpha" (0.05 by default): the cause, the effect, the lag, the kind of
    link ("-->" for a cause, and "o-o" or "x-x" for links whose direction is
    not known or is in conflict), the strength of the link (for "parcorr",
    the partial correlation), and its p-value.

    Wraps:
        - [`tigramite.independence_tests.parcorr.ParCorr`](https://jakobrunge.github.io/tigramite/#tigramite.independence_tests.parcorr.ParCorr)
          from tigramite, for the "parcorr" test.
        - [`tigramite.independence_tests.robust_parcorr.RobustParCorr`](https://jakobrunge.github.io/tigramite/#tigramite.independence_tests.robust_parcorr.RobustParCorr)
          from tigramite, for the "robust_parcorr" test.
        - [`tigramite.independence_tests.cmiknn.CMIknn`](https://jakobrunge.github.io/tigramite/#tigramite.independence_tests.cmiknn.CMIknn)
          from tigramite, for the "cmiknn" test.

    """

    visual: ClassVar[str] = 'causal_graph'

    def infer(
        self,
        item: base.Dataset,
        *,
        max_lag: int = 5,
        alpha: float = 0.05,
        test: str = 'parcorr',
        **kwargs: Any) -> Inference:
        """Finds the causal links among the label and features.

        Args:
            item: the dataset, whose rows are a time series in order.
            max_lag: the longest lag to test. Defaults to 5.
            alpha: level at which links are significant. Defaults to 0.05.
            test: name of the test of conditional independence. Defaults to
                "parcorr".
            **kwargs: not used.

        Returns:
            The links that were found.

        """
        series, names = _series(item, self.name)
        with _quiet():
            results = self._discover(
                series, _independence_test(test, item.seed), max_lag, alpha)
        table = _links(results, names)
        return Inference(
            table = table,
            fitted = results,
            details = {
                'tool': utilities.describe_tool(self.contents),
                'max_lag': max_lag,
                'alpha': alpha,
                'test': test,
                'links': len(table)},
            effect = False)

    """ Private Methods """

    @abc.abstractmethod
    def _discover(
        self,
        series: Any,
        test: Any,
        max_lag: int,
        alpha: float) -> dict[str, Any]:
        """Returns tigramite's results: its graph and the links' statistics.

        Args:
            series: tigramite's data frame of the time series.
            test: tigramite's test of conditional independence.
            max_lag: the longest lag to test.
            alpha: level at which links are significant.

        Returns:
            The graph and the links' statistics.

        """


@dataclasses.dataclass
class LPCMCI(Tigramite):
    """tigramite's LPCMCI: causal discovery in time series with hidden causes.

    Unlike `pcmci_plus`, it allows for causes that are not among the
    variables (latent confounders), so a link may be "o->" (one end not
    known) or "<->" (caused by a hidden variable) as well as "-->".

    Wraps:
        [`tigramite.lpcmci.LPCMCI.run_lpcmci`](https://jakobrunge.github.io/tigramite/#tigramite.lpcmci.LPCMCI.run_lpcmci)
        from tigramite.

    """

    contents: Any = 'tigramite.lpcmci.LPCMCI'

    def _discover(
        self,
        series: Any,
        test: Any,
        max_lag: int,
        alpha: float) -> dict[str, Any]:
        """Returns the results of `LPCMCI.run_lpcmci`.

        Args:
            series: tigramite's data frame of the time series.
            test: tigramite's test of conditional independence.
            max_lag: the longest lag to test.
            alpha: level at which links are significant.

        Returns:
            The graph and the links' statistics.

        """
        tool = utilities.import_tool(self.contents)
        method = tool(series, cond_ind_test = test, verbosity = 0)
        results: dict[str, Any] = method.run_lpcmci(
            tau_min = 0, tau_max = max_lag, pc_alpha = alpha)
        return results


@dataclasses.dataclass
class PCMCI(Tigramite):
    """tigramite's PCMCI: which variables cause which others at later times.

    It finds the lagged causes (at lags of 1 to "max_lag") of each variable,
    with the PC algorithm and then the momentary conditional independence
    test, which keeps false links rare even when the series are
    autocorrelated.

    Wraps:
        [`tigramite.pcmci.PCMCI.run_pcmci`](https://jakobrunge.github.io/tigramite/#tigramite.pcmci.PCMCI.run_pcmci)
        from tigramite.

    """

    contents: Any = 'tigramite.pcmci.PCMCI'

    def _discover(
        self,
        series: Any,
        test: Any,
        max_lag: int,
        alpha: float) -> dict[str, Any]:
        """Returns the results of `PCMCI.run_pcmci`.

        Args:
            series: tigramite's data frame of the time series.
            test: tigramite's test of conditional independence.
            max_lag: the longest lag to test.
            alpha: level at which links are significant.

        Returns:
            The graph and the links' statistics.

        """
        tool = utilities.import_tool(self.contents)
        method = tool(series, cond_ind_test = test, verbosity = 0)
        results: dict[str, Any] = method.run_pcmci(
            tau_min = 1, tau_max = max_lag, pc_alpha = None,
            alpha_level = alpha)
        return results


@dataclasses.dataclass
class PCMCIPlus(Tigramite):
    """tigramite's PCMCI+: causes at later times and at the same time.

    Like `pcmci`, but it also finds links between variables at the same time
    (lag 0), whose direction may not be known ("o-o").

    Wraps:
        [`tigramite.pcmci.PCMCI.run_pcmciplus`](https://jakobrunge.github.io/tigramite/#tigramite.pcmci.PCMCI.run_pcmciplus)
        from tigramite.

    """

    contents: Any = 'tigramite.pcmci.PCMCI'

    def _discover(
        self,
        series: Any,
        test: Any,
        max_lag: int,
        alpha: float) -> dict[str, Any]:
        """Returns the results of `PCMCI.run_pcmciplus`.

        Args:
            series: tigramite's data frame of the time series.
            test: tigramite's test of conditional independence.
            max_lag: the longest lag to test.
            alpha: level at which links are significant.

        Returns:
            The graph and the links' statistics.

        """
        tool = utilities.import_tool(self.contents)
        method = tool(series, cond_ind_test = test, verbosity = 0)
        results: dict[str, Any] = method.run_pcmciplus(
            tau_min = 0, tau_max = max_lag, pc_alpha = alpha)
        return results


@dataclasses.dataclass
class TimeSeriesEffect(Tigramite):
    """tigramite's causal effect of one series (the treatment) on the label.

    It finds the causes of each variable with `pcmci`, fits a linear model of
    each variable on its causes, and multiplies the coefficients along every
    causal path from "treatment" at "lag" (1 by default) earlier to the
    label (tigramite's `LinearMediation`). The effect is in the units of the
    data, and it is 0 if no causal path joins them. Its confidence interval
    comes from "simulations" bootstrap samples (100 by default). The links
    that `pcmci` found are stored in `tables` as "{name}_links".

    Wraps:
        - [`tigramite.pcmci.PCMCI.run_pcmci`](https://jakobrunge.github.io/tigramite/#tigramite.pcmci.PCMCI.run_pcmci)
          from tigramite, to find the causes of each series.
        - [`tigramite.models.LinearMediation`](https://jakobrunge.github.io/tigramite/#tigramite.models.LinearMediation)
          from tigramite, to estimate the effect.

    """

    contents: Any = 'tigramite.models.LinearMediation'
    visual: ClassVar[str] = 'coefficient_plot'

    def infer(
        self,
        item: base.Dataset,
        *,
        treatment: str | None = None,
        lag: int = 1,
        max_lag: int = 5,
        alpha: float = 0.05,
        test: str = 'parcorr',
        simulations: int = 100,
        **kwargs: Any) -> Inference:
        """Estimates the effect of `treatment` on the label `lag` periods later.

        Args:
            item: the dataset, whose rows are a time series in order. It must
                have a label.
            treatment: name of the column with the treatment. Defaults to
                `None`, but it is required.
            lag: number of periods between the treatment and the label.
                Defaults to 1.
            max_lag: the longest lag to test. Defaults to 5.
            alpha: level at which links are significant. Defaults to 0.05.
            test: name of the test of conditional independence. Defaults to
                "parcorr".
            simulations: number of bootstrap samples. Defaults to 100.
            **kwargs: not used.

        Raises:
            ValueError: if there is no treatment, or `lag` is less than 1 or
                more than `max_lag`.

        Returns:
            The estimate, and the links that were found.

        """
        treatment = _require_treatment(treatment, self.name)
        label = item.require_label()
        if not 1 <= lag <= max_lag:
            message = f'lag must be from 1 to max_lag ({max_lag}), not {lag}'
            raise ValueError(message)
        series, names = _series(item, self.name)
        tool = utilities.import_tool(self.contents)
        cause, effect = names.index(treatment), names.index(label)
        arguments = {'i': cause, 'tau': lag, 'j': effect}
        with _quiet():
            results = self._discover(
                series, _independence_test(test, item.seed), max_lag, alpha)
            mediation = tool(series, data_transform = None)
            mediation.fit_model(
                all_parents = _parents(results['graph']), tau_max = max_lag)
            estimate = float(mediation.get_ce(**arguments))
            mediation.fit_model_bootstrap(
                boot_blocklength = 1, seed = item.seed,
                boot_samples = simulations)
            lower, upper = (
                float(v) for v in mediation.get_bootstrap_of(
                    function = 'get_ce', function_args = arguments,
                    conf_lev = 0.95))
        table = _estimates(
            [f'{treatment} (lag {lag})'], [estimate],
            [(upper - lower) / (2 * _Z_95)], lower = [lower], upper = [upper])
        return Inference(
            table = table,
            fitted = mediation,
            tables = {'links': _links(results, names)},
            details = {
                'treatment': treatment,
                'lag': lag,
                'max_lag': max_lag,
                'alpha': alpha,
                'test': test,
                'simulations': simulations})

    """ Private Methods """

    def _discover(
        self,
        series: Any,
        test: Any,
        max_lag: int,
        alpha: float) -> dict[str, Any]:
        """Returns the results of `PCMCI.run_pcmci` (lagged causes only).

        Args:
            series: tigramite's data frame of the time series.
            test: tigramite's test of conditional independence.
            max_lag: the longest lag to test.
            alpha: level at which links are significant.

        Returns:
            The graph and the links' statistics.

        """
        return PCMCI()._discover(series, test, max_lag, alpha)


""" Private Functions """


def _independence_test(name: str, seed: int | None) -> Any:
    """Returns one of tigramite's tests of conditional independence.

    Wraps:
        - [`tigramite.independence_tests.parcorr.ParCorr`](https://jakobrunge.github.io/tigramite/#tigramite.independence_tests.parcorr.ParCorr)
          from tigramite, for "parcorr".
        - [`tigramite.independence_tests.robust_parcorr.RobustParCorr`](https://jakobrunge.github.io/tigramite/#tigramite.independence_tests.robust_parcorr.RobustParCorr)
          from tigramite, for "robust_parcorr".
        - [`tigramite.independence_tests.cmiknn.CMIknn`](https://jakobrunge.github.io/tigramite/#tigramite.independence_tests.cmiknn.CMIknn)
          from tigramite, for "cmiknn".

    Args:
        name: "parcorr", "robust_parcorr", or "cmiknn".
        seed: seed for the shuffles of "cmiknn".

    Raises:
        ValueError: if `name` is not one of those.

    Returns:
        The test, with analytic p-values (or, for "cmiknn", p-values from
            shuffles).

    """
    if name not in _TESTS:
        message = f'test must be one of {sorted(_TESTS)}, not {name!r}'
        raise ValueError(message)
    tool = utilities.import_tool(_TESTS[name])
    if name == 'cmiknn':
        return tool(significance = 'shuffle_test', seed = seed)
    return tool(significance = 'analytic')


def _links(results: dict[str, Any], names: Sequence[str]) -> pd.DataFrame:
    """Returns the causal links in tigramite's results, one for each row.

    Args:
        results: tigramite's results, with its "graph", "val_matrix", and
            "p_matrix".
        names: the names of the variables, in the order of the results.

    Returns:
        The links, with their "cause", "effect", "lag", kind of "link",
            "strength", and "p_value". A link at lag 0 is listed once.

    """
    graph = results['graph']
    rows = [
        {
            'cause': names[i],
            'effect': names[j],
            'lag': tau,
            'link': str(graph[i, j, tau]),
            'strength': float(results['val_matrix'][i, j, tau]),
            'p_value': float(results['p_matrix'][i, j, tau])}
        for i in range(len(names)) for j in range(len(names))
        for tau in range(graph.shape[2])
        if graph[i, j, tau] != '' and (tau > 0 or i < j)]
    columns = ['cause', 'effect', 'lag', 'link', 'strength', 'p_value']
    return pd.DataFrame(rows, columns = columns)


def _parents(graph: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """Returns the lagged causes of each variable in tigramite's graph.

    Args:
        graph: tigramite's graph, whose entry for cause `i`, effect `j`, and
            lag `tau` is "-->" if `i` at `tau` periods before causes `j`.

    Returns:
        For each variable, its causes, as (variable, negative lag) pairs.

    """
    variables, _, lags = graph.shape
    return {
        j: [
            (i, -tau) for i in range(variables) for tau in range(1, lags)
            if graph[i, j, tau] == '-->']
        for j in range(variables)}


def _series(item: base.Dataset, technique: str) -> tuple[Any, list[str]]:
    """Returns tigramite's data frame of the label and features.

    Wraps:
        [`tigramite.data_processing.DataFrame`](https://jakobrunge.github.io/tigramite/#tigramite.data_processing.DataFrame)
        from tigramite.

    Args:
        item: the dataset, whose rows are a time series in order.
        technique: name of the technique, for messages.

    Raises:
        ValueError: if a variable is not a number or has missing values.

    Returns:
        tigramite's data frame, and the names of its variables (the features
            and the label, in the order of the columns).

    """
    names = [
        c for c in item.data.columns
        if c in item.features or c == item.label]
    unusable = [
        c for c in names
        if base._kind_of(item.data[c]) not in {'numerics', 'booleans'}]
    if unusable:
        message = (
            f'{technique!r} needs numbers, but {unusable} are not: encode or '
            f'remove them first'
        )
        raise ValueError(message)
    values = item.data[names].astype(float)
    if values.isna().to_numpy().any():
        message = f'{technique!r} needs a time series without missing values'
        raise ValueError(message)
    processing = utilities.import_tool('tigramite.data_processing')
    return processing.DataFrame(values.to_numpy(), var_names = names), names
