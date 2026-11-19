"""Draws the chart of the named pipelines on `docs/pipelines.md`.

The page runs `chart` with markdown-exec when the documentation is built, so
the chart always matches `amos.pipelines.PIPELINES`. The chart is HTML (with
Markdown headings between its parts): a
strip of the three workers that a pipeline fills in, a table of the
pipelines for kinds of studies, and a table of the pipeline of each model,
grouped by the family of the model (found by how the pipeline prepares,
criticizes, and draws it). Its styles are in `docs/css/pipelines.css`. To
see the HTML without building the documentation:

    uv run python docs/scripts/pipeline_chart.py

Contents:
    chart: returns the HTML of the chart.
    family_of: returns the family of the model of a pipeline.
    _FAMILIES: the families of models, with a title and a note for each.
    _STEP_ORDER: the steps of an analyst, in the order they are applied.
    _SKIP: the HTML of a step that a pipeline skips.

"""

from __future__ import annotations

import html
import io
from collections.abc import Iterable, Sequence

import amos

# The families of models (by the name of their `_Family` in
# `amos.pipelines`), in the order they are drawn, with a title and a note.
_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ('_TREES', 'Trees',
     'Not scaled; classes weighted (or resampled, for AdaBoost). '
     'Importances, SHAP values, partial dependence, and a tree plot.'),
    ('_MISSING_AWARE_TREES', 'Boosted trees that handle missing values',
     'Missing values left in; not scaled. The evaluators and plots of '
     'trees.'),
    ('_PENALIZED', 'Penalized linear models',
     'Standardized features. Coefficients as importances; a Q-Q plot.'),
    ('_LEAST_SQUARES', 'Least squares',
     'Not scaled; categories encoded against a reference level.'),
    ('_SCALE_SENSITIVE', 'Distances, kernels, and networks',
     'Standardized features. Permutation importance and partial '
     'dependence.'),
    ('_GAUSSIAN', 'Gaussian',
     'Features made more normal (Yeo-Johnson); a calibration curve.'),
    ('_ADDITIVE', 'Additive',
     'Missing values left in; the shape function of each feature.'),
    ('_PRETRAINED', 'Pretrained',
     'TabPFN fills, scales, and weighs the features itself.'),
    ('_STATISTICAL', 'Models for inference',
     'Categories encoded against a reference level; not scaled or '
     'resampled. A regression table and Q-Q, influence, and coefficient '
     'plots.'),
    ('_SURVIVAL', 'Survival',
     'Hazard ratios and Kaplan-Meier survival curves.'),
    ('_BASELINE', 'Baseline',
     'Ignores the features: the score that every other model must beat.'))
# The steps of an analyst, in the order they are applied.
_STEP_ORDER: tuple[str, ...] = (
    'split', 'impute', 'scale', 'encode', 'select', 'sample', 'model')
# The HTML of a step that a pipeline skips.
_SKIP: str = '<span class="pc-skip" title="skipped">—</span>'


""" Public Functions """


def chart() -> str:
    """Returns the chart of the named pipelines.

    Returns:
        Markdown with blocks of HTML: the strip of workers, the table of the
            pipelines for kinds of studies, the table of the pipeline of
            each model (with buttons that show the pipelines of one task),
            and the script of the buttons. The sections have Markdown
            headings, so they are in the page's table of contents.

    """
    models = {
        pipeline.name for pipeline in amos.pipelines._MODEL_PIPELINES}
    pipelines = amos.pipelines.PIPELINES.values()
    studies = [p for p in pipelines if p.name not in models]
    each = [p for p in pipelines if p.name in models]
    blocks = [
        _workers(),
        '## Pipelines for studies',
        f'{len(studies)} pipelines are named for a kind of study. Those that '
        'predict compare several models in an experiment. Those for '
        'inference fit one model and keep its coefficients in the units of '
        'the features, so they neither scale nor resample.',
        _study_table(studies),
        '## Pipelines of each model',
        f'{len(each)} pipelines, one for each model and each task that it '
        'can do. Each prepares the data only as its model needs, so they are '
        'grouped by the family of the model.',
        _filter(each),
        f'<p class="pc-legend"><span>{_SKIP} step skipped</span><span>'
        '<code>grid</code> or <code>random</code>: a hyperparameter search, '
        'scored by balanced accuracy or RMSE</span></p>',
        _model_table(each),
        _SCRIPT]
    return '\n\n'.join(block.strip() for block in blocks) + '\n'


def family_of(pipeline: amos.Pipeline) -> str | None:
    """Returns the family of the model of a pipeline.

    A family is found by how the pipeline prepares its model (whether it
    fills missing values, its scaler, and whether it encodes categories
    against a reference level) and the evaluators and plots that it chooses
    among.

    Args:
        pipeline: the pipeline of one model.

    Returns:
        The name of the family's `_Family` in `amos.pipelines` (such as
            "_TREES"), or `None` if it matches none of `_FAMILIES`.

    """
    analyst = pipeline.analyst
    parameters = pipeline.parameters
    evaluators = parameters.get('auto_evaluator', {}).get('choices')
    plots = parameters.get('auto_plot', {}).get('choices')
    scale = (analyst.get('scale_techniques') or [None])[0]
    for name, _, _ in _FAMILIES:
        family = getattr(amos.pipelines, name)
        if (
                evaluators == list(family.evaluators)
                and plots == list(family.plots)
                and ('impute' in analyst['steps']) == family.impute
                and scale == family.scale
                and ('auto_encode' in parameters) == family.reference):
            return name
    return None


""" Private Functions """


def _chips(names: Iterable[str]) -> str:
    """Returns names as chips.

    Args:
        names: names of techniques or settings.

    Returns:
        The HTML of a `code` chip for each name.

    """
    return ''.join(
        f'<code class="pc-chip">{html.escape(name)}</code>' for name in names)


def _filter(pipelines: Sequence[amos.Pipeline]) -> str:
    """Returns the buttons that show the pipelines of one task.

    Args:
        pipelines: the pipelines of the table that the buttons filter.

    Returns:
        The HTML of the buttons, each with the number of pipelines it shows.

    """
    counts = {
        'all': len(pipelines),
        'classify': sum(p.task == 'classify' for p in pipelines),
        'regress': sum(p.task == 'regress' for p in pipelines)}
    buttons = ''.join(
        f'<button type="button" data-task="{task}" '
        f'aria-pressed="{str(task == "all").lower()}">{task.capitalize()} '
        f'<span>{count}</span></button>'
        for task, count in counts.items())
    return (
        '<div class="pc-filter" role="group" aria-label="Show the pipelines '
        f'of a task">{buttons}</div>\n')


def _model_table(pipelines: Sequence[amos.Pipeline]) -> str:
    """Returns the table of the pipeline of each model, grouped by family.

    Args:
        pipelines: the pipelines of each model.

    Returns:
        The HTML of the table, in a container that scrolls sideways. A
            family's row comes before the rows of its pipelines, and
            pipelines that match no family come last, under "Other".

    """
    steps = _steps_in(pipelines, leave_out = ('model',))
    groups: dict[str | None, list[amos.Pipeline]] = {}
    for pipeline in pipelines:
        groups.setdefault(family_of(pipeline), []).append(pipeline)
    titles = [(name, title, note) for name, title, note in _FAMILIES]
    titles.append((None, 'Other', 'Models that match no family above.'))
    out = io.StringIO()
    out.write(_table_head(steps, ['search'], ['validator'], ['extra plots']))
    for name, title, note in titles:
        members = sorted(
            groups.get(name, []),
            key = lambda p: (_model_of(p), p.task))
        if not members:
            continue
        tasks = ' '.join(sorted({p.task for p in members}))
        out.write(
            f'<tr class="pc-family" data-tasks="{tasks}"><td colspan='
            f'"{len(steps) + 4}"><strong>{html.escape(title)}</strong> '
            f'<span>{html.escape(note)}</span></td></tr>\n')
        for pipeline in members:
            analyst = pipeline.analyst
            out.write(f'<tr data-task="{pipeline.task}">')
            out.write(f'<td>{_name_cell(pipeline, describe = False)}</td>')
            for step in steps:
                cell = (
                    _chips(analyst[f'{step}_techniques'])
                    if step in analyst['steps'] else _SKIP)
                out.write(f'<td class="pc-a">{cell}</td>')
            search = pipeline.parameters.get(
                _model_of(pipeline), {}).get('search')
            out.write(
                f'<td class="pc-a">{_chips([search]) if search else _SKIP}'
                '</td>')
            out.write(
                f'<td class="pc-c">{_chips(pipeline.critic["techniques"][:1])}'
                '</td>')
            extra = pipeline.artist['techniques'][1:]
            out.write(
                f'<td class="pc-r">{_chips(extra) if extra else _SKIP}</td>')
            out.write('</tr>\n')
    out.write('</tbody></table></div>\n')
    return out.getvalue()


def _model_of(pipeline: amos.Pipeline) -> str:
    """Returns the model that the pipeline of one model fits.

    Args:
        pipeline: the pipeline of one model.

    Returns:
        The name of its model.

    """
    return pipeline.analyst['model_techniques'][0]


def _name_cell(pipeline: amos.Pipeline, *, describe: bool) -> str:
    """Returns the cell of a pipeline's name.

    Args:
        pipeline: a named pipeline.
        describe: whether to add the pipeline's description.

    Returns:
        The HTML of the name, the task, the settings that the pipeline
            needs (such as "groups" in the "general" section), and its
            description if `describe` is true.

    """
    cell = (
        f'<span class="pc-name">{html.escape(pipeline.name)}</span>'
        f'<span class="pc-task pc-{pipeline.task}">{pipeline.task}</span>')
    if pipeline.grouped or pipeline.fixed_effects:
        cell += '<span class="pc-needs">needs "groups" in [general]</span>'
    for technique, names in pipeline.required.items():
        needed = ', '.join(f'"{html.escape(name)}"' for name in names)
        cell += (
            f'<span class="pc-needs">needs {needed} in '
            f'[{html.escape(technique)}_parameters]</span>')
    if describe:
        cell += (
            f'<span class="pc-description">'
            f'{html.escape(pipeline.description)}</span>')
    return cell


def _steps_in(
    pipelines: Iterable[amos.Pipeline],
    leave_out: Sequence[str] = ()) -> list[str]:
    """Returns the steps that any of the pipelines' analysts apply.

    Args:
        pipelines: named pipelines.
        leave_out: steps to leave out. Defaults to an empty tuple.

    Returns:
        The steps, in the order of `_STEP_ORDER`.

    """
    used = {step for p in pipelines for step in p.analyst['steps']}
    return [
        step for step in _STEP_ORDER
        if step in used and step not in leave_out]


def _study_table(pipelines: Sequence[amos.Pipeline]) -> str:
    """Returns the table of the pipelines for kinds of studies.

    Args:
        pipelines: the pipelines for kinds of studies.

    Returns:
        The HTML of the table, in a container that scrolls sideways, with
            the techniques of each step of each worker.

    """
    steps = _steps_in(pipelines)
    out = io.StringIO()
    out.write(_table_head(steps, [], ['validate', 'evaluate'], ['draw']))
    for pipeline in pipelines:
        analyst = pipeline.analyst
        out.write('<tr>')
        out.write(f'<td>{_name_cell(pipeline, describe = True)}</td>')
        for step in steps:
            if step not in analyst['steps']:
                out.write(f'<td class="pc-a">{_SKIP}</td>')
                continue
            if step != 'model':
                out.write(
                    f'<td class="pc-a">'
                    f'{_chips(analyst[f"{step}_techniques"])}</td>')
                continue
            if analyst['model_techniques'] == ['auto']:
                note = '<code>auto_model</code> chooses one'
            elif pipeline.fixed_effects:
                note = 'groups as fixed effects, clustered by the first'
            elif analyst.get('design') == 'experiment':
                note = f'compared by {html.escape(analyst["criterion"])}'
            else:
                note = ''
            note = f'<span class="pc-note">{note}</span>' if note else ''
            out.write(
                f'<td class="pc-a">{_chips(pipeline.models)}{note}</td>')
        for technique in pipeline.critic['techniques']:
            out.write(f'<td class="pc-c">{_chips([technique])}</td>')
        out.write(
            f'<td class="pc-r">{_chips(pipeline.artist["techniques"])}</td>')
        out.write('</tr>\n')
    out.write('</tbody></table></div>\n')
    return out.getvalue()


def _table_head(
    steps: Sequence[str],
    more_analyst: Sequence[str],
    critic: Sequence[str],
    artist: Sequence[str]) -> str:
    """Returns the start of a table, through its header.

    Args:
        steps: the analyst's steps that have columns.
        more_analyst: headers of more columns of the analyst.
        critic: headers of the critic's columns.
        artist: headers of the artist's columns.

    Returns:
        The HTML of the table's container, the table's opening tag, its
            header (a row of workers over a row of columns), and the opening
            tag of its body.

    """
    analyst = [*steps, *more_analyst]
    workers = ''.join(
        f'<th colspan="{len(columns)}" class="pc-{key}"><span>{title}</span>'
        '</th>'
        for key, title, columns in (
            ('a', 'Analyst', analyst),
            ('c', 'Critic', critic),
            ('r', 'Artist', artist)))
    columns = ''.join(
        f'<th>{html.escape(column)}</th>'
        for column in (*analyst, *critic, *artist))
    return (
        '<div class="pc-scroll"><table class="pc-table"><thead>'
        f'<tr class="pc-workers"><th></th>{workers}</tr>'
        f'<tr class="pc-columns"><th>pipeline</th>{columns}</tr></thead>'
        '<tbody>\n')


def _workers() -> str:
    """Returns the strip of the three workers that a pipeline fills in.

    Returns:
        The HTML of the analyst's steps (with the steps that only some
            pipelines apply marked), the critic's techniques, and the
            artist's techniques.

    """
    always = {'split', 'encode', 'model'}
    steps = '<i>→</i>'.join(
        f'<code class="pc-chip{"" if step in always else " pc-optional"}">'
        f'{step}</code>'
        for step in _STEP_ORDER)
    return (
        '<div class="pc-workers-strip">'
        '<div class="pc-worker pc-a"><strong>Analyst</strong>'
        f'<div class="pc-chain">{steps}</div><span class="pc-note">Dashed '
        'steps are applied only when the model needs them.</span></div>'
        '<div class="pc-worker pc-c"><strong>Critic</strong>'
        '<div class="pc-chain"><code class="pc-chip">validator</code><i>→</i>'
        '<code class="pc-chip">auto_evaluator</code></div><span '
        'class="pc-note">Cross-validates on the training rows, then applies '
        'the evaluators that suit the fitted model.</span></div>'
        '<div class="pc-worker pc-r"><strong>Artist</strong>'
        '<div class="pc-chain"><code class="pc-chip">auto_plot</code><i>+</i>'
        '<code class="pc-chip pc-optional">extra plots</code></div><span '
        'class="pc-note">Draws the figures that suit the model, and any that '
        'the pipeline adds.</span></div>'
        '</div>\n')


# The script of the buttons that show the pipelines of one task. A family's
# row is hidden when none of its pipelines are shown.
_SCRIPT: str = """<script>
(() => {
  const chart = document.querySelector('.md-content');
  const buttons = chart.querySelectorAll('.pc-filter button');
  const show = task => {
    buttons.forEach(button => button.setAttribute(
      'aria-pressed', String(button.dataset.task === task)));
    chart.querySelectorAll('tr[data-task]').forEach(row => {
      row.hidden = task !== 'all' && row.dataset.task !== task;
    });
    chart.querySelectorAll('tr[data-tasks]').forEach(row => {
      row.hidden = task !== 'all' && !row.dataset.tasks.includes(task);
    });
  };
  buttons.forEach(button => button.addEventListener(
    'click', () => show(button.dataset.task)));
})();
</script>
"""


if __name__ == '__main__':
    print(chart())  # noqa: T201
