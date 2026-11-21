"""Draws the figures shown in the README.

The figures come from the breast cancer study in the README itself: this
script reads the `cancer.ini` settings from the README, runs the study, and
saves each figure that its artist drew, its scorecard, and the regression
tables of its best `logit` and `xgboost` branches in `docs/img`. Run it again
whenever the study in the README changes:

    uv run python docs/scripts/readme_figures.py

"""

from __future__ import annotations

import pathlib
import re
import tempfile

import sklearn.datasets

import amos

ROOT = pathlib.Path(__file__).parent.parent.parent
README = ROOT / 'README.md'
FOLDER = ROOT / 'docs' / 'img'
# The models whose best branches have regression tables in the README.
MODELS: tuple[str, ...] = ('logit', 'xgboost')
SETTINGS = re.compile(
    r'<!-- file: cancer.ini -->\s*```ini\n(?P<text>.*?)```', re.DOTALL)


def main() -> None:
    """Runs the README's study and saves its figures."""
    match = SETTINGS.search(README.read_text(encoding = 'utf-8'))
    if match is None:
        message = 'the README has no cancer.ini settings'
        raise ValueError(message)
    with tempfile.TemporaryDirectory() as folder:
        settings = pathlib.Path(folder) / 'cancer.ini'
        settings.write_text(match['text'], encoding = 'utf-8')
        cancer = sklearn.datasets.load_breast_cancer(as_frame = True)
        project = amos.Project.create(
            settings, item = cancer, clerk = folder)
    FOLDER.mkdir(parents = True, exist_ok = True)
    for name, figure in project.result.figures.items():
        path = FOLDER / f'{name}.png'
        figure.savefig(
            path, dpi = amos.options._FIGURE_DPI, bbox_inches = 'tight')
        print(f'saved {path.relative_to(ROOT)}')  # noqa: T201
    path = project.scorecard.to_image(FOLDER / 'scorecard.png', dpi = 150)
    print(f'saved {path.relative_to(ROOT)}')  # noqa: T201
    # The same tables as the README's example of regression tables.
    branches = sorted(
        project.result.branches, key = lambda b: b.score, reverse = True)
    for model in MODELS:
        dataset = next(b.dataset for b in branches if b.steps['model'] == model)
        table = amos.evaluators.RegressionTable.create(
            {model: dataset}, title = f'The best {model} branch')
        path = table.to_image(FOLDER / f'regression_{model}.png', dpi = 150)
        print(f'saved {path.relative_to(ROOT)}')  # noqa: T201


if __name__ == '__main__':
    main()
