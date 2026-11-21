"""A griffe extension that leaves the "Contents:" lists out of the API pages.

Module docstrings list their classes and functions under "Contents:", which
helps when reading the source. On the website, mkdocstrings' summary tables
list them instead, so this extension removes the lists when the documentation
is built. It is named in the `extensions` of mkdocstrings in `mkdocs.yml`.

Contents:
    HideContents: removes the "Contents:" section of module docstrings.
    remove_contents: returns a docstring without its "Contents:" section.

"""

from __future__ import annotations

from typing import Any

import griffe

""" Public Functions """


def remove_contents(docstring: str) -> str:
    """Returns `docstring` without its "Contents:" section.

    The section starts at a line that is just "Contents:" and runs through
    the indented lines (and blank lines between them) after it.

    Args:
        docstring: the text of a docstring.

    Returns:
        The text without the section, or `docstring` unchanged when it has
        none.

    """
    lines = docstring.splitlines()
    try:
        start = [line.strip() for line in lines].index('Contents:')
    except ValueError:
        return docstring
    end = start + 1
    while end < len(lines) and (
            not lines[end].strip() or lines[end].startswith((' ', '\t'))):
        end += 1
    # A blank line before the next section belongs to that section.
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    kept = lines[:start] + lines[end:]
    return '\n'.join(kept).strip()


""" Extensions """


class HideContents(griffe.Extension):
    """Removes the "Contents:" section of every module's docstring."""

    """ Public Methods """

    def on_module_instance(
            self,
            *,
            mod: griffe.Module,
            **kwargs: Any) -> None:
        """Removes the section from `mod`'s docstring after griffe loads it.

        Args:
            mod: the module that griffe loaded.
            **kwargs: the other arguments that griffe passes to the hook.

        """
        if mod.docstring is not None:
            mod.docstring.value = remove_contents(mod.docstring.value)
