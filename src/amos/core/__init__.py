"""The framework of amos that every stage shares.

These modules hold what is not a step of a stage: the dataset and the base
class of techniques (`base`), the workers and pipelines that arrange the
stages, the project that runs them (`interface`), file management (`clerks`),
reports, metrics, the choosers of the "auto" techniques, options, and
utilities. Each module can also be imported from `amos` itself (such as
`amos.base`).

"""

from __future__ import annotations
