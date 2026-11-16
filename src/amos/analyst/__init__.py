"""The steps of the "analyst" stage, which fits models.

Each step is a package of its own: `splitters`, `imputers`, `scalers`,
`encoders`, `mixers`, `selectors`, `reducers`, `samplers`, `models`, and
`validators`. Each can also be imported from `amos` itself (such as
`amos.models`), which is where its techniques are documented. The base class
of the techniques that fit and transform data is in `amos.transformers`.

"""

from __future__ import annotations
