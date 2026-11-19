"""The steps of the "wrangler" stage, which prepares data.

Each step is a package of its own: `loaders`, `parsers`, `cleaners`,
`mungers`, `mergers`, `shapers`, and `savers`. Each can also be imported
from `amos` itself (such as `amos.cleaners`), which is where its techniques
are documented.

"""

from __future__ import annotations
