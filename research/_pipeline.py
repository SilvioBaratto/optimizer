"""Backward-compatibility shim — re-exports all symbols from ``research.pipeline``.

Once all callers have migrated to ``from research.pipeline import ...``,
this file can be removed (Cycle 5).
"""

import research.pipeline
from research.pipeline import *  # noqa: F403

__all__ = list(research.pipeline.__all__)
