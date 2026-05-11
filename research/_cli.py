"""Backward-compatibility shim — re-exports from ``research.cli``.

Once all callers have migrated to ``from research.cli import ...``,
this file can be removed (Cycle 5).
"""

import research.cli
from research.cli import *  # noqa: F403

__all__ = list(research.cli.__all__)
