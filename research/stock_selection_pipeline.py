"""Backward-compat shim — re-exports from ``research.pipeline`` and ``research.cli``.

This file was the ~1954-line monolith.  All business logic has moved to
``research/pipeline/`` (step modules) and ``research/cli/`` (entry point).
Import this module as before to get the same symbols; new code should import
directly from the canonical locations.
"""

import research.cli
import research.pipeline
from research.cli import *  # noqa: F403
from research.pipeline import *  # noqa: F403

__all__ = list(research.pipeline.__all__) + list(research.cli.__all__)
