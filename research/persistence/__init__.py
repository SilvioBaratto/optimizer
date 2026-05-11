"""DB write-back persistence for the research pipeline.

Re-exports from ``_snapshot.py`` — the canonical location for research-run
snapshot persistence and config-diff helpers.
"""

from research.persistence._snapshot import (
    _diff_from_default,
    _flatten_metrics,
    persist_research_run,
)

__all__ = [
    "_diff_from_default",
    "_flatten_metrics",
    "persist_research_run",
]
