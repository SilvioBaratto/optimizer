"""DiagnosticEntry — Cycle 4 aggregate-diagnostic row schema (issue #792).

Per-flag entry surfaced inside `DriftDiagnostics.entries`. Carries the same
shape as `FlagInstance` plus an optional `ticker` so frontend strips and
overlays can render per-position badges without re-deriving copy from raw
`PositionFlag` codes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.schemas.portfolio.normalized_position import PositionFlag


class DiagnosticEntry(BaseModel):
    """One entry inside `DriftDiagnostics.entries`."""

    model_config = ConfigDict(frozen=True)

    code: PositionFlag
    reason: str = ""
    reference: str | None = None
    ticker: str | None = None
