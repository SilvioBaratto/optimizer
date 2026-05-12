"""Optional DB write-back for the research pipeline (Cycle 5 §12).

Bridges the research runner to ``PortfolioRepository`` so a single
``research_run`` snapshot can be replayed or audited from the API DB.
The writer is idempotent on ``(portfolio_id, snapshot_date,
snapshot_type)`` — a re-run on the same date overwrites the previous
weights, summary metrics, and optimizer-config diff in place.

Public surface
--------------
- :func:`persist_research_run` — orchestrator.
- :func:`_diff_from_default` — config diff vs. ``MeanRiskConfig()`` defaults
  (re-exported for the report renderer).
- :func:`_flatten_metrics` — dotted-key flattening of nested metric dicts
  (re-exported for the report renderer).
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager
from datetime import date
from typing import Any, Protocol

from sqlalchemy.orm import Session

from optimizer.optimization import MeanRiskConfig

_RESEARCH_PORTFOLIO_NAME = "research_run"
_RESEARCH_SNAPSHOT_TYPE = "research_run"


class _SessionFactory(Protocol):
    def __call__(self) -> AbstractContextManager[Session]: ...


def persist_research_run(
    snapshot_date: date,
    weights: dict[str, float],
    metrics: dict[str, Any],
    optimizer_cfg: MeanRiskConfig,
    sector_mapping: dict[str, str],
    turnover: float | None,
    *,
    portfolio_name: str = _RESEARCH_PORTFOLIO_NAME,
    session_factory: _SessionFactory | None = None,
) -> uuid.UUID:
    """Persist one research run as a ``snapshot_type='research_run'`` snapshot.

    ``metrics`` is the nested ``{"gross": {...}, "net": {...},
    "after_tax": {...}}`` shape; each leaf becomes one
    ``SnapshotSummaryEntry`` keyed by its dotted path
    (e.g. ``"gross.sharpe"``).  ``optimizer_cfg`` is recorded as the diff
    vs. ``MeanRiskConfig()`` defaults — only changed fields are stored.

    Returns the ``Portfolio.id`` of the row that owns the snapshot.
    """
    from app.database import database_manager
    from app.repositories.portfolio.portfolio_repository import PortfolioRepository

    effective_factory: _SessionFactory = (
        session_factory if session_factory is not None else database_manager.get_session
    )
    summary = _flatten_metrics(metrics)
    optimizer_diff = _diff_from_default(optimizer_cfg)
    with effective_factory() as session:
        repo = PortfolioRepository(session)
        portfolio = repo.get_or_create(portfolio_name)
        portfolio_id: uuid.UUID = portfolio.id
        repo.create_snapshot(
            portfolio_id,
            snapshot_date,
            _RESEARCH_SNAPSHOT_TYPE,
            weights,
            sector_mapping=sector_mapping,
            summary=summary,
            optimizer_config=optimizer_diff,
            turnover=turnover,
        )
    return portfolio_id


def _diff_from_default(cfg: MeanRiskConfig) -> dict[str, Any]:
    """Return ``dataclasses.asdict`` fields where ``cfg`` differs from defaults."""
    current = dataclasses.asdict(cfg)
    default = dataclasses.asdict(MeanRiskConfig())
    return {k: v for k, v in current.items() if default.get(k) != v}


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Recursively flatten a nested dict using dotted keys."""
    return dict(_iter_flat(metrics, prefix=""))


def _iter_flat(d: dict[str, Any], *, prefix: str) -> Iterator[tuple[str, Any]]:
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _iter_flat(v, prefix=key)
        else:
            yield key, v


__all__ = [
    "_diff_from_default",
    "_flatten_metrics",
    "persist_research_run",
]
