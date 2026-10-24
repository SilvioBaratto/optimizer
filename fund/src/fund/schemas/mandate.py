"""Operational mandate for a fund run — pure serialisable data.

``PortfolioMandate`` carries the *how it runs* knobs (capital, base currency,
drift threshold, HITL gates, run triggers, optional benchmark) that the
agents read once per run. It is deliberately data-only: no ``optimizer`` /
``deepagents`` / ``app`` import, so constructing a mandate stays cheap and
optimizer-free (the MiFID→knob mapping lives in ``ConstraintSet`` / ``ViewSet``).

This module establishes the pydantic v2 ``frozen=True`` pattern
(``model_config = ConfigDict(frozen=True)``) reused by the other schemas: frozen
models are hashable + serialisable, matching the optimizer's frozen-dataclass
ethos. ``Decimal`` capital round-trips through JSON as a string.
"""

from __future__ import annotations

from decimal import Decimal
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class RunTriggers(BaseModel):
    """What causes a run to fire: a cron schedule and/or a drift breach."""

    model_config = ConfigDict(frozen=True)

    cron: bool
    drift: bool


class PortfolioMandate(BaseModel):
    """Per-portfolio operational mandate.

    Fields hold only primitives/tuples plus the nested frozen ``RunTriggers`` —
    fully serialisable. ``capital`` is strictly positive and ``drift_l1_threshold``
    lies in the open interval ``(0, MAX_DRIFT_L1)`` so the drift trigger can fire;
    ``base_currency`` is an ISO-4217 alpha code. ``hitl_gates`` lists the tool names
    gated behind a human approval and defaults to the single ``place_orders`` gate.
    ``remembered_directives`` carries free-text co-pilot steers the adviser chose to
    persist across rounds; the mandate JSON is their source of truth (no scalar
    mirror, no migration).
    """

    model_config = ConfigDict(frozen=True)

    # Two disjoint fully-invested books sit at L1 distance 2, the most
    # ``observe.drift_l1`` can report. The drift monitor breaches only on
    # ``drift_l1 > threshold``, so a threshold of 2 or more is never crossed and
    # silently disables the drift trigger.
    MAX_DRIFT_L1: ClassVar[float] = 2.0

    portfolio_id: str
    capital: Decimal = Field(gt=0)
    base_currency: str = Field(pattern=r"^[A-Z]{3}$")
    drift_l1_threshold: float = Field(gt=0, lt=MAX_DRIFT_L1)
    hitl_gates: tuple[str, ...] = ("place_orders",)
    triggers: RunTriggers
    # Mirrored into portfolio_mandates.benchmark (VARCHAR(32)); PostgreSQL rejects a
    # longer value at flush, so reject it here as a clean validation error instead.
    benchmark: str | None = Field(default=None, max_length=32)
    remembered_directives: tuple[str, ...] = ()


__all__ = ["PortfolioMandate", "RunTriggers"]
