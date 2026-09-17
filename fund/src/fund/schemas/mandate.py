"""Operational mandate for a fund run (D13) — pure serialisable data.

``PortfolioMandate`` carries the *how it runs* knobs (capital, base currency,
drift threshold, HITL gates, run triggers, optional benchmark) that the future
agents read once per run. It is deliberately data-only: no ``optimizer`` /
``deepagents`` / ``app`` import, so constructing a mandate stays cheap and
optimizer-free (the MiFID→knob mapping lives in ``ConstraintSet`` / ``ViewSet``).

This module also establishes the Phase-4 pydantic **v2** ``frozen=True`` pattern
(``model_config = ConfigDict(frozen=True)``) reused by the other schemas: frozen
models are hashable + serialisable, matching the optimizer's frozen-dataclass
ethos. ``Decimal`` capital round-trips through JSON as a string.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class RunTriggers(BaseModel):
    """What causes a run to fire (D6): a cron schedule and/or a drift breach."""

    model_config = ConfigDict(frozen=True)

    cron: bool
    drift: bool


class PortfolioMandate(BaseModel):
    """Per-portfolio operational mandate (D13).

    Fields hold only primitives/tuples plus the nested frozen ``RunTriggers`` —
    fully serialisable. ``capital`` and ``drift_l1_threshold`` are strictly
    positive; ``base_currency`` is an ISO-4217 alpha code (D8). ``hitl_gates``
    lists the tool names gated behind a human approval (D10) and defaults to the
    single ``place_orders`` gate.
    """

    model_config = ConfigDict(frozen=True)

    portfolio_id: str
    capital: Decimal = Field(gt=0)  # positive notional (D13)
    base_currency: str = Field(pattern=r"^[A-Z]{3}$")  # ISO-4217 alpha (D8)
    drift_l1_threshold: float = Field(gt=0)  # L1 drift band (D12/D29)
    hitl_gates: tuple[str, ...] = ("place_orders",)  # gated tools (D10)
    triggers: RunTriggers  # run causes (D6)
    benchmark: str | None = None  # optional tracking benchmark (D36)


__all__ = ["PortfolioMandate", "RunTriggers"]
