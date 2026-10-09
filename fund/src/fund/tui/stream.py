"""Live-deliberation stream DTOs — the UI-ready event the worker marshals to the view.

:func:`fund.agents.graph.stream_fund` drives a running fund round with
``agent.stream(stream_mode="updates")`` and yields one :class:`StreamEvent` per
meaningful pipeline step, in emission order; the Deliberation view renders them live.

This module is deliberately **stdlib-only** at import time, so ``import
fund.tui.stream`` drags in neither ``deepagents`` nor ``langgraph``:
:class:`StreamEvent` is safe to import at ``fund.tui.views`` module top, while the
producer in ``fund.agents.graph`` imports it lazily inside the worker-driven generator.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any


class StepStatus(str, Enum):
    """A pipeline step's liveness marker in the Deliberation timeline."""

    LIVE = "live"
    DONE = "done"
    PENDING = "pending"

    @property
    def glyph(self) -> str:
        """Return the single-character marker the timeline renders for this status."""
        return _STATUS_GLYPHS[self]


_STATUS_GLYPHS = {
    StepStatus.LIVE: "●",
    StepStatus.DONE: "✓",
    StepStatus.PENDING: "·",
}


@dataclass(frozen=True)
class StreamEvent:
    """One ordered, UI-ready update from a streaming fund round.

    Mirrors :class:`fund.observe.TranscriptEntry` so the live timeline and the replay
    transcript share rendering and the live→replay fallback is seamless.

    Attributes:
        kind: ``"step"`` for a pipeline step, or a single terminal marker ending the
            stream — ``"paused"`` (the ``place_orders`` gate is live and awaiting the
            adviser), ``"incomplete"`` (the pipeline ended with no ticket), or
            ``"completed"``.
        run_id: The streamed run.
        index: Zero-based position in the emission order.
        agent: The acting agent (``economist``/``allocator``/``risk``/``executor``/
            ``orchestrator``).
        step: The step identity (e.g. ``"report"``, ``"place_orders"``), or ``None``.
        status: The liveness marker; the view flips a prior ``LIVE`` step to ``DONE``
            as the next event arrives, so markers need no second producer pass.
        text: A short human summary (the subagent report, or the gate "why").
        payload: The interrupt/decision JSON when present, else ``None``.
    """

    kind: str
    run_id: uuid.UUID
    index: int
    agent: str
    step: str | None
    status: StepStatus
    text: str
    payload: dict[str, Any] | None = None


__all__ = ["StepStatus", "StreamEvent"]
