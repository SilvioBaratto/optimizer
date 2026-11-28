"""Live-deliberation stream DTOs: the UI-ready event a streamed fund round yields.

:func:`fund.agents.graph.stream_fund` drives a running fund round with
``agent.stream(stream_mode="updates")`` and yields one :class:`StreamEvent` per
meaningful pipeline step, in emission order, then a terminal event unless an error
propagates; the Deliberation view renders them live.

The module sits outside ``fund.agents`` and ``fund.tui`` because both use it: the
agent layer produces the events, the cockpit renders them, and the agent layer must
not import the cockpit. :mod:`fund.tui.stream` re-exports both names.

It imports only the standard library, so importing it pulls in neither
``deepagents`` nor ``langgraph`` and the Deliberation view can import it at module
top.
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

    @property
    def marker(self) -> str:
        """Return the glyph plus its word, so the status never relies on shape alone."""
        return f"{self.glyph} {self.value}"


_STATUS_GLYPHS = {
    StepStatus.LIVE: "●",
    StepStatus.DONE: "✓",
    StepStatus.PENDING: "·",
}


@dataclass(frozen=True)
class StreamEvent:
    """One ordered, UI-ready update from a streaming fund round.

    Shares the ``agent``, ``step``, ``index``, ``text`` and ``payload`` fields of
    :class:`fund.observe.TranscriptEntry`. The Deliberation view formats a live step
    with ``_stream_detail`` and a replayed step with ``_transcript_detail``, two
    separate functions. Of what they render, only the body below the header and text
    lines comes from a shared helper, ``_detail_payload``, which reads ``agent``,
    ``step`` and ``payload``.

    Attributes:
        kind: ``"step"`` for a pipeline step, or a terminal marker that
            ``stream_fund`` yields last, at most once per stream — ``"paused"`` (the
            ``place_orders`` gate is live and awaiting the user) or
            ``"incomplete"`` (no gate opened: the pipeline ended with no ticket, or
            autonomy froze mid-round and the run was rejected).
        run_id: The streamed run.
        index: Zero-based position in the emission order.
        agent: The acting agent: the delegated subagent's type (such as
            ``economist`` or ``risk``), ``subagent`` when no type is known for the
            report, or ``orchestrator`` for the PM's ``place_orders`` step and the
            terminal markers.
        step: The step identity (e.g. ``"report"``, ``"place_orders"``), or ``None``.
        status: The liveness marker; the view flips a prior ``LIVE`` step to ``DONE``
            as the next event arrives, so markers need no second producer pass.
        text: Human-readable text for the timeline, such as a subagent's report, the
            gate's "why" or the reason the round ended.
        payload: The HITL interrupt a ``paused`` event carries; ``None`` on the
            other events ``stream_fund`` yields.
    """

    kind: str
    run_id: uuid.UUID
    index: int
    agent: str
    step: str | None
    status: StepStatus
    text: str
    payload: dict[str, Any] | None = None

    @property
    def status_text(self) -> str:
        """Return the timeline status cell: glyph plus a word.

        A step shows its liveness (``● live``); a terminal event shows its ``kind``
        instead (``· paused`` / ``· incomplete``), because every terminal event
        carries ``PENDING`` and would otherwise read identically.
        """
        if self.kind == "step":
            return self.status.marker
        return f"{self.status.glyph} {self.kind}"


__all__ = ["StepStatus", "StreamEvent"]
