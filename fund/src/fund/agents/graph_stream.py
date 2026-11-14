"""How a round's end and its steps are shaped into stream events."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from fund.tui.stream import StreamEvent


@dataclass(frozen=True)
class _RoundEnd:
    """How :func:`_settle_round` closed a round the PM agent ran to its end.

    Attributes:
        status: The run's new status: ``"incomplete"`` (no gate was reached),
            ``"rejected"`` (a freeze landed while the round ran) or ``"paused"``
            (the ``place_orders`` gate is live).
        reason: Why an ``incomplete`` round never reached the gate (one of
            :func:`_ungated_end_reason`'s); ``None`` for the other two.
    """

    status: Literal["incomplete", "rejected", "paused"]
    reason: str | None = None


def _tool_calls(message: Any) -> list[dict[str, Any]]:
    """The structured tool calls on a streamed model message (empty when none)."""
    calls = getattr(message, "tool_calls", None) or []
    return [call for call in calls if isinstance(call, dict)]


def _message_text(message: Any) -> str:
    """A streamed message's content as text (``""`` when absent or non-string)."""
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


_UNGATED_END_TEXT = {
    "pm_round_cap": "delegation round cap reached before a ticket",
    "output_truncated": "the model's output limit cut off the PM's order call",
    "invalid_tool_call": "the PM's order call was malformed",
}


def _gate_why(interrupt_value: dict[str, Any] | None) -> str:
    """The HITL gate's 'why' from its first action request (safe default otherwise)."""
    requests = (interrupt_value or {}).get("action_requests") or []
    request = requests[0] if requests else {}
    return request.get("description") or "review before it executes"


def _terminal_event(
    run_id: uuid.UUID,
    index: int,
    *,
    kind: str,
    text: str,
    payload: dict[str, Any] | None = None,
) -> StreamEvent:
    """The single event a streamed round ends on, pending at ``place_orders``."""
    from fund.tui.stream import StepStatus, StreamEvent

    return StreamEvent(
        kind=kind,
        run_id=run_id,
        index=index,
        agent="orchestrator",
        step="place_orders",
        status=StepStatus.PENDING,
        text=text,
        payload=payload,
    )


def _round_end_event(
    end: _RoundEnd,
    run_id: uuid.UUID,
    index: int,
    interrupt_value: dict[str, Any] | None,
) -> StreamEvent:
    """How :func:`stream_fund` tells the adviser the way :func:`_settle_round` ended."""
    if end.status == "paused":
        return _terminal_event(
            run_id,
            index,
            kind="paused",
            text=_gate_why(interrupt_value),
            payload=interrupt_value,
        )
    if end.status == "rejected":
        text = "halted: autonomy was frozen while the round was running"
    else:
        text = _UNGATED_END_TEXT.get(end.reason or "", "pipeline ended before the gate")
    return _terminal_event(run_id, index, kind="incomplete", text=text)
