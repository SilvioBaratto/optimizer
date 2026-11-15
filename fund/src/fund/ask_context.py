"""The compact fund snapshot the Ask agent's ``fund_snapshot`` tool returns.

:func:`build_context` reads a few ``fund.observe`` views, through the caller's
session and checkpointer, and formats them as a short plain-text block. The module
sits outside ``fund.tui`` because the Ask agent (``fund.agents.ask_agent``) calls it
and the agent layer must not import the cockpit; :mod:`fund.tui.ask` re-exports
:func:`build_context`.

At module scope it imports only the standard library and the agent-stack-free
``fund.observe`` read model (whose optimizer/skfolio tracker is itself lazy), so
importing it pulls in neither the agent stack nor ``optimizer``/``skfolio``.
"""

from __future__ import annotations

import uuid
from typing import Any

from fund import observe


def build_context(
    session: Any, saver: Any, focus: uuid.UUID | None, question: str
) -> str:
    """Assemble a compact, grounded context from the read model for ``question``.

    Reads the fund roster size and the pending gates (the first five listed) and,
    when a portfolio is focused, its drift, its held and target holding counts, its
    NAV-series returns and its five most recent decisions, formatted as a short
    plain-text block. It only reads, through the caller's session and saver, and
    builds no model. ``question`` is accepted for future relevance selection; today
    the snapshot is fixed.

    Args:
        session: An injected read-only sync session; not owned here.
        saver: The bootstrapped LangGraph checkpointer (opaque), for the gate reads.
        focus: The focused portfolio, or ``None`` for a fund-wide question.
        question: The adviser's natural-language question (reserved).

    Returns:
        A compact grounded-context string — never empty (at minimum the fund summary).
    """
    portfolios = observe.list_portfolios(session)
    gates = observe.pending_hitl(session, saver, None)
    lines = [
        "FUND STATE",
        f"portfolios: {len(portfolios)} · pending gates: {len(gates)}",
    ]
    for gate in gates[:5]:
        pf = _short(gate.portfolio_id) if gate.portfolio_id else "fund"
        lines.append(f"  gate #{_short(gate.run_id)} · pf {pf} · asof {gate.asof}")
    if focus is not None:
        lines.extend(_focus_lines(session, focus))
    return "\n".join(lines)


def _focus_lines(session: Any, focus: uuid.UUID) -> list[str]:
    """The focused-portfolio grounding block: drift, NAV returns, recent decisions."""
    state = observe.portfolio_state(session, focus)
    summary = observe.performance_summary(session, focus)
    lines = [
        f"FOCUS portfolio {_short(focus)}",
        f"  drift L1 {state.drift_l1:.4f} · "
        f"{len(state.current)} held → {len(state.target)} target",
        f"  returns: max {_fmt_pct(summary.returns.get('max'))} · "
        f"1D {_fmt_pct(summary.returns.get('1D'))}",
    ]
    timeline = observe.audit_timeline(session, portfolio_id=focus, limit=5)
    if timeline:
        lines.append("  recent decisions:")
        lines.extend(
            f"    {entry.when:%m-%d} {entry.actor} {entry.action}" for entry in timeline
        )
    return lines


def _fmt_pct(value: float | None) -> str:
    """A signed-percent cell, or ``n/a`` when the metric is unavailable."""
    return f"{value:+.2%}" if value is not None else "n/a"


def _short(value: Any) -> str:
    """First hex group of a UUID (or its string) — matches the view's compact id."""
    return str(value).split("-", 1)[0]


__all__ = ["build_context"]
