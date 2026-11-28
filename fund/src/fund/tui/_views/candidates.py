"""Plain-text cells and prompts for a rebalance gate's candidate portfolios.

The Decisions view lists every candidate a ``place_orders`` gate carries in one table
and the adviser chooses one. Every row gets the same columns, in the id order the set
arrives in; nothing here sorts, scores or marks a candidate, and the only selection is
the one the adviser makes. A status is a dot and a word, never a colour alone.

Candidates arrive as the JSON dump of :class:`fund.schemas.candidates.Candidate`, so
this module needs only the standard library and keeps the view import-light.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "CANDIDATES_STEP",
    "CANDIDATE_COLUMNS",
    "ORDER_GATE",
    "approve_prompt",
    "candidate_row",
    "decided_gate",
    "find_candidate",
    "gate_names",
    "is_selectable",
    "measures_line",
    "payload_set_hash",
    "pick_line",
    "refused_pick_line",
]

ORDER_GATE = "place_orders"
# The allocator step that records a run's candidate set; the audit repository keeps
# its own name for it, which these import-light modules cannot import.
CANDIDATES_STEP = "compute_candidates"

CANDIDATE_COLUMNS: tuple[str, ...] = (
    "id",
    "label",
    "status",
    "volatility",
    "CVaR 95%",
    "largest",
    "positions",
    "turnover",
    "check",
)

# The measure columns follow the id, label and status ones.
_FIRST_MEASURE_COLUMN = 3

_ABSENT = "—"
_FEASIBLE = "● feasible"
_INFEASIBLE = "○ infeasible"
_PASSED = "● passed"
_FAILED = "○ failed"


def gate_names(interrupt: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return the gated tool names a pending HITL payload asks the adviser about.

    The same reading as ``fund.agents.graph_candidates.pending_gate_names``, kept
    here so the view never imports the agent package (and the optimizer behind it)
    on the event loop.

    Args:
        interrupt: The payload a run is paused on (``action_requests``), or ``None``.

    Returns:
        The names in request order; empty when nothing is pending.
    """
    requests = (interrupt or {}).get("action_requests") or []
    return tuple(
        str(request.get("name")) for request in requests if isinstance(request, dict)
    )


def decided_gate(names: Sequence[str]) -> str | None:
    """Return the gate an approve decides: ``place_orders`` whenever it is pending.

    Args:
        names: The pending gated tool names.

    Returns:
        :data:`ORDER_GATE` when it is among ``names``, else the first name, or
        ``None`` when nothing is pending.
    """
    if ORDER_GATE in names:
        return ORDER_GATE
    return names[0] if names else None


def payload_set_hash(interrupt: Mapping[str, Any] | None) -> str | None:
    """Return the candidate-set digest a ``place_orders`` gate payload carries.

    Args:
        interrupt: The pending HITL payload, or ``None``.

    Returns:
        The ``candidate_set.hash`` of the first ``place_orders`` request, or ``None``
        when the gate carries no set.
    """
    for request in (interrupt or {}).get("action_requests") or []:
        if not isinstance(request, dict) or request.get("name") != ORDER_GATE:
            continue
        summary = (request.get("args") or {}).get("candidate_set")
        digest = summary.get("hash") if isinstance(summary, dict) else None
        return str(digest) if digest else None
    return None


def find_candidate(
    candidates: Sequence[Mapping[str, Any]], candidate_id: str | None
) -> Mapping[str, Any] | None:
    """Return the candidate with ``candidate_id``, or ``None`` for an unknown id."""
    if candidate_id is None:
        return None
    return next(
        (entry for entry in candidates if entry.get("candidate_id") == candidate_id),
        None,
    )


def is_selectable(candidate: Mapping[str, Any]) -> bool:
    """Return whether the adviser may choose ``candidate`` (it is feasible)."""
    return bool(candidate.get("feasible"))


def candidate_row(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """Return one table row for ``candidate``, one cell per :data:`CANDIDATE_COLUMNS`.

    A candidate without measures (its method produced no weights) shows a dash in
    each measure column, so every row keeps the same shape.

    Args:
        candidate: The candidate's JSON dump.

    Returns:
        The cells: id, label, status, volatility, CVaR 95%, largest position,
        number of positions, turnover and the constraint check.
    """
    metrics = candidate.get("metrics") or {}
    return (
        str(candidate.get("candidate_id")),
        str(candidate.get("label")),
        _status(candidate),
        _percent(metrics.get("annualized_volatility")),
        _percent(metrics.get("cvar_95")),
        _percent(metrics.get("largest_position")),
        _count(metrics.get("n_positions")),
        _percent(metrics.get("turnover")),
        _check(candidate.get("constraint_check")),
    )


def pick_line(
    candidate: Mapping[str, Any] | None,
    *,
    has_candidates: bool,
    blocked: str | None = None,
) -> str:
    """Return the line under the table that says what ``a`` would approve.

    Args:
        candidate: The candidate the adviser selected, or ``None``.
        has_candidates: Whether the gate shows any candidate at all.
        blocked: Why an approve is refused whatever is selected, or ``None``.

    Returns:
        The selection and the next step, in words.
    """
    if blocked is not None:
        return f"⚠ {blocked}"
    if not has_candidates:
        return "no candidate set: an approve is refused; reject the run"
    if candidate is None:
        return (
            "selected: none - select a feasible candidate row (enter), then press a; "
            "nothing is chosen for you"
        )
    return (
        f"selected: {candidate.get('candidate_id')} · {candidate.get('label')} - "
        "press a to approve it (a confirm names it)"
    )


def refused_pick_line(candidate_id: str, candidate: Mapping[str, Any] | None) -> str:
    """Return why ``candidate_id`` cannot be selected.

    Args:
        candidate_id: The id the adviser tried to select.
        candidate: That candidate, or ``None`` when the gate shows no such id.

    Returns:
        A line naming the id and the reason; nothing is selected.
    """
    if candidate is None:
        return f"cannot select {candidate_id}: this gate shows no such candidate"
    return (
        f"cannot select {candidate_id}: it is infeasible "
        f"({_breach_summary(candidate)}); nothing is selected"
    )


def approve_prompt(target: str, candidate: Mapping[str, Any]) -> str:
    """Return the confirm text for approving ``target`` with ``candidate``.

    Names the candidate's id and label, every weight the paper orders are placed
    from, and its measures, so the adviser confirms exactly what executes.

    Args:
        target: How the gate is named (its run id and portfolio).
        candidate: The candidate the adviser selected.

    Returns:
        The prompt shown above Confirm | Cancel.
    """
    weights = candidate.get("weights") or {}
    lines = [
        f"Approve {target} with candidate {candidate.get('candidate_id')}?",
        f"{candidate.get('candidate_id')} · {candidate.get('label')}",
        "",
        "paper orders are placed from these weights:",
        *(
            f"  {ticker}: {float(weight):.4f}"
            for ticker, weight in sorted(weights.items())
        ),
        "",
        measures_line(candidate),
    ]
    return "\n".join(lines)


def measures_line(candidate: Mapping[str, Any]) -> str:
    """Return the candidate's measures and check as one line, in column order."""
    named = zip(
        CANDIDATE_COLUMNS[_FIRST_MEASURE_COLUMN:],
        candidate_row(candidate)[_FIRST_MEASURE_COLUMN:],
        strict=True,
    )
    return " · ".join(f"{name} {cell}" for name, cell in named)


def _status(candidate: Mapping[str, Any]) -> str:
    if is_selectable(candidate):
        return _FEASIBLE
    return f"{_INFEASIBLE} · {_breach_summary(candidate)}"


def _breach_summary(candidate: Mapping[str, Any]) -> str:
    """The breach types that make a candidate infeasible, in recorded order."""
    kinds = [
        str(breach.get("type", "breach"))
        for breach in candidate.get("breaches") or ()
        if isinstance(breach, Mapping)
    ]
    return ", ".join(kinds) or "no reason recorded"


def _check(check: Any) -> str:
    if not isinstance(check, Mapping):
        return _ABSENT
    if check.get("passed"):
        return _PASSED
    count = len(check.get("violations") or ())
    return f"{_FAILED} · {count} violation{'s' if count != 1 else ''}"


def _percent(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _ABSENT
    return f"{value:.2%}"


def _count(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return _ABSENT
    return str(value)
