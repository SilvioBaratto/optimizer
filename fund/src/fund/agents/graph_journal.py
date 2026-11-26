"""What an approved round records: the logged weights and the journal row."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import datetime as dt


def _optimizer_weights(session: Any, run_id: uuid.UUID) -> dict[str, float]:
    """The latest allocator ``optimize_portfolio`` weights logged for this run.

    The allocator's bound tool logs the load-bearing decision (constraint set +
    effective optimizer config + the optimizer's weights) inside its closure during
    the initial invoke. Reading them back from the audit trail — never from what
    the PM/LLM passed — keeps ``optimize_portfolio`` the single source of weights.
    Delegates to :meth:`AgentRunRepository.latest_optimizer_weights` so the paper
    ticket and the finalised ``agent_runs.weights`` derive from one source and
    cannot diverge. Returns ``{}`` when the allocator never produced a proposal.
    """
    from fund.audit import AgentRunRepository

    return AgentRunRepository(session).latest_optimizer_weights(run_id)


def _trades_from_lines(
    lines: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Index committed paper-order fill lines by ticker for the overlay.

    ``lines`` come straight from ``paper_orders.lines`` (a JSON column), so the
    values are already JSON-safe. Returns ``None`` when there are no fills.
    """
    if not lines:
        return None
    trades: dict[str, Any] = {}
    for line in lines:
        ticker = line.get("ticker")
        if ticker is None:
            continue
        trades[str(ticker)] = {k: v for k, v in line.items() if k != "ticker"}
    return trades or None


def _drift_vs_prior(
    prior: dict[str, float], target: dict[str, float]
) -> dict[str, Any]:
    """Per-ticker weight delta from prior holdings to the new target weights.

    Covers the union of held and targeted tickers (missing side counts as 0),
    so both exits and new entries show up. Bands-crossed classification is out
    of scope here — ``_finalize_hitl`` has no constraint set — and is deferred.
    """
    out: dict[str, Any] = {}
    for ticker in sorted(set(prior) | set(target)):
        before = float(prior.get(ticker, 0.0))
        after = float(target.get(ticker, 0.0))
        out[ticker] = {
            "prior": before,
            "target": after,
            "delta": round(after - before, 6),
        }
    return out


def _portfolio_narrative(
    asof: dt.date,
    allocation: dict[str, float],
    trades: dict[str, Any] | None,
    drift: dict[str, Any],
) -> str:
    """Deterministic one-line summary of an approved rebalance.

    A pure function of the overlay sections — the same inputs reproduce the
    string byte-for-byte, keeping the overlay reproducible for audit.
    """
    n_fills = len(trades) if trades else 0
    moved = sorted(t for t, d in drift.items() if abs(d.get("delta", 0.0)) > 1e-9)
    moves = ", ".join(moved) if moved else "none"
    return (
        f"Rebalance approved for {asof.isoformat()}: "
        f"{len(allocation)} target position(s), {n_fills} fill(s); "
        f"weight moves: {moves}."
    )


def _write_portfolio_journal(
    session: Any,
    *,
    run_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    asof: dt.date,
    weights: dict[str, float],
    order_lines: list[dict[str, Any]] | None,
    prior_weights: dict[str, float],
) -> None:
    """Upsert the per-portfolio overlay row for an approved rebalance.

    Assembles ``allocation`` (target weights), ``trades`` (committed fills), and
    ``drift`` (vs ``prior_weights``) plus a deterministic narrative, then upserts
    keyed on ``(portfolio_id, asof)``. Repos are imported lazily so a bare
    ``import fund.agents.graph`` stays light (``fund.audit`` loads its langgraph
    bootstrap on demand, so this is defence in depth) and no ``commit`` is issued
    (the caller owns the transaction).
    """
    from portopt_db.repositories.orders.portfolio_journal_repository import (
        PortfolioJournalRepository,
    )

    from fund.audit import OrderRepository

    lines = order_lines
    if lines is None:
        order = OrderRepository(session).latest_for_portfolio(portfolio_id)
        if order is not None and order.asof == asof:
            lines = order.lines

    allocation = {ticker: float(weight) for ticker, weight in weights.items()}
    trades = _trades_from_lines(lines)
    drift = _drift_vs_prior(prior_weights, allocation)
    narrative = _portfolio_narrative(asof, allocation, trades, drift)

    PortfolioJournalRepository(session).upsert(
        portfolio_id,
        asof,
        run_id=run_id,
        trades=trades,
        allocation=allocation,
        drift=drift,
        narrative=narrative,
    )


def _holdings_from_ticket(ticket: Any) -> list[dict[str, Any]]:
    """Snapshot rows for the holdings a paper ticket produced.

    Carries each line's base-currency notional but not its ``shares``: that figure is
    the base notional divided by the listing price, so it is currency-mixed for a
    non-base listing (the performance marks re-derive FX-correct shares for the same
    reason) and would be a wrong holding size.
    """
    return [
        {
            "ticker": line["ticker"],
            "weight": line["weight"],
            "notional": line.get("notional"),
        }
        for line in ticket.lines
    ]
