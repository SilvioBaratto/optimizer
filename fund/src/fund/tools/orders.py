"""``place_orders``: idempotent paper execution, the last allocator node.

Turns a target-weight vector into a **simulated** order ticket (SPEC D5): each
line fills at the **next close strictly after** the decision bar ``asof`` (D30 —
no look-ahead), adjusted by a simple slippage + commission model, and the ticket
is persisted to ``paper_orders``. Costs are charged on the **traded** notional —
the move from the previous ticket's target weights to this one (exits included) —
not on the whole target book.

The write must be **idempotent**: ``place_orders`` sits behind ``interrupt_on``
(HITL), and on ``Command(resume=…)`` the interrupting node re-runs from the top
(SPEC D3), so a second call with the same ``(portfolio_id, asof, weights)`` must
not double-place. The tool looks the ticket up by that key first and returns the
stored one (``idempotent: true``) instead of writing again; the DB-level
``UNIQUE(portfolio_id, asof, weights_hash)`` is the backstop.

Contract (via ``tool_envelope``):

* empty ``weights`` ⇒ ``{ok: false, error}``;
* a ticker with no instrument row, or no close strictly after ``asof`` (no fill
  bar), ⇒ ``{ok: false, error}`` — a paper ticket must fill every line;
* every other failure is caught by the envelope and returned as ``{ok: false}``.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from typing import TYPE_CHECKING, Any

from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository

from fund.audit.orders_repository import OrderRepository, weights_hash
from fund.tools._base import ToolResult, coerce_date, err, ok, tool_envelope

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.orm import Session

# Simple paper-execution cost model (D30): flat notional the weights scale, and
# constant bps for slippage/commission. VWAP/Almgren-Chriss come later (needs
# intraday). These params are NOT part of the idempotency key (which is
# weights-only): overrides are honoured only for a FRESH key. A what-if with
# different cost params for an already-placed key is rejected (vary the key —
# e.g. a distinct portfolio_id) rather than silently re-priced from the stored
# ticket.
_DEFAULT_NOTIONAL = 100_000.0
_DEFAULT_SLIPPAGE_BPS = 5.0
_DEFAULT_COMMISSION_BPS = 1.0


def _coerce_uuid(portfolio_id: uuid.UUID | str) -> uuid.UUID:
    """Normalise ``portfolio_id`` to a ``UUID`` (accepts a canonical string)."""
    if isinstance(portfolio_id, uuid.UUID):
        return portfolio_id
    return uuid.UUID(portfolio_id)


def _next_close(
    repo: YFinanceRepository, ticker: str, asof: dt.date
) -> tuple[dt.date, float]:
    """Return ``(fill_date, close)`` for the first bar strictly after ``asof``.

    Raises ``LookupError`` if the ticker has no instrument row or no priced bar
    after ``asof`` — the envelope turns that into ``{ok: false, error}``.
    """
    instrument = repo.get_instrument_by_yfinance_ticker(ticker)
    if instrument is None:
        raise LookupError(f"unknown ticker {ticker!r}")
    rows = repo.get_price_history(
        instrument.id, start_date=asof + dt.timedelta(days=1), ascending=True
    )
    priced = [(row.date, row.close) for row in rows if row.close is not None]
    if not priced:
        raise LookupError(f"no fill bar after {asof.isoformat()} for {ticker!r}")
    fill_date, close = min(priced, key=lambda pair: pair[0])
    return fill_date, float(close)


def fill_shortfalls(
    session: Session, asof: dt.date, tickers: Iterable[str]
) -> list[str]:
    """Why ``place_orders`` could not fill each of ``tickers`` at ``asof``.

    A read-only dry run of the fill lookup, so a caller can tell before the gated
    ``place_orders`` runs (and consumes its interrupt) that the ticket would fail: an
    unknown ticker, or no close strictly after ``asof`` yet.

    Returns:
        One message per ticker that cannot fill; ``[]`` when every line could.
    """
    repo = YFinanceRepository(session)
    shortfalls: list[str] = []
    for ticker in tickers:
        try:
            _next_close(repo, ticker, asof)
        except LookupError as exc:
            shortfalls.append(str(exc))
    return shortfalls


def _cost_params_match(
    order: Any,
    *,
    notional: float,
    slippage_bps: float,
    commission_bps: float,
) -> bool:
    """Whether a stored ticket was priced with the requested cost params.

    The idempotency key is weights-only (SPEC D3 / ``uq_paper_order_key``), so
    the per-call cost params are NOT in the key. This guards against silently
    returning a ticket priced with *different* params than requested: it
    reconstructs the stored slippage/commission from the persisted lines and
    compares them (with a float tolerance) against the request. Commission is a
    fraction of the line's *traded* notional; tickets stored before turnover
    pricing carry no ``traded_notional`` and were charged on the full line notional.
    """
    if not math.isclose(order.notional, notional, rel_tol=1e-9, abs_tol=1e-9):
        return False
    for line in order.lines:
        if not math.isclose(
            line.get("slippage_bps", 0.0), slippage_bps, rel_tol=1e-9, abs_tol=1e-9
        ):
            return False
        ln = line.get("traded_notional", line.get("notional", 0.0))
        if ln:
            stored_comm_bps = line["commission"] / abs(ln) * 1e4
            if not math.isclose(
                stored_comm_bps, commission_bps, rel_tol=1e-6, abs_tol=1e-9
            ):
                return False
    return True


def _ticket_from_row(order: Any, *, idempotent: bool) -> dict[str, Any]:
    """Render a persisted ``PaperOrder`` row as the JSON-serialisable ticket."""
    return {
        "order_id": str(order.id),
        "portfolio_id": str(order.portfolio_id),
        "asof": order.asof.isoformat(),
        "fill_date": order.fill_date.isoformat(),
        "weights_hash": order.weights_hash,
        "weights": order.weights,
        "notional": order.notional,
        "lines": order.lines,
        "total_commission": order.total_commission,
        "total_slippage_cost": order.total_slippage_cost,
        "status": order.status,
        "idempotent": idempotent,
    }


@tool_envelope
def place_orders(
    session: Session,
    asof: dt.date | str,
    weights: dict[str, float],
    portfolio_id: uuid.UUID | str,
    *,
    notional: float = _DEFAULT_NOTIONAL,
    slippage_bps: float = _DEFAULT_SLIPPAGE_BPS,
    commission_bps: float = _DEFAULT_COMMISSION_BPS,
) -> ToolResult:
    """Place (or return) the idempotent paper ticket for ``weights`` as of ``asof``.

    Args:
        session: A sync ``portopt_db`` session (D1); the tool does not own it.
        asof: The decision bar; fills land at the next close strictly after it
            (no look-ahead). Accepts a ``date`` or an ISO ``YYYY-MM-DD`` string.
        weights: Target ``{ticker: weight}`` vector (from the optimizer).
        portfolio_id: Owning portfolio; part of the idempotency key.
        notional: Gross book the weights scale (paper cost model). Costs apply to
            the part of it that trades against the portfolio's previous ticket.
            Not part of the idempotency key — overrides apply only to a fresh key.
        slippage_bps: Per-fill slippage in basis points of traded notional (buys
            pay up, sells receive less; unchanged names fill at the close). Not
            part of the idempotency key — overrides apply only to a fresh key.
        commission_bps: Per-fill commission in basis points of traded notional.
            Not part of the idempotency key — overrides apply only to a fresh key.

    Returns:
        ``ok`` with the order ticket (``order_id``, ``fill_date``, per-``lines``
        fills, cost totals, ``idempotent`` flag). ``err`` on empty ``weights``,
        an unknown ticker, or a ticker with no fill bar after ``asof``.
    """
    if not weights:
        return err("no weights to place")

    end_date = coerce_date(asof)
    pid = _coerce_uuid(portfolio_id)
    whash = weights_hash(weights)

    repo = OrderRepository(session)
    existing = repo.get_by_key(portfolio_id=pid, asof=end_date, weights_hash=whash)
    if existing is not None:
        if _cost_params_match(
            existing,
            notional=notional,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
        ):
            # HITL re-run (D3): the ticket already exists priced with the same
            # params — return it, don't re-place.
            return ok(_ticket_from_row(existing, idempotent=True))
        # Same weights-only key but different cost params: the stored ticket was
        # priced differently, so returning it would misreport the request.
        return err(
            "a paper ticket already exists for this (portfolio, asof, weights) "
            "with different cost params; re-run with the placed params, or use a "
            "distinct key for a what-if with different "
            "notional/slippage/commission"
        )

    prior = repo.latest_on_or_before(pid, end_date)
    prior_weights = (
        {ticker: float(w) for ticker, w in prior.weights.items()} if prior else {}
    )

    prices = YFinanceRepository(session)
    lines: list[dict[str, Any]] = []
    total_commission = 0.0
    total_slippage_cost = 0.0
    fill_dates: list[dt.date] = []

    for ticker, weight in weights.items():
        fill_date, fill_price = _next_close(prices, ticker, end_date)
        fill_dates.append(fill_date)

        line_notional = weight * notional
        # Only the move away from what is already held trades; holding a name at
        # an unchanged weight costs nothing.
        traded_notional = (weight - prior_weights.get(ticker, 0.0)) * notional
        commission = abs(traded_notional) * commission_bps / 1e4
        slippage_cost = abs(traded_notional) * slippage_bps / 1e4
        # Direction comes from the trade, not the target's sign: a trim of a long is
        # a sell, and an unchanged name fills at the raw close.
        direction = (traded_notional > 0) - (traded_notional < 0)
        effective_price = fill_price * (1.0 + direction * slippage_bps / 1e4)
        # The carried part of the book is valued at the close; only the traded part
        # pays (buys) or gives up (sells) the slippage.
        shares = (
            (line_notional - traded_notional) / fill_price
            + traded_notional / effective_price
            if fill_price and effective_price
            else 0.0
        )

        total_commission += commission
        total_slippage_cost += slippage_cost
        lines.append(
            {
                "ticker": ticker,
                "weight": float(weight),
                "fill_date": fill_date.isoformat(),
                "fill_price": fill_price,
                "effective_price": effective_price,
                "slippage_bps": slippage_bps,
                "notional": line_notional,
                "traded_notional": traded_notional,
                "shares": shares,
                "commission": commission,
                "slippage_cost": slippage_cost,
            }
        )

    # A name dropped from the target is sold out, but gets no ticket line: lines
    # become the held book (positions, NAV), so a zero-weight line would resurrect it.
    for ticker, prior_weight in prior_weights.items():
        if ticker not in weights:
            exit_notional = abs(prior_weight * notional)
            total_commission += exit_notional * commission_bps / 1e4
            total_slippage_cost += exit_notional * slippage_bps / 1e4

    order = repo.create(
        portfolio_id=pid,
        asof=end_date,
        weights_hash=whash,
        weights={k: float(v) for k, v in weights.items()},
        fill_date=min(fill_dates),
        lines=lines,
        notional=notional,
        total_commission=total_commission,
        total_slippage_cost=total_slippage_cost,
    )
    return ok(_ticket_from_row(order, idempotent=False))


__all__ = ["fill_shortfalls", "place_orders"]
