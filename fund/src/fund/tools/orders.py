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
* a held ticker (weight above ``fund.pricing.DUST_WEIGHT``) with no instrument row,
  or no close strictly after ``asof`` (no fill bar), ⇒ ``{ok: false, error}`` — a
  paper ticket must fill every line it carries. A zero or dust weight is no order: it
  gets no line and needs no fill bar or FX rate, though it stays in the stored
  ``weights``;
* weights that are all zero or dust ⇒ ``{ok: false, error}`` — nothing to place;
* a listing quoted in a foreign currency with no FX rate on or just before its fill
  bar ⇒ ``{ok: false, error}`` — ``shares`` must be a base-currency notional over a
  base-currency price, never a unit-mixed ratio;
* every other failure is caught by the envelope and returned as ``{ok: false}``.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fund.audit.orders_repository import OrderRepository, weights_hash
from fund.pricing import (
    BPS_PER_UNIT,
    CostModel,
    FillQuote,
    held_weights,
    price_ticket,
)
from fund.readers import fx_rate_reader, price_reader
from fund.tools._base import ToolResult, coerce_date, err, ok, tool_envelope

if TYPE_CHECKING:
    from collections.abc import Iterable

    from portopt_db.ports import FxRateReader, PriceReader
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

# How far back from a fill bar an FX rate may be carried when the feed has no row
# for that exact day (weekends, holidays), mirroring the NAV engine's forward-fill.
_FX_LOOKBACK_DAYS = 7


@dataclass(frozen=True)
class _Fill:
    """The fill bar of one ticker: when, at what quoted close, in which unit."""

    date: dt.date
    close: float
    unit: str | None


def _coerce_uuid(portfolio_id: uuid.UUID | str) -> uuid.UUID:
    """Normalise ``portfolio_id`` to a ``UUID`` (accepts a canonical string)."""
    if isinstance(portfolio_id, uuid.UUID):
        return portfolio_id
    return uuid.UUID(portfolio_id)


def _next_close(prices: PriceReader, ticker: str, asof: dt.date) -> _Fill:
    """Return the first priced bar strictly after ``asof`` and its quote unit.

    The unit is the bar's raw ``price_unit`` (``GBp`` stays pence), falling back to
    the instrument's trading currency.

    Raises ``LookupError`` if the ticker has no instrument row or no priced bar
    after ``asof`` — the envelope turns that into ``{ok: false, error}``.
    """
    instrument = prices.instrument_for(ticker)
    if instrument is None:
        raise LookupError(f"unknown ticker {ticker!r}")
    rows = prices.price_bars_for(
        instrument.id, start_date=asof + dt.timedelta(days=1), ascending=True
    )
    priced = [(row, float(row.close)) for row in rows if row.close is not None]
    if not priced:
        raise LookupError(f"no fill bar after {asof.isoformat()} for {ticker!r}")
    first, close = min(priced, key=lambda pair: pair[0].date)
    return _Fill(
        date=first.date,
        close=close,
        unit=first.price_unit or instrument.currency_code,
    )


def _base_close(
    fx_rates: FxRateReader, fill: _Fill, ticker: str, base_currency: str
) -> float:
    """The fill close as a price per share in ``base_currency``.

    Rescales a minor-unit quote (pence, cents, agorot) to its major unit, then
    multiplies by the base-per-foreign rate on the fill bar (carried forward at most
    ``_FX_LOOKBACK_DAYS``). A listing already in the base currency needs no rate; one
    with no unit is taken to be in the base currency, as the NAV engine does.

    Raises:
        LookupError: The listing is foreign and the feed has no rate for it.
    """
    from optimizer.fx import normalize_currency_code

    base, _ = normalize_currency_code(base_currency)
    major, scale = normalize_currency_code(fill.unit or base)
    price = fill.close / scale
    if major == base:
        return price
    rates = fx_rates.rates_for(
        base,
        [major],
        start=fill.date - dt.timedelta(days=_FX_LOOKBACK_DAYS),
        end=fill.date,
    )
    if not rates:
        raise LookupError(
            f"no FX rate {major}->{base} on or before {fill.date.isoformat()} "
            f"for {ticker!r}"
        )
    return price * rates[-1].rate


def fill_shortfalls(
    session: Session,
    asof: dt.date,
    tickers: Iterable[str],
    *,
    base_currency: str | None = None,
) -> list[str]:
    """Why ``place_orders`` could not fill each of ``tickers`` at ``asof``.

    A read-only dry run of the fill lookup, so a caller can tell before the gated
    ``place_orders`` runs (and consumes its interrupt) that the ticket would fail: an
    unknown ticker, no close strictly after ``asof`` yet, or (given ``base_currency``)
    a foreign listing with no FX rate for its fill bar. Given a weight mapping, names
    at zero or dust weight are skipped, as ``place_orders`` places no order for them.

    Returns:
        One message per ticker that cannot fill; ``[]`` when every line could.
    """
    prices, fx_rates = price_reader(session), fx_rate_reader(session)
    shortfalls: list[str] = []
    names = held_weights(tickers) if isinstance(tickers, Mapping) else tickers
    for ticker in names:
        try:
            fill = _next_close(prices, ticker, asof)
            if base_currency is not None:
                _base_close(fx_rates, fill, ticker, base_currency)
        except LookupError as exc:
            shortfalls.append(str(exc))
    return shortfalls


def _fill_quotes(
    session: Session,
    tickers: Iterable[str],
    asof: dt.date,
    base_currency: str | None,
) -> dict[str, FillQuote]:
    """Look up where each of ``tickers`` fills: its next close after ``asof``.

    Each quote carries the close as quoted and as a price per share in
    ``base_currency``; without a base currency the quoted close is used as given.

    Raises:
        LookupError: A ticker has no instrument row or no close strictly after
            ``asof``, or (given ``base_currency``) a foreign listing has no FX rate
            for its fill bar.
    """
    prices, fx_rates = price_reader(session), fx_rate_reader(session)
    quotes: dict[str, FillQuote] = {}
    for ticker in tickers:
        fill = _next_close(prices, ticker, asof)
        # Shares divide a base-currency notional, so the price must be one too. Without
        # a base currency (a what-if) the quote is used as given.
        base_price = (
            _base_close(fx_rates, fill, ticker, base_currency)
            if base_currency is not None
            else fill.close
        )
        quotes[ticker] = FillQuote(
            date=fill.date, price=fill.close, base_price=base_price, unit=fill.unit
        )
    return quotes


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
            stored_comm_bps = line["commission"] / abs(ln) * BPS_PER_UNIT
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


def _make_newest(repo: OrderRepository, portfolio_id: uuid.UUID, order: Any) -> None:
    """Re-stamp ``order`` as the portfolio's newest ticket when a later one exists.

    The current book is the newest ticket, ranked by ``created_at`` after the decision
    and fill bars. Approving weights that equal an older ticket of the same bar hands
    that ticket back, and left alone the ticket placed in between would keep ranking
    as the book (NAV, drift) while the positions snapshot points at the old one. The
    stamp is written in SQL on the database clock, like every other ticket's, and the
    row is left untouched when it already is the newest.
    """
    newest = repo.latest_for_portfolio(portfolio_id)
    if newest is None or newest.id == order.id:
        return
    repo.restamp_created_at(order.id)


@tool_envelope
def place_orders(
    session: Session,
    asof: dt.date | str,
    weights: dict[str, float],
    portfolio_id: uuid.UUID | str,
    *,
    base_currency: str | None = None,
    notional: float = _DEFAULT_NOTIONAL,
    slippage_bps: float = _DEFAULT_SLIPPAGE_BPS,
    commission_bps: float = _DEFAULT_COMMISSION_BPS,
) -> ToolResult:
    """Place (or return) the idempotent paper ticket for ``weights`` as of ``asof``.

    A stored ticket returned for the same key becomes the portfolio's newest again, so
    re-approving weights that equal an older ticket of the bar puts the book back on
    that ticket instead of the one placed after it.

    Args:
        session: A sync ``portopt_db`` session (D1); the tool does not own it.
        asof: The decision bar; fills land at the next close strictly after it
            (no look-ahead). Accepts a ``date`` or an ISO ``YYYY-MM-DD`` string.
        weights: Target ``{ticker: weight}`` vector (from the optimizer).
        portfolio_id: Owning portfolio; part of the idempotency key.
        base_currency: The mandate's base currency, in which each line's ``shares``
            are priced. Without it prices are used as quoted (only meaningful for a
            what-if on a single-currency universe).
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
        fills for the held names, cost totals, ``idempotent`` flag). ``err`` on empty
        or all-zero ``weights``, an unknown held ticker, a held ticker with no fill bar
        after ``asof``, or a held foreign listing with no FX rate for its fill bar.
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
            _make_newest(repo, pid, existing)
            return ok(_ticket_from_row(existing, idempotent=True))
        # Same weights-only key but different cost params: the stored ticket was
        # priced differently, so returning it would misreport the request.
        return err(
            "a paper ticket already exists for this (portfolio, asof, weights) "
            "with different cost params; re-run with the placed params, or use a "
            "distinct key for a what-if with different "
            "notional/slippage/commission"
        )

    held = held_weights(weights)
    if not held:
        return err("no non-zero weights to place")

    prior = repo.latest_on_or_before(pid, end_date)
    prior_weights = (
        {ticker: float(w) for ticker, w in prior.weights.items()} if prior else {}
    )
    quotes = _fill_quotes(session, held, end_date, base_currency)
    priced = price_ticket(
        held,
        prior_weights,
        quotes,
        notional,
        CostModel(slippage_bps=slippage_bps, commission_bps=commission_bps),
    )
    order = repo.create(
        portfolio_id=pid,
        asof=end_date,
        weights_hash=whash,
        weights={k: float(v) for k, v in weights.items()},
        fill_date=min(quote.date for quote in quotes.values()),
        lines=[dict(line) for line in priced.lines],
        notional=notional,
        total_commission=priced.total_commission,
        total_slippage_cost=priced.total_slippage_cost,
    )
    return ok(_ticket_from_row(order, idempotent=False))


__all__ = ["fill_shortfalls", "place_orders"]
