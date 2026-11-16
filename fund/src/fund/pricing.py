"""Pricing of a paper order ticket, with no session and no I/O.

``price_ticket`` turns target weights, the previous ticket's weights and the fill
quote of each held name into ticket lines and cost totals (SPEC D30): slippage and
commission are constant bps of the **traded** notional, the move from the previous
ticket's weight, exits included. Keeping the arithmetic free of the database lets
the cost model be tested on hand-built quotes.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypedDict

# Basis points in one unit: a cost in bps divided by this is a fraction of notional.
BPS_PER_UNIT: Final = 10_000.0

# The optimiser returns exact zeros and solver residue for names it does not hold; a
# weight this small is no order and must not need a fill bar of its own.
DUST_WEIGHT: Final = 1e-6


@dataclass(frozen=True)
class CostModel:
    """The constant execution costs a paper fill is charged.

    Attributes:
        slippage_bps: Slippage in basis points of traded notional: buys pay up,
            sells receive less, a name whose weight is unchanged fills at the close.
        commission_bps: Commission in basis points of traded notional.
    """

    slippage_bps: float
    commission_bps: float


@dataclass(frozen=True)
class FillQuote:
    """Where one held name fills: the bar, its quoted close and that close in base.

    Attributes:
        date: The fill bar.
        price: The close as quoted, in ``unit``.
        base_price: The close as a price per share in the base currency, or the
            quoted close when the ticket has no base currency.
        unit: The quote unit (``GBp`` stays pence), ``None`` when unknown.
    """

    date: dt.date
    price: float
    base_price: float
    unit: str | None


class TicketLine(TypedDict):
    """One line of a paper ticket, as stored in ``paper_orders.lines``."""

    ticker: str
    weight: float
    fill_date: str
    currency: str | None
    fill_price: float
    fill_price_base: float
    effective_price: float
    slippage_bps: float
    notional: float
    traded_notional: float
    shares: float
    commission: float
    slippage_cost: float


@dataclass(frozen=True)
class PricedTicket:
    """The lines and cost totals of a priced ticket.

    Attributes:
        lines: One line per held name, in the order of the held weights.
        total_commission: Commission of the lines plus that of the exits.
        total_slippage_cost: Slippage of the lines plus that of the exits.
    """

    lines: tuple[TicketLine, ...]
    total_commission: float
    total_slippage_cost: float


def held_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Return the weights that carry an order: those of magnitude above ``DUST_WEIGHT``.

    Written as ``not <=`` rather than ``>`` so a NaN weight stays held and fails
    loudly downstream instead of being dropped as dust.
    """
    return {t: w for t, w in weights.items() if not abs(w) <= DUST_WEIGHT}


def price_ticket(
    held: Mapping[str, float],
    prior: Mapping[str, float],
    quotes: Mapping[str, FillQuote],
    notional: float,
    costs: CostModel,
) -> PricedTicket:
    """Price the ticket that moves the book from ``prior`` to ``held``.

    Args:
        held: Target weights that carry an order (see :func:`held_weights`).
        prior: The previous ticket's weights; empty for a first ticket.
        quotes: The fill quote of each held name.
        notional: Gross book the weights scale.
        costs: Slippage and commission per unit of traded notional, in bps.

    Returns:
        A line per held name, and totals that add the cost of selling out each
        prior name missing from ``held``. Such a name gets no line: lines become the
        held book (positions, NAV), so a zero-weight line would resurrect it.

    Raises:
        KeyError: A held name has no quote.
    """
    lines: list[TicketLine] = []
    total_commission = 0.0
    total_slippage_cost = 0.0
    for ticker, weight in held.items():
        line = _price_line(
            ticker, weight, prior.get(ticker, 0.0), quotes[ticker], notional, costs
        )
        total_commission += line["commission"]
        total_slippage_cost += line["slippage_cost"]
        lines.append(line)

    for ticker, prior_weight in prior.items():
        if ticker not in held:
            exit_notional = abs(prior_weight * notional)
            total_commission += exit_notional * costs.commission_bps / BPS_PER_UNIT
            total_slippage_cost += exit_notional * costs.slippage_bps / BPS_PER_UNIT
    return PricedTicket(
        lines=tuple(lines),
        total_commission=total_commission,
        total_slippage_cost=total_slippage_cost,
    )


def _price_line(
    ticker: str,
    weight: float,
    prior_weight: float,
    quote: FillQuote,
    notional: float,
    costs: CostModel,
) -> TicketLine:
    """Price the line of one held name."""
    line_notional = weight * notional
    # Only the move away from what is already held trades; holding a name at
    # an unchanged weight costs nothing.
    traded_notional = (weight - prior_weight) * notional
    commission = abs(traded_notional) * costs.commission_bps / BPS_PER_UNIT
    slippage_cost = abs(traded_notional) * costs.slippage_bps / BPS_PER_UNIT
    # Direction comes from the trade, not the target's sign: a trim of a long is
    # a sell, and an unchanged name fills at the raw close.
    direction = (traded_notional > 0) - (traded_notional < 0)
    slippage_factor = 1.0 + direction * costs.slippage_bps / BPS_PER_UNIT
    effective_price = quote.price * slippage_factor
    return {
        "ticker": ticker,
        "weight": float(weight),
        "fill_date": quote.date.isoformat(),
        "currency": quote.unit,
        "fill_price": quote.price,
        "fill_price_base": quote.base_price,
        "effective_price": effective_price,
        "slippage_bps": costs.slippage_bps,
        "notional": line_notional,
        "traded_notional": traded_notional,
        "shares": _shares(
            line_notional, traded_notional, quote.base_price, slippage_factor
        ),
        "commission": commission,
        "slippage_cost": slippage_cost,
    }


def _shares(
    line_notional: float,
    traded_notional: float,
    base_price: float,
    slippage_factor: float,
) -> float:
    """Shares of a line, sized on base-currency prices; zero without a price."""
    effective_base_price = base_price * slippage_factor
    # The carried part of the book is valued at the close; only the traded part
    # pays (buys) or gives up (sells) the slippage.
    return (
        (line_notional - traded_notional) / base_price
        + traded_notional / effective_base_price
        if base_price and effective_base_price
        else 0.0
    )


__all__ = [
    "BPS_PER_UNIT",
    "DUST_WEIGHT",
    "CostModel",
    "FillQuote",
    "PricedTicket",
    "TicketLine",
    "held_weights",
    "price_ticket",
]
