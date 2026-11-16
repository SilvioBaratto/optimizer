"""Execution venues: where ``place_orders`` places a ticket.

``place_orders`` (``fund.tools.orders``) checks and normalises its arguments into an
:class:`OrderRequest` and hands it to an :class:`ExecutionVenue`. A venue answers
with a :class:`PlacedOrder`: the ticket, or the reason it placed none, which the tool
returns as its error. :class:`PaperVenue` is the venue the tool uses unless the
caller passes another one (``place_orders(venue=...)``, or the run context's
``venue`` for the bound agent tool), so a broker venue can be added without editing
the tool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from fund.audit.orders_repository import OrderRepository, weights_hash
from fund.pricing import BPS_PER_UNIT, held_weights, price_ticket

if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Iterable, Mapping

    from sqlalchemy.orm import Session

    from fund.pricing import CostModel, FillQuote

_EXISTS_WITH_OTHER_COSTS = (
    "a paper ticket already exists for this (portfolio, asof, weights) "
    "with different cost params; re-run with the placed params, or use a "
    "distinct key for a what-if with different "
    "notional/slippage/commission"
)


@dataclass(frozen=True)
class OrderRequest:
    """One order for a venue: a portfolio's target book at a decision bar.

    Attributes:
        portfolio_id: Owning portfolio.
        asof: The decision bar.
        weights: Target ``{ticker: weight}`` vector, zero and dust weights included.
        base_currency: Currency each line's ``shares`` are priced in; ``None`` uses
            prices as quoted.
        costs: Slippage and commission the fills are charged.
        notional: Gross book the weights scale.
    """

    portfolio_id: uuid.UUID
    asof: dt.date
    weights: Mapping[str, float]
    base_currency: str | None
    costs: CostModel
    notional: float


@dataclass(frozen=True)
class PlacedOrder:
    """What a venue did with a request: the ticket it holds for it, or why none.

    Attributes:
        ticket: The JSON-serialisable ticket, newly placed or returned for a re-run;
            ``None`` on a refusal.
        refusal: Why no ticket was placed; ``None`` when ``ticket`` is set.

    Raises:
        ValueError: Both or neither of ``ticket`` and ``refusal`` are set.
    """

    ticket: dict[str, Any] | None = None
    refusal: str | None = None

    def __post_init__(self) -> None:
        if (self.ticket is None) == (self.refusal is None):
            raise ValueError("a placed order holds either a ticket or a refusal")

    @classmethod
    def placed(cls, ticket: dict[str, Any]) -> PlacedOrder:
        """Return the outcome that carries ``ticket``."""
        return cls(ticket=ticket)

    @classmethod
    def refused(cls, reason: str) -> PlacedOrder:
        """Return the outcome that places no ticket, for ``reason``."""
        return cls(refusal=reason)


class ExecutionVenue(Protocol):
    """Turns an order request into a ticket, or into a refusal."""

    def place(self, session: Session, request: OrderRequest) -> PlacedOrder:
        """Place ``request``, or return the ticket already placed for it.

        Args:
            session: The caller's session; the caller owns its transaction.
            request: The order to place.

        Returns:
            The ticket, or the refusal ``place_orders`` returns as its error. An
            exception propagates to ``place_orders``, whose envelope reports it.
        """
        ...


class FillQuoteSource(Protocol):
    """Looks up where each held name of a paper ticket fills."""

    def __call__(
        self,
        session: Session,
        tickers: Iterable[str],
        asof: dt.date,
        base_currency: str | None,
        /,
    ) -> Mapping[str, FillQuote]:
        """Return the fill quote of each of ``tickers`` for decision bar ``asof``.

        Raises:
            LookupError: A ticker cannot fill; the ticket is not placed.
        """
        ...


class PaperVenue:
    """The simulated venue: it stores the ticket in ``paper_orders``, sends no order.

    The placement is idempotent on ``(portfolio_id, asof, weights_hash)``: the stored
    ticket for that key is returned (``idempotent: true``), re-stamped as the
    portfolio's newest when a later ticket exists, and the ``UNIQUE`` key
    ``uq_paper_order_key`` is the backstop. The cost params are not part of the key,
    so a re-run with other ones is refused rather than answered with a ticket priced
    differently.

    Args:
        fill_quotes: Looks up the fill quote of each held name; ``place_orders``
            passes its next-close lookup.
    """

    def __init__(self, fill_quotes: FillQuoteSource) -> None:
        self._fill_quotes = fill_quotes

    def place(self, session: Session, request: OrderRequest) -> PlacedOrder:
        """Place the paper ticket for ``request``, or return the one stored for it.

        Args:
            session: The caller's session; the ticket is flushed, not committed.
            request: The order to place.

        Returns:
            The new or stored ticket, or a refusal when the stored ticket for the key
            was priced with other cost params or every weight is zero or dust.

        Raises:
            LookupError: A held name cannot fill (from the quote source).
        """
        whash = weights_hash(dict(request.weights))
        repo = OrderRepository(session)
        existing = repo.get_by_key(
            portfolio_id=request.portfolio_id, asof=request.asof, weights_hash=whash
        )
        if existing is not None:
            return _replay(repo, existing, request)
        held = held_weights(request.weights)
        if not held:
            return PlacedOrder.refused("no non-zero weights to place")
        return self._place_new(session, repo, request, held, whash)

    def _place_new(
        self,
        session: Session,
        repo: OrderRepository,
        request: OrderRequest,
        held: dict[str, float],
        whash: str,
    ) -> PlacedOrder:
        prior = repo.latest_on_or_before(request.portfolio_id, request.asof)
        prior_weights = (
            {ticker: float(w) for ticker, w in prior.weights.items()} if prior else {}
        )
        quotes = self._fill_quotes(session, held, request.asof, request.base_currency)
        priced = price_ticket(
            held, prior_weights, quotes, request.notional, request.costs
        )
        order = repo.create(
            portfolio_id=request.portfolio_id,
            asof=request.asof,
            weights_hash=whash,
            weights={k: float(v) for k, v in request.weights.items()},
            fill_date=min(quote.date for quote in quotes.values()),
            lines=[dict(line) for line in priced.lines],
            notional=request.notional,
            total_commission=priced.total_commission,
            total_slippage_cost=priced.total_slippage_cost,
        )
        return PlacedOrder.placed(_ticket_from_row(order, idempotent=False))


def _replay(repo: OrderRepository, order: Any, request: OrderRequest) -> PlacedOrder:
    """Return the stored ``order`` for a re-run of its key, or refuse the request."""
    if _cost_params_match(
        order,
        notional=request.notional,
        slippage_bps=request.costs.slippage_bps,
        commission_bps=request.costs.commission_bps,
    ):
        # HITL re-run (D3): the ticket already exists priced with the same
        # params — return it, don't re-place.
        _make_newest(repo, request.portfolio_id, order)
        return PlacedOrder.placed(_ticket_from_row(order, idempotent=True))
    # Same weights-only key but different cost params: the stored ticket was
    # priced differently, so returning it would misreport the request.
    return PlacedOrder.refused(_EXISTS_WITH_OTHER_COSTS)


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


__all__ = [
    "ExecutionVenue",
    "FillQuoteSource",
    "OrderRequest",
    "PaperVenue",
    "PlacedOrder",
]
