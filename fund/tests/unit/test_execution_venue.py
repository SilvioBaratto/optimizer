"""``fund.execution``: the venue ``place_orders`` hands a ticket to.

``place_orders`` builds an ``OrderRequest`` and delegates to an ``ExecutionVenue``,
the paper venue unless one is injected. These tests pin that a fake venue receives
the request and decides the tool's result without any edit to the tool, that the
run context forwards its venue, and that ``PaperVenue`` places, replays and refuses
from an injected quote source (no price panel needed).
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from portopt_db.models.orders.paper_order import PaperOrder
from sqlalchemy import func, select

from fund.agents.toolsets import RunContext, _place_ticket
from fund.config import FundConfig
from fund.execution import OrderRequest, PaperVenue, PlacedOrder
from fund.pricing import CostModel, FillQuote
from fund.tools.orders import place_orders

_ASOF = dt.date(2024, 1, 30)
_FILL = dt.date(2024, 1, 31)
_PID = uuid.UUID("77777777-7777-7777-7777-777777777777")
_WEIGHTS = {"AAA": 0.6, "BBB": 0.4}
_TICKET = {"order_id": "broker-1", "idempotent": False}


class _FakeVenue:
    """Records each request and answers with a scripted placement or exception."""

    def __init__(self, answer: PlacedOrder | Exception) -> None:
        self.answer = answer
        self.calls: list[tuple[object, OrderRequest]] = []

    def place(self, session, request: OrderRequest) -> PlacedOrder:
        """Record the call, then raise or return the scripted answer."""
        self.calls.append((session, request))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.mark.criterion("REQ-FUND-011")
class TestPlaceOrdersDelegatesToTheVenue:
    """``place_orders(venue=...)`` hands its request to the injected venue."""

    def test_the_venue_receives_the_normalised_request(self) -> None:
        """The request carries the coerced ids and dates and the cost model."""
        venue = _FakeVenue(PlacedOrder.placed(_TICKET))
        session = object()

        result = place_orders(
            session,
            _ASOF.isoformat(),
            _WEIGHTS,
            str(_PID),
            base_currency="EUR",
            venue=venue,
        )

        assert result == {"ok": True, "data": _TICKET}
        [(seen_session, request)] = venue.calls
        assert seen_session is session
        assert request == OrderRequest(
            portfolio_id=_PID,
            asof=_ASOF,
            weights=_WEIGHTS,
            base_currency="EUR",
            costs=CostModel(slippage_bps=5.0, commission_bps=1.0),
            notional=100_000.0,
        )

    def test_cost_overrides_reach_the_venue(self) -> None:
        """Notional, slippage and commission overrides are on the request."""
        venue = _FakeVenue(PlacedOrder.placed(_TICKET))

        place_orders(
            object(),
            _ASOF,
            _WEIGHTS,
            _PID,
            notional=250_000.0,
            slippage_bps=7.3,
            commission_bps=1.7,
            venue=venue,
        )

        request = venue.calls[0][1]
        assert request.costs == CostModel(slippage_bps=7.3, commission_bps=1.7)
        assert request.notional == 250_000.0
        assert request.base_currency is None

    def test_a_venue_refusal_is_the_tool_error(self) -> None:
        """A refusal comes back as ``err`` with the venue's text."""
        venue = _FakeVenue(PlacedOrder.refused("market closed"))

        result = place_orders(object(), _ASOF, _WEIGHTS, _PID, venue=venue)

        assert result == {"ok": False, "error": "market closed"}

    def test_a_venue_exception_is_caught_by_the_envelope(self) -> None:
        """An exception from the venue becomes ``err("<Type>: <msg>")``."""
        venue = _FakeVenue(RuntimeError("broker unreachable"))

        result = place_orders(object(), _ASOF, _WEIGHTS, _PID, venue=venue)

        assert result == {"ok": False, "error": "RuntimeError: broker unreachable"}

    def test_empty_weights_are_refused_before_the_venue(self) -> None:
        """Empty weights never reach the venue."""
        venue = _FakeVenue(PlacedOrder.placed(_TICKET))

        result = place_orders(object(), _ASOF, {}, _PID, venue=venue)

        assert result == {"ok": False, "error": "no weights to place"}
        assert venue.calls == []


class TestRunContextVenue:
    """``RunContext.venue`` is the venue the bound ``place_orders`` uses."""

    def test_defaults_to_none(self) -> None:
        """``None`` leaves ``place_orders`` on the paper venue."""
        ctx = RunContext(
            session=object(),
            asof=_ASOF,
            store=None,
            config=FundConfig(),
            run_id=uuid.uuid4(),
            portfolio_id=_PID,
        )

        assert ctx.venue is None

    def test_the_audited_ticket_goes_to_the_context_venue(self) -> None:
        """``_place_ticket`` hands the audited weights to ``ctx.venue``."""
        venue = _FakeVenue(PlacedOrder.placed(_TICKET))
        ctx = RunContext(
            session=object(),
            asof=_ASOF,
            store=None,
            config=FundConfig(),
            run_id=uuid.uuid4(),
            portfolio_id=_PID,
            base_currency="EUR",
            notional=50_000.0,
            venue=venue,
        )

        result = _place_ticket(ctx, _WEIGHTS)

        assert result == {"ok": True, "data": _TICKET}
        request = venue.calls[0][1]
        assert request.weights == _WEIGHTS
        assert request.notional == 50_000.0
        assert request.base_currency == "EUR"


class TestPlacedOrder:
    """A placed order holds a ticket or a refusal."""

    def test_holds_exactly_one_of_ticket_and_refusal(self) -> None:
        """Neither or both is rejected."""
        with pytest.raises(ValueError, match="ticket or a refusal"):
            PlacedOrder()
        with pytest.raises(ValueError, match="ticket or a refusal"):
            PlacedOrder(ticket=_TICKET, refusal="no")


def _quotes(session, tickers, asof, base_currency) -> dict[str, FillQuote]:
    """A fixed quote per ticker: 100 on the bar after ``asof``, in EUR."""
    return {
        ticker: FillQuote(
            date=asof + dt.timedelta(days=1), price=100.0, base_price=100.0, unit="EUR"
        )
        for ticker in tickers
    }


def _request(weights, *, slippage_bps: float = 5.0) -> OrderRequest:
    return OrderRequest(
        portfolio_id=_PID,
        asof=_ASOF,
        weights=weights,
        base_currency="EUR",
        costs=CostModel(slippage_bps=slippage_bps, commission_bps=1.0),
        notional=100_000.0,
    )


def _ticket_count(db_session) -> int:
    return db_session.execute(select(func.count()).select_from(PaperOrder)).scalar_one()


@pytest.mark.criterion("REQ-FUND-010")
class TestPaperVenue:
    """The paper venue over an injected quote source."""

    def test_places_and_stores_a_ticket(self, db_session) -> None:
        """A fresh key is priced from the quotes and stored."""
        placed = PaperVenue(_quotes).place(db_session, _request(_WEIGHTS))

        assert placed.refusal is None
        assert placed.ticket is not None
        assert placed.ticket["fill_date"] == _FILL.isoformat()
        assert placed.ticket["idempotent"] is False
        assert [line["ticker"] for line in placed.ticket["lines"]] == ["AAA", "BBB"]
        assert _ticket_count(db_session) == 1

    def test_a_rerun_returns_the_stored_ticket(self, db_session) -> None:
        """The same key with the same cost params replays the stored ticket."""
        venue = PaperVenue(_quotes)
        first = venue.place(db_session, _request(_WEIGHTS))

        second = venue.place(db_session, _request(_WEIGHTS))

        assert second.ticket is not None
        assert first.ticket is not None
        assert second.ticket["idempotent"] is True
        assert second.ticket["order_id"] == first.ticket["order_id"]
        assert _ticket_count(db_session) == 1

    def test_a_rerun_with_other_cost_params_is_refused(self, db_session) -> None:
        """The same key priced with other params is refused, not re-priced."""
        venue = PaperVenue(_quotes)
        venue.place(db_session, _request(_WEIGHTS))

        placed = venue.place(db_session, _request(_WEIGHTS, slippage_bps=9.0))

        assert placed.ticket is None
        assert placed.refusal is not None
        assert placed.refusal.startswith("a paper ticket already exists")

    def test_dust_weights_have_nothing_to_place(self, db_session) -> None:
        """All-dust weights are refused and nothing is stored."""
        placed = PaperVenue(_quotes).place(db_session, _request({"AAA": 1e-9}))

        assert placed == PlacedOrder.refused("no non-zero weights to place")
        assert _ticket_count(db_session) == 0
