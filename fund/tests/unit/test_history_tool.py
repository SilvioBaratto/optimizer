"""``get_portfolio_history``: the PM's window onto a portfolio's own past.

Drives the leaf tool against seeded overlay / run / order / holdings rows on the
in-memory ``db_session``, asserting the ``{ok, data}`` envelope, the lookback
summary, and that a bad ``asof`` degrades to ``{ok: false}`` rather than raising.
Uses a letter-bearing portfolio UUID so SQLite does not REAL-coerce it.
"""

from __future__ import annotations

import datetime as dt
import uuid

from portopt_db.repositories.orders.portfolio_journal_repository import (
    PortfolioJournalRepository,
)

from fund.audit import AgentRunRepository, OrderRepository, PositionRepository
from fund.tools.history import get_portfolio_history

_PID = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
_ASOF = dt.date(2024, 2, 1)


def _seed_history(db_session) -> None:
    """Seed one overlay, one finalized run, one paper order, and holdings."""
    PortfolioJournalRepository(db_session).upsert(
        _PID,
        dt.date(2024, 1, 20),
        run_id=None,
        trades={"AAA": {"weight": 0.6}},
        allocation={"AAA": 0.6, "BBB": 0.4},
        drift={},
        narrative="rebalance Jan 20",
    )
    PositionRepository(db_session).set_holdings(
        _PID,
        [{"ticker": "AAA", "weight": 0.6}, {"ticker": "BBB", "weight": 0.4}],
        asof=dt.date(2024, 1, 20),
    )
    runs = AgentRunRepository(db_session)
    run = runs.create_run(
        portfolio_id=_PID,
        asof=dt.date(2024, 1, 20),
        seed=1,
        universe=["AAA", "BBB"],
        optimizer_config={},
    )
    runs.finalize_run(run.id, weights={"AAA": 0.6, "BBB": 0.4})
    OrderRepository(db_session).create(
        portfolio_id=_PID,
        asof=dt.date(2024, 1, 20),
        weights_hash="h",
        weights={"AAA": 0.6, "BBB": 0.4},
        fill_date=dt.date(2024, 1, 22),
        lines=[{"ticker": "AAA", "weight": 0.6}],
        notional=100_000.0,
        total_commission=10.0,
        total_slippage_cost=5.0,
    )


class TestGetPortfolioHistory:
    """The tool rolls up a portfolio's recent rebalances, runs, orders, holdings."""

    def test_summarizes_prior_activity(self, db_session) -> None:
        """Seeded overlay/run/order/holdings surface in the lookback summary."""
        _seed_history(db_session)

        result = get_portfolio_history(db_session, _PID, _ASOF, lookback=90)

        assert result["ok"] is True
        data = result["data"]
        assert data["asof"] == "2024-02-01"
        assert data["portfolio_id"] == str(_PID)
        assert data["current_holdings"] == {"AAA": 0.6, "BBB": 0.4}
        assert len(data["rebalances"]) == 1
        assert data["rebalances"][0]["allocation"] == {"AAA": 0.6, "BBB": 0.4}
        assert len(data["runs"]) == 1
        assert data["runs"][0]["status"] == "completed"
        assert len(data["orders"]) == 1

    def test_empty_history_is_ok(self, db_session) -> None:
        """A portfolio with no history returns well-formed empty sections."""
        result = get_portfolio_history(db_session, _PID, _ASOF)

        assert result["ok"] is True
        data = result["data"]
        assert data["rebalances"] == []
        assert data["runs"] == []
        assert data["orders"] == []
        assert data["current_holdings"] == {}

    def test_bad_asof_returns_error_envelope(self, db_session) -> None:
        """A malformed ``asof`` is caught by the envelope, never raised."""
        result = get_portfolio_history(db_session, str(_PID), "not-a-date")

        assert result["ok"] is False
        assert "error" in result

    def test_accepts_string_portfolio_id(self, db_session) -> None:
        """A canonical UUID string is accepted like a ``UUID``."""
        result = get_portfolio_history(db_session, str(_PID), _ASOF)

        assert result["ok"] is True
        assert result["data"]["portfolio_id"] == str(_PID)
