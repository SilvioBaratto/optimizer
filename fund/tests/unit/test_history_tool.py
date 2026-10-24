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
from sqlalchemy import event

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


def _seed_activity_on(db_session, day: dt.date, weights: dict[str, float]) -> None:
    """Seed one overlay, run, order and a replacing holdings snapshot on ``day``."""
    PortfolioJournalRepository(db_session).upsert(
        _PID,
        day,
        run_id=None,
        trades={},
        allocation=weights,
        drift={},
        narrative=f"rebalance {day.isoformat()}",
    )
    runs = AgentRunRepository(db_session)
    run = runs.create_run(
        portfolio_id=_PID,
        asof=day,
        seed=1,
        universe=list(weights),
        optimizer_config={},
    )
    runs.finalize_run(run.id, weights=weights)
    OrderRepository(db_session).create(
        portfolio_id=_PID,
        asof=day,
        weights_hash=f"h-{day.isoformat()}",
        weights=weights,
        fill_date=day + dt.timedelta(days=2),
        lines=[{"ticker": t, "weight": w} for t, w in weights.items()],
        notional=100_000.0,
        total_commission=10.0,
        total_slippage_cost=5.0,
    )
    PositionRepository(db_session).set_holdings(
        _PID,
        [{"ticker": t, "weight": w} for t, w in weights.items()],
        asof=day,
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

    def test_lookback_is_pushed_into_the_runs_query(self, db_session) -> None:
        """Stale runs are excluded by the SQL ``asof`` predicate, not after loading."""
        runs = AgentRunRepository(db_session)
        for asof in (dt.date(2022, 1, 3), dt.date(2024, 1, 20)):
            run = runs.create_run(
                portfolio_id=_PID,
                asof=asof,
                seed=1,
                universe=["AAA"],
                optimizer_config={},
            )
            runs.finalize_run(run.id, weights={"AAA": 1.0})
        statements: list[str] = []

        def _record(conn, cursor, statement, parameters, context, executemany):
            statements.append(statement)

        bind = db_session.get_bind()
        event.listen(bind, "before_cursor_execute", _record)
        try:
            result = get_portfolio_history(db_session, _PID, _ASOF, lookback=90)
        finally:
            event.remove(bind, "before_cursor_execute", _record)

        assert [r["asof"] for r in result["data"]["runs"]] == ["2024-01-20"]
        run_selects = [s for s in statements if "FROM agent_runs" in s]
        assert run_selects
        assert all("agent_runs.asof >=" in s for s in run_selects)


class TestNoLookAhead:
    """Activity and holdings dated after ``asof`` never reach a back-dated round."""

    def test_activity_after_asof_is_excluded(self, db_session) -> None:
        """Overlays, runs and orders dated after ``asof`` are dropped."""
        _seed_history(db_session)
        _seed_activity_on(db_session, dt.date(2024, 3, 1), {"CCC": 1.0})

        data = get_portfolio_history(db_session, _PID, _ASOF, lookback=90)["data"]

        assert [r["as_of"] for r in data["rebalances"]] == ["2024-01-20"]
        assert [r["asof"] for r in data["runs"]] == ["2024-01-20"]
        assert [o["asof"] for o in data["orders"]] == ["2024-01-20"]

    def test_activity_on_asof_is_included(self, db_session) -> None:
        """The upper bound is inclusive: a bar dated exactly ``asof`` is kept."""
        _seed_activity_on(db_session, _ASOF, {"AAA": 1.0})

        data = get_portfolio_history(db_session, _PID, _ASOF, lookback=90)["data"]

        assert [r["as_of"] for r in data["rebalances"]] == ["2024-02-01"]
        assert [r["asof"] for r in data["runs"]] == ["2024-02-01"]
        assert [o["asof"] for o in data["orders"]] == ["2024-02-01"]
        assert data["current_holdings"] == {"AAA": 1.0}

    def test_holdings_are_reconstructed_as_of_asof(self, db_session) -> None:
        """A snapshot written after ``asof`` is replaced by the then-current overlay."""
        _seed_history(db_session)
        _seed_activity_on(db_session, dt.date(2024, 3, 1), {"CCC": 1.0})

        data = get_portfolio_history(db_session, _PID, _ASOF, lookback=90)["data"]

        assert data["current_holdings"] == {"AAA": 0.6, "BBB": 0.4}

    def test_reconstructed_holdings_ignore_the_lookback_window(
        self, db_session
    ) -> None:
        """The last rebalance before ``asof`` still defines holdings when it is older
        than the lookback span."""
        _seed_history(db_session)
        _seed_activity_on(db_session, dt.date(2024, 3, 1), {"CCC": 1.0})

        data = get_portfolio_history(db_session, _PID, _ASOF, lookback=1)["data"]

        assert data["rebalances"] == []
        assert data["current_holdings"] == {"AAA": 0.6, "BBB": 0.4}

    def test_no_holdings_before_the_first_rebalance(self, db_session) -> None:
        """When every rebalance postdates ``asof`` there is nothing held yet."""
        _seed_activity_on(db_session, dt.date(2024, 3, 1), {"CCC": 1.0})

        data = get_portfolio_history(db_session, _PID, _ASOF, lookback=90)["data"]

        assert data["current_holdings"] == {}
        assert data["rebalances"] == []
        assert data["runs"] == []
        assert data["orders"] == []
