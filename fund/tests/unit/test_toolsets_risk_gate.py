"""The bound risk gate: what blocks, and what ``place_orders`` re-checks.

Two holes are pinned here. A drawdown that cannot be measured used to pass the gate
with a warning, so a book with a young listing sailed through; and a risk verdict
stayed valid after the client was re-profiled, because ``place_orders`` re-checked
only the position cap and the instrument screens against the new mandate.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import uuid

from langgraph.store.memory import InMemoryStore
from portopt_db.models import AgentDecision
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument
from sqlalchemy import select

from fund.agents.toolsets import RunContext, bind_toolset
from fund.audit import AgentRunRepository, put_constraint_set
from fund.config import settings
from fund.schemas import ConstraintSet
from fund.schemas.enums import Horizon, ObjectiveChoice, RiskMeasureChoice

_START = dt.date(2023, 11, 1)
_ASOF = dt.date(2024, 1, 30)
_N_DAYS = 101
_PORTFOLIO_ID = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
_SERIES = {"AAA": (100.0, 0.0), "BBB": (50.0, 1.3), "CCC": (25.0, 2.6)}
_WEIGHTS = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
_YEAR_LEAD = 400


def _seed_panel(
    db_session, *, young_days: int | None = None, lead_days: int = 0
) -> None:
    """Seed three oscillating series; ``young_days`` shortens CCC to its last bars.

    ``lead_days`` further bars precede ``_START``, continuing the same oscillation: the
    gate judges the loss ceilings on the worst year, so a verdict needs more than a
    year of history behind the decision date.
    """
    for ticker, (start, phase) in _SERIES.items():
        exchange = Exchange(name=f"EX-{ticker}")
        db_session.add(exchange)
        db_session.flush()
        instrument = Instrument(
            ticker=ticker,
            short_name=ticker,
            exchange_id=exchange.id,
            instrument_type="EQUITY",
            asset_class="equity",
            yfinance_ticker=ticker,
        )
        db_session.add(instrument)
        db_session.flush()
        first = _N_DAYS - young_days if young_days and ticker == "CCC" else 0
        for i in range(first - lead_days, _N_DAYS):
            close = round(start * (1.0 + 0.02 * math.sin(0.5 * i + phase)), 6)
            db_session.add(
                PriceHistory(
                    instrument_id=instrument.id,
                    date=_START + dt.timedelta(days=i),
                    close=close,
                    volume=1000,
                )
            )
    db_session.flush()


def _constraint_set(**overrides: object) -> ConstraintSet:
    kwargs: dict[str, object] = {
        "portfolio_id": str(_PORTFOLIO_ID),
        "base_currency": "EUR",
        "a_gamma": 2.5,
        "objective": ObjectiveChoice.GROWTH,
        "risk_measure": RiskMeasureChoice.VARIANCE,
        "beta": 0.95,
        "nu1": 0.5,
        "nu2": 0.6,
        "nu3": 0.9,
        "horizon": Horizon.LONG,
    }
    kwargs.update(overrides)
    return ConstraintSet(**kwargs)  # type: ignore[arg-type]


def _store(cs: ConstraintSet) -> InMemoryStore:
    """An in-memory store holding ``cs`` as the portfolio's active constraint set."""
    store = InMemoryStore()
    put_constraint_set(store, cs, store_key=settings.constraint_set_store_key)
    return store


def _context(db_session, store: InMemoryStore) -> RunContext:
    """A run context over a freshly created audit run."""
    run = AgentRunRepository(db_session).create_run(
        portfolio_id=_PORTFOLIO_ID,
        asof=_ASOF,
        seed=0,
        universe=list(_SERIES),
        optimizer_config={"step": "toolset"},
    )
    return RunContext(
        session=db_session,
        asof=_ASOF,
        store=store,
        config=settings,
        run_id=run.id,
        portfolio_id=_PORTFOLIO_ID,
    )


def _audit_proposal(ctx: RunContext, weights: dict[str, float]) -> None:
    """Append the allocator decision a bound optimizer would have audited."""
    AgentRunRepository(ctx.session).append_decision(
        ctx.run_id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": weights}),
    )


def _tool(ctx: RunContext, role: str, name: str):
    """The bound tool ``name`` of ``role``."""
    (match,) = [t for t in bind_toolset(role, ctx) if t.name == name]
    return match


def _executor_rows(db_session) -> list[AgentDecision]:
    """The executor decisions a placed ticket appends."""
    return list(
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "executor")
        )
        .scalars()
        .all()
    )


class TestUnmeasurableDrawdownBlocks:
    """A drawdown the gate cannot measure is a failed check, not a footnote."""

    def test_a_history_too_short_to_backtest_fails_the_gate(self, db_session) -> None:
        """With no prices the ceilings were never checked, so the book cannot pass."""
        ctx = _context(db_session, _store(_constraint_set()))
        _audit_proposal(ctx, _WEIGHTS)

        result = _tool(ctx, "risk", "risk_check").invoke({})

        assert result["data"]["passed"] is False
        (violation,) = result["data"]["violations"]
        assert violation["type"] == "drawdown_unverified"
        assert result["data"]["warnings"] == []

    def test_a_young_listing_cannot_wave_a_book_through(self, db_session) -> None:
        """A holding with a fraction of the history leaves the drawdown unverified."""
        _seed_panel(db_session, young_days=12)
        ctx = _context(db_session, _store(_constraint_set()))
        _audit_proposal(ctx, _WEIGHTS)

        result = _tool(ctx, "risk", "risk_check").invoke({})

        assert result["data"]["passed"] is False
        kinds = [v["type"] for v in result["data"]["violations"]]
        assert kinds == ["drawdown_unverified"]

    def test_place_orders_refuses_a_book_whose_drawdown_cannot_be_measured(
        self, db_session
    ) -> None:
        """The refusal reaches the ticket: nothing is placed or audited."""
        _seed_panel(db_session, young_days=12)
        ctx = _context(db_session, _store(_constraint_set()))
        _audit_proposal(ctx, _WEIGHTS)
        _tool(ctx, "risk", "risk_check").invoke({})

        result = _tool(ctx, "orchestrator", "place_orders").invoke({})

        assert result["ok"] is False
        assert _executor_rows(db_session) == []


class TestReprofileInvalidatesAPassedVerdict:
    """``place_orders`` holds the audited weights to the mandate in force now."""

    def test_a_tighter_drawdown_ceiling_after_the_verdict_stops_the_order(
        self, db_session
    ) -> None:
        """The verdict passed at nu2=0.6; the client then accepts only a 0.01% loss."""
        _seed_panel(db_session, lead_days=_YEAR_LEAD)
        store = _store(_constraint_set())
        ctx = _context(db_session, store)
        _audit_proposal(ctx, _WEIGHTS)
        assert _tool(ctx, "risk", "risk_check").invoke({})["data"]["passed"] is True
        put_constraint_set(
            store,
            _constraint_set(nu1=0.00005, nu2=0.0001, nu3=0.0002),
            store_key=settings.constraint_set_store_key,
        )

        result = _tool(ctx, "orchestrator", "place_orders").invoke({})

        assert result["ok"] is False
        assert "drawdown_nu3" in result["error"]
        assert _executor_rows(db_session) == []

    def test_an_unchanged_mandate_still_places_the_order(self, db_session) -> None:
        """The re-check must not turn a clean approval into a refusal."""
        _seed_panel(db_session, lead_days=_YEAR_LEAD)
        ctx = _context(db_session, _store(_constraint_set()))
        _audit_proposal(ctx, _WEIGHTS)
        _tool(ctx, "risk", "risk_check").invoke({})

        result = _tool(ctx, "orchestrator", "place_orders").invoke({})

        assert result["ok"] is True
        assert len(_executor_rows(db_session)) == 1
