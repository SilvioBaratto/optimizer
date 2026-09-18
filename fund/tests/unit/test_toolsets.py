"""T7.3 — per-run toolset binding (``fund.agents.toolsets``).

Each frozen Phase-3 tool takes ``session`` as its first positional arg. Phase 7
binds it, per run, into a langchain ``@tool`` closure that exposes **only**
model-facing args (never a ``Session``) — mirroring the profiler's
``_make_save_profile``. These tests pin, over a seeded SQLite panel + an in-memory
LangGraph store:

* ``TOOLS_BY_AGENT`` keys equal the skill registry's, and its values partition the
  eight real ``fund.tools`` exports (a bijection);
* every bound tool is a ``BaseTool`` whose args schema omits ``session``;
* the closures return the ``{ok, data}`` envelope and never raise;
* the resolved ``ConstraintSet`` is translated into the tools' existing arg dicts;
* the bound ``optimize_portfolio`` / ``place_orders`` append the expected
  ``agent_decision`` rows on the injected session.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid

import pytest
from langchain_core.tools import BaseTool
from langgraph.store.memory import InMemoryStore
from portopt_db.models import AgentDecision
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument
from sqlalchemy import select

from fund import tools as fund_tools
from fund.agents.skills import SKILLS_BY_AGENT
from fund.agents.toolsets import TOOLS_BY_AGENT, RunContext, bind_toolset
from fund.audit import AgentRunRepository, put_constraint_set
from fund.config import settings
from fund.schemas import ConstraintSet
from fund.schemas.constraint_set import EsgPolicy
from fund.schemas.enums import GicsSector, Horizon, ObjectiveChoice, RiskMeasureChoice

_START = dt.date(2024, 1, 1)
_N_DAYS = 40
_ASOF = dt.date(2024, 1, 30)  # decision bar (index 29); bars 30..39 are future
_UNIVERSE = ["AAA", "BBB", "CCC"]
_WEIGHTS = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
# A UUID with hex letters: an all-digit UUID gets coerced to a float by SQLite's
# numeric affinity when a UUID column round-trips through ``refresh``.
_PORTFOLIO_ID = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
# (start, phase): distinct phases de-correlate the three return series so
# pre-selection keeps them all (a pure linear drift would make identical returns
# that ``drop_correlated`` prunes to nothing).
_SERIES = {"AAA": (100.0, 0.0), "BBB": (50.0, 1.3), "CCC": (25.0, 2.6)}


def _close_on(start: float, phase: float, i: int) -> float:
    """Deterministic close for day ``i``: a phased oscillation off ``start``."""
    return round(start * (1.0 + 0.02 * math.sin(0.5 * i + phase)), 6)


def _seed_panel(db_session) -> None:
    """Seed an ``_N_DAYS`` x 3 close panel extending past ``_ASOF``."""
    for ticker, (start, phase) in _SERIES.items():
        ex = Exchange(name=f"EX-{ticker}")
        db_session.add(ex)
        db_session.flush()
        inst = Instrument(
            ticker=ticker,
            short_name=ticker,
            exchange_id=ex.id,
            instrument_type="EQUITY",
            asset_class="equity",
            yfinance_ticker=ticker,
        )
        db_session.add(inst)
        db_session.flush()
        for i in range(_N_DAYS):
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=_START + dt.timedelta(days=i),
                    close=_close_on(start, phase, i),
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
        "nu1": 0.05,
        "nu2": 0.10,
        "nu3": 0.20,
        "horizon": Horizon.LONG,
    }
    kwargs.update(overrides)
    return ConstraintSet(**kwargs)  # type: ignore[arg-type]


def _make_context(db_session, *, store: InMemoryStore | None = None) -> RunContext:
    """Build a RunContext over a freshly created audit run."""
    run = AgentRunRepository(db_session).create_run(
        portfolio_id=_PORTFOLIO_ID,
        asof=_ASOF,
        seed=0,
        universe=_UNIVERSE,
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


def _store_with_cs(cs: ConstraintSet) -> InMemoryStore:
    store = InMemoryStore()
    put_constraint_set(store, cs, store_key=settings.constraint_set_store_key)
    return store


def _find(tools: list[BaseTool], name: str) -> BaseTool:
    (match,) = [t for t in tools if t.name == name]
    return match


# --- registry shape ---------------------------------------------------------


def test_tools_by_agent_keys_match_skill_registry() -> None:
    assert set(TOOLS_BY_AGENT) == set(SKILLS_BY_AGENT)


def test_tools_by_agent_is_a_bijection_over_fund_tools() -> None:
    assigned = [name for names in TOOLS_BY_AGENT.values() for name in names]
    # partition: every real tool assigned exactly once, nothing extra.
    assert sorted(assigned) == sorted(fund_tools.__all__)
    assert len(assigned) == len(set(assigned)) == len(fund_tools.__all__)


# --- BaseTool shape (session omitted) ---------------------------------------


def test_every_bound_tool_is_a_basetool_omitting_session(db_session) -> None:
    ctx = _make_context(db_session)
    for role in TOOLS_BY_AGENT:
        for tool in bind_toolset(role, ctx):
            assert isinstance(tool, BaseTool)
            assert "session" not in tool.args


def test_empty_roles_bind_no_tools(db_session) -> None:
    ctx = _make_context(db_session)
    assert bind_toolset("profiler", ctx) == []
    assert bind_toolset("executor", ctx) == []


def test_unknown_role_raises_key_error(db_session) -> None:
    ctx = _make_context(db_session)
    with pytest.raises(KeyError):
        bind_toolset("nope", ctx)


def test_run_context_is_frozen(db_session) -> None:
    ctx = _make_context(db_session)
    with pytest.raises((AttributeError, TypeError)):
        ctx.asof = dt.date(2020, 1, 1)  # type: ignore[misc]


# --- envelope round-trips ---------------------------------------------------


def test_bound_get_prices_returns_summary_envelope(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session)
    get_prices = _find(bind_toolset("economist", ctx), "get_prices")

    result = get_prices.invoke({"tickers": _UNIVERSE})

    assert result["ok"] is True
    assert result["data"]["columns"] == _UNIVERSE
    assert result["data"]["asof"] == _ASOF.isoformat()


def test_bound_macro_flags_missing_series(db_session) -> None:
    ctx = _make_context(db_session)
    macro = _find(bind_toolset("economist", ctx), "get_macro_series")

    result = macro.invoke({"names": ["CPIAUCSL"]})

    assert result["ok"] is True
    assert result["data"]["missing"] == ["CPIAUCSL"]


def test_bound_estimate_moments_returns_full_covariance(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session)
    moments = _find(bind_toolset("allocator", ctx), "estimate_moments")

    result = moments.invoke({"universe": _UNIVERSE})

    assert result["ok"] is True
    assert result["data"]["assets"] == _UNIVERSE
    assert len(result["data"]["cov"]) == len(_UNIVERSE)


def test_bound_universe_filter_runs_with_esg_translation(db_session) -> None:
    _seed_panel(db_session)
    cs = _constraint_set(esg=EsgPolicy(exclusions=(GicsSector.ENERGY,)))
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    ufilter = _find(bind_toolset("allocator", ctx), "universe_filter")

    result = ufilter.invoke({"universe": _UNIVERSE})

    # ESG exclusions are translated into the tool's `criteria` arg; the frozen
    # backbone ignores keys it does not model (R2), so the call still succeeds.
    assert result["ok"] is True
    assert set(result["data"]).issubset(set(_UNIVERSE))


def test_universe_filter_without_exclusions(db_session) -> None:
    _seed_panel(db_session)
    cs = _constraint_set()  # default ESG policy → no exclusions
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    ufilter = _find(bind_toolset("allocator", ctx), "universe_filter")

    result = ufilter.invoke({"universe": _UNIVERSE})

    assert result["ok"] is True
    assert set(result["data"]).issubset(set(_UNIVERSE))


def test_bound_risk_check_is_pure_and_omits_session(db_session) -> None:
    cs = _constraint_set()
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    risk = _find(bind_toolset("risk", ctx), "risk_check")

    assert "session" not in risk.args
    result = risk.invoke({"weights": _WEIGHTS})

    assert result["ok"] is True
    assert result["data"]["passed"] is True


def test_bound_backtest_returns_metrics(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session)
    backtest = _find(bind_toolset("risk", ctx), "backtest")

    result = backtest.invoke({"weights": _WEIGHTS})

    assert result["ok"] is True
    assert "sharpe_ratio" in result["data"]["metrics"]


def test_bound_tool_degrades_via_envelope_never_raises(db_session) -> None:
    ctx = _make_context(db_session)
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")

    # No priced assets for the universe: the envelope degrades, never raises.
    result = optimize.invoke({"universe": ["ZZZ"]})

    assert result["ok"] is False
    assert "error" in result


# --- audit: optimize / place_orders append agent_decision rows --------------


def test_bound_optimize_appends_allocator_decision(db_session) -> None:
    _seed_panel(db_session)
    cs = _constraint_set()
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")

    result = optimize.invoke({"universe": _UNIVERSE})

    assert result["ok"] is True
    assert set(result["data"]["weights"]) == set(_UNIVERSE)

    rows = (
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "allocator")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    decision = rows[0]
    assert decision.step == "optimize_portfolio"
    assert decision.constraint_set is not None
    assert decision.constraint_set["a_gamma"] == cs.a_gamma
    # load-bearing facts captured: the mapped optimizer config + the weights.
    assert "weights" in decision.llm_response
    assert "optimizer_config" in decision.llm_response
    assert decision.llm_response_hash is not None


def test_optimize_without_store_uses_default_constraints(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session, store=None)  # no CS to resolve
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")

    result = optimize.invoke({"universe": _UNIVERSE})

    assert result["ok"] is True
    rows = (
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "allocator")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].constraint_set is None  # no CS resolved → none recorded


def test_bound_place_orders_appends_executor_decision(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session)
    place = _find(bind_toolset("orchestrator", ctx), "place_orders")

    result = place.invoke({"weights": _WEIGHTS})

    assert result["ok"] is True
    ticket = result["data"]
    assert ticket["idempotent"] is False

    rows = (
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "executor")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    decision = rows[0]
    assert decision.step == "place_orders"
    assert decision.hitl_decision == {"decision": "approve"}


def test_place_orders_idempotent_rerun_appends_once(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session)
    place = _find(bind_toolset("orchestrator", ctx), "place_orders")

    first = place.invoke({"weights": _WEIGHTS})
    second = place.invoke({"weights": _WEIGHTS})

    assert first["data"]["idempotent"] is False
    assert second["data"]["idempotent"] is True  # HITL re-run (SPEC D3)

    rows = (
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "executor")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1  # the idempotent re-run does not double-log
