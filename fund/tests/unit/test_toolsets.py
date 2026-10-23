"""Per-run toolset binding (``fund.agents.toolsets``).

Each frozen tool takes ``session`` as its first positional arg. ``bind_toolset``
binds it, per run, into a langchain ``@tool`` closure that exposes **only**
model-facing args (never a ``Session``) — mirroring the profiler's
``_make_save_profile``. These tests pin, over a seeded SQLite panel + an in-memory
LangGraph store:

* ``TOOLS_BY_AGENT`` keys equal the skill registry's, and its values partition the
  real ``fund.tools`` exports (a bijection);
* every bound tool is a ``BaseTool`` whose args schema omits ``session``;
* the closures return the ``{ok, data}`` envelope and never raise;
* the resolved ``ConstraintSet`` is translated into the tools' existing arg dicts;
* the bound ``optimize_portfolio`` / ``place_orders`` append the expected
  ``agent_decision`` rows on the injected session.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import datetime as dt
import json
import math
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool
from langgraph.store.memory import InMemoryStore
from optimizer.moments import MomentEstimationConfig
from optimizer.optimization import MeanRiskConfig
from portopt_db.models import AgentDecision
from portopt_db.models.macro.macro_regime import FredObservation
from portopt_db.models.market_data.yfinance_data import PriceHistory, TickerProfile
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.market_data.market_journal_repository import (
    MarketJournalRepository,
)
from sqlalchemy import select

from fund import observe
from fund import tools as fund_tools
from fund.agents import graph as fund_graph
from fund.agents import prompts as fund_prompts
from fund.agents.skills import SKILLS_BY_AGENT, SKILLS_DIR
from fund.agents.toolsets import TOOLS_BY_AGENT, RunContext, bind_toolset
from fund.audit import AgentRunRepository, put_constraint_set
from fund.audit.orders_repository import weights_hash
from fund.config import settings
from fund.schemas import ConstraintSet
from fund.schemas.constraint_set import Bounds, EsgPolicy, UniverseFilters
from fund.schemas.enums import GicsSector, Horizon, ObjectiveChoice, RiskMeasureChoice
from fund.tools import moments as fund_moments
from fund.tools import optimize as fund_optimize

_START = dt.date(2023, 11, 1)
_N_DAYS = 101
# Decision bar (index 90): 90 return rows behind it, clear of the tools' common-sample
# floor; bars 91..100 are future.
_ASOF = dt.date(2024, 1, 30)
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


def _seed_panel(db_session, n_days: int = _N_DAYS) -> None:
    """Seed an ``n_days`` x 3 close panel (default extends past ``_ASOF``)."""
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
        for i in range(n_days):
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=_START + dt.timedelta(days=i),
                    close=_close_on(start, phase, i),
                    volume=1000,
                )
            )
    db_session.flush()


def _seed_long_panel(db_session, n_days: int = 160) -> None:
    """Seed ``n_days`` bars per ticker; the run date is ``_ASOF`` (bar 29)."""
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
        for i in range(n_days):
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


def _make_context(
    db_session,
    *,
    store: InMemoryStore | None = None,
    universe: list[str] = _UNIVERSE,
) -> RunContext:
    """Build a RunContext over a freshly created audit run."""
    run = AgentRunRepository(db_session).create_run(
        portfolio_id=_PORTFOLIO_ID,
        asof=_ASOF,
        seed=0,
        universe=universe,
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


def _classify(
    db_session,
    ticker: str,
    *,
    sector: str | None = None,
    instrument_type: str | None = None,
    name: str | None = None,
) -> None:
    """Give a seeded instrument a profile sector and/or a type and name."""
    instrument = db_session.execute(
        select(Instrument).where(Instrument.yfinance_ticker == ticker)
    ).scalar_one()
    if instrument_type is not None:
        instrument.instrument_type = instrument_type
    if name is not None:
        instrument.name = name
    if sector is not None:
        db_session.add(TickerProfile(instrument_id=instrument.id, sector=sector))
    db_session.flush()


def _audit_proposal(ctx: RunContext, weights: dict[str, float]) -> None:
    """Append an allocator ``optimize_portfolio`` decision carrying ``weights``, as
    the bound optimizer (or an adviser edit) would."""
    AgentRunRepository(ctx.session).append_decision(
        ctx.run_id,
        agent="allocator",
        step="optimize_portfolio",
        llm_response=json.dumps({"weights": weights}),
    )


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


# Skills write a tool invocation as a backticked ``name(args)`` whatever the verb
# ("Call", "Pull ... with", "Run"), so match the invocation itself, not one phrasing.
_CALL_INSTRUCTION = re.compile(r"`(\w+)\(")


def test_skill_call_pattern_matches_any_phrasing_of_an_invocation() -> None:
    body = "Pull the series with `get_macro_series(names)`, then call `risk_check(w)`."

    assert _CALL_INSTRUCTION.findall(body) == ["get_macro_series", "risk_check"]


@pytest.mark.parametrize("role", sorted(SKILLS_BY_AGENT))
def test_skills_only_instruct_calls_to_tools_their_role_binds(role: str) -> None:
    bound = set(TOOLS_BY_AGENT[role])
    for skill in SKILLS_BY_AGENT[role]:
        body = (SKILLS_DIR / skill / "SKILL.md").read_text(encoding="utf-8")
        called = {
            name
            for name in _CALL_INSTRUCTION.findall(body)
            if name in fund_tools.__all__
        }
        assert called <= bound, (
            f"{skill} tells the {role} to call {sorted(called - bound)}, "
            f"which that role does not bind"
        )


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


def test_bound_get_prices_hands_the_economist_numbers_to_read(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session)
    get_prices = _find(bind_toolset("economist", ctx), "get_prices")

    result = get_prices.invoke({"tickers": _UNIVERSE})

    stats = result["data"]["stats"]
    assert set(stats) == set(_UNIVERSE)
    assert all(entry["last"] is not None for entry in stats.values())
    assert "statistics" in get_prices.description.lower()
    assert "non-negative" in get_prices.description.lower()


def test_bound_macro_flags_missing_series(db_session) -> None:
    ctx = _make_context(db_session)
    macro = _find(bind_toolset("economist", ctx), "get_macro_series")

    result = macro.invoke({"names": ["CPIAUCSL"]})

    assert result["ok"] is True
    assert result["data"]["missing"] == ["CPIAUCSL"]


def test_bound_macro_returns_observed_values_for_the_economist(db_session) -> None:
    for day, value in {dt.date(2024, 1, 25): 0.31, dt.date(2024, 1, 26): -0.05}.items():
        db_session.add(FredObservation(series_id="T10Y2Y", date=day, value=value))
    db_session.flush()
    ctx = _make_context(db_session)
    macro = _find(bind_toolset("economist", ctx), "get_macro_series")

    result = macro.invoke({"names": ["T10Y2Y"]})

    assert result["ok"] is True
    assert result["data"]["latest"]["T10Y2Y"] == {"date": "2024-01-26", "value": -0.05}
    assert [p["value"] for p in result["data"]["recent"]["T10Y2Y"]] == [0.31, -0.05]


def test_economist_binds_the_regime_statistics_tool(db_session) -> None:
    ctx = _make_context(db_session)
    tools = bind_toolset("economist", ctx)

    regime = _find(tools, "get_regime_stats")

    assert "session" not in regime.args
    assert set(regime.args) == {"tickers"}


def test_bound_regime_stats_runs_over_the_run_date_panel(db_session) -> None:
    _seed_long_panel(db_session)
    asof = _START + dt.timedelta(days=119)
    ctx = dataclasses.replace(_make_context(db_session), asof=asof)
    regime = _find(bind_toolset("economist", ctx), "get_regime_stats")

    result = regime.invoke({"tickers": _UNIVERSE})

    assert result["ok"] is True
    assert result["data"]["asof"] == asof.isoformat()
    assert result["data"]["window_end"] == asof.isoformat()
    assert "turbulence" in result["data"]
    assert "absorption_ratio" in result["data"]


def test_bound_regime_stats_degrades_via_envelope_on_thin_history(db_session) -> None:
    _seed_panel(db_session)
    # Run date at bar 29: only 29 priced bars behind it, too few for a regime read.
    ctx = dataclasses.replace(
        _make_context(db_session), asof=_START + dt.timedelta(days=29)
    )
    regime = _find(bind_toolset("economist", ctx), "get_regime_stats")

    result = regime.invoke({"tickers": _UNIVERSE})

    assert result["ok"] is False
    assert "observations" in result["error"]


# --- history tools: role binding + envelope round-trips (T9) ------------------


def test_recent_events_binds_to_economist_history_to_orchestrator() -> None:
    assert "get_recent_events" in TOOLS_BY_AGENT["economist"]
    assert "get_portfolio_history" in TOOLS_BY_AGENT["orchestrator"]
    # Each is bound to exactly one role (the bijection guards this globally too).
    assert "get_recent_events" not in TOOLS_BY_AGENT["orchestrator"]
    assert "get_portfolio_history" not in TOOLS_BY_AGENT["economist"]


def test_bound_get_recent_events_returns_window_digests(db_session) -> None:
    MarketJournalRepository(db_session).upsert_journal(
        dt.date(2024, 1, 25),
        macro_deltas={},
        market_moves={},
        news_themes={"themes": {"rates": 1}},
        narrative="digest",
        source_counts={},
    )
    ctx = _make_context(db_session)
    events = _find(bind_toolset("economist", ctx), "get_recent_events")

    result = events.invoke({})

    assert result["ok"] is True
    assert result["data"]["asof"] == _ASOF.isoformat()
    assert result["data"]["count"] == 1


def test_bound_get_portfolio_history_returns_envelope(db_session) -> None:
    ctx = _make_context(db_session)
    history = _find(bind_toolset("orchestrator", ctx), "get_portfolio_history")

    result = history.invoke({})

    assert result["ok"] is True
    assert result["data"]["portfolio_id"] == str(_PORTFOLIO_ID)
    assert result["data"]["current_holdings"] == {}


@pytest.mark.parametrize("window", [-30, 0, 61, 10**9])
def test_get_recent_events_rejects_out_of_range_window(db_session, window) -> None:
    MarketJournalRepository(db_session).upsert_journal(
        dt.date(2024, 1, 25),
        macro_deltas={},
        market_moves={},
        news_themes={"themes": {"rates": 1}},
        narrative="digest",
        source_counts={},
    )
    events = _find(
        bind_toolset("economist", _make_context(db_session)), "get_recent_events"
    )

    result = events.invoke({"window": window})

    # An ``ok`` answer with zero events reads to the economist as "no macro
    # backdrop", so an out-of-range span must be an explicit error.
    assert result["ok"] is False
    assert "window" in result["error"]


@pytest.mark.parametrize("window", [1, 60])
def test_get_recent_events_accepts_window_bounds(db_session, window) -> None:
    events = _find(
        bind_toolset("economist", _make_context(db_session)), "get_recent_events"
    )

    result = events.invoke({"window": window})

    assert result["ok"] is True
    assert result["data"]["window"] == window


@pytest.mark.parametrize("lookback", [-30, 0, 366, 10**9])
def test_get_portfolio_history_rejects_out_of_range_lookback(
    db_session, lookback
) -> None:
    history = _find(
        bind_toolset("orchestrator", _make_context(db_session)),
        "get_portfolio_history",
    )

    result = history.invoke({"lookback": lookback})

    assert result["ok"] is False
    assert "lookback" in result["error"]


@pytest.mark.parametrize("lookback", [1, 365])
def test_get_portfolio_history_accepts_lookback_bounds(db_session, lookback) -> None:
    history = _find(
        bind_toolset("orchestrator", _make_context(db_session)),
        "get_portfolio_history",
    )

    result = history.invoke({"lookback": lookback})

    assert result["ok"] is True
    assert result["data"]["lookback"] == lookback


def test_bound_estimate_moments_returns_full_covariance(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session)
    moments = _find(bind_toolset("allocator", ctx), "estimate_moments")

    result = moments.invoke({"universe": _UNIVERSE})

    assert result["ok"] is True
    assert result["data"]["assets"] == _UNIVERSE
    assert len(result["data"]["cov"]) == len(_UNIVERSE)


def test_universe_filter_drops_esg_excluded_sectors(db_session) -> None:
    _seed_panel(db_session)
    _classify(db_session, "AAA", sector="Energy")
    _classify(db_session, "BBB", sector="Technology")
    _classify(db_session, "CCC", sector="Technology")
    cs = _constraint_set(esg=EsgPolicy(exclusions=(GicsSector.ENERGY,)))
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    ufilter = _find(bind_toolset("allocator", ctx), "universe_filter")

    result = ufilter.invoke({"universe": _UNIVERSE})

    assert result["ok"] is True
    assert "AAA" not in result["data"]
    assert set(result["data"]).issubset({"BBB", "CCC"})
    assert result["mandate_excluded"] == {"AAA": ["esg_exclusion"]}


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
    _audit_proposal(ctx, _WEIGHTS)
    risk = _find(bind_toolset("risk", ctx), "risk_check")

    assert "session" not in risk.args
    result = risk.invoke({})

    assert result["ok"] is True
    assert result["data"]["passed"] is True


def test_bound_risk_check_judges_the_audited_proposal(db_session) -> None:
    _seed_panel(db_session)
    cs = _constraint_set()
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    _find(bind_toolset("allocator", ctx), "optimize_portfolio").invoke(
        {"universe": _UNIVERSE}
    )
    risk = _find(bind_toolset("risk", ctx), "risk_check")

    result = risk.invoke({})

    assert result["ok"] is True
    assert result["data"]["passed"] is True


def test_risk_check_and_backtest_take_no_model_supplied_weights(db_session) -> None:
    ctx = _make_context(db_session)
    risk_tools = bind_toolset("risk", ctx)

    assert "weights" not in _find(risk_tools, "risk_check").args
    assert "weights" not in _find(risk_tools, "backtest").args


def test_risk_check_ignores_a_retyped_weight_vector(db_session) -> None:
    """The risk officer only sees weights as prose and re-types them; judging that
    rounded copy would miss the budget tolerance, so the audited vector is judged."""
    ctx = _make_context(db_session)
    third = 1.0 / 3.0
    _audit_proposal(ctx, {"AAA": third, "BBB": third, "CCC": 1.0 - 2.0 * third})
    risk = _find(bind_toolset("risk", ctx), "risk_check")

    result = risk.invoke({"weights": {"AAA": 0.3333, "BBB": 0.3333, "CCC": 0.3333}})

    assert result["ok"] is True
    assert result["data"]["passed"] is True


def test_risk_check_follows_the_latest_audited_proposal(db_session) -> None:
    cs = _constraint_set(universe_filters=UniverseFilters(max_position_cap=0.4))
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    risk = _find(bind_toolset("risk", ctx), "risk_check")

    _audit_proposal(ctx, {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3})
    assert risk.invoke({})["data"]["passed"] is True

    # A re-optimised proposal supersedes the first: the gate must judge it, not
    # whatever vector an earlier step was shown.
    _audit_proposal(ctx, {"AAA": 0.7, "BBB": 0.2, "CCC": 0.1})
    result = risk.invoke({})

    assert result["data"]["passed"] is False
    assert [v["type"] for v in result["data"]["violations"]] == ["max_weight"]


def test_risk_check_errs_when_no_proposal_was_audited(db_session) -> None:
    ctx = _make_context(db_session)
    risk = _find(bind_toolset("risk", ctx), "risk_check")

    result = risk.invoke({})

    assert result["ok"] is False
    assert "optimize_portfolio" in result["error"]


def test_bound_backtest_returns_metrics(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session)
    _audit_proposal(ctx, _WEIGHTS)
    backtest = _find(bind_toolset("risk", ctx), "backtest")

    result = backtest.invoke({})

    assert result["ok"] is True
    assert "sharpe_ratio" in result["data"]["metrics"]


def test_bound_backtest_flags_optimizer_weights_as_in_sample(db_session) -> None:
    """The optimizer fits on every bar up to ``asof``, so the walk-forward test
    blocks lie inside the weights' own estimation sample: not out-of-sample."""
    n_days = 130
    _seed_panel(db_session, n_days=n_days)
    ctx = dataclasses.replace(
        _make_context(db_session),
        asof=_START + dt.timedelta(days=n_days - 1),
        lookback_days=20,
    )
    _find(bind_toolset("allocator", ctx), "optimize_portfolio").invoke(
        {"universe": _UNIVERSE}
    )
    backtest = _find(bind_toolset("risk", ctx), "backtest")

    result = backtest.invoke({})

    assert result["ok"] is True
    assert result["data"]["n_folds"] >= 1
    assert result["data"]["out_of_sample"] is False


def test_bound_backtest_description_does_not_claim_independent_validation(
    db_session,
) -> None:
    backtest = _find(bind_toolset("risk", _make_context(db_session)), "backtest")

    assert "no leakage" not in backtest.description
    assert "in-sample" in backtest.description


def test_backtest_errs_when_no_proposal_was_audited(db_session) -> None:
    ctx = _make_context(db_session)
    backtest = _find(bind_toolset("risk", ctx), "backtest")

    result = backtest.invoke({})

    assert result["ok"] is False
    assert "optimize_portfolio" in result["error"]


def test_bound_backtest_reports_which_drawdown_tiers_are_breached(db_session) -> None:
    _seed_panel(db_session)
    cs = _constraint_set(nu1=0.0001, nu2=0.0002, nu3=0.0003)  # any drawdown breaches
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    _audit_proposal(ctx, _WEIGHTS)
    backtest = _find(bind_toolset("risk", ctx), "backtest")

    result = backtest.invoke({})

    assert result["ok"] is True
    tiers = result["data"]["drawdown_tiers"]
    assert tiers["breached"] == ["nu1", "nu2", "nu3"]
    assert tiers["max_drawdown"] == result["data"]["full_sample_max_drawdown"]
    assert (tiers["nu1"], tiers["nu2"], tiers["nu3"]) == (0.0001, 0.0002, 0.0003)


def test_bound_backtest_reports_no_breach_inside_every_ceiling(db_session) -> None:
    _seed_panel(db_session)
    cs = _constraint_set(nu1=0.9, nu2=0.95, nu3=1.0)
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    _audit_proposal(ctx, _WEIGHTS)
    backtest = _find(bind_toolset("risk", ctx), "backtest")

    result = backtest.invoke({})

    assert result["data"]["drawdown_tiers"]["breached"] == []


def test_bound_backtest_judges_drawdown_tiers_over_the_full_history(db_session) -> None:
    # The crash sits inside the first lookback window, which the walk-forward test
    # blocks skip. The tier verdict must still see it: the weights lived through it.
    exchange = Exchange(name="EX-AAA")
    db_session.add(exchange)
    db_session.flush()
    inst = Instrument(
        ticker="AAA",
        short_name="AAA",
        exchange_id=exchange.id,
        instrument_type="EQUITY",
        asset_class="equity",
        yfinance_ticker="AAA",
    )
    db_session.add(inst)
    db_session.flush()
    n_days = 120
    for i in range(n_days):
        level = 100.0 * (1.0 - 0.04 * i) if i < 10 else 64.0 * (1.0 + 0.003 * (i - 9))
        close = round(level * (1.0 + 0.0005 * math.sin(i)), 6)
        db_session.add(
            PriceHistory(
                instrument_id=inst.id,
                date=_ASOF - dt.timedelta(days=n_days - 1 - i),
                close=close,
                volume=1000,
            )
        )
    db_session.flush()
    ctx = dataclasses.replace(
        _make_context(db_session, store=_store_with_cs(_constraint_set())),
        lookback_days=10,
    )
    _audit_proposal(ctx, {"AAA": 1.0})
    backtest = _find(bind_toolset("risk", ctx), "backtest")

    tiers = backtest.invoke({})["data"]["drawdown_tiers"]

    assert "nu3" in tiers["breached"]
    assert tiers["max_drawdown"] > 0.3


def test_bound_backtest_rejects_an_audited_proposal_that_misses_the_budget(
    db_session,
) -> None:
    # A partially invested proposal would be replayed as idle cash and understate the
    # drawdown the ceilings are compared against.
    _seed_panel(db_session)
    ctx = _make_context(db_session, store=_store_with_cs(_constraint_set()))
    _audit_proposal(ctx, {"AAA": 0.4, "BBB": 0.2, "CCC": 0.1})
    backtest = _find(bind_toolset("risk", ctx), "backtest")

    result = backtest.invoke({})

    assert result["ok"] is False
    assert "budget" in result["error"]


def test_bound_backtest_honours_a_non_default_constraint_set_budget(
    db_session,
) -> None:
    _seed_panel(db_session)
    cs = _constraint_set(bounds=Bounds(budget=0.5))
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    _audit_proposal(ctx, {"AAA": 0.25, "BBB": 0.15, "CCC": 0.1})
    backtest = _find(bind_toolset("risk", ctx), "backtest")

    result = backtest.invoke({})

    assert result["ok"] is True


def test_bound_backtest_without_a_constraint_set_has_no_tier_verdict(
    db_session,
) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session, store=None)
    _audit_proposal(ctx, _WEIGHTS)
    backtest = _find(bind_toolset("risk", ctx), "backtest")

    result = backtest.invoke({})

    assert result["ok"] is True
    assert "drawdown_tiers" not in result["data"]


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


def test_optimize_audit_persists_the_metrics_fund_status_reads(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session, store=_store_with_cs(_constraint_set()))
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")

    result = optimize.invoke({"universe": _UNIVERSE})

    assert result["ok"] is True
    # Status reads the proposal of a run waiting at its gate (the run that opened the
    # round is still pending until it pauses).
    AgentRunRepository(db_session).mark_paused(ctx.run_id)
    state = observe.portfolio_state(db_session, _PORTFOLIO_ID)
    assert state.metrics == pytest.approx(result["data"]["metrics"])
    assert set(state.metrics) == {"mean", "standard_deviation", "sharpe_ratio"}


def test_bound_optimize_persists_the_optimizer_metrics(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session, store=_store_with_cs(_constraint_set()))
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")

    result = optimize.invoke({"universe": _UNIVERSE})

    assert result["ok"] is True
    (decision,) = (
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "allocator")
        )
        .scalars()
        .all()
    )
    logged = json.loads(decision.llm_response)["metrics"]
    assert logged == pytest.approx(result["data"]["metrics"])
    assert set(logged) == {"mean", "standard_deviation", "sharpe_ratio"}


def test_bound_optimize_records_the_universe_on_the_run(db_session) -> None:
    """A run is opened with an empty universe (the allocator picks its candidates at
    runtime). Once ``optimize_portfolio`` succeeds the run holds the tickers the
    optimiser actually allocated over, without the ones it dropped for lack of prices;
    the allocator decision keeps both the requested universe and the dropped tickers,
    so an auditor can see what was asked for and what was left out."""
    _seed_panel(db_session)
    ctx = _make_context(db_session, universe=[])
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")
    requested = [*_UNIVERSE, "ZZZ"]

    result = optimize.invoke({"universe": requested})

    assert result["ok"] is True
    run = AgentRunRepository(db_session).get_run(ctx.run_id)
    assert run is not None
    assert run.universe == _UNIVERSE
    (decision,) = (
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "allocator")
        )
        .scalars()
        .all()
    )
    payload = json.loads(decision.llm_response)
    assert payload["universe"] == requested
    assert payload["missing"] == ["ZZZ"]


def test_failed_optimize_leaves_the_run_universe_untouched(db_session) -> None:
    ctx = _make_context(db_session, universe=[])
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")

    result = optimize.invoke({"universe": ["ZZZ"]})

    assert result["ok"] is False
    run = AgentRunRepository(db_session).get_run(ctx.run_id)
    assert run is not None
    assert run.universe == []


def test_optimize_applies_and_logs_the_mapped_mifid_config(db_session) -> None:
    """The client's objective / risk measure / aversion / tail beta reach the
    optimizer, and the audit logs the config that was actually applied."""
    _seed_panel(db_session)
    cs = _constraint_set(
        objective=ObjectiveChoice.MAX,
        risk_measure=RiskMeasureChoice.CVAR,
        a_gamma=7.0,
        beta=0.99,
        l2_coef=0.2,
    )
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")

    result = optimize.invoke({"universe": _UNIVERSE})
    assert result["ok"] is True

    (decision,) = (
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "allocator")
        )
        .scalars()
        .all()
    )
    logged = json.loads(decision.llm_response)["optimizer_config"]
    assert logged["objective"] == "maximize_ratio"
    assert logged["risk_measure"] == "cvar"
    assert logged["risk_aversion"] == 7.0
    assert logged["cvar_beta"] == 0.99
    assert logged["l2_coef"] == 0.2
    assert logged["min_weights"] == cs.bounds.min_weights
    assert logged["max_weights"] == cs.bounds.max_weights
    assert logged["budget"] == cs.bounds.budget
    assert decision.constraint_set["objective"] == "max"
    assert decision.constraint_set["risk_measure"] == "cvar"

    # The logged config is not decorative: a protection/variance profile on the
    # same panel produces a different book.
    protection = _find(
        bind_toolset(
            "allocator",
            _make_context(
                db_session,
                store=_store_with_cs(
                    _constraint_set(
                        objective=ObjectiveChoice.PROTECTION,
                        risk_measure=RiskMeasureChoice.VARIANCE,
                    )
                ),
            ),
        ),
        "optimize_portfolio",
    ).invoke({"universe": _UNIVERSE})
    assert protection["ok"] is True
    gap = max(
        abs(result["data"]["weights"][t] - protection["data"]["weights"][t])
        for t in _UNIVERSE
    )
    assert gap > 1e-3


def test_optimize_audits_the_risk_aversion_it_actually_ran_with(db_session) -> None:
    """A tail-measure utility profile has its ``a_gamma`` rescaled by the tool; the
    audit row must carry the value the solver used, not the mapped one."""
    _seed_panel(db_session)
    cs = _constraint_set(risk_measure=RiskMeasureChoice.CVAR, a_gamma=5.0)
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")

    result = optimize.invoke({"universe": _UNIVERSE})

    (decision,) = (
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "allocator")
        )
        .scalars()
        .all()
    )
    logged = json.loads(decision.llm_response)["optimizer_config"]
    assert logged["risk_aversion"] == result["data"]["risk_aversion"]
    assert logged["risk_aversion"] < cs.a_gamma
    assert decision.constraint_set["a_gamma"] == 5.0


def test_defensive_and_aggressive_profiles_allocate_differently(db_session) -> None:
    _seed_panel(db_session)
    weights = {}
    for name, a_gamma, objective in (
        ("defensive", 12.0, ObjectiveChoice.PROTECTION),
        ("aggressive", 1.0, ObjectiveChoice.GROWTH),
    ):
        cs = _constraint_set(a_gamma=a_gamma, objective=objective)
        ctx = _make_context(db_session, store=_store_with_cs(cs))
        optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")
        result = optimize.invoke({"universe": _UNIVERSE})
        assert result["ok"] is True
        weights[name] = result["data"]["weights"]

    l1_distance = sum(
        abs(weights["defensive"][t] - weights["aggressive"][t]) for t in _UNIVERSE
    )
    assert l1_distance > 0.05


def test_optimize_caps_weights_at_max_position_cap(db_session) -> None:
    """A K&E position cap is a hard weight bound, not just a display label."""
    _seed_panel(db_session)
    cap = 0.4
    cs = _constraint_set(universe_filters=UniverseFilters(max_position_cap=cap))
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")

    result = optimize.invoke({"universe": _UNIVERSE})

    # Unconstrained min-variance on this panel is ~0.51 / 0 / 0.49, so the cap binds.
    assert result["ok"] is True
    assert max(result["data"]["weights"].values()) <= cap + 1e-6


def test_optimize_reports_cap_infeasible_for_a_too_small_universe(db_session) -> None:
    _seed_panel(db_session)
    cs = _constraint_set(universe_filters=UniverseFilters(max_position_cap=0.2))
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")

    result = optimize.invoke({"universe": _UNIVERSE})

    # 3 assets x 20% cannot reach a fully invested book; the model is told why
    # rather than receiving an opaque solver failure.
    assert result["ok"] is False
    assert "at least 5" in result["error"]


def test_risk_check_flags_a_weight_above_the_position_cap(db_session) -> None:
    cs = _constraint_set(universe_filters=UniverseFilters(max_position_cap=0.4))
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    _audit_proposal(ctx, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    risk = _find(bind_toolset("risk", ctx), "risk_check")

    result = risk.invoke({})

    assert result["ok"] is True
    assert result["data"]["passed"] is False
    (violation,) = result["data"]["violations"]
    assert violation["type"] == "max_weight"
    assert violation["asset"] == "AAA"
    assert violation["limit"] == 0.4


def _allocate(db_session, **overrides: object) -> dict[str, float]:
    cs = _constraint_set(**overrides)
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")
    result = optimize.invoke({"universe": _UNIVERSE})
    assert result["ok"] is True
    return result["data"]["weights"]


def _l1(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(abs(a[t] - b[t]) for t in _UNIVERSE)


@pytest.mark.parametrize(
    "measure",
    [
        RiskMeasureChoice.CVAR,
        RiskMeasureChoice.CDAR,
        RiskMeasureChoice.MAX_DRAWDOWN,
    ],
)
def test_growth_over_a_tail_measure_is_not_collapsed_into_minimum_risk(
    db_session, measure: RiskMeasureChoice
) -> None:
    """``a_gamma`` is calibrated for variance (~1e-4 per day); a first-order tail
    measure (~1e-2) swamps the return term unless the aversion is rescaled, which
    made an Aggressive growth client's utility objective return the protection
    portfolio."""
    _seed_panel(db_session)

    protection = _allocate(
        db_session,
        objective=ObjectiveChoice.PROTECTION,
        risk_measure=measure,
        a_gamma=12.0,
    )
    aggressive = _allocate(
        db_session,
        objective=ObjectiveChoice.GROWTH,
        risk_measure=measure,
        a_gamma=1.0,
    )

    assert _l1(aggressive, protection) > 0.05


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
    # An audited allocator proposal must exist first: place_orders sources the
    # ticket from the optimizer's audited weights, never the model's arg.
    optimized = _find(bind_toolset("allocator", ctx), "optimize_portfolio").invoke(
        {"universe": _UNIVERSE}
    )
    _find(bind_toolset("risk", ctx), "risk_check").invoke({})
    place = _find(bind_toolset("orchestrator", ctx), "place_orders")

    result = place.invoke({})

    assert result["ok"] is True
    ticket = result["data"]
    assert ticket["idempotent"] is False
    # The ticket carries the optimizer's audited weights.
    assert result["data"]["weights"] == optimized["data"]["weights"]

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


def test_place_orders_description_says_the_call_itself_awaits_approval(
    db_session,
) -> None:
    """The HITL gate pauses at the ``place_orders`` call, so the model-facing text
    must not tell the model to wait for approval before calling it (a model that
    does ends the run with no ticket)."""
    place = _find(
        bind_toolset("orchestrator", _make_context(db_session)), "place_orders"
    )
    description = " ".join(place.description.split())

    assert "after adviser approval" not in description
    assert "pauses" in description
    assert "adviser" in description


def test_place_orders_description_discloses_the_ticket_is_the_audited_proposal(
    db_session,
) -> None:
    place = _find(
        bind_toolset("orchestrator", _make_context(db_session)), "place_orders"
    )
    description = " ".join(place.description.split())

    assert "no weights" in description
    assert "audited" in description


def test_place_orders_idempotent_rerun_appends_once(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session)
    # Establish the audited optimizer weights the ticket hashes on both calls.
    _find(bind_toolset("allocator", ctx), "optimize_portfolio").invoke(
        {"universe": _UNIVERSE}
    )
    _find(bind_toolset("risk", ctx), "risk_check").invoke({})
    place = _find(bind_toolset("orchestrator", ctx), "place_orders")

    first = place.invoke({})
    second = place.invoke({})

    assert first["data"]["idempotent"] is False
    assert second["data"]["idempotent"] is True  # HITL re-run

    rows = (
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "executor")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1  # the idempotent re-run does not double-log


# --- the allocator tools run in the mandate's base currency -------------------


@pytest.mark.parametrize(
    ("tool_name", "role", "args"),
    [
        ("universe_filter", "allocator", {"universe": _UNIVERSE}),
        ("estimate_moments", "allocator", {"universe": _UNIVERSE}),
        ("optimize_portfolio", "allocator", {"universe": _UNIVERSE}),
        ("backtest", "risk", {}),
    ],
)
def test_bound_allocator_tools_forward_the_run_base_currency(
    db_session, monkeypatch: pytest.MonkeyPatch, tool_name: str, role: str, args: dict
) -> None:
    seen: dict[str, object] = {}

    def _spy(*_args: object, **kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"ok": False, "error": "spy"}

    monkeypatch.setattr(fund_tools, tool_name, _spy)
    ctx = dataclasses.replace(_make_context(db_session), base_currency="EUR")
    _audit_proposal(ctx, _WEIGHTS)

    _find(bind_toolset(role, ctx), tool_name).invoke(args)

    assert seen["base_currency"] == "EUR"


def test_bound_estimate_moments_sees_fx_returns_in_the_base_currency(
    db_session,
) -> None:
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    _seed_panel(db_session)
    db_session.query(PriceHistory).update({"price_unit": "USD"})
    # EUR per USD compounds +1% a day, so every USD asset gains ~1% a day in EUR
    # on top of its own USD return.
    FxRatesRepository(db_session).upsert_rates(
        [
            {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "date": _START + dt.timedelta(days=i),
                "rate": 0.90 * 1.01**i,
            }
            for i in range(_N_DAYS)
        ]
    )
    db_session.flush()
    local = _make_context(db_session)
    in_eur = dataclasses.replace(local, base_currency="EUR")

    mu_local = _find(bind_toolset("allocator", local), "estimate_moments").invoke(
        {"universe": _UNIVERSE}
    )["data"]["mu"]
    mu_eur = _find(bind_toolset("allocator", in_eur), "estimate_moments").invoke(
        {"universe": _UNIVERSE}
    )["data"]["mu"]

    for ticker in _UNIVERSE:
        assert mu_eur[ticker] - mu_local[ticker] == pytest.approx(0.01, abs=1e-3)


# --- mandate enforcement: ESG exclusions + K&E instrument restrictions ------


def _classify_all(db_session, sectors: dict[str, str]) -> None:
    """Seed the panel and give each ticker its profile sector."""
    _seed_panel(db_session)
    for ticker, sector in sectors.items():
        _classify(db_session, ticker, sector=sector)


def _risk_check(db_session, cs: ConstraintSet, weights: dict[str, float]) -> dict:
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    _audit_proposal(ctx, weights)
    return _find(bind_toolset("risk", ctx), "risk_check").invoke({})


def _energy_exclusion() -> ConstraintSet:
    return _constraint_set(esg=EsgPolicy(exclusions=(GicsSector.ENERGY,)))


def test_optimize_refuses_a_universe_holding_an_esg_excluded_name(db_session) -> None:
    _seed_panel(db_session)
    _classify(db_session, "AAA", sector="Energy")
    _classify(db_session, "BBB", sector="Technology")
    _classify(db_session, "CCC", sector="Technology")
    ctx = _make_context(db_session, store=_store_with_cs(_energy_exclusion()))
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")

    result = optimize.invoke({"universe": _UNIVERSE})

    assert result["ok"] is False
    assert "AAA" in result["error"]
    assert "esg_exclusion" in result["error"]
    # A refused proposal leaves no audited weights behind to be placed.
    assert (
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "allocator")
        )
        .scalars()
        .all()
        == []
    )


def test_risk_check_fails_a_held_esg_excluded_name(db_session) -> None:
    _classify_all(
        db_session, {"AAA": "Energy", "BBB": "Technology", "CCC": "Utilities"}
    )

    result = _risk_check(db_session, _energy_exclusion(), _WEIGHTS)

    assert result["ok"] is True
    assert result["data"]["passed"] is False
    (violation,) = result["data"]["violations"]
    assert violation["type"] == "esg_exclusion"
    assert violation["asset"] == "AAA"
    assert violation["sector"] == "Energy"


def test_risk_check_ignores_an_excluded_name_held_at_zero(db_session) -> None:
    _classify_all(
        db_session, {"AAA": "Energy", "BBB": "Technology", "CCC": "Utilities"}
    )

    result = _risk_check(
        db_session, _energy_exclusion(), {"AAA": 0.0, "BBB": 0.6, "CCC": 0.4}
    )

    assert result["data"]["passed"] is True


def test_risk_check_fails_closed_when_a_sector_cannot_be_classified(db_session) -> None:
    # No profile row for AAA: an exclusion that cannot be evaluated is a block.
    _seed_panel(db_session)
    _classify(db_session, "BBB", sector="Technology")
    _classify(db_session, "CCC", sector="Utilities")

    result = _risk_check(db_session, _energy_exclusion(), _WEIGHTS)

    assert result["data"]["passed"] is False
    (violation,) = result["data"]["violations"]
    assert violation["type"] == "unclassified_sector"
    assert violation["asset"] == "AAA"


def test_risk_check_does_not_classify_sectors_without_exclusions(db_session) -> None:
    _seed_panel(db_session)  # no profiles at all

    result = _risk_check(db_session, _constraint_set(), _WEIGHTS)

    assert result["data"]["passed"] is True


def test_risk_check_flags_a_leveraged_etf_for_a_no_leverage_client(db_session) -> None:
    _seed_panel(db_session)
    _classify(db_session, "AAA", instrument_type="ETF", name="ProShares Ultra QQQ 2x")
    # A common stock whose name merely contains "Ultra" is not a leveraged product.
    _classify(db_session, "BBB", instrument_type="STOCK", name="Ultra Clean Holdings")
    cs = _constraint_set(universe_filters=UniverseFilters(no_leverage=True))

    result = _risk_check(db_session, cs, _WEIGHTS)

    assert result["data"]["passed"] is False
    (violation,) = result["data"]["violations"]
    assert violation["type"] == "leveraged_instrument"
    assert violation["asset"] == "AAA"


def test_risk_check_flags_a_non_plain_instrument_for_a_no_complex_client(
    db_session,
) -> None:
    _seed_panel(db_session)
    _classify(db_session, "AAA", instrument_type="ETC", name="Gold Bullion Securities")
    cs = _constraint_set(universe_filters=UniverseFilters(no_complex=True))

    result = _risk_check(db_session, cs, _WEIGHTS)

    assert result["data"]["passed"] is False
    (violation,) = result["data"]["violations"]
    assert violation["type"] == "complex_instrument"
    assert violation["asset"] == "AAA"


def test_place_orders_refuses_weights_that_breach_the_current_mandate(
    db_session,
) -> None:
    _classify_all(
        db_session, {"AAA": "Energy", "BBB": "Technology", "CCC": "Utilities"}
    )
    store = _store_with_cs(_constraint_set())
    ctx = _make_context(db_session, store=store)
    _find(bind_toolset("allocator", ctx), "optimize_portfolio").invoke(
        {"universe": _UNIVERSE}
    )
    _find(bind_toolset("risk", ctx), "risk_check").invoke({})
    # The client is re-profiled with an Energy exclusion after the audited proposal.
    put_constraint_set(
        store, _energy_exclusion(), store_key=settings.constraint_set_store_key
    )
    place = _find(bind_toolset("orchestrator", ctx), "place_orders")

    result = place.invoke({})

    assert result["ok"] is False
    assert "esg_exclusion" in result["error"]
    assert (
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "executor")
        )
        .scalars()
        .all()
        == []
    )


# --- one Session shared by every bound tool of a run ------------------------


def test_bound_tool_bodies_of_one_run_are_serialised(db_session, monkeypatch) -> None:
    """LangGraph dispatches parallel tool calls on worker threads; the run's single
    Session is not thread-safe, so no two bound tool bodies may overlap."""
    state = {"active": 0, "peak": 0}
    guard = threading.Lock()

    def slow_tool(*_args: object, **_kwargs: object) -> dict[str, object]:
        with guard:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.05)
        with guard:
            state["active"] -= 1
        return {"ok": True, "data": {}}

    monkeypatch.setattr(fund_tools, "get_prices", slow_tool)
    monkeypatch.setattr(fund_tools, "get_macro_series", slow_tool)
    monkeypatch.setattr(fund_tools, "get_recent_events", slow_tool)
    ctx = _make_context(db_session)
    economist = bind_toolset("economist", ctx)
    calls = [
        (_find(economist, "get_prices"), {"tickers": _UNIVERSE}),
        (_find(economist, "get_macro_series"), {"names": ["T10Y2Y"]}),
        (_find(economist, "get_recent_events"), {}),
        (_find(economist, "get_prices"), {"tickers": _UNIVERSE}),
    ]

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        results = list(pool.map(lambda call: call[0].invoke(call[1]), calls))

    assert all(result["ok"] for result in results)
    assert state["peak"] == 1


def test_a_failed_tool_call_does_not_poison_the_shared_session(
    db_session, monkeypatch
) -> None:
    """A swallowed flush error must roll back only that call, not leave every later
    tool returning ``PendingRollbackError`` and the run's own rows lost."""
    _seed_panel(db_session)
    ctx = _make_context(db_session)

    def colliding_append(self: AgentRunRepository, run_id: uuid.UUID, **_kw: object):
        for _ in range(2):  # same (run_id, decision_index): flush raises IntegrityError
            self.session.add(
                AgentDecision(run_id=run_id, decision_index=99, agent="a", step="s")
            )
        self.session.flush()

    monkeypatch.setattr(AgentRunRepository, "append_decision", colliding_append)
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")
    get_prices = _find(bind_toolset("economist", ctx), "get_prices")

    failed = optimize.invoke({"universe": _UNIVERSE})
    later = get_prices.invoke({"tickers": _UNIVERSE})

    assert failed["ok"] is False
    assert later["ok"] is True
    # The run row created before the tool ran survives the failed call.
    assert AgentRunRepository(db_session).get_run(ctx.run_id) is not None


def test_parallel_optimize_calls_each_append_one_distinct_decision(
    db_session, monkeypatch
) -> None:
    """Two optimize calls in one model turn must not collide on decision_index.

    ``append_decision`` derives the next index from a ``count()`` and inserts later.
    The rendezvous below parks each caller between the two until the other arrives,
    so without the run lock both would hold the same count and collide. Under the
    lock the second caller can never arrive while the first waits; the barrier times
    out and the calls simply run one after the other.
    """
    _seed_panel(db_session)
    ctx = _make_context(db_session, store=_store_with_cs(_constraint_set()))
    optimize = _find(bind_toolset("allocator", ctx), "optimize_portfolio")
    rendezvous = threading.Barrier(2, timeout=0.3)
    original_add = db_session.add

    def add_after_rendezvous(instance: object, *args: object, **kwargs: object):
        if isinstance(instance, AgentDecision):
            with contextlib.suppress(threading.BrokenBarrierError):
                rendezvous.wait()
        return original_add(instance, *args, **kwargs)

    monkeypatch.setattr(db_session, "add", add_after_rendezvous)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: optimize.invoke({"universe": _UNIVERSE}), range(2))
        )

    assert [r["ok"] for r in results] == [True, True]
    indexes = (
        db_session.execute(
            select(AgentDecision.decision_index).where(
                AgentDecision.agent == "allocator"
            )
        )
        .scalars()
        .all()
    )
    assert sorted(indexes) == [0, 1]


def test_place_orders_takes_no_model_supplied_weights(db_session) -> None:
    ctx = _make_context(db_session)
    place = _find(bind_toolset("orchestrator", ctx), "place_orders")

    assert "weights" not in place.args
    # Approval is requested BY calling it, so the text must not say "after approval".
    assert "after adviser approval" not in place.description


# --- the blocking risk gate is recorded and enforced -------------------------


def _risk_rows(db_session) -> list[AgentDecision]:
    return list(
        db_session.execute(
            select(AgentDecision)
            .where(AgentDecision.agent == "risk")
            .order_by(AgentDecision.decision_index)
        )
        .scalars()
        .all()
    )


def _executor_rows(db_session) -> list[AgentDecision]:
    return list(
        db_session.execute(
            select(AgentDecision).where(AgentDecision.agent == "executor")
        )
        .scalars()
        .all()
    )


def test_risk_check_records_its_verdict_against_the_audited_weights(
    db_session,
) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session, store=_store_with_cs(_constraint_set()))
    optimized = _find(bind_toolset("allocator", ctx), "optimize_portfolio").invoke(
        {"universe": _UNIVERSE}
    )

    result = _find(bind_toolset("risk", ctx), "risk_check").invoke({})

    assert result["data"]["passed"] is True
    (row,) = _risk_rows(db_session)
    assert row.step == "risk_check"
    verdict = json.loads(row.llm_response)
    assert verdict["passed"] is True
    assert verdict["violations"] == []
    # Same canonical hash as the paper ticket, so verdict and ticket cross-reference.
    assert verdict["weights_hash"] == weights_hash(optimized["data"]["weights"])
    assert row.llm_response_hash is not None


def test_failed_risk_check_is_recorded_as_a_failed_verdict(db_session) -> None:
    cs = _constraint_set(universe_filters=UniverseFilters(max_position_cap=0.4))
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    _audit_proposal(ctx, _WEIGHTS)  # AAA = 0.5 > 0.4

    _find(bind_toolset("risk", ctx), "risk_check").invoke({})

    (row,) = _risk_rows(db_session)
    verdict = json.loads(row.llm_response)
    assert verdict["passed"] is False
    assert [v["type"] for v in verdict["violations"]] == ["max_weight"]


def test_place_orders_refuses_when_risk_never_checked_the_proposal(db_session) -> None:
    _seed_panel(db_session)
    ctx = _make_context(db_session, store=_store_with_cs(_constraint_set()))
    _find(bind_toolset("allocator", ctx), "optimize_portfolio").invoke(
        {"universe": _UNIVERSE}
    )
    place = _find(bind_toolset("orchestrator", ctx), "place_orders")

    result = place.invoke({})

    assert result["ok"] is False
    assert "risk" in result["error"]
    assert _executor_rows(db_session) == []  # nothing was placed or audited


def test_place_orders_refuses_after_a_failed_risk_verdict(db_session) -> None:
    _seed_panel(db_session)
    cs = _constraint_set(universe_filters=UniverseFilters(max_position_cap=0.4))
    ctx = _make_context(db_session, store=_store_with_cs(cs))
    _audit_proposal(ctx, _WEIGHTS)
    assert (
        _find(bind_toolset("risk", ctx), "risk_check").invoke({})["data"]["passed"]
        is False
    )
    place = _find(bind_toolset("orchestrator", ctx), "place_orders")

    result = place.invoke({})

    assert result["ok"] is False
    assert _executor_rows(db_session) == []


def test_place_orders_refuses_a_proposal_risk_did_not_check(db_session) -> None:
    """A verdict covers the weights it was computed on: a re-optimised proposal
    that supersedes the checked one needs its own pass."""
    _seed_panel(db_session)
    ctx = _make_context(db_session)
    _audit_proposal(ctx, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    _find(bind_toolset("risk", ctx), "risk_check").invoke({})
    _audit_proposal(ctx, {"AAA": 0.2, "BBB": 0.3, "CCC": 0.5})
    place = _find(bind_toolset("orchestrator", ctx), "place_orders")

    result = place.invoke({})

    assert result["ok"] is False
    assert _executor_rows(db_session) == []


# --- views reach the optimizer? (docs must match the wiring) -----------------
#
# Two layers. The wiring guards fail if views, or any prior other than the
# default, become an input to the allocator's tools by a route the model cannot
# see as well as by one it can: a tool arg, a ``RunContext`` field, a config
# handed to the frozen tools, or run code calling the ViewSet -> Black-Litterman
# bridge. The docs guards fail while an agent-facing surface still presents views
# as an optimizer input. When a wiring guard trips, rewrite the docs with it.

_FUND_SRC = Path(__file__).resolve().parents[2] / "src" / "fund"
_SKILLS_DIR = _FUND_SRC / "skills"
_VIEWS_SKILL_DIR = _SKILLS_DIR / "views-construction"

_VIEW_OR_PRIOR_NAME = re.compile(r"view|prior|black|litterman", re.IGNORECASE)
_VIEW_BRIDGE_NAMES = frozenset({"ViewSet", "BlackLittermanConfig"})

# Phrases that tell an agent the optimizer consumes the economist's views.
_VIEW_INPUT_CLAIM = re.compile(
    # the optimizer folds / prices the views
    r"folds? (?:the )?views|folding in the economist|runs black-litterman"
    r"|price into the prior|turn them into a prior"
    # views listed beside the optimizer's real inputs: "universe, views, ..."
    r"|\b(?:universe|moments)\s*[,+]\s*(?:(?:and\s+)?moments\s*[,+]\s*)?"
    r"(?:and\s+)?(?:qualitative\s+)?views"
    r"|\bviews?\s*[,+]\s*(?:and\s+)?(?:an?\s+)?(?:objective|constraints?)"
    # a regime or view driving the Black-Litterman bridge
    r"|\b(?:drives?|feeds?)\b[^.]*black-litterman",
    re.IGNORECASE,
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _prompt_surfaces() -> dict[str, str]:
    surfaces = {"prompts module docstring": fund_prompts.__doc__ or ""}
    for name in fund_prompts.__all__:
        surfaces[name] = getattr(fund_prompts, name)
    for name in (
        "_ECONOMIST_DESCRIPTION",
        "_ALLOCATOR_DESCRIPTION",
        "_RISK_DESCRIPTION",
        "_EXECUTOR_DESCRIPTION",
    ):
        surfaces[f"graph.{name}"] = getattr(fund_graph, name)
    return surfaces


def _skill_surfaces() -> dict[str, str]:
    return {
        f"skills/{path.relative_to(_SKILLS_DIR).as_posix()}": path.read_text(
            encoding="utf-8"
        )
        for path in sorted(_SKILLS_DIR.rglob("*.md"))
    }


_ALL_SURFACES = {**_prompt_surfaces(), **_skill_surfaces()}

# Surfaces a reader could take as describing how the allocator is fed: each must
# say views are advisory rather than merely omit them.
_MUST_LABEL_VIEWS_ADVISORY = (
    "prompts module docstring",
    "PM_SYSTEM_PROMPT",
    "ECONOMIST_SYSTEM_PROMPT",
    "ALLOCATOR_SYSTEM_PROMPT",
    "graph._ALLOCATOR_DESCRIPTION",
    "skills/fund-orchestration/SKILL.md",
    "skills/optimization-objective-map/SKILL.md",
    "skills/macro-regime-read/reference.md",
    "skills/views-construction/SKILL.md",
)


def test_no_allocator_tool_accepts_views_or_a_prior(db_session) -> None:
    # Model-visible args only; the three tests after it cover the routes the
    # model does not see.
    ctx = _make_context(db_session)
    for tool in bind_toolset("allocator", ctx):
        offending = [a for a in tool.args if _VIEW_OR_PRIOR_NAME.search(a)]
        assert not offending, f"{tool.name} now accepts {offending}"


def test_run_context_carries_no_view_or_prior_field() -> None:
    offending = [
        f.name
        for f in dataclasses.fields(RunContext)
        if _VIEW_OR_PRIOR_NAME.search(f.name)
    ]
    assert not offending, f"RunContext now binds {offending}"


def test_allocator_tools_run_on_the_default_prior(db_session, monkeypatch) -> None:
    # Behavioural: whatever the wiring, the configs that actually reach the
    # moment and optimizer builders must be the defaults, so no view-derived
    # prior can have been swapped in.
    _seed_panel(db_session)
    moment_cfgs: list[MomentEstimationConfig] = []
    optimizer_cfgs: list[MeanRiskConfig] = []
    real_build_prior = fund_moments.build_prior
    real_build_mean_risk = fund_optimize.build_mean_risk

    def spy_prior(cfg):
        moment_cfgs.append(cfg)
        return real_build_prior(cfg)

    def spy_mean_risk(cfg):
        optimizer_cfgs.append(cfg)
        return real_build_mean_risk(cfg)

    monkeypatch.setattr(fund_moments, "build_prior", spy_prior)
    monkeypatch.setattr(fund_optimize, "build_mean_risk", spy_mean_risk)
    ctx = _make_context(db_session, store=_store_with_cs(_constraint_set()))
    tools = bind_toolset("allocator", ctx)

    assert _find(tools, "estimate_moments").invoke({"universe": _UNIVERSE})["ok"]
    assert _find(tools, "optimize_portfolio").invoke({"universe": _UNIVERSE})["ok"]

    assert moment_cfgs == [MomentEstimationConfig()]
    assert [c.prior_config for c in optimizer_cfgs] == [None]


def test_no_run_code_reaches_the_viewset_bridge() -> None:
    # Static: ``ViewSet.to_black_litterman_config`` is defined in ``fund.schemas``
    # and called nowhere else, and run code never imports ``optimizer.views``; a
    # caller anywhere (graph, toolsets, a new tool) trips this.
    offenders: list[str] = []
    for path in sorted(_FUND_SRC.rglob("*.py")):
        rel = path.relative_to(_FUND_SRC)
        if rel.parts[0] == "schemas":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute):
                hit = node.attr == "to_black_litterman_config"
            elif isinstance(node, ast.Name):
                hit = node.id in _VIEW_BRIDGE_NAMES
            elif isinstance(node, ast.ImportFrom):
                hit = (node.module or "").startswith("optimizer.views") or any(
                    a.name in _VIEW_BRIDGE_NAMES for a in node.names
                )
            elif isinstance(node, ast.Import):
                hit = any(a.name.startswith("optimizer.views") for a in node.names)
            else:
                hit = False
            if hit:
                offenders.append(f"{rel.as_posix()}:{node.lineno}")
    assert not offenders, f"run code now reaches the views bridge: {offenders}"


@pytest.mark.parametrize("surface", sorted(_ALL_SURFACES))
def test_agent_facing_surface_does_not_present_views_as_optimizer_input(
    surface: str,
) -> None:
    match = _VIEW_INPUT_CLAIM.search(_flat(_ALL_SURFACES[surface]))
    assert match is None, f"{surface} presents views as an input: {match[0]!r}"


@pytest.mark.parametrize("surface", _MUST_LABEL_VIEWS_ADVISORY)
def test_surface_that_mentions_views_labels_them_advisory(surface: str) -> None:
    text = _flat(_ALL_SURFACES[surface]).lower()
    assert re.search(r"\bviews?\b", text), f"{surface} no longer mentions views"
    assert "advisory" in text, f"{surface} mentions views without saying advisory"


def test_views_skill_names_the_tools_it_says_ignore_views() -> None:
    # The skill's claim is only as good as the wiring guards above, so it must
    # name the tools those guards actually cover.
    body = _flat((_VIEWS_SKILL_DIR / "SKILL.md").read_text(encoding="utf-8"))
    for name in ("estimate_moments", "optimize_portfolio"):
        assert name in TOOLS_BY_AGENT["allocator"]
        assert name in body, f"views-construction/SKILL.md no longer names {name}"


def test_views_reference_documents_the_equal_weight_prior_and_factor_gap() -> None:
    ref = _flat((_VIEWS_SKILL_DIR / "reference.md").read_text(encoding="utf-8"))
    lower = ref.lower()
    assert "market_weights" in lower and "equal" in lower
    assert "not wired" in lower
    assert "TimeSeriesFactorModel" in ref and "not built" in lower
