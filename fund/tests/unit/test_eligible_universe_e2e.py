"""End-to-end eligible-universe runs through the real graph with a scripted model.

Unit tests per piece (mandate field, resolver, six tools, PM seed) could all pass while
an outside ticker still reached ``place_orders``. These runs drive the real chain: a
mandate allowlist is resolved once per run, bound to the run context, rendered into
the PM seed and enforced by ``list_universe``, ``universe_filter``,
``estimate_moments``, ``optimize_portfolio``, ``risk_check``, ``place_orders`` and the
adviser edit. Only the language model is scripted; the tools, the audit trail, the
approval gate and the paper ticket are the production ones, over in-memory SQLite.

ESG exclusions are exercised on both halves of the allowlist: single stocks by their
stored sector, and funds by the look-through of their latest sector snapshot against
the client's tolerance. The resolver's own boundary cases live in ``test_eligibility``;
the runs here prove a barred fund is also hidden from ``list_universe``, named in the
PM seed and refused by ``optimize_portfolio``, while commodity and government-bond
funds stay eligible.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest
from _fund_fakes import (
    ASOF,
    N_DAYS,
    PORTFOLIO_ID,
    START,
    UNIVERSE,
    ScriptedFundModel,
    _close_on,
    classify_instruments,
    expected_weights,
    make_constraint_set,
    make_energy_exclusion_constraint_set,
    make_mandate,
    seed_panel,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import PaperOrder
from portopt_db.models.market_data.etf_metadata import ETFSectorWeight
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument
from pydantic import Field
from sqlalchemy import select

from fund.actions import apply_action
from fund.agents import graph as graph_mod
from fund.agents.graph import resume_fund, run_fund
from fund.audit import AgentRunRepository, MandateRepository, put_constraint_set
from fund.audit.positions_repository import PositionRepository
from fund.config import settings
from fund.schemas.constraint_set import EsgPolicy
from fund.schemas.enums import GicsSector

# asset_class, first close, phase: distinct series so the covariance is well-posed.
_ETF_SPECS = {
    "EQETF.DE": ("equity", 100.0, 0.7),
    "GOVETF.L": ("fixed_income", 50.0, 1.9),
    "GLDETF.L": ("commodity", 25.0, 3.1),
}
_ETFS = tuple(_ETF_SPECS)
_FI_SUBCLASS = {"GOVETF.L": "government"}


def _seed_etfs(session: Any) -> None:
    """Add an equity, a government-bond and a commodity ETF to the priced panel."""
    for ticker, (asset_class, first_close, phase) in _ETF_SPECS.items():
        exchange = Exchange(name=f"EX-{ticker}")
        session.add(exchange)
        session.flush()
        instrument = Instrument(
            ticker=ticker,
            short_name=ticker,
            name=f"{ticker} fund",
            exchange_id=exchange.id,
            instrument_type="ETF",
            asset_class=asset_class,
            fi_subclass=_FI_SUBCLASS.get(ticker),
            yfinance_ticker=ticker,
        )
        session.add(instrument)
        session.flush()
        for i in range(N_DAYS):
            session.add(
                PriceHistory(
                    instrument_id=instrument.id,
                    date=START + dt.timedelta(days=i),
                    close=_close_on(first_close, phase, i),
                    volume=1000,
                )
            )
    session.flush()


def _seed_snapshot(
    session: Any, ticker: str, weights: dict[str, float], *, age_days: int = 0
) -> None:
    """Store ``ticker``'s sector snapshot ``age_days`` before the run's decision bar."""
    instrument = session.execute(
        select(Instrument).where(Instrument.yfinance_ticker == ticker)
    ).scalar_one()
    for sector, weight in weights.items():
        session.add(
            ETFSectorWeight(
                instrument_id=instrument.id,
                as_of=ASOF - dt.timedelta(days=age_days),
                sector=sector,
                weight=weight,
            )
        )
    session.flush()


def _tool_results(messages: list[BaseMessage]) -> list[tuple[str, dict[str, Any]]]:
    """The allocator's tool results so far, parsed from their ``ToolMessage`` JSON."""
    results = []
    for message in messages:
        if isinstance(message, ToolMessage):
            content = message.content
            payload = json.loads(content) if isinstance(content, str) else content
            results.append((str(message.name), payload))
    return results


def _last(results: list[tuple[str, dict[str, Any]]], tool: str) -> dict[str, Any]:
    """The most recent result of ``tool`` in an allocator transcript."""
    return next(payload for name, payload in reversed(results) if name == tool)


def _list_universe(_results: Any) -> tuple[str, dict[str, Any]]:
    return "list_universe", {}


def _filter_what_was_listed(results: Any) -> tuple[str, dict[str, Any]]:
    listed = [
        row["ticker"] for row in _last(results, "list_universe")["data"]["instruments"]
    ]
    return "universe_filter", {"universe": listed}


def _moments_of_the_survivors(results: Any) -> tuple[str, dict[str, Any]]:
    return "estimate_moments", {"universe": _last(results, "universe_filter")["data"]}


def _optimize_the_survivors(results: Any) -> tuple[str, dict[str, Any]]:
    return "optimize_portfolio", {"universe": _last(results, "universe_filter")["data"]}


def _optimize(*tickers: str) -> Any:
    def _step(_results: Any) -> tuple[str, dict[str, Any]]:
        return "optimize_portfolio", {"universe": list(tickers)}

    return _step


_FULL_ALLOCATION = [
    _list_universe,
    _filter_what_was_listed,
    _moments_of_the_survivors,
    _optimize_the_survivors,
]


class _ScriptedAllocator(ScriptedFundModel):
    """A fund model whose allocator replays tool-call steps built from earlier results.

    ``steps`` are callables over the allocator's parsed tool results so far, each
    returning the next ``(tool, args)``; once they run out the allocator reports.
    ``transcript`` keeps the latest results for the test to assert on.
    """

    steps: list[Any] = Field(default_factory=list)
    transcript: list[Any] = Field(default_factory=list)

    def _allocator_turn(self, messages: list[BaseMessage]) -> AIMessage:
        results = _tool_results(messages)
        object.__setattr__(self, "transcript", results)
        issued = sum(len(m.tool_calls) for m in messages if isinstance(m, AIMessage))
        if issued < len(self.steps):
            name, args = self.steps[issued](results)
            return AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": f"call_step_{issued}"}],
            )
        return AIMessage(content="allocator report: candidate weights ready.")


@pytest.fixture
def seeds(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every PM seed instruction the graph renders, in order."""
    captured: list[str] = []
    original = graph_mod._run_instruction

    def _spy(*args: Any, **kwargs: Any) -> str:
        text = original(*args, **kwargs)
        captured.append(text)
        return text

    monkeypatch.setattr(graph_mod, "_run_instruction", _spy)
    return captured


def _store(constraint_set: Any = None) -> InMemoryStore:
    store = InMemoryStore()
    put_constraint_set(
        store,
        constraint_set or make_constraint_set(),
        store_key=settings.constraint_set_store_key,
    )
    return store


def _allowlist_mandate(*tickers: str) -> Any:
    return make_mandate().model_copy(update={"eligible_tickers": tuple(tickers)})


def _persist(session: Any, mandate: Any) -> None:
    MandateRepository(session).upsert(mandate)
    session.flush()


def _run_to_gate(session: Any, model: ScriptedFundModel, mandate: Any, saver, store):
    return run_fund(
        model,
        mandate,
        portfolio_id=PORTFOLIO_ID,
        asof=ASOF,
        session=session,
        checkpointer=saver,
        store=store,
    )


def _etf_model(*steps: Any) -> _ScriptedAllocator:
    return _ScriptedAllocator(universe=list(_ETFS), steps=list(steps))


def _approve(session: Any, run_id: Any, saver: Any, store: Any, model: Any) -> None:
    resume_fund(
        run_id,
        "approve",
        session=session,
        checkpointer=saver,
        store=store,
        model=model,
    )


def _held(weights: dict[str, float]) -> set[str]:
    return {ticker for ticker, weight in weights.items() if weight > 1e-6}


# --- 1. an allowlist run ------------------------------------------------------


def test_allowlist_run_allocates_only_the_eligible_etfs_and_places_only_them(
    db_session, seeds
) -> None:
    """The allocator lists exactly the three ETFs, optimizes them, risk passes and the
    approved paper ticket holds only allowlisted names; the PM was told the set."""
    seed_panel(db_session)
    _seed_etfs(db_session)
    mandate = _allowlist_mandate(*_ETFS)
    _persist(db_session, mandate)
    saver, store = MemorySaver(), _store()
    model = _etf_model(*_FULL_ALLOCATION)

    run = _run_to_gate(db_session, model, mandate, saver, store)

    listed = _last(model.transcript, "list_universe")["data"]
    assert [row["ticker"] for row in listed["instruments"]] == list(_ETFS)
    assert listed["total"] == 3
    assert {row["ticker"]: row["asset_class"] for row in listed["instruments"]} == {
        ticker: spec[0] for ticker, spec in _ETF_SPECS.items()
    }
    assert {row["instrument_type"] for row in listed["instruments"]} == {"ETF"}
    assert run.weights and set(run.weights) <= set(_ETFS)
    repo = AgentRunRepository(db_session)
    assert repo.risk_gate(run.run_id, run.weights)["status"] == "passed"
    assert "Eligible universe" in seeds[0]
    assert ", ".join(_ETFS) in seeds[0]

    _approve(db_session, run.run_id, saver, store, _etf_model(*_FULL_ALLOCATION))

    (order,) = db_session.query(PaperOrder).all()
    assert {line["ticker"] for line in order.lines} <= set(_ETFS)
    holdings = {
        p.ticker for p in PositionRepository(db_session).get_holdings(PORTFOLIO_ID)
    }
    assert holdings and holdings <= set(_ETFS)


# --- 2. an out-of-list attempt ------------------------------------------------


def test_optimizing_a_stock_outside_the_allowlist_is_refused_and_the_next_call_succeeds(
    db_session, seeds
) -> None:
    """A stock typed into optimize is refused with its reason and audits nothing; the
    allocator's later call over the listed ETFs succeeds."""
    seed_panel(db_session)
    _seed_etfs(db_session)
    mandate = _allowlist_mandate(*_ETFS)
    _persist(db_session, mandate)
    model = _etf_model(_optimize("AAA", _ETFS[0]), *_FULL_ALLOCATION)

    run = _run_to_gate(db_session, model, mandate, MemorySaver(), _store())

    refused = model.transcript[0][1]
    assert refused["ok"] is False
    assert "AAA (not_in_allowlist)" in refused["error"]
    decisions = [
        d
        for d in AgentRunRepository(db_session).get_run(run.run_id).decisions
        if d.agent == "allocator" and d.step == "optimize_portfolio"
    ]
    assert len(decisions) == 1
    assert set(run.weights) <= set(_ETFS)


# --- 3. the adviser edit ------------------------------------------------------


def _edit(env: dict[str, Any], weights: dict[str, float]) -> None:
    apply_action(
        "edit",
        session=env["session"],
        run_id=env["run_id"],
        portfolio_id=PORTFOLIO_ID,
        checkpointer=env["saver"],
        store=env["store"],
        model=_etf_model(*_FULL_ALLOCATION),
        new_value=weights,
    )


def _paused_etf_run(db_session) -> dict[str, Any]:
    seed_panel(db_session)
    _seed_etfs(db_session)
    mandate = _allowlist_mandate(*_ETFS)
    _persist(db_session, mandate)
    saver, store = MemorySaver(), _store()
    run = _run_to_gate(db_session, _etf_model(*_FULL_ALLOCATION), mandate, saver, store)
    return {
        "session": db_session,
        "run_id": run.run_id,
        "weights": run.weights,
        "saver": saver,
        "store": store,
    }


def _assert_still_paused_with_nothing_committed(env: dict[str, Any]) -> None:
    session = env["session"]
    assert AgentRunRepository(session).get_run(env["run_id"]).status == "paused"
    assert session.query(PaperOrder).count() == 0
    assert PositionRepository(session).get_holdings(PORTFOLIO_ID) == []


def test_an_adviser_edit_into_a_stock_outside_the_allowlist_is_rejected(
    db_session,
) -> None:
    """The edit resolves tickers onto the run's allocator universe, so a stock the
    allocator never held cannot be typed in; nothing is committed."""
    env = _paused_etf_run(db_session)

    with pytest.raises(ValueError, match="AAA"):
        _edit(env, {"AAA": 0.5, _ETFS[0]: 0.5})

    _assert_still_paused_with_nothing_committed(env)


def test_an_adviser_edit_into_a_name_the_mandate_no_longer_lists_is_rejected(
    db_session,
) -> None:
    """Removing a held ETF from the allowlist after the proposal makes an edit that
    still holds it a ``not_eligible`` violation; nothing is committed."""
    env = _paused_etf_run(db_session)
    dropped = next(iter(_held(env["weights"])))
    _persist(db_session, _allowlist_mandate(*(t for t in _ETFS if t != dropped)))
    edited = {ticker: 1.0 / len(_ETFS) for ticker in _ETFS}

    with pytest.raises(ValueError, match="not_eligible"):
        _edit(env, edited)

    _assert_still_paused_with_nothing_committed(env)


# --- 4. ESG on single stocks --------------------------------------------------


def test_an_energy_stock_on_the_allowlist_is_barred_everywhere(
    db_session, seeds
) -> None:
    """With an Energy exclusion the allowlisted energy stock is absent from
    ``list_universe``, named in the PM seed with its reason, refused by optimize, and
    never held."""
    seed_panel(db_session)
    classify_instruments(
        db_session, {"AAA": "Energy", "BBB": "Technology", "CCC": "Utilities"}
    )
    mandate = _allowlist_mandate(*UNIVERSE)
    _persist(db_session, mandate)
    store = _store(make_energy_exclusion_constraint_set())
    model = _ScriptedAllocator(
        universe=UNIVERSE, steps=[_optimize("AAA", "BBB"), *_FULL_ALLOCATION]
    )

    run = _run_to_gate(db_session, model, mandate, MemorySaver(), store)

    assert "AAA (esg:energy)" in model.transcript[0][1]["error"]
    listed = _last(model.transcript, "list_universe")["data"]["instruments"]
    assert [row["ticker"] for row in listed] == ["BBB", "CCC"]
    assert "AAA (esg:energy)" in seeds[0]
    assert "AAA" not in run.weights or run.weights["AAA"] < 1e-6


def test_without_exclusions_every_allowlisted_stock_is_listed(db_session) -> None:
    """The same allowlist and sectors with no ESG exclusion bar nothing."""
    seed_panel(db_session)
    classify_instruments(
        db_session, {"AAA": "Energy", "BBB": "Technology", "CCC": "Utilities"}
    )
    mandate = _allowlist_mandate(*UNIVERSE)
    _persist(db_session, mandate)
    model = _ScriptedAllocator(universe=UNIVERSE, steps=list(_FULL_ALLOCATION))

    _run_to_gate(db_session, model, mandate, MemorySaver(), _store())

    listed = _last(model.transcript, "list_universe")["data"]["instruments"]
    assert [row["ticker"] for row in listed] == list(UNIVERSE)


# --- 4b. ESG look-through on funds ----------------------------------------------


def _energy_policy(tolerance: float = 0.05) -> Any:
    return make_constraint_set().model_copy(
        update={
            "esg": EsgPolicy(
                exclusions=(GicsSector.ENERGY,), look_through_tolerance=tolerance
            )
        }
    )


def _run_with_equity_etf_snapshot(
    db_session: Any,
    *,
    energy: float,
    age_days: int = 0,
    constraint_set: Any,
    steps: list[Any],
) -> tuple[_ScriptedAllocator, Any]:
    """Run the three-ETF allowlist with the equity ETF's snapshot ``energy`` heavy.

    The government-bond and commodity ETFs carry no snapshot at all: they pass on their
    asset class, so any listing of them proves that rule rather than stored data.
    """
    seed_panel(db_session)
    _seed_etfs(db_session)
    _seed_snapshot(
        db_session,
        "EQETF.DE",
        {"energy": energy, "technology": 1.0 - energy},
        age_days=age_days,
    )
    mandate = _allowlist_mandate(*_ETFS)
    _persist(db_session, mandate)
    model = _etf_model(*steps)
    run = _run_to_gate(
        db_session, model, mandate, MemorySaver(), _store(constraint_set)
    )
    return model, run


def _listed(model: _ScriptedAllocator) -> list[str]:
    rows = _last(model.transcript, "list_universe")["data"]["instruments"]
    return [row["ticker"] for row in rows]


def test_an_equity_etf_within_the_tolerance_stays_listed(db_session, seeds) -> None:
    """3% energy against a 5% tolerance is allowed; the others pass on asset class."""
    model, run = _run_with_equity_etf_snapshot(
        db_session, energy=0.03, constraint_set=_energy_policy(), steps=_FULL_ALLOCATION
    )

    assert _listed(model) == list(_ETFS)
    assert "(esg" not in seeds[0]
    assert run.weights and set(run.weights) <= set(_ETFS)


def test_an_equity_etf_over_the_tolerance_is_hidden_named_and_refused(
    db_session, seeds
) -> None:
    """8% energy bars the fund with the dominant sector; the bond and commodity funds
    are still listed and the allocator is told why the equity fund is gone."""
    model, run = _run_with_equity_etf_snapshot(
        db_session,
        energy=0.08,
        constraint_set=_energy_policy(),
        steps=[_optimize("EQETF.DE", "GOVETF.L"), *_FULL_ALLOCATION],
    )

    assert "EQETF.DE (esg:energy)" in model.transcript[0][1]["error"]
    assert _listed(model) == ["GOVETF.L", "GLDETF.L"]
    assert "EQETF.DE (esg:energy)" in seeds[0]
    assert run.weights and set(run.weights) <= {"GOVETF.L", "GLDETF.L"}


def test_an_equity_etf_with_a_stale_snapshot_is_barred_as_unclassified(
    db_session, seeds
) -> None:
    """A snapshot older than the max age cannot show the fund is clean, so it fails
    closed even though its energy share would have been within the tolerance."""
    stale = settings.esg_sector_weights_max_age_days + 1
    model, run = _run_with_equity_etf_snapshot(
        db_session,
        energy=0.03,
        age_days=stale,
        constraint_set=_energy_policy(),
        steps=_FULL_ALLOCATION,
    )

    assert _listed(model) == ["GOVETF.L", "GLDETF.L"]
    assert "EQETF.DE (esg_unclassified)" in seeds[0]
    assert "EQETF.DE" not in run.weights or run.weights["EQETF.DE"] < 1e-6


def test_with_a_zero_tolerance_any_excluded_weight_bars_the_fund(db_session) -> None:
    model, _ = _run_with_equity_etf_snapshot(
        db_session,
        energy=0.01,
        constraint_set=_energy_policy(tolerance=0.0),
        steps=[_list_universe],
    )

    assert _listed(model) == ["GOVETF.L", "GLDETF.L"]


def test_without_exclusions_every_allowlisted_etf_is_listed_whatever_its_snapshot(
    db_session, seeds
) -> None:
    """No ESG exclusion means no look-through: a stale, heavy snapshot is moot."""
    model, _ = _run_with_equity_etf_snapshot(
        db_session,
        energy=0.9,
        age_days=400,
        constraint_set=make_constraint_set(),
        steps=[_list_universe],
    )

    assert _listed(model) == list(_ETFS)
    assert "(esg" not in seeds[0]


# --- 5. the mandate changes before the gate ------------------------------------


def test_removing_a_held_ticker_from_the_mandate_before_approval_stops_the_order(
    db_session,
) -> None:
    """The allowlist is read from the current mandate at resume, so an order holding a
    ticker the adviser has since removed is refused and the run ends incomplete."""
    env = _paused_etf_run(db_session)
    dropped = next(iter(_held(env["weights"])))
    _persist(db_session, _allowlist_mandate(*(t for t in _ETFS if t != dropped)))

    _approve(
        db_session,
        env["run_id"],
        env["saver"],
        env["store"],
        _etf_model(*_FULL_ALLOCATION),
    )

    assert AgentRunRepository(db_session).get_run(env["run_id"]).status == "incomplete"
    assert db_session.query(PaperOrder).count() == 0
    assert PositionRepository(db_session).get_holdings(PORTFOLIO_ID) == []


# --- 6. unrestricted regression -------------------------------------------------


def test_an_empty_allowlist_runs_exactly_as_before_the_feature(
    db_session, seeds
) -> None:
    """No allowlist: the seed carries no eligible block and the optimizer's weights
    are the ones the plain scripted run produces."""
    seed_panel(db_session)
    _seed_etfs(db_session)
    expected = expected_weights(db_session)
    model = ScriptedFundModel(universe=UNIVERSE, weights=expected)

    run = _run_to_gate(db_session, model, make_mandate(), MemorySaver(), _store())

    assert run.weights == pytest.approx(expected)
    assert "Eligible universe" not in seeds[0]
    assert seeds[0].startswith("Produce an allocation")


def test_an_empty_allowlist_lists_the_whole_database_universe(db_session) -> None:
    """Without an allowlist ``list_universe`` still pages the market-cap ranking, now
    with the instrument type on every row."""
    seed_panel(db_session)
    _seed_etfs(db_session)
    model = _ScriptedAllocator(universe=UNIVERSE, steps=[_list_universe])

    _run_to_gate(db_session, model, make_mandate(), MemorySaver(), _store())

    listed = _last(model.transcript, "list_universe")["data"]
    assert listed["total"] == 3 + len(_ETFS)
    assert {row["instrument_type"] for row in listed["instruments"]} == {
        "EQUITY",
        "ETF",
    }
