"""``fund.readouts_compute``: the worker-thread recompute service for drill-downs.

Unit-tests the optimizer-backed recompute directly (never through the TUI), over an
isolated in-memory SQLite price panel. The module imports ``optimizer``/``fund.tools``
at its own module top — legitimate here because it lives outside ``fund.tui`` and is
imported only inside the Deliberation worker thread (the import-light guard covers
``fund.tui.app``, not this module).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from _fund_fakes import ASOF, UNIVERSE, make_constraint_set, seed_panel
from portopt_db.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from fund.agents.toolsets import _optimizer_constraints
from fund.readouts_compute import compute_results
from fund.schemas import ConstraintSet
from fund.tools.optimize import optimize_portfolio


@pytest.fixture
def priced_session() -> Iterator[Session]:
    """A SQLite session seeded with the 40-day AAA/BBB/CCC price panel."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = local()
    seed_panel(session)
    session.commit()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_compute_results_returns_frontier_and_backtest_numbers(
    priced_session: Session,
) -> None:
    """Over a fixture panel, the frontier point and backtest carry real scalars."""
    constraint_set: dict[str, Any] = make_constraint_set().model_dump(mode="json")
    weights = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}
    result = compute_results(
        priced_session, ASOF, list(UNIVERSE), constraint_set, weights
    )
    assert result.frontier is not None
    assert result.frontier_error is None
    assert isinstance(result.frontier.sharpe_ratio, float)
    assert result.frontier.n_observations > 0
    assert result.backtest is not None
    assert result.backtest_error is None
    assert isinstance(result.backtest.max_drawdown, float)


def test_compute_results_degrades_on_empty_inputs(priced_session: Session) -> None:
    """An empty universe / empty weights degrade to documented errors, never a crash."""
    constraint_set: dict[str, Any] = make_constraint_set().model_dump(mode="json")
    result = compute_results(priced_session, ASOF, [], constraint_set, {})
    assert result.frontier is None
    assert result.frontier_error is not None
    assert result.backtest is None
    assert result.backtest_error is not None


def test_compute_risk_returns_verdict_and_sector_breakdown() -> None:
    """A clean allocation passes risk_check and its weights roll up per sector."""
    from fund.readouts_compute import compute_risk

    constraint_set: dict[str, Any] = make_constraint_set().model_dump(mode="json")
    weights = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    sectors = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"}
    result = compute_risk(weights, constraint_set, sectors)
    assert result.error is None
    assert result.passed is True
    assert result.violations == ()
    exposures = dict(result.sector_exposures or ())
    assert exposures["Tech"] == pytest.approx(0.8)
    assert exposures["Energy"] == pytest.approx(0.2)


def test_compute_risk_flags_bound_breach() -> None:
    """A weight over the max bound surfaces a max_weight violation (passed False)."""
    from fund.readouts_compute import compute_risk

    constraint_set: dict[str, Any] = make_constraint_set().model_dump(mode="json")
    constraint_set["bounds"] = {"min_weights": 0.0, "max_weights": 0.4, "budget": 1.0}
    result = compute_risk({"AAA": 0.7, "BBB": 0.3}, constraint_set, None)
    assert result.passed is False
    assert any(v.get("type") == "max_weight" for v in result.violations)


def test_compute_risk_missing_sector_mapping_degrades() -> None:
    """A missing sector mapping degrades to unavailable (None), never a crash."""
    from fund.readouts_compute import compute_risk

    constraint_set: dict[str, Any] = make_constraint_set().model_dump(mode="json")
    result = compute_risk({"AAA": 1.0}, constraint_set, None)
    assert result.sector_exposures is None


def test_compute_results_replays_the_allocator_in_the_base_currency(
    priced_session: Session,
) -> None:
    """The TUI recompute converts prices exactly as the agent's allocator did."""
    from portopt_db.models.market_data.yfinance_data import PriceHistory
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    priced_session.query(PriceHistory).update({"price_unit": "USD"})
    days = sorted({row.date for row in priced_session.query(PriceHistory).all()})
    FxRatesRepository(priced_session).upsert_rates(
        [
            {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "date": day,
                "rate": 0.90 * 1.01**i,
            }
            for i, day in enumerate(days)
        ]
    )
    priced_session.flush()
    constraint_set: dict[str, Any] = make_constraint_set().model_dump(mode="json")
    weights = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}

    local = compute_results(
        priced_session, ASOF, list(UNIVERSE), constraint_set, weights
    )
    in_eur = compute_results(
        priced_session,
        ASOF,
        list(UNIVERSE),
        constraint_set,
        weights,
        base_currency="EUR",
    )

    assert local.backtest is not None
    assert in_eur.backtest is not None
    assert in_eur.backtest.mean - local.backtest.mean == pytest.approx(0.01, abs=2e-3)


def _capped(cap: float) -> dict[str, Any]:
    """The default constraint set re-profiled with a K&E per-position cap."""
    constraint_set = make_constraint_set().model_dump(mode="json")
    constraint_set["universe_filters"] = {"max_position_cap": cap}
    return constraint_set


def test_compute_risk_flags_a_weight_above_the_position_cap() -> None:
    """The live gate tightens max_weights to the K&E cap; the read-out must agree."""
    from fund.readouts_compute import compute_risk

    result = compute_risk({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, _capped(0.25), None)

    assert result.passed is False
    over_cap = {v["asset"] for v in result.violations if v.get("type") == "max_weight"}
    assert over_cap == {"AAA", "BBB"}


def test_compute_results_frontier_replays_the_recorded_optimizer_config(
    priced_session: Session,
) -> None:
    """The frontier point is the solve that produced the recorded weights: the mapped
    objective, risk measure and cap, not a bounds-only min-variance book."""
    constraint_set = _capped(0.4)
    cs = ConstraintSet.model_validate(constraint_set)
    expected = optimize_portfolio(
        priced_session, ASOF, list(UNIVERSE), constraints=_optimizer_constraints(cs)
    )
    bounds_only = optimize_portfolio(
        priced_session,
        ASOF,
        list(UNIVERSE),
        constraints={
            "min_weights": cs.bounds.min_weights,
            "max_weights": cs.bounds.max_weights,
            "budget": cs.bounds.budget,
        },
    )
    assert expected["data"]["metrics"] != bounds_only["data"]["metrics"]

    result = compute_results(
        priced_session,
        ASOF,
        list(UNIVERSE),
        constraint_set,
        expected["data"]["weights"],
    )

    assert result.frontier is not None
    assert result.frontier.sharpe_ratio == pytest.approx(
        expected["data"]["metrics"]["sharpe_ratio"]
    )
    assert result.frontier.mean == pytest.approx(expected["data"]["metrics"]["mean"])


def test_unreadable_recorded_constraint_set_degrades_to_an_error(
    priced_session: Session,
) -> None:
    """A recorded dump that no longer validates is reported, never silently replaced
    by default constraints that would show a verdict for a different mandate."""
    from fund.readouts_compute import compute_risk

    broken: dict[str, Any] = {"bounds": "not-a-mapping"}
    weights = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}

    risk = compute_risk(weights, broken, {"AAA": "Tech"})
    results = compute_results(priced_session, ASOF, list(UNIVERSE), broken, weights)

    assert risk.passed is None
    assert risk.error is not None
    assert risk.sector_exposures is not None
    assert results.frontier is None
    assert results.frontier_error is not None
    assert results.backtest is not None
