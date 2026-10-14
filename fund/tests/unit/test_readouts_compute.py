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

from fund.readouts_compute import compute_results


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
