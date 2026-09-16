"""T3.2 — ``estimate_moments`` + ``optimize_portfolio`` ★ walking skeleton.

These tests pin the allocator critical path as a chain of pure functions over a
seeded SQLite price panel:

* ``estimate_moments`` wraps ``optimizer.moments`` (Ledoit-Wolf **covariance**
  estimator by default, D23) and returns a JSON-serialisable ``{mu, cov}``;
* ``optimize_portfolio`` lets **skfolio** compute the weights — long-only, Σw=1
  (D17) — never the caller;
* both are deterministic: same seeded data ⇒ identical output;
* bad input degrades to ``{ok: false, error}`` (never raised);
* the ★ walking skeleton drives ``get_prices → estimate_moments →
  optimize_portfolio`` and persists an ``agent_runs`` row via
  ``AgentRunRepository`` — proving "the optimizer computes, the audit trail
  records" end to end.
"""

from __future__ import annotations

import datetime as dt
import math

from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument

from fund.audit.repository import AgentRunRepository
from fund.tools.moments import estimate_moments
from fund.tools.optimize import optimize_portfolio
from fund.tools.prices import get_prices

_START = dt.date(2024, 1, 1)
_N_DAYS = 60
_ASOF = _START + dt.timedelta(days=_N_DAYS - 1)
_UNIVERSE = ["AAA", "BBB", "CCC"]
# Deterministic per-ticker starting prices and drift/wave params — no RNG, so the
# whole path is reproducible and identical-seed determinism is trivially testable.
_SERIES = {
    "AAA": (100.0, 0.0008, 0.015),
    "BBB": (50.0, 0.0004, 0.030),
    "CCC": (25.0, 0.0011, 0.020),
}


def _seed_instrument(db_session, ticker: str) -> Instrument:
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
    return inst


def _seed_panel(db_session) -> None:
    """Seed a complete ``_N_DAYS`` x 3 close panel with distinct, drifting series."""
    for ticker, (start, drift, wave) in _SERIES.items():
        inst = _seed_instrument(db_session, ticker)
        for i in range(_N_DAYS):
            close = start * (1.0 + drift * i + wave * math.sin(i / 5.0))
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=_START + dt.timedelta(days=i),
                    close=round(close, 6),
                    volume=1000,
                )
            )
    db_session.flush()


class TestEstimateMoments:
    def test_returns_mu_and_square_covariance(self, db_session) -> None:
        _seed_panel(db_session)

        result = estimate_moments(db_session, _ASOF, _UNIVERSE)

        assert result["ok"] is True
        data = result["data"]
        assert data["assets"] == _UNIVERSE
        assert set(data["mu"]) == set(_UNIVERSE)
        # D23: a covariance estimator yields a full N x N matrix, not a 1-D variance.
        cov = data["cov"]
        assert len(cov) == len(_UNIVERSE)
        assert all(len(row) == len(_UNIVERSE) for row in cov)
        assert data["n_observations"] == _N_DAYS - 1
        assert data["cov_estimator"] == "ledoit_wolf"

    def test_covariance_is_symmetric(self, db_session) -> None:
        _seed_panel(db_session)

        cov = estimate_moments(db_session, _ASOF, _UNIVERSE)["data"]["cov"]

        n = len(cov)
        for i in range(n):
            for j in range(n):
                assert cov[i][j] == cov[j][i]

    def test_empty_universe_is_error(self, db_session) -> None:
        result = estimate_moments(db_session, _ASOF, [])

        assert result["ok"] is False
        assert "error" in result

    def test_no_priced_assets_is_error(self, db_session) -> None:
        result = estimate_moments(db_session, _ASOF, ["ZZZ"])

        assert result["ok"] is False
        assert "error" in result

    def test_missing_ticker_flagged_not_raised(self, db_session) -> None:
        _seed_panel(db_session)

        result = estimate_moments(db_session, _ASOF, ["AAA", "BBB", "ZZZ"])

        assert result["ok"] is True
        assert result["data"]["missing"] == ["ZZZ"]
        assert result["data"]["assets"] == ["AAA", "BBB"]

    def test_deterministic(self, db_session) -> None:
        _seed_panel(db_session)

        first = estimate_moments(db_session, _ASOF, _UNIVERSE)
        second = estimate_moments(db_session, _ASOF, _UNIVERSE)

        assert first == second


class TestOptimizePortfolio:
    def test_weights_are_long_only_and_sum_to_one(self, db_session) -> None:
        _seed_panel(db_session)

        result = optimize_portfolio(db_session, _ASOF, _UNIVERSE)

        assert result["ok"] is True
        weights = result["data"]["weights"]
        assert set(weights) == set(_UNIVERSE)
        assert all(w >= -1e-9 for w in weights.values())  # long-only (D17)
        assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6)  # Σw = 1

    def test_reports_metrics(self, db_session) -> None:
        _seed_panel(db_session)

        metrics = optimize_portfolio(db_session, _ASOF, _UNIVERSE)["data"]["metrics"]

        assert "mean" in metrics
        assert "standard_deviation" in metrics
        assert "sharpe_ratio" in metrics

    def test_empty_universe_is_error(self, db_session) -> None:
        result = optimize_portfolio(db_session, _ASOF, [])

        assert result["ok"] is False
        assert "error" in result

    def test_no_priced_assets_is_error(self, db_session) -> None:
        result = optimize_portfolio(db_session, _ASOF, ["ZZZ"])

        assert result["ok"] is False
        assert "error" in result

    def test_deterministic_identical_seed_identical_weights(self, db_session) -> None:
        _seed_panel(db_session)

        first = optimize_portfolio(db_session, _ASOF, _UNIVERSE)["data"]["weights"]
        second = optimize_portfolio(db_session, _ASOF, _UNIVERSE)["data"]["weights"]

        assert first == second


class TestWalkingSkeleton:
    def test_prices_moments_optimize_persist_agent_run(self, db_session) -> None:
        _seed_panel(db_session)

        # Node 1: prices resolve (summary, not a raw frame).
        prices = get_prices(db_session, _ASOF, _UNIVERSE)
        assert prices["ok"] is True
        assert prices["data"]["missing"] == []

        # Node 2: moments come from the optimizer's covariance estimator.
        moments = estimate_moments(db_session, _ASOF, _UNIVERSE)
        assert moments["ok"] is True

        # Audit: open a pending run, let skfolio compute, stamp the result.
        repo = AgentRunRepository(db_session)
        optimizer_config = {"optimizer": "mean_risk", "objective": "minimize_risk"}
        run = repo.create_run(
            portfolio_id=None,
            asof=_ASOF,
            seed=None,
            universe=_UNIVERSE,
            optimizer_config=optimizer_config,
        )

        # Node 3: the optimizer — not the caller — produces the weights.
        alloc = optimize_portfolio(db_session, _ASOF, _UNIVERSE)
        assert alloc["ok"] is True
        weights = alloc["data"]["weights"]

        finalized = repo.finalize_run(run.id, weights=weights)

        assert finalized is not None
        assert finalized.status == "completed"
        assert finalized.weights == weights
        assert finalized.optimizer_config == optimizer_config
        assert math.isclose(sum(finalized.weights.values()), 1.0, abs_tol=1e-6)

        # The persisted row survives a fresh read.
        reread = repo.get_run(run.id)
        assert reread is not None
        assert reread.weights == weights

    def test_skeleton_weights_are_reproducible(self, db_session) -> None:
        _seed_panel(db_session)

        first = optimize_portfolio(db_session, _ASOF, _UNIVERSE)["data"]["weights"]
        second = optimize_portfolio(db_session, _ASOF, _UNIVERSE)["data"]["weights"]

        assert first == second
