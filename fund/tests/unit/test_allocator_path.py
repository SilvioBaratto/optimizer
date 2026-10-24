"""``estimate_moments`` + ``optimize_portfolio`` ★ walking skeleton.

These tests pin the allocator critical path as a chain of pure functions over a
seeded SQLite price panel:

* ``estimate_moments`` wraps ``optimizer.moments`` (Ledoit-Wolf **covariance**
  estimator by default) and returns a JSON-serialisable ``{mu, cov}``;
* ``optimize_portfolio`` lets **skfolio** compute the weights — long-only, Σw=1
  — never the caller;
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

import pytest
from optimizer.moments import CovEstimatorType, MomentEstimationConfig
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


def _seed_panel(db_session, n_days: int = _N_DAYS) -> None:
    """Seed a complete ``n_days`` x 3 close panel with distinct, drifting series."""
    for ticker, (start, drift, wave) in _SERIES.items():
        inst = _seed_instrument(db_session, ticker)
        for i in range(n_days):
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


def _seed_listing(
    db_session, ticker: str, *, first_day: int, n_bars: int, phase: float = 0.0
) -> None:
    """Seed ``n_bars`` closes for ``ticker`` starting ``first_day`` days in.

    ``phase`` shifts the price wave so several listings do not share one series.
    """
    inst = _seed_instrument(db_session, ticker)
    for i in range(first_day, first_day + n_bars):
        close = 40.0 * (1.0 + 0.0006 * i + 0.025 * math.sin(i / 4.0 + phase))
        db_session.add(
            PriceHistory(
                instrument_id=inst.id,
                date=_START + dt.timedelta(days=i),
                close=round(close, 6),
                volume=1000,
            )
        )
    db_session.flush()


def _seed_single_day(db_session) -> None:
    """Seed one close per ticker — a panel too short to derive returns (<2 obs)."""
    for ticker, params in _SERIES.items():
        inst = _seed_instrument(db_session, ticker)
        db_session.add(
            PriceHistory(
                instrument_id=inst.id, date=_START, close=params[0], volume=1000
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
        # a covariance estimator yields a full N x N matrix, not a 1-D variance.
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

    def test_insufficient_history_is_error(self, db_session) -> None:
        # A single observation can't yield returns → the <2-obs guard fires.
        _seed_single_day(db_session)

        result = estimate_moments(db_session, _ASOF, _UNIVERSE)

        assert result["ok"] is False
        assert "error" in result

    def test_young_listing_is_dropped_instead_of_truncating_the_sample(
        self, db_session
    ) -> None:
        # A 10-bar listing beside 60-bar assets must not shrink everyone's
        # return sample to its own 9 observations.
        _seed_panel(db_session)
        _seed_listing(db_session, "NEW", first_day=_N_DAYS - 10, n_bars=10)

        result = estimate_moments(db_session, _ASOF, [*_UNIVERSE, "NEW"])

        assert result["ok"] is True
        data = result["data"]
        assert data["assets"] == _UNIVERSE
        assert data["dropped"] == ["NEW"]
        assert data["n_observations"] == _N_DAYS - 1
        assert len(data["cov"]) == len(_UNIVERSE)

    def test_single_bar_listing_is_flagged_dropped(self, db_session) -> None:
        # A one-bar listing has no return at all; it must be reported rather than
        # vanish from both ``assets`` and ``dropped``.
        _seed_panel(db_session)
        _seed_listing(db_session, "ONE", first_day=_N_DAYS - 1, n_bars=1)

        result = estimate_moments(db_session, _ASOF, [*_UNIVERSE, "ONE"])

        assert result["ok"] is True
        assert result["data"]["assets"] == _UNIVERSE
        assert result["data"]["dropped"] == ["ONE"]

    def test_listing_past_the_absolute_floor_is_dropped_when_shallow_for_the_panel(
        self, db_session
    ) -> None:
        # 25 bars clears a flat one-month floor, yet beside 60-bar assets it would
        # still cut everyone's sample to 24 rows.
        _seed_panel(db_session)
        _seed_listing(db_session, "NEW", first_day=_N_DAYS - 25, n_bars=25)

        result = estimate_moments(db_session, _ASOF, [*_UNIVERSE, "NEW"])

        assert result["ok"] is True
        data = result["data"]
        assert data["assets"] == _UNIVERSE
        assert data["dropped"] == ["NEW"]
        assert data["n_observations"] == _N_DAYS - 1

    def test_listing_is_judged_against_the_panel_depth_not_a_fixed_count(
        self, db_session
    ) -> None:
        # 40 bars is past any flat floor, but next to 300-bar assets keeping it would
        # truncate their sample from 299 return rows to 39.
        deep_days = 300
        _seed_panel(db_session, n_days=deep_days)
        _seed_listing(db_session, "NEW", first_day=deep_days - 40, n_bars=40)
        asof = _START + dt.timedelta(days=deep_days - 1)

        result = estimate_moments(db_session, asof, [*_UNIVERSE, "NEW"])

        assert result["ok"] is True
        data = result["data"]
        assert data["assets"] == _UNIVERSE
        assert data["dropped"] == ["NEW"]
        assert data["n_observations"] == deep_days - 1

    def test_modestly_younger_listing_is_kept(self, db_session) -> None:
        # 40 of 60 bars costs the panel a third of its sample, not most of it, so
        # the asset stays in.
        _seed_panel(db_session)
        _seed_listing(db_session, "NEW", first_day=_N_DAYS - 40, n_bars=40)

        result = estimate_moments(db_session, _ASOF, [*_UNIVERSE, "NEW"])

        assert result["ok"] is True
        data = result["data"]
        assert data["assets"] == [*_UNIVERSE, "NEW"]
        assert data["dropped"] == []
        assert data["n_observations"] == 39

    def test_universe_wider_than_the_sample_supports_is_error(self, db_session) -> None:
        # 14 assets with 25 bars each: no asset is young next to the others, but 24
        # return rows cannot condition a 14 x 14 covariance (need 2 per asset).
        tickers = [f"T{i:02d}" for i in range(14)]
        for i, ticker in enumerate(tickers):
            _seed_listing(db_session, ticker, first_day=0, n_bars=25, phase=i)
        asof = _START + dt.timedelta(days=24)

        result = estimate_moments(db_session, asof, tickers)

        assert result["ok"] is False
        assert "insufficient" in result["error"]
        assert "28" in result["error"]

    def test_sample_below_the_observation_floor_is_error(self, db_session) -> None:
        # Every asset has only 10 bars: after dropping the short ones nothing is
        # left to estimate moments from.
        for ticker in _UNIVERSE:
            _seed_listing(db_session, ticker, first_day=_N_DAYS - 10, n_bars=10)

        result = estimate_moments(db_session, _ASOF, _UNIVERSE)

        assert result["ok"] is False
        assert "insufficient" in result["error"]

    def test_custom_config_is_honored(self, db_session) -> None:
        # Passing an explicit config exercises the `config is not None` branch and
        # swaps the covariance estimator away from the Ledoit-Wolf default.
        _seed_panel(db_session)
        config = MomentEstimationConfig(cov_estimator=CovEstimatorType.EMPIRICAL)

        result = estimate_moments(db_session, _ASOF, _UNIVERSE, config=config)

        assert result["ok"] is True
        assert result["data"]["cov_estimator"] == "empirical"

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
        assert all(w >= -1e-9 for w in weights.values())  # long-only
        assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6)

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

    def test_insufficient_history_is_error(self, db_session) -> None:
        _seed_single_day(db_session)

        result = optimize_portfolio(db_session, _ASOF, _UNIVERSE)

        assert result["ok"] is False
        assert "error" in result

    def test_min_weight_floor_constraint_is_applied(self, db_session) -> None:
        # A min-weight floor forbids the optimiser from zeroing any asset out; if
        # the override were dropped (default min=0) min-variance could do exactly
        # that, so this pins that recognized constraint keys reach the config.
        _seed_panel(db_session)

        result = optimize_portfolio(
            db_session, _ASOF, _UNIVERSE, constraints={"min_weights": 0.1}
        )

        assert result["ok"] is True
        weights = result["data"]["weights"]
        assert all(w >= 0.1 - 1e-9 for w in weights.values())

    def test_max_weight_cap_constraint_is_applied(self, db_session) -> None:
        _seed_panel(db_session)

        result = optimize_portfolio(
            db_session, _ASOF, _UNIVERSE, constraints={"max_weights": 0.4}
        )

        assert result["ok"] is True
        weights = result["data"]["weights"]
        assert all(w <= 0.4 + 1e-6 for w in weights.values())

    def test_objective_and_risk_measure_strings_reach_the_optimizer(
        self, db_session
    ) -> None:
        # A string value is coerced to the optimizer enum; a dropped key would
        # leave the minimum-variance default and reproduce its weights.
        _seed_panel(db_session)

        default = optimize_portfolio(db_session, _ASOF, _UNIVERSE)["data"]["weights"]
        result = optimize_portfolio(
            db_session,
            _ASOF,
            _UNIVERSE,
            constraints={"objective": "maximize_ratio", "risk_measure": "cvar"},
        )

        assert result["ok"] is True
        weights = result["data"]["weights"]
        assert max(abs(weights[t] - default[t]) for t in _UNIVERSE) > 1e-3

    def test_cap_too_tight_for_the_universe_names_the_minimum_assets(
        self, db_session
    ) -> None:
        _seed_panel(db_session)

        result = optimize_portfolio(
            db_session, _ASOF, _UNIVERSE, constraints={"max_weights": 0.2}
        )

        assert result["ok"] is False
        assert "at least 5 assets" in result["error"]

    def test_unknown_constraint_keys_are_ignored(self, db_session) -> None:
        # Unrecognized keys are filtered out, so the result matches the default.
        _seed_panel(db_session)

        default = optimize_portfolio(db_session, _ASOF, _UNIVERSE)["data"]["weights"]
        result = optimize_portfolio(
            db_session, _ASOF, _UNIVERSE, constraints={"bogus_key": 123}
        )

        assert result["ok"] is True
        assert result["data"]["weights"] == default

    def test_missing_ticker_reported_with_assets_and_n_observations(
        self, db_session
    ) -> None:
        _seed_panel(db_session)

        result = optimize_portfolio(db_session, _ASOF, ["AAA", "BBB", "ZZZ"])

        assert result["ok"] is True
        data = result["data"]
        assert data["missing"] == ["ZZZ"]
        assert data["assets"] == ["AAA", "BBB"]
        assert set(data["weights"]) == {"AAA", "BBB"}
        assert data["n_observations"] == _N_DAYS - 1

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

        reread = repo.get_run(run.id)
        assert reread is not None
        assert reread.weights == weights

    def test_skeleton_weights_are_reproducible(self, db_session) -> None:
        _seed_panel(db_session)

        first = optimize_portfolio(db_session, _ASOF, _UNIVERSE)["data"]["weights"]
        second = optimize_portfolio(db_session, _ASOF, _UNIVERSE)["data"]["weights"]

        assert first == second


def _synthetic_returns(seed: int = 0):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = 8
    mu = rng.uniform(2e-4, 1.2e-3, n)
    loadings = rng.normal(size=(n, n)) * 0.004
    cov = loadings @ loadings.T + np.diag(rng.uniform(1e-4, 3e-4, n))
    return pd.DataFrame(
        rng.multivariate_normal(mu, cov, size=750),
        columns=[f"T{i}" for i in range(n)],
    )


def _utility_config(measure, risk_aversion):
    from dataclasses import replace

    from optimizer.optimization import MeanRiskConfig, ObjectiveFunctionType

    return replace(
        MeanRiskConfig(),
        objective=ObjectiveFunctionType.MAXIMIZE_UTILITY,
        risk_measure=measure,
        risk_aversion=risk_aversion,
    )


class TestRiskAversionCalibration:
    """``a_gamma`` is quoted for variance utility; other measures need rescaling."""

    @pytest.mark.parametrize("measure", ["cvar", "cdar", "max_drawdown"])
    def test_equal_weight_pays_the_same_penalty_as_under_variance(
        self, measure: str
    ) -> None:
        import numpy as np
        from optimizer.optimization import RiskMeasureType
        from skfolio import Portfolio

        from fund.tools.optimize import _calibrate_risk_aversion

        returns = _synthetic_returns()
        config = _utility_config(RiskMeasureType(measure), 5.0)

        calibrated = _calibrate_risk_aversion(config, returns)

        reference = Portfolio(X=returns, weights=np.full(returns.shape[1], 1 / 8))
        assert calibrated.risk_aversion * getattr(reference, measure) == pytest.approx(
            5.0 * reference.variance
        )
        assert calibrated.risk_aversion < 0.1

    @pytest.mark.parametrize("measure", ["variance", "semi_variance"])
    def test_second_moment_measures_keep_the_quoted_scale(self, measure: str) -> None:
        from optimizer.optimization import RiskMeasureType

        from fund.tools.optimize import _calibrate_risk_aversion

        config = _utility_config(RiskMeasureType(measure), 5.0)

        assert _calibrate_risk_aversion(config, _synthetic_returns()) == config

    def test_objectives_that_ignore_risk_aversion_are_untouched(self) -> None:
        from dataclasses import replace

        from optimizer.optimization import MeanRiskConfig, RiskMeasureType

        from fund.tools.optimize import _calibrate_risk_aversion

        config = replace(MeanRiskConfig(), risk_measure=RiskMeasureType.CVAR)

        assert _calibrate_risk_aversion(config, _synthetic_returns()) == config

    @pytest.mark.parametrize("measure", ["cvar", "cdar", "max_drawdown"])
    def test_more_aversion_stays_closer_to_the_minimum_risk_portfolio(
        self, measure: str
    ) -> None:
        from dataclasses import replace

        import numpy as np
        from optimizer.optimization import (
            MeanRiskConfig,
            ObjectiveFunctionType,
            RiskMeasureType,
            build_mean_risk,
        )

        from fund.tools.optimize import _calibrate_risk_aversion

        returns = _synthetic_returns()
        risk_measure = RiskMeasureType(measure)

        def weights(config: MeanRiskConfig) -> np.ndarray:
            model = build_mean_risk(_calibrate_risk_aversion(config, returns))
            model.fit(returns)
            return np.asarray(model.weights_)

        protection = weights(
            replace(
                MeanRiskConfig(),
                objective=ObjectiveFunctionType.MINIMIZE_RISK,
                risk_measure=risk_measure,
            )
        )
        distances = [
            float(np.abs(weights(_utility_config(risk_measure, a)) - protection).sum())
            for a in (12.0, 5.0, 1.0)
        ]

        assert 0.05 < distances[0] < distances[1] < distances[2]


def test_optimize_reports_the_risk_aversion_it_ran_with(db_session) -> None:
    from optimizer.optimization import RiskMeasureType

    _seed_panel(db_session)
    config = _utility_config(RiskMeasureType.CVAR, 5.0)

    result = optimize_portfolio(
        db_session, _ASOF, _UNIVERSE, constraints=_as_constraints(config)
    )

    assert 0.0 < result["data"]["risk_aversion"] < config.risk_aversion


def _as_constraints(config) -> dict:
    """A ``MeanRiskConfig`` as the ``constraints`` dict the tool consumes."""
    from fund.tools.optimize import _CONSTRAINT_FIELDS

    return {name: getattr(config, name) for name in _CONSTRAINT_FIELDS}


def _prices_from_returns(returns):
    import pandas as pd

    prices = 100 * (1 + returns).cumprod()
    prices.index = pd.bdate_range("2020-01-01", periods=len(prices))
    return prices


class TestSolverNoiseIsCleaned:
    """The optimizer's own output must clear the deterministic ``risk_check`` gate.

    ``MAXIMIZE_UTILITY`` under a binding cap returns raw weights carrying ~1e-9
    solver noise (a slightly negative weight, or a cap hit exceeded by a hair),
    which a ``1e-9`` weight tolerance reads as a breach.
    """

    # Seeds whose raw skfolio solution breaches a 1e-9 tolerance under the 0.3 cap.
    @pytest.mark.parametrize("seed", [0, 3, 6, 8, 9, 11])
    def test_capped_utility_weights_pass_risk_check(
        self, monkeypatch: pytest.MonkeyPatch, seed: int
    ) -> None:
        from optimizer.optimization import (
            MeanRiskConfig,
            ObjectiveFunctionType,
            RiskMeasureType,
        )

        from fund.tools import optimize as optimize_mod
        from fund.tools.risk import risk_check

        frame = _prices_from_returns(_synthetic_returns(seed))
        monkeypatch.setattr(
            optimize_mod, "load_price_frame", lambda *_a, **_k: (frame, [])
        )
        config = MeanRiskConfig(
            objective=ObjectiveFunctionType.MAXIMIZE_UTILITY,
            risk_measure=RiskMeasureType.VARIANCE,
            max_weights=0.3,
            risk_aversion=1.0,
        )

        result = optimize_portfolio(
            None, _ASOF, list(frame.columns), constraints=_as_constraints(config)
        )

        weights = result["data"]["weights"]
        assert all(0.0 <= w <= 0.3 for w in weights.values())
        assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8)
        verdict = risk_check(
            weights, constraints={"min_weights": 0.0, "max_weights": 0.3, "budget": 1.0}
        )
        assert verdict["data"]["violations"] == []
        assert verdict["data"]["passed"] is True

    def test_reported_metrics_describe_the_cleaned_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from skfolio import Portfolio

        from fund.tools import optimize as optimize_mod

        returns = _synthetic_returns(0)
        frame = _prices_from_returns(returns)
        monkeypatch.setattr(
            optimize_mod, "load_price_frame", lambda *_a, **_k: (frame, [])
        )

        data = optimize_portfolio(None, _ASOF, list(frame.columns))["data"]

        replayed = Portfolio(
            X=optimize_mod.prices_to_returns(frame),
            weights=[data["weights"][a] for a in data["assets"]],
        )
        assert data["metrics"]["standard_deviation"] == pytest.approx(
            replayed.standard_deviation
        )
