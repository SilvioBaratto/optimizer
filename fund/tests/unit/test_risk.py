"""T3.4 — ``risk_check`` + ``backtest``: the risk agent's blocking-gate primitives.

Pins both tools as pure functions with the ``{ok,data}`` / ``{ok:false,error}``
contract:

* ``risk_check`` validates a weight vector against a ConstraintSet placeholder
  (``min_weights`` / ``max_weights`` / ``budget``): in-norm weights ⇒
  ``passed=True`` with no violations; out-of-norm ⇒ ``passed=False`` with a
  populated ``violations`` list. Pure math on the weights — no DB, no RNG;
* ``backtest`` holds the given weights fixed and evaluates them out-of-sample with
  a walk-forward split (``shuffle=False`` — no future leak, SPEC gotcha) over a
  seeded price panel, returning skfolio ``Portfolio`` metrics. The out-of-sample
  window strictly follows the training block, so temporal order is preserved;
* both degrade bad input to ``{ok: false, error}`` (never raised) and are
  deterministic (same seed/args ⇒ identical output).
"""

from __future__ import annotations

import datetime as dt
import math

from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument

from fund.tools.risk import backtest, risk_check

_START = dt.date(2024, 1, 1)
_N_DAYS = 60
_ASOF = _START + dt.timedelta(days=_N_DAYS - 1)
# Deterministic per-ticker (start, drift, wave) — no RNG, so the whole path is
# reproducible and identical-seed determinism is trivially testable.
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


class TestRiskCheck:
    def test_long_only_fully_invested_passes(self) -> None:
        result = risk_check({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})

        assert result["ok"] is True
        assert result["data"]["passed"] is True
        assert result["data"]["violations"] == []

    def test_negative_weight_flags_min_violation(self) -> None:
        # Sum is still 1.0 (budget ok) so the min-weight breach is isolated.
        result = risk_check({"AAA": 1.2, "BBB": -0.2})

        assert result["ok"] is True
        data = result["data"]
        assert data["passed"] is False
        assert any(
            v["type"] == "min_weight" and v["asset"] == "BBB"
            for v in data["violations"]
        )

    def test_concentration_cap_flags_max_violation(self) -> None:
        result = risk_check({"AAA": 0.7, "BBB": 0.3}, constraints={"max_weights": 0.5})

        data = result["data"]
        assert data["passed"] is False
        assert any(
            v["type"] == "max_weight" and v["asset"] == "AAA"
            for v in data["violations"]
        )

    def test_budget_breach_flags_budget_violation(self) -> None:
        # Weights each in [0, 1] but sum to 0.8 ≠ 1 → only the budget breaches.
        result = risk_check({"AAA": 0.5, "BBB": 0.3})

        data = result["data"]
        assert data["passed"] is False
        assert any(v["type"] == "budget" for v in data["violations"])

    def test_empty_weights_is_error(self) -> None:
        result = risk_check({})

        assert result["ok"] is False
        assert "error" in result

    def test_bad_constraint_value_is_error(self) -> None:
        result = risk_check({"AAA": 1.0}, constraints={"max_weights": "big"})

        assert result["ok"] is False
        assert "error" in result

    def test_deterministic(self) -> None:
        weights = {"AAA": 0.6, "BBB": 0.4}

        assert risk_check(weights) == risk_check(weights)


class TestBacktest:
    def test_reports_metrics_full_sample_default(self, db_session) -> None:
        _seed_panel(db_session)

        result = backtest(db_session, _ASOF, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})

        assert result["ok"] is True
        data = result["data"]
        metrics = data["metrics"]
        assert {"mean", "standard_deviation", "sharpe_ratio", "max_drawdown"} <= set(
            metrics
        )
        # The default walk-forward window is far larger than the seed → no split is
        # possible, so the tool falls back to the full sample.
        assert data["n_folds"] == 0
        assert data["n_observations"] == _N_DAYS - 1

    def test_walk_forward_out_of_sample_respects_temporal_order(
        self, db_session
    ) -> None:
        _seed_panel(db_session)

        result = backtest(
            db_session,
            _ASOF,
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            window={"train_size": 20, "test_size": 10, "purged_size": 0},
        )

        assert result["ok"] is True
        data = result["data"]
        assert data["n_folds"] >= 1
        # Out-of-sample excludes the initial training block: fewer observations, and
        # the evaluated window starts strictly after the sample start (no leak).
        assert data["n_observations"] < _N_DAYS - 1
        assert data["window_start"] > data["sample_start"]

    def test_missing_ticker_flagged_not_raised(self, db_session) -> None:
        _seed_panel(db_session)

        result = backtest(db_session, _ASOF, {"AAA": 0.6, "ZZZ": 0.4})

        assert result["ok"] is True
        assert result["data"]["missing"] == ["ZZZ"]

    def test_empty_weights_is_error(self, db_session) -> None:
        result = backtest(db_session, _ASOF, {})

        assert result["ok"] is False
        assert "error" in result

    def test_no_priced_assets_is_error(self, db_session) -> None:
        result = backtest(db_session, _ASOF, {"ZZZ": 1.0})

        assert result["ok"] is False
        assert "error" in result

    def test_bad_window_is_error(self, db_session) -> None:
        _seed_panel(db_session)

        result = backtest(
            db_session, _ASOF, {"AAA": 0.5, "BBB": 0.5}, window={"test_size": 0}
        )

        assert result["ok"] is False
        assert "error" in result

    def test_deterministic(self, db_session) -> None:
        _seed_panel(db_session)
        window = {"train_size": 20, "test_size": 10, "purged_size": 0}
        weights = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}

        first = backtest(db_session, _ASOF, weights, window=window)
        second = backtest(db_session, _ASOF, weights, window=window)

        assert first == second
