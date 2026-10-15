"""Performance tracker — portfolio metrics (SPEC Phase 05 T8).

Whole-series risk (vol/Sharpe/maxDD) comes from ``skfolio.BasePortfolio`` and is
cross-checked against an independent numpy computation; benchmark-relative
(excess/TE/beta) and trailing-horizon returns are hand-rolled. Every field
degrades to ``None`` on short/empty/benchmark-less history.
"""

from __future__ import annotations

import datetime as dt
import uuid

import numpy as np
import pytest

from fund.performance import NavPoint, portfolio_metrics

_PID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_NAVS = [100.0, 102.0, 101.0, 103.0, 106.0]


def _points(navs, benchmarks=None):
    base = dt.date(2026, 10, 1)
    benchmarks = benchmarks if benchmarks is not None else [None] * len(navs)
    return [
        NavPoint(date=base + dt.timedelta(days=i), nav=nav, benchmark=bench)
        for i, (nav, bench) in enumerate(zip(navs, benchmarks, strict=True))
    ]


def _returns(values):
    arr = np.asarray(values, dtype=float)
    return arr[1:] / arr[:-1] - 1.0


def test_metrics_vol_sharpe_maxdd_match_independent_numpy():
    """vol/Sharpe/maxDD match a numpy hand computation to tolerance."""
    summary = portfolio_metrics(_PID, _points(_NAVS))
    rets = _returns(_NAVS)
    exp_vol = float(np.std(rets, ddof=1) * np.sqrt(252.0))
    exp_sharpe = float((rets.mean() / np.std(rets, ddof=1)) * np.sqrt(252.0))
    cum = np.cumsum(rets)
    exp_maxdd = float(np.max(np.maximum.accumulate(cum) - cum))

    assert summary.volatility == pytest.approx(exp_vol, rel=1e-6)
    assert summary.sharpe == pytest.approx(exp_sharpe, rel=1e-6)
    assert summary.max_drawdown == pytest.approx(exp_maxdd, rel=1e-6)


def test_metrics_trailing_returns():
    """``max`` and ``1D`` returns are the cumulative and last-bar moves."""
    summary = portfolio_metrics(_PID, _points(_NAVS))
    assert summary.returns["max"] == pytest.approx(106.0 / 100.0 - 1.0)
    assert summary.returns["1D"] == pytest.approx(106.0 / 103.0 - 1.0)
    # A 5-day window has no bar a month/year back.
    assert summary.returns["1M"] is None
    assert summary.returns["1Y"] is None


def test_metrics_benchmark_relative_beta_and_excess():
    """beta = cov/var; excess = port − bench over each horizon."""
    benchs = [50.0, 51.0, 50.0, 51.0, 52.0]
    summary = portfolio_metrics(_PID, _points(_NAVS, benchs))

    pr = _returns(_NAVS)
    br = _returns(benchs)
    exp_beta = float(np.cov(pr, br, ddof=1)[0, 1] / np.var(br, ddof=1))
    assert summary.beta == pytest.approx(exp_beta, rel=1e-6)

    exp_excess_max = (106.0 / 100.0 - 1.0) - (52.0 / 50.0 - 1.0)
    assert summary.excess["max"] == pytest.approx(exp_excess_max, rel=1e-6)
    exp_te = float(np.std(pr - br, ddof=1) * np.sqrt(252.0))
    assert summary.tracking_error == pytest.approx(exp_te, rel=1e-6)


def test_metrics_no_benchmark_leaves_bench_fields_none():
    """Without a benchmark, excess/TE/beta are None but risk metrics remain."""
    summary = portfolio_metrics(_PID, _points(_NAVS))
    assert summary.beta is None
    assert summary.tracking_error is None
    assert all(v is None for v in summary.excess.values())
    assert summary.volatility is not None


def test_metrics_short_history_degrades_to_none():
    """A single NAV point yields no returns and all-None metrics (no raise)."""
    summary = portfolio_metrics(_PID, _points([100.0]))
    assert summary.volatility is None
    assert summary.sharpe is None
    assert summary.max_drawdown is None
    assert all(v is None for v in summary.returns.values())


def test_metrics_empty_series_is_all_none():
    """An empty series returns an all-None summary without raising."""
    summary = portfolio_metrics(_PID, [])
    assert summary.portfolio_id == _PID
    assert summary.volatility is None
    assert summary.returns["max"] is None


def test_metrics_partial_returns_survive_mid_series_zero():
    """A mid-series zero NAV ends the return series but keeps the valid prefix.

    A zero denominator makes only *that* pair ill-defined — the returns already
    computed must still feed the risk measures rather than being discarded.
    """
    summary = portfolio_metrics(_PID, _points([100.0, 105.0, 0.0, 110.0]))
    # prefix returns: +5%, -100% → a 100% drawdown is computable, not None.
    assert summary.max_drawdown == pytest.approx(1.0, rel=1e-6)
    assert summary.volatility is not None


def test_metrics_nan_benchmark_leaves_te_and_beta_none():
    """A NaN in the benchmark must degrade tracking error / beta to None, not NaN."""
    nan = float("nan")
    summary = portfolio_metrics(_PID, _points(_NAVS, [50.0, 51.0, nan, 52.0, 53.0]))
    assert summary.tracking_error is None
    assert summary.beta is None
