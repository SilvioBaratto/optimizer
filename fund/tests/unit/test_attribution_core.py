"""Phase 06 Task 1 — attribution math core + NAV return accessor (DB-independent).

Synthetic, in-memory frames only (mirroring ``performance.py``'s pure-compute
tests): a known-beta ground truth exercises exposure recovery, the period-return
decomposition's sum-to-total invariant, the Huber→OLS short-history fallback, and
the honest-degradation paths. No database, no agent stack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fund.attribution import (
    AttributionResult,
    ExposureResult,
    MacroAttributionConfig,
    RegressionMethod,
    attribute,
    estimate_exposures,
)
from fund.performance import NavPoint, nav_return_series


def _synthetic(
    *,
    n: int,
    betas: dict[str, float],
    alpha: float = 0.0,
    noise: float = 3e-4,
    seed: int = 0,
) -> tuple[pd.Series, pd.DataFrame]:
    """A nav-return series = ``alpha + Σ β·factor + noise`` over ``n`` business days."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    factors = pd.DataFrame({name: rng.normal(0, 0.01, n) for name in betas}, index=idx)
    beta_vec = np.array([betas[c] for c in factors.columns])
    nav = alpha + factors.to_numpy() @ beta_vec + rng.normal(0, noise, n)
    return pd.Series(nav, index=idx, name="nav"), factors


def test_attribution_core_recovers_known_betas():
    """Huber default recovers the ground-truth loadings to tolerance on clean data."""
    nav, factors = _synthetic(n=80, betas={"rates": 0.8, "credit": -0.3}, alpha=5e-4)
    config = MacroAttributionConfig.for_default()

    result = estimate_exposures(nav, factors, config=config)

    assert result.method_used is RegressionMethod.HUBER
    assert not result.insufficient_history
    assert result.n_obs == 80
    assert result.exposures["rates"] == pytest.approx(0.8, abs=0.02)
    assert result.exposures["credit"] == pytest.approx(-0.3, abs=0.02)
    assert result.alpha == pytest.approx(5e-4, abs=5e-4)
    assert result.r2 is not None and result.r2 > 0.95


def test_attribution_core_contributions_reconstruct_total():
    """contributions + residual == total period return, by construction."""
    nav, factors = _synthetic(n=60, betas={"rates": 0.5, "usd": -0.2})
    config = MacroAttributionConfig.for_default()

    exposure = estimate_exposures(nav, factors, config=config)
    result = attribute(nav, factors, exposure)

    assert isinstance(result, AttributionResult)
    assert set(result.contributions) == {"rates", "usd"}
    assert sum(result.contributions.values()) + result.residual == pytest.approx(
        result.total_return, abs=1e-9
    )
    # Total is the additive sum of simple nav returns over the aligned window.
    assert result.total_return == pytest.approx(float(nav.sum()), abs=1e-9)


@pytest.mark.parametrize("method", [RegressionMethod.OLS, RegressionMethod.RIDGE])
def test_attribution_core_method_selectable(method):
    """OLS / RIDGE are selectable and reported on ample history."""
    nav, factors = _synthetic(n=80, betas={"rates": 0.6, "credit": -0.4})
    config = MacroAttributionConfig.for_default(method=method)

    result = estimate_exposures(nav, factors, config=config)

    assert result.method_used is method
    assert not result.insufficient_history
    assert set(result.exposures) == {"rates", "credit"}


def test_attribution_core_ols_fallback_on_short_history():
    """n_obs < 3·(K+1) forces the closed-form OLS fallback and flags it; no raise."""
    # Four factors with the default floor of twelve: the fallback band spans
    # twelve up to (but not including) fifteen observations, so thirteen lands in it.
    nav, factors = _synthetic(
        n=13, betas={"rates": 0.5, "credit": -0.2, "usd": 0.1, "equity": 0.9}
    )
    config = MacroAttributionConfig.for_default()

    result = estimate_exposures(nav, factors, config=config)

    assert result.method_used is RegressionMethod.OLS
    assert result.insufficient_history
    assert set(result.exposures) == {"rates", "credit", "usd", "equity"}


def test_attribution_core_too_few_obs_returns_empty_not_raise():
    """Below min_observations: empty exposures, flagged insufficient, never raises."""
    nav, factors = _synthetic(
        n=8, betas={"rates": 0.5, "credit": -0.2, "usd": 0.1, "equity": 0.9}
    )
    config = MacroAttributionConfig.for_default()

    result = estimate_exposures(nav, factors, config=config)

    assert result.exposures == {}
    assert result.insufficient_history
    assert result.n_obs == 8


def test_attribution_core_constant_factor_dropped_not_crash():
    """A zero-variance factor (collinear with alpha) is dropped, never crashes."""
    nav, factors = _synthetic(n=60, betas={"rates": 0.5, "credit": -0.2})
    factors["flat"] = 0.001  # constant → zero variance
    config = MacroAttributionConfig.for_default()

    result = estimate_exposures(nav, factors, config=config)

    assert "flat" not in result.exposures
    assert set(result.exposures) == {"rates", "credit"}


def test_attribution_core_infinite_periods_are_dropped_like_nan():
    """A +/-inf NAV or factor period leaves the sample instead of poisoning the fit."""
    nav, factors = _synthetic(n=80, betas={"rates": 0.8, "credit": -0.3}, alpha=5e-4)
    nav.iloc[10] = np.inf
    factors.iloc[25, 1] = -np.inf
    config = MacroAttributionConfig.for_default()

    exposure = estimate_exposures(nav, factors, config=config)
    result = attribute(nav, factors, exposure)

    assert exposure.n_obs == 78
    assert exposure.estimation_index is not None
    assert nav.index[10] not in exposure.estimation_index
    assert factors.index[25] not in exposure.estimation_index
    assert exposure.r2 is not None and np.isfinite(exposure.r2)
    assert exposure.exposures["rates"] == pytest.approx(0.8, abs=0.02)
    assert np.isfinite(result.total_return)
    assert np.isfinite(result.residual)
    assert result.total_return == pytest.approx(
        float(nav.iloc[exposure.estimation_index.map(nav.index.get_loc)].sum()),
        abs=1e-9,
    )


def test_attribution_infinite_period_without_exposures_stays_finite():
    """The no-exposure total skips +/-inf NAV periods as it skips NaN ones."""
    nav, _ = _synthetic(n=30, betas={"rates": 0.5})
    nav.iloc[4] = -np.inf
    empty_exposure = ExposureResult(
        exposures={},
        alpha=0.0,
        n_obs=0,
        r2=None,
        method_used=RegressionMethod.HUBER,
        insufficient_history=True,
    )

    result = attribute(nav, pd.DataFrame(index=nav.index), empty_exposure)

    assert np.isfinite(result.total_return)
    assert result.total_return == pytest.approx(float(nav.drop(nav.index[4]).sum()))
    assert result.residual == pytest.approx(result.total_return)


def test_attribution_all_factors_missing_residual_equals_total():
    """With no exposures the whole period return is idiosyncratic residual."""
    nav, _ = _synthetic(n=30, betas={"rates": 0.5})
    empty_exposure = ExposureResult(
        exposures={},
        alpha=0.0,
        n_obs=0,
        r2=None,
        method_used=RegressionMethod.HUBER,
        insufficient_history=True,
    )

    result = attribute(nav, pd.DataFrame(index=nav.index), empty_exposure)

    assert result.contributions == {}
    assert result.factor_names == ()
    assert result.residual == pytest.approx(result.total_return, abs=1e-12)
    assert result.total_return == pytest.approx(float(nav.sum()), abs=1e-12)


def test_nav_return_series_nav_source_simple_returns():
    """nav_return_series yields date-indexed simple returns off NavPoint.nav."""
    points = [
        NavPoint(date=pd.Timestamp("2026-01-02").date(), nav=100.0, benchmark=200.0),
        NavPoint(date=pd.Timestamp("2026-01-03").date(), nav=110.0, benchmark=210.0),
        NavPoint(date=pd.Timestamp("2026-01-04").date(), nav=99.0, benchmark=None),
    ]
    series = nav_return_series(points, source="nav")

    assert isinstance(series, pd.Series)
    assert series.index.tolist() == [
        pd.Timestamp("2026-01-03"),
        pd.Timestamp("2026-01-04"),
    ]
    assert series.tolist() == pytest.approx([0.10, 99.0 / 110.0 - 1.0])


def test_nav_return_series_benchmark_source_drops_missing():
    """source='benchmark' uses NavPoint.benchmark and skips the None point."""
    points = [
        NavPoint(date=pd.Timestamp("2026-01-02").date(), nav=100.0, benchmark=200.0),
        NavPoint(date=pd.Timestamp("2026-01-03").date(), nav=110.0, benchmark=220.0),
        NavPoint(date=pd.Timestamp("2026-01-04").date(), nav=99.0, benchmark=None),
    ]
    series = nav_return_series(points, source="benchmark")

    assert series.index.tolist() == [pd.Timestamp("2026-01-03")]
    assert series.tolist() == pytest.approx([0.10])


def test_nav_return_series_too_short_is_empty():
    """Fewer than two usable points yields an empty float series, not a raise."""
    points = [
        NavPoint(date=pd.Timestamp("2026-01-02").date(), nav=100.0, benchmark=None)
    ]
    series = nav_return_series(points, source="nav")

    assert series.empty
    assert series.dtype == float


def test_nav_return_series_rejects_unknown_source():
    """An unknown source is a programmer error — raise ValueError."""
    points = [
        NavPoint(date=pd.Timestamp("2026-01-02").date(), nav=100.0, benchmark=None)
    ]
    with pytest.raises(ValueError):
        nav_return_series(points, source="bogus")  # type: ignore[arg-type]
