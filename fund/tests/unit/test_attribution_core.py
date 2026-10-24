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


def test_attribution_core_ridge_recovers_betas_on_return_scale_factors():
    """RIDGE on ~1e-2 scale factors keeps the exposures instead of shrinking to zero."""
    truth = {"rates": 0.8, "credit": -0.3, "equity": 0.9}
    nav, factors = _synthetic(n=250, betas=truth, alpha=5e-4)

    ridge = estimate_exposures(
        nav,
        factors,
        config=MacroAttributionConfig.for_default(method=RegressionMethod.RIDGE),
    )
    ols = estimate_exposures(
        nav,
        factors,
        config=MacroAttributionConfig.for_default(method=RegressionMethod.OLS),
    )

    assert ridge.method_used is RegressionMethod.RIDGE
    for name, beta in truth.items():
        assert ridge.exposures[name] == pytest.approx(beta, abs=0.05)
        assert ridge.exposures[name] == pytest.approx(ols.exposures[name], abs=0.02)
    assert ridge.r2 is not None and ridge.r2 > 0.95


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


@pytest.mark.parametrize("method", list(RegressionMethod))
@pytest.mark.parametrize("level", [0.0, 1e-4])
def test_attribution_flat_nav_degrades_instead_of_raising(method, level):
    """A NAV that never moves (zero-variance returns) degrades, never raises.

    A constant non-zero return is as unidentifiable as a zero one; it must not fit
    near-zero betas under any regressor.
    """
    _, factors = _synthetic(n=40, betas={"rates": 0.5, "equity": 0.9})
    flat_nav = pd.Series(level, index=factors.index, name="nav")

    result = estimate_exposures(
        flat_nav, factors, config=MacroAttributionConfig.for_default(method=method)
    )

    assert result.exposures == {}
    assert result.insufficient_history
    assert result.r2 is None
    assert result.n_obs == 40


def test_attribution_unrelated_value_error_from_the_fit_propagates(monkeypatch):
    """Only a flat NAV is degraded; any other ValueError is a defect and surfaces.

    skfolio/sklearn raise ValueError for shape and alignment bugs too, so swallowing
    it would silently turn a broken fit into a "no data" attribution.
    """
    nav, factors = _synthetic(n=40, betas={"rates": 0.5, "equity": 0.9})

    def _broken_fit(*args, **kwargs):
        raise ValueError("operands could not be broadcast together")

    monkeypatch.setattr("optimizer.factors.fit_factor_prior", _broken_fit)

    with pytest.raises(ValueError, match="broadcast"):
        estimate_exposures(nav, factors, config=MacroAttributionConfig.for_default())


@pytest.fixture
def fit_calls(monkeypatch) -> list[int]:
    """Count real ``fit_factor_prior`` runs without changing what they return."""
    import optimizer.factors as factors_mod

    real = factors_mod.fit_factor_prior
    calls: list[int] = []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(factors_mod, "fit_factor_prior", counting)
    return calls


def test_estimate_exposures_identical_inputs_fit_once(fit_calls):
    """The cockpit re-reads the brief every poll; an unchanged series is not refit."""
    nav, factors = _synthetic(n=60, betas={"rates": 0.5, "equity": 0.9})
    config = MacroAttributionConfig.for_default()

    first = estimate_exposures(nav, factors, config=config)
    second = estimate_exposures(nav.copy(), factors.copy(), config=config)

    assert len(fit_calls) == 1
    assert second == first


def test_exposure_results_with_equal_but_distinct_indexes_compare_equal():
    """``==`` must not compare the estimation index element-wise: the cache hands
    out copies, so two equal estimates rarely share the same index object."""
    idx = pd.date_range("2025-01-01", periods=5, freq="B")
    fields = {
        "exposures": {"rates": 0.5},
        "alpha": 0.0,
        "n_obs": 5,
        "r2": 0.9,
        "method_used": RegressionMethod.OLS,
        "insufficient_history": False,
    }

    assert ExposureResult(**fields, estimation_index=idx) == ExposureResult(
        **fields, estimation_index=idx.copy()
    )


def test_estimate_exposures_cache_hands_out_independent_copies(fit_calls):
    """A caller mutating its result must not corrupt what the next poll receives."""
    nav, factors = _synthetic(n=60, betas={"rates": 0.5, "equity": 0.9})
    config = MacroAttributionConfig.for_default()

    first = estimate_exposures(nav, factors, config=config)
    pristine = dict(first.exposures)
    first.exposures["rates"] = 99.0
    second = estimate_exposures(nav, factors, config=config)

    assert second.exposures == pristine


def test_estimate_exposures_refits_when_the_series_changes(fit_calls):
    """A new bar, a revised return or another regressor each invalidate the memo."""
    nav, factors = _synthetic(n=60, betas={"rates": 0.5, "equity": 0.9})
    config = MacroAttributionConfig.for_default()
    estimate_exposures(nav, factors, config=config)

    estimate_exposures(nav.iloc[:-1], factors.iloc[:-1], config=config)
    revised = nav.copy()
    revised.iloc[10] += 1e-3
    estimate_exposures(revised, factors, config=config)
    estimate_exposures(
        nav,
        factors,
        config=MacroAttributionConfig.for_default(method=RegressionMethod.OLS),
    )

    assert len(fit_calls) == 4


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


def test_attribute_window_bars_explains_only_the_trailing_bars():
    """window_bars sums over the last bars with betas still fitted on the history."""
    nav, factors = _synthetic(n=60, betas={"rates": 0.5, "usd": -0.2})
    exposure = estimate_exposures(
        nav, factors, config=MacroAttributionConfig.for_default()
    )

    result = attribute(nav, factors, exposure, window_bars=1)

    assert result.total_return == pytest.approx(float(nav.iloc[-1]), abs=1e-12)
    for name, beta in exposure.exposures.items():
        assert result.contributions[name] == pytest.approx(
            beta * float(factors[name].iloc[-1]), abs=1e-12
        )
    assert sum(result.contributions.values()) + result.residual == pytest.approx(
        result.total_return, abs=1e-12
    )


def test_attribute_window_bars_stale_factor_contributes_nothing():
    """A factor with no value on the window bar keeps its share in the residual."""
    nav, factors = _synthetic(n=60, betas={"rates": 0.5, "usd": -0.2})
    exposure = estimate_exposures(
        nav, factors, config=MacroAttributionConfig.for_default()
    )
    stale = factors.copy()
    stale.loc[stale.index[-1], "rates"] = np.nan

    result = attribute(nav, stale, exposure, window_bars=1)

    assert result.contributions["rates"] == 0.0
    assert result.total_return == pytest.approx(float(nav.iloc[-1]), abs=1e-12)
    assert sum(result.contributions.values()) + result.residual == pytest.approx(
        result.total_return, abs=1e-12
    )


def test_attribute_window_bars_without_exposures_is_window_total():
    """Degraded exposures still report the window's NAV return, all as residual."""
    nav, _ = _synthetic(n=30, betas={"rates": 0.5})
    empty_exposure = ExposureResult(
        exposures={},
        alpha=0.0,
        n_obs=0,
        r2=None,
        method_used=RegressionMethod.HUBER,
        insufficient_history=True,
    )

    result = attribute(
        nav, pd.DataFrame(index=nav.index), empty_exposure, window_bars=2
    )

    assert result.total_return == pytest.approx(float(nav.iloc[-2:].sum()), abs=1e-12)
    assert result.residual == pytest.approx(result.total_return, abs=1e-12)


@pytest.mark.parametrize("bars", [0, -3])
def test_attribution_window_bars_must_be_positive(bars):
    """A non-positive window is a programmer error on both the config and the call."""
    with pytest.raises(ValueError):
        MacroAttributionConfig(attribution_bars=bars)
    nav, factors = _synthetic(n=30, betas={"rates": 0.5})
    exposure = estimate_exposures(
        nav, factors, config=MacroAttributionConfig.for_default()
    )
    with pytest.raises(ValueError):
        attribute(nav, factors, exposure, window_bars=bars)


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
