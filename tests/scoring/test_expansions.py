"""Tests for scoring expansions: risk-free rate wiring, annualization,
perf/risk measure scorers, robust IR alignment, and the online measure helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.measures import PerfMeasure, RatioMeasure, RiskMeasure
from skfolio.optimization import EqualWeighted

from optimizer.exceptions import ConfigurationError
from optimizer.optimization import RatioMeasureType, RiskMeasureType
from optimizer.scoring import (
    PerfMeasureType,
    ScorerConfig,
    build_online_measure,
    build_scorer,
)
from optimizer.scoring._factory import _build_ir_scorer

N_ASSETS = 6
N_OBS = 252
TICKERS = [f"A{i:02d}" for i in range(N_ASSETS)]
DATES = pd.date_range("2021-01-04", periods=N_OBS, freq="B")


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    data = rng.normal(loc=0.0008, scale=0.012, size=(N_OBS, N_ASSETS))
    return pd.DataFrame(data, index=DATES, columns=TICKERS)


@pytest.fixture(scope="module")
def fitted(returns: pd.DataFrame) -> EqualWeighted:
    return EqualWeighted().fit(returns)


# ---------------------------------------------------------------------------
# risk_free_rate wiring (previously a dead config field)
# ---------------------------------------------------------------------------


class TestRiskFreeRateWiring:
    def test_positive_rf_lowers_sharpe(
        self, fitted: EqualWeighted, returns: pd.DataFrame
    ) -> None:
        base = build_scorer(ScorerConfig.for_sharpe())(fitted, returns)
        with_rf = build_scorer(ScorerConfig.for_sharpe_with_rf(0.0005))(fitted, returns)
        assert with_rf < base

    def test_zero_rf_matches_bare_measure(
        self, fitted: EqualWeighted, returns: pd.DataFrame
    ) -> None:
        scorer = build_scorer(ScorerConfig.for_sharpe())
        expected = float(fitted.predict(returns).sharpe_ratio)
        assert scorer(fitted, returns) == pytest.approx(expected)

    def test_rf_does_not_mutate_across_calls(
        self, fitted: EqualWeighted, returns: pd.DataFrame
    ) -> None:
        scorer = build_scorer(ScorerConfig.for_sharpe_with_rf(0.0005))
        first = scorer(fitted, returns)
        second = scorer(fitted, returns)
        assert first == pytest.approx(second)


# ---------------------------------------------------------------------------
# annualization_factor wiring
# ---------------------------------------------------------------------------


class TestAnnualizationFactor:
    def test_annualization_changes_annualized_sharpe(
        self, fitted: EqualWeighted, returns: pd.DataFrame
    ) -> None:
        base = build_scorer(
            ScorerConfig(ratio_measure=RatioMeasureType.ANNUALIZED_SHARPE_RATIO)
        )(fitted, returns)
        weekly = build_scorer(
            ScorerConfig(
                ratio_measure=RatioMeasureType.ANNUALIZED_SHARPE_RATIO,
                annualization_factor=52.0,
            )
        )(fitted, returns)
        assert base != pytest.approx(weekly)

    def test_non_positive_annualization_rejected(self) -> None:
        with pytest.raises(ValueError, match="annualization_factor"):
            ScorerConfig(annualization_factor=0.0)

    def test_ir_annualization_factor_used(self) -> None:
        rng = np.random.default_rng(3)
        asset = pd.DataFrame(
            rng.normal(0.001, 0.01, (N_OBS, N_ASSETS)), index=DATES, columns=TICKERS
        )
        bm = pd.Series(np.zeros(N_OBS), index=DATES)
        model = EqualWeighted().fit(asset)
        s252 = _build_ir_scorer(bm, annualization_factor=252)(model, asset)
        s52 = _build_ir_scorer(bm, annualization_factor=52)(model, asset)
        # IR scales with sqrt(factor); more periods/year -> larger magnitude.
        assert abs(s252) > abs(s52)


# ---------------------------------------------------------------------------
# perf / risk measure scorers
# ---------------------------------------------------------------------------


class TestPerfMeasureScorer:
    def test_mean_scorer_matches_portfolio_mean(
        self, fitted: EqualWeighted, returns: pd.DataFrame
    ) -> None:
        scorer = build_scorer(ScorerConfig.for_perf_measure(PerfMeasureType.MEAN))
        assert scorer(fitted, returns) == pytest.approx(fitted.predict(returns).mean)

    def test_perf_measure_preset_clears_ratio(self) -> None:
        cfg = ScorerConfig.for_perf_measure()
        assert cfg.ratio_measure is None
        assert cfg.perf_measure == PerfMeasureType.MEAN


class TestRiskMeasureScorer:
    def test_variance_scorer_is_sign_flipped(
        self, fitted: EqualWeighted, returns: pd.DataFrame
    ) -> None:
        scorer = build_scorer(ScorerConfig.for_risk_measure(RiskMeasureType.VARIANCE))
        variance = fitted.predict(returns).variance
        # risk is minimised: scorer returns -variance (higher is better).
        assert scorer(fitted, returns) == pytest.approx(-variance)

    def test_cvar_risk_measure_builds(
        self, fitted: EqualWeighted, returns: pd.DataFrame
    ) -> None:
        scorer = build_scorer(ScorerConfig.for_risk_measure(RiskMeasureType.CVAR))
        assert scorer(fitted, returns) < 0.0

    def test_all_risk_measures_map(self) -> None:
        for rm in RiskMeasureType:
            scorer = build_scorer(ScorerConfig.for_risk_measure(rm))
            assert callable(scorer)


class TestMeasureExclusivity:
    def test_two_measures_rejected(self) -> None:
        with pytest.raises(ValueError, match="at most one"):
            ScorerConfig(
                ratio_measure=RatioMeasureType.SHARPE_RATIO,
                risk_measure=RiskMeasureType.VARIANCE,
            )

    def test_perf_and_risk_rejected(self) -> None:
        with pytest.raises(ValueError, match="at most one"):
            ScorerConfig(
                ratio_measure=None,
                perf_measure=PerfMeasureType.MEAN,
                risk_measure=RiskMeasureType.CVAR,
            )


# ---------------------------------------------------------------------------
# robust IR alignment
# ---------------------------------------------------------------------------


class TestIRAlignment:
    def test_missing_dates_raise(self, returns: pd.DataFrame) -> None:
        short_bm = pd.Series(np.zeros(N_OBS - 5), index=DATES[:-5])
        scorer = build_scorer(
            ScorerConfig.for_information_ratio(), benchmark_returns=short_bm
        )
        model = EqualWeighted().fit(returns)
        with pytest.raises(ConfigurationError, match="missing"):
            scorer(model, returns)

    def test_nan_benchmark_raises(self, returns: pd.DataFrame) -> None:
        bm = pd.Series(np.zeros(N_OBS), index=DATES)
        bm.iloc[10] = np.nan
        scorer = build_scorer(
            ScorerConfig.for_information_ratio(), benchmark_returns=bm
        )
        model = EqualWeighted().fit(returns)
        with pytest.raises(ConfigurationError, match="missing"):
            scorer(model, returns)

    def test_non_series_benchmark_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="Series"):
            _build_ir_scorer(np.zeros(N_OBS))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# online measure helper
# ---------------------------------------------------------------------------


class TestBuildOnlineMeasure:
    def test_default_is_sharpe(self) -> None:
        assert build_online_measure() is RatioMeasure.SHARPE_RATIO

    def test_ratio_measure(self) -> None:
        cfg = ScorerConfig.for_sortino()
        assert build_online_measure(cfg) is RatioMeasure.SORTINO_RATIO

    def test_perf_measure(self) -> None:
        cfg = ScorerConfig.for_perf_measure(PerfMeasureType.ANNUALIZED_MEAN)
        assert build_online_measure(cfg) is PerfMeasure.ANNUALIZED_MEAN

    def test_risk_measure(self) -> None:
        cfg = ScorerConfig.for_risk_measure(RiskMeasureType.CVAR)
        assert build_online_measure(cfg) is RiskMeasure.CVAR

    def test_information_ratio_rejected(self) -> None:
        cfg = ScorerConfig.for_information_ratio()
        with pytest.raises(ConfigurationError, match="Information Ratio"):
            build_online_measure(cfg)

    def test_custom_rejected(self) -> None:
        cfg = ScorerConfig.for_custom()
        with pytest.raises(ConfigurationError, match="custom"):
            build_online_measure(cfg)
