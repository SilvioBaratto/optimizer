"""Tests for Information Ratio scorer (issue #21).

Verifies:
- ``RatioMeasureType.INFORMATION_RATIO`` is a valid enum member
- ``build_scorer(ScorerConfig(ratio_measure=INFORMATION_RATIO), benchmark_returns=...)``
  returns a callable
- Scorer returns a higher value for better benchmark-relative performance
- Scorer raises ``ValueError`` when ``benchmark_returns`` is ``None``
- ``ScorerConfig.for_information_ratio()`` preset works

Note on scorer API
------------------
skfolio's ``make_scorer`` produces a ``_PortfolioScorer`` whose ``__call__``
signature is ``(estimator, X)`` — it calls ``estimator.predict(X)`` internally,
then applies the score function to the resulting ``Portfolio``.  Tests that
exercise the scorer directly therefore pass ``(fitted_model, X)``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import EqualWeighted

from optimizer.optimization import RatioMeasureType
from optimizer.scoring import ScorerConfig, build_scorer
from optimizer.scoring._factory import _build_ir_scorer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_ASSETS = 10
N_OBS = 252
TICKERS = [f"A{i:02d}" for i in range(N_ASSETS)]
DATES = pd.date_range("2021-01-04", periods=N_OBS, freq="B")


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0005, scale=0.01, size=(N_OBS, N_ASSETS))
    return pd.DataFrame(data, index=DATES, columns=TICKERS)


@pytest.fixture(scope="module")
def benchmark_returns(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight benchmark."""
    return returns.mean(axis=1).rename("benchmark")


# ---------------------------------------------------------------------------
# TestRatioMeasureTypeIR — enum membership
# ---------------------------------------------------------------------------


class TestRatioMeasureTypeIR:
    def test_information_ratio_is_member(self) -> None:
        assert RatioMeasureType.INFORMATION_RATIO in RatioMeasureType

    def test_information_ratio_value(self) -> None:
        assert RatioMeasureType.INFORMATION_RATIO == "information_ratio"

    def test_information_ratio_is_str(self) -> None:
        assert isinstance(RatioMeasureType.INFORMATION_RATIO, str)


# ---------------------------------------------------------------------------
# TestScorerConfigIR — config & preset
# ---------------------------------------------------------------------------


class TestScorerConfigIR:
    def test_for_information_ratio_preset(self) -> None:
        cfg = ScorerConfig.for_information_ratio()
        assert cfg.ratio_measure == RatioMeasureType.INFORMATION_RATIO

    def test_for_information_ratio_is_frozen(self) -> None:
        cfg = ScorerConfig.for_information_ratio()
        with pytest.raises((AttributeError, TypeError)):
            cfg.ratio_measure = RatioMeasureType.SHARPE_RATIO  # type: ignore[misc]

    def test_for_information_ratio_is_hashable(self) -> None:
        cfg = ScorerConfig.for_information_ratio()
        assert isinstance(hash(cfg), int)


# ---------------------------------------------------------------------------
# TestBuildScorerIR — factory
# ---------------------------------------------------------------------------


class TestBuildScorerIR:
    def test_returns_callable(self, benchmark_returns: pd.Series) -> None:
        cfg = ScorerConfig.for_information_ratio()
        scorer = build_scorer(cfg, benchmark_returns=benchmark_returns)
        assert callable(scorer)

    def test_missing_benchmark_raises_value_error(self) -> None:
        cfg = ScorerConfig.for_information_ratio()
        with pytest.raises(ValueError, match="benchmark_returns is required"):
            build_scorer(cfg)

    def test_missing_benchmark_none_raises_value_error(self) -> None:
        cfg = ScorerConfig.for_information_ratio()
        with pytest.raises(ValueError, match="benchmark_returns is required"):
            build_scorer(cfg, benchmark_returns=None)

    def test_ir_scorer_with_direct_config(
        self, benchmark_returns: pd.Series
    ) -> None:
        cfg = ScorerConfig(ratio_measure=RatioMeasureType.INFORMATION_RATIO)
        scorer = build_scorer(cfg, benchmark_returns=benchmark_returns)
        assert callable(scorer)

    def test_non_ir_measures_ignore_benchmark(
        self, benchmark_returns: pd.Series
    ) -> None:
        """benchmark_returns kwarg is silently ignored for non-IR measures."""
        cfg = ScorerConfig.for_sharpe()
        scorer = build_scorer(cfg, benchmark_returns=benchmark_returns)
        assert callable(scorer)


# ---------------------------------------------------------------------------
# TestIRScoreRanking — IR correctness
# ---------------------------------------------------------------------------


class TestIRScoreRanking:
    def test_positive_ir_for_positive_active_return(self) -> None:
        """Portfolio that consistently beats benchmark → IR > 0."""
        rng = np.random.default_rng(99)
        asset_returns = pd.DataFrame(
            rng.normal(0.001, 0.01, (N_OBS, N_ASSETS)),
            index=DATES,
            columns=TICKERS,
        )
        # Benchmark always 0.0005 below equal-weight portfolio
        bm = asset_returns.mean(axis=1) - 0.0005

        cfg = ScorerConfig.for_information_ratio()
        scorer = build_scorer(cfg, benchmark_returns=bm)

        model = EqualWeighted()
        model.fit(asset_returns)
        score = scorer(model, asset_returns)
        assert score > 0.0

    def test_negative_ir_for_negative_active_return(self) -> None:
        """Portfolio that consistently underperforms benchmark → IR < 0."""
        rng = np.random.default_rng(99)
        asset_returns = pd.DataFrame(
            rng.normal(0.001, 0.01, (N_OBS, N_ASSETS)),
            index=DATES,
            columns=TICKERS,
        )
        # Benchmark always 0.0005 above equal-weight portfolio
        bm = asset_returns.mean(axis=1) + 0.0005

        cfg = ScorerConfig.for_information_ratio()
        scorer = build_scorer(cfg, benchmark_returns=bm)

        model = EqualWeighted()
        model.fit(asset_returns)
        score = scorer(model, asset_returns)
        assert score < 0.0

    def test_larger_active_return_gives_higher_ir(self) -> None:
        """Higher mean active return → higher IR (same tracking error)."""
        rng = np.random.default_rng(7)
        base_data = rng.normal(0.0, 0.01, (N_OBS, N_ASSETS))

        # Dataset A: mean return 0.001 per asset vs zero benchmark
        returns_a = pd.DataFrame(
            base_data + 0.001, index=DATES, columns=TICKERS
        )
        # Dataset B: mean return 0.0005 per asset vs zero benchmark
        returns_b = pd.DataFrame(
            base_data + 0.0005, index=DATES, columns=TICKERS
        )
        bm = pd.Series(np.zeros(N_OBS), index=DATES)

        cfg = ScorerConfig.for_information_ratio()
        scorer = build_scorer(cfg, benchmark_returns=bm)

        model_a = EqualWeighted()
        model_b = EqualWeighted()
        model_a.fit(returns_a)
        model_b.fit(returns_b)

        score_a = scorer(model_a, returns_a)
        score_b = scorer(model_b, returns_b)
        assert score_a > score_b

    def test_lower_tracking_error_gives_higher_ir(self) -> None:
        """Same active return but lower tracking error → higher IR."""
        rng = np.random.default_rng(7)
        bm = pd.Series(np.zeros(N_OBS), index=DATES)

        # Low tracking error: tight dispersion around positive mean
        low_te_data = rng.normal(0.001, 0.002, (N_OBS, N_ASSETS))
        # High tracking error: wide dispersion around same mean
        high_te_data = rng.normal(0.001, 0.02, (N_OBS, N_ASSETS))

        returns_low = pd.DataFrame(low_te_data, index=DATES, columns=TICKERS)
        returns_high = pd.DataFrame(high_te_data, index=DATES, columns=TICKERS)

        cfg = ScorerConfig.for_information_ratio()
        scorer = build_scorer(cfg, benchmark_returns=bm)

        model_low = EqualWeighted()
        model_high = EqualWeighted()
        model_low.fit(returns_low)
        model_high.fit(returns_high)

        score_low = scorer(model_low, returns_low)
        score_high = scorer(model_high, returns_high)
        assert score_low > score_high

    def test_ir_is_finite_for_typical_data(
        self, returns: pd.DataFrame, benchmark_returns: pd.Series
    ) -> None:
        cfg = ScorerConfig.for_information_ratio()
        scorer = build_scorer(cfg, benchmark_returns=benchmark_returns)

        model = EqualWeighted()
        model.fit(returns)
        assert math.isfinite(scorer(model, returns))


# ---------------------------------------------------------------------------
# TestBuildIRScorerEdgeCases — edge cases
# ---------------------------------------------------------------------------


class TestBuildIRScorerEdgeCases:
    def test_zero_tracking_error_returns_zero(self) -> None:
        """When portfolio returns = benchmark exactly (TE=0), IR is 0 not inf."""
        bm_data = np.full(N_OBS, 0.001)
        bm = pd.Series(bm_data, index=DATES)

        # Portfolio = benchmark exactly (equal_weight of identical returns)
        equal_returns = pd.DataFrame(
            np.full((N_OBS, N_ASSETS), 0.001), index=DATES, columns=TICKERS
        )
        scorer = _build_ir_scorer(bm)

        model = EqualWeighted()
        model.fit(equal_returns)
        score = scorer(model, equal_returns)
        assert score == 0.0
