"""Tests for the analyst-signal -> Black-Litterman view adapter."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from skfolio.prior import BlackLitterman

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.views import (
    AnalystSignal,
    BlackLittermanConfig,
    PriceTargetStatistic,
    ViewUncertaintyMethod,
    build_analyst_bl_views,
    build_black_litterman,
    build_black_litterman_config_from_signals,
    implied_return_from_price_target,
    recommendation_confidence,
)


class TestPriceTargetStatistic:
    def test_members(self) -> None:
        assert {s.value for s in PriceTargetStatistic} == {
            "mean",
            "median",
            "low",
            "high",
        }


class TestAnalystSignal:
    def test_decimal_prices_coerced_to_float(self) -> None:
        sig = AnalystSignal(
            current_price=Decimal("100.5"),
            target_mean=Decimal("110.0"),
        )
        assert isinstance(sig.current_price, float)
        assert isinstance(sig.target_mean, float)
        assert sig.current_price == 100.5
        assert sig.target_mean == 110.0

    def test_none_votes_coerced_to_zero(self) -> None:
        sig = AnalystSignal(current_price=100.0, strong_buy=None)  # type: ignore[arg-type]
        assert sig.strong_buy == 0
        assert sig.total_votes == 0

    def test_total_votes(self) -> None:
        sig = AnalystSignal(
            current_price=100.0,
            strong_buy=3,
            buy=2,
            hold=1,
            sell=1,
            strong_sell=0,
        )
        assert sig.total_votes == 7

    def test_target_selects_statistic(self) -> None:
        sig = AnalystSignal(
            current_price=100.0,
            target_mean=110.0,
            target_median=108.0,
            target_low=90.0,
            target_high=130.0,
        )
        assert sig.target(PriceTargetStatistic.MEAN) == 110.0
        assert sig.target(PriceTargetStatistic.MEDIAN) == 108.0
        assert sig.target(PriceTargetStatistic.LOW) == 90.0
        assert sig.target(PriceTargetStatistic.HIGH) == 130.0

    def test_frozen(self) -> None:
        sig = AnalystSignal(current_price=100.0)
        with pytest.raises(AttributeError):
            sig.current_price = 200.0  # type: ignore[misc]


class TestImpliedReturnFromPriceTarget:
    def test_basic_upside(self) -> None:
        assert implied_return_from_price_target(100.0, 110.0) == pytest.approx(0.10)

    def test_basic_downside(self) -> None:
        assert implied_return_from_price_target(100.0, 90.0) == pytest.approx(-0.10)

    def test_decimal_inputs(self) -> None:
        r = implied_return_from_price_target(Decimal("100"), Decimal("125"))
        assert r == pytest.approx(0.25)

    def test_horizon_compounding_deannualises(self) -> None:
        # 12-month +21% target over 2 periods compounds to ~10% per period.
        r = implied_return_from_price_target(100.0, 121.0, horizon_periods=2.0)
        assert r == pytest.approx(0.10)

    def test_horizon_simple_deannualises(self) -> None:
        r = implied_return_from_price_target(
            100.0, 120.0, horizon_periods=2.0, compounding=False
        )
        assert r == pytest.approx(0.10)

    def test_horizon_252_daily_scale(self) -> None:
        # A large annual target shrinks to a tiny per-day view.
        r = implied_return_from_price_target(100.0, 120.0, horizon_periods=252.0)
        assert 0.0 < r < 0.001

    def test_nonpositive_current_raises(self) -> None:
        with pytest.raises(DataError, match="current_price"):
            implied_return_from_price_target(0.0, 110.0)

    def test_nonpositive_target_raises(self) -> None:
        with pytest.raises(DataError, match="target_price"):
            implied_return_from_price_target(100.0, -5.0)

    def test_nonpositive_horizon_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="horizon_periods"):
            implied_return_from_price_target(100.0, 110.0, horizon_periods=0.0)


class TestRecommendationConfidence:
    def test_unanimous_is_max(self) -> None:
        sig = AnalystSignal(current_price=100.0, strong_buy=10)
        assert recommendation_confidence(sig, cap=1.0) == pytest.approx(1.0)

    def test_max_split_is_zero(self) -> None:
        sig = AnalystSignal(current_price=100.0, strong_buy=5, strong_sell=5)
        assert recommendation_confidence(sig, cap=1.0) == pytest.approx(0.0)

    def test_partial_agreement_between(self) -> None:
        sig = AnalystSignal(current_price=100.0, strong_buy=6, buy=4)
        conf = recommendation_confidence(sig, cap=1.0)
        assert 0.0 < conf < 1.0

    def test_cap_scales_confidence(self) -> None:
        sig = AnalystSignal(current_price=100.0, strong_buy=10)
        assert recommendation_confidence(sig, cap=0.5) == pytest.approx(0.5)

    def test_more_agreement_higher_confidence(self) -> None:
        agree = AnalystSignal(current_price=100.0, strong_buy=9, buy=1)
        split = AnalystSignal(current_price=100.0, strong_buy=5, sell=5)
        assert recommendation_confidence(agree) > recommendation_confidence(split)

    def test_insufficient_votes_raises(self) -> None:
        sig = AnalystSignal(current_price=100.0)
        with pytest.raises(DataError, match="votes"):
            recommendation_confidence(sig, min_votes=1)

    def test_cap_out_of_range_raises(self) -> None:
        sig = AnalystSignal(current_price=100.0, strong_buy=1)
        with pytest.raises(ConfigurationError, match="cap"):
            recommendation_confidence(sig, cap=1.5)


class TestBuildAnalystBlViews:
    def test_basic_views(self) -> None:
        signals = {
            "AAPL": AnalystSignal(current_price=100.0, target_mean=110.0),
            "MSFT": AnalystSignal(current_price=200.0, target_mean=190.0),
        }
        views, confidences = build_analyst_bl_views(signals)
        assert views == ("AAPL == 0.100000", "MSFT == -0.050000")
        assert confidences is None

    def test_precision_formatting_avoids_scientific_notation(self) -> None:
        signals = {"AAPL": AnalystSignal(current_price=100.0, target_mean=120.0)}
        views, _ = build_analyst_bl_views(
            signals, horizon_periods=252.0, precision=8
        )
        # fixed-point, no 'e' exponent
        assert "e" not in views[0]
        assert views[0].startswith("AAPL == 0.000")

    def test_statistic_selection(self) -> None:
        signals = {
            "AAPL": AnalystSignal(
                current_price=100.0, target_mean=110.0, target_high=150.0
            )
        }
        views, _ = build_analyst_bl_views(
            signals, statistic=PriceTargetStatistic.HIGH
        )
        assert views == ("AAPL == 0.500000",)

    def test_skip_incomplete_skips_missing_target(self) -> None:
        signals = {
            "AAPL": AnalystSignal(current_price=100.0, target_mean=110.0),
            "MSFT": AnalystSignal(current_price=200.0),  # no target
        }
        views, _ = build_analyst_bl_views(signals)
        assert views == ("AAPL == 0.100000",)

    def test_skip_incomplete_false_raises(self) -> None:
        signals = {"MSFT": AnalystSignal(current_price=200.0)}
        with pytest.raises(DataError, match="target"):
            build_analyst_bl_views(signals, skip_incomplete=False)

    def test_empty_result_raises(self) -> None:
        signals = {"MSFT": AnalystSignal(current_price=200.0)}
        with pytest.raises(DataError, match="no usable analyst views"):
            build_analyst_bl_views(signals)

    def test_with_confidence_aligned(self) -> None:
        signals = {
            "AAPL": AnalystSignal(
                current_price=100.0, target_mean=110.0, strong_buy=8, buy=2
            ),
            "MSFT": AnalystSignal(
                current_price=200.0, target_mean=210.0, strong_buy=5, strong_sell=5
            ),
        }
        views, confidences = build_analyst_bl_views(
            signals, with_confidence=True, confidence_cap=0.5
        )
        assert len(views) == 2
        assert confidences is not None
        assert len(confidences) == len(views)
        assert all(0.0 <= c <= 0.5 for c in confidences)
        # AAPL (strong agreement) more confident than MSFT (max split -> 0)
        assert confidences[0] > confidences[1]

    def test_with_confidence_skips_ticker_missing_votes(self) -> None:
        signals = {
            "AAPL": AnalystSignal(
                current_price=100.0, target_mean=110.0, strong_buy=8
            ),
            "MSFT": AnalystSignal(current_price=200.0, target_mean=210.0),  # no votes
        }
        views, confidences = build_analyst_bl_views(
            signals, with_confidence=True, min_votes=1
        )
        # MSFT skipped so views and confidences stay aligned at length 1
        assert views == ("AAPL == 0.100000",)
        assert confidences is not None
        assert len(confidences) == 1

    def test_negative_precision_raises(self) -> None:
        signals = {"AAPL": AnalystSignal(current_price=100.0, target_mean=110.0)}
        with pytest.raises(ConfigurationError, match="precision"):
            build_analyst_bl_views(signals, precision=-1)


class TestBuildBlackLittermanConfigFromSignals:
    def test_he_litterman_when_no_confidence(self) -> None:
        signals = {"AAPL": AnalystSignal(current_price=100.0, target_mean=110.0)}
        cfg = build_black_litterman_config_from_signals(signals)
        assert isinstance(cfg, BlackLittermanConfig)
        assert cfg.views == ("AAPL == 0.100000",)
        assert cfg.uncertainty_method == ViewUncertaintyMethod.HE_LITTERMAN
        assert cfg.view_confidences is None

    def test_idzorek_when_confidence(self) -> None:
        signals = {
            "AAPL": AnalystSignal(
                current_price=100.0, target_mean=110.0, strong_buy=9, buy=1
            )
        }
        cfg = build_black_litterman_config_from_signals(
            signals, with_confidence=True
        )
        assert cfg.uncertainty_method == ViewUncertaintyMethod.IDZOREK
        assert cfg.view_confidences is not None
        assert len(cfg.view_confidences) == 1

    def test_bl_kwargs_forwarded(self) -> None:
        signals = {"AAPL": AnalystSignal(current_price=100.0, target_mean=110.0)}
        cfg = build_black_litterman_config_from_signals(signals, tau=0.1)
        assert cfg.tau == 0.1


class TestIntegration:
    """End-to-end: analyst signals -> config -> fitted BL posterior."""

    def test_analyst_views_fit_bl(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        # 12-month targets, de-annualised to the daily return scale.
        signals = {
            "AAPL": AnalystSignal(
                current_price=100.0,
                target_mean=125.0,
                strong_buy=8,
                buy=2,
            ),
            "JPM": AnalystSignal(
                current_price=100.0,
                target_mean=108.0,
                strong_buy=4,
                hold=4,
                sell=2,
            ),
        }
        cfg = build_black_litterman_config_from_signals(
            signals,
            horizon_periods=252.0,
            with_confidence=True,
            precision=8,
        )
        prior = build_black_litterman(cfg)
        assert isinstance(prior, BlackLitterman)
        prior.fit(returns)
        rd = prior.return_distribution_
        assert rd.mu is not None
        assert rd.mu.shape == (returns.shape[1],)
        assert np.all(np.isfinite(rd.mu))
