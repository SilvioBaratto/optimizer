"""Empty-data and exception-path tests for risk_analytics_service (issue #825).

Existing tests in tests/unit/test_risk_analytics_service.py already cover:
  - compute_var: historical + parametric happy paths (COVERED)
  - compute_var: raises ValueError for insufficient data (COVERED)
  - compute_correlation: symmetric matrix, cluster labels (COVERED)
  - compute_correlation: raises ValueError for single asset (COVERED)
  - compute_factor_exposure: weighted exposures, empty scores → {} (COVERED)
  - compute_concentration: HHI math, empty portfolio → zero summary (COVERED)
  - compute_liquidity: ADDV formula, missing ADDV → None (COVERED)

Remaining gaps identified by this audit:
  - _assert_sufficient_data: series with exactly 1 observation triggers raise
  - _assert_at_least_two_assets: single-item list triggers raise
  - _surviving_tickers: empty prices DataFrame → empty list (no IndexError)
  - compute_concentration: empty weights → summary all zeros (already in
    existing test; only assert_json_safe on the empty result is missing)
  - compute_liquidity: zero-value ADDV triggers None branch (addv <= 0)
  - _compute_liquidity_summary: all-None assets → zero weighted avg
  - _build_asset_exposures: score with None standardized_score is skipped
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests._fixtures.assertions import assert_json_safe

_PORTFOLIO_ID = "00000000-0000-0000-0000-000000000002"


# ---------------------------------------------------------------------------
# Pure helpers — directly testable without the service
# ---------------------------------------------------------------------------


class TestAssertSufficientData:
    """_assert_sufficient_data raises for empty or single-row series."""

    def test_when_series_empty_then_raises_value_error(self) -> None:
        from app.services.risk.risk_analytics_service import _assert_sufficient_data

        with pytest.raises(ValueError, match="[Ii]nsufficient"):
            _assert_sufficient_data(pd.Series(dtype=float), lookback=252)

    def test_when_series_has_one_row_then_raises_value_error(self) -> None:
        from app.services.risk.risk_analytics_service import _assert_sufficient_data

        single_row = pd.Series([0.01])
        with pytest.raises(ValueError, match="[Ii]nsufficient"):
            _assert_sufficient_data(single_row, lookback=252)

    def test_when_series_has_two_rows_then_does_not_raise(self) -> None:
        from app.services.risk.risk_analytics_service import _assert_sufficient_data

        two_rows = pd.Series([0.01, -0.01])
        # Must NOT raise — two rows is the minimum
        _assert_sufficient_data(two_rows, lookback=2)


class TestAssertAtLeastTwoAssets:
    """_assert_at_least_two_assets raises for zero or one-item lists."""

    def test_when_empty_list_then_raises_value_error(self) -> None:
        from app.services.risk.risk_analytics_service import _assert_at_least_two_assets

        with pytest.raises(ValueError, match="[Aa]t least 2"):
            _assert_at_least_two_assets([])

    def test_when_single_asset_then_raises_value_error(self) -> None:
        from app.services.risk.risk_analytics_service import _assert_at_least_two_assets

        with pytest.raises(ValueError, match="[Aa]t least 2"):
            _assert_at_least_two_assets(["AAPL"])

    def test_when_two_assets_then_does_not_raise(self) -> None:
        from app.services.risk.risk_analytics_service import _assert_at_least_two_assets

        _assert_at_least_two_assets(["AAPL", "MSFT"])


class TestSurvivingTickers:
    """_surviving_tickers returns [] for empty prices frame."""

    def test_when_prices_empty_then_returns_empty_list(self) -> None:
        from app.services.risk.risk_analytics_service import _surviving_tickers

        result = _surviving_tickers({"AAPL": 0.6, "MSFT": 0.4}, pd.DataFrame())

        assert result == []

    def test_when_prices_has_only_some_tickers_then_only_matching_returned(
        self,
    ) -> None:
        from app.services.risk.risk_analytics_service import _surviving_tickers

        prices = pd.DataFrame({"AAPL": [100.0]})
        result = _surviving_tickers({"AAPL": 0.6, "MSFT": 0.4}, prices)

        assert result == ["AAPL"]


class TestBuildAssetExposures:
    """_build_asset_exposures skips scores with None standardized_score."""

    def test_when_all_scores_have_none_standardized_score_then_empty_dict(
        self,
    ) -> None:
        from app.services.risk.risk_analytics_service import _build_asset_exposures

        score = MagicMock()
        score.ticker = "AAPL"
        score.factor_type = "momentum"
        score.standardized_score = None

        result = _build_asset_exposures([score])

        assert result == {}

    def test_when_some_scores_none_then_only_valid_included(self) -> None:
        from app.services.risk.risk_analytics_service import _build_asset_exposures

        good = MagicMock()
        good.ticker = "AAPL"
        good.factor_type = "quality"
        good.standardized_score = 0.7

        bad = MagicMock()
        bad.ticker = "MSFT"
        bad.factor_type = "momentum"
        bad.standardized_score = None

        result = _build_asset_exposures([good, bad])

        assert "AAPL" in result
        assert "MSFT" not in result


# ---------------------------------------------------------------------------
# compute_liquidity — zero ADDV branch
# ---------------------------------------------------------------------------


class TestComputeLiquidityZeroAddv:
    """ADDV of exactly zero is treated the same as None (addv <= 0 branch)."""

    def setup_method(self) -> None:
        self.session = MagicMock()
        from app.services.risk.risk_analytics_service import RiskAnalyticsService

        self.service = RiskAnalyticsService(self.session)

    def test_when_addv_is_zero_then_days_to_liquidate_is_none(self) -> None:
        with (
            patch.object(self.service, "_fetch_addv", return_value=0.0),
            patch.object(
                self.service, "_fetch_asset_names", return_value={"AAPL": "Apple"}
            ),
        ):
            result = self.service.compute_liquidity(
                {"AAPL": 0.5}, lookback_days=20, participation_rate=0.1
            )

        assert result["assets"][0]["days_to_liquidate"] is None
        assert result["assets"][0]["liquidity_cost"] is None
        assert_json_safe(result)

    def test_when_all_addv_zero_then_weighted_avg_is_zero(self) -> None:
        with (
            patch.object(self.service, "_fetch_addv", return_value=0.0),
            patch.object(
                self.service,
                "_fetch_asset_names",
                return_value={"AAPL": "Apple", "MSFT": "Microsoft"},
            ),
        ):
            result = self.service.compute_liquidity(
                {"AAPL": 0.6, "MSFT": 0.4}, lookback_days=20, participation_rate=0.1
            )

        assert result["summary"]["weighted_avg_days_to_liquidate"] == 0.0


# ---------------------------------------------------------------------------
# compute_concentration — json safety on empty weights
# ---------------------------------------------------------------------------


class TestComputeConcentrationJsonSafe:
    """Empty and non-empty concentration results are JSON-safe."""

    def setup_method(self) -> None:
        self.session = MagicMock()
        from app.services.risk.risk_analytics_service import RiskAnalyticsService

        self.service = RiskAnalyticsService(self.session)

    def test_when_weights_empty_then_result_is_json_safe(self) -> None:
        with patch.object(self.service, "_fetch_asset_names", return_value={}):
            result = self.service.compute_concentration({}, n=5)

        assert_json_safe(result)

    def test_when_weights_present_then_result_is_json_safe(self) -> None:
        with patch.object(
            self.service,
            "_fetch_asset_names",
            return_value={"AAPL": "Apple", "MSFT": "Microsoft"},
        ):
            result = self.service.compute_concentration(
                {"AAPL": 0.6, "MSFT": 0.4}, n=2
            )

        assert_json_safe(result)


# ---------------------------------------------------------------------------
# compute_factor_exposure — json safety on no-scores path
# ---------------------------------------------------------------------------


class TestComputeFactorExposureJsonSafe:
    """Empty factor-score result is JSON-safe."""

    def setup_method(self) -> None:
        self.session = MagicMock()
        from app.services.risk.risk_analytics_service import RiskAnalyticsService

        self.service = RiskAnalyticsService(self.session)

    def test_when_no_scores_then_result_is_json_safe(self) -> None:
        with patch.object(
            self.service, "_fetch_latest_factor_scores", return_value=[]
        ):
            result = self.service.compute_factor_exposure(
                portfolio_id=_PORTFOLIO_ID,
                weights={"AAPL": 0.6, "MSFT": 0.4},
            )

        assert_json_safe(result)
