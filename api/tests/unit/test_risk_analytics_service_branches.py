"""Branch-coverage tests for app/services/risk/risk_analytics_service.py.

Maps term-missing ranges to concrete conditions not covered by
tests/unit/test_risk_analytics_service.py:

  Lines 233-242  — _fetch_weighted_returns: no surviving tickers → empty Series
  Lines 246-249  — _fetch_prices: instrument not found (repo returns None)
  Lines 257-265  — _fetch_asset_names: instrument found with name / short_name / fallback
  Lines 269-281  — _fetch_addv: instrument not found, empty dollar_volumes list
  Line  378      — _build_asset_exposures: standardized_score is None → row skipped
  Line  498      — _compute_liquidity_summary: total_weight == 0 edge case

Do NOT duplicate tests already in tests/unit/test_risk_analytics_service.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.risk.risk_analytics_service import (
    RiskAnalyticsService,
    _build_asset_exposures,
    _build_liquidity_asset,
    _compute_concentration_summary,
    _compute_liquidity_summary,
    _surviving_tickers,
)

_PORTFOLIO_ID = "00000000-0000-0000-0000-000000000002"
_WEIGHTS = {"AAPL": 0.6, "MSFT": 0.4}


# ---------------------------------------------------------------------------
# _fetch_weighted_returns — no surviving tickers returns empty Series (lines 233-242)
# ---------------------------------------------------------------------------


class TestFetchWeightedReturnsNoSurvivors:
    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)

    def test_when_all_tickers_absent_from_prices_then_empty_series_returned(
        self,
    ) -> None:
        # prices has columns but none match weight keys
        prices = pd.DataFrame({"OTHER": [100.0, 101.0, 102.0]})
        with patch.object(self.service, "_fetch_prices", return_value=prices):
            returns = self.service._fetch_weighted_returns({"AAPL": 1.0}, lookback=252)
        assert returns.empty

    def test_when_prices_empty_dataframe_then_empty_series_returned(self) -> None:
        with patch.object(
            self.service, "_fetch_prices", return_value=pd.DataFrame()
        ):
            returns = self.service._fetch_weighted_returns({"AAPL": 1.0}, lookback=252)
        assert returns.empty


# ---------------------------------------------------------------------------
# _surviving_tickers — pure function branches
# ---------------------------------------------------------------------------


class TestSurvivingTickers:
    def test_when_prices_empty_then_returns_empty_list(self) -> None:
        result = _surviving_tickers({"AAPL": 0.5}, pd.DataFrame())
        assert result == []

    def test_when_no_overlap_then_returns_empty_list(self) -> None:
        prices = pd.DataFrame({"MSFT": [100.0, 101.0]})
        result = _surviving_tickers({"AAPL": 1.0}, prices)
        assert result == []

    def test_when_partial_overlap_then_only_matching_tickers_returned(self) -> None:
        prices = pd.DataFrame({"AAPL": [100.0, 101.0], "OTHER": [50.0, 51.0]})
        result = _surviving_tickers({"AAPL": 0.6, "MISSING": 0.4}, prices)
        assert result == ["AAPL"]


# ---------------------------------------------------------------------------
# _fetch_prices — instrument not found branch (lines 246-249)
# ---------------------------------------------------------------------------


class TestFetchPricesWithRows:
    """Line 241: the dict comprehension {r.date: float(r.close) for r in rows if r.close}."""

    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)

    def test_when_rows_present_and_close_not_none_then_ticker_in_prices(self) -> None:
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            instrument = MagicMock()
            instrument.id = "inst-1"
            mock_repo.get_instrument_by_yfinance_ticker.return_value = instrument

            row = MagicMock()
            row.date = "2025-01-10"
            row.close = 150.0
            mock_repo.get_price_history.return_value = [row]

            prices = self.service._fetch_prices({"AAPL": 1.0}, lookback=5)

        assert "AAPL" in prices.columns
        assert 150.0 in prices["AAPL"].values

    def test_when_row_close_is_none_then_row_excluded_from_dict(self) -> None:
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            instrument = MagicMock()
            instrument.id = "inst-1"
            mock_repo.get_instrument_by_yfinance_ticker.return_value = instrument

            row_none_close = MagicMock()
            row_none_close.date = "2025-01-10"
            row_none_close.close = None  # filtered by `if r.close`

            row_valid = MagicMock()
            row_valid.date = "2025-01-11"
            row_valid.close = 155.0

            mock_repo.get_price_history.return_value = [row_none_close, row_valid]

            prices = self.service._fetch_prices({"AAPL": 1.0}, lookback=5)

        assert "AAPL" in prices.columns
        assert len(prices) == 1


class TestFetchLatestFactorScores:
    """Lines 246-249: _fetch_latest_factor_scores delegates to FactorRepository."""

    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)

    def test_delegates_to_factor_repository(self) -> None:
        mock_score = MagicMock()
        with patch(
            "app.services.risk.risk_analytics_service.FactorRepository"
        ) as mock_repo_cls:
            mock_repo_cls.return_value.get_scores_by_tickers_and_date_range.return_value = [
                mock_score
            ]
            result = self.service._fetch_latest_factor_scores(["AAPL"])

        assert result == [mock_score]
        mock_repo_cls.return_value.get_scores_by_tickers_and_date_range.assert_called_once()

    def test_passes_90_day_window(self) -> None:

        with patch(
            "app.services.risk.risk_analytics_service.FactorRepository"
        ) as mock_repo_cls:
            mock_repo_cls.return_value.get_scores_by_tickers_and_date_range.return_value = []
            self.service._fetch_latest_factor_scores(["AAPL"])

        call_kwargs = mock_repo_cls.return_value.get_scores_by_tickers_and_date_range.call_args.kwargs
        end = call_kwargs["end_date"]
        start = call_kwargs["start_date"]
        assert (end - start).days >= 89


class TestFetchPricesInstrumentNotFound:
    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)

    def test_when_instrument_not_in_repo_then_ticker_skipped(self) -> None:
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            mock_repo.get_instrument_by_yfinance_ticker.return_value = None
            prices = self.service._fetch_prices({"UNKNOWN": 1.0}, lookback=30)
        assert prices.empty or "UNKNOWN" not in prices.columns

    def test_when_instrument_found_but_no_price_rows_then_ticker_skipped(self) -> None:
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            instrument = MagicMock()
            instrument.id = "inst-1"
            mock_repo.get_instrument_by_yfinance_ticker.return_value = instrument
            mock_repo.get_price_history.return_value = []  # empty rows
            prices = self.service._fetch_prices({"AAPL": 1.0}, lookback=30)
        assert prices.empty or "AAPL" not in prices.columns


# ---------------------------------------------------------------------------
# _fetch_asset_names — name / short_name / fallback (lines 257-265)
# ---------------------------------------------------------------------------


class TestFetchAssetNames:
    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)

    def _mock_repo(self, instrument: object) -> None:
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo_cls.return_value.get_instrument_by_yfinance_ticker.return_value = (
                instrument
            )
            return self.service._fetch_asset_names(["AAPL"])

    def test_when_instrument_none_then_ticker_used_as_name(self) -> None:
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo_cls.return_value.get_instrument_by_yfinance_ticker.return_value = (
                None
            )
            names = self.service._fetch_asset_names(["AAPL"])
        assert names["AAPL"] == "AAPL"

    def test_when_instrument_has_name_then_name_used(self) -> None:
        instrument = MagicMock()
        instrument.name = "Apple Inc"
        instrument.short_name = "Apple"
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo_cls.return_value.get_instrument_by_yfinance_ticker.return_value = (
                instrument
            )
            names = self.service._fetch_asset_names(["AAPL"])
        assert names["AAPL"] == "Apple Inc"

    def test_when_instrument_name_none_but_short_name_present_then_short_name_used(
        self,
    ) -> None:
        instrument = MagicMock()
        instrument.name = None
        instrument.short_name = "Apple"
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo_cls.return_value.get_instrument_by_yfinance_ticker.return_value = (
                instrument
            )
            names = self.service._fetch_asset_names(["AAPL"])
        assert names["AAPL"] == "Apple"

    def test_when_both_name_and_short_name_none_then_ticker_fallback(self) -> None:
        instrument = MagicMock()
        instrument.name = None
        instrument.short_name = None
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo_cls.return_value.get_instrument_by_yfinance_ticker.return_value = (
                instrument
            )
            names = self.service._fetch_asset_names(["AAPL"])
        assert names["AAPL"] == "AAPL"


# ---------------------------------------------------------------------------
# _fetch_addv — instrument not found + empty dollar_volumes (lines 269-281)
# ---------------------------------------------------------------------------


class TestFetchAddv:
    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)

    def test_when_instrument_not_found_then_returns_none(self) -> None:
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo_cls.return_value.get_instrument_by_yfinance_ticker.return_value = (
                None
            )
            result = self.service._fetch_addv("AAPL", 20)
        assert result is None

    def test_when_price_rows_empty_then_returns_none(self) -> None:
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            instrument = MagicMock()
            instrument.id = "inst-1"
            mock_repo.get_instrument_by_yfinance_ticker.return_value = instrument
            mock_repo.get_price_history.return_value = []
            result = self.service._fetch_addv("AAPL", 20)
        assert result is None

    def test_when_rows_have_none_volume_then_row_excluded_from_mean(self) -> None:
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            instrument = MagicMock()
            instrument.id = "inst-1"
            mock_repo.get_instrument_by_yfinance_ticker.return_value = instrument

            row_valid = MagicMock()
            row_valid.volume = 1_000_000
            row_valid.close = 150.0

            row_none_vol = MagicMock()
            row_none_vol.volume = None
            row_none_vol.close = 150.0

            mock_repo.get_price_history.return_value = [row_valid, row_none_vol]
            result = self.service._fetch_addv("AAPL", 20)

        # Only 1 valid row; ADDV = 1_000_000 * 150.0
        assert result == pytest.approx(1_000_000 * 150.0)

    def test_when_all_rows_have_none_volume_then_returns_none(self) -> None:
        with patch(
            "app.services.risk.risk_analytics_service.YFinanceRepository"
        ) as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            instrument = MagicMock()
            instrument.id = "inst-1"
            mock_repo.get_instrument_by_yfinance_ticker.return_value = instrument

            row = MagicMock()
            row.volume = None
            row.close = 150.0
            mock_repo.get_price_history.return_value = [row]

            result = self.service._fetch_addv("AAPL", 20)
        assert result is None


# ---------------------------------------------------------------------------
# _build_asset_exposures — score None branch (line 378)
# ---------------------------------------------------------------------------


class TestBuildAssetExposures:
    def test_when_standardized_score_none_then_factor_skipped(self) -> None:
        score = MagicMock()
        score.ticker = "AAPL"
        score.factor_type = "momentum"
        score.standardized_score = None  # ← branch under test

        result = _build_asset_exposures([score])

        assert result == {}

    def test_when_standardized_score_present_then_included(self) -> None:
        score = MagicMock()
        score.ticker = "AAPL"
        score.factor_type = "momentum"
        score.standardized_score = 0.7

        result = _build_asset_exposures([score])

        assert result == {"AAPL": {"momentum": pytest.approx(0.7)}}

    def test_when_mixed_none_and_present_scores_then_only_present_included(
        self,
    ) -> None:
        s1 = MagicMock()
        s1.ticker = "AAPL"
        s1.factor_type = "momentum"
        s1.standardized_score = 0.5

        s2 = MagicMock()
        s2.ticker = "MSFT"
        s2.factor_type = "value"
        s2.standardized_score = None

        result = _build_asset_exposures([s1, s2])

        assert "AAPL" in result
        assert "MSFT" not in result


# ---------------------------------------------------------------------------
# _compute_liquidity_summary — total_weight == 0 branch (line 498)
# ---------------------------------------------------------------------------


class TestComputeLiquiditySummaryZeroWeight:
    def test_when_valid_assets_have_zero_weight_then_returns_zero_avg(self) -> None:
        # All valid assets have weight=0 → total_weight=0 → branch at line 498
        assets = [
            {
                "weight": 0.0,
                "days_to_liquidate": 5.0,
            }
        ]
        result = _compute_liquidity_summary({"AAPL": 0.0}, assets)
        assert result["weighted_avg_days_to_liquidate"] == 0.0

    def test_when_no_valid_assets_then_returns_zero_avg(self) -> None:
        # All assets have days_to_liquidate=None → valid=[] branch
        assets = [
            {
                "weight": 0.5,
                "days_to_liquidate": None,
            }
        ]
        result = _compute_liquidity_summary({"AAPL": 0.5}, assets)
        assert result["weighted_avg_days_to_liquidate"] == 0.0


# ---------------------------------------------------------------------------
# _build_liquidity_asset — addv <= 0 branch
# ---------------------------------------------------------------------------


class TestBuildLiquidityAsset:
    def test_when_addv_zero_then_null_fields_returned(self) -> None:
        result = _build_liquidity_asset("AAPL", 0.5, {"AAPL": "Apple Inc"}, 0.0, 0.1)
        assert result["avg_daily_volume"] is None
        assert result["days_to_liquidate"] is None
        assert result["liquidity_cost"] is None

    def test_when_addv_negative_then_null_fields_returned(self) -> None:
        result = _build_liquidity_asset("AAPL", 0.5, {"AAPL": "Apple Inc"}, -1.0, 0.1)
        assert result["avg_daily_volume"] is None

    def test_when_addv_positive_then_days_computed(self) -> None:
        result = _build_liquidity_asset("AAPL", 0.5, {"AAPL": "Apple Inc"}, 1_000_000.0, 0.1)
        expected_days = 0.5 / (1_000_000.0 * 0.1)
        assert result["days_to_liquidate"] == pytest.approx(expected_days)


# ---------------------------------------------------------------------------
# _compute_concentration_summary — hhi == 0 effective_n branch
# ---------------------------------------------------------------------------


class TestComputeConcentrationSummaryHhiZero:
    def test_when_weights_empty_then_all_zeros(self) -> None:
        result = _compute_concentration_summary({}, n=5)
        assert result["hhi"] == 0.0
        assert result["effective_n"] == 0.0
        assert result["top_n_ratio"] == 0.0
