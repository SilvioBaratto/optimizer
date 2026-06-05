"""Unit tests for Trading212 universe filters and FilterPipelineImpl.

Covers:
- MarketCapFilter
- PriceFilter
- LiquidityFilter
- DataCoverageFilter
- HistoricalDataFilter
- FilterPipelineImpl (short-circuit, stats, empty pipeline)

No network is touched: YFinanceClient is always injected as a MagicMock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.trading212.filters.data_coverage import DataCoverageFilter
from app.services.universe.trading212.filters.historical_data import (
    HistoricalDataFilter,
)
from app.services.universe.trading212.filters.liquidity import LiquidityFilter
from app.services.universe.trading212.filters.market_cap import MarketCapFilter
from app.services.universe.trading212.filters.pipeline import FilterPipelineImpl
from app.services.universe.trading212.filters.price import PriceFilter

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CFG = UniverseBuilderConfig()
TICKER = "AAPL"


def _make_hist(n_rows: int) -> pd.DataFrame:
    """Return a DataFrame with *n_rows* rows (column content irrelevant)."""
    return pd.DataFrame({"Close": [100.0] * n_rows})


def _complete_data_coverage_dict() -> dict[str, Any]:
    """Build a dict that satisfies every *required* InstitutionalFieldSpec."""
    data: dict[str, Any] = {}
    for spec in CFG.institutional_fields.values():
        if not spec.required:
            continue
        # Provide all fields listed in the spec
        for f in spec.fields:
            data[f] = 1.0
    return data


# ---------------------------------------------------------------------------
# MarketCapFilter
# ---------------------------------------------------------------------------


class TestMarketCapFilter:
    def setup_method(self) -> None:
        self.f = MarketCapFilter(config=CFG)

    def test_when_empty_dict_then_false(self) -> None:
        passed, reason = self.f.filter({}, TICKER)
        assert passed is False
        assert reason

    def test_when_no_market_cap_key_then_false(self) -> None:
        passed, _ = self.f.filter({"currentPrice": 10.0}, TICKER)
        assert passed is False

    @pytest.mark.parametrize(
        "mcap, expected",
        [
            (99_999_999, False),
            (100_000_000, True),
            (500_000_000, True),
        ],
    )
    def test_boundary_market_cap(self, mcap: float, expected: bool) -> None:
        passed, _ = self.f.filter({"marketCap": mcap}, TICKER)
        assert passed is expected

    def test_when_above_min_then_reason_contains_dollar_sign(self) -> None:
        _, reason = self.f.filter({"marketCap": 100_000_000}, TICKER)
        assert "$" in reason


# ---------------------------------------------------------------------------
# PriceFilter
# ---------------------------------------------------------------------------


class TestPriceFilter:
    def setup_method(self) -> None:
        self.f = PriceFilter(config=CFG)

    def test_when_empty_dict_then_false(self) -> None:
        passed, _ = self.f.filter({}, TICKER)
        assert passed is False

    def test_when_no_price_field_then_false(self) -> None:
        passed, _ = self.f.filter({"marketCap": 1e9}, TICKER)
        assert passed is False

    @pytest.mark.parametrize(
        "price, expected",
        [
            (4.99, False),
            (5.0, True),
            (10_000.0, True),
            (10_000.01, False),
        ],
    )
    def test_boundary_current_price(self, price: float, expected: bool) -> None:
        passed, _ = self.f.filter({"currentPrice": price}, TICKER)
        assert passed is expected

    def test_fallback_to_regular_market_price_when_current_absent(self) -> None:
        passed, _ = self.f.filter({"regularMarketPrice": 50.0}, TICKER)
        assert passed is True

    def test_fallback_below_min_price_is_rejected(self) -> None:
        passed, _ = self.f.filter({"regularMarketPrice": 1.0}, TICKER)
        assert passed is False

    def test_current_price_none_falls_through_to_regular_market(self) -> None:
        # currentPrice=None → falsy → or evaluates regularMarketPrice
        passed, _ = self.f.filter(
            {"currentPrice": None, "regularMarketPrice": 20.0}, TICKER
        )
        assert passed is True


# ---------------------------------------------------------------------------
# LiquidityFilter
# ---------------------------------------------------------------------------


class TestLiquidityFilter:
    def setup_method(self) -> None:
        self.f = LiquidityFilter(config=CFG)

    def test_when_empty_dict_then_false(self) -> None:
        passed, _ = self.f.filter({}, TICKER)
        assert passed is False

    def test_when_no_market_cap_then_false(self) -> None:
        passed, reason = self.f.filter({"currentPrice": 20.0}, TICKER)
        assert passed is False
        assert "market cap" in reason.lower()

    def test_when_no_price_then_false(self) -> None:
        passed, reason = self.f.filter({"marketCap": 10_000_000_000}, TICKER)
        assert passed is False
        assert "price" in reason.lower()

    def test_when_no_volume_then_false(self) -> None:
        passed, reason = self.f.filter(
            {"marketCap": 10_000_000_000, "currentPrice": 20.0}, TICKER
        )
        assert passed is False
        assert "volume" in reason.lower()

    def test_large_cap_passes_on_exact_tier_boundaries(self) -> None:
        # large_cap tier: min_adv_dollars=10M, min_adv_shares=500_000
        # price=20, vol=500_000 → dollar=10M (not <), shares=500k (not <) → True
        data = {
            "marketCap": 10_000_000_000,
            "currentPrice": 20.0,
            "averageVolume": 500_000,
        }
        passed, _ = self.f.filter(data, TICKER)
        assert passed is True

    def test_large_cap_fails_when_shares_one_below_threshold(self) -> None:
        # price=21 → dollar=21*499_999=10_499_979 ≥ 10M (passes dollar check)
        # but shares=499_999 < 500_000 → fails share check
        data = {
            "marketCap": 10_000_000_000,
            "currentPrice": 21.0,
            "averageVolume": 499_999,
        }
        passed, reason = self.f.filter(data, TICKER)
        assert passed is False
        assert "shares" in reason.lower()

    def test_large_cap_fails_when_dollar_volume_too_low(self) -> None:
        # price=10, vol=500_000 → dollar=5M < 10M required
        data = {
            "marketCap": 10_000_000_000,
            "currentPrice": 10.0,
            "averageVolume": 500_000,
        }
        passed, reason = self.f.filter(data, TICKER)
        assert passed is False
        assert "adv" in reason.lower() or "$" in reason

    def test_falls_back_to_average_volume_10days(self) -> None:
        data = {
            "marketCap": 10_000_000_000,
            "currentPrice": 20.0,
            "averageVolume10days": 600_000,
        }
        passed, _ = self.f.filter(data, TICKER)
        assert passed is True

    def test_large_cap_passes_with_healthy_numbers(self) -> None:
        data = {
            "marketCap": 10_000_000_000,
            "currentPrice": 20.0,
            "averageVolume": 600_000,
        }
        passed, _ = self.f.filter(data, TICKER)
        assert passed is True


# ---------------------------------------------------------------------------
# DataCoverageFilter
# ---------------------------------------------------------------------------


class TestDataCoverageFilter:
    def setup_method(self) -> None:
        self.f = DataCoverageFilter(config=CFG)

    def test_when_empty_dict_then_false(self) -> None:
        passed, _ = self.f.filter({}, TICKER)
        assert passed is False

    def test_when_all_required_categories_satisfied_then_true(self) -> None:
        data = _complete_data_coverage_dict()
        passed, reason = self.f.filter(data, TICKER)
        assert passed is True
        assert "100%" in reason

    def test_when_market_cap_missing_then_false(self) -> None:
        data = _complete_data_coverage_dict()
        del data["marketCap"]
        passed, reason = self.f.filter(data, TICKER)
        assert passed is False
        assert "market_cap" in reason or "missing" in reason.lower()

    def test_when_both_52week_fields_missing_then_false(self) -> None:
        # sector_industry and 52week_range use all_required=True
        data = _complete_data_coverage_dict()
        del data["fiftyTwoWeekHigh"]
        del data["fiftyTwoWeekLow"]
        passed, _ = self.f.filter(data, TICKER)
        assert passed is False

    def test_when_at_least_one_price_field_present_then_category_passes(self) -> None:
        # price spec uses at_least_one=True; drop regularMarketPrice, keep currentPrice
        data = _complete_data_coverage_dict()
        data.pop("regularMarketPrice", None)
        data["currentPrice"] = 10.0
        passed, _ = self.f.filter(data, TICKER)
        assert passed is True

    def test_when_optional_beta_missing_then_still_passes(self) -> None:
        data = _complete_data_coverage_dict()
        data.pop("beta", None)
        passed, _ = self.f.filter(data, TICKER)
        assert passed is True


# ---------------------------------------------------------------------------
# HistoricalDataFilter
# ---------------------------------------------------------------------------


class TestHistoricalDataFilter:
    def _make_filter(self, hist_return_value: Any) -> HistoricalDataFilter:
        mock_client = MagicMock()
        mock_client.fetch_history.return_value = hist_return_value
        return HistoricalDataFilter(config=CFG, _yf_client=mock_client)

    def test_when_history_gte_750_days_then_true(self) -> None:
        mock_client = MagicMock()
        mock_client.fetch_history.return_value = _make_hist(750)
        f = HistoricalDataFilter(config=CFG, _yf_client=mock_client)
        passed, _ = f.filter({}, TICKER)
        assert passed is True
        mock_client.fetch_history.assert_called_once_with(
            TICKER, period="5y", max_retries=1, min_rows=1
        )

    def test_when_history_749_days_then_false(self) -> None:
        f = self._make_filter(_make_hist(749))
        passed, _ = f.filter({}, TICKER)
        assert passed is False

    def test_when_history_is_none_then_false(self) -> None:
        f = self._make_filter(None)
        passed, _ = f.filter({}, TICKER)
        assert passed is False

    def test_when_history_is_empty_df_then_false(self) -> None:
        f = self._make_filter(pd.DataFrame())
        passed, _ = f.filter({}, TICKER)
        assert passed is False

    def test_yf_client_lazy_singleton_not_called_when_injected(self) -> None:
        mock_client = MagicMock()
        mock_client.fetch_history.return_value = _make_hist(800)
        f = HistoricalDataFilter(config=CFG, _yf_client=mock_client)
        # Accessing yf_client property should return the injected mock directly
        assert f.yf_client is mock_client


# ---------------------------------------------------------------------------
# FilterPipelineImpl
# ---------------------------------------------------------------------------


@dataclass
class _FakeFilter:
    """Lightweight stand-in that satisfies the InstrumentFilter protocol."""

    name: str
    filter: Any  # MagicMock assigned per test


class TestFilterPipelineImpl:
    def test_empty_pipeline_returns_true_with_no_filters_message(self) -> None:
        pipeline = FilterPipelineImpl()
        passed, reason = pipeline.apply({}, TICKER)
        assert passed is True
        assert "No filters" in reason

    def test_single_passing_filter_returns_true(self) -> None:
        pipeline = FilterPipelineImpl()
        f = _FakeFilter(name="Filter1", filter=MagicMock(return_value=(True, "ok")))
        pipeline.add_filter(f)  # type: ignore[arg-type]
        passed, _ = pipeline.apply({}, TICKER)
        assert passed is True
        assert pipeline.get_summary()["Filter1"]["passed"] == 1
        assert pipeline.get_summary()["Filter1"]["failed"] == 0

    def test_single_failing_filter_returns_false_with_prefix(self) -> None:
        pipeline = FilterPipelineImpl()
        f = _FakeFilter(
            name="Filter1", filter=MagicMock(return_value=(False, "too small"))
        )
        pipeline.add_filter(f)  # type: ignore[arg-type]
        passed, reason = pipeline.apply({}, TICKER)
        assert passed is False
        assert reason == "[Filter1] too small"
        assert pipeline.get_summary()["Filter1"]["failed"] == 1

    def test_short_circuit_second_filter_not_called_when_first_fails(self) -> None:
        pipeline = FilterPipelineImpl()
        f1 = _FakeFilter(
            name="Filter1", filter=MagicMock(return_value=(False, "x"))
        )
        f2 = _FakeFilter(
            name="Filter2", filter=MagicMock(return_value=(True, "ok"))
        )
        pipeline.add_filter(f1)  # type: ignore[arg-type]
        pipeline.add_filter(f2)  # type: ignore[arg-type]

        passed, reason = pipeline.apply({}, TICKER)

        assert passed is False
        assert "[Filter1]" in reason
        f2.filter.assert_not_called()

    def test_stats_accumulate_across_multiple_apply_calls(self) -> None:
        pipeline = FilterPipelineImpl()
        f = _FakeFilter(name="F", filter=MagicMock(return_value=(True, "ok")))
        pipeline.add_filter(f)  # type: ignore[arg-type]

        pipeline.apply({}, TICKER)
        pipeline.apply({}, TICKER)

        assert pipeline.get_summary()["F"]["passed"] == 2

    def test_add_filter_returns_self_for_fluent_chaining(self) -> None:
        pipeline = FilterPipelineImpl()
        f = _FakeFilter(name="F", filter=MagicMock(return_value=(True, "ok")))
        result = pipeline.add_filter(f)  # type: ignore[arg-type]
        assert result is pipeline

    def test_all_filters_pass_increments_each_passed_counter(self) -> None:
        pipeline = FilterPipelineImpl()
        for name in ("A", "B", "C"):
            f = _FakeFilter(name=name, filter=MagicMock(return_value=(True, "ok")))
            pipeline.add_filter(f)  # type: ignore[arg-type]

        passed, reason = pipeline.apply({}, TICKER)

        assert passed is True
        assert reason == "Passed all filters"
        for name in ("A", "B", "C"):
            assert pipeline.get_summary()[name]["passed"] == 1
            assert pipeline.get_summary()[name]["failed"] == 0

    def test_get_summary_returns_copy(self) -> None:
        pipeline = FilterPipelineImpl()
        summary = pipeline.get_summary()
        summary["injected"] = {"passed": 99, "failed": 0}
        assert "injected" not in pipeline.get_summary()
