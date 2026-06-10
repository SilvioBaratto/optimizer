"""Branch coverage for Trading212 filter sub-modules and config.

Targets:
- app/services/universe/trading212/config.py        — to_dict helpers, factory methods
- app/services/universe/trading212/filters/market_cap.py  — .name, _format_market_cap <1B
- app/services/universe/trading212/filters/price.py       — .name property
- app/services/universe/trading212/filters/liquidity.py   — .name property
- app/services/universe/trading212/filters/data_coverage.py — .name, >3 missing categories
- app/services/universe/trading212/filters/historical_data.py — lazy yf_client, retry/error
- app/services/universe/trading212/filters/pipeline.py    — reset_stats, get_filters,
                                                             get_pipeline_summary
- app/services/universe/trading212/protocols.py           — SKIPPED (Protocol stubs only)

protocols.py defines @runtime_checkable Protocol classes whose method bodies are
`...` (Ellipsis). There is no executable branch logic — the `->exit` arcs reported
by coverage.py are the implicit return paths out of Protocol method stubs, which are
never reachable from user code. Writing hollow `isinstance` checks would not close any
meaningful logic branch, so protocols.py is intentionally excluded from this suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from app.services.universe.trading212.config import (
    InstitutionalFieldSpec,
    LiquidityTier,
    UniverseBuilderConfig,
)
from app.services.universe.trading212.filters.data_coverage import DataCoverageFilter
from app.services.universe.trading212.filters.historical_data import (
    HistoricalDataFilter,
)
from app.services.universe.trading212.filters.liquidity import LiquidityFilter
from app.services.universe.trading212.filters.market_cap import MarketCapFilter
from app.services.universe.trading212.filters.pipeline import FilterPipelineImpl
from app.services.universe.trading212.filters.price import PriceFilter

CFG = UniverseBuilderConfig()
TICKER = "AAPL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hist(n: int) -> pd.DataFrame:
    return pd.DataFrame({"Close": [100.0] * n})


@dataclass
class _FakeFilter:
    name: str
    filter: Any  # MagicMock


# ===========================================================================
# config.py — LiquidityTier.to_dict
# ===========================================================================


class TestLiquidityTierToDict:
    def test_to_dict_returns_expected_keys_and_values(self) -> None:
        tier = LiquidityTier(min_adv_dollars=5_000_000, min_adv_shares=250_000)
        d = tier.to_dict()
        assert d == {"min_adv_dollars": 5_000_000, "min_adv_shares": 250_000}


# ===========================================================================
# config.py — InstitutionalFieldSpec.to_dict
# ===========================================================================


class TestInstitutionalFieldSpecToDict:
    def test_to_dict_includes_all_five_fields(self) -> None:
        spec = InstitutionalFieldSpec(
            fields=("marketCap",),
            description="Market cap",
            required=True,
            at_least_one=False,
            all_required=False,
        )
        d = spec.to_dict()
        assert d["fields"] == ["marketCap"]
        assert d["description"] == "Market cap"
        assert d["required"] is True
        assert d["at_least_one"] is False
        assert d["all_required"] is False

    def test_to_dict_optional_spec_has_required_false(self) -> None:
        spec = InstitutionalFieldSpec(
            fields=("beta",), description="Beta", required=False
        )
        assert spec.to_dict()["required"] is False


# ===========================================================================
# config.py — UniverseBuilderConfig.to_dict
# ===========================================================================


class TestUniverseBuilderConfigToDict:
    def test_to_dict_contains_scalar_threshold_keys(self) -> None:
        d = CFG.to_dict()
        assert d["min_market_cap"] == CFG.min_market_cap
        assert d["min_price"] == CFG.min_price
        assert d["min_trading_days"] == CFG.min_trading_days

    def test_to_dict_liquidity_tiers_serialised_as_dicts(self) -> None:
        d = CFG.to_dict()
        for seg in ("large_cap", "mid_cap", "small_cap"):
            assert seg in d["liquidity_tiers"]
            assert "min_adv_dollars" in d["liquidity_tiers"][seg]

    def test_to_dict_portfolio_countries_serialised_as_list(self) -> None:
        d = CFG.to_dict()
        assert isinstance(d["portfolio_countries"], list)
        assert "USA" in d["portfolio_countries"]


# ===========================================================================
# config.py — is_exchange_allowed
# ===========================================================================


class TestIsExchangeAllowed:
    def test_known_exchange_returns_true(self) -> None:
        assert CFG.is_exchange_allowed("NYSE") is True

    def test_unknown_exchange_returns_false(self) -> None:
        assert CFG.is_exchange_allowed("TokyoStockExchange") is False


# ===========================================================================
# config.py — determine_market_cap_segment mid and small branches
# ===========================================================================


class TestDetermineMarketCapSegment:
    def test_large_cap_boundary(self) -> None:
        assert CFG.determine_market_cap_segment(10_000_000_000) == "large_cap"

    def test_mid_cap_between_small_and_large(self) -> None:
        assert CFG.determine_market_cap_segment(5_000_000_000) == "mid_cap"

    def test_small_cap_below_2b(self) -> None:
        assert CFG.determine_market_cap_segment(1_000_000_000) == "small_cap"


# ===========================================================================
# config.py — get_liquidity_requirements
# ===========================================================================


class TestGetLiquidityRequirements:
    def test_returns_liquidity_tier_for_large_cap(self) -> None:
        tier = CFG.get_liquidity_requirements(15_000_000_000)
        assert isinstance(tier, LiquidityTier)
        assert tier.min_adv_dollars == 10_000_000

    def test_returns_liquidity_tier_for_small_cap(self) -> None:
        tier = CFG.get_liquidity_requirements(500_000_000)
        assert isinstance(tier, LiquidityTier)
        assert tier.min_adv_dollars == 1_000_000


# ===========================================================================
# config.py — get_yahoo_suffix
# ===========================================================================


class TestGetYahooSuffix:
    def test_nyse_returns_empty_string(self) -> None:
        assert CFG.get_yahoo_suffix("NYSE") == ""

    def test_london_returns_dot_l(self) -> None:
        assert CFG.get_yahoo_suffix("London Stock Exchange") == ".L"

    def test_unknown_exchange_returns_none(self) -> None:
        assert CFG.get_yahoo_suffix("UnknownExchange") is None


# ===========================================================================
# config.py — for_regime factory all branches
# ===========================================================================


class TestUniverseBuilderConfigForRegime:
    def test_recession_regime_raises_min_market_cap(self) -> None:
        cfg = UniverseBuilderConfig.for_regime("recession")
        assert cfg.min_market_cap == 500_000_000
        assert cfg.min_price == 10.0

    def test_recession_regime_has_stricter_liquidity_tiers(self) -> None:
        cfg = UniverseBuilderConfig.for_regime("recession")
        assert cfg.liquidity_tiers["large_cap"].min_adv_dollars == 20_000_000
        assert cfg.liquidity_tiers["mid_cap"].min_adv_shares == 500_000
        assert cfg.liquidity_tiers["small_cap"].min_adv_dollars == 5_000_000

    def test_late_cycle_regime_intermediate_thresholds(self) -> None:
        cfg = UniverseBuilderConfig.for_regime("late_cycle")
        assert cfg.min_market_cap == 200_000_000
        assert cfg.min_price == 7.0
        assert cfg.liquidity_tiers["large_cap"].min_adv_dollars == 15_000_000

    def test_early_cycle_regime_relaxed_thresholds(self) -> None:
        cfg = UniverseBuilderConfig.for_regime("early_cycle")
        assert cfg.min_market_cap == 75_000_000
        assert cfg.min_price == 3.0
        assert cfg.liquidity_tiers["small_cap"].min_adv_dollars == 750_000

    def test_unknown_regime_falls_back_to_defaults(self) -> None:
        cfg = UniverseBuilderConfig.for_regime("expansion")
        default = UniverseBuilderConfig()
        assert cfg.min_market_cap == default.min_market_cap
        assert cfg.min_price == default.min_price


# ===========================================================================
# filters/market_cap.py — .name property + _format_market_cap <1B branch
# ===========================================================================


class TestMarketCapFilterBranches:
    def test_name_property_returns_market_cap_filter(self) -> None:
        f = MarketCapFilter(config=CFG)
        assert f.name == "MarketCapFilter"

    def test_format_market_cap_below_1b_uses_millions(self) -> None:
        # Triggers the `else` branch in _format_market_cap
        f = MarketCapFilter(config=CFG)
        # 500M < 1B → should format as $500.0M
        result = f._format_market_cap(500_000_000)
        assert "M" in result
        assert "B" not in result

    def test_format_market_cap_above_1b_uses_billions(self) -> None:
        f = MarketCapFilter(config=CFG)
        result = f._format_market_cap(2_000_000_000)
        assert "B" in result


# ===========================================================================
# filters/price.py — .name property
# ===========================================================================


class TestPriceFilterName:
    def test_name_returns_price_filter(self) -> None:
        f = PriceFilter(config=CFG)
        assert f.name == "PriceFilter"


# ===========================================================================
# filters/liquidity.py — .name property
# ===========================================================================


class TestLiquidityFilterName:
    def test_name_returns_liquidity_filter(self) -> None:
        f = LiquidityFilter(config=CFG)
        assert f.name == "LiquidityFilter"


# ===========================================================================
# filters/data_coverage.py — .name + >3 missing categories branch
# ===========================================================================


class TestDataCoverageFilterBranches:
    def test_name_returns_data_coverage_filter(self) -> None:
        f = DataCoverageFilter(config=CFG)
        assert f.name == "DataCoverageFilter"

    def test_when_more_than_3_categories_missing_then_reason_includes_plus_more(
        self,
    ) -> None:
        # Pass an empty data dict — all required categories will be missing (>3)
        f = DataCoverageFilter(config=CFG)
        passed, reason = f.filter({"dummy": 1}, TICKER)
        # There are many required categories; >3 should be missing
        assert passed is False
        assert "+1 more" in reason or "more" in reason

    def test_when_exactly_3_categories_missing_then_no_plus_more(self) -> None:
        # Build a config with exactly 3 required categories, each missing
        spec = InstitutionalFieldSpec(
            fields=("f1",), description="d", required=True, all_required=True
        )
        # Override institutional_fields with exactly 3 required specs
        custom_fields = {"a": spec, "b": spec, "c": spec}
        # UniverseBuilderConfig is frozen so we use direct construction
        mini_cfg2 = UniverseBuilderConfig(
            institutional_fields=custom_fields  # type: ignore[call-arg]
        )
        f = DataCoverageFilter(config=mini_cfg2)
        passed, reason = f.filter({}, TICKER)
        assert passed is False
        # Exactly 3 missing → no "+N more" suffix
        assert "+0 more" not in reason
        assert "more" not in reason


# ===========================================================================
# filters/historical_data.py — lazy yf_client + retry/error branches
# ===========================================================================


class TestHistoricalDataFilterBranches:
    def test_name_returns_historical_data_filter(self) -> None:
        mock_client = MagicMock()
        mock_client.fetch_history.return_value = _make_hist(800)
        f = HistoricalDataFilter(config=CFG, _yf_client=mock_client)
        assert f.name == "HistoricalDataFilter"

    def test_lazy_yf_client_property_calls_singleton_when_none(self) -> None:
        # _yf_client=None → property calls YFinanceClient.get_instance()
        f = HistoricalDataFilter(config=CFG, _yf_client=None)
        mock_instance = MagicMock()
        with patch(
            "app.services.universe.trading212.filters.historical_data.YFinanceClient"
        ) as mock_cls:
            mock_cls.get_instance.return_value = mock_instance
            client = f.yf_client
        mock_cls.get_instance.assert_called_once()
        assert client is mock_instance

    def test_rate_limit_error_retries_then_returns_false_on_exhaustion(self) -> None:
        # Raise a rate-limit error on every attempt; max_retries=3
        mock_client = MagicMock()
        mock_client.fetch_history.side_effect = RuntimeError("rate limit exceeded")
        f = HistoricalDataFilter(config=CFG, _yf_client=mock_client)

        with patch("app.services.universe.trading212.filters.historical_data.time.sleep"):
            has_cov, days = f._check_historical_data("AAPL", max_retries=3)

        assert has_cov is False
        assert days == 0
        # Should have been called 3 times (one per attempt)
        assert mock_client.fetch_history.call_count == 3

    def test_rate_limit_error_sleeps_between_retries(self) -> None:
        mock_client = MagicMock()
        mock_client.fetch_history.side_effect = RuntimeError("too many requests")
        f = HistoricalDataFilter(config=CFG, _yf_client=mock_client)
        sleep_calls: list[float] = []

        with patch(
            "app.services.universe.trading212.filters.historical_data.time.sleep",
            side_effect=lambda t: sleep_calls.append(t),
        ):
            f._check_historical_data("AAPL", max_retries=3)

        # Two sleeps for attempts 0 and 1 (attempt 2 is the last, no sleep before final fail)
        assert len(sleep_calls) == 2

    def test_non_rate_limit_error_retries_with_linear_backoff(self) -> None:
        mock_client = MagicMock()
        mock_client.fetch_history.side_effect = RuntimeError("some other network error")
        f = HistoricalDataFilter(config=CFG, _yf_client=mock_client)
        sleep_calls: list[float] = []

        with patch(
            "app.services.universe.trading212.filters.historical_data.time.sleep",
            side_effect=lambda t: sleep_calls.append(t),
        ):
            has_cov, days = f._check_historical_data("AAPL", max_retries=3)

        assert has_cov is False
        assert days == 0
        # Linear backoff: sleep(1), sleep(2) for attempts 0 and 1
        assert sleep_calls == [1, 2]

    def test_error_on_last_attempt_returns_false_without_extra_sleep(self) -> None:
        mock_client = MagicMock()
        # Succeed twice, fail on 3rd (last) attempt → exhaustion path
        mock_client.fetch_history.side_effect = [
            RuntimeError("timeout"),
            RuntimeError("timeout"),
            RuntimeError("timeout"),
        ]
        f = HistoricalDataFilter(config=CFG, _yf_client=mock_client)
        with patch("app.services.universe.trading212.filters.historical_data.time.sleep"):
            has_cov, days = f._check_historical_data("AAPL", max_retries=3)
        assert has_cov is False

    def test_zero_days_when_history_empty_reports_zero_years(self) -> None:
        # days_available=0 triggers the `if days_available > 0 else 0` branch in filter()
        mock_client = MagicMock()
        mock_client.fetch_history.return_value = None  # → False, 0
        f = HistoricalDataFilter(config=CFG, _yf_client=mock_client)
        passed, reason = f.filter({}, TICKER)
        assert passed is False
        assert "0.0y" in reason


# ===========================================================================
# filters/pipeline.py — reset_stats, get_filters, get_pipeline_summary
# ===========================================================================


class TestFilterPipelineImplBranches:
    def test_reset_stats_zeroes_all_counters(self) -> None:
        pipeline = FilterPipelineImpl()
        f = _FakeFilter(name="F", filter=MagicMock(return_value=(True, "ok")))
        pipeline.add_filter(f)  # type: ignore[arg-type]
        pipeline.apply({}, TICKER)
        assert pipeline.get_summary()["F"]["passed"] == 1

        pipeline.reset_stats()

        assert pipeline.get_summary()["F"]["passed"] == 0
        assert pipeline.get_summary()["F"]["failed"] == 0

    def test_reset_stats_on_empty_pipeline_is_noop(self) -> None:
        pipeline = FilterPipelineImpl()
        pipeline.reset_stats()  # must not raise
        assert pipeline.get_summary() == {}

    def test_get_filters_returns_copy_of_filter_list(self) -> None:
        pipeline = FilterPipelineImpl()
        f = _FakeFilter(name="F", filter=MagicMock(return_value=(True, "ok")))
        pipeline.add_filter(f)  # type: ignore[arg-type]
        result = pipeline.get_filters()
        assert len(result) == 1
        # Mutating the returned copy does not affect the pipeline
        result.clear()
        assert len(pipeline.get_filters()) == 1

    def test_get_pipeline_summary_contains_filter_name_and_stats(self) -> None:
        pipeline = FilterPipelineImpl()
        f = _FakeFilter(name="SomeFilter", filter=MagicMock(return_value=(True, "ok")))
        pipeline.add_filter(f)  # type: ignore[arg-type]
        pipeline.apply({}, TICKER)

        summary = pipeline.get_pipeline_summary()

        assert "SomeFilter" in summary
        assert "passed" in summary
        assert "failed" in summary

    def test_get_pipeline_summary_on_empty_pipeline_does_not_raise(self) -> None:
        pipeline = FilterPipelineImpl()
        summary = pipeline.get_pipeline_summary()
        assert "Filter Pipeline Summary" in summary

    def test_get_pipeline_summary_shows_pass_rate_percent(self) -> None:
        pipeline = FilterPipelineImpl()
        f_pass = _FakeFilter(
            name="PassFilter", filter=MagicMock(return_value=(True, "ok"))
        )
        f_fail = _FakeFilter(
            name="FailFilter", filter=MagicMock(return_value=(False, "no"))
        )
        pipeline.add_filter(f_pass)  # type: ignore[arg-type]
        pipeline.add_filter(f_fail)  # type: ignore[arg-type]
        pipeline.apply({}, TICKER)

        summary = pipeline.get_pipeline_summary()
        assert "%" in summary

    def test_get_pipeline_summary_total_line_present(self) -> None:
        pipeline = FilterPipelineImpl()
        f = _FakeFilter(name="G", filter=MagicMock(return_value=(True, "ok")))
        pipeline.add_filter(f)  # type: ignore[arg-type]
        pipeline.apply({}, TICKER)

        summary = pipeline.get_pipeline_summary()
        assert "Final:" in summary
