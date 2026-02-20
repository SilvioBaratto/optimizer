"""Tests for universe screening configuration."""

from __future__ import annotations

import pytest

from optimizer.universe import (
    ExchangeRegion,
    HysteresisConfig,
    InvestabilityScreenConfig,
)


class TestExchangeRegion:
    def test_members(self) -> None:
        assert set(ExchangeRegion) == {
            ExchangeRegion.US,
            ExchangeRegion.EUROPE,
        }

    def test_str_serialization(self) -> None:
        assert ExchangeRegion.US.value == "us"
        assert ExchangeRegion.EUROPE.value == "europe"


class TestHysteresisConfig:
    def test_creation(self) -> None:
        cfg = HysteresisConfig(entry=100.0, exit_=80.0)
        assert cfg.entry == 100.0
        assert cfg.exit_ == 80.0

    def test_frozen(self) -> None:
        cfg = HysteresisConfig(entry=100.0, exit_=80.0)
        with pytest.raises(AttributeError):
            cfg.entry = 50.0  # type: ignore[misc]

    def test_exit_above_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="exit_.*must be <= entry"):
            HysteresisConfig(entry=100.0, exit_=150.0)

    def test_equal_entry_exit_allowed(self) -> None:
        cfg = HysteresisConfig(entry=100.0, exit_=100.0)
        assert cfg.entry == cfg.exit_


class TestInvestabilityScreenConfig:
    def test_default_values(self) -> None:
        cfg = InvestabilityScreenConfig()
        assert cfg.market_cap.entry == 200_000_000
        assert cfg.market_cap.exit_ == 150_000_000
        assert cfg.addv_12m.entry == 750_000
        assert cfg.addv_3m.entry == 500_000
        assert cfg.trading_frequency.entry == 0.95
        assert cfg.price_us.entry == 3.0
        assert cfg.price_europe.entry == 2.0
        assert cfg.min_trading_history == 252
        assert cfg.min_ipo_seasoning == 60
        assert cfg.min_annual_reports == 3
        assert cfg.min_quarterly_reports == 8
        assert cfg.exchange_region == ExchangeRegion.US
        assert cfg.mcap_percentile_entry == 0.10
        assert cfg.mcap_percentile_exit == 0.075

    def test_frozen(self) -> None:
        cfg = InvestabilityScreenConfig()
        with pytest.raises(AttributeError):
            cfg.min_trading_history = 100  # type: ignore[misc]

    def test_percentile_exit_above_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="mcap_percentile_exit"):
            InvestabilityScreenConfig(
                mcap_percentile_entry=0.10,
                mcap_percentile_exit=0.15,
            )

    def test_percentile_entry_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="mcap_percentile_exit"):
            InvestabilityScreenConfig(
                mcap_percentile_entry=1.5,
                mcap_percentile_exit=0.10,
            )

    def test_custom_values(self) -> None:
        cfg = InvestabilityScreenConfig(
            market_cap=HysteresisConfig(entry=500_000_000, exit_=400_000_000),
            min_trading_history=126,
            exchange_region=ExchangeRegion.EUROPE,
        )
        assert cfg.market_cap.entry == 500_000_000
        assert cfg.min_trading_history == 126
        assert cfg.exchange_region == ExchangeRegion.EUROPE


class TestFactoryMethods:
    def test_for_developed_markets(self) -> None:
        cfg = InvestabilityScreenConfig.for_developed_markets()
        assert cfg.market_cap.entry == 200_000_000
        assert cfg.min_trading_history == 252

    def test_for_broad_universe(self) -> None:
        cfg = InvestabilityScreenConfig.for_broad_universe()
        assert cfg.market_cap.entry == 100_000_000
        assert cfg.min_trading_history == 126
        assert cfg.min_annual_reports == 2

    def test_for_small_cap(self) -> None:
        cfg = InvestabilityScreenConfig.for_small_cap()
        assert cfg.market_cap.entry == 50_000_000
        assert cfg.addv_12m.entry == 250_000
        assert cfg.price_us.entry == 1.0
