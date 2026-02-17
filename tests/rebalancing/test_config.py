"""Tests for rebalancing configs."""

from __future__ import annotations

import pytest

from optimizer.rebalancing import (
    TRADING_DAYS,
    CalendarRebalancingConfig,
    RebalancingFrequency,
    ThresholdRebalancingConfig,
    ThresholdType,
)


class TestRebalancingFrequency:
    def test_members(self) -> None:
        assert set(RebalancingFrequency) == {
            RebalancingFrequency.MONTHLY,
            RebalancingFrequency.QUARTERLY,
            RebalancingFrequency.SEMIANNUAL,
            RebalancingFrequency.ANNUAL,
        }

    def test_str_serialization(self) -> None:
        assert RebalancingFrequency.MONTHLY.value == "monthly"
        assert RebalancingFrequency.ANNUAL.value == "annual"


class TestThresholdType:
    def test_members(self) -> None:
        assert set(ThresholdType) == {
            ThresholdType.ABSOLUTE,
            ThresholdType.RELATIVE,
        }


class TestTradingDays:
    def test_monthly(self) -> None:
        assert TRADING_DAYS[RebalancingFrequency.MONTHLY] == 21

    def test_quarterly(self) -> None:
        assert TRADING_DAYS[RebalancingFrequency.QUARTERLY] == 63

    def test_semiannual(self) -> None:
        assert TRADING_DAYS[RebalancingFrequency.SEMIANNUAL] == 126

    def test_annual(self) -> None:
        assert TRADING_DAYS[RebalancingFrequency.ANNUAL] == 252


class TestCalendarRebalancingConfig:
    def test_defaults(self) -> None:
        cfg = CalendarRebalancingConfig()
        assert cfg.frequency == RebalancingFrequency.QUARTERLY
        assert cfg.trading_days == 63

    def test_frozen(self) -> None:
        cfg = CalendarRebalancingConfig()
        with pytest.raises(AttributeError):
            cfg.frequency = RebalancingFrequency.MONTHLY  # type: ignore[misc]

    def test_for_monthly(self) -> None:
        cfg = CalendarRebalancingConfig.for_monthly()
        assert cfg.frequency == RebalancingFrequency.MONTHLY
        assert cfg.trading_days == 21

    def test_for_quarterly(self) -> None:
        cfg = CalendarRebalancingConfig.for_quarterly()
        assert cfg.trading_days == 63

    def test_for_annual(self) -> None:
        cfg = CalendarRebalancingConfig.for_annual()
        assert cfg.trading_days == 252


class TestThresholdRebalancingConfig:
    def test_defaults(self) -> None:
        cfg = ThresholdRebalancingConfig()
        assert cfg.threshold_type == ThresholdType.ABSOLUTE
        assert cfg.threshold == 0.05

    def test_frozen(self) -> None:
        cfg = ThresholdRebalancingConfig()
        with pytest.raises(AttributeError):
            cfg.threshold = 0.10  # type: ignore[misc]

    def test_for_absolute(self) -> None:
        cfg = ThresholdRebalancingConfig.for_absolute(threshold=0.03)
        assert cfg.threshold_type == ThresholdType.ABSOLUTE
        assert cfg.threshold == 0.03

    def test_for_relative(self) -> None:
        cfg = ThresholdRebalancingConfig.for_relative(threshold=0.20)
        assert cfg.threshold_type == ThresholdType.RELATIVE
        assert cfg.threshold == 0.20
