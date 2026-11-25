"""Tests for the strategies configs and enum."""

from __future__ import annotations

import dataclasses

import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.strategies import BacktestConfig, CostConfig, MovingAverageKind


class TestMovingAverageKind:
    """``MovingAverageKind`` is a string enum."""

    def test_members_carry_their_lowercase_names(self) -> None:
        """The values are the lowercase names, usable as plain strings."""
        assert MovingAverageKind.SMA == "sma"
        assert MovingAverageKind.EMA == "ema"

    def test_a_string_converts_to_the_member(self) -> None:
        """A configuration file can name the kind as text."""
        assert MovingAverageKind("ema") is MovingAverageKind.EMA


class TestCostConfig:
    """``CostConfig`` holds the proportional trading cost."""

    def test_default_is_ten_basis_points(self) -> None:
        """The default one-way cost is 10 bps per unit of traded notional."""
        assert CostConfig().proportional_bps == 10.0

    def test_zero_cost_is_allowed(self) -> None:
        """A frictionless backtest is a legitimate benchmark."""
        assert CostConfig(proportional_bps=0.0).proportional_bps == 0.0

    @pytest.mark.parametrize("bps", [-0.1, float("nan"), float("inf")])
    def test_a_negative_or_non_finite_cost_raises(self, bps: float) -> None:
        """A cost must be a finite, non-negative number of basis points."""
        with pytest.raises(ConfigurationError, match="proportional_bps"):
            CostConfig(proportional_bps=bps)

    def test_is_frozen(self) -> None:
        """A shared configuration cannot be mutated by accident."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            CostConfig().proportional_bps = 1.0  # type: ignore[misc]


class TestBacktestConfig:
    """``BacktestConfig`` holds the backtest conventions."""

    def test_defaults(self) -> None:
        """The defaults are a daily, one-period-lag, 10 bps backtest."""
        config = BacktestConfig()
        assert config.periods_per_year == 252
        assert config.execution_lag == 1
        assert config.costs == CostConfig()
        assert config.initial_capital == 1.0

    def test_zero_lag_is_allowed(self) -> None:
        """Lag zero is permitted so the look-ahead leak can be demonstrated."""
        assert BacktestConfig(execution_lag=0).execution_lag == 0

    def test_a_non_positive_periods_per_year_raises(self) -> None:
        """At least one period per year is needed to annualise."""
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            BacktestConfig(periods_per_year=0)

    def test_a_negative_execution_lag_raises(self) -> None:
        """A negative lag would trade on information from the future."""
        with pytest.raises(ConfigurationError, match="execution_lag"):
            BacktestConfig(execution_lag=-1)

    @pytest.mark.parametrize("capital", [0.0, -5.0, float("nan"), float("inf")])
    def test_a_non_positive_or_non_finite_capital_raises(self, capital: float) -> None:
        """The starting equity must be a finite positive amount."""
        with pytest.raises(ConfigurationError, match="initial_capital"):
            BacktestConfig(initial_capital=capital)

    def test_is_frozen(self) -> None:
        """A shared configuration cannot be mutated by accident."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            BacktestConfig().execution_lag = 2  # type: ignore[misc]
