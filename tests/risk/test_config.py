"""Tests for the Value-at-Risk configuration."""

from __future__ import annotations

import dataclasses
import math

import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.risk import VarConfig, VarMethod


class TestVarMethod:
    """``VarMethod`` names the four estimators."""

    def test_members_are_strings(self) -> None:
        """A method compares equal to its lower-case name."""
        assert VarMethod.HISTORICAL == "historical"
        assert VarMethod.PARAMETRIC == "parametric"
        assert VarMethod.CORNISH_FISHER == "cornish_fisher"
        assert VarMethod.MONTE_CARLO == "monte_carlo"

    def test_there_are_four_methods(self) -> None:
        """No estimator is missing or left over."""
        assert len(VarMethod) == 4


class TestVarConfig:
    """``VarConfig`` defaults and validation."""

    def test_defaults(self) -> None:
        """The defaults are a one-period historical VaR at 95% confidence."""
        config = VarConfig()
        assert config.confidence_level == 0.95
        assert config.method is VarMethod.HISTORICAL
        assert config.horizon == 1
        assert config.n_simulations == 10_000
        assert config.random_state is None
        assert config.min_observations == 30

    def test_is_immutable(self) -> None:
        """A shared config cannot be changed behind a caller's back."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            VarConfig().horizon = 5  # type: ignore[misc]

    def test_accepts_the_boundary_values(self) -> None:
        """The smallest allowed settings are accepted."""
        config = VarConfig(
            horizon=1, n_simulations=100, random_state=0, min_observations=10
        )
        assert config.n_simulations == 100

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"confidence_level": 0.0}, "confidence_level"),
            ({"confidence_level": 1.0}, "confidence_level"),
            ({"confidence_level": -0.5}, "confidence_level"),
            ({"confidence_level": math.nan}, "confidence_level"),
            ({"method": "historical"}, "method"),
            ({"horizon": 0}, "horizon"),
            ({"horizon": 2.5}, "horizon"),
            ({"horizon": True}, "horizon"),
            ({"n_simulations": 99}, "n_simulations"),
            ({"random_state": -1}, "random_state"),
            ({"min_observations": 9}, "min_observations"),
        ],
    )
    def test_invalid_values_raise(
        self, overrides: dict[str, object], message: str
    ) -> None:
        """Out-of-range settings fail when the config is built."""
        with pytest.raises(ConfigurationError, match=message):
            VarConfig(**overrides)  # type: ignore[arg-type]
