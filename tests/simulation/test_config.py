"""Tests for the validation of the simulation configs.

Every rejected value is a number a caller can plausibly pass by mistake, and each
one has to fail when the config is built, not deep inside a simulation.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.simulation import BootstrapConfig, GbmConfig

_GBM = GbmConfig(initial_price=100.0, mu=0.05, sigma=0.2, n_steps=10, n_paths=100)
_BOOTSTRAP = BootstrapConfig(n_paths=100, n_steps=10)


class TestGbmConfig:
    """``GbmConfig`` validates its settings at construction."""

    def test_defaults(self) -> None:
        """A daily step, antithetic variates and no fixed seed are the defaults."""
        assert _GBM.dt == pytest.approx(1.0 / 252.0)
        assert _GBM.antithetic is True
        assert _GBM.random_state is None

    def test_zero_volatility_and_negative_drift_are_valid(self) -> None:
        """A deterministic decay is a legitimate scenario."""
        config = dataclasses.replace(_GBM, sigma=0.0, mu=-0.1)
        assert config.sigma == 0.0

    def test_config_is_frozen(self) -> None:
        """A shared config cannot be edited behind the caller's back."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            _GBM.sigma = 0.5  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("initial_price", 0.0),
            ("initial_price", -5.0),
            ("initial_price", math.nan),
            ("initial_price", math.inf),
            ("mu", math.nan),
            ("mu", math.inf),
            ("sigma", -0.1),
            ("sigma", math.nan),
            ("sigma", math.inf),
            ("dt", 0.0),
            ("dt", -1.0),
            ("dt", math.nan),
            ("n_steps", 0),
            ("n_paths", 0),
            ("random_state", -1),
            ("initial_price", "100"),
            ("sigma", None),
            ("dt", True),
            ("n_steps", 2.5),
            ("n_steps", True),
            ("n_paths", math.nan),
            ("n_paths", "10"),
            ("random_state", 1.5),
            ("random_state", True),
        ],
    )
    def test_invalid_setting_raises(self, field: str, value: object) -> None:
        """Each invalid value, wrong type included, names the offending field."""
        with pytest.raises(ConfigurationError, match=field):
            dataclasses.replace(_GBM, **{field: value})

    def test_numpy_integers_are_counts_and_seeds(self) -> None:
        """Values read from numpy arrays are valid settings."""
        config = dataclasses.replace(
            _GBM, n_steps=np.int64(5), n_paths=np.int32(7), random_state=np.int64(3)
        )
        assert config.n_steps == 5


class TestBootstrapConfig:
    """``BootstrapConfig`` validates its settings at construction."""

    def test_defaults(self) -> None:
        """The default is the i.i.d. bootstrap without a fixed seed."""
        assert _BOOTSTRAP.block_size == 1
        assert _BOOTSTRAP.random_state is None

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("n_paths", 0),
            ("n_steps", 0),
            ("block_size", 0),
            ("random_state", -3),
            ("n_paths", 2.5),
            ("block_size", True),
            ("n_steps", math.nan),
            ("random_state", 0.5),
        ],
    )
    def test_invalid_setting_raises(self, field: str, value: object) -> None:
        """Each invalid value, wrong type included, names the offending field."""
        with pytest.raises(ConfigurationError, match=field):
            dataclasses.replace(_BOOTSTRAP, **{field: value})
