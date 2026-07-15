"""Tests for HMM regime configuration."""

from __future__ import annotations

import dataclasses

import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.regime import HMMCovarianceType, HMMFeatureType, HMMRegimeConfig


class TestHMMFeatureType:
    def test_when_listed_then_three_members_present(self) -> None:
        members = {m.name for m in HMMFeatureType}
        assert members == {"RETURN", "RETURN_VOL", "RETURN_VOL_SKEW"}


class TestHMMCovarianceType:
    def test_when_listed_then_four_members_present(self) -> None:
        members = {m.name for m in HMMCovarianceType}
        assert members == {"FULL", "DIAG", "TIED", "SPHERICAL"}


class TestHMMRegimeConfig:
    def test_when_default_then_two_regime_return_vol(self) -> None:
        cfg = HMMRegimeConfig()
        assert cfg.n_regimes == 2
        assert cfg.feature == HMMFeatureType.RETURN_VOL
        assert cfg.covariance_type == HMMCovarianceType.DIAG

    def test_when_constructed_then_frozen_dataclass(self) -> None:
        cfg = HMMRegimeConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.n_regimes = 3  # type: ignore[misc]

    def test_when_n_regimes_below_two_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="n_regimes"):
            HMMRegimeConfig(n_regimes=1)

    def test_when_vol_window_below_two_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="vol_window"):
            HMMRegimeConfig(vol_window=1)

    def test_when_skew_window_below_two_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="skew_window"):
            HMMRegimeConfig(skew_window=1)

    def test_when_min_observations_below_thirty_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="min_observations"):
            HMMRegimeConfig(min_observations=29)

    def test_when_min_observations_at_thirty_then_valid(self) -> None:
        cfg = HMMRegimeConfig(min_observations=30)
        assert cfg.min_observations == 30


class TestPresets:
    def test_when_for_two_regime_then_two_states(self) -> None:
        cfg = HMMRegimeConfig.for_two_regime()
        assert cfg.n_regimes == 2
        assert cfg.feature == HMMFeatureType.RETURN_VOL

    def test_when_for_three_regime_then_three_states(self) -> None:
        cfg = HMMRegimeConfig.for_three_regime()
        assert cfg.n_regimes == 3
        assert cfg.feature == HMMFeatureType.RETURN_VOL
