"""Validation and calendar params on the online configs."""

from __future__ import annotations

import pytest

from optimizer.online import (
    CovarianceForecastConfig,
    OnlinePredictConfig,
)


def test_when_warmup_size_zero_then_raises() -> None:
    with pytest.raises(ValueError, match="warmup_size must be >= 1"):
        OnlinePredictConfig(warmup_size=0)


def test_when_test_size_zero_then_raises() -> None:
    with pytest.raises(ValueError, match="test_size must be >= 1"):
        OnlinePredictConfig(test_size=0)


def test_when_purged_size_negative_then_raises() -> None:
    with pytest.raises(ValueError, match="purged_size must be >= 0"):
        OnlinePredictConfig(purged_size=-1)


def test_when_freq_offset_without_freq_then_raises() -> None:
    with pytest.raises(ValueError, match="freq_offset requires freq"):
        OnlinePredictConfig(freq_offset="2D")


def test_when_calendar_monthly_preset_then_sets_freq() -> None:
    cfg = OnlinePredictConfig.for_calendar_monthly()
    assert cfg.freq == "MS"
    assert cfg.test_size == 1


def test_when_daily_rebalance_preset_then_no_freq() -> None:
    cfg = OnlinePredictConfig.for_daily_rebalance(warmup_size=100)
    assert cfg.freq is None
    assert cfg.warmup_size == 100
    assert cfg.test_size == 1


def test_when_config_defaults_then_backward_compatible() -> None:
    cfg = OnlinePredictConfig()
    assert cfg.warmup_size == 252
    assert cfg.test_size == 1
    assert cfg.purged_size == 0
    assert cfg.n_jobs is None
    assert cfg.verbose is False


def test_when_cov_forecast_train_size_zero_then_raises() -> None:
    with pytest.raises(ValueError, match="train_size must be >= 1"):
        CovarianceForecastConfig(train_size=0)


def test_when_cov_forecast_test_size_zero_then_raises() -> None:
    with pytest.raises(ValueError, match="test_size must be >= 1"):
        CovarianceForecastConfig(test_size=0)


def test_when_cov_forecast_purged_negative_then_raises() -> None:
    with pytest.raises(ValueError, match="purged_size must be >= 0"):
        CovarianceForecastConfig(purged_size=-1)


def test_when_configs_frozen_then_immutable() -> None:
    cfg = OnlinePredictConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.warmup_size = 10  # type: ignore[misc]
