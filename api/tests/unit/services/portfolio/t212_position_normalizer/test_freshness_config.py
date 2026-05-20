"""Tests for FreshnessConfig + default_freshness_config (issue #794)."""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import FrozenInstanceError

import pytest

import app.config as config_module
from app.services.portfolio.t212_position_normalizer.freshness_config import (
    FreshnessConfig,
    default_freshness_config,
)


def _reload_with_env(monkeypatch: pytest.MonkeyPatch, **env: str | None) -> None:
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    importlib.reload(config_module)


class TestFreshnessConfigShape:
    def test_when_instantiated_then_is_frozen_dataclass(self) -> None:
        cfg = FreshnessConfig(stale_price_threshold_hours=24)
        assert dataclasses.is_dataclass(cfg)
        with pytest.raises(FrozenInstanceError):
            cfg.stale_price_threshold_hours = 99  # type: ignore[misc]

    def test_when_constructed_then_carries_threshold(self) -> None:
        cfg = FreshnessConfig(stale_price_threshold_hours=72)
        assert cfg.stale_price_threshold_hours == 72


class TestDefaultFreshnessConfig:
    def test_when_env_unset_then_default_is_48(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reload_with_env(monkeypatch, STALE_PRICE_THRESHOLD_HOURS=None)
        try:
            cfg = default_freshness_config()
            assert cfg.stale_price_threshold_hours == 48
        finally:
            importlib.reload(config_module)

    def test_when_env_set_to_integer_then_value_honoured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reload_with_env(monkeypatch, STALE_PRICE_THRESHOLD_HOURS="12")
        try:
            cfg = default_freshness_config()
            assert cfg.stale_price_threshold_hours == 12
        finally:
            importlib.reload(config_module)

    def test_when_env_set_to_non_integer_then_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic import ValidationError

        monkeypatch.setenv("STALE_PRICE_THRESHOLD_HOURS", "not-a-number")
        try:
            with pytest.raises(ValidationError):
                importlib.reload(config_module)
        finally:
            monkeypatch.delenv("STALE_PRICE_THRESHOLD_HOURS", raising=False)
            importlib.reload(config_module)


class TestImportTimeNoInstantiation:
    def test_when_module_imported_then_no_module_level_instance(self) -> None:
        """Factory must be the only construction site at module scope."""
        import app.services.portfolio.t212_position_normalizer.freshness_config as mod

        public = {
            name
            for name in vars(mod)
            if not name.startswith("_")
            and isinstance(vars(mod)[name], FreshnessConfig)
        }
        assert public == set()
