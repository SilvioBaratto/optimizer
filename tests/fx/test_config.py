"""Tests for FxConfig validation and presets."""

from __future__ import annotations

import dataclasses

import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.fx import BaseCurrency, FxConfig, FxConversionMode


class TestFxConfigValidation:
    """__post_init__ validation."""

    def test_negative_fill_limit_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="fill_limit"):
            FxConfig(fill_limit=-1)

    def test_zero_fill_limit_allowed(self) -> None:
        cfg = FxConfig(fill_limit=0)
        assert cfg.fill_limit == 0

    def test_is_frozen(self) -> None:
        cfg = FxConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.fill_limit = 10  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        # Frozen + primitive/enum fields → hashable (serialisable config).
        assert hash(FxConfig.for_eur_base()) == hash(FxConfig.for_eur_base())


class TestFxConfigPresets:
    """Factory-style class-method presets."""

    def test_for_eur_base(self) -> None:
        cfg = FxConfig.for_eur_base()
        assert cfg.base_currency == BaseCurrency.EUR
        assert cfg.mode == FxConversionMode.TO_BASE

    def test_for_gbp_base(self) -> None:
        cfg = FxConfig.for_gbp_base()
        assert cfg.base_currency == BaseCurrency.GBP
        assert cfg.mode == FxConversionMode.TO_BASE

    def test_for_usd_base(self) -> None:
        cfg = FxConfig.for_usd_base()
        assert cfg.base_currency == BaseCurrency.USD
        assert cfg.mode == FxConversionMode.TO_BASE

    def test_for_decomposition(self) -> None:
        cfg = FxConfig.for_decomposition(BaseCurrency.GBP)
        assert cfg.mode == FxConversionMode.DECOMPOSE
        assert cfg.base_currency == BaseCurrency.GBP

    def test_for_strict_conversion(self) -> None:
        cfg = FxConfig.for_strict_conversion()
        assert cfg.strict is True
        assert cfg.mode == FxConversionMode.TO_BASE
