"""Tests for research.pipeline._regime — Step 6 regime classification."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


class TestClassifyAndTilt:
    def test_when_no_db_manager_then_skips_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from optimizer.factors._config import FactorGroupType, MacroRegime
        from research.pipeline._regime import classify_and_tilt

        assembly = MagicMock()
        assembly.macro_data = pd.DataFrame()
        assembly.fred_data = pd.DataFrame()
        assembly.te_observations = pd.DataFrame()
        assembly.sentiment_data = pd.DataFrame()

        monkeypatch.setattr(
            "research.pipeline._regime.assemble_regime_data",
            lambda **_: pd.DataFrame(),
        )
        monkeypatch.setattr(
            "research.pipeline._regime.classify_regime",
            lambda _: MacroRegime.EXPANSION,
        )
        monkeypatch.setattr(
            "research.pipeline._regime.apply_regime_tilts",
            lambda group_weights, regime, config: {
                FactorGroupType.VALUE: 1.2,
                FactorGroupType.MOMENTUM: 0.8,
            },
        )

        regime, tilts = classify_and_tilt(assembly)
        assert regime == MacroRegime.EXPANSION
        assert isinstance(tilts, dict)

    def test_when_db_manager_provided_then_caches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from optimizer.factors._config import MacroRegime
        from research.pipeline._regime import classify_and_tilt

        assembly = MagicMock()
        assembly.macro_data = pd.DataFrame()
        assembly.fred_data = pd.DataFrame()
        assembly.te_observations = pd.DataFrame()
        assembly.sentiment_data = pd.DataFrame()

        db_manager = MagicMock()

        monkeypatch.setattr(
            "research.pipeline._regime.assemble_regime_data",
            lambda **_: pd.DataFrame(),
        )
        monkeypatch.setattr(
            "research.pipeline._regime.classify_regime",
            lambda _: MacroRegime.SLOWDOWN,
        )
        monkeypatch.setattr(
            "research.pipeline._regime.apply_regime_tilts",
            lambda group_weights, regime, config: {},
        )
        cache_called = []

        def _fake_cache(db_mgr, regime):
            cache_called.append(True)

        monkeypatch.setattr(
            "research.pipeline._regime._cache_regime_classification",
            _fake_cache,
        )

        classify_and_tilt(assembly, db_manager=db_manager)
        assert len(cache_called) == 1
