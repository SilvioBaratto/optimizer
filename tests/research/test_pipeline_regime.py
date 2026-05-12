"""Tests for research.pipeline._regime — Step 6 regime classification."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest


def _stub_assembly() -> MagicMock:
    assembly = MagicMock()
    assembly.macro_data = pd.DataFrame()
    assembly.fred_data = pd.DataFrame()
    assembly.te_observations = pd.DataFrame()
    assembly.sentiment_data = pd.DataFrame()
    return assembly


def _patch_regime_steps(
    monkeypatch: pytest.MonkeyPatch,
    regime_value: object,
    tilts: dict[object, float] | None = None,
) -> None:
    monkeypatch.setattr(
        "research.pipeline._regime.assemble_regime_data",
        lambda **_: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "research.pipeline._regime.classify_regime",
        lambda _: regime_value,
    )
    monkeypatch.setattr(
        "research.pipeline._regime.apply_regime_tilts",
        lambda group_weights, regime, config: tilts or {},
    )


class TestClassifyAndTiltNoopWhenNoPersistence:
    def test_when_no_regime_persistence_then_returns_regime_and_tilts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from optimizer.factors._config import FactorGroupType, MacroRegime
        from research.pipeline._regime import classify_and_tilt

        _patch_regime_steps(
            monkeypatch,
            regime_value=MacroRegime.EXPANSION,
            tilts={FactorGroupType.VALUE: 1.2, FactorGroupType.MOMENTUM: 0.8},
        )

        regime, tilts = classify_and_tilt(_stub_assembly())

        assert regime == MacroRegime.EXPANSION
        assert isinstance(tilts, dict)

    def test_when_regime_persistence_none_then_no_persistence_and_no_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from optimizer.factors._config import MacroRegime
        from research.pipeline._regime import classify_and_tilt

        _patch_regime_steps(monkeypatch, regime_value=MacroRegime.EXPANSION)

        # Must not raise even though no persistence backend is wired
        regime, _tilts = classify_and_tilt(_stub_assembly(), regime_persistence=None)

        assert regime == MacroRegime.EXPANSION


class TestClassifyAndTiltWithProtocol:
    def test_when_persistence_provided_then_upsert_called_with_us_and_regime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from optimizer.factors._config import MacroRegime
        from research.pipeline._regime import classify_and_tilt

        _patch_regime_steps(monkeypatch, regime_value=MacroRegime.SLOWDOWN)

        persistence = MagicMock()

        classify_and_tilt(_stub_assembly(), regime_persistence=persistence)

        persistence.upsert_regime_classification.assert_called_once_with(
            country="US", regime=MacroRegime.SLOWDOWN.value
        )

    def test_when_regime_persistence_provided_then_cache_called_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from optimizer.factors._config import MacroRegime
        from research.pipeline._regime import classify_and_tilt

        _patch_regime_steps(monkeypatch, regime_value=MacroRegime.RECESSION)

        persistence = MagicMock()

        classify_and_tilt(_stub_assembly(), regime_persistence=persistence)

        assert persistence.upsert_regime_classification.call_count == 1


class TestPipelineRegimeImportIsolation:
    def test_when_pipeline_regime_imported_then_app_repositories_not_loaded(
        self,
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import research.pipeline._regime; import sys; "
                    "matches = [k for k in sys.modules if k.startswith('app.repositories')]; "
                    "assert not matches, f'app.repositories leaked: {matches}'"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"app.repositories leaked on import of research.pipeline._regime.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_when_pipeline_regime_imported_then_app_database_not_loaded(
        self,
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import research.pipeline._regime; import sys; "
                    "leaks = [k for k in sys.modules if 'app.repositories' in k "
                    "or k == 'app.database']; "
                    "assert not leaks, f'app coupling leaked: {leaks}'"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"app.repositories or app.database leaked on import.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
