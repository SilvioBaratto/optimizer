"""Tests for ``step_screen`` (issue #699 — Step 3 of the wizard)."""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pandas as pd
import pytest

from app.services.pipeline_builder import session_store, steps

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Assembly:
    """Minimal stub exposing fields used by ``step_screen``."""

    prices: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    volumes: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    fundamentals: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    financial_statements: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)


def _seed(run_config: dict | None = None, assembly: _Assembly | None = None) -> str:
    sid = session_store.create_session(dict(run_config or {}))
    if assembly is not None:
        session_store.update_session(sid, assembly=assembly)
    return sid


def _fetch(sid: str) -> session_store._PipelineSession:
    s = session_store.get_session(sid)
    assert s is not None
    return s


def _passing(n: int) -> pd.Index:
    return pd.Index([f"T{i}" for i in range(n)])


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------


def test_when_assembly_missing_then_value_error_raised():
    sid = _seed()
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="assembly not set"):
            steps.step_screen(sid, session, params={}, on_progress=lambda **_: None)
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Default preset path
# ---------------------------------------------------------------------------


def test_when_default_preset_then_screen_investable_called():
    assembly = _Assembly()
    sid = _seed(assembly=assembly)
    session = _fetch(sid)
    try:
        with (
            patch(
                "app.services.pipeline_builder.steps.screen_investable",
                return_value=_passing(500),
            ) as mock_screen,
            patch(
                "app.services.pipeline_builder.steps.screen_universe"
            ) as mock_universe,
        ):
            out = steps.step_screen(
                sid, session, params={}, on_progress=lambda **_: None
            )
            mock_screen.assert_called_once_with(assembly)
            mock_universe.assert_not_called()

        assert out == {
            "n_investable": 500,
            "preset": "developed_markets",
            "band_warning": False,
            "band_low": 300,
            "band_high": 1500,
        }
        assert len(_fetch(sid).investable) == 500
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Non-default preset paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["broad_universe", "small_cap", "large_cap"])
def test_when_non_default_preset_then_screen_universe_called(preset):
    assembly = _Assembly()
    sid = _seed(assembly=assembly)
    session = _fetch(sid)
    sentinel_config = object()
    try:
        with (
            patch.dict(
                steps._PRESET_FACTORIES,
                {preset: lambda: sentinel_config},
                clear=False,
            ),
            patch(
                "app.services.pipeline_builder.steps.screen_universe",
                return_value=_passing(800),
            ) as mock_universe,
            patch(
                "app.services.pipeline_builder.steps.screen_investable"
            ) as mock_default,
        ):
            out = steps.step_screen(
                sid,
                session,
                params={"preset": preset},
                on_progress=lambda **_: None,
            )
            mock_default.assert_not_called()
            mock_universe.assert_called_once()
            _, kwargs = mock_universe.call_args
            assert kwargs["fundamentals"] is assembly.fundamentals
            assert kwargs["price_history"] is assembly.prices
            assert kwargs["volume_history"] is assembly.volumes
            assert kwargs["financial_statements"] is assembly.financial_statements
            assert kwargs["config"] is sentinel_config

        assert out["preset"] == preset
        assert out["n_investable"] == 800
        assert out["band_warning"] is False
    finally:
        session_store.delete_session(sid)


def test_when_unknown_preset_then_value_error():
    assembly = _Assembly()
    sid = _seed(assembly=assembly)
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="unknown preset"):
            steps.step_screen(
                sid,
                session,
                params={"preset": "bogus"},
                on_progress=lambda **_: None,
            )
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Floor + band
# ---------------------------------------------------------------------------


def test_when_floor_breached_then_runtime_error_with_message():
    assembly = _Assembly()
    sid = _seed(assembly=assembly)
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps.screen_investable",
            return_value=_passing(50),
        ):
            with pytest.raises(RuntimeError, match=r"only 50 tickers"):
                steps.step_screen(sid, session, params={}, on_progress=lambda **_: None)
    finally:
        session_store.delete_session(sid)


def test_when_band_breached_above_then_band_warning_true_no_exception():
    assembly = _Assembly()
    sid = _seed(assembly=assembly)
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps.screen_investable",
            return_value=_passing(2000),
        ):
            out = steps.step_screen(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert out["band_warning"] is True
        assert out["n_investable"] == 2000
    finally:
        session_store.delete_session(sid)


def test_when_band_breached_below_then_band_warning_true_no_exception():
    assembly = _Assembly()
    sid = _seed(assembly=assembly)
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps.screen_investable",
            return_value=_passing(250),
        ):
            out = steps.step_screen(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert out["band_warning"] is True
    finally:
        session_store.delete_session(sid)


def test_when_failure_then_investable_not_persisted():
    assembly = _Assembly()
    sid = _seed(assembly=assembly)
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps.screen_investable",
            return_value=_passing(50),
        ):
            with pytest.raises(RuntimeError):
                steps.step_screen(sid, session, params={}, on_progress=lambda **_: None)
        assert _fetch(sid).investable is None
    finally:
        session_store.delete_session(sid)
