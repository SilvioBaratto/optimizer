"""Tests for DatabaseManager.get_session lazy-init guard (issues #562, #563)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.database import DatabaseManager


def _arm_initialized(mgr: DatabaseManager) -> None:
    """Pre-arm a fresh manager as if initialize() had already run."""
    mgr._is_initialized = True
    mgr._session_factory = MagicMock(return_value=MagicMock())


def _make_init_side_effect(mgr: DatabaseManager):
    """Return a side-effect that arms *mgr* on call (mimics real initialize)."""

    def _fake_init() -> None:
        _arm_initialized(mgr)

    return _fake_init


def test_get_session_calls_initialize_when_not_initialized() -> None:
    mgr = DatabaseManager()
    with patch.object(
        mgr, "initialize", side_effect=_make_init_side_effect(mgr),
    ) as init_mock:
        with mgr.get_session():
            pass
    assert init_mock.call_count == 1


def test_get_session_does_not_call_initialize_when_already_initialized() -> None:
    mgr = DatabaseManager()
    _arm_initialized(mgr)
    with patch.object(mgr, "initialize") as init_mock:
        with mgr.get_session():
            pass
    init_mock.assert_not_called()


def test_get_session_propagates_initialize_failure() -> None:
    mgr = DatabaseManager()
    with patch.object(mgr, "initialize", side_effect=RuntimeError("db down")):
        with pytest.raises(RuntimeError, match="db down"):
            with mgr.get_session():
                pass
