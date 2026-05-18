"""Tests for ``step_regime`` (issue #705 — Step 9 of the wizard)."""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest

from app.services.pipeline_builder import session_store, steps

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _Regime(Enum):
    EXPANSION = "expansion"


class _Group(Enum):
    MOMENTUM = "MOMENTUM"
    VALUE = "VALUE"


class _Assembly:
    pass


def _seed(assembly=None) -> str:
    sid = session_store.create_session({})
    if assembly is not None:
        session_store.update_session(sid, assembly=assembly)
    return sid


def _fetch(sid: str) -> session_store._PipelineSession:
    s = session_store.get_session(sid)
    assert s is not None
    return s


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------


def test_when_assembly_missing_then_value_error_raised():
    sid = _seed(assembly=None)
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="assembly not set"):
            steps.step_regime(sid, session, params={}, on_progress=lambda **_: None)
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Happy path — no persist
# ---------------------------------------------------------------------------


def test_when_classify_runs_no_persist_then_payload_and_session_populated():
    assembly = _Assembly()
    sid = _seed(assembly=assembly)
    session = _fetch(sid)

    tilts = {_Group.MOMENTUM: 1.2, _Group.VALUE: 0.8}
    try:
        with patch(
            "app.services.pipeline_builder.steps.classify_and_tilt",
            return_value=(_Regime.EXPANSION, tilts),
        ) as mock_clf:
            out = steps.step_regime(
                sid, session, params={}, on_progress=lambda **_: None
            )
            mock_clf.assert_called_once()
            args, kwargs = mock_clf.call_args
            assert args[0] is assembly
            assert kwargs.get("regime_persistence") is None

        assert out["regime"] == "expansion"
        assert out["tilts"] == {"MOMENTUM": 1.2, "VALUE": 0.8}
        assert out["persisted"] is False
        assert "Regime Preview" in out["note"]

        regime_result = _fetch(sid).regime_result
        assert regime_result["regime"] == "expansion"
        assert regime_result["tilts"] == {"MOMENTUM": 1.2, "VALUE": 0.8}
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Persist path — adapter forwards through MacroRegimeRepository
# ---------------------------------------------------------------------------


def test_when_persist_true_then_adapter_passed_and_persisted_flag_true():
    assembly = _Assembly()
    sid = _seed(assembly=assembly)
    session = _fetch(sid)
    captured: list = []
    tilts = {_Group.MOMENTUM: 1.0}

    def _fake_classify(_assembly, *, regime_persistence=None):
        captured.append(regime_persistence)
        if regime_persistence is not None:
            regime_persistence.upsert_regime_classification(
                country="US", regime="expansion"
            )
        return _Regime.EXPANSION, tilts

    fake_repo_cls = MagicMock()
    fake_repo = MagicMock()
    fake_repo_cls.return_value = fake_repo

    fake_session = MagicMock()

    @contextmanager
    def _fake_get_session():
        yield fake_session

    fake_db = MagicMock()
    fake_db.get_session = _fake_get_session

    try:
        with (
            patch(
                "app.services.pipeline_builder.steps.classify_and_tilt",
                side_effect=_fake_classify,
            ),
            patch(
                "app.services.pipeline_builder.steps.database_manager",
                new=fake_db,
            ),
            patch(
                "app.repositories.macro.macro_regime_repository.MacroRegimeRepository",
                new=fake_repo_cls,
            ),
        ):
            out = steps.step_regime(
                sid,
                session,
                params={"persist_regime": True},
                on_progress=lambda **_: None,
            )

        assert out["persisted"] is True
        adapter = captured[0]
        assert adapter is not None
        # Adapter actually invoked the repo path.
        fake_repo_cls.assert_called_once_with(fake_session)
        fake_repo.upsert_regime_classification.assert_called_once_with(
            country="US", regime="expansion"
        )
        fake_session.commit.assert_called_once()
    finally:
        session_store.delete_session(sid)


def test_when_persist_false_default_then_no_adapter_constructed():
    assembly = _Assembly()
    sid = _seed(assembly=assembly)
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps.classify_and_tilt",
            return_value=(_Regime.EXPANSION, {}),
        ) as mock_clf:
            out = steps.step_regime(
                sid, session, params={}, on_progress=lambda **_: None
            )
        _, kwargs = mock_clf.call_args
        assert kwargs["regime_persistence"] is None
        assert out["persisted"] is False
    finally:
        session_store.delete_session(sid)
