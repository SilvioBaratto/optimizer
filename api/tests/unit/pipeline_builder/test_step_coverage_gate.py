"""Tests for ``step_coverage_gate`` (issue #704 — Step 8 of the wizard)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from app.services.pipeline_builder import session_store, steps
from optimizer.exceptions import FactorCoverageError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _is_report(significant: list[str]) -> SimpleNamespace:
    return SimpleNamespace(significant_factors=list(significant))


def _oos_result(icir: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(mean_oos_icir=pd.Series(icir))


def _seed(*, is_result=None, oos_result=None) -> str:
    sid = session_store.create_session({})
    if is_result is not None:
        session_store.update_session(sid, is_result=is_result)
    if oos_result is not None:
        session_store.update_session(sid, oos_result=oos_result)
    return sid


def _fetch(sid: str) -> session_store._PipelineSession:
    s = session_store.get_session(sid)
    assert s is not None
    return s


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_when_is_result_missing_then_value_error_raised():
    sid = _seed(is_result=None, oos_result=_oos_result({"a": 0.1}))
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="is_result not set"):
            steps.step_coverage_gate(
                sid, session, params={}, on_progress=lambda **_: None
            )
    finally:
        session_store.delete_session(sid)


def test_when_oos_result_missing_then_value_error_raised():
    sid = _seed(is_result=_is_report(["a"]), oos_result=None)
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="oos_result not set"):
            steps.step_coverage_gate(
                sid, session, params={}, on_progress=lambda **_: None
            )
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_when_three_factors_pass_both_sets_then_gate_passes():
    is_result = _is_report(["momentum", "size", "value", "quality"])
    oos_result = _oos_result(
        {"momentum": 0.4, "size": 0.2, "value": 0.1, "lowvol": 0.3, "quality": -0.1}
    )
    sid = _seed(is_result=is_result, oos_result=oos_result)
    session = _fetch(sid)
    try:
        out = steps.step_coverage_gate(
            sid, session, params={}, on_progress=lambda **_: None
        )

        assert out == {
            "passing_factors": ["momentum", "size", "value"],
            "is_only_factors": ["quality"],
            "oos_only_factors": ["lowvol"],
            "n_passing": 3,
            "min_factors": 2,
        }
        assert _fetch(sid).coverage_result == out
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Gate failure
# ---------------------------------------------------------------------------


def test_when_n_passing_below_default_min_factors_then_raises():
    is_result = _is_report(["momentum"])
    oos_result = _oos_result({"momentum": 0.4, "size": 0.2})
    sid = _seed(is_result=is_result, oos_result=oos_result)
    session = _fetch(sid)
    try:
        with pytest.raises(FactorCoverageError, match="Factor coverage gate failed"):
            steps.step_coverage_gate(
                sid, session, params={}, on_progress=lambda **_: None
            )
    finally:
        session_store.delete_session(sid)


def test_when_min_factors_param_override_then_lower_threshold_passes():
    is_result = _is_report(["momentum"])
    oos_result = _oos_result({"momentum": 0.4, "size": 0.2})
    sid = _seed(is_result=is_result, oos_result=oos_result)
    session = _fetch(sid)
    try:
        out = steps.step_coverage_gate(
            sid,
            session,
            params={"min_factors": 1},
            on_progress=lambda **_: None,
        )
        assert out["n_passing"] == 1
        assert out["min_factors"] == 1
        assert out["passing_factors"] == ["momentum"]
        assert out["oos_only_factors"] == ["size"]
    finally:
        session_store.delete_session(sid)


def test_when_gate_fails_then_session_not_updated():
    is_result = _is_report(["momentum"])
    oos_result = _oos_result({"momentum": 0.4})
    sid = _seed(is_result=is_result, oos_result=oos_result)
    session = _fetch(sid)
    try:
        with pytest.raises(FactorCoverageError):
            steps.step_coverage_gate(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert _fetch(sid).coverage_result is None
    finally:
        session_store.delete_session(sid)
