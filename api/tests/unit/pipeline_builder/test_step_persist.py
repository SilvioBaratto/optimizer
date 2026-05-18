"""Tests for ``step_persist`` (issue #710 — terminal step)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from app.services.pipeline_builder import session_store, steps

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _Assembly:
    def __init__(self):
        self.sector_mapping = {"AAPL": "TECH"}


def _result() -> SimpleNamespace:
    return SimpleNamespace(weights=pd.Series({"AAPL": 1.0}))


def _report(checklist_passed: bool, pass_count: int = 17) -> dict:
    return {
        "checklist_passed": checklist_passed,
        "pass_count": pass_count,
        "metrics": {"Portfolio (after-tax)": {"sharpe": 1.2}},
    }


def _seed(
    *,
    assembly=None,
    optimize_result=None,
    report_result=None,
    run_config: dict | None = None,
) -> str:
    sid = session_store.create_session(dict(run_config or {}))
    for key, val in (
        ("assembly", assembly),
        ("optimize_result", optimize_result),
        ("report_result", report_result),
    ):
        if val is not None:
            session_store.update_session(sid, **{key: val})
    return sid


def _fetch(sid: str) -> session_store._PipelineSession:
    s = session_store.get_session(sid)
    assert s is not None
    return s


# ---------------------------------------------------------------------------
# Precondition matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing, expected_msg",
    [
        ("assembly", "assembly not set"),
        ("optimize_result", "optimize_result not set"),
        ("report_result", "report_result not set"),
    ],
)
def test_when_required_field_missing_then_value_error(missing, expected_msg):
    kwargs = {
        "assembly": _Assembly(),
        "optimize_result": _result(),
        "report_result": _report(True),
    }
    kwargs[missing] = None
    sid = _seed(
        run_config={"persist": True, "cost_bps": 10.0},
        **kwargs,
    )
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match=expected_msg):
            steps.step_persist(sid, session, params={}, on_progress=lambda **_: None)
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# persist + pass → persisted
# ---------------------------------------------------------------------------


def test_when_persist_flag_on_and_checklist_passed_then_snapshot_called():
    assembly = _Assembly()
    result = _result()
    report = _report(True, 17)
    sid = _seed(
        assembly=assembly,
        optimize_result=result,
        report_result=report,
        run_config={"persist": True, "cost_bps": 10.0},
    )
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps._persist_research_snapshot"
        ) as mock_snap:
            out = steps.step_persist(
                sid, session, params={}, on_progress=lambda **_: None
            )
            mock_snap.assert_called_once()
            _, kwargs = mock_snap.call_args
            assert kwargs["result"] is result
            assert kwargs["assembly"] is assembly
            assert kwargs["metrics"] == report["metrics"]
            assert kwargs["cost_bps"] == 10.0
        assert out["persisted"] is True
        assert out["reason"] == "persisted"
        assert out["pass_count"] == 17
        assert out["checklist_passed"] is True
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# persist + fail → checklist_failed
# ---------------------------------------------------------------------------


def test_when_persist_flag_on_but_checklist_failed_then_not_persisted():
    sid = _seed(
        assembly=_Assembly(),
        optimize_result=_result(),
        report_result=_report(False, 15),
        run_config={"persist": True, "cost_bps": 10.0},
    )
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps._persist_research_snapshot"
        ) as mock_snap:
            out = steps.step_persist(
                sid, session, params={}, on_progress=lambda **_: None
            )
            mock_snap.assert_not_called()
        assert out["persisted"] is False
        assert out["reason"] == "checklist_failed"
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# persist flag off → persist_flag_off
# ---------------------------------------------------------------------------


def test_when_persist_flag_off_then_persist_flag_off_reason():
    sid = _seed(
        assembly=_Assembly(),
        optimize_result=_result(),
        report_result=_report(True),
        run_config={"persist": False, "cost_bps": 10.0},
    )
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps._persist_research_snapshot"
        ) as mock_snap:
            out = steps.step_persist(
                sid, session, params={}, on_progress=lambda **_: None
            )
            mock_snap.assert_not_called()
        assert out["persisted"] is False
        assert out["reason"] == "persist_flag_off"
    finally:
        session_store.delete_session(sid)


def test_when_persist_missing_from_run_config_then_default_off():
    sid = _seed(
        assembly=_Assembly(),
        optimize_result=_result(),
        report_result=_report(True),
        run_config={"cost_bps": 10.0},
    )
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps._persist_research_snapshot"
        ) as mock_snap:
            out = steps.step_persist(
                sid, session, params={}, on_progress=lambda **_: None
            )
            mock_snap.assert_not_called()
        assert out["reason"] == "persist_flag_off"
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# force_persist bypasses run_config[persist] but NOT checklist gate
# ---------------------------------------------------------------------------


def test_when_force_persist_and_checklist_passed_then_persisted():
    sid = _seed(
        assembly=_Assembly(),
        optimize_result=_result(),
        report_result=_report(True),
        run_config={"persist": False, "cost_bps": 10.0},
    )
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps._persist_research_snapshot"
        ) as mock_snap:
            out = steps.step_persist(
                sid,
                session,
                params={"force_persist": True},
                on_progress=lambda **_: None,
            )
            mock_snap.assert_called_once()
        assert out["persisted"] is True
        assert out["reason"] == "persisted"
    finally:
        session_store.delete_session(sid)


def test_when_force_persist_but_checklist_failed_then_not_persisted():
    sid = _seed(
        assembly=_Assembly(),
        optimize_result=_result(),
        report_result=_report(False, 10),
        run_config={"persist": False, "cost_bps": 10.0},
    )
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps._persist_research_snapshot"
        ) as mock_snap:
            out = steps.step_persist(
                sid,
                session,
                params={"force_persist": True},
                on_progress=lambda **_: None,
            )
            mock_snap.assert_not_called()
        assert out["persisted"] is False
        assert out["reason"] == "checklist_failed"
    finally:
        session_store.delete_session(sid)
