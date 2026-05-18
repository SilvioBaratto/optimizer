"""Tests for ``step_report`` (issue #709 — Step 13)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from app.services.pipeline_builder import session_store, steps

_TMP = Path(tempfile.gettempdir())

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _Assembly:
    def __init__(self, assembly_hash: str = "deadbeef"):
        self.assembly_hash = assembly_hash
        self.sector_mapping = {}


def _optimize_result() -> SimpleNamespace:
    return SimpleNamespace(weights=pd.Series({"AAPL": 1.0}))


def _is_report() -> SimpleNamespace:
    return SimpleNamespace(significant_factors=["momentum"])


def _oos_result() -> SimpleNamespace:
    return SimpleNamespace(per_fold_ic=pd.DataFrame({"momentum": [0.05]}))


def _seed(
    *,
    assembly=None,
    optimize_result=None,
    cost_result=None,
    is_result=None,
    oos_result=None,
    regime_result=None,
    run_config: dict | None = None,
) -> str:
    sid = session_store.create_session(dict(run_config or {}))
    updates = {
        "assembly": assembly,
        "optimize_result": optimize_result,
        "cost_result": cost_result,
        "is_result": is_result,
        "oos_result": oos_result,
        "regime_result": regime_result,
    }
    for key, val in updates.items():
        if val is not None:
            session_store.update_session(sid, **{key: val})
    return sid


def _fetch(sid: str) -> session_store._PipelineSession:
    s = session_store.get_session(sid)
    assert s is not None
    return s


def _checklist_rules(pass_count: int, total: int = 17) -> list[dict]:
    return [{"rule": i, "passed": i <= pass_count} for i in range(1, total + 1)]


def _metrics() -> dict:
    return {
        "Portfolio (after-tax)": {"sharpe": 1.2, "ann_return": 0.08},
        "SPY (benchmark)": {"sharpe": 0.9, "ann_return": 0.06},
    }


# ---------------------------------------------------------------------------
# Precondition matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing, expected_msg",
    [
        ("assembly", "assembly not set"),
        ("optimize_result", "optimize_result not set"),
        ("cost_result", "cost_result not set"),
    ],
)
def test_when_required_field_missing_then_value_error(missing, expected_msg):
    kwargs = {
        "assembly": _Assembly(),
        "optimize_result": _optimize_result(),
        "cost_result": {"cost_bps_actual": 18.0},
    }
    kwargs[missing] = None
    sid = _seed(
        run_config={"cost_bps": 10.0, "tax_rate": 0.26},
        **kwargs,
    )
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match=expected_msg):
            steps.step_report(sid, session, params={}, on_progress=lambda **_: None)
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Happy path — 17/17 pass
# ---------------------------------------------------------------------------


def test_when_pass_count_17_then_checklist_passed_true():
    assembly = _Assembly()
    result = _optimize_result()
    rules = _checklist_rules(17)
    metrics = _metrics()
    chart_paths = [_TMP / "chart1.png", _TMP / "chart2.png"]

    sid = _seed(
        assembly=assembly,
        optimize_result=result,
        cost_result={"cost_bps_actual": 18.0},
        is_result=_is_report(),
        oos_result=_oos_result(),
        run_config={
            "cost_bps": 10.0,
            "tax_rate": 0.26,
            "country_map": {"AAPL": "United States"},
        },
    )
    session = _fetch(sid)
    try:
        with (
            patch(
                "app.services.pipeline_builder.steps.report_performance",
                return_value=(17, rules, metrics, chart_paths),
            ) as mock_report,
            patch(
                "app.services.pipeline_builder.steps._render_research_report",
                return_value=_TMP / "report.md",
            ) as mock_render,
        ):
            out = steps.step_report(
                sid, session, params={}, on_progress=lambda **_: None
            )
            mock_report.assert_called_once()
            mock_render.assert_called_once()

            args, kwargs = mock_report.call_args
            assert args[0] is result
            assert args[1] is assembly
            assert args[2] == {"AAPL": "United States"}
            assert kwargs["cost_bps_actual"] == 18.0
            assert kwargs["cost_bps"] == 10.0
            assert kwargs["tax_rate"] == 0.26

        assert out["pass_count"] == 17
        assert out["checklist_total"] == 17
        assert out["checklist_passed"] is True
        assert out["checklist_rules"] == rules
        assert "Portfolio (after-tax)" in out["metrics"]
        assert "output_dir" in out
        assert out["output_dir"].startswith("/")
        assert out["artifact_paths"]["report_md"].endswith("/report.md")
        assert out["artifact_paths"]["weights_csv"].endswith("/weights.csv")
        assert out["artifact_paths"]["metrics_json"].endswith("/metrics.json")
        assert out["artifact_paths"]["checklist_json"].endswith("/checklist.json")
        assert all(p.startswith("/") for p in out["chart_paths"])
        assert _fetch(sid).report_result == out
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Partial fail — 15/17 → checklist_passed False, no exception
# ---------------------------------------------------------------------------


def test_when_pass_count_below_17_then_checklist_passed_false_no_raise():
    assembly = _Assembly()
    rules = _checklist_rules(15)
    sid = _seed(
        assembly=assembly,
        optimize_result=_optimize_result(),
        cost_result={"cost_bps_actual": 12.0},
        is_result=_is_report(),
        oos_result=_oos_result(),
        run_config={"cost_bps": 10.0, "tax_rate": 0.26},
    )
    session = _fetch(sid)
    try:
        with (
            patch(
                "app.services.pipeline_builder.steps.report_performance",
                return_value=(15, rules, _metrics(), []),
            ),
            patch(
                "app.services.pipeline_builder.steps._render_research_report",
                return_value=_TMP / "report.md",
            ),
        ):
            out = steps.step_report(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert out["pass_count"] == 15
        assert out["checklist_passed"] is False
        assert out["checklist_total"] == 17
    finally:
        session_store.delete_session(sid)


def test_when_empty_checklist_then_checklist_passed_false():
    assembly = _Assembly()
    sid = _seed(
        assembly=assembly,
        optimize_result=_optimize_result(),
        cost_result={"cost_bps_actual": 0.0},
        is_result=_is_report(),
        oos_result=_oos_result(),
        run_config={"cost_bps": 10.0, "tax_rate": 0.26},
    )
    session = _fetch(sid)
    try:
        with (
            patch(
                "app.services.pipeline_builder.steps.report_performance",
                return_value=(0, [], {}, []),
            ),
            patch(
                "app.services.pipeline_builder.steps._render_research_report",
                return_value=_TMP / "report.md",
            ),
        ):
            out = steps.step_report(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert out["checklist_passed"] is False
        assert out["pass_count"] == 0
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Tempdir per session
# ---------------------------------------------------------------------------


def test_when_step_runs_then_tempdir_prefix_includes_session_hex():
    assembly = _Assembly()
    sid = _seed(
        assembly=assembly,
        optimize_result=_optimize_result(),
        cost_result={"cost_bps_actual": 18.0},
        is_result=_is_report(),
        oos_result=_oos_result(),
        run_config={"cost_bps": 10.0, "tax_rate": 0.26},
    )
    session = _fetch(sid)
    captured: dict = {}

    def _capture_report(*args, **kwargs):
        captured["output_dir"] = kwargs["output_dir"]
        return (17, _checklist_rules(17), _metrics(), [])

    try:
        with (
            patch(
                "app.services.pipeline_builder.steps.report_performance",
                side_effect=_capture_report,
            ),
            patch(
                "app.services.pipeline_builder.steps._render_research_report",
                return_value=_TMP / "report.md",
            ),
        ):
            out = steps.step_report(
                sid, session, params={}, on_progress=lambda **_: None
            )
        prefix = f"pipeline_{sid[:8]}_"
        assert prefix in captured["output_dir"].name
        assert prefix in out["output_dir"]
    finally:
        session_store.delete_session(sid)
