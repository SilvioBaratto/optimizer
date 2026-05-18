"""Tests for ``step_optimize`` (issue #706 — Step 10 of the wizard)."""

from __future__ import annotations

import logging
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
        self.sector_mapping = {
            "AAPL": "TECH",
            "MSFT": "TECH",
            "XOM": "ENERGY",
        }


def _seed(
    *,
    assembly=None,
    investable=None,
    oos_result=None,
    run_config: dict | None = None,
) -> str:
    cfg = dict(run_config or {})
    sid = session_store.create_session(cfg)
    if assembly is not None:
        session_store.update_session(sid, assembly=assembly)
    if investable is not None:
        session_store.update_session(sid, investable=investable)
    if oos_result is not None:
        session_store.update_session(sid, oos_result=oos_result)
    return sid


def _fetch(sid: str) -> session_store._PipelineSession:
    s = session_store.get_session(sid)
    assert s is not None
    return s


def _oos(per_fold_ic: pd.DataFrame | None = None) -> SimpleNamespace:
    if per_fold_ic is None:
        per_fold_ic = pd.DataFrame({"momentum": [0.05, 0.06]})
    return SimpleNamespace(per_fold_ic=per_fold_ic)


def _opt_result(
    weights: pd.Series,
    is_sharpe: float = 1.2,
    net_sharpe: float | None = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        weights=weights,
        summary={"sharpe_ratio": is_sharpe},
        net_sharpe_ratio=net_sharpe,
        net_returns=None,
    )


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_field, expected_msg",
    [
        ("assembly", "assembly not set"),
        ("investable", "investable not set"),
        ("oos_result", "oos_result not set"),
    ],
)
def test_when_required_field_missing_then_value_error(missing_field, expected_msg):
    kwargs = {
        "assembly": _Assembly(),
        "investable": pd.Index(["AAPL"]),
        "oos_result": _oos(),
    }
    kwargs[missing_field] = None
    sid = _seed(**kwargs)
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match=expected_msg):
            steps.step_optimize(sid, session, params={}, on_progress=lambda **_: None)
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_when_optimize_succeeds_then_session_and_payload_returned():
    assembly = _Assembly()
    investable = pd.Index(["AAPL", "MSFT", "XOM"])
    oos = _oos()
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "XOM": 0.2})
    result = _opt_result(weights, is_sharpe=1.4, net_sharpe=1.1)

    run_config = {
        "n_selected": 20,
        "cost_bps": 12.5,
        "country_map": {"AAPL": "US", "MSFT": "US", "XOM": "US"},
    }
    sid = _seed(
        assembly=assembly,
        investable=investable,
        oos_result=oos,
        run_config=run_config,
    )
    session = _fetch(sid)

    try:
        with patch(
            "app.services.pipeline_builder.steps.optimize_portfolio",
            return_value=result,
        ) as mock_opt:
            out = steps.step_optimize(
                sid,
                session,
                params={"robust": True, "seed": 42},
                on_progress=lambda **_: None,
            )
            args, kwargs = mock_opt.call_args
            assert args[0] is assembly
            assert args[1] is investable
            assert args[2] is oos.per_fold_ic
            assert kwargs["n_selected"] == 20
            assert kwargs["cost_bps"] == 12.5
            assert kwargs["country_map"] == run_config["country_map"]
            assert kwargs["robust"] is True
            assert kwargs["uncertainty_level"] == 0.95
            assert kwargs["seed"] == 42

        assert out["n_selected"] == 3
        assert out["is_sharpe"] == 1.4
        assert out["net_sharpe"] == 1.1
        assert out["hockey_stick_warning"] is False
        assert out["weights"] == [
            {"ticker": "AAPL", "weight": 0.5},
            {"ticker": "MSFT", "weight": 0.3},
            {"ticker": "XOM", "weight": 0.2},
        ]
        assert out["sector_breakdown"] == {"TECH": 0.8, "ENERGY": 0.2}
        assert out["country_breakdown"] == {"US": 1.0}
        assert _fetch(sid).optimize_result is result
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_when_no_country_map_in_run_config_then_country_breakdown_empty():
    assembly = _Assembly()
    investable = pd.Index(["AAPL"])
    weights = pd.Series({"AAPL": 1.0})
    sid = _seed(
        assembly=assembly,
        investable=investable,
        oos_result=_oos(),
        run_config={"n_selected": 20, "cost_bps": 10.0},
    )
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps.optimize_portfolio",
            return_value=_opt_result(weights),
        ) as mock_opt:
            out = steps.step_optimize(
                sid, session, params={}, on_progress=lambda **_: None
            )
        _, kwargs = mock_opt.call_args
        assert kwargs["country_map"] == {}
        assert kwargs["robust"] is False
        assert kwargs["seed"] is None
        assert out["country_breakdown"] == {}
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Progress phases
# ---------------------------------------------------------------------------


def test_when_step_runs_then_coarse_phases_emitted_in_order():
    assembly = _Assembly()
    investable = pd.Index(["AAPL"])
    weights = pd.Series({"AAPL": 1.0})
    sid = _seed(
        assembly=assembly,
        investable=investable,
        oos_result=_oos(),
        run_config={"n_selected": 20, "cost_bps": 10.0},
    )
    session = _fetch(sid)
    phases: list[str] = []

    def on_progress(**kwargs):
        if "phase" in kwargs:
            phases.append(kwargs["phase"])

    try:
        with patch(
            "app.services.pipeline_builder.steps.optimize_portfolio",
            return_value=_opt_result(weights),
        ):
            steps.step_optimize(sid, session, params={}, on_progress=on_progress)
        assert phases[:4] == ["scoring", "selecting", "walk-forward", "optimizing"]
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Hockey-stick detection via logger capture
# ---------------------------------------------------------------------------


def test_when_hockey_stick_warning_logged_then_payload_flag_true():
    assembly = _Assembly()
    investable = pd.Index(["AAPL"])
    weights = pd.Series({"AAPL": 1.0})
    sid = _seed(
        assembly=assembly,
        investable=investable,
        oos_result=_oos(),
        run_config={"n_selected": 20, "cost_bps": 10.0},
    )
    session = _fetch(sid)

    def _emit_warning(*args, **kwargs):
        logging.getLogger("research.optimization._rebalance").warning(
            "[WARN] hockey-stick — outperformance concentrated in period 1 (sharpes=[2.0, -0.5, 0.1])"
        )
        return _opt_result(weights)

    try:
        with patch(
            "app.services.pipeline_builder.steps.optimize_portfolio",
            side_effect=_emit_warning,
        ):
            out = steps.step_optimize(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert out["hockey_stick_warning"] is True
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Solver-failure propagation
# ---------------------------------------------------------------------------


def test_when_optimize_raises_runtime_error_then_propagates():
    assembly = _Assembly()
    investable = pd.Index(["AAPL"])
    sid = _seed(
        assembly=assembly,
        investable=investable,
        oos_result=_oos(),
        run_config={"n_selected": 20, "cost_bps": 10.0},
    )
    session = _fetch(sid)
    err = RuntimeError("solver infeasible")
    try:
        with patch(
            "app.services.pipeline_builder.steps.optimize_portfolio",
            side_effect=err,
        ):
            with pytest.raises(RuntimeError) as exc:
                steps.step_optimize(
                    sid, session, params={}, on_progress=lambda **_: None
                )
            assert exc.value is err
        assert _fetch(sid).optimize_result is None
    finally:
        session_store.delete_session(sid)
