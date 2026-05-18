"""Tests for ``step_validate_is`` (issue #702 — Step 6 of the wizard)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from app.services.pipeline_builder import session_store, steps

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ICResult:
    """Mirror of ``optimizer.factors._validation.ICResult`` fields."""

    factor_name: str
    mean_ic: float
    ic_std: float
    t_stat: float
    p_value: float
    significant: bool


def _report(
    *,
    ic_results=None,
    significant_factors=None,
    vif_scores=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        ic_results=ic_results or [],
        significant_factors=significant_factors or [],
        vif_scores=vif_scores,
    )


def _seed(factor_scores_dict=None, returns_history=None) -> str:
    sid = session_store.create_session({})
    if factor_scores_dict is not None:
        session_store.update_session(sid, factor_scores_dict=factor_scores_dict)
    if returns_history is not None:
        session_store.update_session(sid, returns_history=returns_history)
    return sid


def _fetch(sid: str) -> session_store._PipelineSession:
    s = session_store.get_session(sid)
    assert s is not None
    return s


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_when_factor_scores_dict_missing_then_value_error_raised():
    sid = _seed(factor_scores_dict=None, returns_history=pd.DataFrame())
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="factor_scores_dict not set"):
            steps.step_validate_is(
                sid, session, params={}, on_progress=lambda **_: None
            )
    finally:
        session_store.delete_session(sid)


def test_when_returns_history_missing_then_value_error_raised():
    sid = _seed(factor_scores_dict={"a": pd.DataFrame()}, returns_history=None)
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="returns_history not set"):
            steps.step_validate_is(
                sid, session, params={}, on_progress=lambda **_: None
            )
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_when_validate_is_runs_then_session_and_payload_populated():
    fsd = {"momentum": pd.DataFrame()}
    rh = pd.DataFrame({"r": [0.1]})
    sid = _seed(factor_scores_dict=fsd, returns_history=rh)
    session = _fetch(sid)

    report = _report(
        ic_results=[
            _ICResult("momentum", 0.05, 0.10, 2.0, 0.04, True),
            _ICResult("size", 0.01, 0.15, 0.5, 0.6, False),
        ],
        significant_factors=["momentum"],
        vif_scores=pd.Series({"momentum": 1.1, "size": 6.5}),
    )

    try:
        with patch(
            "app.services.pipeline_builder.steps.validate_is",
            return_value=report,
        ) as mock_validate:
            out = steps.step_validate_is(
                sid, session, params={}, on_progress=lambda **_: None
            )
            mock_validate.assert_called_once_with(fsd, rh)

        assert out["n_significant"] == 1
        assert out["significant_factors"] == ["momentum"]
        assert out["high_vif_factors"] == ["size"]
        assert out["config"] == {
            "newey_west_lags": 4,
            "fdr_alpha": 0.10,
            "t_stat_threshold": 1.645,
        }
        # ic_results carries one entry per ICResult.
        assert len(out["ic_results"]) == 2
        first = out["ic_results"][0]
        assert first["factor_name"] == "momentum"
        assert first["mean_ic"] == 0.05
        assert first["t_stat"] == 2.0
        assert first["significant"] is True

        assert _fetch(sid).is_result is report
    finally:
        session_store.delete_session(sid)


def test_when_vif_scores_none_then_high_vif_factors_empty():
    fsd = {"a": pd.DataFrame()}
    rh = pd.DataFrame()
    sid = _seed(factor_scores_dict=fsd, returns_history=rh)
    session = _fetch(sid)
    report = _report(
        ic_results=[_ICResult("a", 0.0, 0.0, 0.0, 1.0, False)],
        significant_factors=[],
        vif_scores=None,
    )
    try:
        with patch(
            "app.services.pipeline_builder.steps.validate_is",
            return_value=report,
        ):
            out = steps.step_validate_is(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert out["high_vif_factors"] == []
    finally:
        session_store.delete_session(sid)


def test_when_no_factor_above_vif_5_then_high_vif_factors_empty():
    fsd = {"a": pd.DataFrame()}
    rh = pd.DataFrame()
    sid = _seed(factor_scores_dict=fsd, returns_history=rh)
    session = _fetch(sid)
    report = _report(
        ic_results=[],
        significant_factors=[],
        vif_scores=pd.Series({"a": 2.1, "b": 4.9}),
    )
    try:
        with patch(
            "app.services.pipeline_builder.steps.validate_is",
            return_value=report,
        ):
            out = steps.step_validate_is(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert out["high_vif_factors"] == []
    finally:
        session_store.delete_session(sid)
