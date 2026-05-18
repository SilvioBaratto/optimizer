"""Tests for ``step_validate_oos`` (issue #703 — Step 7 of the wizard)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from app.services.pipeline_builder import session_store, steps


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


def _result(
    *, n_folds: int, mean_ic: dict[str, float], mean_icir: dict[str, float]
) -> SimpleNamespace:
    return SimpleNamespace(
        n_folds=n_folds,
        mean_oos_ic=pd.Series(mean_ic),
        mean_oos_icir=pd.Series(mean_icir),
    )


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_when_factor_scores_dict_missing_then_value_error_raised():
    sid = _seed(factor_scores_dict=None, returns_history=pd.DataFrame())
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="factor_scores_dict not set"):
            steps.step_validate_oos(
                sid, session, params={}, on_progress=lambda **_: None
            )
    finally:
        session_store.delete_session(sid)


def test_when_returns_history_missing_then_value_error_raised():
    sid = _seed(factor_scores_dict={"a": pd.DataFrame()}, returns_history=None)
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="returns_history not set"):
            steps.step_validate_oos(
                sid, session, params={}, on_progress=lambda **_: None
            )
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_when_validate_oos_succeeds_then_session_and_sorted_payload_returned():
    fsd = {"momentum": pd.DataFrame()}
    rh = pd.DataFrame({"r": [0.1]})
    sid = _seed(factor_scores_dict=fsd, returns_history=rh)
    session = _fetch(sid)

    result = _result(
        n_folds=6,
        mean_ic={"momentum": 0.05, "size": 0.10, "value": -0.02},
        mean_icir={"momentum": 0.4, "size": 0.7, "value": -0.1},
    )

    try:
        with patch(
            "app.services.pipeline_builder.steps.validate_oos",
            return_value=result,
        ) as mock_validate:
            out = steps.step_validate_oos(
                sid, session, params={}, on_progress=lambda **_: None
            )
            mock_validate.assert_called_once_with(fsd, rh)

        assert out["n_folds"] == 6
        assert out["config"] == {
            "train_periods": 8,
            "val_periods": 4,
            "step_periods": 2,
        }
        # Sorted by oos_mean_ic desc.
        names = [r["factor_name"] for r in out["oos_results"]]
        assert names == ["size", "momentum", "value"]
        assert out["oos_results"][0] == {
            "factor_name": "size",
            "oos_mean_ic": 0.10,
            "oos_icir": 0.7,
        }
        assert _fetch(sid).oos_result is result
    finally:
        session_store.delete_session(sid)


def test_when_factor_missing_from_icir_then_payload_icir_is_none():
    fsd = {"a": pd.DataFrame()}
    rh = pd.DataFrame()
    sid = _seed(factor_scores_dict=fsd, returns_history=rh)
    session = _fetch(sid)
    result = _result(
        n_folds=4,
        mean_ic={"a": 0.03, "b": 0.01},
        mean_icir={"a": 0.2},  # b absent
    )
    try:
        with patch(
            "app.services.pipeline_builder.steps.validate_oos",
            return_value=result,
        ):
            out = steps.step_validate_oos(
                sid, session, params={}, on_progress=lambda **_: None
            )
        b_record = next(r for r in out["oos_results"] if r["factor_name"] == "b")
        assert b_record["oos_icir"] is None
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# n_folds == 0 abort
# ---------------------------------------------------------------------------


def test_when_validate_oos_raises_n_folds_zero_then_message_propagates_verbatim():
    sid = _seed(
        factor_scores_dict={"a": pd.DataFrame()}, returns_history=pd.DataFrame()
    )
    session = _fetch(sid)
    msg = (
        "OOS validation produced 0 folds "
        "(train_periods=8, val_periods=4, step_periods=2). "
        "Increase history or reduce train_periods."
    )
    err = RuntimeError(msg)
    try:
        with patch(
            "app.services.pipeline_builder.steps.validate_oos",
            side_effect=err,
        ):
            with pytest.raises(RuntimeError) as exc:
                steps.step_validate_oos(
                    sid, session, params={}, on_progress=lambda **_: None
                )
            assert exc.value is err
            assert str(exc.value) == msg
    finally:
        session_store.delete_session(sid)


def test_when_validate_oos_raises_then_session_not_updated():
    sid = _seed(
        factor_scores_dict={"a": pd.DataFrame()}, returns_history=pd.DataFrame()
    )
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps.validate_oos",
            side_effect=RuntimeError("0 folds"),
        ):
            with pytest.raises(RuntimeError):
                steps.step_validate_oos(
                    sid, session, params={}, on_progress=lambda **_: None
                )
        assert _fetch(sid).oos_result is None
    finally:
        session_store.delete_session(sid)
