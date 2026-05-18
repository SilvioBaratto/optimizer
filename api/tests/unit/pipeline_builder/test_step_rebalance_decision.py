"""Tests for ``step_rebalance_decision`` (issue #707 — Step 11)."""

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
    def __init__(self, sector_mapping: dict[str, str] | None = None):
        self.sector_mapping = sector_mapping or {}


def _result(weights: pd.Series) -> SimpleNamespace:
    return SimpleNamespace(weights=weights)


def _seed(
    *,
    optimize_result=None,
    assembly: _Assembly | None = None,
) -> str:
    sid = session_store.create_session({})
    if optimize_result is not None:
        session_store.update_session(sid, optimize_result=optimize_result)
    if assembly is not None:
        session_store.update_session(sid, assembly=assembly)
    return sid


def _fetch(sid: str) -> session_store._PipelineSession:
    s = session_store.get_session(sid)
    assert s is not None
    return s


def _make_weights(n: int) -> pd.Series:
    return pd.Series({f"T{i}": 1.0 / n for i in range(n)})


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------


def test_when_optimize_result_missing_then_value_error_raised():
    sid = _seed(optimize_result=None, assembly=_Assembly())
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="optimize_result not set"):
            steps.step_rebalance_decision(
                sid, session, params={}, on_progress=lambda **_: None
            )
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------


def test_when_no_prev_weights_then_cold_start_returned():
    weights = _make_weights(20)
    sid = _seed(optimize_result=_result(weights), assembly=_Assembly())
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps._write_last_review_date"
        ) as mock_write:
            out = steps.step_rebalance_decision(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert out["decision"] is False
        assert out["reason"] == "cold_start"
        # Cold start also writes the review date.
        mock_write.assert_called_once()
    finally:
        session_store.delete_session(sid)


def test_when_force_cold_start_then_prev_weights_ignored():
    weights = _make_weights(20)
    sid = _seed(optimize_result=_result(weights), assembly=_Assembly())
    session = _fetch(sid)
    try:
        with patch("app.services.pipeline_builder.steps._write_last_review_date"):
            out = steps.step_rebalance_decision(
                sid,
                session,
                params={
                    "force_cold_start": True,
                    "prev_weights": [0.5, 0.5],
                },
                on_progress=lambda **_: None,
            )
        assert out["reason"] == "cold_start"
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Normal path
# ---------------------------------------------------------------------------


def test_when_prev_weights_supplied_then_decide_rebalance_invoked_and_write_called():
    weights = _make_weights(20)
    sid = _seed(optimize_result=_result(weights), assembly=_Assembly())
    session = _fetch(sid)
    try:
        with (
            patch(
                "app.services.pipeline_builder.steps._decide_rebalance",
                return_value=(True, "calendar_review"),
            ) as mock_dec,
            patch(
                "app.services.pipeline_builder.steps._write_last_review_date"
            ) as mock_write,
        ):
            out = steps.step_rebalance_decision(
                sid,
                session,
                params={"prev_weights": [0.5] * 20},
                on_progress=lambda **_: None,
            )
            mock_dec.assert_called_once()
            _, kwargs = mock_dec.call_args
            assert kwargs["prev_weights"] is not None
            mock_write.assert_called_once()
        assert out["decision"] is True
        assert out["reason"] == "calendar_review"
    finally:
        session_store.delete_session(sid)


def test_when_decision_false_and_not_cold_start_then_no_write():
    weights = _make_weights(20)
    sid = _seed(optimize_result=_result(weights), assembly=_Assembly())
    session = _fetch(sid)
    try:
        with (
            patch(
                "app.services.pipeline_builder.steps._decide_rebalance",
                return_value=(False, "within_threshold"),
            ),
            patch(
                "app.services.pipeline_builder.steps._write_last_review_date"
            ) as mock_write,
        ):
            out = steps.step_rebalance_decision(
                sid,
                session,
                params={"prev_weights": [0.5] * 20},
                on_progress=lambda **_: None,
            )
            mock_write.assert_not_called()
        assert out["decision"] is False
        assert out["reason"] == "within_threshold"
    finally:
        session_store.delete_session(sid)


def test_when_last_review_date_override_then_used_verbatim():
    weights = _make_weights(20)
    sid = _seed(optimize_result=_result(weights), assembly=_Assembly())
    session = _fetch(sid)
    try:
        with (
            patch(
                "app.services.pipeline_builder.steps._decide_rebalance",
                return_value=(False, "within_threshold"),
            ) as mock_dec,
            patch(
                "app.services.pipeline_builder.steps._read_last_review_date"
            ) as mock_read,
            patch("app.services.pipeline_builder.steps._write_last_review_date"),
        ):
            steps.step_rebalance_decision(
                sid,
                session,
                params={
                    "prev_weights": [0.5] * 20,
                    "last_review_date": "2024-01-15",
                },
                on_progress=lambda **_: None,
            )
            mock_read.assert_not_called()
            _, kwargs = mock_dec.call_args
            assert kwargs["last_review_date"] == pd.Timestamp("2024-01-15")
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Weight-count warnings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_weights", [10, 14, 31, 50])
def test_when_n_weights_outside_band_then_weight_count_warning_true(n_weights):
    weights = _make_weights(n_weights)
    sid = _seed(optimize_result=_result(weights), assembly=_Assembly())
    session = _fetch(sid)
    try:
        with patch("app.services.pipeline_builder.steps._write_last_review_date"):
            out = steps.step_rebalance_decision(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert out["n_weights"] == n_weights
        assert out["weight_count_warning"] is True
    finally:
        session_store.delete_session(sid)


@pytest.mark.parametrize("n_weights", [15, 20, 30])
def test_when_n_weights_within_band_then_weight_count_warning_false(n_weights):
    weights = _make_weights(n_weights)
    sid = _seed(optimize_result=_result(weights), assembly=_Assembly())
    session = _fetch(sid)
    try:
        with patch("app.services.pipeline_builder.steps._write_last_review_date"):
            out = steps.step_rebalance_decision(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert out["weight_count_warning"] is False
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Missing sectors
# ---------------------------------------------------------------------------


def test_when_sectors_absent_then_missing_sectors_listed():
    weights = _make_weights(20)
    sid = _seed(optimize_result=_result(weights), assembly=_Assembly())
    session = _fetch(sid)
    try:
        with (
            patch(
                "app.services.pipeline_builder.steps._missing_gics_sectors",
                return_value=["Energy", "Utilities"],
            ),
            patch("app.services.pipeline_builder.steps._write_last_review_date"),
        ):
            out = steps.step_rebalance_decision(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert out["missing_sectors"] == ["Energy", "Utilities"]
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


def test_when_step_runs_then_session_updated_with_payload():
    weights = _make_weights(20)
    sid = _seed(optimize_result=_result(weights), assembly=_Assembly())
    session = _fetch(sid)
    try:
        with patch("app.services.pipeline_builder.steps._write_last_review_date"):
            out = steps.step_rebalance_decision(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert _fetch(sid).rebalance_result == out
    finally:
        session_store.delete_session(sid)
