"""Tests for ``step_clean_returns`` (issue #700 — Step 4 of the wizard)."""

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
    """Stub for ``DataAssembly`` — only fields the handler touches."""


def _seed(assembly: _Assembly | None, investable: pd.Index | None) -> str:
    sid = session_store.create_session({})
    if assembly is not None:
        session_store.update_session(sid, assembly=assembly)
    if investable is not None:
        session_store.update_session(sid, investable=investable)
    return sid


def _fetch(sid: str) -> session_store._PipelineSession:
    s = session_store.get_session(sid)
    assert s is not None
    return s


def _clean_df(n_days: int = 250, n_tickers: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    cols = [f"T{i}" for i in range(n_tickers)]
    return pd.DataFrame(0.0, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_when_assembly_missing_then_value_error_raised():
    sid = _seed(assembly=None, investable=pd.Index(["A"]))
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="assembly not set"):
            steps.step_clean_returns(
                sid, session, params={}, on_progress=lambda **_: None
            )
    finally:
        session_store.delete_session(sid)


def test_when_investable_missing_then_value_error_raised():
    sid = _seed(assembly=_Assembly(), investable=None)
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="investable not set"):
            steps.step_clean_returns(
                sid, session, params={}, on_progress=lambda **_: None
            )
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_when_clean_returns_succeeds_then_session_updated_and_payload_returned():
    assembly = _Assembly()
    investable = pd.Index(["A", "B", "C"])
    sid = _seed(assembly=assembly, investable=investable)
    session = _fetch(sid)
    df = _clean_df(n_days=300, n_tickers=3)
    try:
        with patch(
            "app.services.pipeline_builder.steps._materialise_clean_returns",
            return_value=df,
        ) as mock_fn:
            out = steps.step_clean_returns(
                sid, session, params={}, on_progress=lambda **_: None
            )
            mock_fn.assert_called_once_with(assembly, investable)

        assert out == {
            "n_days": 300,
            "n_tickers": 3,
            "return_start": str(df.index[0].date()),
            "return_end": str(df.index[-1].date()),
            "preprocessing_steps": [
                "DataValidator",
                "OutlierTreater",
                "SectorImputer",
                "RegressionImputer",
            ],
        }
        assert _fetch(sid).clean_returns is df
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# NaN/inf abort gate
# ---------------------------------------------------------------------------


def test_when_materialise_raises_then_runtime_error_propagates():
    sid = _seed(assembly=_Assembly(), investable=pd.Index(["A"]))
    session = _fetch(sid)
    err = RuntimeError("clean_returns contains NaN or inf after preprocessing.")
    try:
        with patch(
            "app.services.pipeline_builder.steps._materialise_clean_returns",
            side_effect=err,
        ):
            with pytest.raises(RuntimeError) as exc:
                steps.step_clean_returns(
                    sid, session, params={}, on_progress=lambda **_: None
                )
            assert exc.value is err
    finally:
        session_store.delete_session(sid)


def test_when_materialise_raises_then_session_not_updated():
    sid = _seed(assembly=_Assembly(), investable=pd.Index(["A"]))
    session = _fetch(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps._materialise_clean_returns",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError):
                steps.step_clean_returns(
                    sid, session, params={}, on_progress=lambda **_: None
                )
        assert _fetch(sid).clean_returns is None
    finally:
        session_store.delete_session(sid)
