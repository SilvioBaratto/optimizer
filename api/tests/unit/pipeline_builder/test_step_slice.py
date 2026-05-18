"""Tests for ``step_slice`` (issue #698 — Step 2 of the wizard)."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from app.services.pipeline_builder import session_store, steps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=20, freq="B")
    return pd.DataFrame({"A": range(20), "B": range(20, 40)}, index=idx)


@dataclasses.dataclass(frozen=True)
class _Assembly:
    prices: pd.DataFrame
    assembly_hash: str = "deadbeef"


def _seed_session(run_config: dict, assembly: _Assembly | None) -> str:
    """Insert a real session and overwrite its assembly field."""
    sid = session_store.create_session(dict(run_config))
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
    sid = session_store.create_session({})
    session = session_store.get_session(sid)
    try:
        with pytest.raises(ValueError, match="assembly not set"):
            steps.step_slice(sid, session, params={}, on_progress=lambda **_: None)
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# No-op path
# ---------------------------------------------------------------------------


def test_when_both_dates_none_then_no_op_payload_returned():
    prices = _make_prices()
    assembly = _Assembly(prices=prices)
    sid = _seed_session({}, assembly)
    session = session_store.get_session(sid)
    try:
        out = steps.step_slice(sid, session, params={}, on_progress=lambda **_: None)
        assert out == {
            "sliced": False,
            "price_start": str(prices.index[0].date()),
            "price_end": str(prices.index[-1].date()),
            "n_trading_days": 20,
        }
        # Session assembly object unchanged.
        assert _fetch(sid).assembly is assembly
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Sliced paths
# ---------------------------------------------------------------------------


def test_when_both_dates_set_then_assembly_replaced_and_payload_reports_slice():
    prices = _make_prices()
    assembly = _Assembly(prices=prices)
    run_config = {"start_date": "2024-01-08", "end_date": "2024-01-15"}
    sid = _seed_session(run_config, assembly)
    session = session_store.get_session(sid)
    try:
        out = steps.step_slice(sid, session, params={}, on_progress=lambda **_: None)
        assert out["sliced"] is True
        assert out["price_start"] == "2024-01-08"
        assert out["price_end"] == "2024-01-15"
        assert out["n_trading_days"] == 6
        # Session got a fresh assembly with sliced prices.
        new_assembly = _fetch(sid).assembly
        assert new_assembly is not assembly
        assert len(new_assembly.prices) == 6
        assert new_assembly.assembly_hash == "deadbeef"
    finally:
        session_store.delete_session(sid)


def test_when_only_start_date_set_then_slice_is_open_ended_to_right():
    prices = _make_prices()
    assembly = _Assembly(prices=prices)
    sid = _seed_session({"start_date": "2024-01-15"}, assembly)
    session = session_store.get_session(sid)
    try:
        out = steps.step_slice(sid, session, params={}, on_progress=lambda **_: None)
        assert out["sliced"] is True
        assert out["price_start"] == "2024-01-15"
        assert out["price_end"] == str(prices.index[-1].date())
    finally:
        session_store.delete_session(sid)


def test_when_only_end_date_set_then_slice_is_open_ended_to_left():
    prices = _make_prices()
    assembly = _Assembly(prices=prices)
    sid = _seed_session({"end_date": "2024-01-10"}, assembly)
    session = session_store.get_session(sid)
    try:
        out = steps.step_slice(sid, session, params={}, on_progress=lambda **_: None)
        assert out["sliced"] is True
        assert out["price_start"] == str(prices.index[0].date())
        assert out["price_end"] == "2024-01-10"
    finally:
        session_store.delete_session(sid)


def test_when_params_override_then_takes_precedence_over_run_config():
    prices = _make_prices()
    assembly = _Assembly(prices=prices)
    run_config = {"start_date": "2024-01-02", "end_date": "2024-01-31"}
    sid = _seed_session(run_config, assembly)
    session = session_store.get_session(sid)
    try:
        out = steps.step_slice(
            sid,
            session,
            params={"start_date": "2024-01-10", "end_date": "2024-01-12"},
            on_progress=lambda **_: None,
        )
        assert out["price_start"] == "2024-01-10"
        assert out["price_end"] == "2024-01-12"
    finally:
        session_store.delete_session(sid)


def test_when_params_have_only_start_then_other_falls_back_to_run_config():
    prices = _make_prices()
    assembly = _Assembly(prices=prices)
    run_config = {"end_date": "2024-01-19"}
    sid = _seed_session(run_config, assembly)
    session = session_store.get_session(sid)
    try:
        out = steps.step_slice(
            sid,
            session,
            params={"start_date": "2024-01-15"},
            on_progress=lambda **_: None,
        )
        assert out["price_start"] == "2024-01-15"
        assert out["price_end"] == "2024-01-19"
    finally:
        session_store.delete_session(sid)
