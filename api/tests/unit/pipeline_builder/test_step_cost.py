"""Tests for ``step_cost`` (issue #708 — Step 12)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from app.services.pipeline_builder import session_store, steps

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _result(weights: pd.Series) -> SimpleNamespace:
    return SimpleNamespace(weights=weights)


def _seed(*, optimize_result=None, run_config: dict | None = None) -> str:
    sid = session_store.create_session(dict(run_config or {}))
    if optimize_result is not None:
        session_store.update_session(sid, optimize_result=optimize_result)
    return sid


def _fetch(sid: str) -> session_store._PipelineSession:
    s = session_store.get_session(sid)
    assert s is not None
    return s


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------


def test_when_optimize_result_missing_then_value_error_raised():
    sid = _seed(optimize_result=None)
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="optimize_result not set"):
            steps.step_cost(sid, session, params={}, on_progress=lambda **_: None)
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Happy path — UK + US mix
# ---------------------------------------------------------------------------


def test_when_uk_and_us_weights_then_weighted_cost_matches_hand_calc():
    weights = pd.Series({"VOD.L": 0.4, "AAPL": 0.6})
    country_map = {"VOD.L": "United Kingdom", "AAPL": "United States"}
    sid = _seed(
        optimize_result=_result(weights),
        run_config={"country_map": country_map, "cost_bps": 10.0},
    )
    session = _fetch(sid)
    # UK total = 50 + 8 + 0 = 58; US total = 0 + 3 + 12 = 15
    expected = 0.4 * 58 + 0.6 * 15  # 23.2 + 9 = 32.2
    try:
        out = steps.step_cost(sid, session, params={}, on_progress=lambda **_: None)
        assert out["cost_bps_actual"] == pytest.approx(expected)
        assert out["cost_bps_assumed"] == 10.0
        assert out["exceeds_assumed"] is True
        per_country = {row["country"]: row for row in out["per_country"]}
        assert set(per_country) == {"United Kingdom", "United States"}
        uk = per_country["United Kingdom"]
        assert uk["stamp"] == 50.0
        assert uk["spread"] == 8.0
        assert uk["fx"] == 0.0
        assert uk["total"] == 58.0
        us = per_country["United States"]
        assert us["total"] == 15.0
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Overrides do not leak into module-level dict
# ---------------------------------------------------------------------------


def test_when_override_supplied_then_cost_uses_override_and_module_dict_unchanged():
    from research.pipeline._optimize import COUNTRY_COSTS_BPS

    weights = pd.Series({"ENI.MI": 1.0})
    country_map = {"ENI.MI": "Italy"}
    sid = _seed(
        optimize_result=_result(weights),
        run_config={"country_map": country_map, "cost_bps": 5.0},
    )
    session = _fetch(sid)
    italy_before = dict(COUNTRY_COSTS_BPS["Italy"])
    try:
        out = steps.step_cost(
            sid,
            session,
            params={"country_cost_overrides": {"Italy": {"stamp": 25.0}}},
            on_progress=lambda **_: None,
        )
        # Italy override: 25 + 8 + 12 = 45
        assert out["cost_bps_actual"] == pytest.approx(45.0)
        italy_row = next(r for r in out["per_country"] if r["country"] == "Italy")
        assert italy_row["stamp"] == 25.0
        assert italy_row["total"] == 45.0
        # Module-level dict untouched.
        assert COUNTRY_COSTS_BPS["Italy"] == italy_before
    finally:
        session_store.delete_session(sid)


def test_when_override_adds_new_country_then_used_for_that_country():
    weights = pd.Series({"7203.T": 1.0})
    country_map = {"7203.T": "Japan"}
    sid = _seed(
        optimize_result=_result(weights),
        run_config={"country_map": country_map, "cost_bps": 10.0},
    )
    session = _fetch(sid)
    try:
        out = steps.step_cost(
            sid,
            session,
            params={
                "country_cost_overrides": {
                    "Japan": {"stamp": 5.0, "spread": 4.0, "fx": 8.0}
                }
            },
            on_progress=lambda **_: None,
        )
        # Japan total = 17
        assert out["cost_bps_actual"] == pytest.approx(17.0)
        japan = next(r for r in out["per_country"] if r["country"] == "Japan")
        assert japan["total"] == 17.0
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Empty country_map → all tickers use _DEFAULT_COSTS
# ---------------------------------------------------------------------------


def test_when_no_country_map_then_default_costs_used():
    weights = pd.Series({"X": 1.0})
    sid = _seed(
        optimize_result=_result(weights),
        run_config={"cost_bps": 10.0},
    )
    session = _fetch(sid)
    try:
        out = steps.step_cost(sid, session, params={}, on_progress=lambda **_: None)
        # _DEFAULT_COSTS = stamp 0, spread 6, fx 12 → 18
        assert out["cost_bps_actual"] == pytest.approx(18.0)
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# exceeds_assumed flag
# ---------------------------------------------------------------------------


def test_when_actual_below_assumed_then_exceeds_assumed_false():
    weights = pd.Series({"AAPL": 1.0})
    country_map = {"AAPL": "United States"}
    sid = _seed(
        optimize_result=_result(weights),
        run_config={"country_map": country_map, "cost_bps": 50.0},
    )
    session = _fetch(sid)
    try:
        out = steps.step_cost(sid, session, params={}, on_progress=lambda **_: None)
        assert out["cost_bps_actual"] == pytest.approx(15.0)
        assert out["cost_bps_assumed"] == 50.0
        assert out["exceeds_assumed"] is False
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


def test_when_step_runs_then_session_updated_with_payload():
    weights = pd.Series({"AAPL": 1.0})
    sid = _seed(
        optimize_result=_result(weights),
        run_config={"country_map": {"AAPL": "United States"}, "cost_bps": 10.0},
    )
    session = _fetch(sid)
    try:
        out = steps.step_cost(sid, session, params={}, on_progress=lambda **_: None)
        assert _fetch(sid).cost_result == out
    finally:
        session_store.delete_session(sid)
