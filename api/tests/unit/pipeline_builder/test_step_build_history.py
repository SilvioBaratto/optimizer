"""Tests for ``step_build_history`` (issue #701 — Step 5 of the wizard)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.pipeline_builder import session_store, steps

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Assembly:
    """Stub for ``DataAssembly`` — handler does not introspect fields."""


def _seed(
    run_config: dict | None = None,
    *,
    assembly: _Assembly | None,
    investable: pd.Index | None,
) -> str:
    sid = session_store.create_session(dict(run_config or {}))
    if assembly is not None:
        session_store.update_session(sid, assembly=assembly)
    if investable is not None:
        session_store.update_session(sid, investable=investable)
    return sid


def _fetch(sid: str) -> session_store._PipelineSession:
    s = session_store.get_session(sid)
    assert s is not None
    return s


def _health(succeeded: int = 18, total: int = 20, failed: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        succeeded_dates=succeeded,
        total_dates=total,
        failed_dates=failed,
    )


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_when_assembly_missing_then_value_error_raised():
    sid = _seed(assembly=None, investable=pd.Index(["A"]))
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="assembly not set"):
            steps.step_build_history(
                sid, session, params={}, on_progress=lambda **_: None
            )
    finally:
        session_store.delete_session(sid)


def test_when_investable_missing_then_value_error_raised():
    sid = _seed(assembly=_Assembly(), investable=None)
    session = _fetch(sid)
    try:
        with pytest.raises(ValueError, match="investable not set"):
            steps.step_build_history(
                sid, session, params={}, on_progress=lambda **_: None
            )
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Happy path — market proxy present
# ---------------------------------------------------------------------------


def test_when_proxy_and_build_succeed_then_session_updated_and_payload_returned():
    assembly = _Assembly()
    investable = pd.Index(["A", "B"])
    sid = _seed(assembly=assembly, investable=investable)
    session = _fetch(sid)
    proxy = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
    )
    fsd = {"momentum": pd.DataFrame()}
    rh = pd.DataFrame({"r": [0.1]})
    health = _health(succeeded=15, total=16, failed=1)
    try:
        with (
            patch(
                "app.services.pipeline_builder.steps._load_market_proxy_from_db",
                return_value=proxy,
            ) as mock_proxy,
            patch(
                "app.services.pipeline_builder.steps.build_history",
                return_value=(fsd, rh, health),
            ) as mock_build,
        ):
            out = steps.step_build_history(
                sid, session, params={}, on_progress=lambda **_: None
            )
            mock_proxy.assert_called_once()
            mock_build.assert_called_once()
            _, kwargs = mock_build.call_args
            assert kwargs["rebalance_freq"] == 63
            assert kwargs["market_prices"] is proxy

        assert out == {
            "succeeded_dates": 15,
            "total_dates": 16,
            "failed_dates": 1,
            "n_factors": 1,
            "rebalance_freq": 63,
            "market_proxy_loaded": True,
        }
        updated = _fetch(sid)
        assert updated.factor_scores_dict is fsd
        assert updated.returns_history is rh
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Market proxy not present
# ---------------------------------------------------------------------------


def test_when_proxy_returns_none_then_market_proxy_loaded_false():
    sid = _seed(assembly=_Assembly(), investable=pd.Index(["A"]))
    session = _fetch(sid)
    try:
        with (
            patch(
                "app.services.pipeline_builder.steps._load_market_proxy_from_db",
                return_value=None,
            ),
            patch(
                "app.services.pipeline_builder.steps.build_history",
                return_value=({}, pd.DataFrame(), _health()),
            ) as mock_build,
        ):
            out = steps.step_build_history(
                sid, session, params={}, on_progress=lambda **_: None
            )
            _, kwargs = mock_build.call_args
            assert kwargs["market_prices"] is None

        assert out["market_proxy_loaded"] is False
        assert out["n_factors"] == 0
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Rebalance frequency override
# ---------------------------------------------------------------------------


def test_when_rebalance_freq_in_run_config_then_passed_to_build_history():
    sid = _seed(
        run_config={"rebalance_freq": 21},
        assembly=_Assembly(),
        investable=pd.Index(["A"]),
    )
    session = _fetch(sid)
    try:
        with (
            patch(
                "app.services.pipeline_builder.steps._load_market_proxy_from_db",
                return_value=None,
            ),
            patch(
                "app.services.pipeline_builder.steps.build_history",
                return_value=({}, pd.DataFrame(), _health()),
            ) as mock_build,
        ):
            out = steps.step_build_history(
                sid, session, params={}, on_progress=lambda **_: None
            )
        _, kwargs = mock_build.call_args
        assert kwargs["rebalance_freq"] == 21
        assert out["rebalance_freq"] == 21
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Ticker override
# ---------------------------------------------------------------------------


def test_when_params_ticker_set_then_helper_called_with_override():
    sid = _seed(assembly=_Assembly(), investable=pd.Index(["A"]))
    session = _fetch(sid)
    try:
        with (
            patch(
                "app.services.pipeline_builder.steps._load_market_proxy_from_db",
                return_value=None,
            ) as mock_proxy,
            patch(
                "app.services.pipeline_builder.steps.build_history",
                return_value=({}, pd.DataFrame(), _health()),
            ),
        ):
            steps.step_build_history(
                sid,
                session,
                params={"market_proxy_ticker": "VTI"},
                on_progress=lambda **_: None,
            )
        _, kwargs = mock_proxy.call_args
        assert kwargs["tickers"] == ("VTI",)
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# DB session closed before build_history runs
# ---------------------------------------------------------------------------


def test_when_step_runs_then_db_session_released_before_build_history():
    """The proxy helper must complete (and release the DB connection)
    before ``build_history`` is invoked. Capture call order via a
    parent mock attaching both children."""
    sid = _seed(assembly=_Assembly(), investable=pd.Index(["A"]))
    session = _fetch(sid)

    parent = MagicMock()
    parent.proxy.return_value = None
    parent.build.return_value = ({}, pd.DataFrame(), _health())

    try:
        with (
            patch(
                "app.services.pipeline_builder.steps._load_market_proxy_from_db",
                new=parent.proxy,
            ),
            patch(
                "app.services.pipeline_builder.steps.build_history",
                new=parent.build,
            ),
        ):
            steps.step_build_history(
                sid, session, params={}, on_progress=lambda **_: None
            )

        names = [c[0] for c in parent.mock_calls]
        assert names.index("proxy") < names.index("build")
    finally:
        session_store.delete_session(sid)
