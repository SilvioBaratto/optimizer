"""Tests for ``step_load`` (issue #697 — Step 1 of the wizard).

Covers:

* happy path: ``load_data`` invoked with the configured base currency,
  ``assembly`` + ``country_map`` persisted onto the session, JSON-safe
  summary dict returned;
* upfront ``n_selected`` gate raises ``RuntimeError`` for values outside
  the inclusive ``[15, 30]`` band;
* ``_assert_assembly_size`` failures from ``load_data`` propagate
  verbatim (no swallowing);
* progress callback emits ``db_preflight`` → ``assembling`` → ``done``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.pipeline_builder import session_store, steps

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _fake_assembly(n_tickers: int = 2500, n_days: int = 1500) -> SimpleNamespace:
    idx = pd.date_range("2021-01-04", periods=n_days, freq="B")
    cols = [f"T{i}" for i in range(n_tickers)]
    prices = pd.DataFrame(1.0, index=idx, columns=cols)
    return SimpleNamespace(
        prices=prices,
        n_tickers=n_tickers,
        n_trading_days=n_days,
        assembly_hash="deadbeef",
    )


# ---------------------------------------------------------------------------
# n_selected gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_selected", [10, 14])
def test_when_n_selected_below_15_then_runtime_error(n_selected):
    sid = session_store.create_session({"n_selected": n_selected})
    session = session_store.get_session(sid)
    try:
        with pytest.raises(RuntimeError, match=r"n_selected=\d+ outside \[15, 30\]"):
            steps.step_load(sid, session, params={}, on_progress=lambda **_: None)
    finally:
        session_store.delete_session(sid)


@pytest.mark.parametrize("n_selected", [31, 35, 50])
def test_when_n_selected_above_30_then_runtime_error(n_selected):
    sid = session_store.create_session({"n_selected": n_selected})
    session = session_store.get_session(sid)
    try:
        with pytest.raises(RuntimeError, match=r"n_selected=\d+ outside \[15, 30\]"):
            steps.step_load(sid, session, params={}, on_progress=lambda **_: None)
    finally:
        session_store.delete_session(sid)


def test_when_n_selected_default_missing_then_falls_back_to_20():
    sid = session_store.create_session({})
    session = session_store.get_session(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps.load_data",
            return_value=(_fake_assembly(), {"AAPL": "US"}, MagicMock()),
        ):
            result = steps.step_load(
                sid, session, params={}, on_progress=lambda **_: None
            )
        assert result["n_tickers"] == 2500
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_when_load_data_succeeds_then_returns_json_safe_summary():
    sid = session_store.create_session({"n_selected": 20, "base_currency": "USD"})
    session = session_store.get_session(sid)
    try:
        assembly = _fake_assembly(n_tickers=2700, n_days=1400)
        with patch(
            "app.services.pipeline_builder.steps.load_data",
            return_value=(assembly, {"AAPL": "US", "TLT": "US"}, MagicMock()),
        ) as mock_load:
            result = steps.step_load(
                sid, session, params={}, on_progress=lambda **_: None
            )
            mock_load.assert_called_once()
            _, kwargs = mock_load.call_args
            assert kwargs["base_currency"] == "USD"

        assert result == {
            "n_tickers": 2700,
            "n_trading_days": 1400,
            "assembly_hash": "deadbeef",
            "base_currency": "USD",
            "price_start": str(assembly.prices.index[0]),
            "price_end": str(assembly.prices.index[-1]),
        }
    finally:
        session_store.delete_session(sid)


def test_when_load_data_succeeds_then_assembly_persisted_in_session():
    sid = session_store.create_session({"n_selected": 20})
    session = session_store.get_session(sid)
    try:
        assembly = _fake_assembly()
        with patch(
            "app.services.pipeline_builder.steps.load_data",
            return_value=(assembly, {"AAPL": "US"}, MagicMock()),
        ):
            steps.step_load(sid, session, params={}, on_progress=lambda **_: None)
        updated = session_store.get_session(sid)
        assert updated is not None
        assert updated.assembly is assembly
        assert updated.run_config["country_map"] == {"AAPL": "US"}
    finally:
        session_store.delete_session(sid)


def test_when_base_currency_missing_then_defaults_to_eur():
    sid = session_store.create_session({"n_selected": 20})
    session = session_store.get_session(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps.load_data",
            return_value=(_fake_assembly(), {}, MagicMock()),
        ) as mock_load:
            steps.step_load(sid, session, params={}, on_progress=lambda **_: None)
        _, kwargs = mock_load.call_args
        assert kwargs["base_currency"] == "EUR"
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


def test_when_load_data_runs_then_progress_phases_emitted_in_order():
    sid = session_store.create_session({"n_selected": 20})
    session = session_store.get_session(sid)
    phases: list[str] = []

    def on_progress(**kwargs):
        if "phase" in kwargs:
            phases.append(kwargs["phase"])

    try:
        with patch(
            "app.services.pipeline_builder.steps.load_data",
            return_value=(_fake_assembly(), {}, MagicMock()),
        ):
            steps.step_load(sid, session, params={}, on_progress=on_progress)
        assert phases == ["db_preflight", "assembling", "done"]
    finally:
        session_store.delete_session(sid)


# ---------------------------------------------------------------------------
# Abort gate propagation
# ---------------------------------------------------------------------------


def test_when_load_data_raises_assembly_size_then_propagates_unchanged():
    sid = session_store.create_session({"n_selected": 20})
    session = session_store.get_session(sid)
    err = RuntimeError("DataAssembly: n_tickers=500 below floor 2000.")
    try:
        with patch(
            "app.services.pipeline_builder.steps.load_data",
            side_effect=err,
        ):
            with pytest.raises(RuntimeError) as exc:
                steps.step_load(sid, session, params={}, on_progress=lambda **_: None)
            assert exc.value is err
    finally:
        session_store.delete_session(sid)


def test_when_load_data_raises_then_session_not_updated():
    sid = session_store.create_session({"n_selected": 20})
    session = session_store.get_session(sid)
    try:
        with patch(
            "app.services.pipeline_builder.steps.load_data",
            side_effect=RuntimeError("preflight failed"),
        ):
            with pytest.raises(RuntimeError):
                steps.step_load(sid, session, params={}, on_progress=lambda **_: None)
        updated = session_store.get_session(sid)
        assert updated is not None
        assert updated.assembly is None
        assert "country_map" not in updated.run_config
    finally:
        session_store.delete_session(sid)
