"""Pure-unit tests for the _run_bulk_calibrate background-job wrapper.

Calls the wrapper directly (no HTTP, no client, no DB fixtures).
Covers three branches:
  1. success          -- service succeeds, running status set with total, no failure
  2. CancelledError   -- wrapper exits silently, no failure status recorded
  3. generic exception -- failure status recorded with error message

Also exercises the cancel_event=None default branch (all tests omit it).
"""

from __future__ import annotations

import threading
from concurrent.futures import CancelledError
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from app.api.v1.macro.macro_calibration import _run_bulk_calibrate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOB_ID = "jid-calibrate"
_COUNTRIES = ["USA", "UK"]
_FORCE_REFRESH = False

_SERVICE_PATH = "app.api.v1.macro.macro_calibration.run_bulk_calibrate"
_SVC_INSTANCE_PATH = "app.api.v1.macro.macro_calibration._calibrate_job_service"

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _has_running_with_total(mock_update: MagicMock, expected_total: int) -> bool:
    return any(
        kw.get("status") == "running" and kw.get("total") == expected_total
        for _, kw in mock_update.call_args_list
    )


def _has_failed_call(mock_update: MagicMock, expected_error: str) -> bool:
    return any(
        kw.get("status") == "failed" and kw.get("error") == expected_error
        for _, kw in mock_update.call_args_list
    )


def _no_failed_call(mock_update: MagicMock) -> bool:
    return not any(
        kw.get("status") == "failed"
        for _, kw in mock_update.call_args_list
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_when_service_succeeds_then_running_set_with_total_and_no_failure() -> None:
    """Success branch: running status with total=2 recorded; service called; no failure."""
    with ExitStack() as stack:
        mock_calibrate = stack.enter_context(
            patch(_SERVICE_PATH, return_value=None)
        )
        mock_update = stack.enter_context(
            patch(f"{_SVC_INSTANCE_PATH}.update_job", new_callable=MagicMock)
        )

        # omitting cancel_event exercises the default-branch (None -> Event())
        _run_bulk_calibrate(JOB_ID, _COUNTRIES, _FORCE_REFRESH)

    mock_calibrate.assert_called_once_with(
        _COUNTRIES, _FORCE_REFRESH, on_progress=mock_calibrate.call_args.kwargs["on_progress"]
    )
    _, kwargs = mock_calibrate.call_args
    assert "on_progress" in kwargs, "run_bulk_calibrate must be called with on_progress kwarg"

    assert _has_running_with_total(mock_update, 2), (
        "expected update_job called with status='running' and total=2"
    )
    assert _no_failed_call(mock_update), "no failure update expected on success"


def test_when_cancelled_then_returns_without_failure_update() -> None:
    """CancelledError branch: wrapper returns silently; no failure status recorded."""
    with ExitStack() as stack:
        stack.enter_context(patch(_SERVICE_PATH, side_effect=CancelledError()))
        mock_update = stack.enter_context(
            patch(f"{_SVC_INSTANCE_PATH}.update_job", new_callable=MagicMock)
        )

        _run_bulk_calibrate(JOB_ID, _COUNTRIES, _FORCE_REFRESH)  # must not propagate

    assert _no_failed_call(mock_update), "CancelledError must not trigger a failed update"


def test_when_service_raises_then_failed_status_recorded() -> None:
    """Generic exception branch: failure status recorded with correct error message."""
    with ExitStack() as stack:
        stack.enter_context(patch(_SERVICE_PATH, side_effect=Exception("kaboom")))
        mock_update = stack.enter_context(
            patch(f"{_SVC_INSTANCE_PATH}.update_job", new_callable=MagicMock)
        )

        _run_bulk_calibrate(JOB_ID, _COUNTRIES, _FORCE_REFRESH)  # must not propagate

    assert _has_failed_call(mock_update, "kaboom"), (
        "expected update_job called with status='failed' and error='kaboom'"
    )


def test_when_cancel_event_provided_then_used_not_defaulted() -> None:
    """cancel_event-provided branch: an explicit Event is accepted (not defaulted)."""
    event = threading.Event()
    with ExitStack() as stack:
        mock_calibrate = stack.enter_context(patch(_SERVICE_PATH, return_value=None))
        stack.enter_context(
            patch(f"{_SVC_INSTANCE_PATH}.update_job", new_callable=MagicMock)
        )
        _run_bulk_calibrate(JOB_ID, _COUNTRIES, _FORCE_REFRESH, cancel_event=event)

    mock_calibrate.assert_called_once()
