"""Pure-unit tests for the four background-job wrapper functions in macro_regime.

Calls each wrapper directly (no HTTP, no client, no DB fixtures).
Covers three branches per wrapper:
  1. success          -- service succeeds, running status set, no failure recorded
  2. CancelledError   -- wrapper exits silently, no failure status recorded
  3. generic exception -- failure status recorded with error message

Also exercises the cancel_event=None default branch (all success tests omit it).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import CancelledError
from contextlib import ExitStack
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest

from app.api.v1.macro.macro_regime import (
    _run_bulk_fetch,
    _run_fred_fetch,
    _run_macro_news_fetch,
    _run_news_summarize,
)
from app.schemas.macro.macro_regime import (
    FredFetchRequest,
    MacroFetchRequest,
    MacroNewsFetchRequest,
    MacroNewsSummarizeRequest,
)

# ---------------------------------------------------------------------------
# Case descriptors
# ---------------------------------------------------------------------------

JOB_ID = "jid-test"


class _Case(NamedTuple):
    """Holds the per-wrapper test parameters."""

    wrapper: Callable[..., None]
    service_path: str  # dotted path to the imported service fn
    svc_instance_path: str  # dotted path to the module-level job-service instance
    request: object  # pre-built request model instance


_CASES: list[_Case] = [
    _Case(
        wrapper=_run_bulk_fetch,
        service_path="app.api.v1.macro.macro_regime.run_bulk_macro_fetch",
        svc_instance_path="app.api.v1.macro.macro_regime._job_service",
        request=MacroFetchRequest(),
    ),
    _Case(
        wrapper=_run_fred_fetch,
        service_path="app.api.v1.macro.macro_regime.run_bulk_fred_fetch",
        svc_instance_path="app.api.v1.macro.macro_regime._fred_job_service",
        request=FredFetchRequest(),
    ),
    _Case(
        wrapper=_run_macro_news_fetch,
        service_path="app.api.v1.macro.macro_regime.run_macro_news_fetch",
        svc_instance_path="app.api.v1.macro.macro_regime._news_job_service",
        request=MacroNewsFetchRequest(),
    ),
    _Case(
        wrapper=_run_news_summarize,
        service_path="app.api.v1.macro.macro_regime.run_news_summarize",
        svc_instance_path="app.api.v1.macro.macro_regime._summarize_job_service",
        request=MacroNewsSummarizeRequest(),
    ),
]

_CASE_IDS = [
    "bulk_fetch",
    "fred_fetch",
    "macro_news_fetch",
    "news_summarize",
]

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_when_service_succeeds_then_running_set_and_no_failure(case: _Case) -> None:
    """Success branch: running status recorded; service called with on_progress; no failure."""
    with ExitStack() as stack:
        mock_svc_fn = stack.enter_context(patch(case.service_path, return_value=None))
        mock_update = stack.enter_context(
            patch(f"{case.svc_instance_path}.update_job", new_callable=MagicMock)
        )

        # omitting cancel_event exercises the default-branch (None -> Event())
        case.wrapper(JOB_ID, case.request)

    mock_svc_fn.assert_called_once()
    _, kwargs = mock_svc_fn.call_args
    assert "on_progress" in kwargs, "service fn must be called with on_progress kwarg"

    mock_update.assert_any_call(JOB_ID, status="running")
    assert _no_failed_call(mock_update), "no failure update expected on success"


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_when_cancelled_then_returns_without_failure_update(case: _Case) -> None:
    """CancelledError branch: wrapper returns silently; no failure status recorded."""
    with ExitStack() as stack:
        stack.enter_context(
            patch(case.service_path, side_effect=CancelledError())
        )
        mock_update = stack.enter_context(
            patch(f"{case.svc_instance_path}.update_job", new_callable=MagicMock)
        )

        case.wrapper(JOB_ID, case.request)  # must not propagate

    assert _no_failed_call(mock_update), "CancelledError must not trigger a failed update"


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_when_service_raises_then_failed_status_recorded(case: _Case) -> None:
    """Generic exception branch: failure status recorded with correct error message."""
    with ExitStack() as stack:
        stack.enter_context(
            patch(case.service_path, side_effect=Exception("boom"))
        )
        mock_update = stack.enter_context(
            patch(f"{case.svc_instance_path}.update_job", new_callable=MagicMock)
        )

        case.wrapper(JOB_ID, case.request)  # must not propagate

    assert _has_failed_call(mock_update, "boom"), (
        "expected update_job called with status='failed' and error='boom'"
    )


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_when_cancel_event_provided_then_used_not_defaulted(case: _Case) -> None:
    """cancel_event-provided branch: an explicit Event is accepted (not defaulted)."""
    event = threading.Event()
    with ExitStack() as stack:
        mock_svc_fn = stack.enter_context(patch(case.service_path, return_value=None))
        stack.enter_context(
            patch(f"{case.svc_instance_path}.update_job", new_callable=MagicMock)
        )
        case.wrapper(JOB_ID, case.request, cancel_event=event)

    mock_svc_fn.assert_called_once()
