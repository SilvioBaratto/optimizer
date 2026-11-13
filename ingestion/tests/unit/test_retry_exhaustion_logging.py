"""``retry_with_backoff`` returns None when it gives up; the log must say why.

Sixteen callers treat None as "failed after retries" and report nothing else, so the
last error is only recoverable from this log line.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from app.services.infrastructure.retry import retry_with_backoff

_LOGGER = "app.services.infrastructure.retry"


def _always_raises() -> int:
    raise RuntimeError("boom")


def _retry_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == _LOGGER]


def test_exhaustion_logs_the_last_exception(caplog: pytest.LogCaptureFixture) -> None:
    """After the last failed attempt one WARNING carries the exception."""
    with (
        patch("app.services.infrastructure.retry.time.sleep"),
        caplog.at_level(logging.WARNING, logger=_LOGGER),
    ):
        result = retry_with_backoff(_always_raises, max_retries=2)

    records = _retry_records(caplog)
    assert result is None
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].exc_info is not None
    assert str(records[0].exc_info[1]) == "boom"


def test_exhaustion_by_rate_limit_logs_the_last_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Giving up on a rate-limited action is logged like any other exhaustion."""
    with (
        patch("app.services.infrastructure.retry.time.sleep"),
        caplog.at_level(logging.WARNING, logger=_LOGGER),
    ):
        result = retry_with_backoff(
            _always_raises, max_retries=2, is_rate_limit_error=lambda _exc: True
        )

    records = _retry_records(caplog)
    assert result is None
    assert len(records) == 1
    exc_info = records[0].exc_info
    assert exc_info is not None
    assert str(exc_info[1]) == "boom"


def test_exhaustion_by_invalid_result_logs_the_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A result rejected by ``is_valid`` is named as the reason, not an exception."""
    with (
        patch("app.services.infrastructure.retry.time.sleep"),
        caplog.at_level(logging.WARNING, logger=_LOGGER),
    ):
        result = retry_with_backoff(lambda: 0, max_retries=2, is_valid=lambda v: v > 0)

    records = _retry_records(caplog)
    assert result is None
    assert len(records) == 1
    assert "is_valid" in records[0].getMessage()
    assert records[0].exc_info is None


def test_invalid_result_after_an_earlier_exception_is_not_blamed_on_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The reason is the last attempt's outcome, not a stale earlier exception."""
    outcomes = iter([RuntimeError("first"), 0])

    def flaky() -> int:
        """Raise on the first call, then return an invalid (zero) result."""
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    with (
        patch("app.services.infrastructure.retry.time.sleep"),
        caplog.at_level(logging.WARNING, logger=_LOGGER),
    ):
        retry_with_backoff(flaky, max_retries=2, is_valid=lambda v: v > 0)

    records = _retry_records(caplog)
    assert len(records) == 1
    assert "is_valid" in records[0].getMessage()


def test_success_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """A first-try success leaves no record."""
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        result = retry_with_backoff(lambda: 5, max_retries=3)

    assert result == 5
    assert _retry_records(caplog) == []
