"""Scheduler error-surfacing tests (issue #849).

The scheduler's ``except`` blocks must surface failures (log with traceback)
rather than swallow them silently. This locks the behaviour for the
reference-index refresh path so a regression to a bare ``except: pass`` fails.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

from app.config import settings
from app.services.jobs import scheduler


def test_when_benchmark_lookup_fails_then_error_is_logged(caplog) -> None:
    """when the portfolio benchmark lookup raises, the failure is logged."""
    with patch.object(
        scheduler.database_manager,
        "get_session",
        side_effect=RuntimeError("db down"),
    ):
        with caplog.at_level(logging.ERROR, logger="app.services.jobs.scheduler"):
            result = scheduler._resolve_benchmark_tickers()

    # Degrades to the operator-configured benchmarks instead of crashing.
    assert result == sorted(set(settings.benchmark_tickers))
    # The failure is surfaced: logged at ERROR with a traceback (logger.exception).
    matching = [
        r
        for r in caplog.records
        if "portfolio benchmark lookup failed" in r.getMessage()
    ]
    assert matching, "scheduler swallowed the benchmark lookup failure silently"
    assert any(r.exc_info is not None for r in matching)
