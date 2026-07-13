"""Startup-time bootstrap for reference-index benchmarks.

Ensures every ticker listed in ``settings.benchmark_tickers`` has an
:class:`~app.models.universe.Instrument` row and recent price history.

Runs in a daemon thread at worker startup (after ``init_db``, alongside
scheduler start). On a fresh database this incurs a ~30-second yfinance fetch
for the full benchmark set; on subsequent restarts the delta is empty and the
helper returns immediately.

Failures are logged and swallowed — startup must never abort because yfinance
is rate-limited or a single ticker is delisted.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.repositories.market_data.yfinance_repository import YFinanceRepository
from app.services.market_data.reference_index_seeder import seed_reference_indices
from app.services.market_data.yfinance import YFinanceClient

logger = logging.getLogger(__name__)


SessionFactory = Callable[[], AbstractContextManager[Session]]


def _compute_delta(
    coverage: dict[str, tuple[int, date | None]],
    today: date,
    stale_days: int,
) -> list[str]:
    """Return tickers whose price history is missing or stale."""
    stale_cutoff = today - timedelta(days=stale_days)
    delta: list[str] = []
    for ticker, (count, latest) in coverage.items():
        if count == 0 or latest is None or latest < stale_cutoff:
            delta.append(ticker)
    return delta


def bootstrap_benchmarks(
    session_factory: SessionFactory,
    yf_client: YFinanceClient,
    *,
    tickers: list[str],
    stale_days: int = 2,
    today: date | None = None,
) -> dict[str, object]:
    """Seed any benchmark whose price history is missing or stale.

    Parameters
    ----------
    session_factory:
        Callable returning a context-managed SQLAlchemy ``Session`` (typically
        ``database_manager.get_session``).
    yf_client:
        Configured ``YFinanceClient`` instance (caller owns lifecycle).
    tickers:
        Canonical benchmark list — usually ``settings.benchmark_tickers``.
    stale_days:
        Re-seed thresholds. Latest price date older than ``today − stale_days``
        triggers a refetch.
    today:
        Override for tests; defaults to ``date.today()``.

    Returns
    -------
    Diagnostic dict with keys ``checked``, ``seeded``, ``up_to_date``, ``errors``.
    Always returns — caller does not need to handle exceptions.
    """
    today = today or date.today()
    summary: dict[str, object] = {
        "checked": len(tickers),
        "seeded": [],
        "up_to_date": [],
        "errors": [],
    }

    if not tickers:
        logger.info("benchmark_bootstrap: no tickers configured, skipping")
        return summary

    try:
        with session_factory() as session:
            repo = YFinanceRepository(session)
            coverage = repo.get_benchmark_coverage(tickers)
    except Exception as exc:
        # Surfaced, not swallowed: logged with traceback and recorded in
        # summary["errors"] so the caller sees the failure in the result.
        logger.exception("benchmark_bootstrap: coverage query failed")
        summary["errors"] = [f"coverage_query: {exc}"]
        return summary

    delta = _compute_delta(coverage, today, stale_days)
    summary["up_to_date"] = [t for t in tickers if t not in delta]

    if not delta:
        logger.info(
            "benchmark_bootstrap: all %d benchmarks up to date (stale_days=%d)",
            len(tickers),
            stale_days,
        )
        return summary

    logger.info(
        "benchmark_bootstrap: seeding %d/%d benchmarks (missing or stale): %s",
        len(delta),
        len(tickers),
        delta,
    )
    try:
        result = seed_reference_indices(delta, yf_client)
        summary["seeded"] = delta
        summary["errors"] = list(result.errors)
        logger.info(
            "benchmark_bootstrap: seeded=%d created=%d already=%d errors=%d",
            result.tickers_processed,
            result.instruments_created,
            result.instruments_already_present,
            len(result.errors),
        )
    except Exception as exc:
        # Surfaced, not swallowed: logged with traceback and recorded in
        # summary["errors"]; partial bootstrap progress is still returned.
        logger.exception("benchmark_bootstrap: seed_reference_indices raised")
        summary["errors"] = [f"seed: {exc}"]

    return summary
