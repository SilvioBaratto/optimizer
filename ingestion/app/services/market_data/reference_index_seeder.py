"""Service that seeds reference-index instruments and fetches their price history.

Reference indices (e.g. SPY) are benchmark instruments that must exist in the
database for HMM regime detection and portfolio performance calculations, but
they are *not* members of any tradable T212 universe.  The T212 universe
builder filters for ``instrument_type == 'STOCK'`` and would silently ignore
ETFs, so seeding is handled here as a separate, explicit operation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select

from app.models.universe.universe import Exchange, Instrument
from app.repositories.market_data.yfinance_repository import YFinanceRepository
from app.services._shared._progress import ProgressCallback, _noop
from app.services.market_data.yfinance import YFinanceClient
from app.services.market_data.yfinance_data_service import YFinanceDataService

logger = logging.getLogger(__name__)


@dataclass
class SeedResult:
    """Aggregated outcome of a reference-index seeding run."""

    tickers_processed: int = 0
    instruments_created: int = 0
    instruments_already_present: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def seed_reference_indices(
    tickers: list[str],
    yf_client: YFinanceClient,
    *,
    on_progress: ProgressCallback = _noop,
) -> SeedResult:
    """Idempotently seed instrument rows and fetch price history for *tickers*.

    For each ticker the function:

    1. Opens a DB session via ``database_manager.get_session()``.
    2. Looks up or creates an :class:`~app.models.universe.Instrument` row
       (``instrument_type='ETF'``, exchange = NYSE).
    3. Calls ``YFinanceDataService.fetch_and_store`` with ``period="5y"`` and
       ``mode="incremental"`` to populate price history.
    4. Commits on success; rolls back and logs on any exception, then continues
       to the next ticker so one bad ticker cannot abort the whole batch.

    Args:
        tickers: yfinance ticker symbols to seed (e.g. ``["SPY", "QQQ"]``).
        yf_client: Configured :class:`~app.services.market_data.yfinance.YFinanceClient` instance.
        on_progress: Optional callback for job-progress updates.

    Returns:
        :class:`SeedResult` with counts and error messages.
    """
    # Import here to mirror the pattern in run_bulk_yfinance_fetch and avoid
    # circular imports at module level.
    from app.database import database_manager

    result = SeedResult(tickers_processed=len(tickers))
    all_errors: list[str] = []

    with database_manager.get_session() as session:
        repo = YFinanceRepository(session)
        service = YFinanceDataService(repo, yf_client)

        # Resolve NYSE exchange id once up front — avoids a per-ticker SELECT.
        nyse_id = session.execute(
            select(Exchange.id).where(Exchange.name == "NYSE")
        ).scalar_one_or_none()

        if nyse_id is None:
            # Reference indices are seeded independently of the T212 universe (the
            # docstring's "separate, explicit operation"), so on a cold DB the
            # NYSE exchange they attach to may not exist yet. Create it rather
            # than failing — otherwise a fresh deploy seeds no benchmarks until
            # the first universe build happens to create an NYSE row.
            nyse = Exchange(name="NYSE")
            session.add(nyse)
            session.flush()
            nyse_id = nyse.id
            logger.info("Created NYSE exchange row for reference-index seeding")

        on_progress(total=len(tickers))

        for idx, ticker in enumerate(tickers, 1):
            ticker_upper = ticker.upper()
            on_progress(current=idx, current_ticker=ticker_upper)

            try:
                instrument = repo.get_instrument_by_yfinance_ticker(ticker_upper)

                if instrument is None:
                    instrument = Instrument(
                        ticker=ticker_upper,
                        short_name=ticker_upper,
                        yfinance_ticker=ticker_upper,
                        instrument_type="ETF",
                        currency_code="USD",
                        exchange_id=nyse_id,
                    )
                    session.add(instrument)
                    # Flush to get the auto-generated id before calling fetch_and_store.
                    session.flush()
                    result.instruments_created += 1
                    logger.info("Created Instrument row for %s", ticker_upper)
                else:
                    result.instruments_already_present += 1
                    logger.debug("Instrument already present for %s", ticker_upper)

                fetch_result = service.fetch_and_store(
                    instrument_id=instrument.id,
                    yfinance_ticker=ticker_upper,
                    period="5y",
                    mode="incremental",
                    exchange_name="NYSE",
                    currency_code="USD",
                )

                for k, v in fetch_result["counts"].items():
                    result.counts[k] = result.counts.get(k, 0) + v

                for err in fetch_result["errors"]:
                    all_errors.append(f"{ticker_upper}: {err}")

                session.commit()

            except Exception as exc:
                # Per-ticker seed isolation: a failure is logged, recorded in
                # all_errors (surfaced to the caller), and rolled back so the
                # next ticker seeds cleanly — one bad ticker never aborts seed.
                logger.error("Failed to seed %s: %s", ticker_upper, exc)
                all_errors.append(f"{ticker_upper}: {exc}")
                session.rollback()

    result.errors = all_errors

    on_progress(
        status="completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        errors=all_errors,
        result={
            "tickers_processed": result.tickers_processed,
            "instruments_created": result.instruments_created,
            "instruments_already_present": result.instruments_already_present,
            "counts": result.counts,
            "error_count": len(all_errors),
        },
    )

    logger.info(
        "Reference index seeding completed: %d tickers, %d created, %d errors",
        result.tickers_processed,
        result.instruments_created,
        len(all_errors),
    )

    return result
