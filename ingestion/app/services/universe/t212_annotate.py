"""Trading 212 annotation step (SPEC D14).

Runs *after* the yfinance universe build. Trading 212 is an add-on: this step
attaches each T212 ticker to the matching yfinance instrument by ISIN. Because
the Screener build carries no ISIN, ISIN is fetched lazily here (one call per
active instrument) and matched against T212's ``isin -> ticker`` metadata.

Skips cleanly when Trading 212 is not configured.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from app.database import database_manager
from app.repositories.universe.universe_repository import UniverseRepository
from app.services._shared import ProgressCallback, _noop
from app.services.universe.universe_build_service import (
    Trading212NotConfiguredError,
    build_trading212_client,
)

logger = logging.getLogger(__name__)

IsinLookup = Callable[[str], str | None]


def _default_isin_lookup(ticker: str) -> str | None:
    """Fetch an instrument's ISIN from yfinance (best effort; None on failure)."""
    import yfinance as yf

    try:
        return yf.Ticker(ticker).isin or None
    except Exception:  # best-effort lazy lookup; never fail the whole step
        return None


def run_t212_annotate(
    *,
    isin_lookup: IsinLookup | None = None,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Attach T212 tickers to yfinance instruments by ISIN.

    Args:
        isin_lookup: ``ticker -> ISIN`` resolver (defaults to a yfinance lookup).
        on_progress: Optional progress callback.

    Returns:
        ``{"annotated": int, "total": int, "skipped": bool}``.
    """
    try:
        client = build_trading212_client()
    except Trading212NotConfiguredError as exc:
        logger.info("t212_annotate: skipped — %s", exc)
        on_progress(status="skipped")
        return {"annotated": 0, "total": 0, "skipped": True}

    lookup = isin_lookup or _default_isin_lookup

    isin_to_t212: dict[str, str] = {}
    for inst in client.get_instruments():
        isin = inst.get("isin")
        t212_ticker = inst.get("ticker")
        if isin and t212_ticker:
            isin_to_t212[isin] = t212_ticker

    annotated = 0
    with database_manager.get_session() as session:
        repo = UniverseRepository(session)
        instruments = list(repo.get_active_instruments())
        total = len(instruments)
        for i, instrument in enumerate(instruments, start=1):
            isin = lookup(instrument.ticker)
            t212_ticker = isin_to_t212.get(isin) if isin else None
            if t212_ticker and repo.set_t212_ticker(
                ticker=instrument.ticker,
                exchange_id=instrument.exchange_id,
                t212_ticker=t212_ticker,
            ):
                annotated += 1
            on_progress(current=i, total=total, current_stock=instrument.ticker)
        session.commit()

    logger.info("t212_annotate: annotated %d/%d instruments", annotated, total)
    summary = {"annotated": annotated, "total": total, "skipped": False}
    on_progress(status="completed", result=summary)
    return summary
