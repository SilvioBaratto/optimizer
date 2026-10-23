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

from portopt_db.repositories.universe.universe_repository import UniverseRepository

from app.database import database_manager
from app.services._shared import ProgressCallback, _noop
from app.services.market_data.yfinance import RequestLostError, get_yfinance_client
from app.services.universe.universe_build_service import (
    Trading212NotConfiguredError,
    build_trading212_client,
)

logger = logging.getLogger(__name__)

IsinLookup = Callable[[str], str | None]

# Yahoo having no ISIN for a ticker is a normal answer (None); a lookup that raises
# is a lost request. This many lost requests in a row means the source is down, so
# the step stops and fails rather than grinding through the universe and reporting
# a clean run with nothing annotated.
_MAX_CONSECUTIVE_LOOKUP_ERRORS = 20


class IsinLookupError(RuntimeError):
    """The ISIN source failed (throttled, unreachable), which is not "no ISIN"."""


def _default_isin_lookup(ticker: str) -> str | None:
    """Fetch an instrument's ISIN through the shared, throttled yfinance client.

    Args:
        ticker: yfinance symbol of the instrument.

    Returns:
        The ISIN, or ``None`` when Yahoo has none for the symbol.

    Raises:
        IsinLookupError: Every attempt failed (rate limit, DNS, TLS, a garbled
            body), so Yahoo never said whether it holds an ISIN.
    """
    try:
        return get_yfinance_client().metadata.fetch_isin(ticker, raise_on_lost=True)
    except RequestLostError as exc:
        msg = f"ISIN lookup for {ticker} lost: {exc}"
        raise IsinLookupError(msg) from exc


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
        ``{"annotated": int, "total": int, "skipped": bool, "lookup_errors": int}``.

    Raises:
        IsinLookupError: Too many ISIN lookups failed in a row; annotations made
            before the outage are committed.
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
    lookup_errors = 0
    consecutive_errors = 0
    with database_manager.get_session() as session:
        repo = UniverseRepository(session)
        instruments = list(repo.get_active_instruments())
        total = len(instruments)
        for i, instrument in enumerate(instruments, start=1):
            looked_up: str | None = None
            try:
                looked_up = lookup(instrument.ticker)
                consecutive_errors = 0
            except Exception as exc:
                lookup_errors += 1
                consecutive_errors += 1
                logger.warning(
                    "t212_annotate: ISIN lookup failed for %s: %s",
                    instrument.ticker,
                    exc,
                )
                if consecutive_errors >= _MAX_CONSECUTIVE_LOOKUP_ERRORS:
                    break
            t212_ticker = isin_to_t212.get(looked_up) if looked_up else None
            if t212_ticker and repo.set_t212_ticker(
                ticker=instrument.ticker,
                exchange_id=instrument.exchange_id,
                t212_ticker=t212_ticker,
            ):
                annotated += 1
            on_progress(current=i, total=total, current_stock=instrument.ticker)
        session.commit()

    if consecutive_errors >= _MAX_CONSECUTIVE_LOOKUP_ERRORS:
        msg = (
            f"{consecutive_errors} consecutive ISIN lookups failed; stopped after "
            f"annotating {annotated}/{total} instruments"
        )
        raise IsinLookupError(msg)

    logger.info(
        "t212_annotate: annotated %d/%d instruments (%d lookup errors)",
        annotated,
        total,
        lookup_errors,
    )
    summary = {
        "annotated": annotated,
        "total": total,
        "skipped": False,
        "lookup_errors": lookup_errors,
    }
    on_progress(status="completed", result=summary)
    return summary
