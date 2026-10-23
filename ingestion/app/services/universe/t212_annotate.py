"""Trading 212 annotation step (SPEC D14).

Runs *after* the yfinance universe build. Trading 212 is an add-on: this step
attaches each T212 ticker to the matching yfinance instrument by ISIN. Because
the Screener build carries no ISIN, ISIN is fetched lazily here (one call per
active instrument) and matched against T212's ``isin -> ticker`` metadata.

Skips cleanly when Trading 212 is not configured.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from portopt_db.repositories.universe.universe_repository import UniverseRepository

from app.database import database_manager
from app.services._shared import ProgressCallback, _noop
from app.services.market_data.yfinance import RequestLostError, get_yfinance_client
from app.services.universe.canonical import split_currency
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


def _pick_t212_ticker(
    candidates: dict[str, str | None], currency: str | None
) -> str | None:
    """Choose the T212 line that matches an instrument's quote currency.

    Trading 212 lists one ISIN on several venues and currencies, so the ISIN alone
    does not identify the line. A sole line is unambiguous and returned as is. With
    several, the one quoted in the instrument's own unit wins (GBp and GBX are the
    same unit), then the one in the same major currency; a tier holding more than one
    line, or no tier holding any, is ambiguous and yields ``None``.

    Args:
        candidates: T212 ticker -> its ``currencyCode`` for one ISIN.
        currency: The instrument's quote currency, if known.

    Returns:
        The matching T212 ticker, or ``None`` when there is no unambiguous match.
    """
    if len(candidates) == 1:
        return next(iter(candidates))
    if currency is None:
        return None
    unit = split_currency(currency)
    tiers = (
        [t for t, c in candidates.items() if c and split_currency(c) == unit],
        [t for t, c in candidates.items() if c and split_currency(c)[0] == unit[0]],
    )
    for tier in tiers:
        if tier:
            return tier[0] if len(tier) == 1 else None
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
        ``{"annotated": int, "total": int, "skipped": bool, "lookup_errors": int,
        "cleared": int}``; ``cleared`` counts stale T212 tickers removed.

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

    isin_to_t212: defaultdict[str, dict[str, str | None]] = defaultdict(dict)
    for inst in client.get_instruments():
        t212_isin = inst.get("isin")
        t212_symbol = inst.get("ticker")
        if t212_isin and t212_symbol:
            isin_to_t212[t212_isin][t212_symbol] = inst.get("currencyCode")

    annotated = 0
    cleared = 0
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
            t212_ticker = (
                _pick_t212_ticker(
                    isin_to_t212.get(looked_up, {}), instrument.currency_code
                )
                if looked_up
                else None
            )
            # A known ISIN with no unambiguous T212 line proves the stored
            # cross-reference stale; a missing ISIN proves nothing.
            stale = (
                t212_ticker is None
                and bool(looked_up)
                and instrument.t212_ticker is not None
            )
            if (t212_ticker is not None or stale) and repo.set_t212_ticker(
                ticker=instrument.ticker,
                exchange_id=instrument.exchange_id,
                t212_ticker=t212_ticker,
            ):
                if stale:
                    cleared += 1
                else:
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
        "t212_annotate: annotated %d/%d instruments, cleared %d (%d lookup errors)",
        annotated,
        total,
        cleared,
        lookup_errors,
    )
    summary = {
        "annotated": annotated,
        "total": total,
        "skipped": False,
        "lookup_errors": lookup_errors,
        "cleared": cleared,
    }
    on_progress(status="completed", result=summary)
    return summary
