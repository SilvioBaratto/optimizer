"""Trading 212 annotation step (SPEC D14).

Runs *after* the yfinance universe build. Trading 212 is a tradability overlay on
that universe, never its source: this step attaches each T212 ticker to the
matching yfinance instrument by ISIN and records in ``is_t212_tradeable`` whether
the broker's catalogue carries the instrument at all. Because the Screener build
carries no ISIN, ISIN is fetched lazily here (only for instruments without a
stored one) and matched against T212's ``isin -> ticker`` metadata.

The flag is tri-state. An instrument whose ISIN is unknown keeps whatever it had
(NULL, "not evaluated"), because Yahoo having no ISIN says nothing about the
broker. The universe itself is never narrowed here; a run that must be executable
at the broker restricts itself with
``UniverseRepository.get_t212_tradeable_tickers``.

Skips cleanly when Trading 212 is not configured, leaving every flag untouched.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from portopt_db.repositories.universe.pinned_instrument_repository import (
    PinnedInstrumentRepository,
)
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

# Lookups take about a second each, so writes are committed every this many
# instruments rather than once at the end: a crash loses at most one batch and row
# locks are held for one short run of UPDATEs, not for the hours the loop lasts.
_COMMIT_EVERY = 100


class IsinLookupError(RuntimeError):
    """The ISIN source failed (throttled, unreachable), which is not "no ISIN"."""


@dataclass(frozen=True)
class _Target:
    """Plain snapshot of an active instrument, readable after its session closed."""

    ticker: str
    exchange_id: Any
    isin: str | None
    currency_code: str | None
    t212_ticker: str | None
    is_t212_tradeable: bool | None


@dataclass(frozen=True)
class _Update:
    """The pending writes for one instrument.

    Attributes:
        ticker: Instrument ticker within its exchange.
        exchange_id: Owning exchange row id.
        t212_ticker: Value for ``instruments.t212_ticker``; ``None`` clears it.
        write_ticker: Whether ``t212_ticker`` is written at all.
        tradeable: Value for ``instruments.is_t212_tradeable``; ``None`` leaves the
            flag as it is.
    """

    ticker: str
    exchange_id: Any
    t212_ticker: str | None
    write_ticker: bool
    tradeable: bool | None


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


def _plan_update(
    target: _Target, isin: str | None, isin_to_t212: dict[str, dict[str, str | None]]
) -> _Update | None:
    if not isin:
        return None
    lines = isin_to_t212.get(isin, {})
    t212_ticker = _pick_t212_ticker(lines, target.currency_code)
    # A known ISIN with no unambiguous T212 line proves a stored cross-reference
    # stale; a missing ISIN proves nothing.
    write_ticker = not (t212_ticker is None and target.t212_ticker is None)
    tradeable = _is_tradeable(isin, isin_to_t212)
    write_flag = tradeable is not None and target.is_t212_tradeable is not tradeable
    if not write_ticker and not write_flag:
        return None
    return _Update(
        target.ticker,
        target.exchange_id,
        t212_ticker,
        write_ticker,
        tradeable if write_flag else None,
    )


def _is_tradeable(
    isin: str, isin_to_t212: dict[str, dict[str, str | None]]
) -> bool | None:
    """Whether the broker catalogue carries ``isin``, or ``None`` if it proves nothing.

    Any line counts, even when the choice among several stays ambiguous: the
    instrument is tradeable, only its ``t212_ticker`` is unknown. A catalogue with
    no ISIN-bearing line at all (an empty answer) cannot show that anything is
    absent, so it flags nothing rather than marking the whole universe untradeable.
    """
    if not isin_to_t212:
        return None
    return isin in isin_to_t212


def _apply_updates(updates: list[_Update]) -> tuple[int, int]:
    """Write one batch in its own short transaction.

    Returns:
        ``(annotated, cleared)`` row counts for the batch.
    """
    if not updates:
        return 0, 0
    annotated = cleared = 0
    with database_manager.get_session() as session:
        repo = UniverseRepository(session)
        for update in updates:
            if update.write_ticker and repo.set_t212_ticker(
                ticker=update.ticker,
                exchange_id=update.exchange_id,
                t212_ticker=update.t212_ticker,
            ):
                if update.t212_ticker is None:
                    cleared += 1
                else:
                    annotated += 1
            if update.tradeable is not None:
                repo.set_t212_tradeable(
                    ticker=update.ticker,
                    exchange_id=update.exchange_id,
                    tradeable=update.tradeable,
                )
        session.commit()
    return annotated, cleared


def run_t212_annotate(
    *,
    isin_lookup: IsinLookup | None = None,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Attach T212 tickers to yfinance instruments by ISIN and flag tradability.

    Args:
        isin_lookup: ``ticker -> ISIN`` resolver (defaults to a yfinance lookup).
        on_progress: Optional progress callback.

    Pinned instruments are left out entirely: their ``t212_ticker`` is set by the
    operator (``add-instrument --t212-ticker``), and the ISIN match that drives
    this step would clear it for lack of an ISIN or pick a wrong-currency twin.

    Returns:
        ``{"annotated": int, "total": int, "skipped": bool, "lookup_errors": int,
        "cleared": int, "pinned_skipped": int, "tradeable": int, "untradeable":
        int}``; ``cleared`` counts stale T212 tickers removed, ``total`` excludes
        the ``pinned_skipped`` rows. ``tradeable`` and ``untradeable`` count the
        instruments whose ISIN was matched against the catalogue this run, whether
        or not their flag changed; the rest of ``total`` is "not evaluated".

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

    # Snapshot first and look up with no session open: holding a connection (and
    # the row locks of every UPDATE so far) across thousands of network calls would
    # block concurrent writers on ``instruments``.
    with database_manager.get_session() as session:
        pinned = PinnedInstrumentRepository(session).pinned_tickers()
        active = UniverseRepository(session).get_active_instruments()
        targets = [
            _Target(
                i.ticker,
                i.exchange_id,
                i.isin,
                i.currency_code,
                i.t212_ticker,
                i.is_t212_tradeable,
            )
            for i in active
            if i.yfinance_ticker not in pinned
        ]
        pinned_skipped = len(active) - len(targets)
    if pinned_skipped:
        logger.info(
            "t212_annotate: left %d pinned instruments untouched", pinned_skipped
        )
    total = len(targets)

    annotated = 0
    cleared = 0
    tradeable = 0
    untradeable = 0
    lookup_errors = 0
    consecutive_errors = 0
    pending: list[_Update] = []
    for i, target in enumerate(targets, start=1):
        isin = target.isin
        if not isin:
            try:
                isin = lookup(target.ticker)
                consecutive_errors = 0
            except Exception as exc:
                lookup_errors += 1
                consecutive_errors += 1
                logger.warning(
                    "t212_annotate: ISIN lookup failed for %s: %s", target.ticker, exc
                )
                if consecutive_errors >= _MAX_CONSECUTIVE_LOOKUP_ERRORS:
                    break
        update = _plan_update(target, isin, isin_to_t212)
        if update is not None:
            pending.append(update)
        verdict = _is_tradeable(isin, isin_to_t212) if isin else None
        if verdict is True:
            tradeable += 1
        elif verdict is False:
            untradeable += 1
        if i % _COMMIT_EVERY == 0:
            done, dropped = _apply_updates(pending)
            annotated += done
            cleared += dropped
            pending = []
        on_progress(current=i, total=total, current_stock=target.ticker)
    done, dropped = _apply_updates(pending)
    annotated += done
    cleared += dropped

    if consecutive_errors >= _MAX_CONSECUTIVE_LOOKUP_ERRORS:
        msg = (
            f"{consecutive_errors} consecutive ISIN lookups failed; stopped after "
            f"annotating {annotated}/{total} instruments"
        )
        raise IsinLookupError(msg)

    logger.info(
        "t212_annotate: annotated %d/%d instruments, cleared %d, %d tradeable, "
        "%d not on the broker (%d lookup errors)",
        annotated,
        total,
        cleared,
        tradeable,
        untradeable,
        lookup_errors,
    )
    summary = {
        "annotated": annotated,
        "total": total,
        "skipped": False,
        "lookup_errors": lookup_errors,
        "cleared": cleared,
        "pinned_skipped": pinned_skipped,
        "tradeable": tradeable,
        "untradeable": untradeable,
    }
    on_progress(status="completed", result=summary)
    return summary
