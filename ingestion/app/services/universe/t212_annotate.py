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

A pin does not exempt an instrument from the tradability judgement: the broker
scope of a mandate reads the flag for exactly the tickers a mandate pins. Only the
broker line itself (``t212_ticker``) and a looked-up ISIN are left alone on the
operator's own (``manual``) pins, because the operator curated them.

Skips cleanly when Trading 212 is not configured, leaving every flag untouched.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

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

# The pin source of ``add-instrument --pin``. Mandate pins (``mandate:<id>``) only
# keep a name in the universe; the operator's own pins also carry a hand-picked
# broker line.
_OPERATOR_PIN_SOURCE = "manual"


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
    operator_pinned: bool


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
        new_isin: ISIN looked up for a row that had none, to be stored; ``None``
            when there is nothing to store.
    """

    ticker: str
    exchange_id: Any
    t212_ticker: str | None
    write_ticker: bool
    tradeable: bool | None
    new_isin: str | None


class _Written(NamedTuple):
    """Row counts for one batch of writes."""

    annotated: int
    cleared: int
    isin_stored: int


@dataclass
class _Tally:
    """Running counts of one annotation run.

    Attributes:
        annotated: T212 tickers written onto instruments.
        cleared: Stale T212 tickers removed.
        isin_stored: Looked-up ISINs written onto rows that had none.
        tradeable: Instruments whose ISIN the catalogue lists.
        untradeable: Instruments whose ISIN the catalogue does not list.
        lookup_errors: ISIN lookups that failed.
        consecutive_errors: ISIN lookups that failed since the last success.
    """

    annotated: int = 0
    cleared: int = 0
    isin_stored: int = 0
    tradeable: int = 0
    untradeable: int = 0
    lookup_errors: int = 0
    consecutive_errors: int = 0

    @property
    def outage(self) -> bool:
        """Whether enough lookups failed in a row to call the ISIN source down."""
        return self.consecutive_errors >= _MAX_CONSECUTIVE_LOOKUP_ERRORS

    def add_written(self, written: _Written) -> None:
        """Add one committed batch's row counts."""
        self.annotated += written.annotated
        self.cleared += written.cleared
        self.isin_stored += written.isin_stored

    def count_verdict(self, verdict: bool | None) -> None:
        """Count one instrument's tradability; ``None`` was not evaluated."""
        if verdict is True:
            self.tradeable += 1
        elif verdict is False:
            self.untradeable += 1

    def summary(self, *, total: int, pinned_skipped: int) -> dict[str, Any]:
        """Return the step's result for a run that completed.

        Args:
            total: Active instruments the run covered.
            pinned_skipped: Rows whose broker line was left to the operator.

        Returns:
            The summary ``run_t212_annotate`` documents, in its key order.
        """
        return {
            "annotated": self.annotated,
            "total": total,
            "skipped": False,
            "lookup_errors": self.lookup_errors,
            "cleared": self.cleared,
            "pinned_skipped": pinned_skipped,
            "tradeable": self.tradeable,
            "untradeable": self.untradeable,
            "isin_stored": self.isin_stored,
        }


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
    # stale; a missing ISIN proves nothing, and neither does a catalogue that
    # lists no ISIN at all (an empty or reshaped answer), which would otherwise
    # clear every stored line.
    write_ticker = (
        bool(isin_to_t212)
        and not target.operator_pinned
        and not (t212_ticker is None and target.t212_ticker is None)
    )
    tradeable = _is_tradeable(isin, isin_to_t212)
    write_flag = tradeable is not None and target.is_t212_tradeable is not tradeable
    # An ISIN the row did not have is one Yahoo just supplied; keep it so the next
    # run neither repeats the request nor leaves ISIN-keyed checks blind to the row.
    # The operator's own rows keep exactly the identifiers they were given.
    new_isin = None if target.isin or target.operator_pinned else isin
    if not (write_ticker or write_flag or new_isin):
        return None
    return _Update(
        target.ticker,
        target.exchange_id,
        t212_ticker,
        write_ticker,
        tradeable if write_flag else None,
        new_isin,
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


def _apply_updates(updates: list[_Update]) -> _Written:
    """Write one batch in its own short transaction.

    Returns:
        The ``annotated``, ``cleared`` and ``isin_stored`` row counts for the batch.
    """
    if not updates:
        return _Written(0, 0, 0)
    annotated = cleared = isin_stored = 0
    with database_manager.get_session() as session:
        repo = UniverseRepository(session)
        for update in updates:
            if update.new_isin and repo.set_isin_if_missing(
                ticker=update.ticker,
                exchange_id=update.exchange_id,
                isin=update.new_isin,
            ):
                isin_stored += 1
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
    return _Written(annotated, cleared, isin_stored)


def run_t212_annotate(
    *,
    isin_lookup: IsinLookup | None = None,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Attach T212 tickers to yfinance instruments by ISIN and flag tradability.

    Args:
        isin_lookup: ``ticker -> ISIN`` resolver (defaults to a yfinance lookup).
        on_progress: Optional progress callback.

    Every active instrument is judged for tradability, pinned or not. The ones
    the operator pinned themselves (``add-instrument --pin``) keep their
    ``t212_ticker`` and ISIN as given: the ISIN match that drives this step would
    clear a hand-set broker line for lack of an ISIN or pick a wrong-currency twin.
    A catalogue with no ISIN-bearing line proves nothing, so it also leaves every
    stored ``t212_ticker`` alone.

    Returns:
        ``{"annotated": int, "total": int, "skipped": bool, "lookup_errors": int,
        "cleared": int, "pinned_skipped": int, "tradeable": int, "untradeable":
        int, "isin_stored": int}``; ``cleared`` counts stale T212 tickers removed,
        ``pinned_skipped`` the rows whose broker line was left to the operator.
        ``tradeable`` and ``untradeable`` count the instruments whose ISIN was
        matched against the catalogue this run, whether or not their flag
        changed; the rest of ``total`` is "not evaluated". ``isin_stored`` counts
        ISINs looked up for rows that had none and written onto them.

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
    isin_to_t212 = _isin_catalogue(client.get_instruments())
    targets = _snapshot_targets()
    pinned_skipped = sum(t.operator_pinned for t in targets)
    _log_scope(pinned_skipped, catalogue_lists_isins=bool(isin_to_t212))
    tally = _annotate(targets, lookup, isin_to_t212, on_progress)
    return _finish(tally, len(targets), pinned_skipped, on_progress)


def _isin_catalogue(
    instruments: list[dict[str, Any]],
) -> defaultdict[str, dict[str, str | None]]:
    """Map each ISIN the broker lists to its T212 lines and their currencies.

    A line without an ISIN or a ticker cannot be matched and is left out.
    """
    isin_to_t212: defaultdict[str, dict[str, str | None]] = defaultdict(dict)
    for inst in instruments:
        t212_isin = inst.get("isin")
        t212_symbol = inst.get("ticker")
        if t212_isin and t212_symbol:
            isin_to_t212[t212_isin][t212_symbol] = inst.get("currencyCode")
    return isin_to_t212


def _snapshot_targets() -> list[_Target]:
    """Read every active instrument, marking the operator's own pins."""
    # Snapshot first and look up with no session open: holding a connection (and
    # the row locks of every UPDATE so far) across thousands of network calls would
    # block concurrent writers on ``instruments``.
    with database_manager.get_session() as session:
        operator_pins = PinnedInstrumentRepository(session).pinned_tickers(
            source=_OPERATOR_PIN_SOURCE
        )
        active = UniverseRepository(session).get_active_instruments()
        return [
            _Target(
                i.ticker,
                i.exchange_id,
                i.isin,
                i.currency_code,
                i.t212_ticker,
                i.is_t212_tradeable,
                i.yfinance_ticker in operator_pins,
            )
            for i in active
        ]


def _log_scope(pinned_skipped: int, *, catalogue_lists_isins: bool) -> None:
    """Log what the run will leave alone before it starts."""
    if pinned_skipped:
        logger.info(
            "t212_annotate: left the broker ticker of %d pinned instruments untouched",
            pinned_skipped,
        )
    if not catalogue_lists_isins:
        logger.warning(
            "t212_annotate: the Trading 212 catalogue lists no ISIN, so stored "
            "broker tickers and tradability flags are left as they are"
        )


def _resolve_isin(target: _Target, lookup: IsinLookup, tally: _Tally) -> str | None:
    """Return the instrument's ISIN, looking it up when none is stored.

    A failed lookup is counted in ``tally`` and leaves the stored (empty) value.
    """
    if target.isin:
        return target.isin
    try:
        isin = lookup(target.ticker)
    except Exception as exc:
        tally.lookup_errors += 1
        tally.consecutive_errors += 1
        logger.warning(
            "t212_annotate: ISIN lookup failed for %s: %s", target.ticker, exc
        )
        return target.isin
    tally.consecutive_errors = 0
    return isin


def _annotate(
    targets: list[_Target],
    lookup: IsinLookup,
    isin_to_t212: dict[str, dict[str, str | None]],
    on_progress: ProgressCallback,
) -> _Tally:
    """Judge each instrument and write the changes in batches.

    Stops at the lookup that makes the ISIN source an outage, without
    reporting that instrument; the changes planned before it are still written.
    """
    tally = _Tally()
    pending: list[_Update] = []
    for i, target in enumerate(targets, start=1):
        isin = _resolve_isin(target, lookup, tally)
        if tally.outage:
            break
        update = _plan_update(target, isin, isin_to_t212)
        if update is not None:
            pending.append(update)
        tally.count_verdict(_is_tradeable(isin, isin_to_t212) if isin else None)
        if i % _COMMIT_EVERY == 0:
            tally.add_written(_apply_updates(pending))
            pending = []
        on_progress(current=i, total=len(targets), current_stock=target.ticker)
    tally.add_written(_apply_updates(pending))
    return tally


def _finish(
    tally: _Tally, total: int, pinned_skipped: int, on_progress: ProgressCallback
) -> dict[str, Any]:
    """Fail an outage run, otherwise log, report and return the summary.

    Raises:
        IsinLookupError: The run stopped because the ISIN source was down.
    """
    if tally.outage:
        msg = (
            f"{tally.consecutive_errors} consecutive ISIN lookups failed; stopped "
            f"after annotating {tally.annotated}/{total} instruments"
        )
        raise IsinLookupError(msg)
    logger.info(
        "t212_annotate: annotated %d/%d instruments, cleared %d, stored %d ISINs, "
        "%d tradeable, %d not on the broker (%d lookup errors)",
        tally.annotated,
        total,
        tally.cleared,
        tally.isin_stored,
        tally.tradeable,
        tally.untradeable,
        tally.lookup_errors,
    )
    summary = tally.summary(total=total, pinned_skipped=pinned_skipped)
    on_progress(status="completed", result=summary)
    return summary
