"""FX-rate feed (SPEC Phase 05 T0/T3).

Fetches daily foreign-exchange rates for every currency in the stored universe
against a base currency (default EUR), cross-via-USD, and upserts them into
``fx_rates``. Pure ingestion — reads ``instruments`` / ``price_history`` and
writes via ``FxRatesRepository``; it must **not** import ``optimizer`` (the
``optimizer.fx`` conversion runs only in the fund tracker).

Cross-via-USD: for a base ``B`` and quote ``Q``, the stored rate (B per 1 Q) is
``(USD per 1 Q) / (USD per 1 B)``, each leg read from Yahoo's ``{CCY}USD=X``
series (with the USD leg of USD itself being the constant 1). This covers any
currency pair without needing a direct ``{B}{Q}=X`` quote, which Yahoo often
lacks.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd
from sqlalchemy import select

from app.services._shared import ProgressCallback, _noop

logger = logging.getLogger(__name__)

# yfinance quotes some markets in a currency sub-unit (pence, cents, agorot).
# Map each to its ISO-4217 major so the rate feed keys on majors only — the
# minor-unit ÷100 scaling is the fund price reader's job, not the feed's.
_MINOR_UNIT_TO_MAJOR = {
    "GBX": "GBP",
    "GBP": "GBP",
    "ZAC": "ZAR",
    "ZAR": "ZAR",
    "ILA": "ILS",
    "ILS": "ILS",
}


def _to_major(code: str | None) -> str | None:
    """Normalize a currency / price-unit code to its ISO-4217 major unit.

    Args:
        code: A currency code or price unit (e.g. ``"USD"``, ``"GBp"``).

    Returns:
        The three-letter major code (sub-units collapsed, e.g. ``GBp`` → ``GBP``),
        or ``None`` when the input is blank or not a recognizable ISO code.
    """
    if not code:
        return None
    stripped = str(code).strip()
    if not stripped:
        return None
    mapped = _MINOR_UNIT_TO_MAJOR.get(stripped.upper())
    if mapped:
        return mapped
    upper = stripped.upper()
    if len(upper) == 3 and upper.isalpha():
        return upper
    return None


def _distinct_currencies(session: Any, base: str) -> set[str]:
    """Return the distinct major quote currencies present in the universe.

    Unions ``instruments.currency_code`` with ``price_history.price_unit`` (so a
    NULL ``currency_code`` still contributes its listing currency), normalizes to
    major units, and drops the base currency (which is never its own quote).

    Args:
        session: Open SQLAlchemy session.
        base: Base currency to exclude from the quote set.

    Returns:
        Major ISO-4217 codes to fetch rates for; empty when the universe is bare.
    """
    from portopt_db.models.market_data.yfinance_data import PriceHistory
    from portopt_db.models.universe.universe import Instrument

    majors: set[str] = set()
    for (code,) in session.execute(select(Instrument.currency_code).distinct()):
        major = _to_major(code)
        if major:
            majors.add(major)
    for (unit,) in session.execute(select(PriceHistory.price_unit).distinct()):
        major = _to_major(unit)
        if major:
            majors.add(major)
    majors.discard(base.upper())
    return majors


def _close_by_date(hist: pd.DataFrame | None) -> dict[dt.date, float]:
    """Extract a ``{date: close}`` mapping from a yfinance history frame.

    Args:
        hist: History DataFrame with a ``Close`` column and a DatetimeIndex, or
            ``None`` / empty when the fetch failed.

    Returns:
        Calendar-date to close-price mapping; empty when there is no usable data.
    """
    if hist is None or getattr(hist, "empty", True) or "Close" not in hist.columns:
        return {}
    close = hist["Close"]
    out: dict[dt.date, float] = {}
    for ts, value in zip(pd.DatetimeIndex(close.index), close.tolist(), strict=False):
        if value is None or pd.isna(value):
            continue
        out[ts.date()] = float(value)
    return out


def run_fx_fetch(
    yf_client: Any,
    *,
    base: str = "EUR",
    period: str = "5y",
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Fetch and persist daily FX rates for the universe's currencies.

    Derives the quote-currency set dynamically from the stored universe, fetches
    the USD leg (``{CCY}USD=X``) for each currency plus the base, computes
    base-per-quote rates cross-via-USD on aligned dates, and upserts them. A
    failing or empty leg is logged and skipped rather than sinking the run; if
    the base's own USD leg is unavailable nothing can be converted and the run
    completes writing no rows.

    Args:
        yf_client: Configured yfinance client exposing ``fetch_history``.
        base: Base currency the stored rates are expressed in.
        period: Lookback window passed to ``fetch_history`` (backfill depth).
        on_progress: Callback receiving progress + terminal-status updates.

    Returns:
        Dict with ``rows_total`` (rows upserted) and ``error_count``.
    """
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    from app.database import database_manager

    base = base.upper()
    now = dt.datetime.now(dt.timezone.utc)
    errors: list[str] = []

    with database_manager.get_session() as session:
        quotes = _distinct_currencies(session, base)
        if not quotes:
            result = {"rows_total": 0, "error_count": 0}
            logger.info("fx_fetch: no quote currencies in the universe; nothing to do")
            on_progress(status="completed", finished_at=now.isoformat(), result=result)
            return result

        # Every currency needs its USD leg except USD itself (leg == 1).
        needed = ({base} | quotes) - {"USD"}
        on_progress(total=len(needed))
        usd_legs: dict[str, dict[dt.date, float]] = {}
        for idx, ccy in enumerate(sorted(needed), 1):
            on_progress(current=idx, current_currency=ccy)
            symbol = f"{ccy}USD=X"
            try:
                series = _close_by_date(yf_client.fetch_history(symbol, period=period))
            except Exception as exc:  # one pair must not sink the rest
                logger.warning("fx_fetch: failed %s: %s", symbol, exc)
                errors.append(f"{symbol}: {exc}")
                continue
            if series:
                usd_legs[ccy] = series
            else:
                logger.warning("fx_fetch: no data for %s", symbol)
                errors.append(f"{symbol}: empty")

        base_leg = None if base == "USD" else usd_legs.get(base)
        base_dates = set(base_leg) if base_leg is not None else None
        if base != "USD" and base_leg is None:
            result = {"rows_total": 0, "error_count": len(errors)}
            logger.warning(
                "fx_fetch: base USD leg (%sUSD=X) unavailable; wrote no rows", base
            )
            on_progress(status="completed", finished_at=now.isoformat(), result=result)
            return result

        rows = _build_rows(base, quotes, usd_legs, base_leg, base_dates)
        repo = FxRatesRepository(session)
        written = repo.upsert_rates(rows)
        session.commit()

    result = {"rows_total": written, "error_count": len(errors)}
    logger.info("fx_fetch: upserted %d rate(s) for base %s", written, base)
    on_progress(
        status="completed",
        finished_at=now.isoformat(),
        errors=errors,
        result=result,
    )
    return result


def _build_rows(
    base: str,
    quotes: set[str],
    usd_legs: dict[str, dict[dt.date, float]],
    base_leg: dict[dt.date, float] | None,
    base_dates: set[dt.date] | None,
) -> list[dict[str, Any]]:
    """Assemble base-per-quote rate rows from the fetched USD legs.

    Args:
        base: Base currency.
        quotes: Quote currencies to emit rows for.
        usd_legs: ``{currency: {date: usd_per_unit}}`` for fetched legs.
        base_leg: The base currency's USD leg, or ``None`` when base is USD.
        base_dates: Dates the base leg covers, or ``None`` when base is USD.

    Returns:
        Upsert-ready row dicts; a quote with no usable aligned dates is skipped.
    """
    rows: list[dict[str, Any]] = []
    for quote in sorted(quotes):
        if quote == base:
            continue
        if quote == "USD":
            quote_leg: dict[dt.date, float] | None = None
            quote_dates = base_dates if base_dates is not None else set()
        else:
            quote_leg = usd_legs.get(quote)
            if quote_leg is None:
                continue
            quote_dates = set(quote_leg)
        dates = quote_dates if base_dates is None else (quote_dates & base_dates)
        for date in sorted(dates):
            base_usd = 1.0 if base_leg is None else base_leg[date]
            if base_usd == 0:
                continue
            quote_usd = 1.0 if quote_leg is None else quote_leg[date]
            rows.append(
                {
                    "base_currency": base,
                    "quote_currency": quote,
                    "date": date,
                    "rate": quote_usd / base_usd,
                }
            )
    return rows


__all__ = ["run_fx_fetch"]
