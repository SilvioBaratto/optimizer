"""``get_prices``: the first node of the allocator critical path.

Reads daily price history out of the shared ``portopt_db`` layer via
``YFinanceRepository`` and hands the agent a JSON-serialisable shape/coverage
**summary** plus a bounded set of per-ticker statistics (level, horizon changes,
realised volatility, drawdown) — never the raw price matrix. The tool is a pure
function of ``(session, asof, tickers, field)``: same seeded data + same args ⇒
identical output.

Contract (via ``tool_envelope``):

* empty ``tickers`` or an unknown price column ⇒ ``{ok: false, error}``;
* a ticker with no instrument row or no priced days ⇒ **flagged** in
  ``data["missing"]``, never raised;
* every other failure (e.g. a malformed ``asof``) is caught by the envelope and
  returned as ``{ok: false, error}``.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository

from fund.performance import convert_prices_to_base
from fund.tools._base import (
    ToolResult,
    coerce_date,
    err,
    ok,
    summarize_frame,
    tool_envelope,
)
from fund.tools._stats import level_stats, price_stats

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# OHLCV-and-actions columns that live on ``PriceHistory`` and can be pulled as a
# numeric series. A request for anything else is a caller error, not a crash.
_PRICE_FIELDS: frozenset[str] = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dividends",
        "stock_splits",
        "capital_gains",
    }
)


# Columns quoted in the listing currency. Only these are FX-converted: volume and
# split ratios are unit-free, and an FX rate would corrupt them.
_PRICE_LEVEL_FIELDS: frozenset[str] = frozenset({"open", "high", "low", "close"})

# A series whose latest bar lags ``asof`` by more than this many calendar days is
# not tradable data: it covers the longest exchange closures (Golden Week, ~8 days)
# but not a halted, delisted or un-ingested name, whose forward-filled tail would
# read as a zero-variance asset to the optimiser.
_MAX_STALE_DAYS = 10


def load_price_frame(
    session: Session,
    asof: dt.date | str,
    tickers: Sequence[str],
    *,
    field: str = "close",
    base_currency: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a wide price frame for ``tickers`` as of ``asof``, plus the missing.

    Shared loader behind ``get_prices`` and the moment/optimize tools, so every
    allocator node re-derives the same panel from the same DB state. The
    frame carries one column per *priced* ticker (in requested order) on a sorted
    ``DatetimeIndex``. ``missing`` lists requested tickers with no instrument row
    or no priced days, plus those that are not investable at ``asof``: delisted on
    or before it, or stale (latest value more than ``_MAX_STALE_DAYS`` calendar days
    behind it; for a sparse non-price column, the latest bar of any column). Such a
    name is excluded rather than forward-filled into a flat, zero-variance tail
    that a minimum-variance optimiser would over-weight. The caller validates
    ``field`` and non-empty ``tickers``.

    Args:
        session: A sync ``portopt_db`` session; the loader does not own it.
        asof: Inclusive upper bound; rows strictly after it are excluded.
        tickers: yfinance tickers, in the order the panel columns should follow.
        field: Which ``PriceHistory`` column to pull (default ``"close"``).
        base_currency: ISO-4217 code to express price-level columns in.
            ``price_history`` stores each listing in its own currency
            (``GBp``/``USD``/...), so an allocator that derives returns from the raw
            panel optimises over a different series than the NAV tracker marks;
            passing the mandate's base currency FX-converts (and rescales minor
            units) first. ``None`` leaves the panel in listing currency, which is
            only sound for a single-currency universe.

    Returns:
        ``(frame, missing)`` — the wide price frame and the absent, delisted or
        stale tickers.

    Raises:
        FxCoverageError: ``base_currency`` is set, the panel holds foreign tickers
            and none of their currencies has an FX rate under that base.
    """
    end_date = coerce_date(asof)
    repo = YFinanceRepository(session)

    series_by_ticker: dict[str, pd.Series] = {}
    currency_map: dict[str, str] = {}
    excluded: list[str] = []
    for ticker in tickers:
        instrument = repo.get_instrument_by_yfinance_ticker(ticker)
        if instrument is None:
            continue
        if instrument.delisted_at is not None and instrument.delisted_at <= end_date:
            excluded.append(ticker)
            continue
        rows = repo.get_price_history(instrument.id, end_date=end_date)
        values = {
            row.date: float(getattr(row, field))
            for row in rows
            if getattr(row, field) is not None
        }
        if values:
            # A price level is dense, so its own last value is the freshness test: a
            # recent row whose close is NULL must not hide a weeks-old last close.
            # A sparse column (dividends) is legitimately NULL on most days, so it
            # is judged on the latest bar of any column instead.
            last_bar = (
                max(values)
                if field in _PRICE_LEVEL_FIELDS
                else max(row.date for row in rows)
            )
            if (end_date - last_bar).days > _MAX_STALE_DAYS:
                excluded.append(ticker)
                continue
            series_by_ticker[ticker] = pd.Series(values)
            unit = next((r.price_unit for r in rows if r.price_unit), None)
            unit = unit or instrument.currency_code
            if unit:
                currency_map[ticker] = unit

    if excluded:
        logger.warning(
            "price panel as of %s excludes delisted/stale tickers: %s",
            end_date,
            excluded,
        )

    frame = pd.DataFrame(series_by_ticker)
    if not frame.empty:
        # A DatetimeIndex lets ``summarize_frame`` report the panel's date span.
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        if base_currency is not None and field in _PRICE_LEVEL_FIELDS:
            frame = convert_prices_to_base(
                session, frame, currency_map, base_currency=base_currency
            )
    priced = set(frame.columns)
    missing = [ticker for ticker in tickers if ticker not in priced]
    return frame, missing


@tool_envelope
def get_prices(
    session: Session,
    asof: dt.date | str,
    tickers: Sequence[str],
    *,
    field: str = "close",
) -> ToolResult:
    """Return a price-panel summary and per-ticker statistics as of ``asof``.

    Args:
        session: A sync ``portopt_db`` session; the tool does not own it.
        asof: Inclusive upper bound; rows strictly after it are excluded (no
            look-ahead). Accepts a ``date`` or an ISO ``YYYY-MM-DD`` string.
        tickers: yfinance tickers to resolve, in the order the panel columns
            should follow.
        field: Which ``PriceHistory`` column to pull (default ``"close"``).

    Returns:
        ``ok`` with a shape/coverage summary of the wide price frame, plus
        ``asof``, ``field``, ``requested``, ``missing`` (requested tickers absent
        from the panel) and ``stats``: per priced ticker its ``last`` value and
        ``last_date``, ``change_1m``/``change_3m``/``change_12m`` and, for the
        price-level columns, the trailing-year ``realised_vol``, ``drawdown`` and
        ``max_drawdown`` (non-negative fractions below the running peak: ``0.2``
        is 20% down, as for every ``max_drawdown`` in ``fund``). Statistics are in
        each listing's own currency (returns
        are unit-free; levels are not comparable across currencies). Any other
        column reports level differences and a ``zscore`` instead. ``err`` on empty
        ``tickers`` or an unknown ``field``.
    """
    if not tickers:
        return err("no tickers requested")
    if field not in _PRICE_FIELDS:
        return err(f"unknown column {field!r}; expected one of {sorted(_PRICE_FIELDS)}")

    frame, missing = load_price_frame(session, asof, tickers, field=field)

    summary = summarize_frame(frame, name="prices")
    summary["asof"] = coerce_date(asof).isoformat()
    summary["field"] = field
    summary["requested"] = list(tickers)
    summary["missing"] = missing
    summary["stats"] = (
        price_stats(frame) if field in _PRICE_LEVEL_FIELDS else level_stats(frame)
    )
    return ok(summary)


__all__ = ["get_prices", "load_price_frame"]
