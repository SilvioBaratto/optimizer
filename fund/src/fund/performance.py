"""Performance tracker — NAV / returns / risk / per-holding read-outs.

The compute core behind the oversight TUI's "how are we doing?" question. It
reconstructs the paper portfolio's FX-normalized NAV time series, computes
multi-horizon returns + risk metrics versus the mandate benchmark, and marks each
holding to market. Reads ``paper_orders`` / ``positions`` / ``price_history`` /
``fx_rates`` from an injected session and FX-converts listing-currency prices to
the mandate base currency via ``optimizer.fx`` (the only package allowed to do so
in the fund).

Agent-stack-free: imports ``optimizer`` / ``skfolio`` (allowed) but never the
deep-agent stack (``deepagents`` / ``langgraph``) nor ``fund.audit`` — so the
daily NAV-mark scheduler step can lazy-import it without pulling that weight into
a bare ``import fund.scheduler``. Degrades gracefully: short/empty/absent history
yields an empty series or ``None`` fields, never a fabricated number.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NavPoint:
    """One marked-to-market portfolio value in the mandate base currency.

    Attributes:
        date: Calendar date of the mark.
        nav: Base-currency portfolio value on ``date``.
        benchmark: Benchmark level FX-normalized to base and date-aligned, or
            ``None`` when the mandate has no benchmark or none aligns.
    """

    date: dt.date
    nav: float
    benchmark: float | None


@dataclass(frozen=True)
class PerformanceSummary:
    """Multi-horizon returns + risk metrics over a portfolio's NAV series.

    Every field degrades gracefully: a horizon with insufficient history is
    ``None`` rather than a fabricated number; benchmark-relative fields are
    ``None`` when no benchmark series aligns to the window.

    Attributes:
        portfolio_id: The portfolio these metrics describe.
        returns: Trailing fractional returns keyed by horizon
            (``"1D"``/``"1M"``/``"YTD"``/``"1Y"``/``"max"``).
        excess: Portfolio-minus-benchmark return per horizon.
        tracking_error: Stddev of the portfolio-minus-benchmark return series.
        max_drawdown: Worst peak-to-trough decline (a non-negative fraction).
        volatility: Annualized standard deviation of NAV returns.
        sharpe: Annualized Sharpe ratio of NAV returns.
        beta: Slope of NAV returns regressed on benchmark returns.
    """

    portfolio_id: uuid.UUID
    returns: dict[str, float | None]
    excess: dict[str, float | None]
    tracking_error: float | None
    max_drawdown: float | None
    volatility: float | None
    sharpe: float | None
    beta: float | None


@dataclass(frozen=True)
class HoldingMark:
    """Per-holding mark-to-market snapshot in the mandate base currency.

    Attributes:
        ticker: The held instrument's yfinance ticker.
        shares: FX-correct share count (notional_base / price_base@fill).
        value: Current base-currency market value (shares × current base price).
        weight: Share of the marked portfolio value this holding represents.
        return_1d: Trailing 1-day price return, or ``None`` with too little data.
        return_1m: Trailing ~1-month price return, or ``None``.
        return_1y: Trailing ~1-year price return, or ``None``.
        contribution: ``weight × return_1d`` (contribution to 1-day return), or
            ``None`` when the 1-day return is unavailable.
        unrealized_pnl: Current value minus FX-correct cost basis, or ``None``.
    """

    ticker: str
    shares: float
    value: float
    weight: float
    return_1d: float | None
    return_1m: float | None
    return_1y: float | None
    contribution: float | None
    unrealized_pnl: float | None


def _to_base_prices(
    prices: pd.DataFrame,
    *,
    base_currency: str,
    currency_map: dict[str, str],
    fx_rates: pd.DataFrame,
    fill_limit: int = 5,
) -> pd.DataFrame:
    """Convert a listing-currency price frame to the mandate base currency.

    Wraps ``optimizer.fx``: minor-unit listings (``GBp``/``ZAc``/``ILA``) are
    rescaled to their major unit, then multiplied by the base-per-foreign FX rate.
    Tickers whose currency has no FX column are left unconverted (logged, not
    raised); an unsupported base currency returns the prices unchanged.

    Args:
        prices: Wide price frame (``DatetimeIndex`` rows, ticker columns), in
            listing currency. ``Decimal`` values are coerced to float.
        base_currency: ISO-4217 code to express all prices in.
        currency_map: ``{ticker: raw price_unit}`` — raw sub-unit codes (``GBp``),
            never pre-normalized to the major (``GBP``), or the ÷100 scale is lost.
        fx_rates: Rate frame (``DatetimeIndex`` rows, major-currency columns), each
            cell base-per-one-foreign.
        fill_limit: Forward-fill horizon for aligning rates to the price index.

    Returns:
        A base-currency price frame of the same shape; the input unchanged when it
        is empty or the base currency is unsupported.
    """
    if prices.empty:
        return prices
    from optimizer.fx import (
        BaseCurrency,
        FxConfig,
        FxConversionMode,
        build_fx_converter,
    )

    try:
        base = BaseCurrency(base_currency.upper())
    except ValueError:
        logger.warning(
            "performance: unsupported base currency %r; prices left unconverted",
            base_currency,
        )
        return prices.astype("float64")

    config = FxConfig(
        base_currency=base,
        mode=FxConversionMode.TO_BASE,
        fill_limit=fill_limit,
        require_full_coverage=False,
    )
    converter = build_fx_converter(config, fx_rates=fx_rates, currency_map=currency_map)
    return converter.fit_transform(prices)


def _load_price_frame(
    session: Any,
    tickers: list[str],
    *,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load a wide close-price frame + raw currency map for the given tickers.

    Joins each yfinance ticker to its instrument (``instruments.yfinance_ticker``)
    and reads ``price_history`` closes into a ``DatetimeIndex`` × ticker frame. The
    currency map keys each ticker to its **raw** ``price_unit`` (e.g. ``GBp``),
    falling back to ``instruments.currency_code``.

    Args:
        session: Open SQLAlchemy session.
        tickers: yfinance tickers to load.
        start: Inclusive lower date bound; ``None`` leaves it open.
        end: Inclusive upper date bound; ``None`` leaves it open.

    Returns:
        ``(prices, currency_map)`` — the price frame (empty when nothing loads) and
        the raw ``{ticker: price_unit}`` map.
    """
    from portopt_db.repositories.market_data.yfinance_repository import (
        YFinanceRepository,
    )

    repo = YFinanceRepository(session)
    series: dict[str, pd.Series] = {}
    currency_map: dict[str, str] = {}
    for ticker in tickers:
        instrument = repo.get_instrument_by_yfinance_ticker(ticker)
        if instrument is None:
            continue
        rows = repo.get_price_history(
            instrument.id, start_date=start, end_date=end, ascending=True
        )
        closes = {r.date: float(r.close) for r in rows if r.close is not None}
        if not closes:
            continue
        series[ticker] = pd.Series(closes)
        unit = next((r.price_unit for r in rows if r.price_unit), None)
        unit = unit or instrument.currency_code
        if unit:
            currency_map[ticker] = unit
    if not series:
        return pd.DataFrame(), currency_map
    prices = pd.DataFrame(series).sort_index()
    prices.index = pd.DatetimeIndex(prices.index)
    return prices, currency_map


def _load_fx_rates_frame(
    session: Any,
    base: str,
    quotes: list[str] | None = None,
    *,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Load FX rates into a dates × quote-currency frame for ``optimizer.fx``.

    Pivots ``fx_rates`` rows for one base currency into a frame whose columns are
    major quote currencies and whose cells are base-per-one-foreign.

    Args:
        session: Open SQLAlchemy session.
        base: Base currency the rates are expressed in.
        quotes: Restrict to these quote currencies; ``None`` loads all.
        start: Inclusive lower date bound; ``None`` leaves it open.
        end: Inclusive upper date bound; ``None`` leaves it open.

    Returns:
        A ``DatetimeIndex`` × currency-column rate frame; empty when no rows match.
    """
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    rows = FxRatesRepository(session).get_rates(base.upper(), quotes, start, end)
    if not rows:
        return pd.DataFrame()
    data: dict[str, dict[dt.date, float]] = {}
    for row in rows:
        data.setdefault(row.quote_currency, {})[row.date] = float(row.rate)
    frame = pd.DataFrame(data).sort_index()
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def _major_currency(code: str) -> str:
    """Return the ISO major unit for a (possibly sub-unit) price code."""
    from optimizer.fx import normalize_currency_code

    major, _ = normalize_currency_code(code)
    return major


def _price_asof(prices: pd.DataFrame, ticker: str, ts: Any) -> float | None:
    """Last valid base price for ``ticker`` at or before ``ts`` (no look-ahead)."""
    value: Any = prices[ticker].asof(ts)
    if pd.isna(value):
        return None
    return float(value)


def _ticket_shares(ticket: Any, base_prices: pd.DataFrame) -> dict[str, float]:
    """Re-derive FX-correct shares for a paper ticket from its lines.

    ``shares = notional_base / price_base@fill`` — the stored ``lines.shares`` is
    currency-mixed (base notional ÷ listing price) and is deliberately ignored.
    """
    fill_ts = pd.Timestamp(ticket.fill_date)
    shares: dict[str, float] = {}
    for line in ticket.lines:
        ticker = line.get("ticker")
        notional = line.get("notional")
        if ticker is None or notional is None or ticker not in base_prices.columns:
            continue
        price = _price_asof(base_prices, ticker, fill_ts)
        if not price:
            continue
        shares[ticker] = float(notional) / price
    return shares


def _active_shares(
    ticket_shares: list[tuple[dt.date, dict[str, float]]], day: dt.date
) -> dict[str, float]:
    """The holdings in force on ``day``: the latest ticket filled at or before it.

    Paper tickets are full target re-allocations (``positions`` is replaced, not
    accumulated, on each approve), so the current book is the most recent fill —
    not the sum of every fill.
    """
    active: dict[str, float] = {}
    for fill_date, shares in ticket_shares:
        if fill_date <= day:
            active = shares
        else:
            break
    return active


def build_nav_series(
    portfolio_id: uuid.UUID,
    *,
    session: Any,
    base_currency: str,
    benchmark: str | None = None,
    asof: dt.date | None = None,
) -> list[NavPoint]:
    """Reconstruct the paper portfolio's NAV series, marked to market in base ccy.

    Shares are re-derived forward from ``paper_orders.lines`` (no look-ahead: a
    fill counts from its ``fill_date``) as ``notional_base / price_base@fill``,
    with ``price_base`` FX-converted from listing currency via ``optimizer.fx``.
    Each trading day at or after the first fill is marked with that day's
    base-currency close (carried forward over gaps). The benchmark level, when
    given, is pulled from ``price_history``, FX-normalized to base, and date-aligned.

    Args:
        portfolio_id: Portfolio whose paper book to reconstruct.
        session: Open SQLAlchemy session (reader; no commit).
        base_currency: Mandate base currency to express NAV in.
        benchmark: Optional benchmark ticker; its level is FX-normalized to base.
        asof: Upper date bound for the series; ``None`` runs to the latest price.

    Returns:
        NAV points ordered by date; empty when the portfolio has no fills or no
        overlapping prices.
    """
    from portopt_db.models.orders.paper_order import PaperOrder
    from sqlalchemy import select

    tickets = list(
        session.execute(
            select(PaperOrder)
            .where(PaperOrder.portfolio_id == portfolio_id)
            .order_by(PaperOrder.fill_date, PaperOrder.asof)
        )
        .scalars()
        .all()
    )
    if not tickets:
        return []

    load_tickers = sorted({line["ticker"] for t in tickets for line in t.lines})
    if benchmark and benchmark not in load_tickers:
        load_tickers.append(benchmark)

    first_fill = min(t.fill_date for t in tickets)
    raw_prices, currency_map = _load_price_frame(
        session, load_tickers, start=first_fill, end=asof
    )
    if raw_prices.empty:
        return []

    quotes = sorted({_major_currency(code) for code in currency_map.values()})
    fx_rates = _load_fx_rates_frame(
        session, base_currency, quotes, start=first_fill, end=asof
    )
    base_prices = _to_base_prices(
        raw_prices,
        base_currency=base_currency,
        currency_map=currency_map,
        fx_rates=fx_rates,
    ).ffill()

    ticket_shares = [
        (ticket.fill_date, _ticket_shares(ticket, base_prices)) for ticket in tickets
    ]
    ticket_shares = [(d, s) for d, s in ticket_shares if s]
    if not ticket_shares:
        return []
    earliest = ticket_shares[0][0]

    points: list[NavPoint] = []
    for ts in base_prices.index:
        day = ts.date()
        if day < earliest:
            continue
        shares = _active_shares(ticket_shares, day)
        nav = 0.0
        marked = False
        for ticker, qty in shares.items():
            if ticker not in base_prices.columns:
                continue
            price = base_prices.at[ts, ticker]
            if pd.notna(price):
                nav += qty * float(price)
                marked = True
        if not marked:
            continue
        bench_level: float | None = None
        if benchmark and benchmark in base_prices.columns:
            raw = base_prices.at[ts, benchmark]
            if pd.notna(raw):
                bench_level = float(raw)
        points.append(NavPoint(date=day, nav=nav, benchmark=bench_level))
    return points


__all__ = [
    "HoldingMark",
    "NavPoint",
    "PerformanceSummary",
    "build_nav_series",
]
