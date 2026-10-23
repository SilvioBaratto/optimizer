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
import itertools
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HORIZONS = ("1D", "1M", "YTD", "1Y", "max")

# Calendar days of bars read before a book's earliest fill so the fill-day price, and
# the 5-row FX forward-fill behind it, still resolve across weekends and holidays.
_MARK_LOOKBACK_DAYS = 30


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

    ``shares`` / ``value`` / ``weight`` are ``None`` for a holding with no usable
    price (degraded, never fabricated).

    Attributes:
        ticker: The held instrument's yfinance ticker.
        shares: FX-correct share count (notional_base / price_base@fill), or ``None``.
        value: Current base-currency market value (shares × current base price), or
            ``None``.
        weight: Share of the marked portfolio value this holding represents, or
            ``None``.
        return_1d: Trailing 1-day price return, or ``None`` with too little data.
        return_1m: Trailing ~1-month price return, or ``None``.
        return_1y: Trailing ~1-year price return, or ``None``.
        contribution: ``weight × return_1d`` (contribution to 1-day return), or
            ``None`` when either input is unavailable.
        unrealized_pnl: Current value minus FX-correct cost basis, or ``None``.
    """

    ticker: str
    shares: float | None
    value: float | None
    weight: float | None
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


def convert_prices_to_base(
    session: Any,
    prices: pd.DataFrame,
    currency_map: dict[str, str],
    *,
    base_currency: str,
) -> pd.DataFrame:
    """FX-convert a listing-currency price frame to ``base_currency`` from the DB.

    Loads the ``fx_rates`` the frame's date span needs and applies the same
    conversion the NAV tracker uses, so an allocator reading prices through this
    helper optimises over the base-currency series NAV is later marked on.

    Args:
        session: Open SQLAlchemy session the FX rates are read from.
        prices: Wide price frame (``DatetimeIndex`` rows, ticker columns), in
            listing currency.
        currency_map: ``{ticker: raw price_unit}`` — raw sub-unit codes (``GBp``),
            never pre-normalized to the major (``GBP``).
        base_currency: ISO-4217 code to express every price in.

    Returns:
        The base-currency frame; ``prices`` unchanged when it is empty or no ticker
        carries a currency. Tickers whose currency has a rate column are converted;
        a currency missing among several present is left unconverted (logged).

    Raises:
        FxCoverageError: Foreign tickers exist but none of their currencies has a
            rate under ``base_currency`` — converting would silently mix currencies.
    """
    if prices.empty or not currency_map:
        return prices
    quotes = sorted({_major_currency(code) for code in currency_map.values()})
    days = pd.DatetimeIndex(prices.index)
    fx_rates = _load_fx_rates_frame(
        session,
        base_currency,
        quotes,
        start=min(days).date(),
        end=max(days).date(),
    )
    _require_fx_coverage(base_currency, currency_map, fx_rates)
    return _to_base_prices(
        prices,
        base_currency=base_currency,
        currency_map=currency_map,
        fx_rates=fx_rates,
    )


def _finite(value: float) -> float | None:
    """The value as a float, or ``None`` when NaN/inf — never a fabricated metric."""
    return float(value) if np.isfinite(value) else None


def _major_currency(code: str) -> str:
    """Return the ISO major unit for a (possibly sub-unit) price code."""
    from optimizer.fx import normalize_currency_code

    major, _ = normalize_currency_code(code)
    return major


class FxCoverageError(RuntimeError):
    """Raised when a mandate has foreign holdings but no FX coverage for its base.

    Distinguishes a *misconfiguration* (the FX feed never populated this base —
    e.g. a non-EUR mandate while the ingestion feed only writes EUR-base rows)
    from the per-ticker ``T6`` graceful-degradation case (one currency missing
    among several present). The former would silently mark NAV in mixed/listing
    currency, so it fails loud instead; the NAV-mark step skips that portfolio.
    """


def _require_fx_coverage(
    base_currency: str, currency_map: dict[str, str], fx_rates: pd.DataFrame
) -> None:
    """Raise :class:`FxCoverageError` when foreign holdings have *zero* FX coverage.

    Fires only when at least one held currency differs from the base *and none* of
    those foreign currencies has a rate column — the "wrong base / feed never ran"
    case. Partial coverage (some foreign currencies present, some missing) is left
    to per-ticker graceful degradation in :func:`_to_base_prices`.

    Args:
        base_currency: Mandate base currency the NAV is expressed in.
        currency_map: ``{ticker: raw price_unit}`` for the loaded holdings.
        fx_rates: The rate frame loaded for ``base_currency`` (quote-currency
            columns); empty when the feed has no rows for this base.

    Raises:
        FxCoverageError: Foreign holdings exist but no foreign currency is covered.
    """
    base_major = _major_currency(base_currency)
    foreign = {_major_currency(code) for code in currency_map.values()} - {base_major}
    if not foreign:
        return
    available = set(fx_rates.columns) if not fx_rates.empty else set()
    if not (foreign & available):
        raise FxCoverageError(
            f"no FX rates for base {base_major!r} covering {sorted(foreign)}; the "
            "ingestion FX feed populates EUR base only, so a non-EUR mandate cannot "
            "be marked. Run the FX feed for this base or set the mandate to EUR."
        )


def _expressed_in_base(
    ticker: str,
    base_currency: str,
    currency_map: dict[str, str],
    fx_rates: pd.DataFrame,
) -> bool:
    """Whether :func:`_to_base_prices` leaves ``ticker`` priced in ``base_currency``.

    The converter skips a foreign ticker whose currency has no rate column rather
    than raising, so its prices stay in listing currency; this mirrors that rule so
    callers can tell a converted series from a raw one. A ticker with no known
    currency is treated as base, as the converter does.
    """
    code = currency_map.get(ticker)
    if code is None:
        return True
    major = _major_currency(code)
    if major == _major_currency(base_currency):
        return True
    available = {str(col).upper() for col in fx_rates.columns}
    return major in available


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
    Each line is priced at **its own** ``fill_date`` (fills stagger across exchange
    calendars; the ticket-level ``fill_date`` is only the earliest of them).

    Prices at the fill-bar *close*, not the slippage-adjusted ``effective_price``,
    so value@fill equals the deployed base notional — NAV is gross of the modelled
    slippage/commission (see :func:`build_nav_series`).
    """
    shares: dict[str, float] = {}
    for line in ticket.lines:
        ticker = line.get("ticker")
        notional = line.get("notional")
        if ticker is None or notional is None or ticker not in base_prices.columns:
            continue
        fill_ts = pd.Timestamp(line.get("fill_date") or ticket.fill_date)
        price = _price_asof(base_prices, ticker, fill_ts)
        if not price:
            continue
        shares[ticker] = float(notional) / price
    return shares


def _as_date(value: Any) -> dt.date | None:
    """Coerce a stored line ``fill_date`` (ISO string or date) to a ``date``.

    Returns ``None`` for a missing or unparseable value so the caller can fall back to
    the ticket-level fill date.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _ticket_book(
    ticket: Any, base_prices: pd.DataFrame
) -> dict[str, tuple[float, dt.date, float]]:
    """Per-line ``(FX-correct shares, own fill date, base notional)`` for a ticket.

    Shares reuse :func:`_ticket_shares` — each line sized at the base price on *its*
    own ``fill_date``. Pairing the fill date and base notional lets the NAV loop carry
    a line that has not yet filled at book value instead of marking it early at a price
    it was never bought at (the staggered-fill look-ahead the ticket-level key hid).
    """
    shares = _ticket_shares(ticket, base_prices)
    book: dict[str, tuple[float, dt.date, float]] = {}
    for line in ticket.lines:
        ticker = line.get("ticker")
        notional = line.get("notional")
        if ticker not in shares or notional is None:
            continue
        fill_date = _as_date(line.get("fill_date")) or ticket.fill_date
        book[ticker] = (shares[ticker], fill_date, float(notional))
    return book


def _active_book(
    ticket_books: list[tuple[dt.date, dict[str, tuple[float, dt.date, float]]]],
    day: dt.date,
) -> dict[str, tuple[float, dt.date, float]]:
    """The book in force on ``day``: the latest ticket filled at or before it.

    Mirrors the full-re-allocation rule (paper tickets replace ``positions`` on each
    approve rather than accumulate), so the active book is the most recent ticket's,
    not the union of every fill. Each line keeps its own fill date so the NAV loop can
    defer a line that has not yet filled on ``day``.
    """
    active: dict[str, tuple[float, dt.date, float]] = {}
    for fill_date, book in ticket_books:
        if fill_date <= day:
            active = book
        else:
            break
    return active


def _book_value(
    book: dict[str, tuple[float, dt.date, float]],
    ts: pd.Timestamp,
    base_prices: pd.DataFrame,
) -> float | None:
    """Base-currency value of ``book`` on bar ``ts``; ``None`` when nothing marks.

    A line that has not yet filled on the bar is carried at its book notional rather
    than marked early at a price it was never bought at (look-ahead) or dropped, which
    would read the undeployed capital as a drawdown and recover it as a fake gain. A
    filled line with no price on the bar contributes nothing.
    """
    day = ts.date()
    value = 0.0
    marked = False
    for ticker, (qty, line_fill, notional) in book.items():
        if line_fill > day:
            value += notional
            marked = True
            continue
        if ticker not in base_prices.columns:
            continue
        price = base_prices.at[ts, ticker]
        if pd.notna(price):
            value += qty * float(price)
            marked = True
    return value if marked else None


def _chain_scales(
    ticket_books: list[tuple[dt.date, dict[str, tuple[float, dt.date, float]]]],
    base_prices: pd.DataFrame,
) -> list[float]:
    """The factor that rescales each ticket's book so NAV is continuous at its fill.

    Every ticket is sized off the same flat notional, not the portfolio's marked
    value, so a new ticket's book is worth ``sum(weights) x notional`` on its first
    bar regardless of what the previous book had earned. Each book after the first
    is multiplied by the factor that makes it worth exactly what the previous
    (already rescaled) book is worth on that bar — a rebalance then conserves value
    and the series keeps its accumulated P&L (time-weighted-return convention).

    When either side cannot be valued on the switch bar the previous factor is
    carried forward instead of resetting to the flat notional. The first ticket keeps
    its own notional (factor ``1.0``).

    This is the single source of the per-ticket scale: :func:`build_nav_series` marks
    the rescaled books and :func:`holding_marks` applies the same factor to share
    counts and cost basis, so holdings and NAV cannot disagree after a rebalance.
    """
    scales: list[float] = []
    previous_book: dict[str, tuple[float, dt.date, float]] | None = None
    scale = 1.0
    for fill_day, book in ticket_books:
        if previous_book is not None:
            position = int(base_prices.index.searchsorted(pd.Timestamp(fill_day)))
            if position < len(base_prices.index):
                ts = base_prices.index[position]
                previous = _book_value(previous_book, ts, base_prices)
                incoming = _book_value(book, ts, base_prices)
                if previous is not None and incoming:
                    scale = previous / incoming
        scales.append(scale)
        previous_book = _scaled_book(book, scale)
    return scales


def _scaled_book(
    book: dict[str, tuple[float, dt.date, float]], scale: float
) -> dict[str, tuple[float, dt.date, float]]:
    """``book`` with every line's shares and base notional multiplied by ``scale``."""
    return {
        ticker: (qty * scale, line_fill, notional * scale)
        for ticker, (qty, line_fill, notional) in book.items()
    }


def _chain_link_books(
    ticket_books: list[tuple[dt.date, dict[str, tuple[float, dt.date, float]]]],
    base_prices: pd.DataFrame,
) -> list[tuple[dt.date, dict[str, tuple[float, dt.date, float]]]]:
    """Rescale each ticket's book by its :func:`_chain_scales` factor."""
    scales = _chain_scales(ticket_books, base_prices)
    return [
        (fill_day, _scaled_book(book, scale))
        for (fill_day, book), scale in zip(ticket_books, scales, strict=True)
    ]


def build_nav_series(
    portfolio_id: uuid.UUID,
    *,
    session: Any,
    base_currency: str,
    benchmark: str | None = None,
    asof: dt.date | None = None,
    since: dt.date | None = None,
) -> list[NavPoint]:
    """Reconstruct the paper portfolio's NAV series, marked to market in base ccy.

    Shares are re-derived forward from ``paper_orders.lines`` (no look-ahead: a
    fill counts from its ``fill_date``) as ``notional_base / price_base@fill``,
    with ``price_base`` FX-converted from listing currency via ``optimizer.fx``.
    Each trading day at or after the first fill is marked with that day's
    base-currency close (carried forward over gaps). The benchmark level, when
    given, is pulled from ``price_history``, FX-normalized to base, and date-aligned.

    Rebalances conserve NAV: each ticket after the first is rescaled so the series
    is continuous on its first active bar (see :func:`_chain_link_books`), instead of
    snapping back to the flat ticket notional and discarding accumulated P&L.

    NAV is **gross** of paper execution costs: shares are reconstructed from the
    fill-bar *close*, so NAV at the fill equals the deployed base notional. The
    slippage and commission the paper fill modelled (``tools.orders``, default
    ~5 bps and ~1 bps) are intentionally excluded — this tracks deployed capital,
    not a cost-adjusted book.

    Args:
        portfolio_id: Portfolio whose paper book to reconstruct.
        session: Open SQLAlchemy session (reader; no commit).
        base_currency: Mandate base currency to express NAV in.
        benchmark: Optional benchmark ticker; its level is FX-normalized to base,
            or left ``None`` when its currency has no FX rate to convert it.
        asof: Upper date bound for the series; ``None`` runs to the latest price.
        since: Lower date bound on the *emitted* points — shares are still derived
            from the full ticket history and prices carried forward from the first
            fill, but only bars on/after ``since`` are returned. ``None`` emits the
            full series from the first fill (backfill). Lets the daily mark re-emit
            only recent bars instead of rewriting the whole history every run.

    Returns:
        NAV points ordered by date; empty when the portfolio has no fills, no
        overlapping prices, or no bars on/after ``since``.

    Raises:
        FxCoverageError: The mandate holds foreign-currency instruments but no FX
            rate for their currencies exists under ``base_currency`` (e.g. a
            non-EUR mandate against the EUR-only feed) — marking would silently mix
            currencies, so it fails loud instead.
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

    holding_tickers = {line["ticker"] for t in tickets for line in t.lines}
    load_tickers = sorted(holding_tickers)
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
    # Guard on the HOLDINGS' currencies only: a foreign benchmark that lacks a rate
    # degrades gracefully (it is a reference line, not part of NAV) and must not
    # block marking of an otherwise all-base-currency book.
    holding_currency_map = {
        ticker: code
        for ticker, code in currency_map.items()
        if ticker in holding_tickers
    }
    _require_fx_coverage(base_currency, holding_currency_map, fx_rates)
    base_prices = _to_base_prices(
        raw_prices,
        base_currency=base_currency,
        currency_map=currency_map,
        fx_rates=fx_rates,
    ).ffill()

    # A benchmark whose currency could not be converted is dropped, not emitted raw:
    # a local-currency index level beside a base-currency NAV would make excess,
    # tracking error and beta compare two currencies with no FX return between them.
    bench_in_base = (
        benchmark is not None
        and benchmark in base_prices.columns
        and _expressed_in_base(benchmark, base_currency, currency_map, fx_rates)
    )

    ticket_books = [
        (ticket.fill_date, _ticket_book(ticket, base_prices)) for ticket in tickets
    ]
    ticket_books = [(d, b) for d, b in ticket_books if b]
    if not ticket_books:
        return []
    ticket_books = _chain_link_books(ticket_books, base_prices)
    earliest = ticket_books[0][0]
    mark_from = earliest if since is None else max(earliest, since)

    points: list[NavPoint] = []
    for ts in base_prices.index:
        day = ts.date()
        if day < mark_from:
            continue
        nav = _book_value(_active_book(ticket_books, day), ts, base_prices)
        if nav is None:
            continue
        bench_level: float | None = None
        if benchmark and bench_in_base:
            raw = base_prices.at[ts, benchmark]
            if pd.notna(raw):
                bench_level = float(raw)
        points.append(NavPoint(date=day, nav=nav, benchmark=bench_level))
    return points


def _linear_returns(values: list[float]) -> list[float]:
    """Per-period simple returns, stopping at the first ill-defined (zero) denominator.

    A zero denominator makes only *that* pair undefined; the valid returns already
    computed are kept (the series is truncated there), so a portfolio whose value
    touches zero still yields the metrics its history up to that point supports.
    """
    returns: list[float] = []
    for prev, cur in itertools.pairwise(values):
        if prev == 0:
            break
        returns.append(cur / prev - 1.0)
    return returns


def nav_return_series(nav_points: list[NavPoint], *, source: str = "nav") -> pd.Series:
    """Per-period simple returns of a NAV series as a date-indexed float ``Series``.

    The public return-series accessor the macro-attribution engine regresses on.
    Reuses :func:`_linear_returns`, so a zero denominator truncates the series at
    that point rather than fabricating a value. Each return is indexed by the later
    date of its pair (``nav_points[1:]``), matching :func:`portfolio_metrics`.

    Args:
        nav_points: The NAV series from :func:`build_nav_series` (ascending dates).
        source: ``"nav"`` uses :attr:`NavPoint.nav`; ``"benchmark"`` uses
            :attr:`NavPoint.benchmark`, skipping points whose benchmark is ``None``
            so a late-starting or partial benchmark still yields its own return
            window.

    Returns:
        Simple returns on a ``DatetimeIndex``; an empty float series when fewer than
        two usable points exist.

    Raises:
        ValueError: ``source`` is neither ``"nav"`` nor ``"benchmark"``.
    """
    if source == "nav":
        pairs = [(p.date, float(p.nav)) for p in nav_points]
    elif source == "benchmark":
        pairs = [
            (p.date, float(p.benchmark)) for p in nav_points if p.benchmark is not None
        ]
    else:
        raise ValueError(f"source must be 'nav' or 'benchmark', got {source!r}")

    if len(pairs) < 2:
        return pd.Series([], index=pd.DatetimeIndex([]), dtype=float)

    dates = [d for d, _ in pairs]
    returns = _linear_returns([v for _, v in pairs])
    index = pd.DatetimeIndex(dates[1 : 1 + len(returns)])
    return pd.Series(returns, index=index, dtype=float)


def _nav_asof(points: list[NavPoint], target: dt.date) -> float | None:
    """The NAV on the latest point at or before ``target`` (points sorted ascending)."""
    base: float | None = None
    for point in points:
        if point.date <= target:
            base = point.nav
        else:
            break
    return base


def _window_return(
    points: list[NavPoint], last: NavPoint, delta: dt.timedelta
) -> float | None:
    """Trailing return from ``last.date - delta`` to the last point, or ``None``."""
    base = _nav_asof(points, last.date - delta)
    if not base:
        return None
    return last.nav / base - 1.0


def _ytd_return(points: list[NavPoint], last: NavPoint) -> float | None:
    """Return since prior year-end, falling back to inception for a new series."""
    base = _nav_asof(points, dt.date(last.date.year - 1, 12, 31))
    if base is None:
        base = points[0].nav
    if not base:
        return None
    return last.nav / base - 1.0


def _horizon_returns(points: list[NavPoint]) -> dict[str, float | None]:
    """Trailing fractional returns for each horizon; ``None`` when history is short."""
    out: dict[str, float | None] = dict.fromkeys(_HORIZONS)
    if len(points) < 2:
        return out
    last = points[-1]
    if points[0].nav:
        out["max"] = last.nav / points[0].nav - 1.0
    if points[-2].nav:
        out["1D"] = last.nav / points[-2].nav - 1.0
    out["1M"] = _window_return(points, last, dt.timedelta(days=30))
    out["1Y"] = _window_return(points, last, dt.timedelta(days=365))
    out["YTD"] = _ytd_return(points, last)
    return out


def _risk_measures(
    returns: list[float], dates: list[dt.date], annualization_factor: float
) -> tuple[float | None, float | None, float | None]:
    """Whole-series (volatility, Sharpe, max drawdown) via ``skfolio.BasePortfolio``.

    Volatility and Sharpe need at least two returns (sample dispersion); max
    drawdown needs one. Any NaN or failure degrades the affected field to ``None``.
    The returns are NAV-to-NAV simple returns, so the drawdown is taken on the
    compounded wealth path; skfolio's default sums returns, which overstates the
    decline of a compounding NAV.
    """
    from skfolio import BasePortfolio

    try:
        portfolio = BasePortfolio(
            returns=np.asarray(returns, dtype=float),
            observations=list(dates),
            annualization_factor=annualization_factor,
            compounded=True,
        )
        max_drawdown = _finite(float(portfolio.max_drawdown))
        if len(returns) >= 2:
            volatility = _finite(float(portfolio.annualized_standard_deviation))
            sharpe = _finite(float(portfolio.annualized_sharpe_ratio))
        else:
            volatility = sharpe = None
    except Exception:
        logger.warning("performance: risk measures unavailable", exc_info=True)
        return None, None, None
    return volatility, sharpe, max_drawdown


def portfolio_metrics(
    portfolio_id: uuid.UUID,
    nav_points: list[NavPoint],
    *,
    annualization_factor: float = 252.0,
) -> PerformanceSummary:
    """Compute multi-horizon returns + risk metrics over a NAV series.

    Whole-series volatility/Sharpe/max-drawdown come from ``skfolio.BasePortfolio``
    on the NAV's linear returns; trailing-horizon returns and the
    benchmark-relative figures (excess per horizon, annualized tracking error,
    beta) are computed directly. Every field degrades to ``None`` on insufficient
    history. Benchmark-relative fields run over the sub-series where a benchmark
    level is present (a benchmark priced later than the first fill leaves leading
    ``None`` points), so a late-starting benchmark no longer blanks them all; a
    ``NaN`` benchmark level (as opposed to a missing one) still degrades tracking
    error and beta to ``None``.

    Args:
        portfolio_id: Portfolio the metrics describe.
        nav_points: The NAV series from :func:`build_nav_series` (ascending dates).
        annualization_factor: Periods per year for annualizing vol/Sharpe/TE.

    Returns:
        A :class:`PerformanceSummary`; all-``None`` fields when the series is empty.
    """
    navs = [p.nav for p in nav_points]
    port_returns = _linear_returns(navs)
    # Align observations to the (possibly truncated) returns so BasePortfolio gets
    # equal-length inputs when a zero denominator cut the series short.
    return_dates = [p.date for p in nav_points[1 : 1 + len(port_returns)]]

    volatility = sharpe = max_drawdown = None
    if port_returns:
        volatility, sharpe, max_drawdown = _risk_measures(
            port_returns, return_dates, annualization_factor
        )

    returns = _horizon_returns(nav_points)
    excess: dict[str, float | None] = dict.fromkeys(_HORIZONS)
    tracking_error: float | None = None
    beta: float | None = None

    # Restrict to the points that carry a benchmark level. A NaN benchmark is kept
    # (NaN is not None) so it poisons TE/beta to None via _finite rather than
    # silently shrinking the window; a genuinely missing (None) leading benchmark
    # is trimmed so a late-starting benchmark still yields metrics over its window.
    bench_window = [p for p in nav_points if p.benchmark is not None]
    if len(bench_window) >= 2:
        bench_navs = [
            float(p.benchmark) for p in bench_window if p.benchmark is not None
        ]
        window_navs = [p.nav for p in bench_window]
        bench_points = [
            NavPoint(date=p.date, nav=b, benchmark=None)
            for p, b in zip(bench_window, bench_navs, strict=True)
        ]
        port_window_points = [
            NavPoint(date=p.date, nav=p.nav, benchmark=None) for p in bench_window
        ]
        bench_returns = _linear_returns(bench_navs)
        window_returns = _linear_returns(window_navs)
        # Compare port vs benchmark over the *same* window so each excess horizon
        # is like-for-like (the full-series `returns` can span more than the window).
        port_window_horizon = _horizon_returns(port_window_points)
        bench_horizon = _horizon_returns(bench_points)
        for horizon in _HORIZONS:
            port_r, bench_r = port_window_horizon[horizon], bench_horizon[horizon]
            if port_r is not None and bench_r is not None:
                excess[horizon] = port_r - bench_r
        if len(window_returns) >= 2 and len(bench_returns) == len(window_returns):
            port_arr = np.asarray(window_returns, dtype=float)
            bench_arr = np.asarray(bench_returns, dtype=float)
            tracking_error = _finite(
                float(np.std(port_arr - bench_arr, ddof=1))
                * np.sqrt(annualization_factor)
            )
            bench_var = float(np.var(bench_arr, ddof=1))
            if bench_var > 0:
                beta = _finite(
                    float(np.cov(port_arr, bench_arr, ddof=1)[0, 1]) / bench_var
                )

    return PerformanceSummary(
        portfolio_id=portfolio_id,
        returns=returns,
        excess=excess,
        tracking_error=tracking_error,
        max_drawdown=max_drawdown,
        volatility=volatility,
        sharpe=sharpe,
        beta=beta,
    )


def _degraded_mark(ticker: str) -> HoldingMark:
    """A holding that could not be priced: every numeric field ``None``."""
    return HoldingMark(
        ticker=ticker,
        shares=None,
        value=None,
        weight=None,
        return_1d=None,
        return_1m=None,
        return_1y=None,
        contribution=None,
        unrealized_pnl=None,
    )


def _series_returns(
    series: pd.Series,
) -> tuple[float | None, float | None, float | None]:
    """Trailing (1D, 1M, 1Y) returns of a base-price series (ascending, non-NaN)."""
    current = float(series.iloc[-1])
    last_ts = series.index[-1]

    r1d: float | None = None
    if len(series) >= 2:
        prev = float(series.iloc[-2])
        if prev:
            r1d = current / prev - 1.0

    def _window(days: int) -> float | None:
        base: Any = series.asof(last_ts - pd.Timedelta(days=days))
        if pd.isna(base) or float(base) == 0:
            return None
        return current / float(base) - 1.0

    return r1d, _window(30), _window(365)


def _active_ticket(session: Any, portfolio_id: uuid.UUID, asof: dt.date | None) -> Any:
    """The latest paper ticket filled at or before ``asof`` (``None`` when absent)."""
    from portopt_db.models.orders.paper_order import PaperOrder
    from sqlalchemy import select

    stmt = select(PaperOrder).where(PaperOrder.portfolio_id == portfolio_id)
    if asof is not None:
        stmt = stmt.where(PaperOrder.fill_date <= asof)
    stmt = stmt.order_by(PaperOrder.fill_date.desc(), PaperOrder.asof.desc())
    return session.execute(stmt).scalars().first()


def _tickets_through(
    session: Any, portfolio_id: uuid.UUID, asof: dt.date | None
) -> list[Any]:
    """Every paper ticket filled at or before ``asof``, oldest first (the NAV chain)."""
    from portopt_db.models.orders.paper_order import PaperOrder
    from sqlalchemy import select

    stmt = select(PaperOrder).where(PaperOrder.portfolio_id == portfolio_id)
    if asof is not None:
        stmt = stmt.where(PaperOrder.fill_date <= asof)
    stmt = stmt.order_by(PaperOrder.fill_date, PaperOrder.asof)
    return list(session.execute(stmt).scalars().all())


def _ticket_lines(ticket: Any) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """A ticket's tickers in line order, and the (last) line for each of them."""
    line_by_ticker: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line in ticket.lines:
        ticker = line.get("ticker")
        if ticker is None:
            continue
        if ticker not in line_by_ticker:
            order.append(ticker)
        line_by_ticker[ticker] = line
    return order, line_by_ticker


def _book_base_prices(
    session: Any,
    order: list[str],
    *,
    base_currency: str,
    start: dt.date | None,
    end: dt.date | None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Base-currency closes for a book: ``(raw, forward-filled)``, or ``None``.

    ``None`` when no holding has any price in the window. ``raw`` keeps each holding's
    own trading days (NaN elsewhere); the forward-filled copy is what values are
    marked on.

    Raises:
        FxCoverageError: See :func:`_require_fx_coverage`.
    """
    raw_prices, currency_map = _load_price_frame(session, order, start=start, end=end)
    if raw_prices.empty:
        return None
    quotes = sorted({_major_currency(code) for code in currency_map.values()})
    fx_rates = _load_fx_rates_frame(
        session, base_currency, quotes, start=start, end=end
    )
    _require_fx_coverage(base_currency, currency_map, fx_rates)
    base_raw = _to_base_prices(
        raw_prices,
        base_currency=base_currency,
        currency_map=currency_map,
        fx_rates=fx_rates,
    )
    return base_raw, base_raw.ffill()


def _marked_value(
    ticker: str, shares: dict[str, float], base_prices: pd.DataFrame
) -> tuple[float, float] | None:
    """``(shares, value)`` at the holding's latest base close; ``None`` if unpriced."""
    qty = shares.get(ticker)
    if qty is None or ticker not in base_prices.columns:
        return None
    marked = base_prices[ticker].dropna()
    if marked.empty:
        return None
    return qty, qty * float(marked.iloc[-1])


def holding_marks(
    portfolio_id: uuid.UUID,
    *,
    session: Any,
    base_currency: str,
    asof: dt.date | None = None,
) -> list[HoldingMark]:
    """Mark each current holding to market in the mandate base currency.

    Takes the latest paper ticket at or before ``asof`` as the current book,
    re-derives FX-correct shares (``notional_base / price_base@fill``, ignoring the
    stored currency-mixed ``lines.shares``), and marks each holding with its latest
    base-currency close. Per holding it reports value, portfolio weight, trailing
    1D/1M/1Y price returns, 1-day contribution (``weight × return_1d``), and
    unrealized P&L (value minus the deployed base cost basis). A holding with no
    usable price degrades to ``None`` fields rather than raising.

    Value marks on the carried-forward (``asof``) close, but returns are computed
    on each holding's *own* real closes, so a carry-forward over a day the holding
    did not trade cannot manufacture a 0 % move. Like :func:`build_nav_series`,
    value and P&L are gross of the modelled slippage/commission.

    The trailing returns need a year of closes, so this reads each holding's whole
    price history; a caller that only needs the book's weights (the drift poll) should
    use :func:`marked_weights`.

    Args:
        portfolio_id: Portfolio whose current book to mark.
        session: Open SQLAlchemy session (reader; no commit).
        base_currency: Mandate base currency to express values in.
        asof: Mark date upper bound; ``None`` uses the latest available prices.

    Returns:
        One :class:`HoldingMark` per line of the active ticket, in ticket order;
        empty when the portfolio has no paper orders.

    Raises:
        FxCoverageError: Foreign holdings exist but no FX rate covers their currency
            under ``base_currency`` (see :func:`build_nav_series`).
    """
    tickets = _tickets_through(session, portfolio_id, asof)
    if not tickets:
        return []
    ticket = tickets[-1]

    order, line_by_ticker = _ticket_lines(ticket)
    # Earlier tickets' instruments are loaded too: the current ticket's scale depends
    # on the value of every book it replaced.
    chain_tickers = sorted(
        {line["ticker"] for t in tickets for line in t.lines if line.get("ticker")}
    )
    book = _book_base_prices(
        session, chain_tickers, base_currency=base_currency, start=None, end=asof
    )
    if book is None:
        return [_degraded_mark(ticker) for ticker in order]
    base_raw, base_prices = book
    current_book = _ticket_book(ticket, base_prices)
    scale = 1.0
    if current_book:
        ticket_books = [(t.fill_date, _ticket_book(t, base_prices)) for t in tickets]
        scale = _chain_scales([(d, b) for d, b in ticket_books if b], base_prices)[-1]
    # Holdings sit on the chain-linked NAV scale, so they sum to the latest NAV and
    # the cost basis is the value carried into the rebalance, not the flat notional.
    shares = {
        ticker: qty * scale
        for ticker, qty in _ticket_shares(ticket, base_prices).items()
    }

    computed: dict[str, tuple[float, float, pd.Series]] = {}
    for ticker in order:
        marked = _marked_value(ticker, shares, base_prices)
        if marked is None:
            continue
        qty, value = marked
        # Value marks on the ffilled (asof) close; returns run on the holding's own
        # real closes so a carry-forward over a non-trading day cannot fake a move.
        computed[ticker] = (qty, value, base_raw[ticker].dropna())

    total = sum(value for _, value, _ in computed.values())

    marks: list[HoldingMark] = []
    for ticker in order:
        if ticker not in computed:
            marks.append(_degraded_mark(ticker))
            continue
        qty, value, series = computed[ticker]
        weight = value / total if total else None
        return_1d, return_1m, return_1y = _series_returns(series)
        contribution = (
            weight * return_1d if weight is not None and return_1d is not None else None
        )
        notional = line_by_ticker[ticker].get("notional")
        pnl = value - float(notional) * scale if notional is not None else None
        marks.append(
            HoldingMark(
                ticker=ticker,
                shares=qty,
                value=value,
                weight=weight,
                return_1d=return_1d,
                return_1m=return_1m,
                return_1y=return_1y,
                contribution=contribution,
                unrealized_pnl=pnl,
            )
        )
    return marks


def marked_weights(
    portfolio_id: uuid.UUID,
    *,
    session: Any,
    base_currency: str,
    asof: dt.date | None = None,
) -> dict[str, float] | None:
    """Market-value weights of the current paper book, from a bounded price read.

    The same weights :func:`holding_marks` reports (shares × latest base close over
    the book's total), but loading only the bars from ``_MARK_LOOKBACK_DAYS`` before
    the book's earliest fill instead of each holding's whole listing history. The
    drift poll runs this for every portfolio every couple of seconds on the TUI event
    loop, so its cost must scale with the book's age, not the instruments'.

    ``None`` — never a partial map — when the portfolio has no paper ticket, a holding
    has no price in the window (including one whose last bar before its fill is more
    than ``_MARK_LOOKBACK_DAYS`` older), or the total value is zero: weights over only
    the priced holdings would misstate the book.

    Args:
        portfolio_id: Portfolio whose current book to weight.
        session: Open SQLAlchemy session (reader; no commit).
        base_currency: Mandate base currency to express values in.
        asof: Mark date upper bound; ``None`` uses the latest available prices.

    Returns:
        ``{ticker: weight}`` as shares of NAV including the cash residual the ticket
        left undeployed (``notional`` minus the line notionals), so a book approved with
        ``Bounds.budget < 1`` marks on the same basis as its target instead of being
        renormalised to 1 and reading as permanently drifted; the weights therefore sum
        to 1 for a fully invested book. ``None`` when the book cannot be marked.

    Raises:
        FxCoverageError: Foreign holdings exist but no FX rate covers their currency
            under ``base_currency`` (see :func:`build_nav_series`).
    """
    ticket = _active_ticket(session, portfolio_id, asof)
    if ticket is None:
        return None

    order, line_by_ticker = _ticket_lines(ticket)
    fills = [
        _as_date(line.get("fill_date")) or ticket.fill_date
        for line in ticket.lines
        if line.get("ticker") is not None
    ]
    window_start = min(fills, default=ticket.fill_date) - dt.timedelta(
        days=_MARK_LOOKBACK_DAYS
    )
    book = _book_base_prices(
        session, order, base_currency=base_currency, start=window_start, end=asof
    )
    if book is None:
        return None
    _, base_prices = book
    shares = _ticket_shares(ticket, base_prices)

    values: dict[str, float] = {}
    for ticker in order:
        marked = _marked_value(ticker, shares, base_prices)
        if marked is None:
            return None
        values[ticker] = marked[1]
    deployed = sum(
        float(line["notional"])
        for line in line_by_ticker.values()
        if line.get("notional") is not None
    )
    cash = max(0.0, float(ticket.notional or 0.0) - deployed)
    total = sum(values.values()) + cash
    if total <= 0:
        return None
    return {ticker: value / total for ticker, value in values.items()}


__all__ = [
    "FxCoverageError",
    "HoldingMark",
    "NavPoint",
    "PerformanceSummary",
    "build_nav_series",
    "convert_prices_to_base",
    "holding_marks",
    "marked_weights",
    "nav_return_series",
    "portfolio_metrics",
]
