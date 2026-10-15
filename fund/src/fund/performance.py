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
    since: dt.date | None = None,
) -> list[NavPoint]:
    """Reconstruct the paper portfolio's NAV series, marked to market in base ccy.

    Shares are re-derived forward from ``paper_orders.lines`` (no look-ahead: a
    fill counts from its ``fill_date``) as ``notional_base / price_base@fill``,
    with ``price_base`` FX-converted from listing currency via ``optimizer.fx``.
    Each trading day at or after the first fill is marked with that day's
    base-currency close (carried forward over gaps). The benchmark level, when
    given, is pulled from ``price_history``, FX-normalized to base, and date-aligned.

    NAV is **gross** of paper execution costs: shares are reconstructed from the
    fill-bar *close*, so NAV at the fill equals the deployed base notional. The
    slippage and commission the paper fill modelled (``tools.orders``, default
    ~5 bps and ~1 bps) are intentionally excluded — this tracks deployed capital,
    not a cost-adjusted book.

    Args:
        portfolio_id: Portfolio whose paper book to reconstruct.
        session: Open SQLAlchemy session (reader; no commit).
        base_currency: Mandate base currency to express NAV in.
        benchmark: Optional benchmark ticker; its level is FX-normalized to base.
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

    ticket_shares = [
        (ticket.fill_date, _ticket_shares(ticket, base_prices)) for ticket in tickets
    ]
    ticket_shares = [(d, s) for d, s in ticket_shares if s]
    if not ticket_shares:
        return []
    earliest = ticket_shares[0][0]
    mark_from = earliest if since is None else max(earliest, since)

    points: list[NavPoint] = []
    for ts in base_prices.index:
        day = ts.date()
        if day < mark_from:
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
    """
    from skfolio import BasePortfolio

    try:
        portfolio = BasePortfolio(
            returns=np.asarray(returns, dtype=float),
            observations=list(dates),
            annualization_factor=annualization_factor,
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
    from portopt_db.models.orders.paper_order import PaperOrder
    from sqlalchemy import select

    stmt = select(PaperOrder).where(PaperOrder.portfolio_id == portfolio_id)
    if asof is not None:
        stmt = stmt.where(PaperOrder.fill_date <= asof)
    stmt = stmt.order_by(PaperOrder.fill_date.desc(), PaperOrder.asof.desc())
    ticket = session.execute(stmt).scalars().first()
    if ticket is None:
        return []

    line_by_ticker: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line in ticket.lines:
        ticker = line.get("ticker")
        if ticker is None:
            continue
        if ticker not in line_by_ticker:
            order.append(ticker)
        line_by_ticker[ticker] = line

    raw_prices, currency_map = _load_price_frame(session, order, end=asof)
    if raw_prices.empty:
        return [_degraded_mark(ticker) for ticker in order]

    quotes = sorted({_major_currency(code) for code in currency_map.values()})
    fx_rates = _load_fx_rates_frame(session, base_currency, quotes, end=asof)
    _require_fx_coverage(base_currency, currency_map, fx_rates)
    base_raw = _to_base_prices(
        raw_prices,
        base_currency=base_currency,
        currency_map=currency_map,
        fx_rates=fx_rates,
    )
    base_prices = base_raw.ffill()
    shares = _ticket_shares(ticket, base_prices)

    computed: dict[str, tuple[float, float, pd.Series]] = {}
    for ticker in order:
        qty = shares.get(ticker)
        if qty is None or ticker not in base_prices.columns:
            continue
        marked = base_prices[ticker].dropna()
        if marked.empty:
            continue
        # Value marks on the ffilled (asof) close; returns run on the holding's own
        # real closes so a carry-forward over a non-trading day cannot fake a move.
        returns_series = base_raw[ticker].dropna()
        computed[ticker] = (qty, qty * float(marked.iloc[-1]), returns_series)

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
        pnl = value - float(notional) if notional is not None else None
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


__all__ = [
    "FxCoverageError",
    "HoldingMark",
    "NavPoint",
    "PerformanceSummary",
    "build_nav_series",
    "holding_marks",
    "portfolio_metrics",
]
