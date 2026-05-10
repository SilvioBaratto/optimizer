"""Pure computation for dashboard performance metrics.

All functions are stateless — no DB access. They accept pandas
DataFrames/Series and return plain dicts matching the response schema.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timezone
from typing import Any

import numpy as np
import pandas as pd

from optimizer.rebalancing import compute_drifted_weights

ANNUALIZATION_FACTOR = 252
SPARKLINE_POINTS = 30
ROLLING_VOL_WINDOW = 21
ROLLING_SHARPE_WINDOW = 63


# ---------------------------------------------------------------------------
# Individual KPI helpers
# ---------------------------------------------------------------------------


def _total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def _annualized_return(returns: pd.Series) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    total = (1 + returns).prod()
    return float(total ** (ANNUALIZATION_FACTOR / n) - 1)


def _sharpe_ratio(returns: pd.Series) -> float:
    ann_ret = _annualized_return(returns)
    ann_vol = _volatility(returns)
    if ann_vol == 0.0:
        return 0.0
    return ann_ret / ann_vol


def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return float(dd.min())


def _volatility(returns: pd.Series) -> float:
    return float(returns.std() * math.sqrt(ANNUALIZATION_FACTOR))


def _cvar_95(returns: pd.Series) -> float:
    cutoff = returns.quantile(0.05)
    tail = returns[returns <= cutoff]
    if tail.empty:
        return 0.0
    return float(tail.mean())


# ---------------------------------------------------------------------------
# Portfolio returns
# ---------------------------------------------------------------------------


def _portfolio_returns(
    prices: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Compute weighted portfolio return series from price DataFrame."""
    available = [t for t in weights if t in prices.columns]
    if not available:
        raise ValueError("No price data for any portfolio ticker")

    w = np.array([weights[t] for t in available])
    w = w / w.sum()  # re-normalize to available tickers

    rets = prices[available].pct_change().dropna()
    return (rets * w).sum(axis=1)


# ---------------------------------------------------------------------------
# Sparklines (trailing rolling values)
# ---------------------------------------------------------------------------


def _sparkline_cumulative(returns: pd.Series) -> list[float]:
    """Trailing cumulative return index (last SPARKLINE_POINTS values)."""
    cum = (1 + returns).cumprod()
    pts = cum.iloc[-SPARKLINE_POINTS:] if len(cum) >= SPARKLINE_POINTS else cum
    # Normalize to start at 100
    base = pts.iloc[0] if len(pts) > 0 else 1.0
    return (pts / base * 100).round(2).tolist()


def _sparkline_rolling_vol(returns: pd.Series) -> list[float]:
    vol = returns.rolling(ROLLING_VOL_WINDOW).std() * math.sqrt(ANNUALIZATION_FACTOR)
    pts = vol.dropna().iloc[-SPARKLINE_POINTS:]
    return pts.round(4).tolist()


def _sparkline_rolling_sharpe(returns: pd.Series) -> list[float]:
    roll_mean = returns.rolling(ROLLING_SHARPE_WINDOW).mean()
    roll_std = returns.rolling(ROLLING_SHARPE_WINDOW).std()
    sharpe = (roll_mean / roll_std.replace(0, np.nan)) * math.sqrt(ANNUALIZATION_FACTOR)
    pts = sharpe.dropna().iloc[-SPARKLINE_POINTS:]
    return pts.round(4).tolist()


def _sparkline_drawdown(returns: pd.Series) -> list[float]:
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    pts = dd.iloc[-SPARKLINE_POINTS:] if len(dd) >= SPARKLINE_POINTS else dd
    return pts.round(4).tolist()


def _sparkline_rolling_cvar(returns: pd.Series) -> list[float]:
    def _cvar_window(x: pd.Series) -> float:
        cutoff = x.quantile(0.05)
        tail = x[x <= cutoff]
        return float(tail.mean()) if len(tail) > 0 else 0.0

    cvar = returns.rolling(ROLLING_SHARPE_WINDOW).apply(_cvar_window, raw=False)
    pts = cvar.dropna().iloc[-SPARKLINE_POINTS:]
    return pts.round(4).tolist()


# ---------------------------------------------------------------------------
# Change delta (month-over-month)
# ---------------------------------------------------------------------------

# Sanity bound on per-window KPI magnitudes. Real-world ratios (Sharpe,
# Calmar, …) sit well below this; values beyond it indicate numerical
# instability — e.g. a near-flat prior window inflates Sharpe to 1e+16
# because float std() returns ~1e-23 instead of exactly 0.
_KPI_MAGNITUDE_LIMIT: float = 1e3


def _compute_change(returns: pd.Series, kpi_fn, lookback: int = 21) -> float:
    """Compute change = current_period_kpi - prior_period_kpi.

    Defensive against near-zero baselines: if either window's KPI is
    non-finite or its magnitude exceeds ``_KPI_MAGNITUDE_LIMIT`` (signalling
    numerical noise from a degenerate window), the function returns 0.0.
    Keeping this absolute-difference contract bounded is what stops the
    Dashboard KPI delta from rendering values like ``Sharpe ▼ -1263.85 %``
    (issue #432).
    """
    n = len(returns)
    if n < 2 * lookback:
        return 0.0
    current = kpi_fn(returns.iloc[-lookback:])
    prior = kpi_fn(returns.iloc[-2 * lookback : -lookback])
    if not _is_plausible(current) or not _is_plausible(prior):
        return 0.0
    return round(current - prior, 6)


def _is_plausible(value: float) -> bool:
    """Return True when ``value`` is finite and within the KPI sanity bound."""
    return math.isfinite(value) and abs(value) <= _KPI_MAGNITUDE_LIMIT


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_performance_metrics(
    weights: dict[str, float],
    prices: pd.DataFrame,
    nav: float | None = None,
) -> dict:
    """Compute all 7 KPIs with sparklines and change deltas.

    Args:
        weights: {yfinance_ticker: weight} from latest portfolio snapshot.
        prices: DataFrame[date × ticker → close] including benchmark.
        nav: Real NAV from broker account snapshot. If None, uses normalised index.

    Returns:
        Dict matching PerformanceMetricsResponse schema shape.
    """
    port_rets = _portfolio_returns(prices, weights)

    if len(port_rets) < SPARKLINE_POINTS:
        raise ValueError(
            f"Insufficient price data: {len(port_rets)} trading days "
            f"(minimum {SPARKLINE_POINTS} required)"
        )

    # Scalar KPI values
    total_ret = _total_return(port_rets)
    ann_ret = _annualized_return(port_rets)
    sharpe = _sharpe_ratio(port_rets)
    max_dd = _max_drawdown(port_rets)
    vol = _volatility(port_rets)
    cvar = _cvar_95(port_rets)

    # NAV
    cum = (1 + port_rets).cumprod()
    if nav is None:
        nav = round(float(cum.iloc[-1]) * 100, 2)  # normalised index starting at 100
    port_value = nav

    # NAV change (last month)
    nav_change = _compute_change(port_rets, _total_return)

    kpis = [
        {
            "label": "Total Return",
            "value": round(total_ret, 4),
            "format": "percent",
            "change": _compute_change(port_rets, _total_return),
            "change_label": "vs last month",
            "sparkline": _sparkline_cumulative(port_rets),
        },
        {
            "label": "Ann. Return",
            "value": round(ann_ret, 4),
            "format": "percent",
            "change": _compute_change(port_rets, _annualized_return),
            "change_label": "vs last month",
            "sparkline": _sparkline_cumulative(port_rets),
        },
        {
            "label": "Sharpe Ratio",
            "value": round(sharpe, 4),
            "format": "ratio",
            "change": _compute_change(port_rets, _sharpe_ratio),
            "change_label": "vs last month",
            "sparkline": _sparkline_rolling_sharpe(port_rets),
        },
        {
            "label": "Max Drawdown",
            "value": round(max_dd, 4),
            "format": "percent",
            "change": _compute_change(port_rets, _max_drawdown),
            "change_label": "vs last month",
            "sparkline": _sparkline_drawdown(port_rets),
        },
        {
            "label": "Portfolio Value",
            "value": port_value,
            "format": "currency",
            "change": nav_change,
            "change_label": "vs last month",
            "sparkline": _sparkline_cumulative(port_rets),
        },
        {
            "label": "Volatility",
            "value": round(vol, 4),
            "format": "percent",
            "change": _compute_change(port_rets, _volatility),
            "change_label": "vs last month",
            "sparkline": _sparkline_rolling_vol(port_rets),
        },
        {
            "label": "CVaR 95%",
            "value": round(cvar, 4),
            "format": "percent",
            "change": _compute_change(port_rets, _cvar_95),
            "change_label": "vs last month",
            "sparkline": _sparkline_rolling_cvar(port_rets),
        },
    ]

    return {
        "kpis": kpis,
        "nav": port_value,
        "nav_change_pct": nav_change,
    }


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------


def compute_equity_curve(
    weights: dict[str, float],
    prices: pd.DataFrame,
    benchmark: str,
) -> dict:
    """Compute dual-line equity curve (portfolio vs benchmark) rebased to 100.

    Args:
        weights: {yfinance_ticker: weight} from latest portfolio snapshot.
        prices: DataFrame[date × ticker → close] including benchmark ticker.
        benchmark: Benchmark ticker column name in *prices*.

    Returns:
        Dict matching EquityCurveResponse schema shape.
    """
    port_rets = _portfolio_returns(prices, weights)

    bench_prices = prices[benchmark].reindex(port_rets.index)
    bench_rets = bench_prices.pct_change().dropna()

    common_idx = port_rets.index.intersection(bench_rets.index)
    if len(common_idx) < 2:
        raise ValueError(
            "Insufficient overlapping price data for portfolio and benchmark"
        )

    port_rets = port_rets.loc[common_idx]
    bench_rets = bench_rets.loc[common_idx]

    port_cum = (1 + port_rets).cumprod()
    bench_cum = (1 + bench_rets).cumprod()

    # Rebase both series to start at 100
    port_cum = port_cum / port_cum.iloc[0] * 100
    bench_cum = bench_cum / bench_cum.iloc[0] * 100

    points = [
        {
            "date": idx.date() if hasattr(idx, "date") else idx,
            "portfolio": round(float(p), 4),
            "benchmark": round(float(b), 4),
        }
        for idx, p, b in zip(common_idx, port_cum, bench_cum, strict=False)
    ]

    port_total_return = round(float(port_cum.iloc[-1] / 100 - 1), 6)
    bench_total_return = round(float(bench_cum.iloc[-1] / 100 - 1), 6)

    return {
        "points": points,
        "portfolio_total_return": port_total_return,
        "benchmark_total_return": bench_total_return,
    }


# ---------------------------------------------------------------------------
# Rolling metrics (Sharpe / volatility / beta time series)
# ---------------------------------------------------------------------------


def _rolling_sharpe(port_rets: pd.Series, window: int) -> pd.Series:
    mean = port_rets.rolling(window).mean()
    std = port_rets.rolling(window).std()
    return (mean / std.replace(0, np.nan)) * math.sqrt(ANNUALIZATION_FACTOR)


def _rolling_volatility(port_rets: pd.Series, window: int) -> pd.Series:
    return port_rets.rolling(window).std() * math.sqrt(ANNUALIZATION_FACTOR)


def _rolling_beta(
    port_rets: pd.Series,
    bench_rets: pd.Series,
    window: int,
) -> pd.Series:
    cov = port_rets.rolling(window).cov(bench_rets)
    var = bench_rets.rolling(window).var()
    return cov / var.replace(0, np.nan)


def compute_rolling_metrics(
    weights: dict[str, float],
    prices: pd.DataFrame,
    benchmark: str,
    window: int = ROLLING_SHARPE_WINDOW,
) -> dict:
    """Compute rolling Sharpe, annualized vol, and beta series over *window* days.

    The three series are returned on a shared date index, i.e. the index
    produced by the rolling computation after dropping leading NaNs.

    Args:
        weights: {ticker: weight} from the latest portfolio snapshot.
        prices: DataFrame[date × ticker → close] including the benchmark column.
        benchmark: Benchmark ticker column name in *prices*.
        window: Rolling-window length in trading days (default: 63 ≈ 3 months).

    Returns:
        Dict matching ``RollingMetricsResponse`` schema shape.
    """
    port_rets = _portfolio_returns(prices, weights)
    bench_prices = prices[benchmark].reindex(port_rets.index)
    bench_rets = bench_prices.pct_change().dropna()

    common_idx = port_rets.index.intersection(bench_rets.index)
    if len(common_idx) < window + 1:
        raise ValueError(
            f"Insufficient overlapping price data for a {window}-day rolling window",
        )

    port_rets = port_rets.loc[common_idx]
    bench_rets = bench_rets.loc[common_idx]

    sharpe = _rolling_sharpe(port_rets, window).dropna()
    vol = _rolling_volatility(port_rets, window).dropna()
    beta = _rolling_beta(port_rets, bench_rets, window).dropna()

    shared = sharpe.index.intersection(vol.index).intersection(beta.index)

    return {
        "window": window,
        "sharpe": _series_to_points(sharpe, shared),
        "volatility": _series_to_points(vol, shared),
        "beta": _series_to_points(beta, shared),
    }


def _series_to_points(series: pd.Series, index: pd.Index) -> list[dict]:
    trimmed = series.loc[index]
    return [
        {
            "date": idx.date() if hasattr(idx, "date") else idx,
            "value": round(float(value), 6),
        }
        for idx, value in zip(trimmed.index, trimmed.values, strict=False)
    ]


# ---------------------------------------------------------------------------
# Allocation sunburst
# ---------------------------------------------------------------------------


def compute_allocation(
    weights: dict[str, float],
    sector_mapping: dict[str, str],
) -> dict:
    """Group ticker weights by sector for a sunburst chart.

    Args:
        weights: {ticker: weight} as raw fractions (e.g. 0.082).
        sector_mapping: {ticker: sector_name}.

    Returns:
        Dict matching AllocationResponse schema shape.
    """
    # Group tickers by sector
    sector_tickers: dict[str, list[tuple[str, float]]] = {}
    for ticker, weight in weights.items():
        sector = sector_mapping.get(ticker, "Unknown")
        sector_tickers.setdefault(sector, []).append((ticker, weight * 100))

    # Build nodes with children sorted by value descending
    nodes: list[dict[str, Any]] = []
    for sector, tickers in sector_tickers.items():
        tickers.sort(key=lambda t: t[1], reverse=True)
        children = [{"name": t, "value": round(v, 4)} for t, v in tickers]
        sector_value = round(sum(v for _, v in tickers), 4)
        nodes.append({"name": sector, "value": sector_value, "children": children})

    # Sort sectors by total weight descending
    nodes.sort(key=lambda n: float(n["value"]), reverse=True)

    return {
        "nodes": nodes,
        "total_positions": len(weights),
        "total_sectors": len(nodes),
    }


# ---------------------------------------------------------------------------
# Drift analysis
# ---------------------------------------------------------------------------


def _actual_weights_from_positions(
    positions: list[dict],
) -> dict[str, float] | None:
    """Compute actual weights from broker positions by market value.

    Returns None if no positions have a usable yfinance_ticker + current_price.
    """
    valid = [
        p
        for p in positions
        if p.get("yfinance_ticker") and p.get("current_price") is not None
    ]
    if not valid:
        return None

    market_values: dict[str, float] = {}
    for p in valid:
        ticker = p["yfinance_ticker"]
        mv = p["current_price"] * p["quantity"]
        market_values[ticker] = market_values.get(ticker, 0.0) + mv

    total = sum(market_values.values())
    if total == 0.0:
        return None

    return {t: mv / total for t, mv in market_values.items()}


def _actual_weights_from_prices(
    target_weights: dict[str, float],
    prices_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute drifted weights from price changes since snapshot date."""
    common = sorted(t for t in target_weights if t in prices_df.columns)
    if not common or len(prices_df) < 1:
        raise ValueError("No price data available for drift computation")

    base_prices = prices_df[common].iloc[0]
    current_prices = prices_df[common].iloc[-1]
    returns = (current_prices / base_prices - 1).values.astype(np.float64)

    w_array = np.array([target_weights[t] for t in common], dtype=np.float64)
    w_sum = w_array.sum()
    if w_sum > 0:
        w_array = w_array / w_sum

    drifted = compute_drifted_weights(w_array, returns)

    # Scale back to original total weight for the common subset
    return {t: float(drifted[i]) * w_sum for i, t in enumerate(common)}


def compute_drift(
    target_weights: dict[str, float],
    broker_positions: list[dict],
    threshold: float,
    prices_df: pd.DataFrame | None = None,
) -> dict:
    """Compute per-ticker drift between target and actual weights.

    Args:
        target_weights: {yfinance_ticker: weight} from latest snapshot.
        broker_positions: List of position dicts with keys
            yfinance_ticker, name, quantity, current_price.
        threshold: Absolute drift threshold for breach flag.
        prices_df: Price DataFrame for fallback path (only used when
            broker positions are unavailable).

    Returns:
        Dict matching DriftResponse schema shape.
    """
    # Build name lookup from broker positions
    name_lookup: dict[str, str | None] = {}
    for p in broker_positions:
        yt = p.get("yfinance_ticker")
        if yt:
            name_lookup[yt] = p.get("name")

    # Determine actual weights
    actual = _actual_weights_from_positions(broker_positions)

    if actual is None:
        if prices_df is None or prices_df.empty:
            raise ValueError("No price data available for drift computation")
        actual = _actual_weights_from_prices(target_weights, prices_df)

    # Build per-ticker drift entries
    entries: list[dict[str, Any]] = []
    for ticker, target_w in target_weights.items():
        actual_w = actual.get(ticker, 0.0)
        drift_val = actual_w - target_w
        entries.append(
            {
                "ticker": ticker,
                "name": name_lookup.get(ticker),
                "target": round(target_w, 6),
                "actual": round(actual_w, 6),
                "drift": round(drift_val, 6),
                "breached": abs(drift_val) > threshold,
            }
        )

    # Sort by abs(drift) descending
    entries.sort(key=lambda e: abs(float(e["drift"])), reverse=True)

    total_drift = round(sum(abs(float(e["drift"])) for e in entries), 6)
    breached_count = sum(1 for e in entries if bool(e["breached"]))

    return {
        "entries": entries,
        "total_drift": total_drift,
        "breached_count": breached_count,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Market snapshot
# ---------------------------------------------------------------------------


def get_market_snapshot(
    fred_data: dict[str, tuple[float, float]],
    spy_prices: list[float],
    bond_yield: tuple[float, float],
    as_of: date,
) -> dict:
    """Compute market snapshot from pre-fetched data.

    Args:
        fred_data: {series_id: (latest, previous)} for VIXCLS and DTWEXBGS.
        spy_prices: [prev_close, latest_close] for SPY.
        bond_yield: (yield_value, day_change) for US 10Y.
        as_of: Most recent data date.

    Returns:
        Dict matching MarketSnapshotResponse schema (snake_case keys).
    """
    vix_latest, vix_prev = fred_data["VIXCLS"]
    usd_latest, usd_prev = fred_data["DTWEXBGS"]

    return {
        "vix": round(vix_latest, 4),
        "vix_change": round(vix_latest - vix_prev, 4),
        "sp500_return": round((spy_prices[1] - spy_prices[0]) / spy_prices[0], 6),
        "ten_year_yield": round(bond_yield[0], 4),
        "yield_change": round(bond_yield[1], 4),
        "usd_index": round(usd_latest, 4),
        "usd_change": round(usd_latest - usd_prev, 4),
        "as_of": datetime.combine(as_of, time.min, tzinfo=timezone.utc),
    }


def _sector_return_for_period(
    tickers_in_sector: list[str],
    sector_weights: list[float],
    prices: pd.DataFrame,
    start_idx: int,
) -> float:
    """Compute sector return = weighted average of ticker returns.

    Uses within-sector normalized weights. Skips tickers absent
    from prices.columns without raising.
    """
    available = [
        (t, w)
        for t, w in zip(tickers_in_sector, sector_weights, strict=False)
        if t in prices.columns
    ]
    if not available:
        return 0.0

    tickers, weights = zip(*available, strict=False)
    total_w = sum(weights)
    if total_w == 0.0:
        return 0.0

    start_prices = prices[list(tickers)].iloc[start_idx]
    end_prices = prices[list(tickers)].iloc[-1]
    ticker_returns = (end_prices / start_prices - 1).fillna(0.0)

    return float(
        sum(
            (w / total_w) * ticker_returns[t]
            for t, w in zip(tickers, weights, strict=False)
        )
    )


def compute_asset_class_returns(
    weights: dict[str, float],
    sector_mapping: dict[str, str],
    prices: pd.DataFrame,
    today: date,
) -> dict:
    """Compute sector returns over 1D, 1W, 1M, and YTD periods.

    Args:
        weights:        {yfinance_ticker: weight} from latest snapshot.
        sector_mapping: {yfinance_ticker: sector_name}.
        prices:         DataFrame[date × ticker → close]. Must cover from
                        Jan 1 of current year to today.
        today:          Reference date for YTD boundary (usually date.today()).

    Returns:
        Dict matching AssetClassReturnsResponse schema (snake_case keys).

    Raises:
        ValueError: If prices has fewer than 2 rows, or YTD start is
                    not found in price history.
    """
    n_rows = len(prices)
    if n_rows < 2:
        raise ValueError(f"Insufficient price data: {n_rows} rows (minimum 2 required)")

    # Defensive: ensure DatetimeIndex so that Timestamp comparisons work
    # even if the caller passed a DataFrame with a plain date index.
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)

    # --- YTD start index ---
    ytd_cutoff = date(today.year, 1, 1)
    ytd_candidates = prices.index[prices.index >= pd.Timestamp(ytd_cutoff)]
    if ytd_candidates.empty:
        raise ValueError(f"No price data on or after {ytd_cutoff} for YTD computation")
    _loc = prices.index.get_loc(ytd_candidates[0])
    ytd_start_idx: int = _loc if isinstance(_loc, int) else 0

    # Validate windows — fall back gracefully for shorter windows
    min_rows = {"1D": 2, "1W": 6, "1M": 22}

    # Start indices (prices.iloc[idx] is the "start" price, iloc[-1] is "end")
    period_start_idx: dict[str, int | None] = {
        "1D": -2 if n_rows >= min_rows["1D"] else None,
        "1W": -6 if n_rows >= min_rows["1W"] else None,
        "1M": -22 if n_rows >= min_rows["1M"] else None,
        "YTD": ytd_start_idx,
    }

    # Must have at least 1D
    if period_start_idx["1D"] is None:
        raise ValueError(f"Insufficient price data: {n_rows} rows (minimum 2 required)")

    # --- Group tickers by sector, preserving portfolio weight order ---
    sector_data: dict[str, dict] = {}
    for ticker, w in weights.items():
        sector = sector_mapping.get(ticker, "Unknown")
        if sector not in sector_data:
            sector_data[sector] = {"tickers": [], "weights": [], "total_w": 0.0}
        sector_data[sector]["tickers"].append(ticker)
        sector_data[sector]["weights"].append(w)
        sector_data[sector]["total_w"] += w

    # Sort sectors by descending portfolio weight (consistent with allocation)
    sorted_sectors = sorted(
        sector_data.items(), key=lambda kv: kv[1]["total_w"], reverse=True
    )

    rows = []
    for sector, data in sorted_sectors:
        tickers = data["tickers"]
        sw = data["weights"]
        if data["total_w"] == 0.0:
            continue

        row: dict[str, object] = {"name": sector}
        for label, idx in period_start_idx.items():
            if idx is not None:
                row[label] = round(
                    _sector_return_for_period(tickers, sw, prices, idx), 6
                )
            else:
                row[label] = 0.0
        rows.append(row)

    as_of = prices.index[-1]
    as_of_date = as_of.date() if hasattr(as_of, "date") else as_of

    return {"returns": rows, "as_of": as_of_date}
