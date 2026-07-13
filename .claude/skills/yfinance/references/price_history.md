# Price History — `yf.download` and `Ticker.history`

Two entry points for OHLCV data. Use `yf.download` for bulk retrieval of many tickers; use `Ticker.history` when you already have a `Ticker` object and want richer parameters (pre/post, error behavior).

## yf.download

```python
data = yf.download(
    tickers,                  # str or list — "AAPL" or ["AAPL", "MSFT"]
    period="1mo",             # see "Valid periods" below
    interval="1d",            # see "Valid intervals" below
    start=None,               # str or datetime — "2020-01-01"
    end=None,                 # str or datetime — "2024-01-01"
    group_by="column",        # "column" (default) or "ticker"
    auto_adjust=True,         # Adjust OHLC for splits/dividends
    repair=False,             # Repair known data issues (see config.md)
    actions=True,             # Include dividends and stock splits
    threads=True,             # Multi-threaded download
    proxy=None,
    progress=True,
    timeout=10,
    multi_level_index=True,   # Multi-level columns for multi-ticker
)
```

**Column order:** `Open, High, Low, Close, Volume`. `auto_adjust=True` adjusts OHLC in place and removes the separate `Adj Close` column.

> **v1.2.1 / v1.2.2 — thread-safe.** Previously a race in shared state could corrupt DataFrames under `threads=True`. Now safe from async workers and worker pools without extra locking.

## Ticker.history

```python
history = ticker.history(
    period="1mo",
    interval="1d",
    start=None,
    end=None,
    prepost=False,            # Include pre/post market data
    auto_adjust=True,
    repair=False,
    keepna=False,             # Keep NaN rows
    rounding=False,           # Round prices to 2 decimals
    raise_errors=False,       # Raise exceptions vs return empty
)
# Columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
```

## Valid periods

`1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

## Valid intervals

`1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`

## Period ↔ interval constraints

Yahoo enforces history limits that silently truncate data. Most common:

- **1m**: max 7 days of history
- **2m–90m**: max 60 days
- **1h**: max 730 days
- **1d and above**: full history

Pass `start`/`end` for explicit ranges; pass `period` only for rolling lookback.

## Multi-Level Columns

When downloading multiple tickers, yfinance returns multi-level columns by default.

```python
# group_by="column" (default): Level 0 = OHLCV, Level 1 = Ticker
data = yf.download(["AAPL", "MSFT"], group_by="column")
data["Close"]["AAPL"]

# group_by="ticker": Level 0 = Ticker, Level 1 = OHLCV
data = yf.download(["AAPL", "MSFT"], group_by="ticker")
data["AAPL"]["Close"]

# Disable multi-level (recent versions)
data = yf.download(["AAPL", "MSFT"], multi_level_index=False)

# CSV round-trip — re-read with header=[0, 1]
data.to_csv("prices.csv")
df = pd.read_csv("prices.csv", header=[0, 1], index_col=0, parse_dates=True)
```

## pandas 3+ read-only gotcha (v1.2.0)

As of v1.2.0, `history()` output is memory-consolidated — a single contiguous block instead of many. With pandas 3+, in-place mutations on the returned DataFrame may raise `ValueError: output array is read-only`. yfinance's internal repair path was fixed; if you hit this in your own code, call `.copy()` before mutating:

```python
df = ticker.history(period="2y", repair=True).copy()
df["Close"] *= 1.0   # safe
```
