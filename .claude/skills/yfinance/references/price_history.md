# Price History — `yf.download`, `Ticker.history`, `PriceHistory`

Two entry points for OHLCV data. Use `yf.download` for bulk retrieval of many tickers; use
`Ticker.history` when you already have a `Ticker` object and want richer parameters (pre/post,
back-adjust, error behavior). `Ticker.history` delegates to the low-level `PriceHistory` scraper.

## yf.download

```python
data = yf.download(
    tickers,                  # str or list — "AAPL" or ["AAPL", "MSFT"]
    start=None,               # inclusive; str "YYYY-MM-DD" or datetime
    end=None,                 # EXCLUSIVE; end="2023-01-01" -> last point 2022-12-31
    actions=False,            # dividends/splits columns — NOTE: default False here
    threads=True,             # multi-threaded download (bool or int)
    auto_adjust=True,         # adjust OHLC for splits/dividends
    back_adjust=False,        # back-adjust to mimic true historical prices
    prepost=False,            # include pre/post market
    repair=False,             # repair known data issues (see config.md)
    keepna=False,             # keep NaN rows
    progress=True,            # print a download progress bar
    interval="1d",
    group_by="column",        # "column" (default) or "ticker"
    ignore_tz=None,           # see timezone note below
    rounding=False,           # round to 2 decimals
    timeout=10,               # seconds (fractional ok); default 10 (pass None to disable)
    session=None,
    multi_level_index=True,   # multi-level columns even for a single ticker
    period="1mo",             # used only when start AND end are None
)
```

> **`actions` default differs from `history()`.** `download()` defaults `actions=False`;
> `Ticker.history()` defaults `actions=True`. If you need Dividends / Stock Splits columns from
> `download`, pass `actions=True` explicitly.

> **No `proxy` parameter.** Proxy is global-only (see `config.md`). `download()` *does* expose a
> `progress=True` bar toggle and `back_adjust=False` (both shown above); only `proxy` is missing.

**Column order:** `Open, High, Low, Close, Volume`. `auto_adjust=True` adjusts OHLC in place and
removes the separate `Adj Close` column.

> **Reentrant since v1.4.0** — shared module globals were removed, so `download()` is safe from
> async workers and worker pools without extra locking. v1.4.1 also preserves the
> Date/Datetime index name.

### Timezone (`ignore_tz`) — behavior changed in 1.4.0

`ignore_tz` has no fixed default: it is `False` for intraday intervals and `True` for
day-and-above intervals.

- `ignore_tz=True` → tz-naive index.
- `ignore_tz=False` → **the most-common exchange timezone across the requested tickers**
  (before v1.4.0 this always returned UTC — localized intraday downloads were wrongly UTC).

## Ticker.history

```python
history = ticker.history(
    period="1mo",             # "1mo" if start & end None; can combine with start/end
    interval="1d",
    start=None,               # inclusive
    end=None,                 # exclusive
    prepost=False,
    actions=True,             # NOTE: default True here (vs download's False)
    auto_adjust=True,
    back_adjust=False,        # back-adjust to mimic true historical prices
    repair=False,
    keepna=False,
    rounding=False,           # False = precision suggested by Yahoo
    timeout=10,
    raise_errors=False,       # DEPRECATED (emits DeprecationWarning); use yf.config.debug.hide_exceptions=False
)
# Columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
```

`get_history_metadata(repair=<sentinel>)` returns exchange/timezone/instrument metadata for the
most recent fetch. Its `repair` default is a **sentinel** — when not passed it inherits the
repair setting from the previous `history()` call (v1.4.0 added the flag).

## PriceHistory (low-level)

`Ticker.history()` delegates to `yfinance.scrapers.history.PriceHistory`. You rarely construct
it directly, but it exposes the same `history()` plus corporate-action accessors that take
`period` and `repair`:

```python
ph.get_actions(period="max")                 # DataFrame
ph.get_dividends(period="max", repair=False) # Series
ph.get_splits(period="max", repair=False)    # Series
ph.get_capital_gains(period="max", repair=False)  # Series
ph.get_history_metadata()                    # dict
```

## Valid periods

`1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

## Valid intervals

`1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`

## Period ↔ interval constraints

Yahoo enforces history limits that silently truncate data:

- **1m**: max 7 days of history
- **2m–90m**: max 60 days
- **1h**: max 730 days
- **1d and above**: full history
- **Intraday cannot extend beyond the last 60 days.**

> **`30m` is resampled.** yfinance fetches 30-minute bars from Yahoo as `15m` and resamples
> internally to work around a Yahoo API bug — the values are derived, not native 30m bars.

Pass `start`/`end` for explicit ranges; pass `period` only for rolling lookback.

## Multi-Level Columns

```python
# group_by="column" (default): Level 0 = OHLCV, Level 1 = Ticker
data = yf.download(["AAPL", "MSFT"], group_by="column")
data["Close"]["AAPL"]

# group_by="ticker": Level 0 = Ticker, Level 1 = OHLCV
data = yf.download(["AAPL", "MSFT"], group_by="ticker")
data["AAPL"]["Close"]

# Disable multi-level (returns flat columns even for lists)
data = yf.download(["AAPL", "MSFT"], multi_level_index=False)

# CSV round-trip — re-read with header=[0, 1]
data.to_csv("prices.csv")
df = pd.read_csv("prices.csv", header=[0, 1], index_col=0, parse_dates=True)
```

`multi_level_index=True` by default, so even a single-ticker `download` returns multi-level
columns unless you set it `False`.

## pandas 3+ read-only gotcha (v1.2.0)

`history()` output is memory-consolidated (one contiguous block). With pandas 3+, in-place
mutation may raise `ValueError: output array is read-only`. Call `.copy()` before mutating:

```python
df = ticker.history(period="2y", repair=True).copy()
df["Close"] *= 1.0   # safe
```

(v1.6.0 fixed an internal read-only `Adj Close` crash in the dividend-adjust repair path.)
