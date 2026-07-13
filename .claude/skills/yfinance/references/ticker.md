# Ticker & Tickers

The central entry point. `Ticker` wraps one symbol; `Tickers` wraps many but is just a thin container around individual `Ticker` objects — for bulk prices use `yf.download` (see `price_history.md`) instead.

## Construction

```python
import yfinance as yf

ticker = yf.Ticker("AAPL", session=None)

# Multiple tickers — individual objects accessible via .tickers dict
tickers = yf.Tickers("AAPL MSFT GOOG", session=None)
# Or: yf.Tickers(["AAPL", "MSFT", "GOOG"])
tickers.tickers["AAPL"].info
```

`session` accepts any `requests.Session`-compatible object (e.g. `curl_cffi.requests.Session`) — useful for custom headers, cookies, or reuse of a single impersonated session across many calls.

## Stock Properties

| Property | Returns | Description |
|---|---|---|
| `info` | dict | Complete stock info (slow, cached) |
| `fast_info` | `FastInfo` | Key metrics (fast, fewer fields) |
| `news` | list[dict] | Recent news articles |
| `dividends` | Series | Historical dividends |
| `splits` | Series | Historical stock splits |
| `actions` | DataFrame | Dividends + splits combined |
| `capital_gains` | Series | Capital-gains distributions (funds) |
| `shares_full` | DataFrame | Historical shares outstanding |
| `get_shares_full(start, end)` | DataFrame | Same, range-bounded |
| `isin` | str | ISIN identifier |
| `options` | tuple | Available option expiry dates |
| `option_chain(date)` | `OptionChain` | Calls and puts for one expiry |
| `valuation_measures` | DataFrame | **v1.3.0+** — 9 valuation metrics × historical periods |

### fast_info fields

`currency`, `dayHigh`, `dayLow`, `exchange`, `fiftyDayAverage`, `lastPrice`, `lastVolume`, `marketCap`, `open`, `previousClose`, `quoteType`, `regularMarketPreviousClose`, `shares`, `tenDayAverageVolume`, `threeMonthAverageVolume`, `timezone`, `twoHundredDayAverage`, `yearChange`, `yearHigh`, `yearLow`

Use `fast_info` when you need a handful of live-ish metrics; it's much cheaper than `info` because it hits a smaller Yahoo endpoint. Fall back to `info` only when you need the long tail of descriptive fields (company summary, officers, fullTimeEmployees, etc.).

### valuation_measures (v1.3.0+)

Valuation metrics pulled from Yahoo's Statistics page. DataFrame rows × time-period columns:

| Metric row |
|---|
| Market Cap |
| Enterprise Value |
| Trailing P/E |
| Forward P/E |
| PEG Ratio (5yr expected) |
| Price/Sales |
| Price/Book |
| Enterprise Value/Revenue |
| Enterprise Value/EBITDA |

Columns are the current period plus historical quarters and year-ends.

```python
t = yf.Ticker("AAPL")
vm = t.valuation_measures
vm.loc["Trailing P/E"]          # time series of trailing P/E
vm.iloc[:, 0]                   # all 9 metrics for the most recent period
```

## Options

```python
ticker = yf.Ticker("AAPL")
ticker.options                  # tuple of expiry date strings
chain = ticker.option_chain("2026-06-19")
chain.calls                     # DataFrame
chain.puts                      # DataFrame
```

## News

```python
for item in ticker.news:
    print(item["title"], item["link"])
```

Each entry typically has `title`, `link`, `publisher`, `providerPublishTime`, `type`, `thumbnail`, `relatedTickers`.
