# Ticker & Tickers

The central entry point. `Ticker` wraps one symbol; `Tickers` wraps many but is a thin
container around individual `Ticker` objects — for bulk prices use `yf.download` (see
`price_history.md`) instead.

## Construction

```python
import yfinance as yf

ticker = yf.Ticker("AAPL", session=None)          # constructor is (ticker, session=None)

# Multiple tickers — individual objects via the .tickers dict (UPPERCASE keys)
tickers = yf.Tickers("AAPL MSFT GOOG", session=None)
# Or: yf.Tickers(["AAPL", "MSFT", "GOOG"])
tickers.tickers["AAPL"].info
```

- `session` accepts any `requests.Session`-compatible object (`curl_cffi` session if installed).
- **No `lang` / `region` constructor args.** Locale is set GLOBALLY via `yf.config` locale (see `config.md`), not per-`Ticker`.
- **No `proxy` parameter** — configure proxy globally (see `config.md`).
- `Tickers.tickers` keys are uppercased regardless of input case.

## Property ↔ get_* method pairs

Almost every accessor comes as a **property** and a parallel **`get_*()` method**. The property
takes no arguments; the `get_*()` method is where optional parameters live (e.g. `as_dict=`,
`freq=`, `period=`, `limit=`).

```python
ticker.dividends              # property
ticker.get_dividends(period="max")   # method with params
```

## Stock properties

| Property | Method form | Returns | Description |
|---|---|---|---|
| `info` | `get_info()` | dict | Complete stock info (slow, scraped, cached) |
| `fast_info` | `get_fast_info()` | `FastInfo` | Key metrics (fast, fewer fields) |
| `news` | `get_news(count=…, tab=…)` | list[dict] | Recent news; `tab` ∈ `"news"`, `"all"`, `"press releases"` |
| `dividends` | `get_dividends(period)` | Series | Historical dividends |
| `splits` | `get_splits(period)` | Series | Historical stock splits |
| `actions` | `get_actions(period)` | DataFrame | Dividends + splits + capital gains |
| `capital_gains` | `get_capital_gains(period)` | Series | Capital-gains distributions (funds) |
| `get_shares_full(start, end)` | — | DataFrame | Historical shares outstanding |
| `isin` | `get_isin()` | str | ISIN identifier |
| `options` | — | tuple | Available option expiry dates |
| `option_chain(date)` | — | `Options` namedtuple | `.calls`, `.puts`, `.underlying` for one expiry |
| `valuation` | `get_valuation_measures(freq='quarterly', periods=5)` | DataFrame | 9 valuation metrics × historical periods |
| `funds_data` | `get_funds_data()` | `FundsData` | ETF/fund holdings object (see `funds.md`) |
| `calendar` | — | dict | Next earnings / dividend events |
| `earnings_dates` | `get_earnings_dates(limit, offset)` | DataFrame | Past/future earnings dates (`offset` added post-1.3.0) |
| `sec_filings` | `get_sec_filings()` | list[dict] | SEC filings |

Financial-statement, analyst, and ownership properties live in `financials.md` /
`analysis.md`.

### fast_info fields

`currency`, `dayHigh`, `dayLow`, `exchange`, `fiftyDayAverage`, `lastPrice`, `lastVolume`,
`marketCap`, `open`, `previousClose`, `quoteType`, `regularMarketPreviousClose`, `shares`,
`tenDayAverageVolume`, `threeMonthAverageVolume`, `timezone`, `twoHundredDayAverage`,
`yearChange`, `yearHigh`, `yearLow`

Use `fast_info` when you need a handful of live-ish metrics; it's much cheaper than `info`. Fall
back to `info` only for the long tail of descriptive fields (company summary, officers,
fullTimeEmployees, etc.).

### valuation

9 metrics pulled from Yahoo's timeseries API (v1.5.1 — was an HTML Statistics scrape before, so
row/field shapes may differ from older code). The `valuation` property takes no args; use
`get_valuation_measures(freq='quarterly', periods=5)` for the parameterised form. DataFrame rows ×
time-period columns:

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

```python
t = yf.Ticker("AAPL")
vm = t.valuation                # or t.get_valuation_measures(freq="quarterly", periods=5)
vm.loc["Trailing P/E"]          # time series of trailing P/E
vm.iloc[:, 0]                   # all 9 metrics for the most recent period
```

## Options

```python
ticker.options                  # tuple of expiry date strings
chain = ticker.option_chain(ticker.options[0])   # namedtuple 'Options'
chain.calls                     # DataFrame
chain.puts                      # DataFrame
chain.underlying                # dict of underlying quote fields
```

Read `options` before calling `option_chain` — the canonical pattern is
`option_chain(options[0])`. The return is a namedtuple `Options(calls, puts, underlying)`
(attribute access), not an `OptionChain` type.

## Real-time streaming

```python
ticker.live()                   # stream this symbol via WebSocket (see websocket.md)
tickers.live()                  # batch stream across the Tickers container
```

## News

```python
for item in ticker.news:
    content = item["content"]                      # each item = {'id', 'content'}
    print(content["title"], content["clickThroughUrl"]["url"])
# get_news(count=…, tab="press releases") to page or switch feeds
```

Each item has only top-level keys `id` and `content`. The real fields live under
`item["content"]`: `title`, `summary`, `pubDate`, `provider`, `canonicalUrl`,
`clickThroughUrl` (a dict with `.url`), `thumbnail`, `contentType`. The old flat
`title`/`link`/`publisher`/`providerPublishTime` schema is gone.
