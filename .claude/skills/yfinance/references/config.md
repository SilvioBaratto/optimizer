# Configuration, Auth, Caching, Price Repair

## Proxy — now global only

`proxy` was **removed as a per-call / constructor argument** across `Ticker`, `Search`,
`Lookup`, `Market`, `Sector`, `Industry`, `screen()`, and `download()` through the 1.4.x–1.6.0
line. Configure it globally instead (v1.5.1 also normalizes configured proxy strings).

```python
import yfinance as yf

# Global config (preferred). Assign on the nested namespace — flat attribute
# assignment like `yf.config.proxy = ...` silently no-ops (it does NOT reach network.*):
yf.config.network.proxy = "http://proxy:8080"
yf.config.network.retries = 5              # exponential backoff (1s, 2s, 4s, ...)

# Deprecated-but-working legacy path: the function yf.set_config(...) forwards to network.*
yf.set_config(proxy="http://proxy:8080", retries=5)   # emits DeprecationWarning
```

> Do **not** pass `proxy=` to `yf.Ticker(...)`, `yf.download(...)`, `yf.Search(...)`, etc. — it
> is no longer accepted there. Cross-check the advanced config docs
> (https://ranaroussi.github.io/yfinance/advanced/config.html) for the current recommended
> config API before starting a new project.

## Debug configuration

```python
yf.config.debug.hide_exceptions = False    # default is True (exceptions suppressed -> empty results); set False to surface them
yf.config.debug.logging = True             # verbose logging
yf.enable_debug_mode()                     # DEPRECATED (only sets DEBUG-level logging); use yf.config.debug.logging = True
```

## curl_cffi is optional (v1.4.0)

`curl_cffi` is **still a declared dependency** (`Requires-Dist: curl_cffi>=0.15`, no extra
marker). What changed in v1.4.0 is a **runtime fallback**: yfinance uses the `requests`
package instead if `curl_cffi` cannot be imported.

- The declared minimum is **`>= 0.15`** (matches the METADATA pin and the code's fallback warning).
- If you don't, `requests` is used transparently.
- `frozendict` is also no longer a hard dependency (dropped v1.4.0; internal fallback).

```python
# Optional: use an impersonated curl_cffi session if you have curl_cffi installed
try:
    from curl_cffi import requests as cffi_requests
    session = cffi_requests.Session(impersonate="chrome")
except ImportError:
    import requests
    session = requests.Session()

ticker = yf.Ticker("AAPL", session=session)
```

## Auth — Yahoo account login (v1.4.0)

`yf.Auth` logs in to a Yahoo account, unlocking subscription-tier data. Login state and
subscription tier are resolved via Yahoo's subscriptions API (v1.5.1), and login cookies are
preserved across cookie-strategy switches.

```python
auth = yf.Auth(...)   # see https://ranaroussi.github.io/yfinance/reference/yfinance.auth.html
```

Consult the upstream Auth reference for the exact constructor — do not hardcode credentials;
wire them through your secret store.

## Caching

yfinance caches timezone data by default.

```python
yf.set_tz_cache_location("/path/to/cache")
```

**Default cache paths:**

| OS | Path |
|---|---|
| macOS | `~/Library/Caches/py-yfinance` |
| Linux | `~/.cache/py-yfinance` |
| Windows | `%LOCALAPPDATA%\py-yfinance` |

## Price repair

When `repair=True` (on `history()`, `download()`, or the action getters), yfinance detects and
fixes:

| # | Issue | Notes |
|---|---|---|
| 1 | Missing dividend adjustment | Prices not adjusted after dividend |
| 2 | Missing stock-split adjustment | v1.1.0 reduced false positives from benign price jumps |
| 3 | Missing data | Gaps filled from adjacent intervals |
| 4 | Corrupt data | Outlier detection and replacement |
| 5 | 100x currency errors | Wrong unit (e.g. pence vs pounds) |
| 6 | Dividend amount errors | Incorrect dividend values |
| 7 | Capital-gains double-counting | Fund distributions counted as gain + dividend (v1.1.0) |

```python
df = ticker.history(period="2y", repair=True)
if "Repaired?" in df.columns:
    repaired = df[df["Repaired?"] == True]
    print(f"Repaired {len(repaired)} rows")
```

> **Sub-unit currencies now stay put (v1.6.0).** `repair=True` **no longer permanently
> converts GBp / ZAc / ILA (pence / cents / agorot) prices to the main currency** — repaired
> sub-unit prices remain in their quoted sub-unit. Convert explicitly if you need
> main-currency values. v1.5.1 also fixed the unit-switch sometimes applying twice, false-
> positive bad dividends from premarket games, and NaN-volume errors; v1.6.0 fixed a
> read-only `Adj Close` crash in the dividend-adjust path.

> **pandas 3+ gotcha (v1.2.0):** `history()` output is memory-consolidated as one block.
> Mutating in place may raise `ValueError: output array is read-only`. Call `.copy()` before
> assigning. yfinance's own repair path was fixed internally — the warning is for *your* code.

## Sessions

Any parameter that accepts `session=...` takes a `requests.Session`-compatible object. Use a
`curl_cffi.requests.Session` (if `curl_cffi` is installed) to pool connections and share cookies
across many calls, or a plain `requests.Session` with auth headers / SOCKS proxy tuned
elsewhere. See the curl_cffi-optional snippet above for a portable construction.
