# Configuration, Caching, Price Repair

## Network configuration

```python
import yfinance as yf

# Proxy
yf.config.network.proxy = "http://proxy:8080"

# Retries — exponential backoff (1s, 2s, 4s, ...)
yf.config.network.retries = 5
```

## Debug configuration

```python
# Hide exceptions (return empty results instead of raising)
yf.config.debug.hide_exceptions = False   # default — set True to suppress

# Verbose logging
yf.config.debug.logging = True

# Full debug mode (verbose + expose internals)
yf.enable_debug_mode()
```

> **v1.0 deprecation (Dec 2025):** direct attribute assignment on `yf.config.*` still works but emits `DeprecationWarning` in favor of a new config method. Check `yf.config.__dict__` or the [advanced config docs](https://ranaroussi.github.io/yfinance/advanced/config.html) for the current recommended API before starting a new project.

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
| Windows | `%LOCALAPPDATA%\py-yfinance\Cache` |

## Price repair

When `repair=True` (on `history()` or `download()`), yfinance detects and fixes:

| # | Issue | Notes |
|---|---|---|
| 1 | Missing dividend adjustment | Prices not adjusted after dividend |
| 2 | Missing stock-split adjustment | v1.1.0 reduced false positives from benign price jumps |
| 3 | Missing data | Gaps filled from adjacent intervals |
| 4 | Corrupt data | Outlier detection and replacement |
| 5 | 100x currency errors | Wrong unit (e.g. pence vs pounds) |
| 6 | Dividend repair | Incorrect dividend amounts |
| 7 | Capital-gains double-counting | Fund distributions counted as both gain and dividend (**v1.1.0**) |

```python
df = ticker.history(period="2y", repair=True)
if "Repaired?" in df.columns:
    repaired = df[df["Repaired?"] == True]
    print(f"Repaired {len(repaired)} rows")
```

> **pandas 3+ gotcha (v1.2.0):** `history()` output is memory-consolidated as one block. Mutating in place may raise `ValueError: output array is read-only`. Call `.copy()` before assigning. yfinance's own repair path was fixed internally in 1.2.0 — the warning is only for *your* code.

## Sessions

Any parameter that accepts `session=...` takes a `requests.Session`-compatible object. Use `curl_cffi.requests.Session` (the default transport under the hood since the curl_cffi migration) to pool connections and share cookies across many calls, or pass a custom session with auth headers / SOCKS proxy / retries tuned elsewhere.

```python
from curl_cffi import requests

session = requests.Session(impersonate="chrome")
ticker = yf.Ticker("AAPL", session=session)
```
