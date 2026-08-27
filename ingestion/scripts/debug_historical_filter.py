"""TEMP repro: why HistoricalDataFilter passed 0/3255 in the bulk universe build.

Standalone the filter passes (AAPL 5y = 1254 rows > 750). In the bulk build it
rejected 100%. Hypothesis: the build fetches history for thousands of tickers
CONCURRENTLY through the shared YFinanceClient, and Yahoo's crumb/session
collapses under that load → fetch_history returns empty → every instrument is
rejected (the filter treats "couldn't fetch" the same as "too little history").

This script runs the SAME filter serial vs concurrent over real large-caps that
all genuinely have 5y history, and prints a row-by-row PASS/FAIL so the two
modes can be compared.

Debug it: set breakpoints in
  HistoricalDataFilter._check_historical_data  (historical_data.py)
  YFinanceClient.fetch_history / the facade    (yfinance/_facade.py)
and launch the "DEBUG: historical filter repro" config, then watch `hist`,
`len(hist)`, and any exception per ticker in the debugger console.

Run:
    python -m scripts.debug_historical_filter --workers 8 --mode both
(cwd = ingestion/, with CURL_CA_BUNDLE set — the launch config does this via .env)
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services.universe.trading212.filters.historical_data import (
    HistoricalDataFilter,
)

from app.services.market_data.yfinance import YFinanceClient
from app.services.universe.trading212.config import UniverseBuilderConfig

# Real names that unquestionably have 5y of daily history → every one SHOULD
# pass HistoricalDataFilter. Any FAIL here is a fetch/crumb problem, not "young".
TICKERS: list[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "V",
    "WMT",
    "KO",
    "PEP",
    "XOM",
    "CVX",
    "PG",
    "JNJ",
    "UNH",
    "HD",
    "BAC",
    "DIS",
    "SAP.DE",
    "SIE.DE",
    "ALV.DE",
    "BAS.DE",
    "BMW.DE",
    "DTE.DE",
    "BAYN.DE",
    "VOW3.DE",
    "MC.PA",
    "OR.PA",
    "AIR.PA",
    "BNP.PA",
    "SAN.PA",
    "TTE.PA",
    "HSBA.L",
    "BP.L",
    "SHEL.L",
    "AZN.L",
    "GSK.L",
    "ULVR.L",
]


def _run_one(
    f: HistoricalDataFilter, ticker: str
) -> tuple[str, bool, str, float, str | None]:
    """Run the filter for one ticker; return (ticker, passed, reason, secs, error)."""
    started = time.time()
    try:
        ok, reason = f.filter({}, ticker)  # data arg unused by this filter
        return ticker, ok, reason, time.time() - started, None
    except Exception as exc:  # surface, don't swallow — this is a diagnostic
        return ticker, False, "", time.time() - started, repr(exc)


def _run_one_pipeline(
    mapper: object, f: HistoricalDataFilter, ticker: str
) -> tuple[str, bool, str, float, str | None]:
    """Faithful per-instrument path: fetch_basic_data (fetch_info) THEN the
    HistoricalDataFilter (fetch_history) — exactly what _process_single_instrument
    does. This is what the real build runs; history-only did not reproduce it.
    """
    started = time.time()
    try:
        basic = mapper.fetch_basic_data(ticker)  # type: ignore[attr-defined]
        if not basic:
            return (
                ticker,
                False,
                "fetch_basic_data returned None (fetch_info failed)",
                time.time() - started,
                None,
            )
        ok, reason = f.filter(basic, ticker)
        return ticker, ok, reason, time.time() - started, None
    except Exception as exc:
        return ticker, False, "", time.time() - started, repr(exc)


def _run(mode: str, workers: int, tickers: list[str], pipeline: bool = False) -> int:
    cfg = UniverseBuilderConfig()
    f = HistoricalDataFilter(cfg)  # shares the singleton YFinanceClient
    from app.services.universe.trading212.ticker_mapper import YFinanceTickerMapper

    mapper = YFinanceTickerMapper(cfg)
    label = f"{mode}{' +fetch_info' if pipeline else ''}"
    print(
        f"\n=== {label}  (workers={workers}, tickers={len(tickers)}, "
        f"min_trading_days={cfg.min_trading_days}) ==="
    )

    def _one(tk: str):
        return _run_one_pipeline(mapper, f, tk) if pipeline else _run_one(f, tk)

    results: list[tuple[str, bool, str, float, str | None]] = []
    if workers <= 1:
        for tk in tickers:
            results.append(_one(tk))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_one, tk): tk for tk in tickers}
            for fut in as_completed(futures):
                results.append(fut.result())

    passed = 0
    fails: list[tuple] = []
    for tk, ok, reason, _dt, err in sorted(results):
        passed += int(ok)
        if not ok:
            fails.append((tk, reason, err))
    # Print only fails in bulk mode (too many rows otherwise); all rows if small.
    show = (
        results
        if len(results) <= 60
        else [(tk, False, r, 0.0, e) for tk, r, e in fails]
    )
    for tk, ok, reason, _dt, err in sorted(show):
        print(f"  [{'PASS' if ok else 'FAIL'}] {tk:9}  {reason or err}")
    print(f"  -> {passed}/{len(tickers)} passed  ({label})")
    return passed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workers", type=int, default=20, help="concurrent workers (build default 20)"
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat the ticker list N times to sustain load",
    )
    ap.add_argument(
        "--mode",
        choices=["serial", "concurrent", "pipeline", "both"],
        default="pipeline",
        help="pipeline = faithful fetch_info->fetch_history per ticker (like the real build)",
    )
    args = ap.parse_args()
    tickers = TICKERS * args.repeat

    if args.mode in ("serial", "both"):
        _run("SERIAL", 1, tickers)
    if args.mode in ("concurrent", "both"):
        _reset()
        _run("CONCURRENT", args.workers, tickers)
    if args.mode in ("pipeline", "both"):
        _reset()
        _run("PIPELINE", args.workers, tickers, pipeline=True)


def _reset() -> None:
    """Reset the singleton so each run starts clean (fresh session/crumb/breaker)."""
    reset = getattr(YFinanceClient, "reset_instance", None)
    if callable(reset):
        reset()


if __name__ == "__main__":
    main()
