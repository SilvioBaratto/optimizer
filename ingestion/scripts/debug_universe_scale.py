"""LARGE-SCALE repro — nail the exact trigger of the HistoricalDataFilter collapse.

The bulk universe build rejected 3255/3255 on HistoricalDataFilter while a
40-ticker repro passes 100%. Hypothesis: at scale, Yahoo throttles the *chart*
(history) endpoint after N requests, so fetch_history returns empty/errors while
fetch_info (quoteSummary) keeps working — and the filter converts that into a
silent drop.

This pulls the REAL Trading 212 stock universe (needs TRADING_212_API_KEY), maps
a sample to yfinance, then runs the per-instrument path
(map -> fetch_info -> Ticker.history) at the build's 20-worker concurrency. It
records, in completion order, whether info and history each succeeded — so the
THROTTLE ONSET (the index where history starts failing while info still works)
is visible. History is read from the cached Ticker rather than through the
facade, which hides both exceptions and retries of empty answers.

    python -m scripts.debug_universe_scale --limit 500 --workers 20

Debug it via the "DEBUG: universe scale repro" launch config: breakpoint
fetch_history_raw and watch the raw response at the onset index.
"""

from __future__ import annotations

import argparse
import itertools
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import yfinance as yf

from app.services.market_data.yfinance import YFinanceClient
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.trading212.ticker_mapper import YFinanceTickerMapper
from app.services.universe.universe_build_service import build_trading212_client

_counter = itertools.count(1)
_lock = threading.Lock()


def fetch_history_fixed(
    client: YFinanceClient,
    symbol: str,
    *,
    period: str = "5y",
    min_rows: int = 1,
    fresh_tries: int = 3,
):
    """PROTOTYPE of the ingestion fix.

    yfinance ``Ticker`` is NOT thread-safe (yfinance skill gotcha #11): under
    concurrency ``ticker.history()`` can memoize an empty result ON the Ticker
    instance. Our facade caches Ticker objects by symbol, so the poisoned
    instance is returned forever -> the filter silently drops a live stock.

    Fix: treat an empty result as a *technical failure*, not "no history".
    Evict the cached Ticker and retry with a brand-new instance (proven to
    recover). Only genuinely dataless symbols stay empty after this.
    """
    hist = client.fetch_history(symbol, period=period, max_retries=1, min_rows=min_rows)
    if hist is not None and not hist.empty:
        return hist
    for _ in range(fresh_tries):
        fresh = yf.Ticker(symbol)
        hist = fresh.history(period=period)
        if hist is not None and not hist.empty:
            client.cache.put(symbol, fresh)  # replace the poisoned instance
            return hist
    return hist


def fetch_history_raw(
    client: YFinanceClient,
    symbol: str,
    *,
    attempts: int,
    period: str = "5y",
) -> tuple[Any, str | None]:
    """Request *symbol*'s history from the cached Ticker, keeping rate-limit errors.

    The facade swallows every exception and ends a call after one request when
    Yahoo answers empty, so through it a throttled request is indistinguishable
    from "Yahoo has no data" and an empty answer cannot be asked for again -
    which is what ``--hist-retries`` and the serial re-fetch exist to test.
    Going straight to the Ticker bypasses the facade's circuit breaker too, so
    an open breaker cannot masquerade as a run of empty answers.

    ``Ticker.history`` re-raises only ``YFRateLimitError``. Every other failure
    comes back as an empty frame because yfinance's ``hide_exceptions`` stays
    on, deliberately, so those still read as an empty answer here.

    Args:
        client: Facade whose cached Ticker (and rate limiter) is used.
        symbol: yfinance ticker.
        attempts: Total requests allowed; a value below 1 still makes one.
        period: Lookback period passed to ``Ticker.history``.

    Returns:
        ``(frame, error)``. *frame* is the first non-empty frame, else the
        last answer received (``None`` when the last attempt raised). *error*
        is the ``repr`` of that last attempt's exception, in practice a
        ``YFRateLimitError``, and ``None`` when it did not raise.
    """
    hist = None
    error: str | None = None
    for _ in range(max(1, attempts)):
        try:
            # repair=True mirrors the facade default, whose scikit-learn
            # dependency is part of what this script investigates.
            hist = client.get_ticker(symbol).history(period=period, repair=True)
            error = None
        except Exception as exc:
            hist, error = None, repr(exc)[:120]
            continue
        if hist is not None and not hist.empty:
            return hist, None
    return hist, error


# T212 metadata endpoints are strictly rate-limited (~1 req/min on /instruments).
# Cache the raw exchanges+instruments once so repeated debug runs of THIS script
# don't trip 429 — the investigation target is yfinance, not T212.
_CACHE = Path(__file__).with_name(".t212_universe_cache.json")


def _real_universe_pairs(
    cfg: UniverseBuilderConfig,
) -> list[tuple[str, dict[str, Any]]]:
    """(exchange_name, instrument) for every allowed-exchange STOCK — real T212."""
    if _CACHE.exists():
        raw = json.loads(_CACHE.read_text(encoding="utf-8"))
        exchanges, instruments = raw["exchanges"], raw["instruments"]
        print(f"(loaded T212 universe from cache: {_CACHE.name})")
    else:
        client = build_trading212_client()
        exchanges = client.get_exchanges()
        instruments = client.get_instruments()
        _CACHE.write_text(
            json.dumps({"exchanges": exchanges, "instruments": instruments}),
            encoding="utf-8",
        )
        print(f"(cached T212 universe to {_CACHE.name})")
    allowed = set(cfg.get_allowed_exchanges())

    sched_to_ex: dict[Any, dict[str, Any]] = {}
    for ex in exchanges:
        for sched in ex.get("workingSchedules", []):
            sched_to_ex[sched["id"]] = ex

    pairs: list[tuple[str, dict[str, Any]]] = []
    for inst in instruments:
        if inst.get("type") != "STOCK":
            continue
        ex = sched_to_ex.get(inst.get("workingScheduleId"))
        if not ex:
            continue
        name = ex.get("name")
        if name and name in allowed:
            pairs.append((name, inst))
    return pairs


def _probe(
    mapper: YFinanceTickerMapper,
    client: YFinanceClient,
    exchange_name: str,
    inst: dict,
    hist_retries: int = 1,
) -> dict[str, Any]:
    """Per-instrument path: map -> fetch_info -> Ticker.history. Record outcome.

    A failure inside the probe is recorded under ``error`` instead of raised:
    ``Future.result()`` would re-raise it in the main thread and discard every
    other result.
    """
    symbol = inst.get("shortName") or inst.get("ticker") or "?"
    out: dict[str, Any] = {"symbol": symbol, "exchange": exchange_name}
    try:
        yf = mapper.discover(symbol, exchange_name)
        out["yf"] = yf
        out["mapped"] = bool(yf)
        if yf:
            out["info_ok"] = bool(client.fetch_info(yf, max_retries=1, min_fields=5))
            hist, out["hist_err"] = fetch_history_raw(client, yf, attempts=hist_retries)
            if hist is None or hist.empty:
                out["hist_rows"] = 0
            else:
                out["hist_rows"] = len(hist)
                # Capture the real span so a short row-count can be classified as
                # genuinely-young (recent first date) vs gappy/partial (old first
                # date, sparse rows) rather than assumed "short history".
                out["first"] = str(hist.index[0].date())
                out["last"] = str(hist.index[-1].date())
    except Exception as exc:
        out["error"] = repr(exc)[:120]
    with _lock:
        out["idx"] = next(_counter)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=500, help="how many stocks to probe")
    ap.add_argument("--workers", type=int, default=20, help="build default is 20")
    ap.add_argument(
        "--hist-retries",
        type=int,
        default=1,
        help="history requests per symbol; >1 re-asks after an empty answer or "
        "an error",
    )
    ap.add_argument(
        "--fix",
        action="store_true",
        help="re-fetch empties via the prototype fetch_history_fixed (evict + fresh Ticker)",
    )
    args = ap.parse_args()

    cfg = UniverseBuilderConfig()
    mapper = YFinanceTickerMapper(cfg)
    client = YFinanceClient.get_instance()

    print("Fetching real Trading 212 universe...")
    pairs = _real_universe_pairs(cfg)
    print(
        f"Universe: {len(pairs)} allowed-exchange STOCKs; probing first {args.limit} "
        f"at {args.workers} workers (min_trading_days={cfg.min_trading_days})"
    )
    sample = pairs[: args.limit]

    started = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(_probe, mapper, client, exn, inst, args.hist_retries)
            for exn, inst in sample
        ]
        for fut in as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: r["idx"])
    mapped = [r for r in results if r.get("mapped")]
    info_ok = [r for r in mapped if r.get("info_ok")]
    hist_ok = [r for r in mapped if r.get("hist_rows", 0) >= cfg.min_trading_days]
    hist_empty = [
        r
        for r in mapped
        if r.get("hist_rows", 0) == 0 and not r.get("hist_err") and not r.get("error")
    ]
    hist_err = [r for r in mapped if r.get("hist_err")]
    probe_err = [r for r in results if r.get("error")]

    # Throttle onset: first completion where info worked but history came back empty.
    onset = next(
        (
            r
            for r in results
            if r.get("info_ok") and r.get("hist_rows", 0) == 0 and not r.get("error")
        ),
        None,
    )

    print(f"\n===== RESULTS ({time.time() - started:.0f}s) =====")
    print(f"probed              : {len(results)}")
    print(f"mapped (yf found)   : {len(mapped)}")
    print(f"fetch_info ok       : {len(info_ok)}")
    print(f"fetch_history >=750 : {len(hist_ok)}")
    print(f"history EMPTY (0)   : {len(hist_empty)}")
    print(f"history ERROR       : {len(hist_err)}")
    print(f"probe ERROR         : {len(probe_err)}")
    if onset:
        print(
            f"\n*** THROTTLE ONSET at completion idx {onset['idx']}: "
            f"{onset['symbol']} ({onset['yf']}) info_ok={onset['info_ok']} hist_rows=0 ***"
        )
    # Health timeline: history-ok rate per 50 completions → shows the collapse point.
    print("\nhistory-ok rate per 50 completions (info_ok in parens):")
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in mapped:
        buckets[(r["idx"] - 1) // 50].append(r)
    for b in sorted(buckets):
        rs = buckets[b]
        hok = sum(1 for r in rs if r.get("hist_rows", 0) >= cfg.min_trading_days)
        iok = sum(1 for r in rs if r.get("info_ok"))
        print(
            f"  idx {b * 50 + 1:4}-{b * 50 + 50:<4}: history {hok:2}/{len(rs):2}  (info {iok}/{len(rs)})"
        )
    if hist_err:
        print("\nsample history errors:")
        for r in hist_err[:5]:
            print(f"  {r['symbol']} ({r['yf']}): {r['hist_err']}")

    if probe_err:
        print("\nsample probe errors:")
        for r in probe_err[:5]:
            print(f"  {r['symbol']} ({r.get('yf')}): {r['error']}")

    # Short-history diagnosis: 0 < rows < min_trading_days. Classify each as
    #   YOUNG   -> first trade date is recent; the listing is genuinely too short
    #              to ever reach min_trading_days (a legitimate rejection), OR
    #   GAPPY   -> first trade date is old enough (span >= min years) but rows are
    #              sparse -> partial data / calendar / thin trading, NOT "young".
    min_years = cfg.min_trading_days / 252.0
    short = sorted(
        (r for r in mapped if 0 < r.get("hist_rows", 0) < cfg.min_trading_days),
        key=lambda r: r.get("hist_rows", 0),
    )
    if short:
        print(
            f"\nSHORT HISTORY (0 < rows < {cfg.min_trading_days}, "
            f"~{min_years:.1f}y): {len(short)} tickers"
        )
        print(
            f"  {'symbol':12} {'yf':12} {'rows':>5} {'first':>11} {'last':>11} "
            f"{'span_y':>7} {'rows/yr':>7}  class"
        )
        young = gappy = 0
        for r in short:
            rows = r["hist_rows"]
            first, last = r.get("first"), r.get("last")
            if first and last:
                span_y = (
                    date.fromisoformat(last) - date.fromisoformat(first)
                ).days / 365.25
            else:
                span_y = 0.0
            per_yr = rows / span_y if span_y > 0.05 else 0.0
            # span shorter than the min window => cannot physically reach the floor.
            klass = "YOUNG" if span_y < min_years else "GAPPY"
            young += klass == "YOUNG"
            gappy += klass == "GAPPY"
            print(
                f"  {r['symbol']:12} {r['yf']!s:12} {rows:5} {first!s:>11} "
                f"{last!s:>11} {span_y:7.2f} {per_yr:7.0f}  {klass}"
            )
        print(
            f"  => YOUNG (genuine short listing): {young}   GAPPY (partial/calendar): {gappy}"
        )

    # Discriminator: re-fetch every EMPTY serially, in isolation, with retries.
    # Recovered => the in-run empty was load/session related (transient).
    # Still empty => genuine (delisted / wrong suffix / truly < min_rows history).
    if hist_empty:
        how = (
            "fetch_history_fixed (evict + fresh Ticker)"
            if args.fix
            else "raw Ticker.history (up to 3 attempts)"
        )
        print(f"\nre-fetching {len(hist_empty)} EMPTY tickers SERIALLY via {how}:")
        recovered = 0
        for r in hist_empty:
            err: str | None = None
            try:
                if args.fix:
                    h = fetch_history_fixed(client, r["yf"], period="5y", min_rows=1)
                else:
                    h, err = fetch_history_raw(client, r["yf"], attempts=3)
                n = 0 if h is None or h.empty else len(h)
            except Exception as exc:
                n, err = 0, repr(exc)[:120]
            if n > 0:
                recovered += 1
                tag = f"RECOVERED {n} rows"
            else:
                tag = f"still failing: {err}" if err else "still empty"
            print(f"  {r['symbol']:12} ({r['yf']:14}) -> {tag}")
        verdict = (
            "FIX WORKS"
            if args.fix and recovered == len(hist_empty)
            else (
                "FIX PARTIAL"
                if args.fix
                else ("load/session" if recovered else "genuine")
            )
        )
        print(f"  => recovered {recovered}/{len(hist_empty)}  [{verdict}]")


if __name__ == "__main__":
    main()
