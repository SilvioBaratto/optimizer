"""TEMP: list every Trading 212 tradable instrument.

Source = /api/v0/equity/metadata/instruments (per docs/api.md). Uses the cached
response written by ingestion/scripts/debug_universe_scale.py when present (that
endpoint is rate-limited to 1 req/50s, so we avoid re-hitting it while a build
runs); falls back to a live fetch otherwise.

Prints a breakdown (by type / exchange / currency) and writes the full list to
CSV.

    python scripts/list_t212_instruments.py [out.csv]
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_CACHE = Path("ingestion/scripts/.t212_universe_cache.json")


def _load() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if _CACHE.exists():
        raw = json.loads(_CACHE.read_text(encoding="utf-8"))
        print(f"(source: cache {_CACHE})")
        return raw["exchanges"], raw["instruments"]
    # Live fallback — needs T212 creds in the env (see docs/api.md Authentication).
    sys.path.insert(0, "ingestion")
    from app.services.universe.universe_build_service import build_trading212_client

    c = build_trading212_client()
    print("(source: live T212 API)")
    return c.get_exchanges(), c.get_instruments()


def _exchange_names(exchanges: list[dict[str, Any]]) -> dict[Any, str]:
    """workingScheduleId -> exchange name."""
    m: dict[Any, str] = {}
    for ex in exchanges:
        for sched in ex.get("workingSchedules", []):
            m[sched["id"]] = ex.get("name", "?")
    return m


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/trading212_instruments.csv")
    exchanges, instruments = _load()
    sched_to_ex = _exchange_names(exchanges)

    by_type: Counter[str] = Counter()
    by_exchange: Counter[str] = Counter()
    by_currency: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for inst in instruments:
        ex_name = sched_to_ex.get(inst.get("workingScheduleId"), "?")
        by_type[inst.get("type", "?")] += 1
        by_exchange[ex_name] += 1
        by_currency[inst.get("currencyCode", "?")] += 1
        rows.append(
            {
                "ticker": inst.get("ticker", ""),
                "shortName": inst.get("shortName", ""),
                "name": inst.get("name", ""),
                "type": inst.get("type", ""),
                "currencyCode": inst.get("currencyCode", ""),
                "isin": inst.get("isin", ""),
                "exchange": ex_name,
                "addedOn": inst.get("addedOn", ""),
            }
        )

    rows.sort(key=lambda r: (r["type"], r["exchange"], r["ticker"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nTOTAL instruments: {len(instruments)}   exchanges: {len(exchanges)}")
    print("\nby type:")
    for t, n in by_type.most_common():
        print(f"  {t:16} {n:6}")
    print("\nby exchange:")
    for e, n in by_exchange.most_common():
        print(f"  {e:24} {n:6}")
    print("\ntop 15 currencies:")
    for c, n in by_currency.most_common(15):
        print(f"  {c:6} {n:6}")
    print(f"\nfull list -> {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
