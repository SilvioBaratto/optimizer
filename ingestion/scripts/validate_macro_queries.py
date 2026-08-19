"""Validate that every MACRO_SEARCH_QUERIES entry returns recent content.

Runs each query against live yfinance search and asserts that:

1. yfinance returns at least one result.
2. At least one result passes the recency filter (<= 60 days old).

Exits with status 1 if any query regresses, 0 otherwise.  Intended to be
run manually when modifying MACRO_SEARCH_QUERIES or on a weekly schedule
to catch upstream search-indexing regressions early.

Usage (from repo root):

    python ingestion/scripts/validate_macro_queries.py
    python ingestion/scripts/validate_macro_queries.py --json         # machine-readable
    python ingestion/scripts/validate_macro_queries.py --min-recent 2 # stricter

The script never hits the database and makes only read-only calls to the
live yfinance search endpoint, so it is safe to run ad hoc.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add repo root + ingestion/ to sys.path so ``app.services.*`` imports work
# when the script is invoked directly (without installing the ingestion package).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "ingestion"))

import yfinance as yf
from dateutil import parser as dateutil_parser

from app.services.market_data.yfinance.news.macro_news import MACRO_SEARCH_QUERIES

_MAX_AGE_DAYS = 60
_DEFAULT_MAX_RESULTS = 8


# ---------------------------------------------------------------------------
# Helpers (mirrors logic in app.services.market_data.yfinance.news aggregator/_process)
# ---------------------------------------------------------------------------


def _extract_publisher(raw: dict[str, Any]) -> str:
    content = raw.get("content", raw)
    provider = content.get("provider")
    if isinstance(provider, dict):
        return provider.get("displayName", "") or ""
    return content.get("publisher", "") or ""


def _extract_pub_time(raw: dict[str, Any]) -> Any:
    content = raw.get("content", raw)
    return content.get("pubDate", content.get("providerPublishTime", "N/A"))


def _parse_date(pt: Any) -> datetime | None:
    if pt is None or pt == "N/A":
        return None
    try:
        if isinstance(pt, int):
            return datetime.fromtimestamp(pt)
        if isinstance(pt, str):
            d = dateutil_parser.isoparse(pt)
            if d.tzinfo is not None:
                d = d.replace(tzinfo=None)
            return d
    except Exception:
        return None
    return None


def _is_recent(pt: Any, cutoff: datetime) -> bool:
    d = _parse_date(pt)
    return d is not None and d >= cutoff


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------


def evaluate_query(query: str, cutoff: datetime) -> dict[str, Any]:
    """Run a single yfinance.Search query and report totals + recency.

    Returns a dict with ``total``, ``recent``, ``publishers`` (set[str]),
    and an optional ``error`` string.
    """
    try:
        result = yf.Search(query, max_results=_DEFAULT_MAX_RESULTS)
        news = result.news or []
    except Exception as exc:
        return {
            "total": 0,
            "recent": 0,
            "publishers": set(),
            "error": str(exc),
        }

    recent = 0
    publishers: set[str] = set()
    for raw in news:
        publishers.add(_extract_publisher(raw))
        if _is_recent(_extract_pub_time(raw), cutoff):
            recent += 1

    return {
        "total": len(news),
        "recent": recent,
        "publishers": publishers,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-recent",
        type=int,
        default=1,
        help="Minimum recent articles each query must return (default: 1)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text report",
    )
    args = parser.parse_args()

    cutoff = datetime.now() - timedelta(days=_MAX_AGE_DAYS)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for query in MACRO_SEARCH_QUERIES:
        result = evaluate_query(query, cutoff)
        passed = result["error"] is None and result["recent"] >= args.min_recent
        rows.append(
            {
                "query": query,
                "total": result["total"],
                "recent": result["recent"],
                "publishers": sorted(result["publishers"]),
                "passed": passed,
                "error": result["error"],
            },
        )
        if not passed:
            reason = (
                f"error: {result['error']}"
                if result["error"]
                else f"{result['recent']} recent < {args.min_recent}"
            )
            failures.append(f"{query!r}: {reason}")

    if args.json:
        print(json.dumps({"rows": rows, "failures": failures}, indent=2))
    else:
        print(f"{'Query':<45} | {'Total':>5} | {'Recent':>6} | Status")
        print("-" * 90)
        for row in rows:
            status = "PASS" if row["passed"] else "FAIL"
            print(
                f"{row['query']:<45} | "
                f"{row['total']:>5} | "
                f"{row['recent']:>6} | {status}",
            )
        print()
        if failures:
            print(f"✗ {len(failures)} of {len(rows)} queries failed:")
            for f in failures:
                print(f"  - {f}")
        else:
            print(f"✓ all {len(rows)} queries passed (min recent={args.min_recent})")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
