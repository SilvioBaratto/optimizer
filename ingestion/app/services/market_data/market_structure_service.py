"""Sector / industry market-structure ingestion (SPEC B1 / OQ4).

Sweeps the 11 ``yf.Sector`` keys across a configured region set and persists a
point-in-time snapshot (overview + industry taxonomy + regional top companies)
per (sector, region). yfinance exposes no enumeration of supported regions, so
``DEFAULT_REGIONS`` is an explicit curated list of the regions Yahoo scopes the
rollups for; the effective set is logged each run (no silent cap).

Market-wide + slow-moving → run by its own weekly scheduler step, not the daily
per-ticker loop.

Yahoo returns the same sector overview (market cap, weight, counts) whatever
``region`` is requested; only the industry / top-company lists vary.  The
overview columns are therefore persisted on the ``OVERVIEW_REGION`` row alone and
left NULL elsewhere, so a regional snapshot never carries another scope's totals.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from portopt_db.repositories.market_data.market_structure_repository import _int, _num

from app.services._shared import ProgressCallback, _noop
from app.services.market_data.yfinance import YFinanceClient
from app.services.market_data.yfinance.market.sectors import SECTOR_KEYS

logger = logging.getLogger(__name__)

# Yahoo answers regions it does not scope (NL, CH) with the US rollup, so they
# are left out; the sweep also skips any region whose list mirrors the US one.
DEFAULT_REGIONS: tuple[str, ...] = (
    "US",
    "GB",
    "DE",
    "FR",
    "IT",
    "ES",
    "SE",
    "CA",
)

OVERVIEW_REGION = "US"


# Yahoo answers a region it does not scope with the US rollup, so the US list is
# the reference a region's top companies are compared against.
_REFERENCE_REGION = OVERVIEW_REGION


def _top_symbols(data: dict[str, Any]) -> frozenset[str]:
    return frozenset(
        str(c["symbol"]) for c in data.get("top_companies") or [] if c.get("symbol")
    )


def run_market_structure_fetch(
    yf_client: YFinanceClient,
    *,
    regions: tuple[str, ...] = DEFAULT_REGIONS,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Fetch and persist sector/industry rollups for every (sector, region) pair.

    Args:
        yf_client: Configured yfinance client.
        regions: ISO country codes to sweep; defaults to ``DEFAULT_REGIONS``.
        on_progress: Callback receiving progress updates.

    Returns:
        Dict with regions swept, row counts per layer (sectors, industries,
        top companies), the ``sector/region`` pairs skipped because Yahoo
        returned the US rollup for a region it does not scope, and the total
        error count.
    """
    from portopt_db.repositories.market_data.market_structure_repository import (
        MarketStructureRepository,
    )

    from app.database import database_manager

    now = datetime.now(timezone.utc)
    as_of = now.date()
    logger.info(
        "Market-structure sweep: %d sectors x %d regions (%s)",
        len(SECTOR_KEYS),
        len(regions),
        ",".join(regions),
    )

    with database_manager.get_session() as session:
        repo = MarketStructureRepository(session)
        total = len(SECTOR_KEYS) * len(regions)
        on_progress(total=total)

        errors: list[str] = []
        skipped_unscoped: list[str] = []
        reference_symbols: dict[str, frozenset[str]] = {}
        sectors_written = 0
        industries_written = 0
        companies_written = 0
        idx = 0

        # The reference region goes first so every other region can be compared.
        for region in sorted(regions, key=lambda r: r != _REFERENCE_REGION):
            for key in SECTOR_KEYS:
                idx += 1
                on_progress(current=idx, current_sector=key, current_region=region)
                try:
                    data = yf_client.sectors.fetch_sector(key, region=region)
                    if not data:
                        logger.warning("No data for sector %s/%s", key, region)
                        errors.append(f"{key}/{region}: no data")
                        continue
                    symbols = _top_symbols(data)
                    if region == _REFERENCE_REGION:
                        reference_symbols[key] = symbols
                    elif symbols and symbols == reference_symbols.get(key):
                        logger.warning(
                            "Sector %s/%s mirrors the %s list; Yahoo does not "
                            "scope this region, skipping",
                            key,
                            region,
                            _REFERENCE_REGION,
                        )
                        skipped_unscoped.append(f"{key}/{region}")
                        continue
                    ov = (
                        (data.get("overview") or {})
                        if region == OVERVIEW_REGION
                        else {}
                    )
                    repo.upsert_sector_snapshot(
                        key,
                        region,
                        as_of,
                        name=data.get("name"),
                        symbol=data.get("symbol"),
                        market_cap=_num(ov.get("market_cap")),
                        market_weight=_num(ov.get("market_weight")),
                        companies_count=_int(ov.get("companies_count")),
                        industries_count=_int(ov.get("industries_count")),
                        employee_count=_int(ov.get("employee_count")),
                    )
                    n_industries = repo.upsert_industries(
                        key, region, as_of, data.get("industries") or []
                    )
                    n_companies = repo.upsert_top_companies(
                        key, region, as_of, data.get("top_companies") or []
                    )
                    session.commit()
                    sectors_written += 1
                    industries_written += n_industries
                    companies_written += n_companies
                except Exception as e:  # one bad sector/region must not abort
                    logger.warning("Failed sector %s/%s: %s", key, region, e)
                    errors.append(f"{key}/{region}: {e}")
                    session.rollback()

        result = {
            "regions": list(regions),
            "sectors_written": sectors_written,
            "industries_written": industries_written,
            "top_companies_written": companies_written,
            "skipped_unscoped": skipped_unscoped,
            "error_count": len(errors),
        }
        logger.info(
            "Market-structure sweep done: %d sector rows, %d industries, %d companies",
            sectors_written,
            industries_written,
            companies_written,
        )
        on_progress(
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            errors=errors,
            result=result,
        )

    return result
