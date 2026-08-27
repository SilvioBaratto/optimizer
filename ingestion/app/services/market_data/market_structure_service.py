"""Sector / industry market-structure ingestion (SPEC B1 / OQ4).

Sweeps the 11 ``yf.Sector`` keys across a configured region set and persists a
point-in-time snapshot (overview + industry taxonomy + regional top companies)
per (sector, region). yfinance exposes no enumeration of supported regions, so
``DEFAULT_REGIONS`` is an explicit curated list covering the universe's markets;
the effective set is logged each run (no silent cap).

Market-wide + slow-moving → run by its own weekly scheduler step, not the daily
per-ticker loop.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.services._shared import ProgressCallback, _noop
from app.services.market_data.yfinance import YFinanceClient
from app.services.market_data.yfinance.market.sectors import SECTOR_KEYS

logger = logging.getLogger(__name__)

DEFAULT_REGIONS: tuple[str, ...] = (
    "US",
    "GB",
    "DE",
    "FR",
    "IT",
    "ES",
    "NL",
    "CH",
    "SE",
    "CA",
)


def _num(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _int(v: Any) -> int | None:
    f = _num(v)
    return int(f) if f is not None else None


def run_market_structure_fetch(
    yf_client: YFinanceClient,
    *,
    regions: tuple[str, ...] = DEFAULT_REGIONS,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Fetch + persist sector/industry rollups for every (sector, region)."""
    from app.database import database_manager
    from app.repositories.market_data.market_structure_repository import (
        MarketStructureRepository,
    )

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
        sectors_written = 0
        industries_written = 0
        companies_written = 0
        idx = 0

        for region in regions:
            for key in SECTOR_KEYS:
                idx += 1
                on_progress(current=idx, current_sector=key, current_region=region)
                try:
                    data = yf_client.sectors.fetch_sector(key, region=region)
                    if not data:
                        continue
                    ov = data.get("overview") or {}
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
                    sectors_written += 1
                    industries_written += repo.upsert_industries(
                        key, region, as_of, data.get("industries") or []
                    )
                    companies_written += repo.upsert_top_companies(
                        key, region, as_of, data.get("top_companies") or []
                    )
                    session.commit()
                except Exception as e:  # one bad sector/region must not abort
                    logger.warning("Failed sector %s/%s: %s", key, region, e)
                    errors.append(f"{key}/{region}: {e}")
                    session.rollback()

        result = {
            "regions": list(regions),
            "sectors_written": sectors_written,
            "industries_written": industries_written,
            "top_companies_written": companies_written,
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
            finished_at=now.isoformat(),
            errors=errors,
            result=result,
        )

    return result
