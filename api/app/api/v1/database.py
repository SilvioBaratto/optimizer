"""FastAPI router for database management endpoints."""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import database_manager, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/database", tags=["Database"])

# Tables managed by the application (derived from models)
APP_TABLES: list[str] = [
    "economic_indicators",
    "trading_economics_indicators",
    "bond_yields",
    "exchanges",
    "instruments",
    "ticker_profiles",
    "price_history",
    "financial_statements",
    "dividends",
    "stock_splits",
    "analyst_recommendations",
    "analyst_price_targets",
    "institutional_holders",
    "mutual_fund_holders",
    "insider_transactions",
    "ticker_news",
]


def _mask_url(url: str) -> str:
    """Mask password in a database URL."""
    # postgresql://user:password@host:port/db -> postgresql://user:***@host:port/db
    try:
        if "@" in url and ":" in url.split("@")[0]:
            prefix, rest = url.split("@", 1)
            scheme_user, _ = prefix.rsplit(":", 1)
            return f"{scheme_user}:***@{rest}"
    except Exception:
        pass
    return "***"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@router.get("/health")
def db_health(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Run a SELECT 1 health check and return latency."""
    start = time.perf_counter()
    try:
        result = db.execute(text("SELECT 1"))
        result.fetchone()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "healthy": True,
            "latency_ms": latency_ms,
            "database_url": _mask_url(settings.database_url),
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.error("Database health check failed: %s", exc)
        return {
            "healthy": False,
            "latency_ms": latency_ms,
            "database_url": _mask_url(settings.database_url),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Detailed status
# ---------------------------------------------------------------------------


@router.get("/status")
def db_status() -> dict[str, Any]:
    """Return detailed database manager status (pool info, config)."""
    return database_manager.get_detailed_status()


# ---------------------------------------------------------------------------
# Table info with row counts
# ---------------------------------------------------------------------------


@router.get("/tables")
def db_tables(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """List application tables with row counts."""
    tables: list[dict[str, Any]] = []

    for table_name in APP_TABLES:
        # Check if table exists
        exists_result = db.execute(
            text(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables "
                "  WHERE table_schema = 'public' AND table_name = :name"
                ")"
            ),
            {"name": table_name},
        )
        exists = exists_result.scalar()

        if exists:
            count_result = db.execute(
                text(f'SELECT COUNT(*) FROM "{table_name}"')  # noqa: S608
            )
            row_count = count_result.scalar()
        else:
            row_count = None

        tables.append(
            {
                "table_name": table_name,
                "exists": bool(exists),
                "row_count": row_count,
            }
        )

    return tables


# ---------------------------------------------------------------------------
# Truncate a single table
# ---------------------------------------------------------------------------


@router.delete("/tables/{table_name}")
def db_clear_table(
    table_name: str,
    confirm: bool = Query(False, description="Must be true to actually truncate"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Truncate a single application table."""
    if table_name not in APP_TABLES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table '{table_name}' is not a managed application table.",
        )

    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pass ?confirm=true to truncate the table.",
        )

    db.execute(text(f'TRUNCATE TABLE "{table_name}" CASCADE'))
    db.commit()
    logger.info("Truncated table: %s", table_name)

    return {"table": table_name, "status": "truncated"}


# ---------------------------------------------------------------------------
# Truncate all application tables
# ---------------------------------------------------------------------------


@router.delete("/tables")
def db_clear_all(
    confirm: bool = Query(False, description="Must be true to actually truncate"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Truncate all application tables."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pass ?confirm=true to truncate all tables.",
        )

    cleared: list[str] = []
    errors: list[str] = []

    for table_name in APP_TABLES:
        try:
            db.execute(text(f'TRUNCATE TABLE "{table_name}" CASCADE'))
            cleared.append(table_name)
        except Exception as exc:
            logger.error("Failed to truncate %s: %s", table_name, exc)
            errors.append(f"{table_name}: {exc}")
            db.rollback()

    if cleared:
        db.commit()

    logger.info("Truncated %d tables", len(cleared))
    return {"cleared": cleared, "errors": errors}
