"""FastAPI router for database management endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import database_manager, get_db
from app.repositories.database_admin_repository import (
    APP_TABLES,
    DatabaseAdminRepository,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/database", tags=["Database"])


def _mask_url(url: str) -> str:
    """Mask password in a database URL."""
    try:
        if "@" in url and ":" in url.split("@")[0]:
            prefix, rest = url.split("@", 1)
            scheme_user, _ = prefix.rsplit(":", 1)
            return f"{scheme_user}:***@{rest}"
    except Exception:
        pass
    return "***"


def _get_repo(db: Session = Depends(get_db)) -> DatabaseAdminRepository:
    return DatabaseAdminRepository(db)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@router.get("/health")
def db_health(repo: DatabaseAdminRepository = Depends(_get_repo)) -> dict[str, Any]:
    """Run a SELECT 1 health check and return latency."""
    healthy, latency_ms = repo.check_health()
    result: dict[str, Any] = {
        "healthy": healthy,
        "latency_ms": latency_ms,
        "database_url": _mask_url(settings.database_url),
    }
    return result


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
def db_tables(
    repo: DatabaseAdminRepository = Depends(_get_repo),
) -> list[dict[str, Any]]:
    """List application tables with row counts."""
    return repo.get_table_info(APP_TABLES)


# ---------------------------------------------------------------------------
# Truncate a single table
# ---------------------------------------------------------------------------


@router.delete("/tables/{table_name}")
def db_clear_table(
    table_name: str,
    confirm: bool = Query(False, description="Must be true to actually truncate"),
    repo: DatabaseAdminRepository = Depends(_get_repo),
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
    repo.truncate_table(table_name)
    return {"table": table_name, "status": "truncated"}


# ---------------------------------------------------------------------------
# Truncate all application tables
# ---------------------------------------------------------------------------


@router.delete("/tables")
def db_clear_all(
    confirm: bool = Query(False, description="Must be true to actually truncate"),
    repo: DatabaseAdminRepository = Depends(_get_repo),
) -> dict[str, Any]:
    """Truncate all application tables."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pass ?confirm=true to truncate all tables.",
        )
    cleared, errors = repo.truncate_tables(APP_TABLES)
    return {"cleared": cleared, "errors": errors}
