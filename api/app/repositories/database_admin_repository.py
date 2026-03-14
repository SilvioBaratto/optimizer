"""Repository for database administration operations.

Encapsulates health checks, table introspection, and truncation behind
a typed interface. Table names are validated against an allowlist before
any DDL is executed.
"""

import logging
import time
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.repositories.base import RepositoryBase

logger = logging.getLogger(__name__)

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

_ALLOWED_TABLES: frozenset[str] = frozenset(APP_TABLES)


class DatabaseAdminRepository(RepositoryBase):
    """Sync repository for database introspection and truncation."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def check_health(self) -> tuple[bool, float]:
        """Run ``SELECT 1`` and return ``(healthy, latency_ms)``."""
        start = time.perf_counter()
        try:
            result = self.session.execute(text("SELECT 1"))
            result.fetchone()
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return True, latency_ms
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error("Health check failed: %s", exc)
            return False, latency_ms

    def get_table_info(self, table_names: list[str]) -> list[dict[str, Any]]:
        """Return existence and row counts for each table in *table_names*."""
        tables: list[dict[str, Any]] = []
        for table_name in table_names:
            exists_result = self.session.execute(
                text(
                    "SELECT EXISTS ("
                    "  SELECT 1 FROM information_schema.tables "
                    "  WHERE table_schema = 'public' AND table_name = :name"
                    ")"
                ),
                {"name": table_name},
            )
            exists = exists_result.scalar()

            row_count: int | None = None
            if exists:
                count_result = self.session.execute(
                    text(f'SELECT COUNT(*) FROM "{table_name}"')  # noqa: S608
                )
                row_count = count_result.scalar()

            tables.append(
                {
                    "table_name": table_name,
                    "exists": bool(exists),
                    "row_count": row_count,
                }
            )
        return tables

    def truncate_table(self, table_name: str) -> None:
        """Truncate a single table (with CASCADE).

        Raises ValueError for unknown tables.
        """
        if table_name not in _ALLOWED_TABLES:
            raise ValueError(
                f"Table '{table_name}' is not a managed application table."
            )
        self.session.execute(text(f'TRUNCATE TABLE "{table_name}" CASCADE'))
        self.session.commit()
        logger.info("Truncated table: %s", table_name)

    def truncate_tables(
        self, table_names: list[str]
    ) -> tuple[list[str], list[str]]:
        """Truncate multiple tables. Returns ``(cleared, errors)``."""
        cleared: list[str] = []
        errors: list[str] = []

        for table_name in table_names:
            try:
                if table_name not in _ALLOWED_TABLES:
                    errors.append(f"{table_name}: not a managed table")
                    continue
                self.session.execute(
                    text(f'TRUNCATE TABLE "{table_name}" CASCADE')
                )
                cleared.append(table_name)
            except Exception as exc:
                logger.error("Failed to truncate %s: %s", table_name, exc)
                errors.append(f"{table_name}: {exc}")
                self.session.rollback()

        if cleared:
            self.session.commit()

        logger.info("Truncated %d tables", len(cleared))
        return cleared, errors
