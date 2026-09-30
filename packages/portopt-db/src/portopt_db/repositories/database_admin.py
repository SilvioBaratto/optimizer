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

from portopt_db.repository import RepositoryBase

logger = logging.getLogger(__name__)

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

_PUBLIC_SCHEMA: str = "public"


def _missing_table_row(table_name: str) -> dict[str, Any]:
    """Placeholder row for a table that is allowlisted but absent from the DB."""
    return {
        "name": table_name,
        "schema": _PUBLIC_SCHEMA,
        "exists": False,
        "row_count": None,
        "size_bytes": None,
        "size_pretty": "—",
    }


class DatabaseAdminRepository(RepositoryBase):
    """Sync repository for database introspection and truncation."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def check_health(self) -> tuple[bool, float]:
        """Probe the database connection and report round-trip latency.

        Returns:
            A ``(healthy, latency_ms)`` pair.  *healthy* is ``False`` when the
            probe query raises — the exception is logged, not re-raised.
        """
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
        """Return introspection rows for each requested table.

        Each row contains:

        * ``name`` — table name
        * ``schema`` — Postgres schema (always ``"public"`` for managed tables)
        * ``exists`` — whether the table is present in ``information_schema``
        * ``row_count`` — exact row count (``None`` when the table is missing)
        * ``size_bytes`` — total relation size in bytes; ``None`` when missing
        * ``size_pretty`` — human-readable size; ``"—"`` when missing

        Args:
            table_names: Tables to introspect; unrecognised or absent tables
                are represented with ``exists=False`` and ``None`` metrics.
        """
        return [self._table_row(name) for name in table_names]

    def _table_row(self, table_name: str) -> dict[str, Any]:
        """Build a single ``TableInfo`` row for *table_name*."""
        if not self._table_exists(table_name):
            return _missing_table_row(table_name)
        return {
            "name": table_name,
            "schema": _PUBLIC_SCHEMA,
            "exists": True,
            "row_count": self._row_count(table_name),
            "size_bytes": self._size_bytes(table_name),
            "size_pretty": self._size_pretty(table_name),
        }

    def _table_exists(self, table_name: str) -> bool:
        result = self.session.execute(
            text(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables "
                "  WHERE table_schema = :schema AND table_name = :name"
                ")"
            ),
            {"schema": _PUBLIC_SCHEMA, "name": table_name},
        )
        return bool(result.scalar())

    def _row_count(self, table_name: str) -> int | None:
        # Direct interpolation is safe: callers must validate against _ALLOWED_TABLES
        # before invoking this private helper — parameterised queries don't support
        # identifiers, so the allowlist is the sole injection defence.
        result = self.session.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
        return result.scalar()

    def _size_bytes(self, table_name: str) -> int | None:
        result = self.session.execute(
            text(
                "SELECT pg_total_relation_size("
                "  format('%I.%I', :schema, :name)::regclass"
                ")"
            ),
            {"schema": _PUBLIC_SCHEMA, "name": table_name},
        )
        return result.scalar()

    def _size_pretty(self, table_name: str) -> str:
        result = self.session.execute(
            text(
                "SELECT pg_size_pretty(pg_total_relation_size("
                "  format('%I.%I', :schema, :name)::regclass"
                "))"
            ),
            {"schema": _PUBLIC_SCHEMA, "name": table_name},
        )
        return str(result.scalar() or "—")

    def truncate_table(self, table_name: str) -> None:
        """Truncate a single table, cascading to all dependent rows.

        Args:
            table_name: Must be present in ``_ALLOWED_TABLES``; any other
                value raises ``ValueError`` before touching the database.

        Raises:
            ValueError: When *table_name* is not a managed application table.
        """
        if table_name not in _ALLOWED_TABLES:
            raise ValueError(
                f"Table '{table_name}' is not a managed application table."
            )
        self.session.execute(text(f'TRUNCATE TABLE "{table_name}" CASCADE'))
        self.session.commit()
        logger.info("Truncated table: %s", table_name)

    def truncate_tables(self, table_names: list[str]) -> tuple[list[str], list[str]]:
        """Truncate each listed table, rolling back per-table on failure.

        Unrecognised table names are rejected without touching the database.
        A single commit is issued for all successfully cleared tables; each
        failure triggers a rollback of that table's statement only.

        Args:
            table_names: Tables to truncate; each must be in ``_ALLOWED_TABLES``
                or it is skipped and its name recorded in the error list.

        Returns:
            A ``(cleared, errors)`` pair where *cleared* holds names that were
            successfully truncated and *errors* holds failure messages.
        """
        cleared: list[str] = []
        errors: list[str] = []

        for table_name in table_names:
            try:
                if table_name not in _ALLOWED_TABLES:
                    errors.append(f"{table_name}: not a managed table")
                    continue
                self.session.execute(text(f'TRUNCATE TABLE "{table_name}" CASCADE'))
                cleared.append(table_name)
            except Exception as exc:
                logger.error("Failed to truncate %s: %s", table_name, exc)
                errors.append(f"{table_name}: {exc}")
                self.session.rollback()

        if cleared:
            self.session.commit()

        logger.info("Truncated %d tables", len(cleared))
        return cleared, errors
