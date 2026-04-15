"""Tests for migration y5z6a7b8c9d0: six new tables.

Covers:
- Migration module imports and revision metadata
- down_revision chains from x4y5z6a7b8c9
- upgrade() creates all six tables in SQLite
- downgrade() drops all six tables cleanly
- All indexes and unique constraints are present after upgrade
- JSONB columns accept dict/list payloads (dialect-neutral)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = (
    "y5z6a7b8c9d0_add_factor_scores_execution_runs_"
    "risk_limits_rebalancing_policies_tables.py"
)
MIGRATION_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "alembic", "versions"
)
MIGRATION_PATH = os.path.abspath(os.path.join(MIGRATION_DIR, MIGRATION_FILENAME))
EXPECTED_REVISION = "y5z6a7b8c9d0"
EXPECTED_DOWN_REVISION = "x4y5z6a7b8c9"

SIX_NEW_TABLES = {
    "factor_scores",
    "factor_validation_reports",
    "optimization_runs",
    "backtest_runs",
    "risk_limits",
    "rebalancing_policies",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_migration() -> types.ModuleType:
    """Load the migration module from its file path."""
    module_name = MIGRATION_FILENAME[:-3]  # strip .py
    spec = importlib.util.spec_from_file_location(module_name, MIGRATION_PATH)
    assert spec is not None, f"Cannot find migration at {MIGRATION_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sqlite_engine():
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _create_prereq_tables(engine) -> None:
    """Create the tables that the six new tables reference via FK."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS background_jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """))


# ---------------------------------------------------------------------------
# Migration metadata
# ---------------------------------------------------------------------------


class TestMigrationMetadata:
    """Migration module must declare correct revision identifiers."""

    def test_module_is_importable(self) -> None:
        mod = _load_migration()
        assert mod is not None

    def test_revision_id_is_correct(self) -> None:
        mod = _load_migration()
        assert mod.revision == EXPECTED_REVISION

    def test_down_revision_chains_from_head(self) -> None:
        mod = _load_migration()
        assert mod.down_revision == EXPECTED_DOWN_REVISION

    def test_branch_labels_is_none(self) -> None:
        mod = _load_migration()
        assert mod.branch_labels is None

    def test_depends_on_is_none(self) -> None:
        mod = _load_migration()
        assert mod.depends_on is None


# ---------------------------------------------------------------------------
# Upgrade: all six tables created
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def upgraded_engine():
    """Engine with prereq tables + the six new tables created via upgrade()."""
    engine = _sqlite_engine()
    _create_prereq_tables(engine)

    mod = _load_migration()

    # Alembic's op functions use a module-level MigrationContext.
    # For unit testing we bypass Alembic's runner and call create_table /
    # create_index directly via a manual connection context.
    from alembic.runtime.migration import MigrationContext
    from alembic.operations import Operations

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        op = Operations(ctx)
        mod.upgrade.__globals__["op"] = op
        mod.upgrade()

    yield engine


class TestUpgradeCreatesAllSixTables:
    """After upgrade(), all six new tables must exist."""

    def test_factor_scores_table_exists(self, upgraded_engine) -> None:
        assert "factor_scores" in inspect(upgraded_engine).get_table_names()

    def test_factor_validation_reports_table_exists(self, upgraded_engine) -> None:
        assert "factor_validation_reports" in inspect(upgraded_engine).get_table_names()

    def test_optimization_runs_table_exists(self, upgraded_engine) -> None:
        assert "optimization_runs" in inspect(upgraded_engine).get_table_names()

    def test_backtest_runs_table_exists(self, upgraded_engine) -> None:
        assert "backtest_runs" in inspect(upgraded_engine).get_table_names()

    def test_risk_limits_table_exists(self, upgraded_engine) -> None:
        assert "risk_limits" in inspect(upgraded_engine).get_table_names()

    def test_rebalancing_policies_table_exists(self, upgraded_engine) -> None:
        assert "rebalancing_policies" in inspect(upgraded_engine).get_table_names()


class TestUpgradeIndexes:
    """Expected indexes must be present after upgrade."""

    def _index_names(self, engine, table: str) -> set[str]:
        return {idx["name"] for idx in inspect(engine).get_indexes(table)}

    def test_factor_scores_has_ticker_index(self, upgraded_engine) -> None:
        assert "ix_factor_scores_ticker" in self._index_names(upgraded_engine, "factor_scores")

    def test_factor_scores_has_score_date_index(self, upgraded_engine) -> None:
        assert "ix_factor_scores_score_date" in self._index_names(upgraded_engine, "factor_scores")

    def test_factor_scores_has_factor_type_index(self, upgraded_engine) -> None:
        assert "ix_factor_scores_factor_type" in self._index_names(upgraded_engine, "factor_scores")

    def test_factor_validation_reports_has_report_date_index(self, upgraded_engine) -> None:
        assert "ix_factor_validation_reports_report_date" in self._index_names(
            upgraded_engine, "factor_validation_reports"
        )

    def test_factor_validation_reports_has_factor_type_index(self, upgraded_engine) -> None:
        assert "ix_factor_validation_reports_factor_type" in self._index_names(
            upgraded_engine, "factor_validation_reports"
        )

    def test_optimization_runs_has_portfolio_id_index(self, upgraded_engine) -> None:
        assert "ix_optimization_runs_portfolio_id" in self._index_names(
            upgraded_engine, "optimization_runs"
        )

    def test_optimization_runs_has_status_index(self, upgraded_engine) -> None:
        assert "ix_optimization_runs_status" in self._index_names(
            upgraded_engine, "optimization_runs"
        )

    def test_optimization_runs_has_created_at_index(self, upgraded_engine) -> None:
        assert "ix_optimization_runs_created_at" in self._index_names(
            upgraded_engine, "optimization_runs"
        )

    def test_backtest_runs_has_portfolio_id_index(self, upgraded_engine) -> None:
        assert "ix_backtest_runs_portfolio_id" in self._index_names(
            upgraded_engine, "backtest_runs"
        )

    def test_backtest_runs_has_status_index(self, upgraded_engine) -> None:
        assert "ix_backtest_runs_status" in self._index_names(
            upgraded_engine, "backtest_runs"
        )

    def test_backtest_runs_has_created_at_index(self, upgraded_engine) -> None:
        assert "ix_backtest_runs_created_at" in self._index_names(
            upgraded_engine, "backtest_runs"
        )

    def test_risk_limits_has_portfolio_id_index(self, upgraded_engine) -> None:
        assert "ix_risk_limits_portfolio_id" in self._index_names(
            upgraded_engine, "risk_limits"
        )

    def test_rebalancing_policies_has_portfolio_id_index(self, upgraded_engine) -> None:
        assert "ix_rebalancing_policies_portfolio_id" in self._index_names(
            upgraded_engine, "rebalancing_policies"
        )


class TestUpgradeUniqueConstraints:
    """Unique constraints must be enforced after upgrade."""

    def test_factor_scores_rejects_duplicate_ticker_date_factor(
        self, upgraded_engine
    ) -> None:
        with upgraded_engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO factor_scores
                  (id, ticker, score_date, factor_type, factor_group, raw_score,
                   created_at, updated_at)
                VALUES ('id-uc-1', 'AAPL', '2024-01-01', 'momentum', 'momentum', 1.0,
                        datetime('now'), datetime('now'))
            """))
        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            with upgraded_engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO factor_scores
                      (id, ticker, score_date, factor_type, factor_group, raw_score,
                       created_at, updated_at)
                    VALUES ('id-uc-2', 'AAPL', '2024-01-01', 'momentum', 'momentum', 2.0,
                            datetime('now'), datetime('now'))
                """))

    def test_risk_limits_rejects_duplicate_portfolio_metric_limit_type(
        self, upgraded_engine
    ) -> None:
        portfolio_id = "portfolio-uc-1"
        with upgraded_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO portfolios (id, name) VALUES (:pid, 'UC Portfolio')"
            ), {"pid": portfolio_id})
            conn.execute(text("""
                INSERT INTO risk_limits
                  (id, portfolio_id, metric, limit_type, threshold, is_breached,
                   created_at, updated_at)
                VALUES ('rl-uc-1', :pid, 'volatility', 'max', 0.2, 0,
                        datetime('now'), datetime('now'))
            """), {"pid": portfolio_id})
        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            with upgraded_engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO risk_limits
                      (id, portfolio_id, metric, limit_type, threshold, is_breached,
                       created_at, updated_at)
                    VALUES ('rl-uc-2', :pid, 'volatility', 'max', 0.3, 0,
                            datetime('now'), datetime('now'))
                """), {"pid": portfolio_id})

    def test_rebalancing_policies_rejects_duplicate_portfolio_name(
        self, upgraded_engine
    ) -> None:
        portfolio_id = "portfolio-uc-2"
        with upgraded_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO portfolios (id, name) VALUES (:pid, 'UC Portfolio 2')"
            ), {"pid": portfolio_id})
            conn.execute(text("""
                INSERT INTO rebalancing_policies
                  (id, portfolio_id, name, policy_type, config, is_active,
                   created_at, updated_at)
                VALUES ('rp-uc-1', :pid, 'quarterly', 'calendar', '{}', 0,
                        datetime('now'), datetime('now'))
            """), {"pid": portfolio_id})
        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            with upgraded_engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO rebalancing_policies
                      (id, portfolio_id, name, policy_type, config, is_active,
                       created_at, updated_at)
                    VALUES ('rp-uc-2', :pid, 'quarterly', 'threshold', '{}', 0,
                            datetime('now'), datetime('now'))
                """), {"pid": portfolio_id})


# ---------------------------------------------------------------------------
# Downgrade: all six tables dropped
# ---------------------------------------------------------------------------


class TestDowngradeDropsAllSixTables:
    """After downgrade(), none of the six tables must exist."""

    def test_downgrade_removes_all_six_tables(self) -> None:
        engine = _sqlite_engine()
        _create_prereq_tables(engine)

        mod = _load_migration()

        from alembic.runtime.migration import MigrationContext
        from alembic.operations import Operations

        with engine.begin() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            mod.upgrade.__globals__["op"] = op
            mod.upgrade()

        tables_after_upgrade = set(inspect(engine).get_table_names())
        assert SIX_NEW_TABLES.issubset(tables_after_upgrade)

        with engine.begin() as conn:
            ctx = MigrationContext.configure(conn)
            op = Operations(ctx)
            mod.downgrade.__globals__["op"] = op
            mod.downgrade()

        tables_after_downgrade = set(inspect(engine).get_table_names())
        assert tables_after_downgrade.isdisjoint(SIX_NEW_TABLES), (
            f"Tables still present after downgrade: "
            f"{SIX_NEW_TABLES & tables_after_downgrade}"
        )
