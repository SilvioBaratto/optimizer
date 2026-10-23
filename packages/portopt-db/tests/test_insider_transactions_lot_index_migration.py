"""``insider_transactions.lot_index``: model key + Alembic revision.

Yahoo can report several real insider transactions that render identically
(same insider, day and text) but differ in share count. ``lot_index`` is part of
the natural key so those lots coexist instead of collapsing to one row. The
revision is replayed against in-memory SQLite through a raw ``MigrationContext``
+ ``Operations`` (no Postgres needed, so it runs in CI).
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import uuid
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from portopt_db.models.market_data.yfinance_data import InsiderTransaction
from portopt_db.models.universe.universe import Exchange, Instrument

MIGRATION_FILENAME = "b4d6f8a0c2e5_add_lot_index_to_insider_transactions.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "b4d6f8a0c2e5"
EXPECTED_DOWN_REVISION = "a3c5e7f9b1d4"

TABLE = "insider_transactions"
CONSTRAINT = "uq_insider_tx_row"
KEY_BEFORE = ["instrument_id", "insider_name", "start_date", "transaction_type"]
KEY_AFTER = [*KEY_BEFORE, "lot_index"]


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "insider_lot_index_migration", MIGRATION_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _engine_with_pre_migration_table():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    meta = sa.MetaData()
    sa.Table(
        TABLE,
        meta,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("instrument_id", sa.String(36), nullable=False),
        sa.Column("insider_name", sa.String(500), nullable=False),
        sa.Column("transaction_type", sa.String(200), nullable=False),
        sa.Column("shares", sa.BigInteger()),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.UniqueConstraint(*KEY_BEFORE, name=CONSTRAINT),
    )
    meta.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO insider_transactions "
                "(id, instrument_id, insider_name, transaction_type, shares, "
                "start_date) VALUES "
                "('r1', 'i1', 'Alice', 'Grant', 5000, '2024-01-10')"
            )
        )
    return engine


def _run(engine, fn_name: str) -> None:
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    mod = _load_migration()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(mod, fn_name)()


def _unique_columns(engine) -> list[str]:
    uniques = {
        u["name"]: u["column_names"]
        for u in inspect(engine).get_unique_constraints(TABLE)
    }
    return uniques[CONSTRAINT]


def test_revision_chains_from_earnings_surprise_rescale() -> None:
    mod = _load_migration()
    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_adds_lot_index_and_widens_the_natural_key() -> None:
    engine = _engine_with_pre_migration_table()
    _run(engine, "upgrade")

    cols = {c["name"]: c for c in inspect(engine).get_columns(TABLE)}
    assert cols["lot_index"]["nullable"] is False
    assert _unique_columns(engine) == KEY_AFTER


def test_upgrade_keeps_existing_rows_at_lot_zero() -> None:
    engine = _engine_with_pre_migration_table()
    _run(engine, "upgrade")

    with engine.connect() as conn:
        rows = conn.execute(
            sa.text("SELECT shares, lot_index FROM insider_transactions")
        )
        assert rows.all() == [(5000, 0)]


def test_downgrade_restores_the_original_key() -> None:
    engine = _engine_with_pre_migration_table()
    _run(engine, "upgrade")
    _run(engine, "downgrade")

    cols = {c["name"] for c in inspect(engine).get_columns(TABLE)}
    assert "lot_index" not in cols
    assert _unique_columns(engine) == KEY_BEFORE


def test_downgrade_collapses_extra_lots_so_the_old_key_can_be_restored() -> None:
    engine = _engine_with_pre_migration_table()
    _run(engine, "upgrade")
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO insider_transactions "
                "(id, instrument_id, insider_name, transaction_type, shares, "
                "start_date, lot_index) VALUES "
                "('r2', 'i1', 'Alice', 'Grant', 12000, '2024-01-10', 1)"
            )
        )

    _run(engine, "downgrade")

    with engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT shares FROM insider_transactions"))
        assert rows.all() == [(5000,)]


def test_model_key_includes_lot_index() -> None:
    constraint = next(
        c for c in InsiderTransaction.__table__.constraints if c.name == CONSTRAINT
    )
    assert [col.name for col in constraint.columns] == KEY_AFTER


def test_same_key_rows_with_distinct_lot_index_coexist(db_session) -> None:
    ex = Exchange(name="NASDAQ")
    db_session.add(ex)
    db_session.flush()
    inst = Instrument(
        ticker="MSFT",
        short_name="MSFT",
        exchange_id=ex.id,
        instrument_type="EQUITY",
        asset_class="equity",
    )
    db_session.add(inst)
    db_session.flush()

    def lot(shares: int, lot_index: int) -> InsiderTransaction:
        return InsiderTransaction(
            id=uuid.uuid4(),
            instrument_id=inst.id,
            insider_name="COLEMAN AMY",
            transaction_type="Stock Award(Grant) at price 0.00 per share.",
            start_date=dt.date(2025, 9, 15),
            shares=shares,
            lot_index=lot_index,
        )

    db_session.add_all([lot(4022, 0), lot(4358, 1)])
    db_session.flush()
    assert db_session.query(InsiderTransaction).count() == 2

    db_session.add(lot(999, 1))
    with pytest.raises(IntegrityError):
        db_session.flush()
