"""Tests for the ``repair_etf_fund_snapshots`` Alembic revision.

The ingestion fixes only correct new writes, so this revision repairs the
snapshot rows already stored under earlier ``as_of`` dates: valuation
reciprocals become multiples, ``0.0`` / negative no-data values become NULL,
and all-NULL asset-class snapshots (which would shadow the last real
allocation as the newest ``as_of``) are deleted.

Replays ``upgrade()`` / ``downgrade()`` against an in-memory SQLite engine via
a raw ``MigrationContext`` + ``Operations`` (no Postgres needed), using the real
model tables so a column rename breaks the test rather than the migration.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from portopt_db.models.market_data.etf_metadata import (
    ETFAssetClass,
    ETFEquityHoldings,
)

MIGRATION_FILENAME = "3aa457e38da0_repair_etf_fund_snapshots.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "3aa457e38da0"
EXPECTED_DOWN_REVISION = "d8f0b2c4e6a1"

EQUITY = ETFEquityHoldings.__table__
ASSETS = ETFAssetClass.__table__

_VALUATION = (
    "price_to_earnings",
    "price_to_book",
    "price_to_sales",
    "price_to_cashflow",
)
_AS_OF = dt.date(2026, 9, 1)


def _load_migration() -> ModuleType:
    """Import the revision module directly from its file path."""
    spec = importlib.util.spec_from_file_location(
        "repair_etf_fund_snapshots_migration", MIGRATION_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def engine() -> Engine:
    """In-memory SQLite holding only the two tables the revision touches."""
    eng = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    EQUITY.create(eng)
    ASSETS.create(eng)
    return eng


def _equity_row(engine: Engine, **values: Any) -> uuid.UUID:
    instrument_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            sa.insert(EQUITY).values(
                instrument_id=instrument_id, as_of=_AS_OF, **values
            )
        )
    return instrument_id


def _asset_row(engine: Engine, **values: Any) -> uuid.UUID:
    instrument_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            sa.insert(ASSETS).values(
                instrument_id=instrument_id, as_of=_AS_OF, **values
            )
        )
    return instrument_id


def _equity(engine: Engine, instrument_id: uuid.UUID) -> dict[str, float | None]:
    with engine.connect() as conn:
        row = conn.execute(
            sa.select(EQUITY).where(EQUITY.c.instrument_id == instrument_id)
        ).one()
    return {
        col: None if row._mapping[col] is None else float(row._mapping[col])
        for col in _VALUATION
    }


def _asset_instruments(engine: Engine) -> set[uuid.UUID]:
    with engine.connect() as conn:
        return {r[0] for r in conn.execute(sa.select(ASSETS.c.instrument_id))}


def _run(engine: Engine, step: str) -> None:
    mod = _load_migration()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(mod, step)()


def test_revision_chains_from_the_nav_history_head() -> None:
    """The revision sits directly on the ``nav_history`` revision."""
    mod = _load_migration()

    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_inverts_stored_reciprocals_into_multiples(engine: Engine) -> None:
    """SPY as the pre-fix writer stored it: P/E 0.04132 is really 1/24.2."""
    spy = _equity_row(
        engine,
        price_to_earnings=0.04132,
        price_to_book=0.19034,
        price_to_sales=0.27459,
        price_to_cashflow=0.05326,
    )

    _run(engine, "upgrade")

    out = _equity(engine, spy)
    assert out["price_to_earnings"] == pytest.approx(24.2, abs=0.05)
    assert out["price_to_book"] == pytest.approx(5.254, abs=0.01)
    assert out["price_to_sales"] == pytest.approx(3.642, abs=0.01)
    assert out["price_to_cashflow"] == pytest.approx(18.78, abs=0.05)


def test_upgrade_nulls_zero_and_negative_no_data_values(engine: Engine) -> None:
    """A bond ETF carries Yahoo's 0.0 sentinel in every valuation column; a
    negative reciprocal (negative earnings) has no meaningful multiple."""
    agg = _equity_row(
        engine,
        price_to_earnings=0.0,
        price_to_book=0.0,
        price_to_sales=0.0,
        price_to_cashflow=0.0,
    )
    arkk = _equity_row(engine, price_to_earnings=-0.02, price_to_book=0.16675)

    _run(engine, "upgrade")

    assert all(v is None for v in list(_equity(engine, agg).values())[:4])
    mixed = _equity(engine, arkk)
    assert mixed["price_to_earnings"] is None
    assert mixed["price_to_book"] == pytest.approx(6.0, abs=0.01)


def test_upgrade_keeps_missing_valuation_values_missing(engine: Engine) -> None:
    """A metric Yahoo never reported stays NULL rather than becoming a multiple."""
    jaga = _equity_row(engine, price_to_earnings=0.05)

    _run(engine, "upgrade")

    out = _equity(engine, jaga)
    assert out["price_to_earnings"] == pytest.approx(20.0, abs=0.01)
    assert out["price_to_book"] is None


def test_upgrade_deletes_only_all_null_asset_class_snapshots(engine: Engine) -> None:
    """A genuine 0.0 allocation is data; only a row with every bucket NULL goes."""
    empty = _asset_row(engine)
    allocated = _asset_row(engine, stock_pct=0.6)
    all_bonds = _asset_row(engine, stock_pct=0.0, bond_pct=1.0)

    _run(engine, "upgrade")

    assert _asset_instruments(engine) == {allocated, all_bonds}
    assert empty not in _asset_instruments(engine)


def test_downgrade_restores_the_reciprocal_form_and_keeps_asset_classes(
    engine: Engine,
) -> None:
    """Downgrade re-inverts so the data matches what the pre-fix writer
    produced; rows nulled or deleted by upgrade are not resurrected."""
    spy = _equity_row(engine, price_to_earnings=0.04132, price_to_book=0.19034)
    kept = _asset_row(engine, stock_pct=0.6)

    _run(engine, "upgrade")
    _run(engine, "downgrade")

    out = _equity(engine, spy)
    assert out["price_to_earnings"] == pytest.approx(0.04132, abs=1e-5)
    assert out["price_to_book"] == pytest.approx(0.19034, abs=1e-5)
    assert _asset_instruments(engine) == {kept}
