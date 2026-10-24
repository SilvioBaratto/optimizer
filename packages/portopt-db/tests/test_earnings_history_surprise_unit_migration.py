"""Tests for the ``rescale_earnings_history_surprise_percent`` Alembic revision.

``earnings_history.surprise_percent`` used to hold Yahoo's raw fraction while
``earnings_dates`` held percent points; the revision rescales the stored rows so
both tables share one unit. Replayed against in-memory SQLite via a raw
``MigrationContext`` + ``Operations`` (no Postgres needed, so it runs in CI).
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "a3c5e7f9b1d4_rescale_earnings_history_surprise_percent.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "a3c5e7f9b1d4"
EXPECTED_DOWN_REVISION = "e9a1c3d5f7b2"

FRACTIONS = [0.1181, -0.05, None, 0.0]


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "earnings_history_surprise_unit_migration", MIGRATION_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _engine_with_fraction_rows():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    meta = sa.MetaData()
    table = sa.Table(
        "earnings_history",
        meta,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("surprise_percent", sa.Numeric(20, 6)),
    )
    meta.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            table.insert(),
            [{"id": str(uuid.uuid4()), "surprise_percent": v} for v in FRACTIONS],
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


def _stored(engine) -> list[float | None]:
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text("SELECT surprise_percent FROM earnings_history")
        ).all()
    values = [None if r[0] is None else float(r[0]) for r in rows]
    return sorted(values, key=lambda v: (v is None, v))


def test_revision_chains_from_asset_class_positions() -> None:
    mod = _load_migration()
    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_rescales_fractions_to_percent_points_and_keeps_nulls() -> None:
    engine = _engine_with_fraction_rows()
    _run(engine, "upgrade")

    assert _stored(engine) == [
        pytest.approx(-5.0),
        pytest.approx(0.0),
        pytest.approx(11.81),
        None,
    ]


def test_downgrade_restores_the_fractions() -> None:
    engine = _engine_with_fraction_rows()
    _run(engine, "upgrade")
    _run(engine, "downgrade")

    assert _stored(engine) == [
        pytest.approx(-0.05),
        pytest.approx(0.0),
        pytest.approx(0.1181),
        None,
    ]
