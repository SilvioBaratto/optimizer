"""Tests for migration c8d0bb63d826: correct the seeded VGK ISIN.

``a7b8c9d0e1f2`` seeded VGK with US9220427671, which embeds another security's
CUSIP. Databases that already ran that seed keep the wrong code, so a forward
revision rewrites it. The revision is replayed here against in-memory SQLite via
a raw ``MigrationContext`` + ``Operations`` (the UPDATE is dialect-neutral).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

MIGRATION_FILENAME = "c8d0bb63d826_correct_vgk_isin.py"
MIGRATION_PATH = str(
    (
        Path(__file__).parent / ".." / "alembic" / "versions" / MIGRATION_FILENAME
    ).resolve()
)

EXPECTED_REVISION = "c8d0bb63d826"
EXPECTED_DOWN_REVISION = "d8f0b2c4e6a1"

WRONG_ISIN = "US9220427671"
RIGHT_ISIN = "US9220428745"


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "correct_vgk_isin_migration", MIGRATION_PATH
    )
    assert spec and spec.loader, f"Cannot find migration at {MIGRATION_PATH}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def engine() -> Engine:
    """In-memory SQLite engine holding a minimal ``instruments`` table."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE instruments ("
                "id TEXT PRIMARY KEY, ticker TEXT NOT NULL, isin TEXT, "
                "exchange_id TEXT NOT NULL)"
            )
        )
    return eng


def _insert(engine: Engine, row_id: str, ticker: str, isin: str | None) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO instruments (id, ticker, isin, exchange_id) "
                "VALUES (:id, :ticker, :isin, 'nyse')"
            ),
            {"id": row_id, "ticker": ticker, "isin": isin},
        )


def _isin_by_id(engine: Engine) -> dict[str, str | None]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, isin FROM instruments")).all()
    return {row[0]: row[1] for row in rows}


def _run(engine: Engine, step: str) -> None:
    mod = _load_migration()
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            getattr(mod, step)()


def test_revision_chains_from_the_current_head() -> None:
    """The revision sits directly on the previous Alembic head."""
    mod = _load_migration()
    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION


def test_upgrade_rewrites_the_wrong_vgk_isin(engine: Engine) -> None:
    """A VGK row seeded with the wrong code is rewritten to the right one."""
    _insert(engine, "vgk", "VGK", WRONG_ISIN)

    _run(engine, "upgrade")

    assert _isin_by_id(engine)["vgk"] == RIGHT_ISIN


def test_upgrade_leaves_a_vgk_row_with_another_isin_alone(engine: Engine) -> None:
    """Only the exact seeded typo is rewritten; NULL and other codes survive."""
    _insert(engine, "vgk-null", "VGK", None)
    _insert(engine, "vgk-other", "VGK", "US0000000000")

    _run(engine, "upgrade")

    isins = _isin_by_id(engine)
    assert isins["vgk-null"] is None
    assert isins["vgk-other"] == "US0000000000"


def test_upgrade_does_not_touch_other_tickers(engine: Engine) -> None:
    """The rewrite is scoped to VGK, even for rows carrying the same wrong code."""
    _insert(engine, "vwo", "VWO", "US9220428588")
    _insert(engine, "stray", "XYZ", WRONG_ISIN)

    _run(engine, "upgrade")

    isins = _isin_by_id(engine)
    assert isins["vwo"] == "US9220428588"
    assert isins["stray"] == WRONG_ISIN


def test_upgrade_is_idempotent(engine: Engine) -> None:
    """Replaying the upgrade on an already corrected row changes nothing."""
    _insert(engine, "vgk", "VGK", WRONG_ISIN)

    _run(engine, "upgrade")
    _run(engine, "upgrade")

    assert _isin_by_id(engine)["vgk"] == RIGHT_ISIN


def test_downgrade_never_restores_the_wrong_isin(engine: Engine) -> None:
    """Stepping down must not reintroduce the known-bad code."""
    _insert(engine, "vgk", "VGK", WRONG_ISIN)

    _run(engine, "upgrade")
    _run(engine, "downgrade")

    assert _isin_by_id(engine)["vgk"] == RIGHT_ISIN
