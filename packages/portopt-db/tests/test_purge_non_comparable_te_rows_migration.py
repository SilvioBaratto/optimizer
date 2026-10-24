"""Tests for the ``purge_non_comparable_te_rows`` Alembic revision.

Before the scraper learned to keep non-comparable measures off shared keys, rows
such as Japan's capacity-utilisation index (103 points) or France's employment
level filed as ``non_farm_payrolls`` were upserted next to the US series. The
snapshot table is keyed ``(country, indicator_key)`` and an upsert never deletes,
so the scraper fix alone leaves them behind. The revision deletes them from the
snapshot table (judged by its ``unit`` column) and, for the same
``(country, indicator_key)`` pairs, from the unit-less observation history.

Replays ``upgrade()`` / ``downgrade()`` against an in-memory SQLite engine via a
raw ``MigrationContext`` + ``Operations`` (no Postgres needed, so it runs in CI).
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from portopt_db.models.macro.macro_regime import (
    TradingEconomicsIndicator,
    TradingEconomicsObservation,
)

MIGRATION_FILENAME = "a5ac0fa9b0ec_purge_non_comparable_te_rows.py"
MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_PATH = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())

EXPECTED_REVISION = "a5ac0fa9b0ec"
EXPECTED_DOWN_REVISION = "d8f0b2c4e6a1"

_DAY = dt.date(2026, 10, 1)


def _load_migration() -> ModuleType:
    """Import the revision module directly from its file path."""
    spec = importlib.util.spec_from_file_location(
        "purge_non_comparable_te_rows_migration", MIGRATION_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(engine, step: str) -> None:
    mod = _load_migration()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(mod, step)()


@pytest.fixture
def engine():
    """In-memory SQLite engine holding just the two Trading Economics tables."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TradingEconomicsIndicator.__table__.create(eng)
    TradingEconomicsObservation.__table__.create(eng)
    return eng


def _snapshot(country: str, key: str, unit: str | None) -> TradingEconomicsIndicator:
    return TradingEconomicsIndicator(
        country=country, indicator_key=key, value=1.0, previous=0.5, unit=unit
    )


def _observation(country: str, key: str) -> TradingEconomicsObservation:
    return TradingEconomicsObservation(
        country=country, indicator_key=key, date=_DAY, value=1.0
    )


def _snapshot_pairs(engine) -> set[tuple[str, str]]:
    with Session(engine) as session:
        rows = session.execute(
            select(
                TradingEconomicsIndicator.country,
                TradingEconomicsIndicator.indicator_key,
            )
        ).all()
    return {(country, key) for country, key in rows}


def _observation_pairs(engine) -> set[tuple[str, str]]:
    with Session(engine) as session:
        rows = session.execute(
            select(
                TradingEconomicsObservation.country,
                TradingEconomicsObservation.indicator_key,
            )
        ).all()
    return {(country, key) for country, key in rows}


def _seed(engine, *rows) -> None:
    with Session(engine) as session:
        session.add_all(rows)
        session.commit()


def test_module_is_importable() -> None:
    """The revision file loads as a module."""
    assert _load_migration() is not None


def test_revision_chains_from_the_nav_history_head() -> None:
    """The revision hangs off the nav_history head without opening a branch."""
    mod = _load_migration()

    assert mod.revision == EXPECTED_REVISION
    assert mod.down_revision == EXPECTED_DOWN_REVISION
    assert mod.branch_labels is None


class TestSnapshotTable:
    """Latest-snapshot rows are judged by their stored unit and country."""

    def test_row_in_a_foreign_unit_is_deleted_and_the_comparable_row_kept(
        self, engine
    ) -> None:
        """A different measure under a shared key is purged; the US rate stays."""
        _seed(
            engine,
            _snapshot("Japan", "capacity_utilization", "points"),
            _snapshot("USA", "capacity_utilization", "percent"),
        )

        _run(engine, "upgrade")

        assert _snapshot_pairs(engine) == {("USA", "capacity_utilization")}

    @pytest.mark.parametrize(
        ("key", "foreign_unit"),
        [
            ("capacity_utilization", "points"),
            ("leading_economic_index", "percent"),
            ("factory_orders", "Net Balance"),
            ("personal_savings", "percent of GDP"),
            ("employment_change", "percent"),
        ],
    )
    def test_every_unit_checked_key_loses_its_foreign_unit_row(
        self, engine, key: str, foreign_unit: str
    ) -> None:
        """Each unit-checked key purges a row quoted in a different unit."""
        _seed(engine, _snapshot("Brazil", key, foreign_unit))

        _run(engine, "upgrade")

        assert _snapshot_pairs(engine) == set()

    @pytest.mark.parametrize(
        ("key", "unit"),
        [
            ("capacity_utilization", "Percent"),
            ("capacity_utilization", " percent "),
            ("leading_economic_index", "Points"),
            ("factory_orders", "percent"),
            ("personal_savings", "percent"),
            ("employment_change", "Thousand"),
        ],
    )
    def test_expected_unit_is_kept_whatever_its_case_or_padding(
        self, engine, key: str, unit: str
    ) -> None:
        """Unit comparison ignores case and surrounding whitespace."""
        _seed(engine, _snapshot("Germany", key, unit))

        _run(engine, "upgrade")

        assert _snapshot_pairs(engine) == {("Germany", key)}

    @pytest.mark.parametrize("unit", [None, "", "   "])
    def test_row_without_a_unit_cannot_be_judged_and_is_kept(
        self, engine, unit: str | None
    ) -> None:
        """A missing unit gives no evidence of a mismatch, so the row stays."""
        _seed(engine, _snapshot("France", "capacity_utilization", unit))

        _run(engine, "upgrade")

        assert _snapshot_pairs(engine) == {("France", "capacity_utilization")}

    @pytest.mark.parametrize(
        "key",
        ["non_farm_payrolls", "initial_jobless_claims", "continuing_jobless_claims"],
    )
    def test_us_only_series_survive_for_the_us_alone(self, engine, key: str) -> None:
        """US-only labour series are purged from every other country."""
        _seed(
            engine,
            _snapshot("USA", key, "Thousand"),
            _snapshot("France", key, "Thousand"),
            _snapshot("Germany", key, "Thousand"),
        )

        _run(engine, "upgrade")

        assert _snapshot_pairs(engine) == {("USA", key)}

    def test_keys_without_a_comparability_rule_are_untouched(self, engine) -> None:
        """Keys outside the comparability rules are never deleted."""
        _seed(
            engine,
            _snapshot("Japan", "gdp_growth_rate", "percent"),
            _snapshot("Japan", "trade_balance", "JPY Billion"),
            _snapshot("France", "inflation_rate", "points"),
        )

        _run(engine, "upgrade")

        assert _snapshot_pairs(engine) == {
            ("Japan", "gdp_growth_rate"),
            ("Japan", "trade_balance"),
            ("France", "inflation_rate"),
        }


class TestObservationTable:
    """Observation history has no unit, so it follows the snapshot verdict."""

    def test_history_of_a_purged_snapshot_pair_is_deleted_with_it(self, engine) -> None:
        """History of a purged snapshot pair goes with it."""
        _seed(
            engine,
            _snapshot("Japan", "capacity_utilization", "points"),
            _snapshot("USA", "capacity_utilization", "percent"),
            _observation("Japan", "capacity_utilization"),
            _observation("USA", "capacity_utilization"),
        )

        _run(engine, "upgrade")

        assert _observation_pairs(engine) == {("USA", "capacity_utilization")}

    def test_us_only_history_of_other_countries_is_deleted_without_a_snapshot(
        self, engine
    ) -> None:
        """US-only keys are purged from other countries' history by country alone."""
        _seed(
            engine,
            _observation("France", "non_farm_payrolls"),
            _observation("USA", "non_farm_payrolls"),
            _observation("Germany", "initial_jobless_claims"),
        )

        _run(engine, "upgrade")

        assert _observation_pairs(engine) == {("USA", "non_farm_payrolls")}

    def test_history_with_no_snapshot_to_judge_by_is_kept(self, engine) -> None:
        """Unit-checked history with no snapshot row cannot be judged and stays."""
        _seed(engine, _observation("Japan", "capacity_utilization"))

        _run(engine, "upgrade")

        assert _observation_pairs(engine) == {("Japan", "capacity_utilization")}

    def test_history_of_a_comparable_snapshot_pair_is_kept(self, engine) -> None:
        """History of a snapshot pair that passes the unit check stays."""
        _seed(
            engine,
            _snapshot("Germany", "factory_orders", "percent"),
            _observation("Germany", "factory_orders"),
        )

        _run(engine, "upgrade")

        assert _observation_pairs(engine) == {("Germany", "factory_orders")}

    def test_history_of_other_keys_is_untouched(self, engine) -> None:
        """History of a key outside the rules is untouched."""
        _seed(
            engine,
            _snapshot("Japan", "capacity_utilization", "points"),
            _observation("Japan", "gdp_growth_rate"),
        )

        _run(engine, "upgrade")

        assert _observation_pairs(engine) == {("Japan", "gdp_growth_rate")}


def test_running_the_cleanup_twice_changes_nothing_more(engine) -> None:
    """A replayed upgrade finds nothing left to delete."""
    _seed(
        engine,
        _snapshot("Japan", "capacity_utilization", "points"),
        _snapshot("USA", "capacity_utilization", "percent"),
        _observation("Japan", "capacity_utilization"),
        _observation("USA", "capacity_utilization"),
    )

    _run(engine, "upgrade")
    _run(engine, "upgrade")

    assert _snapshot_pairs(engine) == {("USA", "capacity_utilization")}
    assert _observation_pairs(engine) == {("USA", "capacity_utilization")}


def test_downgrade_is_declared_irreversible(engine) -> None:
    """Deleted rows cannot be restored, so downgrade refuses."""
    with pytest.raises(NotImplementedError):
        _run(engine, "downgrade")
