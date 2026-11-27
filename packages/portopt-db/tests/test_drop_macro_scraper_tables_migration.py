"""Revision 8ed6504aee6a, which drops the six tables the removed macro scrapers filled.

The revision is replayed by path against in-memory SQLite, so it runs in CI without
PostgreSQL. The six tables have no model any more, so the schema of the previous head
is described once in ``_TABLE_SPECS`` and used both to build the tables the upgrade
must drop and to check the tables the downgrade must bring back.
"""

from __future__ import annotations

import datetime
import importlib.util
import uuid
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, Table, create_engine, func, inspect, select, text
from sqlalchemy.pool import StaticPool

from portopt_db.models.macro.macro_regime import (
    FredObservation,
    MacroNews,
    MacroNewsTheme,
)

MIGRATION_DIR = Path(__file__).parent / ".." / "alembic" / "versions"
MIGRATION_FILENAME = "2026_11_27_1913-8ed6504aee6a_drop_macro_scraper_tables.py"

EXPECTED_REVISION = "8ed6504aee6a"
EXPECTED_DOWN_REVISION = "2266e24c509f"

_FRED_SERIES_ID = "BAMLH0A0HYM2"
_FRED_VALUE = 3.25
_FRED_DATE = datetime.date(2026, 11, 26)
_NEWS_ID = "news-1"
_NEWS_THEME = "inflation"


class _Column(NamedTuple):
    """A column as SQLite reports it: its declared type text and nullability."""

    name: str
    sql_type: str
    nullable: bool
    has_server_default: bool = False


class _TableSpec(NamedTuple):
    columns: tuple[_Column, ...]
    unique_constraints: dict[str, tuple[str, ...]]
    indexes: dict[str, tuple[str, ...]]


def _float_columns(*names: str) -> tuple[_Column, ...]:
    return tuple(_Column(name, "FLOAT", True) for name in names)


_ID = _Column("id", "CHAR(32)", False)
_COUNTRY = _Column("country", "VARCHAR(100)", False)
_INDICATOR_KEY = _Column("indicator_key", "VARCHAR(100)", False)
_MATURITY = _Column("maturity", "VARCHAR(10)", False)
_DATE = _Column("date", "DATE", False)
_REFERENCE_DATE = _Column("reference_date", "DATE", True)
_TIMESTAMPS = (
    _Column("created_at", "DATETIME", False, has_server_default=True),
    _Column("updated_at", "DATETIME", False, has_server_default=True),
)
_FORECAST_COLUMNS = (
    *_float_columns(
        "last_inflation",
        "inflation_6m",
        "inflation_10y_avg",
        "gdp_growth_6m",
        "earnings_12m",
        "eps_expected_12m",
        "peg_ratio",
        "lt_rate_forecast",
    ),
    _REFERENCE_DATE,
)

# The six tables as the previous head leaves them, in the order the downgrade
# recreates them (the reverse of the order the upgrade drops them).
_TABLE_SPECS: dict[str, _TableSpec] = {
    "economic_indicators": _TableSpec(
        columns=(_ID, _COUNTRY, *_FORECAST_COLUMNS, *_TIMESTAMPS),
        unique_constraints={"uq_economic_indicator_country": ("country",)},
        indexes={},
    ),
    "economic_indicator_observations": _TableSpec(
        columns=(_ID, _COUNTRY, _DATE, *_FORECAST_COLUMNS, *_TIMESTAMPS),
        unique_constraints={"uq_econ_obs_country_date": ("country", "date")},
        indexes={
            "ix_econ_observations_country": ("country",),
            "ix_econ_observations_date": ("date",),
        },
    ),
    "trading_economics_indicators": _TableSpec(
        columns=(
            _ID,
            _COUNTRY,
            _INDICATOR_KEY,
            *_float_columns("value", "previous"),
            _Column("unit", "VARCHAR(50)", True),
            _Column("reference", "VARCHAR(100)", True),
            _Column("raw_name", "VARCHAR(200)", True),
            *_TIMESTAMPS,
        ),
        unique_constraints={
            "uq_te_indicator_country_key": ("country", "indicator_key")
        },
        indexes={"ix_trading_economics_indicators_country": ("country",)},
    ),
    "trading_economics_observations": _TableSpec(
        columns=(
            _ID,
            _COUNTRY,
            _INDICATOR_KEY,
            _DATE,
            *_float_columns("value"),
            *_TIMESTAMPS,
        ),
        unique_constraints={
            "uq_te_obs_country_key_date": ("country", "indicator_key", "date")
        },
        indexes={
            "ix_te_observations_country": ("country",),
            "ix_te_observations_date": ("date",),
        },
    ),
    "bond_yields": _TableSpec(
        columns=(
            _ID,
            _COUNTRY,
            _MATURITY,
            *_float_columns("yield_value", "day_change", "month_change", "year_change"),
            _REFERENCE_DATE,
            *_TIMESTAMPS,
        ),
        unique_constraints={"uq_bond_yield_country_maturity": ("country", "maturity")},
        indexes={"ix_bond_yields_country": ("country",)},
    ),
    "bond_yield_observations": _TableSpec(
        columns=(
            _ID,
            _COUNTRY,
            _MATURITY,
            _DATE,
            *_float_columns("yield_value"),
            *_TIMESTAMPS,
        ),
        unique_constraints={
            "uq_bond_obs_country_mat_date": ("country", "maturity", "date")
        },
        indexes={
            "ix_bond_observations_country": ("country",),
            "ix_bond_observations_date": ("date",),
        },
    ),
}

_DROPPED_TABLES = tuple(_TABLE_SPECS)

_FRED_TABLE: Table = FredObservation.__table__
_NEWS_TABLE: Table = MacroNews.__table__
_NEWS_THEME_TABLE: Table = MacroNewsTheme.__table__


def _column_ddl(column: _Column) -> str:
    not_null = "" if column.nullable else " NOT NULL"
    default = " DEFAULT CURRENT_TIMESTAMP" if column.has_server_default else ""
    return f"{column.name} {column.sql_type}{not_null}{default}"


def _create_previous_head_table(engine: Engine, table: str, spec: _TableSpec) -> None:
    definitions = [_column_ddl(column) for column in spec.columns]
    definitions.append("PRIMARY KEY (id)")
    definitions.extend(
        f"CONSTRAINT {name} UNIQUE ({', '.join(columns)})"
        for name, columns in spec.unique_constraints.items()
    )
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE {table} ({', '.join(definitions)})"))
        for index_name, columns in spec.indexes.items():
            conn.execute(
                text(f"CREATE INDEX {index_name} ON {table} ({', '.join(columns)})")
            )


def _load() -> ModuleType:
    path = str((MIGRATION_DIR / MIGRATION_FILENAME).resolve())
    spec = importlib.util.spec_from_file_location(Path(MIGRATION_FILENAME).stem, path)
    assert spec and spec.loader
    revision_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision_module)
    return revision_module


def _run(engine: Engine, migration_step: Callable[[], None]) -> None:
    with engine.begin() as conn:
        migration_context = MigrationContext.configure(conn)
        with Operations.context(migration_context):
            migration_step()


def _table_names(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def _row_count(engine: Engine, table: Table) -> int:
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar_one()


@pytest.fixture
def engine() -> Engine:
    """SQLite shaped like the previous head, with one row in each kept macro table."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table, spec in _TABLE_SPECS.items():
        _create_previous_head_table(engine, table, spec)
    for kept_table in (_FRED_TABLE, _NEWS_TABLE, _NEWS_THEME_TABLE):
        kept_table.create(engine)
    news_row_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            _FRED_TABLE.insert().values(
                id=uuid.uuid4(),
                series_id=_FRED_SERIES_ID,
                date=_FRED_DATE,
                value=_FRED_VALUE,
            )
        )
        conn.execute(
            _NEWS_TABLE.insert().values(id=news_row_id, news_id=_NEWS_ID, title="CPI")
        )
        conn.execute(
            _NEWS_THEME_TABLE.insert().values(
                id=uuid.uuid4(), news_id=news_row_id, theme=_NEWS_THEME
            )
        )
    return engine


@pytest.mark.criterion("REQ-DB-010")
def test_revision_follows_the_trading_212_drop() -> None:
    """Validates revision 8ed6504aee6a: it extends the previous Alembic head."""
    revision_module = _load()
    assert revision_module.revision == EXPECTED_REVISION
    assert revision_module.down_revision == EXPECTED_DOWN_REVISION


@pytest.mark.criterion("REQ-DB-010")
def test_upgrade_drops_the_six_tables_and_keeps_the_fred_and_news_tables(
    engine: Engine,
) -> None:
    """Validates the upgrade() of revision 8ed6504aee6a: only the six tables go."""
    tables_before = _table_names(engine)
    assert set(_DROPPED_TABLES) <= tables_before

    _run(engine, _load().upgrade)

    assert _table_names(engine) == tables_before - set(_DROPPED_TABLES)
    assert _row_count(engine, _FRED_TABLE) == 1
    assert _row_count(engine, _NEWS_TABLE) == 1
    assert _row_count(engine, _NEWS_THEME_TABLE) == 1
    with engine.connect() as conn:
        fred_row = conn.execute(select(_FRED_TABLE.c.series_id, _FRED_TABLE.c.value))
        assert tuple(fred_row.one()) == (_FRED_SERIES_ID, _FRED_VALUE)


@pytest.mark.criterion("REQ-DB-010")
def test_downgrade_recreates_the_six_tables_empty_with_their_indexes(
    engine: Engine,
) -> None:
    """Validates the downgrade() of revision 8ed6504aee6a: the six tables return
    empty, each with its columns, primary key, unique constraints and indexes."""
    revision_module = _load()
    _run(engine, revision_module.upgrade)
    assert not set(_DROPPED_TABLES) & _table_names(engine)

    _run(engine, revision_module.downgrade)

    inspector = inspect(engine)
    for table, spec in _TABLE_SPECS.items():
        columns = inspector.get_columns(table)
        assert [
            (info["name"], str(info["type"]), info["nullable"]) for info in columns
        ] == [(c.name, c.sql_type, c.nullable) for c in spec.columns], table
        assert {info["name"] for info in columns if info["default"] is not None} == {
            c.name for c in spec.columns if c.has_server_default
        }, table
        assert inspector.get_pk_constraint(table)["constrained_columns"] == ["id"]
        assert {
            info["name"]: tuple(info["column_names"])
            for info in inspector.get_unique_constraints(table)
        } == spec.unique_constraints, table
        assert {
            info["name"]: tuple(info["column_names"])
            for info in inspector.get_indexes(table)
        } == spec.indexes, table
        with engine.connect() as conn:
            row_count = conn.execute(text(f"SELECT count(*) FROM {table}"))
            assert row_count.scalar_one() == 0, table
