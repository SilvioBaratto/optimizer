"""Purge Trading Economics rows that mix non-comparable measures under one key.

Trading Economics reuses one row label for different measures per country: Japan's
capacity utilisation is an index in points, the US one a rate in percent; France's
"Non Farm Payrolls" is an employment level, the US one a monthly change. The
scraper used to file them all under one canonical ``indicator_key``, so a rule such
as ``capacity_utilization > 80`` flagged Japan permanently. The scraper now skips
those rows, but ``trading_economics_indicators`` is keyed
``(country, indicator_key)`` and an upsert never deletes, so the stale rows would
otherwise live on, never refreshed.

This deletes them from the snapshot table, judged by its ``unit`` column, and the
accumulated history of the same ``(country, indicator_key)`` pairs from
``trading_economics_observations``, which keeps no unit of its own. History with no
snapshot row to judge it by is kept, as is any row without a unit. The scraper only
writes comparable rows, so nothing deleted here comes back.

The rules below are a frozen copy of the scraper's comparability rules at the time
of the fix, so the migration keeps replaying the same way when the scraper changes.

Revision ID: a5ac0fa9b0ec
Revises: d8f0b2c4e6a1
Create Date: 2026-10-24
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a5ac0fa9b0ec"
down_revision: str | Sequence[str] | None = "d8f0b2c4e6a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Lower-cased units a key's rows must be quoted in to share its canonical key.
_EXPECTED_UNITS: dict[str, tuple[str, ...]] = {
    "capacity_utilization": ("percent",),
    "leading_economic_index": ("points",),
    "factory_orders": ("percent",),
    "personal_savings": ("percent",),
    "employment_change": ("thousand",),
}

# Series defined for the US only; other country pages reuse the label for an
# unrelated measure in the same unit.
_US_ONLY_KEYS: tuple[str, ...] = (
    "non_farm_payrolls",
    "initial_jobless_claims",
    "continuing_jobless_claims",
)
_US = "USA"

# Lightweight table constructs keep the migration independent of the ORM models,
# which keep evolving while this revision must replay unchanged.
_indicators = sa.table(
    "trading_economics_indicators",
    sa.column("country", sa.String),
    sa.column("indicator_key", sa.String),
    sa.column("unit", sa.String),
)
_observations = sa.table(
    "trading_economics_observations",
    sa.column("country", sa.String),
    sa.column("indicator_key", sa.String),
)


def _foreign_unit(table: sa.TableClause) -> sa.ColumnElement[bool]:
    """Condition: the row is quoted in a unit its key does not allow.

    A blank unit never matches, so a row that cannot be checked is kept.
    """
    return sa.or_(
        *(
            sa.and_(
                table.c.indicator_key == key,
                sa.func.trim(sa.func.coalesce(table.c.unit, "")) != "",
                sa.func.lower(sa.func.trim(table.c.unit)).not_in(units),
            )
            for key, units in _EXPECTED_UNITS.items()
        )
    )


def _us_only_series_elsewhere(table: sa.TableClause) -> sa.ColumnElement[bool]:
    """Condition: the row files a US-only key under another country."""
    return sa.and_(
        table.c.indicator_key.in_(_US_ONLY_KEYS),
        table.c.country != _US,
    )


def upgrade() -> None:
    """Delete non-comparable snapshot rows and the history filed under them."""
    judged_by_snapshot = sa.exists().where(
        _indicators.c.country == _observations.c.country,
        _indicators.c.indicator_key == _observations.c.indicator_key,
        _foreign_unit(_indicators),
    )
    # History first: its verdict is read off the snapshot rows deleted next.
    op.execute(
        sa.delete(_observations).where(
            sa.or_(judged_by_snapshot, _us_only_series_elsewhere(_observations))
        )
    )
    op.execute(
        sa.delete(_indicators).where(
            sa.or_(_foreign_unit(_indicators), _us_only_series_elsewhere(_indicators))
        )
    )


def downgrade() -> None:
    """Not reversible.

    The deleted rows held a different measure than their key names, so restoring
    them would put the corruption back. Restore from a pre-upgrade dump if the
    rows are needed.
    """
    raise NotImplementedError(
        "a5ac0fa9b0ec is one-way: the deleted Trading Economics rows mixed "
        "non-comparable measures and are not restorable. Recover from a dump if "
        "they are needed."
    )
