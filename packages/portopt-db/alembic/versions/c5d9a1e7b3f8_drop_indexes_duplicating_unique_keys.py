"""Drop plain indexes that duplicate a UNIQUE key on the same columns.

A UNIQUE constraint is already enforced by its own b-tree index, so a plain index
over the same columns in the same order is a byte-identical copy: every
INSERT/UPSERT maintains both and the storage is doubled, with no query able to
prefer the copy. Nine tables carried such a pair (the observation snapshots, the
per-instrument one-row tables and ``market_journal``).

An index is only dropped when a unique constraint or full unique index still
covers its columns, so a legacy database whose unique key is missing keeps its
only lookup path. Idempotent for tables or indexes that are already absent.

Revision ID: c5d9a1e7b3f8
Revises: e3a5c7b9d1f4
Create Date: 2026-10-24
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c5d9a1e7b3f8"
down_revision: str | Sequence[str] | None = "e3a5c7b9d1f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_REDUNDANT: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "trading_economics_observations",
        "ix_te_obs_country_key_date",
        ("country", "indicator_key", "date"),
    ),
    (
        "bond_yield_observations",
        "ix_bond_obs_country_maturity_date",
        ("country", "maturity", "date"),
    ),
    ("economic_indicators", "ix_economic_indicators_country", ("country",)),
    (
        "analyst_price_targets",
        "ix_analyst_price_targets_instrument_id",
        ("instrument_id",),
    ),
    ("market_journal", "ix_market_journal_as_of", ("as_of",)),
    ("etf_metadata", "ix_etf_metadata_instrument_id", ("instrument_id",)),
    ("major_holders", "ix_major_holders_instrument_id", ("instrument_id",)),
    (
        "insider_purchases",
        "ix_insider_purchases_instrument_id",
        ("instrument_id",),
    ),
    (
        "ticker_profile_extras",
        "ix_ticker_profile_extras_instrument_id",
        ("instrument_id",),
    ),
)


def _enforced_unique_columns(
    inspector: sa.Inspector, table: str
) -> set[tuple[str | None, ...]]:
    """Ordered column tuples the table already enforces as unique for every row."""
    unique = {
        tuple(uc["column_names"]) for uc in inspector.get_unique_constraints(table)
    }
    for index in inspector.get_indexes(table):
        options = index.get("dialect_options") or {}
        partial = options.get("postgresql_where") or options.get("sqlite_where")
        if index.get("unique") and not partial:
            unique.add(tuple(index["column_names"]))
    return unique


def upgrade() -> None:
    """Drop each redundant plain index whose columns a unique key still covers."""
    if op.get_context().as_sql:
        # No connection to inspect: the rendered script targets a database built
        # by this chain, where every listed unique key exists, so only the
        # index's presence is uncertain.
        for table, index_name, _columns in _REDUNDANT:
            op.drop_index(index_name, table_name=table, if_exists=True)
        return
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    for table, index_name, columns in _REDUNDANT:
        if table not in tables:
            continue
        plain = {
            ix["name"] for ix in inspector.get_indexes(table) if not ix.get("unique")
        }
        if index_name in plain and columns in _enforced_unique_columns(
            inspector, table
        ):
            op.drop_index(index_name, table_name=table)


def downgrade() -> None:
    """Recreate the plain indexes dropped by ``upgrade``."""
    if op.get_context().as_sql:
        for table, index_name, columns in reversed(_REDUNDANT):
            op.create_index(index_name, table, list(columns), if_not_exists=True)
        return
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    for table, index_name, columns in reversed(_REDUNDANT):
        if table not in tables:
            continue
        present = {ix["name"] for ix in inspector.get_indexes(table)}
        if index_name not in present:
            op.create_index(index_name, table, list(columns))
