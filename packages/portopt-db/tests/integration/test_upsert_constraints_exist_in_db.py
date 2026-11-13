"""Every upsert conflict target exists as a constraint in the migrated database.

``test_upsert_constraint_names`` checks the names against the SQLAlchemy models;
this one checks them against the schema the migrations actually built, which is
what ``ON CONFLICT ON CONSTRAINT`` resolves at runtime. It catches a model that
declares a constraint no migration created.
"""

from __future__ import annotations

import importlib

import pytest
from _upsert_sites import collect_upsert_sites
from sqlalchemy import text
from sqlalchemy.engine import Connection

pytestmark = pytest.mark.integration

_FIND_CONSTRAINT = text(
    "SELECT 1 FROM pg_constraint "
    "WHERE conname = :name AND conrelid = to_regclass(:table) "
    "AND contype IN ('u', 'p')"
)


def test_every_upsert_constraint_exists_in_postgres(pg_connection: Connection) -> None:
    """Each conflict constraint is a unique or primary-key constraint of its table."""
    missing: list[str] = []
    for site in collect_upsert_sites():
        model = getattr(importlib.import_module(site.module), site.model_name)
        found = pg_connection.execute(
            _FIND_CONSTRAINT,
            {"name": site.constraint_name, "table": model.__table__.fullname},
        ).first()
        if found is None:
            missing.append(f"{site.file.name}:{site.line} {site.constraint_name}")

    assert missing == []
