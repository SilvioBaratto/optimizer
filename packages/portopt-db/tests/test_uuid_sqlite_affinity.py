"""UUID keys survive SQLite's type-affinity rules.

SQLite derives a column's affinity from its declared type name. ``UUID`` contains
none of the TEXT keywords, so the column would get NUMERIC affinity and a hex id that
reads as a number (all digits, or digits around one ``e``) would be stored as a REAL —
a random ``uuid4`` does that often enough to fail a full test run now and then.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from portopt_db.models import Exchange


@pytest.mark.parametrize(
    "hex_id",
    [
        "12345678-1234-4123-8123-123456789012",
        "12345678-1234-4123-8123-123456789e12",
    ],
    ids=["all-digits", "exponent"],
)
def test_a_numeric_looking_uuid_primary_key_round_trips(db_session, hex_id) -> None:
    """A UUID whose hex form parses as a number reads back as the same UUID."""
    pk = uuid.UUID(hex_id)
    db_session.add(Exchange(id=pk, name="EX-affinity"))
    db_session.flush()
    db_session.expunge_all()

    stored = db_session.scalars(
        select(Exchange).where(Exchange.name == "EX-affinity")
    ).one()

    assert stored.id == pk
