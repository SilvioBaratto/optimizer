"""Shared internals for the per-domain seed builders (issue #804).

Builders insert via ``session.add`` + ``session.flush()`` and never call
``session.commit()`` — the SAVEPOINT ``db_session`` fixture owns the
transaction boundary.
"""

from __future__ import annotations

from typing import TypeVar

from sqlalchemy.orm import Session

T = TypeVar("T")


def add_and_flush(session: Session, obj: T) -> T:
    """Add a single ORM row and flush so its PK / FKs are assigned."""
    session.add(obj)
    session.flush()
    return obj
