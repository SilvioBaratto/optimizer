"""Compatibility shim — moved to ``portopt_db.base`` (P1 of the portopt-db
extraction). Re-exported here so existing ``app.models._shared`` imports keep
working until models move (P2); removed in P4.2.
"""

from portopt_db.base import (
    Base,
    BaseModel,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

__all__ = ["Base", "BaseModel", "TimestampMixin", "UUIDPrimaryKeyMixin"]
