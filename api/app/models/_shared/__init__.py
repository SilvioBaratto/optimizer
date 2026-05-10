"""Shared Models.

Cross-cutting base classes and mixins used by all domain models.
"""

from app.models._shared.base import (
    Base,
    BaseModel,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

__all__ = [
    "Base",
    "BaseModel",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
]
