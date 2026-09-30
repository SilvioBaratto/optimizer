"""SQLAlchemy declarative Base + shared mixins for the portopt database.

Single source of the ORM registry: every model in ``portopt_db.models``
inherits this ``Base`` so ``Base.metadata`` holds one complete schema.
"""

import uuid
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """ORM registry root for all portopt_db models.

    Registers timezone-aware DateTime as the default column type for
    ``datetime`` annotations, so individual models need not repeat the
    dialect option on every timestamp column.
    """

    type_annotation_map: ClassVar[dict[Any, Any]] = {
        datetime: DateTime(timezone=True),
    }


class TimestampMixin:
    """Adds server-managed ``created_at`` and ``updated_at`` audit columns.

    Both columns default to the DB server clock on INSERT; ``updated_at``
    refreshes automatically on every UPDATE without any application-side
    involvement.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class UUIDPrimaryKeyMixin:
    """Adds a UUID v4 primary key column named ``id``.

    The default is generated Python-side via ``uuid.uuid4``, so the value is
    available before the row is flushed to the database.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False
    )


class BaseModel(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Abstract model combining a UUID PK with server-managed audit timestamps.

    Cannot be mapped to a table directly (``__abstract__ = True``); subclass
    it to inherit the ``id``, ``created_at``, and ``updated_at`` columns.
    """

    __abstract__ = True

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict mapping column name to value for this instance.

        Covers only mapped table columns; ORM relationships are excluded.
        """
        return {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"<{class_name}(id={getattr(self, 'id', None)})>"


__all__ = ["Base", "BaseModel", "TimestampMixin", "UUIDPrimaryKeyMixin"]
