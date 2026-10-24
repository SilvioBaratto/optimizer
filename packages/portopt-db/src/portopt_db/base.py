"""SQLAlchemy declarative Base + shared mixins for the portopt database.

Single source of the ORM registry: every model in ``portopt_db.models``
inherits this ``Base`` so ``Base.metadata`` holds one complete schema.
"""

import uuid
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql.compiler import SQLCompiler, TypeCompiler
from sqlalchemy.sql.expression import FunctionElement


class Base(DeclarativeBase):
    """ORM registry root for all portopt_db models.

    Registers timezone-aware DateTime as the default column type for
    ``datetime`` annotations, so individual models need not repeat the
    dialect option on every timestamp column.
    """

    type_annotation_map: ClassVar[dict[Any, Any]] = {
        datetime: DateTime(timezone=True),
    }


class ClockTimestamp(FunctionElement[datetime]):
    """SQL expression for the DB server's wall clock at statement execution.

    PostgreSQL's ``now()`` is frozen at transaction start, so rows written
    minutes apart inside one long transaction (an agent run) would all carry
    the same stale instant. ``clock_timestamp()`` advances within a
    transaction. SQLite has no such function; the stand-in below keeps
    millisecond resolution so the test database orders rows the same way.
    """

    type = DateTime(timezone=True)
    name = "clock_timestamp"
    inherit_cache = True


@compiles(ClockTimestamp)
def _compile_clock_timestamp(
    element: ClockTimestamp, compiler: SQLCompiler, **kw: Any
) -> str:
    return "clock_timestamp()"


@compiles(ClockTimestamp, "sqlite")
def _compile_clock_timestamp_sqlite(
    element: ClockTimestamp, compiler: SQLCompiler, **kw: Any
) -> str:
    return "STRFTIME('%Y-%m-%d %H:%M:%f', 'now')"


@compiles(UUID, "sqlite")
def _compile_uuid_sqlite(element: UUID[Any], compiler: TypeCompiler, **kw: Any) -> str:
    # SQLite reads affinity off the declared type name, and ``UUID`` would get NUMERIC:
    # a stored hex id that parses as a number (all digits, or one ``e``) turns REAL.
    return "CHAR(32)"


class TimestampMixin:
    """Adds server-clocked ``created_at`` and ``updated_at`` audit columns.

    Both columns are stamped with the DB server clock at statement time on
    INSERT; ``updated_at`` refreshes automatically on every ORM/Core UPDATE
    without any application-side involvement.
    """

    # ``default`` governs every ORM/Core write; ``server_default`` is kept only
    # so raw SQL inserts and the migrated schema still get a value.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=ClockTimestamp(),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=ClockTimestamp(),
        server_default=func.now(),
        onupdate=ClockTimestamp(),
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


__all__ = [
    "Base",
    "BaseModel",
    "ClockTimestamp",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
]
