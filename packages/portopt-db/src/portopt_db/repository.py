"""Base repository: session holder + shared idempotent ``_upsert`` (ON CONFLICT)
and a generic CRUD mixin. Synchronous SQLAlchemy 2.0.
"""

from collections.abc import Sequence
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel
from sqlalchemy import Table, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, Session

from portopt_db.base import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


def _get_table(model: type[DeclarativeBase]) -> Table:
    return cast(Table, model.__table__)


class RepositoryBase:
    """Minimal base for all repositories. Provides session and shared upsert."""

    def __init__(self, session: Session):
        self.session = session

    def _upsert(
        self,
        model: type,
        rows: list[dict[str, Any]],
        constraint_name: str | None = None,
        update_columns: list[str] | None = None,
        index_elements: list[str] | None = None,
    ) -> int:
        """Insert rows with ON CONFLICT DO UPDATE. Returns count of rows processed.

        The conflict target is one of, exactly:
        - ``constraint_name`` — a named unique constraint (e.g.
          ``uq_economic_indicator_country``), or
        - ``index_elements`` — the column name(s) of a unique index, for
          column-level ``unique=True`` columns that carry no named constraint
          (e.g. ``exchanges.name``).
        """
        if not rows:
            return 0

        if (constraint_name is None) == (index_elements is None):
            raise ValueError(
                "_upsert requires exactly one of constraint_name / index_elements"
            )

        tbl = _get_table(model)
        stmt = pg_insert(tbl).values(rows)

        update_dict: dict[str, Any]
        if update_columns:
            update_dict = {col: stmt.excluded[col] for col in update_columns}
        else:
            exclude = {"id", "created_at"}
            update_dict = {
                col.name: stmt.excluded[col.name]
                for col in tbl.columns
                if col.name not in exclude
            }

        # Always stamp updated_at with the server's current time on conflict.
        # excluded.updated_at is unreliable: server_default columns are omitted
        # from the proposed-INSERT VALUES list, so excluded reflects the original
        # insert timestamp or NULL — never the conflict timestamp.
        if "updated_at" in update_dict:
            update_dict["updated_at"] = func.now()

        if constraint_name is not None:
            stmt = stmt.on_conflict_do_update(
                constraint=constraint_name,
                set_=update_dict,
            )
        else:
            stmt = stmt.on_conflict_do_update(
                index_elements=index_elements,
                set_=update_dict,
            )

        self.session.execute(stmt)
        return len(rows)


class BaseRepository(
    RepositoryBase,
    Generic[ModelType, CreateSchemaType, UpdateSchemaType],
):
    """Generic CRUD repository for a single SQLAlchemy model, bound at construction.

    Flush-based: every write calls ``flush()`` then ``refresh()`` to populate server
    defaults (e.g. generated PKs, ``created_at``).  The caller controls commit/rollback.
    """

    def __init__(self, model: type[ModelType], session: Session):
        super().__init__(session)
        self.model = model

    def get(self, id: Any) -> ModelType | None:
        """Fetch a single record by primary key.

        Args:
            id: Primary key value for the lookup.

        Returns:
            The mapped instance, or ``None`` if no row matches.
        """
        id_column = cast(Any, self.model).id
        stmt = select(self.model).where(id_column == id)
        result = self.session.execute(stmt)
        return result.scalar_one_or_none()

    def get_by_field(self, field: str, value: Any) -> ModelType | None:
        """Fetch the first record matching ``field == value``.

        Args:
            field: Attribute name on the model to filter by.
            value: Value to compare against.

        Returns:
            The matched instance, or ``None`` if no row matches.
        """
        column = getattr(self.model, field)
        stmt = select(self.model).where(column == value)
        result = self.session.execute(stmt)
        return result.scalar_one_or_none()

    def get_multi(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        order_by: str | None = None,
        desc: bool = True,
    ) -> Sequence[ModelType]:
        """Fetch a paginated slice of records.

        Falls back to ``created_at`` ordering when ``order_by`` is ``None`` or
        names an attribute that does not exist on the model.

        Args:
            skip: Number of rows to skip before returning results.
            limit: Maximum number of rows to return.
            order_by: Model attribute name to sort by.
            desc: Sort descending when ``True`` (default); ascending otherwise.

        Returns:
            Sequence of matched instances, possibly empty.
        """
        stmt = select(self.model)

        if order_by and hasattr(self.model, order_by):
            column = getattr(self.model, order_by)
            stmt = stmt.order_by(column.desc() if desc else column.asc())
        elif hasattr(self.model, "created_at"):
            column = cast(Any, self.model).created_at
            stmt = stmt.order_by(column.desc() if desc else column.asc())

        stmt = stmt.offset(skip).limit(limit)
        result = self.session.execute(stmt)
        return result.scalars().all()

    def get_all(self) -> Sequence[ModelType]:
        stmt = select(self.model)
        result = self.session.execute(stmt)
        return result.scalars().all()

    def create(self, obj_in: CreateSchemaType) -> ModelType:
        """Insert a new record and return the refreshed instance.

        Args:
            obj_in: Pydantic create schema; all fields are written.

        Returns:
            The persisted instance with server-generated values (e.g. PK, timestamps)
            populated via ``refresh()``.
        """
        obj_data = obj_in.model_dump()
        db_obj = self.model(**obj_data)
        self.session.add(db_obj)
        self.session.flush()
        self.session.refresh(db_obj)
        return db_obj

    def create_from_dict(self, obj_data: dict[str, Any]) -> ModelType:
        """Insert a new record from a raw mapping and return the refreshed instance.

        Args:
            obj_data: Column-name to value mapping passed directly to the model constructor.

        Returns:
            The persisted instance with server-generated values populated.
        """
        db_obj = self.model(**obj_data)
        self.session.add(db_obj)
        self.session.flush()
        self.session.refresh(db_obj)
        return db_obj

    def update(self, id: Any, obj_in: UpdateSchemaType) -> ModelType | None:
        """Apply a partial update via a Pydantic schema.

        Only fields explicitly set in ``obj_in`` are written (``exclude_unset=True``),
        so callers may omit unchanged fields for patch semantics.

        Args:
            id: Primary key of the record to update.
            obj_in: Pydantic update schema; unset fields are ignored.

        Returns:
            The refreshed instance, or ``None`` if no record with ``id`` exists.
        """
        db_obj = self.get(id)
        if not db_obj:
            return None

        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)

        self.session.flush()
        self.session.refresh(db_obj)
        return db_obj

    def update_from_dict(self, id: Any, obj_data: dict[str, Any]) -> ModelType | None:
        """Apply a partial update from a raw mapping.

        Keys in ``obj_data`` that do not correspond to model attributes are silently
        skipped, so callers may pass superset dicts without error.

        Args:
            id: Primary key of the record to update.
            obj_data: Column-name to value mapping; unknown keys are ignored.

        Returns:
            The refreshed instance, or ``None`` if no record with ``id`` exists.
        """
        db_obj = self.get(id)
        if not db_obj:
            return None

        for field, value in obj_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)

        self.session.flush()
        self.session.refresh(db_obj)
        return db_obj

    def delete(self, id: Any) -> bool:
        """Delete a record by primary key.

        Args:
            id: Primary key of the record to remove.

        Returns:
            ``True`` if the record existed and was deleted; ``False`` if not found.
        """
        db_obj = self.get(id)
        if not db_obj:
            return False
        self.session.delete(db_obj)
        self.session.flush()
        return True

    def count(self) -> int:
        stmt = select(func.count()).select_from(self.model)
        result = self.session.execute(stmt)
        return result.scalar_one()

    def exists(self, id: Any) -> bool:
        id_column = cast(Any, self.model).id
        stmt = select(func.count()).where(id_column == id)
        result = self.session.execute(stmt)
        return result.scalar_one() > 0

    def exists_by_field(self, field: str, value: Any) -> bool:
        column = getattr(self.model, field)
        stmt = select(func.count()).where(column == value)
        result = self.session.execute(stmt)
        return result.scalar_one() > 0


__all__ = [
    "BaseRepository",
    "CreateSchemaType",
    "ModelType",
    "RepositoryBase",
    "UpdateSchemaType",
]
