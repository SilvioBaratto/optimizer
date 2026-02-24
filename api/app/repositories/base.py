"""Base Repository Module.

Generic base repository providing common CRUD operations for all entities.
Uses synchronous SQLAlchemy 2.0+ patterns with proper type hints.
"""

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.base import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Generic repository with synchronous CRUD operations.

    Type Parameters:
        ModelType: SQLAlchemy model class
        CreateSchemaType: Pydantic schema for creation
        UpdateSchemaType: Pydantic schema for updates

    Usage::

        class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
            def __init__(self, session: Session):
                super().__init__(User, session)
    """

    def __init__(self, model: type[ModelType], session: Session):
        self.model = model
        self.session = session

    def get(self, id: Any) -> ModelType | None:
        id_column = self.model.id
        stmt = select(self.model).where(id_column == id)
        result = self.session.execute(stmt)
        return result.scalar_one_or_none()

    def get_by_field(self, field: str, value: Any) -> ModelType | None:
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
        stmt = select(self.model)

        if order_by and hasattr(self.model, order_by):
            column = getattr(self.model, order_by)
            stmt = stmt.order_by(column.desc() if desc else column.asc())
        elif hasattr(self.model, "created_at"):
            column = self.model.created_at
            stmt = stmt.order_by(column.desc() if desc else column.asc())

        stmt = stmt.offset(skip).limit(limit)
        result = self.session.execute(stmt)
        return result.scalars().all()

    def get_all(self) -> Sequence[ModelType]:
        stmt = select(self.model)
        result = self.session.execute(stmt)
        return result.scalars().all()

    def create(self, obj_in: CreateSchemaType) -> ModelType:
        obj_data = obj_in.model_dump()
        db_obj = self.model(**obj_data)
        self.session.add(db_obj)
        self.session.flush()
        self.session.refresh(db_obj)
        return db_obj

    def create_from_dict(self, obj_data: dict) -> ModelType:
        db_obj = self.model(**obj_data)
        self.session.add(db_obj)
        self.session.flush()
        self.session.refresh(db_obj)
        return db_obj

    def update(self, id: Any, obj_in: UpdateSchemaType) -> ModelType | None:
        db_obj = self.get(id)
        if not db_obj:
            return None

        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)

        self.session.flush()
        self.session.refresh(db_obj)
        return db_obj

    def update_from_dict(self, id: Any, obj_data: dict) -> ModelType | None:
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
        id_column = self.model.id
        stmt = select(func.count()).where(id_column == id)
        result = self.session.execute(stmt)
        return result.scalar_one() > 0

    def exists_by_field(self, field: str, value: Any) -> bool:
        column = getattr(self.model, field)
        stmt = select(func.count()).where(column == value)
        result = self.session.execute(stmt)
        return result.scalar_one() > 0
