"""
Base Repository Module
======================

Generic base repository providing common CRUD operations for all entities.
Uses SQLAlchemy 2.0+ async patterns with proper type hints.
"""

from typing import Generic, TypeVar, Optional, Sequence, Any, Type
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.models.base import Base


ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Generic repository with CRUD operations.

    Type Parameters:
        ModelType: SQLAlchemy model class
        CreateSchemaType: Pydantic schema for creation
        UpdateSchemaType: Pydantic schema for updates

    Usage:
        class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
            def __init__(self, session: AsyncSession):
                super().__init__(User, session)
    """

    def __init__(self, model: Type[ModelType], session: AsyncSession):
        """
        Initialize repository with model and session.

        Args:
            model: SQLAlchemy model class
            session: Async database session
        """
        self.model = model
        self.session = session

    async def get(self, id: Any) -> Optional[ModelType]:
        """
        Get a single record by ID.

        Args:
            id: Primary key value

        Returns:
            Model instance or None if not found
        """
        id_column = getattr(self.model, "id")
        stmt = select(self.model).where(id_column == id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_field(
        self,
        field: str,
        value: Any
    ) -> Optional[ModelType]:
        """
        Get a single record by any field.

        Args:
            field: Field name to filter by
            value: Value to match

        Returns:
            Model instance or None if not found
        """
        column = getattr(self.model, field)
        stmt = select(self.model).where(column == value)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_multi(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        desc: bool = True
    ) -> Sequence[ModelType]:
        """
        Get multiple records with pagination.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            order_by: Field name to order by (default: created_at if exists)
            desc: Sort descending if True

        Returns:
            Sequence of model instances
        """
        stmt = select(self.model)

        # Apply ordering if field exists
        if order_by and hasattr(self.model, order_by):
            column = getattr(self.model, order_by)
            stmt = stmt.order_by(column.desc() if desc else column.asc())
        elif hasattr(self.model, "created_at"):
            column = getattr(self.model, "created_at")
            stmt = stmt.order_by(column.desc() if desc else column.asc())

        stmt = stmt.offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_all(self) -> Sequence[ModelType]:
        """
        Get all records (use with caution on large tables).

        Returns:
            Sequence of all model instances
        """
        stmt = select(self.model)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def create(self, obj_in: CreateSchemaType) -> ModelType:
        """
        Create a new record.

        Args:
            obj_in: Pydantic schema with creation data

        Returns:
            Created model instance
        """
        obj_data = obj_in.model_dump()
        db_obj = self.model(**obj_data)
        self.session.add(db_obj)
        await self.session.flush()
        await self.session.refresh(db_obj)
        return db_obj

    async def create_from_dict(self, obj_data: dict) -> ModelType:
        """
        Create a new record from dictionary.

        Args:
            obj_data: Dictionary with creation data

        Returns:
            Created model instance
        """
        db_obj = self.model(**obj_data)
        self.session.add(db_obj)
        await self.session.flush()
        await self.session.refresh(db_obj)
        return db_obj

    async def update(
        self,
        id: Any,
        obj_in: UpdateSchemaType
    ) -> Optional[ModelType]:
        """
        Update an existing record.

        Args:
            id: Primary key value
            obj_in: Pydantic schema with update data

        Returns:
            Updated model instance or None if not found
        """
        db_obj = await self.get(id)
        if not db_obj:
            return None

        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)

        await self.session.flush()
        await self.session.refresh(db_obj)
        return db_obj

    async def update_from_dict(
        self,
        id: Any,
        obj_data: dict
    ) -> Optional[ModelType]:
        """
        Update an existing record from dictionary.

        Args:
            id: Primary key value
            obj_data: Dictionary with update data

        Returns:
            Updated model instance or None if not found
        """
        db_obj = await self.get(id)
        if not db_obj:
            return None

        for field, value in obj_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)

        await self.session.flush()
        await self.session.refresh(db_obj)
        return db_obj

    async def delete(self, id: Any) -> bool:
        """
        Delete a record by ID.

        Args:
            id: Primary key value

        Returns:
            True if deleted, False if not found
        """
        db_obj = await self.get(id)
        if not db_obj:
            return False
        await self.session.delete(db_obj)
        await self.session.flush()
        return True

    async def count(self) -> int:
        """
        Count total records.

        Returns:
            Total count of records
        """
        stmt = select(func.count()).select_from(self.model)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def exists(self, id: Any) -> bool:
        """
        Check if a record exists by ID.

        Args:
            id: Primary key value

        Returns:
            True if exists, False otherwise
        """
        id_column = getattr(self.model, "id")
        stmt = select(func.count()).where(id_column == id)
        result = await self.session.execute(stmt)
        return result.scalar_one() > 0

    async def exists_by_field(self, field: str, value: Any) -> bool:
        """
        Check if a record exists by field value.

        Args:
            field: Field name to check
            value: Value to match

        Returns:
            True if exists, False otherwise
        """
        column = getattr(self.model, field)
        stmt = select(func.count()).where(column == value)
        result = await self.session.execute(stmt)
        return result.scalar_one() > 0
