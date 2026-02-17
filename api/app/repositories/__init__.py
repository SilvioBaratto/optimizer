"""
Repositories Package
====================

This package contains repository classes that abstract database operations.
The Repository Pattern separates data access logic from business logic.

File Structure
--------------
repositories/
├── __init__.py       # This file - exports repository classes
├── base.py           # Generic base repository with CRUD operations
└── <entity>_repository.py  # Entity-specific repositories

Base Repository Pattern
-----------------------
Create a generic base repository for common CRUD operations:

```python
# repositories/base.py
from typing import Generic, TypeVar, Optional, Sequence, Any
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.models.base import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    \"\"\"
    Generic repository with CRUD operations.

    Provides:
    - get(id): Get single record by ID
    - get_multi(skip, limit): Get paginated records
    - create(obj_in): Create new record
    - update(id, obj_in): Update existing record
    - delete(id): Delete record
    - count(): Count total records
    \"\"\"

    def __init__(self, model: type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session

    async def get(self, id: Any) -> Optional[ModelType]:
        \"\"\"Get a single record by ID.\"\"\"
        stmt = select(self.model).where(self.model.id == id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_field(self, field: str, value: Any) -> Optional[ModelType]:
        \"\"\"Get a single record by any field.\"\"\"
        column = getattr(self.model, field)
        stmt = select(self.model).where(column == value)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_multi(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: str = "created_at",
        desc: bool = True
    ) -> Sequence[ModelType]:
        \"\"\"Get multiple records with pagination.\"\"\"
        column = getattr(self.model, order_by)
        order = column.desc() if desc else column.asc()
        stmt = select(self.model).order_by(order).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def create(self, obj_in: CreateSchemaType) -> ModelType:
        \"\"\"Create a new record.\"\"\"
        obj_data = obj_in.model_dump()
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
        \"\"\"Update an existing record.\"\"\"
        db_obj = await self.get(id)
        if not db_obj:
            return None

        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)

        await self.session.flush()
        await self.session.refresh(db_obj)
        return db_obj

    async def delete(self, id: Any) -> bool:
        \"\"\"Delete a record by ID.\"\"\"
        db_obj = await self.get(id)
        if not db_obj:
            return False
        await self.session.delete(db_obj)
        await self.session.flush()
        return True

    async def count(self) -> int:
        \"\"\"Count total records.\"\"\"
        stmt = select(func.count()).select_from(self.model)
        result = await self.session.execute(stmt)
        return result.scalar_one()
```

Entity-Specific Repository
--------------------------
Extend base repository for entity-specific operations:

```python
# repositories/user_repository.py
from typing import Optional, Sequence
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    \"\"\"Repository for User-specific database operations.\"\"\"

    def __init__(self, session: AsyncSession):
        super().__init__(User, session)

    async def get_by_email(self, email: str) -> Optional[User]:
        \"\"\"Get user by email address.\"\"\"
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_with_items(self, user_id: str) -> Optional[User]:
        \"\"\"Get user with eagerly loaded items.\"\"\"
        stmt = (
            select(User)
            .where(User.id == user_id)
            .options(selectinload(User.items))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def search(
        self,
        query: str,
        skip: int = 0,
        limit: int = 100
    ) -> Sequence[User]:
        \"\"\"Search users by email or name.\"\"\"
        search_pattern = f"%{query}%"
        stmt = (
            select(User)
            .where(
                or_(
                    User.email.ilike(search_pattern),
                    User.full_name.ilike(search_pattern)
                )
            )
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_active_users(self) -> Sequence[User]:
        \"\"\"Get all active users.\"\"\"
        stmt = select(User).where(User.is_active == True)
        result = await self.session.execute(stmt)
        return result.scalars().all()
```

Using Repositories in Services
------------------------------
```python
# services/user_service.py
class UserService:
    def __init__(self, session: AsyncSession):
        self.repository = UserRepository(session)

    async def get_user(self, user_id: str) -> Optional[User]:
        return await self.repository.get(user_id)
```

Best Practices
--------------
1. **Single Responsibility**: One repository per entity/aggregate
2. **Generic Base**: Use generics for type-safe CRUD operations
3. **Async Operations**: Use async/await for all database operations
4. **Session Injection**: Inject session via constructor, don't create it
5. **Flush vs Commit**: Use flush() in repos, commit() in service/route layer
6. **Eager Loading**: Use selectinload/joinedload for relationships
7. **No Business Logic**: Repositories handle data access only
8. **Query Builders**: Complex queries should return query builders when needed
"""

# Import and export your repositories here:
# from app.repositories.base import BaseRepository
# from app.repositories.user_repository import UserRepository
from app.repositories.universe_repository import UniverseRepository
from app.repositories.yfinance_repository import YFinanceRepository
from app.repositories.macro_regime_repository import MacroRegimeRepository

__all__ = [
    # "BaseRepository",
    # "UserRepository",
    "UniverseRepository",
    "YFinanceRepository",
    "MacroRegimeRepository",
]
