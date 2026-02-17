"""
API Version 1 Package
=====================

This package contains all v1 API endpoints organized by domain.

File Structure
--------------
v1/
├── __init__.py      # This file - package documentation
├── router.py        # Main router - imports and includes all domain routers
└── <domain>.py      # Domain-specific endpoints (one file per resource)

Creating a New Route Module
---------------------------
Each domain module should follow this pattern:

```python
# api/v1/users.py
\"\"\"User management endpoints\"\"\"

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.user import UserCreate, UserResponse, UserUpdate
from app.services.user_service import UserService
from app.dependencies import get_current_user

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    \"\"\"
    List all users with pagination.

    - **skip**: Number of records to skip (default: 0)
    - **limit**: Maximum number of records to return (default: 100)
    \"\"\"
    service = UserService(db)
    return await service.get_multi(skip=skip, limit=limit)


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    \"\"\"Create a new user.\"\"\"
    service = UserService(db)
    return await service.create(user_in)


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    \"\"\"Get a specific user by ID.\"\"\"
    service = UserService(db)
    user = await service.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_in: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    \"\"\"Update an existing user.\"\"\"
    service = UserService(db)
    user = await service.update(user_id, user_in)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    \"\"\"Delete a user.\"\"\"
    service = UserService(db)
    success = await service.delete(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
```

Router Registration (router.py)
-------------------------------
```python
from fastapi import APIRouter
from app.api.v1.users import router as users_router
from app.api.v1.items import router as items_router

api_router = APIRouter()
api_router.include_router(users_router)
api_router.include_router(items_router)
```

Best Practices
--------------
1. **Use Dependency Injection**: Inject database sessions and services via Depends()
2. **Type Hints**: Always use Pydantic schemas for request/response types
3. **HTTP Status Codes**: Use appropriate status codes (201 for create, 204 for delete)
4. **Documentation**: Write docstrings - they appear in OpenAPI/Swagger docs
5. **Error Handling**: Raise HTTPException with meaningful error messages
6. **Tags**: Use tags=["Resource"] for OpenAPI organization
7. **Async**: Use async def for all endpoints when using async database
"""
