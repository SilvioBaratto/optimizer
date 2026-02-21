"""
Services Package
================

This package contains business logic services that orchestrate operations
between repositories, external APIs, and other services.

File Structure
--------------
services/
├── __init__.py           # This file - exports service classes
├── base.py               # Base service class (optional)
└── <domain>_service.py   # Domain-specific services

Service Layer Responsibilities
------------------------------
- Business logic and validation
- Orchestrating repository operations
- External API integrations
- Event publishing
- Transaction management
- Caching strategies

Service Pattern
---------------
```python
# services/user_service.py
from typing import Optional, Sequence
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.repositories.user_repository import UserRepository
from app.core.security import get_password_hash, verify_password
from app.core.exceptions import NotFoundError, ConflictError


class UserService:
    \"\"\"
    Service for user-related business logic.

    Handles:
    - User CRUD with business rules
    - Password hashing
    - Email uniqueness validation
    - User authentication
    \"\"\"

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = UserRepository(session)

    async def get(self, user_id: str) -> Optional[User]:
        \"\"\"Get user by ID.\"\"\"
        return await self.repository.get(user_id)

    async def get_by_email(self, email: str) -> Optional[User]:
        \"\"\"Get user by email.\"\"\"
        return await self.repository.get_by_email(email)

    async def get_multi(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> Sequence[User]:
        \"\"\"Get paginated list of users.\"\"\"
        return await self.repository.get_multi(skip=skip, limit=limit)

    async def create(self, user_in: UserCreate) -> User:
        \"\"\"
        Create a new user.

        Raises:
            ConflictError: If email already exists
        \"\"\"
        # Check email uniqueness
        existing = await self.repository.get_by_email(user_in.email)
        if existing:
            raise ConflictError(f"Email {user_in.email} already registered")

        # Hash password before storing
        user_data = user_in.model_dump()
        user_data["hashed_password"] = get_password_hash(user_data.pop("password"))

        # Create user
        user = User(**user_data)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)

        return user

    async def update(
        self,
        user_id: str,
        user_in: UserUpdate
    ) -> User:
        \"\"\"
        Update an existing user.

        Raises:
            NotFoundError: If user not found
            ConflictError: If new email already exists
        \"\"\"
        user = await self.repository.get(user_id)
        if not user:
            raise NotFoundError(f"User {user_id} not found")

        update_data = user_in.model_dump(exclude_unset=True)

        # Check email uniqueness if changing
        if "email" in update_data and update_data["email"] != user.email:
            existing = await self.repository.get_by_email(update_data["email"])
            if existing:
                raise ConflictError(f"Email {update_data['email']} already registered")

        # Hash new password if provided
        if "password" in update_data:
            update_data["hashed_password"] = get_password_hash(
                update_data.pop("password")
            )

        # Update fields
        for field, value in update_data.items():
            setattr(user, field, value)

        await self.session.commit()
        await self.session.refresh(user)

        return user

    async def delete(self, user_id: str) -> bool:
        \"\"\"
        Delete a user.

        Raises:
            NotFoundError: If user not found
        \"\"\"
        user = await self.repository.get(user_id)
        if not user:
            raise NotFoundError(f"User {user_id} not found")

        await self.session.delete(user)
        await self.session.commit()
        return True

    async def authenticate(
        self,
        email: str,
        password: str
    ) -> Optional[User]:
        \"\"\"
        Authenticate user by email and password.

        Returns:
            User if authentication successful, None otherwise
        \"\"\"
        user = await self.repository.get_by_email(email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user
```

Using Services in Routes
------------------------
```python
# api/v1/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.user_service import UserService
from app.schemas.user import UserCreate, UserResponse
from app.core.exceptions import NotFoundError, ConflictError

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    service = UserService(db)
    try:
        return await service.create(user_in)
    except ConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
```

Service with External APIs
--------------------------
```python
# services/payment_service.py
import httpx
from app.config import settings


class PaymentService:
    \"\"\"Service for payment processing.\"\"\"

    def __init__(self, session: AsyncSession):
        self.session = session
        self.api_key = settings.stripe_api_key
        self.base_url = "https://api.stripe.com/v1"

    async def create_payment_intent(
        self,
        amount: int,
        currency: str = "usd"
    ) -> dict:
        \"\"\"Create a Stripe payment intent.\"\"\"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/payment_intents",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data={"amount": amount, "currency": currency}
            )
            response.raise_for_status()
            return response.json()
```

Best Practices
--------------
1. **Business Logic Only**: Keep HTTP/API concerns in routes
2. **Transaction Boundaries**: Services control commit/rollback
3. **Repository Composition**: Services can use multiple repositories
4. **Exception Handling**: Raise domain exceptions, not HTTP exceptions
5. **Dependency Injection**: Inject session/repos via constructor
6. **Async Operations**: Use async/await consistently
7. **Single Responsibility**: One service per domain/aggregate
8. **Testing**: Services should be easily testable with mocked repos
9. **Caching**: Implement caching strategies in services
10. **Logging**: Log important business operations
"""

# Import and export your services here:
# from app.services.user_service import UserService
# from app.services.auth_service import AuthService
from app.services.trading212 import (
    BuildProgress,
    BuildResult,
    Trading212Client,
    UniverseBuilder,
    UniverseBuilderConfig,
    YFinanceTickerMapper,
)

__all__ = [
    # "UserService",
    # "AuthService",
    "UniverseBuilder",
    "BuildResult",
    "BuildProgress",
    "Trading212Client",
    "YFinanceTickerMapper",
    "UniverseBuilderConfig",
]
