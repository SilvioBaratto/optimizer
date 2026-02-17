"""
Core Package
============

This package contains core functionality used across the application.

File Structure
--------------
core/
    __init__.py       # This file - exports core utilities
    security.py       # Password hashing, JWT tokens, authentication
    auth.py           # Authentication dependencies and utilities
    events.py         # Event system for pub/sub patterns (optional)
    cache.py          # Caching utilities (Redis, in-memory)

Security Module Example
-----------------------
```python
# core/security.py
from datetime import datetime, timedelta
from typing import Optional, Any

from jose import jwt, JWTError
from passlib.context import CryptContext

from app.config import settings


# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
ALGORITHM = "HS256"


def create_access_token(
    subject: str | Any,
    expires_delta: Optional[timedelta] = None
) -> str:
    \"\"\"Create a JWT access token.\"\"\"
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)

    to_encode = {"exp": expire, "sub": str(subject), "type": "access"}
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(subject: str | Any) -> str:
    \"\"\"Create a JWT refresh token with longer expiration.\"\"\"
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[str]:
    \"\"\"Verify a JWT token and return the subject.\"\"\"
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    \"\"\"Verify a password against its hash.\"\"\"
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    \"\"\"Hash a password using bcrypt.\"\"\"
    return pwd_context.hash(password)
```

Auth Dependencies Example
-------------------------
```python
# core/auth.py
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.core.security import verify_token
from app.services.user_service import UserService
from app.models.user import User


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    \"\"\"Get the current authenticated user from JWT token.\"\"\"
    user_id = verify_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    service = UserService(db)
    user = await service.get(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    \"\"\"Get current user and verify they are active.\"\"\"
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_active_user)
) -> User:
    \"\"\"Get current user and verify they are a superuser.\"\"\"
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user
```
"""

# Import and export core utilities here:
# from app.core.security import (
#     create_access_token,
#     create_refresh_token,
#     verify_token,
#     verify_password,
#     get_password_hash,
# )
# from app.core.auth import (
#     get_current_user,
#     get_current_active_user,
#     get_current_superuser,
# )

__all__ = [
    # "create_access_token",
    # "create_refresh_token",
    # "verify_token",
    # "verify_password",
    # "get_password_hash",
    # "get_current_user",
    # "get_current_active_user",
    # "get_current_superuser",
]
