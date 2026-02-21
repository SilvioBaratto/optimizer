"""
Schemas Package
===============

This package contains Pydantic models for request/response validation and serialization.

File Structure
--------------
schemas/
├── __init__.py       # This file - exports all schemas
├── base.py           # Common base schemas and mixins
└── <entity>.py       # Entity-specific schemas (user.py, item.py, etc.)

Schema Naming Conventions
-------------------------
For each entity, create these schema types:
- <Entity>Base     - Shared fields between create/update
- <Entity>Create   - Fields required for creation
- <Entity>Update   - Fields for updates (all optional)
- <Entity>Response - Fields returned in responses
- <Entity>InDB     - Database representation (includes DB-only fields)

Example Schema Organization
---------------------------
```python
# schemas/user.py
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, EmailStr, ConfigDict


class UserBase(BaseModel):
    \"\"\"Base schema with shared fields.\"\"\"
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, max_length=255, description="Full name")
    is_active: bool = Field(True, description="Whether user is active")


class UserCreate(UserBase):
    \"\"\"Schema for creating a new user.\"\"\"
    password: str = Field(..., min_length=8, description="User password")


class UserUpdate(BaseModel):
    \"\"\"Schema for updating a user. All fields optional.\"\"\"
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=255)
    password: Optional[str] = Field(None, min_length=8)
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    \"\"\"Schema for user responses.\"\"\"
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="User ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class UserInDB(UserResponse):
    \"\"\"Schema with DB-only fields (internal use).\"\"\"
    hashed_password: str
```

Common Schema Patterns
----------------------
```python
# schemas/base.py
from datetime import datetime, timezone
from typing import Optional, Generic, TypeVar, List
from pydantic import BaseModel, Field, ConfigDict

DataT = TypeVar("DataT")


class TimestampSchema(BaseModel):
    \"\"\"Mixin for timestamp fields.\"\"\"
    created_at: datetime
    updated_at: datetime


class PaginatedResponse(BaseModel, Generic[DataT]):
    \"\"\"Generic paginated response schema.\"\"\"
    items: List[DataT]
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")


class ErrorResponse(BaseModel):
    \"\"\"Standard error response.\"\"\"
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    code: Optional[str] = Field(None, description="Error code")


class SuccessResponse(BaseModel):
    \"\"\"Standard success response.\"\"\"
    success: bool = True
    message: Optional[str] = None


# Validators example
from pydantic import field_validator

class UserCreate(UserBase):
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase letter")
        return v
```

Best Practices
--------------
1. **Use Field()**: Always use Field() for validation and documentation
2. **ConfigDict**: Use model_config = ConfigDict(from_attributes=True) for ORM
3. **Separate Create/Update**: Different validation needs for each operation
4. **Optional Updates**: All fields in Update schemas should be Optional
5. **EmailStr**: Use EmailStr for email validation
6. **Datetime Handling**: Use timezone-aware datetimes
7. **Validators**: Use @field_validator for custom validation
8. **Documentation**: Field descriptions appear in OpenAPI docs
9. **Generic Responses**: Use generics for paginated responses
10. **Nested Schemas**: Compose schemas for complex responses
"""

from datetime import datetime, timezone
from typing import Any, List, Optional

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


# ===========================
# Message/Echo Schemas
# ===========================


class MessageResponse(BaseModel):
    """Simple message response"""

    message: str


class EchoRequest(BaseModel):
    """Echo request body"""

    message: str = Field(..., description="Message to echo back")
    metadata: dict | None = Field(default=None, description="Optional metadata")


class EchoResponse(BaseModel):
    """Echo response"""

    echo: str
    received_at: datetime = Field(default_factory=utc_now)
    metadata: dict | None = None


# ===========================
# Item CRUD Schemas
# ===========================


class ItemBase(BaseModel):
    """Base item schema"""

    name: str = Field(..., min_length=1, max_length=100, description="Item name")
    description: str | None = Field(
        default=None, max_length=500, description="Item description"
    )
    price: float | None = Field(default=None, ge=0, description="Item price")
    is_active: bool = Field(default=True, description="Whether item is active")


class ItemCreate(ItemBase):
    """Schema for creating an item"""

    pass


class ItemUpdate(BaseModel):
    """Schema for updating an item (all fields optional)"""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    price: float | None = Field(default=None, ge=0)
    is_active: bool | None = None


class ItemResponse(ItemBase):
    """Schema for item response"""

    model_config = {"from_attributes": True}

    id: str = Field(..., description="Item ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class ItemListResponse(BaseModel):
    """Schema for list of items response"""

    items: list[ItemResponse]
    total: int = Field(..., description="Total number of items")


# ===========================
# Generic API Schemas
# ===========================


class ErrorResponse(BaseModel):
    """Standard error response"""

    error: str
    detail: str | None = None
    code: str | None = None


class SuccessResponse(BaseModel):
    """Standard success response"""

    success: bool = True
    message: str | None = None
    data: Any | None = None


# Import trading212 schemas
from app.schemas.trading212 import (
    BuildJobResponse,
    BuildProgressResponse,
    BuildResultResponse,
    CacheStatsResponse,
    ExchangeResponse,
    InstrumentListResponse,
    InstrumentResponse,
    UniverseBuildRequest,
    UniverseStatsResponse,
)

# Import yfinance data schemas
from app.schemas.yfinance_data import (
    AnalystPriceTargetResponse,
    AnalystRecommendationResponse,
    DividendResponse,
    FinancialStatementResponse,
    InsiderTransactionResponse,
    InstitutionalHolderResponse,
    MutualFundHolderResponse,
    PriceHistoryResponse,
    StockSplitResponse,
    TickerNewsResponse,
    TickerProfileResponse,
    YFinanceFetchJobResponse,
    YFinanceFetchProgress,
    YFinanceFetchRequest,
    YFinanceSingleFetchRequest,
    YFinanceSingleFetchResponse,
)

# Export all schemas
__all__ = [
    "AnalystPriceTargetResponse",
    "AnalystRecommendationResponse",
    "BuildJobResponse",
    "BuildProgressResponse",
    "BuildResultResponse",
    "CacheStatsResponse",
    "DividendResponse",
    "EchoRequest",
    "EchoResponse",
    "ErrorResponse",
    "ExchangeResponse",
    "FinancialStatementResponse",
    "InsiderTransactionResponse",
    "InstitutionalHolderResponse",
    "InstrumentListResponse",
    "InstrumentResponse",
    "ItemBase",
    "ItemCreate",
    "ItemListResponse",
    "ItemResponse",
    "ItemUpdate",
    "MessageResponse",
    "MutualFundHolderResponse",
    "PriceHistoryResponse",
    "StockSplitResponse",
    "SuccessResponse",
    "TickerNewsResponse",
    "TickerProfileResponse",
    "UniverseBuildRequest",
    "UniverseStatsResponse",
    "YFinanceFetchJobResponse",
    "YFinanceFetchProgress",
    "YFinanceFetchRequest",
    "YFinanceSingleFetchRequest",
    "YFinanceSingleFetchResponse",
    "utc_now",
]
