"""
Utilities Package
=================

This package contains utility functions and helpers used across the application.

File Structure
--------------
utils/
    __init__.py       # This file - exports utility functions
    helpers.py        # General helper functions
    validators.py     # Custom validation utilities
    formatters.py     # Data formatting utilities
    datetime_utils.py # Date/time helpers
    pagination.py     # Pagination utilities

Common Utilities Example
------------------------
```python
# utils/helpers.py
import re
import uuid
from typing import Any, Optional


def generate_uuid() -> str:
    \"\"\"Generate a unique UUID string.\"\"\"
    return str(uuid.uuid4())


def slugify(text: str) -> str:
    \"\"\"Convert text to URL-friendly slug.\"\"\"
    text = text.lower().strip()
    text = re.sub(r'[^\\w\\s-]', '', text)
    text = re.sub(r'[\\s_-]+', '-', text)
    return text


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    \"\"\"Truncate text to max length with suffix.\"\"\"
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_get(dictionary: dict, *keys: str, default: Any = None) -> Any:
    \"\"\"Safely get nested dictionary values.\"\"\"
    for key in keys:
        try:
            dictionary = dictionary[key]
        except (KeyError, TypeError):
            return default
    return dictionary
```

Pagination Utilities Example
----------------------------
```python
# utils/pagination.py
from typing import Generic, TypeVar, Sequence, Optional
from pydantic import BaseModel, Field
from math import ceil


T = TypeVar("T")


class PaginationParams(BaseModel):
    \"\"\"Query parameters for pagination.\"\"\"
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")

    @property
    def skip(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size


class PaginatedResponse(BaseModel, Generic[T]):
    \"\"\"Generic paginated response.\"\"\"
    items: Sequence[T]
    total: int
    page: int
    page_size: int
    pages: int

    @classmethod
    def create(
        cls,
        items: Sequence[T],
        total: int,
        params: PaginationParams
    ) -> "PaginatedResponse[T]":
        return cls(
            items=items,
            total=total,
            page=params.page,
            page_size=params.page_size,
            pages=ceil(total / params.page_size) if total > 0 else 0
        )
```

DateTime Utilities Example
--------------------------
```python
# utils/datetime_utils.py
from datetime import datetime, timezone, timedelta
from typing import Optional


def utc_now() -> datetime:
    \"\"\"Get current UTC datetime.\"\"\"
    return datetime.now(timezone.utc)


def format_datetime(dt: Optional[datetime], format: str = "%Y-%m-%d %H:%M:%S") -> str:
    \"\"\"Format datetime to string.\"\"\"
    if dt is None:
        return ""
    return dt.strftime(format)


def parse_datetime(date_string: str, format: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    \"\"\"Parse string to datetime.\"\"\"
    return datetime.strptime(date_string, format)


def time_ago(dt: datetime) -> str:
    \"\"\"Get human-readable time ago string.\"\"\"
    now = utc_now()
    diff = now - dt

    if diff.days > 365:
        years = diff.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
    elif diff.days > 30:
        months = diff.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "just now"
```
"""

# Import and export your utilities here:
# from app.utils.helpers import generate_uuid, slugify, truncate, safe_get
# from app.utils.pagination import PaginationParams, PaginatedResponse
# from app.utils.datetime_utils import utc_now, format_datetime, time_ago

__all__ = [
    # "generate_uuid",
    # "slugify",
    # "truncate",
    # "safe_get",
    # "PaginationParams",
    # "PaginatedResponse",
    # "utc_now",
    # "format_datetime",
    # "time_ago",
]
