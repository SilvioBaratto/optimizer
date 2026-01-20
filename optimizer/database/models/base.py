"""Base model classes and mixins"""

import uuid
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from typing import Any


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models"""
    
    # Global type annotation map for modern SQLAlchemy
    type_annotation_map = {
        datetime: DateTime(timezone=True),
    }


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps"""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


class UUIDPrimaryKeyMixin:
    """Mixin to add UUID primary key"""
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )


class StringUUIDPrimaryKeyMixin:
    """Mixin to add string-based UUID primary key for compatibility"""
    
    id: Mapped[str] = mapped_column(
        String(36),  # UUID string length
        primary_key=True,
        nullable=False
    )


class BaseModel(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Abstract base model combining common patterns
    
    Provides:
    - UUID primary key
    - Created/updated timestamps
    - Common utilities
    """
    
    __abstract__ = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert model instance to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def __repr__(self) -> str:
        """String representation of the model"""
        class_name = self.__class__.__name__
        return f"<{class_name}(id={getattr(self, 'id', None)})>"