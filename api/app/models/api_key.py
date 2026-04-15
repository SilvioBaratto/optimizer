"""ApiKey SQLAlchemy model with HMAC-signed key storage."""

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class ApiKey(BaseModel):
    """Stores hashed API keys for request authentication."""

    __tablename__ = "api_keys"

    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
