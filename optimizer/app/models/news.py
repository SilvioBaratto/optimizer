import uuid
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum
from typing import Optional, TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    String, Integer, Float, DateTime, Text,
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import Enum as SQLEnum

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.macro_regime import CountryRegimeAssessment

class NewsArticle(BaseModel):
    """
    News articles used in regime analysis (optional).
    Simplified to essential metadata only.
    """

    __tablename__ = "news_articles"

    assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("country_regime_assessments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to assessment"
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Article title"
    )

    publisher: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Publisher name"
    )

    published_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Publication date"
    )

    link: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="URL to article"
    )

    # Relationships
    assessment: Mapped["CountryRegimeAssessment"] = relationship(
        "CountryRegimeAssessment",
        back_populates="news_articles"
    )

    __table_args__ = (
        Index('idx_news_assessment_id', 'assessment_id'),
        Index('idx_news_published_date', 'published_date'),
        {
            'comment': 'News articles used in macro regime analysis',
        }
    )

    def __repr__(self) -> str:
        title_preview = self.title[:50] if self.title else "No title"
        return f"<NewsArticle(title='{title_preview}...', publisher='{self.publisher}')>"