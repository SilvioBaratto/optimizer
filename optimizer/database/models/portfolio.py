#!/usr/bin/env python3
"""
Portfolio Models - Database Schema for Saved Portfolios
========================================================

Stores optimized portfolio allocations with their positions.

Tables:
    - portfolios: Main portfolio metadata (id, date, optimization method, metrics)
    - portfolio_positions: Individual stock positions (ticker, weight, sector, etc.)

Usage:
    from app.models.portfolio import Portfolio, PortfolioPosition

    # Create portfolio
    portfolio = Portfolio(
        portfolio_date=date.today(),
        name="BL Optimized 20-Stock",
        optimization_method="black_litterman",
        total_positions=20,
        total_weight=1.0
    )

    # Add positions
    position = PortfolioPosition(
        portfolio_id=portfolio.id,
        ticker="NFLX_US_EQ",
        yfinance_ticker="NFLX",
        weight=0.0239,
        sector="Communication Services",
        ...
    )
"""

import uuid
from datetime import date as date_type
from typing import List, Optional
from decimal import Decimal

from sqlalchemy import (
    String,
    Date,
    Integer,
    Numeric,
    ForeignKey,
    Index,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from optimizer.database.models.base import Base, TimestampMixin


class Portfolio(Base, TimestampMixin):
    """
    Main portfolio table storing optimization metadata.

    Tracks portfolio snapshots created by the optimizer with their
    configuration and aggregated metrics.
    """

    __tablename__ = "portfolios"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False
    )

    # Portfolio identification
    portfolio_date: Mapped[date_type] = mapped_column(
        Date,
        nullable=False,
        index=True,
        comment="Date this portfolio was created/optimized",
    )

    name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Optional portfolio name (e.g., 'BL Optimized 20-Stock')",
    )

    # Optimization configuration
    optimization_method: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Optimization method: 'black_litterman', 'concentrated', 'equal_weight', etc.",
    )

    used_baml_views: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        comment="Whether BAML AI views were used (Black-Litterman only)",
    )

    used_factor_priors: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        comment="Whether factor-based priors were used (fallback mode)",
    )

    # Portfolio metrics
    total_positions: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Number of positions in portfolio"
    )

    total_weight: Mapped[Decimal] = mapped_column(
        Numeric(10, 6),
        nullable=False,
        comment="Sum of all position weights (should be 1.0 for fully invested)",
    )

    risk_aversion: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
        comment="Risk aversion coefficient (delta) used in optimization",
    )

    tau: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 6),
        nullable=True,
        comment="Prior uncertainty parameter (Black-Litterman only)",
    )

    regime: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Macro regime at portfolio creation (EARLY_CYCLE, MID_CYCLE, etc.)",
    )

    # Additional metrics (stored as JSON for flexibility)
    metrics: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional portfolio metrics (expected return, volatility, Sharpe, etc.)",
    )

    # Relationships
    positions: Mapped[List["PortfolioPosition"]] = relationship(
        "PortfolioPosition", back_populates="portfolio", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_portfolios_date_method", "portfolio_date", "optimization_method"),
    )

    def __repr__(self) -> str:
        return (
            f"<Portfolio(id={self.id}, date={self.portfolio_date}, "
            f"method={self.optimization_method}, positions={self.total_positions})>"
        )


class PortfolioPosition(Base, TimestampMixin):
    """
    Individual position within a portfolio.

    Stores the optimized weight and metadata for each stock in the portfolio.
    """

    __tablename__ = "portfolio_positions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False
    )

    # Foreign key to portfolio
    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Ticker identification
    ticker: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="Trading212 ticker (e.g., 'NFLX_US_EQ')"
    )

    yfinance_ticker: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="Yahoo Finance ticker (e.g., 'NFLX')"
    )

    # Position weight (optimized allocation)
    weight: Mapped[Decimal] = mapped_column(
        Numeric(10, 6),
        nullable=False,
        comment="Optimized portfolio weight (0.0 to 1.0)",
    )

    # Company metadata
    company_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    sector: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True
    )

    industry: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    country: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    exchange: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Price at optimization time
    price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 6), nullable=True, comment="Stock price at portfolio creation"
    )

    # Risk metrics
    sharpe_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4), nullable=True
    )

    sortino_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4), nullable=True
    )

    volatility: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 6), nullable=True, comment="Annualized volatility"
    )

    alpha: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6), nullable=True)

    beta: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6), nullable=True)

    max_drawdown: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 6), nullable=True
    )

    annualized_return: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 6), nullable=True
    )

    # Signal metadata
    signal_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Reference to original stock signal (if applicable)",
    )

    signal_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Signal classification (LARGE_GAIN, SMALL_GAIN, etc.)",
    )

    confidence_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Conviction tier (from ConcentratedPortfolioBuilder)
    conviction_tier: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Conviction tier: 1 (highest), 2, 3 (lowest)"
    )

    # Original weight (before Black-Litterman optimization)
    original_weight: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 6),
        nullable=True,
        comment="Weight before optimization (from ConcentratedPortfolioBuilder)",
    )

    # Expected returns (from Black-Litterman)
    equilibrium_return: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 6), nullable=True, comment="Equilibrium expected return (prior)"
    )

    posterior_return: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 6), nullable=True, comment="Posterior expected return (after views)"
    )

    # Additional metadata
    selection_reason: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True, comment="Why this stock was selected/weighted"
    )

    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(
        "Portfolio", back_populates="positions"
    )

    # Indexes
    __table_args__ = (
        Index("ix_portfolio_positions_portfolio_ticker", "portfolio_id", "ticker"),
        Index("ix_portfolio_positions_sector", "sector"),
    )

    def __repr__(self) -> str:
        return (
            f"<PortfolioPosition(ticker={self.ticker}, weight={self.weight}, "
            f"sector={self.sector})>"
        )
