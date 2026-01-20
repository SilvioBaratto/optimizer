"""
Portfolio Mapper - Convert between Portfolio models and DTOs.
"""

from decimal import Decimal
from typing import List

from optimizer.database.models.portfolio import Portfolio, PortfolioPosition
from optimizer.domain.models.portfolio import PortfolioDTO, PositionDTO


class PortfolioMapper:
    """
    Static mapper class for Portfolio conversions.

    Follows the Data Mapper pattern to separate domain logic from persistence.
    """

    @staticmethod
    def position_to_dto(position: PortfolioPosition) -> PositionDTO:
        """
        Convert SQLAlchemy PortfolioPosition model to PositionDTO.

        Args:
            position: Database PortfolioPosition model

        Returns:
            PositionDTO domain object
        """
        return PositionDTO(
            ticker=position.ticker,
            weight=float(position.weight) if position.weight else 0.0,
            instrument_id=str(position.instrument_id) if hasattr(position, 'instrument_id') and position.instrument_id else None,
            signal_id=str(position.signal_id) if position.signal_id else None,
            yfinance_ticker=position.yfinance_ticker,
            company_name=position.company_name,
            sector=position.sector,
            industry=position.industry,
            country=position.country or "USA",
            exchange=position.exchange,
            price=float(position.price) if position.price else None,
            sharpe_ratio=float(position.sharpe_ratio) if position.sharpe_ratio else None,
            sortino_ratio=float(position.sortino_ratio) if position.sortino_ratio else None,
            volatility=float(position.volatility) if position.volatility else None,
            alpha=float(position.alpha) if position.alpha else None,
            beta=float(position.beta) if position.beta else None,
            max_drawdown=float(position.max_drawdown) if position.max_drawdown else None,
            annualized_return=float(position.annualized_return) if position.annualized_return else None,
            signal_type=position.signal_type,
            confidence_level=position.confidence_level or "medium",
            conviction_tier=position.conviction_tier or 2,
            data_quality_score=float(position.data_quality_score) if hasattr(position, 'data_quality_score') and position.data_quality_score else None,
            original_weight=float(position.original_weight) if position.original_weight else None,
            equilibrium_return=float(position.equilibrium_return) if position.equilibrium_return else None,
            posterior_return=float(position.posterior_return) if position.posterior_return else None,
            selection_reason=position.selection_reason or "",
        )

    @staticmethod
    def to_dto(portfolio: Portfolio) -> PortfolioDTO:
        """
        Convert SQLAlchemy Portfolio model to PortfolioDTO.

        Args:
            portfolio: Database Portfolio model

        Returns:
            PortfolioDTO domain object
        """
        positions = [PortfolioMapper.position_to_dto(p) for p in portfolio.positions]

        return PortfolioDTO(
            id=portfolio.id,
            portfolio_date=portfolio.portfolio_date,
            positions=positions,
            name=portfolio.name,
            optimization_method=portfolio.optimization_method or "black_litterman",
            used_baml_views=portfolio.used_baml_views if hasattr(portfolio, 'used_baml_views') else True,
            used_factor_priors=portfolio.used_factor_priors if hasattr(portfolio, 'used_factor_priors') else False,
            total_positions=portfolio.total_positions or len(positions),
            total_weight=float(portfolio.total_weight) if portfolio.total_weight else 1.0,
            risk_aversion=float(portfolio.risk_aversion) if portfolio.risk_aversion else None,
            tau=float(portfolio.tau) if portfolio.tau else None,
            regime=portfolio.regime,
            metrics=portfolio.metrics if portfolio.metrics else {},
            created_at=portfolio.created_at,
            updated_at=portfolio.updated_at if hasattr(portfolio, 'updated_at') else None,
        )

    @staticmethod
    def to_dto_batch(portfolios: List[Portfolio]) -> List[PortfolioDTO]:
        """
        Convert multiple portfolios to DTOs.

        Args:
            portfolios: List of database Portfolio models

        Returns:
            List of PortfolioDTO objects
        """
        return [PortfolioMapper.to_dto(p) for p in portfolios]

    @staticmethod
    def position_from_dto(dto: PositionDTO, portfolio_id) -> PortfolioPosition:
        """
        Convert PositionDTO to SQLAlchemy PortfolioPosition model.

        Args:
            dto: PositionDTO domain object
            portfolio_id: UUID of parent portfolio

        Returns:
            Database PortfolioPosition model
        """
        return PortfolioPosition(
            portfolio_id=portfolio_id,
            ticker=dto.ticker,
            yfinance_ticker=dto.yfinance_ticker,
            weight=Decimal(str(dto.weight)),
            company_name=dto.company_name,
            sector=dto.sector,
            industry=dto.industry,
            country=dto.country,
            exchange=dto.exchange,
            price=Decimal(str(dto.price)) if dto.price else None,
            sharpe_ratio=Decimal(str(dto.sharpe_ratio)) if dto.sharpe_ratio else None,
            sortino_ratio=Decimal(str(dto.sortino_ratio)) if dto.sortino_ratio else None,
            volatility=Decimal(str(dto.volatility)) if dto.volatility else None,
            alpha=Decimal(str(dto.alpha)) if dto.alpha else None,
            beta=Decimal(str(dto.beta)) if dto.beta else None,
            max_drawdown=Decimal(str(dto.max_drawdown)) if dto.max_drawdown else None,
            annualized_return=Decimal(str(dto.annualized_return)) if dto.annualized_return else None,
            signal_id=dto.signal_id,
            signal_type=dto.signal_type,
            confidence_level=dto.confidence_level,
            conviction_tier=dto.conviction_tier,
            original_weight=Decimal(str(dto.original_weight)) if dto.original_weight else None,
            equilibrium_return=Decimal(str(dto.equilibrium_return)) if dto.equilibrium_return else None,
            posterior_return=Decimal(str(dto.posterior_return)) if dto.posterior_return else None,
            selection_reason=dto.selection_reason,
        )
