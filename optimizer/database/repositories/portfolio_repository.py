import logging
from datetime import date
from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, desc

from optimizer.database.database import DatabaseManager
from optimizer.database.models.portfolio import Portfolio, PortfolioPosition as DBPortfolioPosition
from optimizer.database.repositories.base import BaseRepository
from optimizer.domain.models.portfolio import PortfolioDTO, PositionDTO

logger = logging.getLogger(__name__)


class PortfolioRepositoryImpl(BaseRepository[Portfolio]):
    """
    SQLAlchemy implementation of the PortfolioRepository protocol.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize portfolio repository.
        """
        super().__init__(db_manager, Portfolio)

    def _position_to_dto(self, position: DBPortfolioPosition) -> PositionDTO:
        """Convert database position to DTO."""
        return PositionDTO(
            ticker=position.ticker,
            weight=float(position.weight) if position.weight else 0.0,
            instrument_id=(
                str(position.instrument_id)
                if hasattr(position, "instrument_id") and position.instrument_id
                else None
            ),
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
            annualized_return=(
                float(position.annualized_return) if position.annualized_return else None
            ),
            signal_type=position.signal_type,
            confidence_level=position.confidence_level or "medium",
            conviction_tier=position.conviction_tier or 2,
            data_quality_score=(
                float(position.data_quality_score)
                if hasattr(position, "data_quality_score") and position.data_quality_score
                else None
            ),
            original_weight=float(position.original_weight) if position.original_weight else None,
            equilibrium_return=(
                float(position.equilibrium_return) if position.equilibrium_return else None
            ),
            posterior_return=(
                float(position.posterior_return) if position.posterior_return else None
            ),
            selection_reason=position.selection_reason or "",
        )

    def _to_dto(self, portfolio: Portfolio) -> PortfolioDTO:
        """Convert database portfolio to DTO."""
        positions = [self._position_to_dto(p) for p in portfolio.positions]

        return PortfolioDTO(
            id=portfolio.id,
            portfolio_date=portfolio.portfolio_date,
            positions=positions,
            name=portfolio.name,
            optimization_method=portfolio.optimization_method or "black_litterman",
            used_baml_views=(
                portfolio.used_baml_views if hasattr(portfolio, "used_baml_views") else True
            ),
            used_factor_priors=(
                portfolio.used_factor_priors if hasattr(portfolio, "used_factor_priors") else False
            ),
            total_positions=portfolio.total_positions or len(positions),
            total_weight=float(portfolio.total_weight) if portfolio.total_weight else 1.0,
            risk_aversion=float(portfolio.risk_aversion) if portfolio.risk_aversion else None,
            tau=float(portfolio.tau) if portfolio.tau else None,
            regime=portfolio.regime,
            metrics=portfolio.metrics if portfolio.metrics else {},
            created_at=portfolio.created_at,
            updated_at=portfolio.updated_at if hasattr(portfolio, "updated_at") else None,
        )

    def save(self, portfolio_dto: PortfolioDTO) -> UUID:
        """
        Save a new portfolio to the database.
        """
        with self._get_session() as session:
            # Create portfolio record
            portfolio = Portfolio(
                id=portfolio_dto.id,
                portfolio_date=portfolio_dto.portfolio_date,
                name=portfolio_dto.name,
                optimization_method=portfolio_dto.optimization_method,
                used_baml_views=portfolio_dto.used_baml_views,
                used_factor_priors=portfolio_dto.used_factor_priors,
                total_positions=portfolio_dto.total_positions,
                total_weight=Decimal(str(portfolio_dto.total_weight)),
                risk_aversion=(
                    Decimal(str(portfolio_dto.risk_aversion))
                    if portfolio_dto.risk_aversion
                    else None
                ),
                tau=Decimal(str(portfolio_dto.tau)) if portfolio_dto.tau else None,
                regime=portfolio_dto.regime,
                metrics=portfolio_dto.metrics,
            )

            session.add(portfolio)
            session.flush()  # Get portfolio ID

            # Create position records
            for pos_dto in portfolio_dto.positions:
                position = DBPortfolioPosition(
                    portfolio_id=portfolio.id,
                    ticker=pos_dto.ticker,
                    yfinance_ticker=pos_dto.yfinance_ticker,
                    weight=Decimal(str(pos_dto.weight)),
                    company_name=pos_dto.company_name,
                    sector=pos_dto.sector,
                    industry=pos_dto.industry,
                    country=pos_dto.country,
                    exchange=pos_dto.exchange,
                    price=Decimal(str(pos_dto.price)) if pos_dto.price else None,
                    sharpe_ratio=(
                        Decimal(str(pos_dto.sharpe_ratio)) if pos_dto.sharpe_ratio else None
                    ),
                    sortino_ratio=(
                        Decimal(str(pos_dto.sortino_ratio)) if pos_dto.sortino_ratio else None
                    ),
                    volatility=Decimal(str(pos_dto.volatility)) if pos_dto.volatility else None,
                    alpha=Decimal(str(pos_dto.alpha)) if pos_dto.alpha else None,
                    beta=Decimal(str(pos_dto.beta)) if pos_dto.beta else None,
                    max_drawdown=(
                        Decimal(str(pos_dto.max_drawdown)) if pos_dto.max_drawdown else None
                    ),
                    annualized_return=(
                        Decimal(str(pos_dto.annualized_return))
                        if pos_dto.annualized_return
                        else None
                    ),
                    signal_id=pos_dto.signal_id,
                    signal_type=pos_dto.signal_type,
                    confidence_level=pos_dto.confidence_level,
                    conviction_tier=pos_dto.conviction_tier,
                    original_weight=(
                        Decimal(str(pos_dto.original_weight)) if pos_dto.original_weight else None
                    ),
                    equilibrium_return=(
                        Decimal(str(pos_dto.equilibrium_return))
                        if pos_dto.equilibrium_return
                        else None
                    ),
                    posterior_return=(
                        Decimal(str(pos_dto.posterior_return)) if pos_dto.posterior_return else None
                    ),
                    selection_reason=pos_dto.selection_reason,
                )
                session.add(position)

            session.commit()

            logger.info(
                f"Saved portfolio {portfolio.id} with {len(portfolio_dto.positions)} positions"
            )
            return portfolio.id

    def get_by_id(self, portfolio_id: UUID) -> Optional[PortfolioDTO]:
        """
        Fetch a portfolio by its ID.
        """
        with self._get_session() as session:
            query = select(Portfolio).where(Portfolio.id == portfolio_id)
            portfolio = session.execute(query).scalar_one_or_none()

            if portfolio:
                return self._to_dto(portfolio)
            return None

    def get_by_date(self, portfolio_date: date) -> List[PortfolioDTO]:
        """
        Fetch all portfolios for a specific date.

        Args:
            portfolio_date: Portfolio date

        Returns:
            List of PortfolioDTO objects
        """
        with self._get_session() as session:
            query = (
                select(Portfolio)
                .where(Portfolio.portfolio_date == portfolio_date)
                .order_by(desc(Portfolio.created_at))
            )

            results = session.execute(query).scalars().all()
            return [self._to_dto(p) for p in results]

    def get_latest(self, limit: int = 10) -> List[PortfolioDTO]:
        """
        Fetch the most recent portfolios.
        """
        with self._get_session() as session:
            query = select(Portfolio).order_by(desc(Portfolio.created_at)).limit(limit)

            results = session.execute(query).scalars().all()
            return [self._to_dto(p) for p in results]

    def update_positions(
        self,
        portfolio_id: UUID,
        positions: List[PositionDTO],
    ) -> None:
        """
        Update positions for an existing portfolio.
        """
        with self._get_session() as session:
            # Delete existing positions
            session.query(DBPortfolioPosition).filter(
                DBPortfolioPosition.portfolio_id == portfolio_id
            ).delete()

            # Add new positions
            for pos_dto in positions:
                position = DBPortfolioPosition(
                    portfolio_id=portfolio_id,
                    ticker=pos_dto.ticker,
                    weight=Decimal(str(pos_dto.weight)),
                    yfinance_ticker=pos_dto.yfinance_ticker,
                    company_name=pos_dto.company_name,
                    sector=pos_dto.sector,
                    industry=pos_dto.industry,
                    country=pos_dto.country,
                    exchange=pos_dto.exchange,
                )
                session.add(position)

            # Update portfolio totals
            portfolio = session.get(Portfolio, portfolio_id)
            if portfolio:
                portfolio.total_positions = len(positions)
                portfolio.total_weight = Decimal(str(sum(p.weight for p in positions)))

            session.commit()
            logger.info(f"Updated {len(positions)} positions for portfolio {portfolio_id}")

    def delete(self, portfolio_id: UUID) -> bool:
        """
        Delete a portfolio and its positions.
        """
        with self._get_session() as session:
            portfolio = session.get(Portfolio, portfolio_id)
            if portfolio:
                session.delete(portfolio)  # Cascade deletes positions
                session.commit()
                logger.info(f"Deleted portfolio {portfolio_id}")
                return True
            return False
