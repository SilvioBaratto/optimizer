import logging
from datetime import date
from typing import List, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, func, desc
from sqlalchemy.orm import joinedload

from optimizer.database.database import DatabaseManager
from optimizer.database.models.stock_signals import StockSignal, SignalEnum
from optimizer.database.models.universe import Instrument
from optimizer.database.repositories.base import BaseRepository
from optimizer.domain.models.stock_signal import (
    StockSignalDTO,
    SignalType,
    ConfidenceLevel,
    RiskLevel,
)

logger = logging.getLogger(__name__)


class SignalRepositoryImpl(BaseRepository[StockSignal]):
    """
    SQLAlchemy implementation of the SignalRepository protocol.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize signal repository.
        """
        super().__init__(db_manager, StockSignal)

    def _to_dto(self, signal: StockSignal) -> StockSignalDTO:
        """
        Convert SQLAlchemy model to domain DTO.
        """
        # Convert enum values
        signal_type = (
            SignalType(signal.signal_type.value) if signal.signal_type else SignalType.NEUTRAL
        )

        confidence = None
        if signal.confidence_level:
            try:
                confidence = ConfidenceLevel(signal.confidence_level.value)
            except (ValueError, AttributeError):
                confidence = ConfidenceLevel.MEDIUM

        # Convert risk levels
        def to_risk_level(db_risk) -> Optional[RiskLevel]:
            if db_risk is None:
                return None
            try:
                return RiskLevel(db_risk.value)
            except (ValueError, AttributeError):
                return RiskLevel.UNKNOWN

        return StockSignalDTO(
            id=signal.id,
            instrument_id=signal.instrument_id,
            ticker=signal.ticker or "",
            signal_date=signal.signal_date,
            signal_type=signal_type,
            yfinance_ticker=signal.yfinance_ticker,
            exchange_name=signal.exchange_name,
            sector=signal.sector,
            industry=signal.industry,
            close_price=signal.close_price,
            open_price=signal.open_price,
            daily_return=signal.daily_return,
            volume=signal.volume,
            annualized_return=signal.annualized_return,
            volatility=signal.volatility,
            sharpe_ratio=signal.sharpe_ratio,
            sortino_ratio=signal.sortino_ratio,
            max_drawdown=signal.max_drawdown,
            calmar_ratio=signal.calmar_ratio,
            beta=signal.beta,
            alpha=signal.alpha,
            r_squared=signal.r_squared,
            information_ratio=signal.information_ratio,
            benchmark_return=signal.benchmark_return,
            rsi=signal.rsi,
            valuation_score=signal.valuation_score,
            momentum_score=signal.momentum_score,
            quality_score=signal.quality_score,
            growth_score=signal.growth_score,
            technical_score=signal.technical_score,
            volatility_level=to_risk_level(signal.volatility_level),
            beta_risk=to_risk_level(signal.beta_risk),
            debt_risk=to_risk_level(signal.debt_risk),
            liquidity_risk=to_risk_level(signal.liquidity_risk),
            confidence_level=confidence,
            data_quality_score=signal.data_quality_score,
            upside_potential_pct=signal.upside_potential_pct,
            downside_risk_pct=signal.downside_risk_pct,
        )

    def get_by_date(
        self,
        signal_date: date,
        signal_types: Optional[List[str]] = None,
        min_sharpe: Optional[float] = None,
        sector: Optional[str] = None,
    ) -> List[StockSignalDTO]:
        """
        Fetch signals for a specific date with optional filters.
        """
        with self._get_session() as session:
            query = select(StockSignal).where(StockSignal.signal_date == signal_date)

            if signal_types:
                # Convert string types to enum values
                enum_types = [SignalEnum(t.lower()) for t in signal_types]
                query = query.where(StockSignal.signal_type.in_(enum_types))

            if min_sharpe is not None:
                query = query.where(StockSignal.sharpe_ratio >= min_sharpe)

            if sector:
                query = query.where(StockSignal.sector == sector)

            results = session.execute(query).scalars().all()
            return [self._to_dto(s) for s in results]

    def get_latest_date(self, signal_type: Optional[str] = None) -> Optional[date]:
        """
        Get the most recent signal date in the database.
        """
        with self._get_session() as session:
            query = select(func.max(StockSignal.signal_date))

            if signal_type:
                enum_type = SignalEnum(signal_type.lower())
                query = query.where(StockSignal.signal_type == enum_type)

            result = session.execute(query).scalar_one_or_none()
            return result

    def get_by_tickers(
        self,
        tickers: List[str],
        signal_date: Optional[date] = None,
    ) -> Dict[str, StockSignalDTO]:
        """
        Fetch signals for specific tickers.
        """
        if signal_date is None:
            signal_date = self.get_latest_date()
            if signal_date is None:
                return {}

        with self._get_session() as session:
            # First try exact date
            query = (
                select(StockSignal)
                .where(StockSignal.ticker.in_(tickers))
                .where(StockSignal.signal_date == signal_date)
            )

            results = session.execute(query).scalars().all()
            signal_dict = {s.ticker: self._to_dto(s) for s in results}

            # For missing tickers, get most recent signal
            missing = set(tickers) - set(signal_dict.keys())
            if missing:
                for ticker in missing:
                    recent_query = (
                        select(StockSignal)
                        .where(StockSignal.ticker == ticker)
                        .order_by(desc(StockSignal.signal_date))
                        .limit(1)
                    )
                    result = session.execute(recent_query).scalar_one_or_none()
                    if result:
                        signal_dict[ticker] = self._to_dto(result)

            return signal_dict

    def get_large_gain_signals(
        self,
        signal_date: Optional[date] = None,
        min_sharpe: Optional[float] = None,
        max_volatility: Optional[float] = None,
    ) -> List[Tuple[StockSignalDTO, "InstrumentDTO"]]:
        """
        Fetch LARGE_GAIN signals with quality filters.
        """
        from mappers.signal_mapper import SignalMapper
        from mappers.instrument_mapper import InstrumentMapper

        if signal_date is None:
            signal_date = self.get_latest_date(signal_type="large_gain")
            if signal_date is None:
                logger.warning("No LARGE_GAIN signals found in database")
                return []

        with self._get_session() as session:
            query = (
                select(StockSignal, Instrument)
                .join(Instrument, StockSignal.instrument_id == Instrument.id)
                .where(StockSignal.signal_type == SignalEnum.LARGE_GAIN)
                .where(StockSignal.signal_date == signal_date)
            )

            if min_sharpe is not None:
                query = query.where(StockSignal.sharpe_ratio >= min_sharpe)

            if max_volatility is not None:
                query = query.where(StockSignal.volatility <= max_volatility)

            results = session.execute(query).all()

            # Convert to DTOs
            dtos = []
            for signal, instrument in results:
                signal_dto = self._to_dto(signal)
                instrument_dto = InstrumentMapper.to_dto(instrument)
                dtos.append((signal_dto, instrument_dto))

            logger.info(f"Found {len(dtos)} LARGE_GAIN signals for {signal_date}")
            return dtos

    def get_signals_with_instruments(
        self,
        signal_date: date,
        signal_type: SignalEnum = SignalEnum.LARGE_GAIN,
    ) -> List[Tuple[StockSignal, Instrument]]:
        """
        Fetch signals with joined instrument data (returns ORM objects).
        """
        with self._get_session() as session:
            query = (
                select(StockSignal, Instrument)
                .join(Instrument, StockSignal.instrument_id == Instrument.id)
                .options(joinedload(StockSignal.instrument))
                .where(StockSignal.signal_type == signal_type)
                .where(StockSignal.signal_date == signal_date)
            )

            results = session.execute(query).all()
            return [(signal, instrument) for signal, instrument in results]

    def count_by_date(self, signal_date: date) -> Dict[str, int]:
        """
        Count signals by type for a specific date.

        Args:
            signal_date: Date to count signals for

        Returns:
            Dictionary mapping signal_type -> count
        """
        with self._get_session() as session:
            query = (
                select(StockSignal.signal_type, func.count(StockSignal.id))
                .where(StockSignal.signal_date == signal_date)
                .group_by(StockSignal.signal_type)
            )

            results = session.execute(query).all()
            return {signal_type.value: count for signal_type, count in results}
