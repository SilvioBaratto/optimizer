"""
Signal Mapper - Convert between StockSignal models and DTOs.
"""

from typing import Optional, List, Tuple

from optimizer.database.models.stock_signals import StockSignal, SignalEnum, ConfidenceLevelEnum, RiskLevelEnum
from optimizer.database.models.universe import Instrument
from optimizer.domain.models.stock_signal import (
    StockSignalDTO,
    SignalType,
    ConfidenceLevel,
    RiskLevel,
)
from optimizer.domain.models.instrument import InstrumentDTO


class SignalMapper:
    """
    Static mapper class for StockSignal conversions.

    Follows the Data Mapper pattern to separate domain logic from persistence.
    """

    @staticmethod
    def to_dto(signal: StockSignal) -> StockSignalDTO:
        """
        Convert SQLAlchemy StockSignal model to StockSignalDTO.

        Args:
            signal: Database StockSignal model

        Returns:
            StockSignalDTO domain object
        """
        # Convert signal type
        signal_type = SignalMapper._convert_signal_type(signal.signal_type)

        # Convert confidence level
        confidence = SignalMapper._convert_confidence(signal.confidence_level)

        # Convert risk levels
        volatility_level = SignalMapper._convert_risk_level(signal.volatility_level)
        beta_risk = SignalMapper._convert_risk_level(signal.beta_risk)
        debt_risk = SignalMapper._convert_risk_level(signal.debt_risk)
        liquidity_risk = SignalMapper._convert_risk_level(signal.liquidity_risk)

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
            volatility_level=volatility_level,
            beta_risk=beta_risk,
            debt_risk=debt_risk,
            liquidity_risk=liquidity_risk,
            confidence_level=confidence,
            data_quality_score=signal.data_quality_score,
            upside_potential_pct=signal.upside_potential_pct,
            downside_risk_pct=signal.downside_risk_pct,
        )

    @staticmethod
    def to_dto_with_instrument(
        signal: StockSignal,
        instrument: Instrument
    ) -> Tuple[StockSignalDTO, InstrumentDTO]:
        """
        Convert signal and instrument to DTOs.

        Args:
            signal: Database StockSignal model
            instrument: Database Instrument model

        Returns:
            Tuple of (StockSignalDTO, InstrumentDTO)
        """
        from mappers.instrument_mapper import InstrumentMapper

        signal_dto = SignalMapper.to_dto(signal)
        instrument_dto = InstrumentMapper.to_dto(instrument)

        return signal_dto, instrument_dto

    @staticmethod
    def to_dto_batch(
        signals: List[StockSignal]
    ) -> List[StockSignalDTO]:
        """
        Convert multiple signals to DTOs.

        Args:
            signals: List of database StockSignal models

        Returns:
            List of StockSignalDTO objects
        """
        return [SignalMapper.to_dto(s) for s in signals]

    @staticmethod
    def to_dto_batch_with_instruments(
        signals_with_instruments: List[Tuple[StockSignal, Instrument]]
    ) -> List[Tuple[StockSignalDTO, InstrumentDTO]]:
        """
        Convert multiple signal-instrument pairs to DTOs.

        Args:
            signals_with_instruments: List of (StockSignal, Instrument) tuples

        Returns:
            List of (StockSignalDTO, InstrumentDTO) tuples
        """
        return [
            SignalMapper.to_dto_with_instrument(signal, instrument)
            for signal, instrument in signals_with_instruments
        ]

    @staticmethod
    def _convert_signal_type(db_type: Optional[SignalEnum]) -> SignalType:
        """Convert database signal type to domain enum."""
        if db_type is None:
            return SignalType.NEUTRAL

        mapping = {
            SignalEnum.LARGE_DECLINE: SignalType.LARGE_DECLINE,
            SignalEnum.SMALL_DECLINE: SignalType.SMALL_DECLINE,
            SignalEnum.NEUTRAL: SignalType.NEUTRAL,
            SignalEnum.SMALL_GAIN: SignalType.SMALL_GAIN,
            SignalEnum.LARGE_GAIN: SignalType.LARGE_GAIN,
        }

        return mapping.get(db_type, SignalType.NEUTRAL)

    @staticmethod
    def _convert_confidence(db_confidence: Optional[ConfidenceLevelEnum]) -> Optional[ConfidenceLevel]:
        """Convert database confidence to domain enum."""
        if db_confidence is None:
            return None

        mapping = {
            ConfidenceLevelEnum.LOW: ConfidenceLevel.LOW,
            ConfidenceLevelEnum.MEDIUM: ConfidenceLevel.MEDIUM,
            ConfidenceLevelEnum.HIGH: ConfidenceLevel.HIGH,
        }

        return mapping.get(db_confidence, ConfidenceLevel.MEDIUM)

    @staticmethod
    def _convert_risk_level(db_risk: Optional[RiskLevelEnum]) -> Optional[RiskLevel]:
        """Convert database risk level to domain enum."""
        if db_risk is None:
            return None

        mapping = {
            RiskLevelEnum.low: RiskLevel.LOW,
            RiskLevelEnum.medium: RiskLevel.MEDIUM,
            RiskLevelEnum.high: RiskLevel.HIGH,
            RiskLevelEnum.unknown: RiskLevel.UNKNOWN,
        }

        return mapping.get(db_risk, RiskLevel.UNKNOWN)

    @staticmethod
    def dto_to_baml_signal(dto: StockSignalDTO) -> dict:
        """
        Convert DTO to BAML signal data format.

        Used for passing signals to BAML view generation.

        Args:
            dto: StockSignalDTO domain object

        Returns:
            Dictionary in BAML StockSignalData format
        """
        return {
            "ticker": dto.ticker,
            "yfinance_ticker": dto.yfinance_ticker or dto.ticker,
            "sector": dto.sector,
            "industry": dto.industry,
            "signal_type": dto.signal_type.value.upper(),
            "confidence_level": dto.confidence_level.value.upper() if dto.confidence_level else "MEDIUM",
            "valuation_score": dto.valuation_score,
            "momentum_score": dto.momentum_score,
            "quality_score": dto.quality_score,
            "growth_score": dto.growth_score,
            "technical_score": dto.technical_score,
            "annualized_return": dto.annualized_return,
            "volatility": dto.volatility,
            "sharpe_ratio": dto.sharpe_ratio,
            "beta": dto.beta,
            "alpha": dto.alpha,
            "volatility_level": dto.volatility_level.value if dto.volatility_level else None,
            "beta_risk": dto.beta_risk.value if dto.beta_risk else None,
            "debt_risk": dto.debt_risk.value if dto.debt_risk else None,
            "liquidity_risk": dto.liquidity_risk.value if dto.liquidity_risk else None,
            "upside_potential_pct": dto.upside_potential_pct,
            "downside_risk_pct": dto.downside_risk_pct,
            "close_price": dto.close_price,
            "daily_return": dto.daily_return,
        }
