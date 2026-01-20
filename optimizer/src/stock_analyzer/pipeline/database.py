import uuid
from datetime import date as date_type, datetime, timedelta
from typing import Optional, List, Union, Set

from sqlalchemy import select, and_
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from optimizer.src.yfinance import YFinanceClient

from optimizer.database.models.universe import Instrument
from optimizer.database.models.stock_signals import StockSignal
from baml_client.types import StockSignalOutput

from .utils import (
    map_signal_type_to_enum,
    map_confidence_level_to_enum,
    map_risk_level_to_enum,
    safe_float,
)


def get_active_instruments(
    session: Session, max_instruments: Optional[int] = None
) -> List[Instrument]:
    """Fetch active instruments from database."""
    query = (
        select(Instrument)
        .where(and_(Instrument.is_active == True, Instrument.yfinance_ticker.isnot(None)))
        .order_by(Instrument.ticker)
    )

    if max_instruments:
        query = query.limit(max_instruments)

    instruments = session.execute(query).scalars().all()
    return list(instruments)


def get_processed_instrument_ids(session: Session, signal_date: date_type) -> Set[uuid.UUID]:
    """Get instrument IDs that already have signals for the given date."""
    query = select(StockSignal.instrument_id).where(StockSignal.signal_date == signal_date)
    results = session.execute(query).scalars().all()
    return set(results)


def check_for_incomplete_run(
    session: Session, total_instruments: int
) -> Optional[tuple[date_type, int]]:
    """Check for incomplete runs from today or yesterday."""
    today = date_type.today()
    yesterday = today - timedelta(days=1)

    # Check today first
    today_processed = get_processed_instrument_ids(session, today)
    today_count = len(today_processed)

    if 0 < today_count < total_instruments:
        return (today, today_count)

    # Check yesterday
    yesterday_processed = get_processed_instrument_ids(session, yesterday)
    yesterday_count = len(yesterday_processed)

    if 0 < yesterday_count < total_instruments:
        return (yesterday, yesterday_count)

    # Check if today is complete or if yesterday is complete
    if today_count >= total_instruments or yesterday_count >= total_instruments:
        return None

    # No signals found
    return None


def check_signal_exists(
    session: Session, instrument_id: Union[uuid.UUID, str], signal_date: date_type
) -> Optional[StockSignal]:
    """Check if signal already exists for instrument and date."""
    query = select(StockSignal).where(
        and_(
            StockSignal.instrument_id == instrument_id,
            StockSignal.signal_date == signal_date,
        )
    )
    return session.execute(query).scalar_one_or_none()


def fetch_sector_industry(yfinance_ticker: str) -> tuple[Optional[str], Optional[str]]:
    """Fetch sector and industry from yfinance."""
    try:
        client = YFinanceClient.get_instance()
        info = client.fetch_info(yfinance_ticker)
        if info is None:
            return None, None
        return info.get("sector"), info.get("industry")
    except Exception:
        return None, None


def save_signal(
    session: Session,
    instrument_id: Union[uuid.UUID, str],
    signal_output: StockSignalOutput,
    technical_metrics: Optional[dict],
    update_if_exists: bool = True,
    instrument: Optional[Instrument] = None,
) -> bool:
    """Save or update signal in database."""
    try:
        # Get instrument data for denormalization
        if instrument is None:
            instrument = session.get(Instrument, instrument_id)
        else:
            instrument = session.merge(instrument, load=False)

        # Fetch sector/industry
        sector, industry = None, None
        if instrument and instrument.yfinance_ticker:
            sector, industry = fetch_sector_industry(instrument.yfinance_ticker)

        # Prepare denormalized fields
        ticker = instrument.ticker if instrument else None
        yfinance_ticker = instrument.yfinance_ticker if instrument else None
        exchange_name = (
            instrument.exchange.exchange_name if (instrument and instrument.exchange) else None
        )

        # Parse signal date
        signal_date = datetime.fromisoformat(signal_output.signal_date).date()

        # Map enums
        db_signal_type = map_signal_type_to_enum(signal_output.signal_type)
        db_confidence = map_confidence_level_to_enum(signal_output.confidence_level)

        # Check if signal exists
        existing_signal = check_signal_exists(session, instrument_id, signal_date)

        if existing_signal:
            if not update_if_exists:
                return False

            # Update existing signal
            _update_signal_fields(
                existing_signal,
                signal_output,
                technical_metrics,
                db_signal_type,
                db_confidence,
                ticker,
                yfinance_ticker,
                exchange_name,
                sector,
                industry,
            )
            return True
        else:
            # Create new signal
            new_signal = _create_new_signal(
                instrument_id,
                signal_date,
                signal_output,
                technical_metrics,
                db_signal_type,
                db_confidence,
                ticker,
                yfinance_ticker,
                exchange_name,
                sector,
                industry,
            )
            session.add(new_signal)
            return True

    except IntegrityError:
        session.rollback()
        return False

    except Exception:
        session.rollback()
        return False


def _update_signal_fields(
    signal: StockSignal,
    signal_output: StockSignalOutput,
    technical_metrics: Optional[dict],
    db_signal_type,
    db_confidence,
    ticker,
    yfinance_ticker,
    exchange_name,
    sector,
    industry,
) -> None:
    """Update all fields of an existing signal."""
    # Basic signal data
    signal.signal_type = db_signal_type
    signal.ticker = ticker
    signal.yfinance_ticker = yfinance_ticker
    signal.exchange_name = exchange_name
    signal.sector = sector
    signal.industry = industry
    signal.confidence_level = db_confidence

    # Price data
    signal.close_price = safe_float(signal_output.close_price)
    signal.open_price = safe_float(signal_output.open_price)
    signal.daily_return = safe_float(signal_output.daily_return)
    signal.volume = safe_float(signal_output.volume)

    # Technical indicators
    signal.volatility = safe_float(signal_output.volatility)
    signal.rsi = safe_float(signal_output.rsi)
    signal.data_quality_score = float(signal_output.data_quality_score)

    # Signal drivers
    signal.valuation_score = float(signal_output.signal_drivers.valuation_score)
    signal.momentum_score = float(signal_output.signal_drivers.momentum_score)
    signal.quality_score = float(signal_output.signal_drivers.quality_score)
    signal.growth_score = float(signal_output.signal_drivers.growth_score)
    signal.technical_score = safe_float(signal_output.signal_drivers.technical_score)

    # Risk factors
    signal.volatility_level = map_risk_level_to_enum(signal_output.risk_factors.volatility_level)
    signal.beta_risk = map_risk_level_to_enum(signal_output.risk_factors.beta_risk)
    signal.debt_risk = map_risk_level_to_enum(signal_output.risk_factors.debt_risk)
    signal.liquidity_risk = map_risk_level_to_enum(signal_output.risk_factors.liquidity_risk)

    # Technical metrics
    if technical_metrics:
        signal.annualized_return = technical_metrics.get("annualized_return")
        signal.sharpe_ratio = technical_metrics.get("sharpe_ratio")
        signal.sortino_ratio = technical_metrics.get("sortino_ratio")
        signal.max_drawdown = technical_metrics.get("max_drawdown")
        signal.calmar_ratio = technical_metrics.get("calmar_ratio")
        signal.beta = technical_metrics.get("beta")
        signal.alpha = technical_metrics.get("alpha")
        signal.r_squared = technical_metrics.get("r_squared")
        signal.information_ratio = technical_metrics.get("information_ratio")
        signal.benchmark_return = technical_metrics.get("benchmark_return")

    # Price targets
    signal.upside_potential_pct = safe_float(signal_output.upside_potential_pct)
    signal.downside_risk_pct = safe_float(signal_output.downside_risk_pct)


def _create_new_signal(
    instrument_id,
    signal_date,
    signal_output,
    technical_metrics,
    db_signal_type,
    db_confidence,
    ticker,
    yfinance_ticker,
    exchange_name,
    sector,
    industry,
) -> StockSignal:
    """Create a new StockSignal object."""
    return StockSignal(
        instrument_id=instrument_id,
        signal_date=signal_date,
        signal_type=db_signal_type,
        # Denormalized instrument data
        ticker=ticker,
        yfinance_ticker=yfinance_ticker,
        exchange_name=exchange_name,
        sector=sector,
        industry=industry,
        confidence_level=db_confidence,
        # Price data
        close_price=safe_float(signal_output.close_price),
        open_price=safe_float(signal_output.open_price),
        daily_return=safe_float(signal_output.daily_return),
        volume=safe_float(signal_output.volume),
        # Technical indicators
        volatility=safe_float(signal_output.volatility),
        rsi=safe_float(signal_output.rsi),
        data_quality_score=float(signal_output.data_quality_score),
        # Signal drivers
        valuation_score=float(signal_output.signal_drivers.valuation_score),
        momentum_score=float(signal_output.signal_drivers.momentum_score),
        quality_score=float(signal_output.signal_drivers.quality_score),
        growth_score=float(signal_output.signal_drivers.growth_score),
        technical_score=safe_float(signal_output.signal_drivers.technical_score),
        # Risk factors
        volatility_level=map_risk_level_to_enum(signal_output.risk_factors.volatility_level),
        beta_risk=map_risk_level_to_enum(signal_output.risk_factors.beta_risk),
        debt_risk=map_risk_level_to_enum(signal_output.risk_factors.debt_risk),
        liquidity_risk=map_risk_level_to_enum(signal_output.risk_factors.liquidity_risk),
        # Technical metrics
        annualized_return=(
            technical_metrics.get("annualized_return") if technical_metrics else None
        ),
        sharpe_ratio=(technical_metrics.get("sharpe_ratio") if technical_metrics else None),
        sortino_ratio=(technical_metrics.get("sortino_ratio") if technical_metrics else None),
        max_drawdown=(technical_metrics.get("max_drawdown") if technical_metrics else None),
        calmar_ratio=(technical_metrics.get("calmar_ratio") if technical_metrics else None),
        beta=technical_metrics.get("beta") if technical_metrics else None,
        alpha=technical_metrics.get("alpha") if technical_metrics else None,
        r_squared=technical_metrics.get("r_squared") if technical_metrics else None,
        information_ratio=(
            technical_metrics.get("information_ratio") if technical_metrics else None
        ),
        benchmark_return=(technical_metrics.get("benchmark_return") if technical_metrics else None),
        # Price targets
        upside_potential_pct=safe_float(signal_output.upside_potential_pct),
        downside_risk_pct=safe_float(signal_output.downside_risk_pct),
    )
