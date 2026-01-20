"""
Repository Layer - SQLAlchemy implementations of domain repository protocols.

These repositories:
- Implement domain protocols using SQLAlchemy
- Convert between SQLAlchemy models and domain DTOs
- Handle database transactions
- Abstract database details from business logic

Usage:
    from database.repositories import SignalRepositoryImpl, InstrumentRepositoryImpl

    # Create repository with database manager
    signal_repo = SignalRepositoryImpl(database_manager)

    # Use domain methods
    signals = signal_repo.get_large_gain_signals(signal_date=date.today())
"""

from optimizer.database.repositories.base import BaseRepository
from optimizer.database.repositories.signal_repository import SignalRepositoryImpl
from optimizer.database.repositories.instrument_repository import InstrumentRepositoryImpl
from optimizer.database.repositories.macro_regime_repository import MacroRegimeRepositoryImpl
from optimizer.database.repositories.portfolio_repository import PortfolioRepositoryImpl
from optimizer.database.repositories.universe_repository import UniverseRepositoryImpl

__all__ = [
    "BaseRepository",
    "SignalRepositoryImpl",
    "InstrumentRepositoryImpl",
    "MacroRegimeRepositoryImpl",
    "PortfolioRepositoryImpl",
    "UniverseRepositoryImpl",
]
