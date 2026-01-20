"""Database models for Optimizer API"""

# Import base classes
from optimizer.database.models.base import Base

# Import universe models
from optimizer.database.models.universe import Exchange, Instrument

# Import stock signal models
from optimizer.database.models.stock_signals import StockSignal, SignalEnum

# Import signal distribution models
from optimizer.database.models.signal_distribution import SignalDistribution

# Import macro regime models
from optimizer.database.models.macro_regime import (
    # Models
    MacroAnalysisRun,
    MarketIndicators,
    CountryRegimeAssessment,
    EconomicIndicators,
    RegimeTransition,
    # Enums
    RegimeEnum,
    ISMSignalEnum,
    YieldCurveSignalEnum,
    CreditSpreadSignalEnum,
    FactorExposureEnum,
    VIXSignalEnum,
)

from optimizer.database.models.trading_economics import (
    TradingEconomicsSnapshot,
    TradingEconomicsIndicator,
    TradingEconomicsBondYield,
)

from optimizer.database.models.news import (
    NewsArticle,
)

# Import portfolio models
from optimizer.database.models.portfolio import (
    Portfolio,
    PortfolioPosition,
)

# Export all models and enums
__all__ = [
    "Base",
    # Universe models
    "Exchange",
    "Instrument",
    # Stock signal models
    "StockSignal",
    "SignalEnum",
    # Signal distribution models
    "SignalDistribution",
    # Macro regime models
    "MacroAnalysisRun",
    "MarketIndicators",
    "CountryRegimeAssessment",
    "EconomicIndicators",
    "NewsArticle",
    "RegimeTransition",
    "TradingEconomicsSnapshot",
    "TradingEconomicsIndicator",
    "TradingEconomicsBondYield",
    # Portfolio models
    "Portfolio",
    "PortfolioPosition",
    # Macro regime enums
    "RegimeEnum",
    "ISMSignalEnum",
    "YieldCurveSignalEnum",
    "CreditSpreadSignalEnum",
    "FactorExposureEnum",
    "VIXSignalEnum",
]
