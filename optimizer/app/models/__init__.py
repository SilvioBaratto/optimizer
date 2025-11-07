"""Database models for Optimizer API"""

# Import base classes
from app.models.base import Base

# Import universe models
from app.models.universe import (
    Exchange,
    Instrument
)

# Import stock signal models
from app.models.stock_signals import (
    StockSignal,
    SignalEnum
)

# Import signal distribution models
from app.models.signal_distribution import (
    SignalDistribution
)

# Import macro regime models
from app.models.macro_regime import (
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

from app.models.trading_economics import (
    TradingEconomicsSnapshot,
    TradingEconomicsIndicator,
    TradingEconomicsBondYield,
)

from app.models.news import (
    NewsArticle,
)

# Import portfolio models
from app.models.portfolio import (
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