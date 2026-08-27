"""SQLAlchemy models — single source of truth for the shared schema.

Every model module is imported here so ``Base.metadata`` is complete for Alembic
autogenerate and SQLite ``create_all`` in tests.
"""

from portopt_db.base import Base, BaseModel, TimestampMixin, UUIDPrimaryKeyMixin
from portopt_db.models.jobs.background_job import BackgroundJob, BackgroundJobError
from portopt_db.models.macro.macro_regime import (
    BondYield,
    EconomicIndicator,
    MacroCalibration,
    MacroNews,
    MacroNewsSummary,
    MacroNewsTheme,
    TradingEconomicsIndicator,
)
from portopt_db.models.market_data.calendars import (
    EarningsCalendar,
    EconomicEventCalendar,
    IpoCalendar,
    SplitCalendar,
)
from portopt_db.models.market_data.etf_metadata import (
    ETFAssetClass,
    ETFHolding,
    ETFMetadata,
    ETFSectorWeight,
)
from portopt_db.models.market_data.market_structure import (
    SectorIndustry,
    SectorSnapshot,
    SectorTopCompany,
)
from portopt_db.models.market_data.market_summary import MarketSummary
from portopt_db.models.market_data.yfinance_data import (
    AnalystPriceTarget,
    AnalystRecommendation,
    Dividend,
    FinancialStatement,
    InsiderTransaction,
    InstitutionalHolder,
    MutualFundHolder,
    PriceHistory,
    StockSplit,
    TickerNews,
    TickerProfile,
)
from portopt_db.models.universe.universe import Exchange, Instrument

__all__ = [
    "AnalystPriceTarget",
    "AnalystRecommendation",
    "BackgroundJob",
    "BackgroundJobError",
    "Base",
    "BaseModel",
    "BondYield",
    "Dividend",
    "ETFAssetClass",
    "ETFHolding",
    "ETFMetadata",
    "ETFSectorWeight",
    "EarningsCalendar",
    "EconomicEventCalendar",
    "EconomicIndicator",
    "Exchange",
    "FinancialStatement",
    "InsiderTransaction",
    "InstitutionalHolder",
    "Instrument",
    "IpoCalendar",
    "MacroCalibration",
    "MacroNews",
    "MacroNewsSummary",
    "MacroNewsTheme",
    "MarketSummary",
    "MutualFundHolder",
    "PriceHistory",
    "SectorIndustry",
    "SectorSnapshot",
    "SectorTopCompany",
    "SplitCalendar",
    "StockSplit",
    "TickerNews",
    "TickerProfile",
    "TimestampMixin",
    "TradingEconomicsIndicator",
    "UUIDPrimaryKeyMixin",
]
