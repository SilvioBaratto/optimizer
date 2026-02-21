from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from ..news.scraper import ArticleResult


@runtime_checkable
class CacheProtocol(Protocol):
    def get(self, key: str) -> Any | None: ...

    def put(self, key: str, value: Any, ttl: float | None = None) -> None: ...

    def clear(self) -> None: ...

    def size(self) -> int: ...


@runtime_checkable
class RateLimiterProtocol(Protocol):
    def acquire(self, key: str) -> None: ...


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    def check(self) -> None: ...

    def trigger(self) -> None: ...

    def reset(self) -> None: ...

    @property
    def is_active(self) -> bool: ...


@runtime_checkable
class ArticleScraperProtocol(Protocol):
    def fetch(self, url: str) -> ArticleResult: ...

    def fetch_multiple(
        self,
        urls: list[str],
        max_articles: int | None = None,
    ) -> list[ArticleResult]: ...


@runtime_checkable
class YFinanceClientProtocol(Protocol):
    def get_ticker(self, symbol: str) -> Any: ...

    def fetch_info(
        self,
        symbol: str,
        max_retries: int | None = None,
        min_fields: int = 10,
    ) -> dict[str, Any] | None: ...

    def fetch_history(
        self,
        symbol: str,
        period: str = "5y",
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        max_retries: int | None = None,
        min_rows: int = 10,
    ) -> pd.DataFrame | None: ...

    def fetch_price_and_benchmark(
        self,
        symbol: str,
        benchmark: str = "SPY",
        period: str = "5y",
        max_retries: int | None = None,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any] | None]: ...

    def bulk_download(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        period: str = "5y",
        interval: str = "1d",
        threads: bool = True,
        group_by: str = "ticker",
        auto_adjust: bool = False,
        progress: bool = False,
    ) -> pd.DataFrame | None: ...

    def fetch_prices_dataframe(
        self,
        symbols: list[str],
        period: str = "5y",
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame | None: ...


# ---------------------------------------------------------------------------
# Sub-client protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class FinancialsClientProtocol(Protocol):
    def fetch_income_stmt(
        self, symbol: str, quarterly: bool = False, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_balance_sheet(
        self, symbol: str, quarterly: bool = False, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_cashflow(
        self, symbol: str, quarterly: bool = False, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_earnings(
        self, symbol: str, quarterly: bool = False, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_sec_filings(
        self, symbol: str, max_retries: int | None = None
    ) -> list[dict[str, Any]] | None: ...


@runtime_checkable
class AnalysisClientProtocol(Protocol):
    def fetch_recommendations(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_recommendations_summary(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_upgrades_downgrades(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_analyst_price_targets(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_earnings_estimate(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_revenue_estimate(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_earnings_history(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_growth_estimates(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_sustainability(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...


@runtime_checkable
class HoldersClientProtocol(Protocol):
    def fetch_major_holders(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_institutional_holders(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_mutualfund_holders(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_insider_transactions(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_insider_purchases(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_insider_roster_holders(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...


@runtime_checkable
class CorporateActionsClientProtocol(Protocol):
    def fetch_dividends(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.Series | None: ...

    def fetch_splits(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.Series | None: ...

    def fetch_actions(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_capital_gains(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.Series | None: ...

    def fetch_shares_full(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None: ...


@runtime_checkable
class MetadataClientProtocol(Protocol):
    def fetch_isin(self, symbol: str, max_retries: int | None = None) -> str | None: ...

    def fetch_fast_info(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_calendar(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_options_expirations(
        self, symbol: str, max_retries: int | None = None
    ) -> tuple[str, ...] | None: ...

    def fetch_option_chain(
        self, symbol: str, date: str | None = None, max_retries: int | None = None
    ) -> Any | None: ...

    def fetch_earnings_dates(
        self, symbol: str, limit: int = 12, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_history_metadata(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...


@runtime_checkable
class FundsClientProtocol(Protocol):
    def fetch_fund_overview(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_fund_top_holdings(
        self, symbol: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_fund_sector_weightings(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_fund_bond_holdings(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_fund_bond_ratings(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_fund_equity_holdings(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_fund_operations(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_fund_asset_classes(
        self, symbol: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_fund_description(
        self, symbol: str, max_retries: int | None = None
    ) -> str | None: ...


@runtime_checkable
class MarketClientProtocol(Protocol):
    def fetch_status(
        self, market: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_summary(
        self, market: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...


@runtime_checkable
class SectorIndustryClientProtocol(Protocol):
    def fetch_sector_overview(
        self, sector_key: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_sector_top_companies(
        self, sector_key: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_sector_top_etfs(
        self, sector_key: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_sector_top_mutual_funds(
        self, sector_key: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_industry_overview(
        self, industry_key: str, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def fetch_industry_top_companies(
        self, industry_key: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_industry_top_etfs(
        self, industry_key: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...

    def fetch_industry_top_mutual_funds(
        self, industry_key: str, max_retries: int | None = None
    ) -> pd.DataFrame | None: ...


@runtime_checkable
class SearchClientProtocol(Protocol):
    def search(
        self, query: str, max_results: int = 8, max_retries: int | None = None
    ) -> dict[str, Any] | None: ...

    def lookup(
        self,
        query: str,
        asset_type: str = "stock",
        count: int = 25,
        max_retries: int | None = None,
    ) -> list[dict[str, Any]] | None: ...


@runtime_checkable
class ScreenerClientProtocol(Protocol):
    def screen(
        self,
        query: Any,
        offset: int = 0,
        size: int = 25,
        sort_field: str = "ticker",
        sort_asc: bool = True,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None: ...

    def get_predefined_screeners(self) -> dict[str, Any]: ...


@runtime_checkable
class CalendarsClientProtocol(Protocol):
    def fetch_earnings_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None: ...

    def fetch_ipo_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None: ...

    def fetch_splits_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None: ...

    def fetch_economic_events_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None: ...


@runtime_checkable
class StreamingClientProtocol(Protocol):
    def subscribe(self, symbols: list[str]) -> None: ...

    def unsubscribe(self, symbols: list[str]) -> None: ...

    def listen(
        self, message_handler: Callable[[dict[str, Any]], None] | None = None
    ) -> None: ...

    def close(self) -> None: ...
