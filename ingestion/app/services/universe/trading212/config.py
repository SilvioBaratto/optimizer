"""
Universe Builder Configuration — exchange coverage, ticker mapping, and ETF dedup.

Ingestion applies no investability filtering, so this config carries no size /
price / liquidity / history thresholds — only the exchange sets to discover, the
Yahoo suffix map used to resolve tickers, and the ISIN-dedup exchange preference.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UniverseBuilderConfig:
    """Configuration for universe discovery from the Trading 212 API."""

    # Portfolio countries and their Trading 212 stock exchanges.
    portfolio_countries: tuple[str, ...] = (
        "USA",
        "Germany",
        "France",
        "UK",
    )
    country_to_exchanges: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "USA": ("NYSE", "NASDAQ"),
            "Germany": ("Deutsche Börse Xetra",),
            "France": ("Euronext Paris",),
            "UK": ("London Stock Exchange",),
        }
    )

    # Yahoo Finance suffix mapping by exchange. The first five are the stock
    # exchanges; the last three are ETF-only venues (broader UCITS set).
    yahoo_suffix_map: dict[str, str] = field(
        default_factory=lambda: {
            "NYSE": "",
            "NASDAQ": "",
            "London Stock Exchange": ".L",
            "Euronext Paris": ".PA",
            "Deutsche Börse Xetra": ".DE",
            "Borsa Italiana": ".MI",
            "Euronext Amsterdam": ".AS",
            "SIX Swiss Exchange": ".SW",
        }
    )

    # ETF universe: broader UCITS-friendly exchange set (the 5 stock exchanges
    # plus Milan / Amsterdam / SIX, where many bond & multi-asset UCITS ETFs list).
    etf_exchanges: tuple[str, ...] = (
        "NYSE",
        "NASDAQ",
        "Deutsche Börse Xetra",
        "Euronext Paris",
        "London Stock Exchange",
        "Borsa Italiana",
        "Euronext Amsterdam",
        "SIX Swiss Exchange",
    )
    # Share-class / cross-listing dedup: keep the single listing per ISIN on the
    # most-preferred exchange (largest/most-liquid UCITS venues first).
    etf_exchange_preference: tuple[str, ...] = (
        "Deutsche Börse Xetra",
        "London Stock Exchange",
        "Euronext Amsterdam",
        "Euronext Paris",
        "Borsa Italiana",
        "SIX Swiss Exchange",
        "NASDAQ",
        "NYSE",
    )

    def get_allowed_exchanges(self) -> set[str]:
        allowed: set[str] = set()
        for country in self.portfolio_countries:
            allowed.update(self.country_to_exchanges.get(country, ()))
        return allowed

    def get_etf_allowed_exchanges(self) -> set[str]:
        return set(self.etf_exchanges)

    def is_exchange_allowed(self, exchange_name: str) -> bool:
        return exchange_name in self.get_allowed_exchanges()

    def get_yahoo_suffix(self, exchange_name: str) -> str | None:
        return self.yahoo_suffix_map.get(exchange_name)
