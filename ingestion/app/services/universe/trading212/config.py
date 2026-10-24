"""
Universe Builder Configuration — exchange coverage, ticker mapping, and ETF dedup.

Ingestion applies no investability filtering, so this config carries no size /
price / liquidity / history thresholds. The universe scope is exactly the set of
exchanges in ``yahoo_suffix_map`` — an instrument is discoverable iff its exchange
has a Yahoo suffix here (so every persisted row can resolve a real yfinance
ticker). Unmapped venues (e.g. OTC Markets) are skipped entirely.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UniverseBuilderConfig:
    """Configuration for universe discovery from the Trading 212 API."""

    # Yahoo Finance suffix mapping by exchange. This IS the universe scope: only
    # instruments on these exchanges are discovered (both stocks and ETFs). A
    # ``""`` suffix means the bare symbol is the Yahoo ticker (US venues).
    yahoo_suffix_map: dict[str, str] = field(
        default_factory=lambda: {
            "NYSE": "",
            "NASDAQ": "",
            "London Stock Exchange": ".L",
            "London Stock Exchange AIM": ".L",
            "Euronext Paris": ".PA",
            "Euronext Amsterdam": ".AS",
            "Euronext Brussels": ".BR",
            "Euronext Lisbon": ".LS",
            "Deutsche Börse Xetra": ".DE",
            "Gettex": ".DE",
            "Borsa Italiana": ".MI",
            "Bolsa de Madrid": ".MC",
            "SIX Swiss Exchange": ".SW",
            "Wiener Börse": ".VI",
            "Toronto Stock Exchange": ".TO",
        }
    )

    # Share-class / cross-listing dedup (ETFs): keep the single listing per ISIN
    # on the most-preferred exchange (largest/most-liquid UCITS venues first).
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

    # Delisting-reconciliation sanity guard. A venue losing more than this
    # fraction of its active tickers in one run looks like a cut-off source
    # answer rather than a mass delisting, so none of them is marked.
    max_delisting_fraction: float = 0.20
    # Below this many active tickers one delisting already exceeds any sensible
    # fraction, so the guard only applies to venues at least this large.
    delisting_guard_min_tickers: int = 20

    def get_allowed_exchanges(self) -> set[str]:
        """The universe scope: every exchange with a Yahoo suffix mapping."""
        return set(self.yahoo_suffix_map)

    def get_etf_allowed_exchanges(self) -> set[str]:
        """Same scope as stocks: every exchange with a Yahoo suffix mapping."""
        return set(self.yahoo_suffix_map)

    def is_exchange_allowed(self, exchange_name: str) -> bool:
        return exchange_name in self.yahoo_suffix_map

    def get_yahoo_suffix(self, exchange_name: str) -> str | None:
        return self.yahoo_suffix_map.get(exchange_name)
