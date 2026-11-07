"""Venue allowlist contract for the Screener-backed universe (stocks and ETFs).

The allowlist is spread over four tables that must agree: the Yahoo code ->
exchange-name map, the code -> ETF region map, the canonical-listing tiebreak
order, and the config's Yahoo suffix map (plus the trading-calendar map the
price-history guard reads). A venue missing from any of them is dropped or
silently loses a guard, so these tests pin the tables against each other, pin the
original venues against regression, and push quotes from every new venue through
the real transform with injected FX (no network).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("yfinance")

from app.services._shared.trading_calendar import (
    EXCHANGE_NAME_TO_MIC,
    get_expected_trading_sessions,
)
from app.services.universe import yfinance_source
from app.services.universe.trading212.builder import UniverseBuilder
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.yfinance_source import (
    _CODE_TO_CONFIG_NAME,
    _CODE_TO_REGION,
    _EXCHANGE_PREF,
    _STOCK_SORT_FIELD,
    PassThroughTickerMapper,
    YFinanceUniverseSource,
)

_BIG_CAP = 10_000

# The venues the universe covered before the allowlist was widened. Frozen here so
# widening it can never silently re-route or drop one of them.
_ORIGINAL_CODE_TO_NAME: dict[str, str] = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NYQ": "NYSE",
    "PCX": "NYSE",
    "ASE": "NYSE",
    "LSE": "London Stock Exchange",
    "IOB": "London Stock Exchange",
    "PAR": "Euronext Paris",
    "AMS": "Euronext Amsterdam",
    "BRU": "Euronext Brussels",
    "LIS": "Euronext Lisbon",
    "GER": "Deutsche Börse Xetra",
    "MUN": "Gettex",
    "EBS": "SIX Swiss Exchange",
    "VIE": "Wiener Börse",
    "MIL": "Borsa Italiana",
    "MCE": "Bolsa de Madrid",
    "MAD": "Bolsa de Madrid",
    "TOR": "Toronto Stock Exchange",
}

_ORIGINAL_REGION: dict[str, str] = {
    "NMS": "us",
    "NGM": "us",
    "NCM": "us",
    "NYQ": "us",
    "PCX": "us",
    "ASE": "us",
    "LSE": "gb",
    "IOB": "gb",
    "PAR": "fr",
    "AMS": "nl",
    "BRU": "be",
    "LIS": "pt",
    "GER": "de",
    "MUN": "de",
    "EBS": "ch",
    "VIE": "at",
    "MIL": "it",
    "MCE": "es",
    "MAD": "es",
    "TOR": "ca",
}

_ORIGINAL_SUFFIX: dict[str, str] = {
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

# code -> (exchange name, Yahoo suffix, ETF region, quote currency)
_NEW_VENUES: dict[str, tuple[str, str, str, str]] = {
    "JPX": ("Tokyo Stock Exchange", ".T", "jp", "JPY"),
    "ASX": ("Australian Securities Exchange", ".AX", "au", "AUD"),
    "HKG": ("Hong Kong Stock Exchange", ".HK", "hk", "HKD"),
    "SHH": ("Shanghai Stock Exchange", ".SS", "cn", "CNY"),
    "SHZ": ("Shenzhen Stock Exchange", ".SZ", "cn", "CNY"),
    "TAI": ("Taiwan Stock Exchange", ".TW", "tw", "TWD"),
    "TWO": ("Taipei Exchange", ".TWO", "tw", "TWD"),
    "KSC": ("Korea Exchange", ".KS", "kr", "KRW"),
    "KOE": ("KOSDAQ", ".KQ", "kr", "KRW"),
    "SES": ("Singapore Exchange", ".SI", "sg", "SGD"),
    "MEX": ("Bolsa Mexicana de Valores", ".MX", "mx", "MXN"),
    "SAO": ("B3 Brasil Bolsa Balcão", ".SA", "br", "BRL"),
    "TLV": ("Tel Aviv Stock Exchange", ".TA", "il", "ILA"),
    "CPH": ("Nasdaq Copenhagen", ".CO", "dk", "DKK"),
    "STO": ("Nasdaq Stockholm", ".ST", "se", "SEK"),
    "HEL": ("Nasdaq Helsinki", ".HE", "fi", "EUR"),
    "OSL": ("Oslo Børs", ".OL", "no", "NOK"),
    "NEO": ("Cboe Canada", ".NE", "ca", "CAD"),
    "PNK": ("OTC Markets", "", "us", "USD"),
    "OQB": ("OTC Markets", "", "us", "USD"),
    "OQX": ("OTC Markets", "", "us", "USD"),
}

# USD per 1 major unit, rounded from live Yahoo spot. Only the ratios matter.
_USD_PER_MAJOR: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "CHF": 1.1,
    "CAD": 0.73,
    "JPY": 0.0063,
    "AUD": 0.70,
    "MXN": 0.054,
    "BRL": 0.184,
    "HKD": 0.1275,
    "CNY": 0.15,
    "ILS": 0.3165,
    "DKK": 0.1497,
    "SEK": 0.1021,
    "NOK": 0.1082,
    "KRW": 0.000744,
    "TWD": 0.0305,
    "SGD": 0.78,
}


def _major(currency: str) -> str:
    return "ILS" if currency == "ILA" else currency


def _quote(symbol: str, exchange: str, **kw: Any) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "exchange": exchange,
        "currency": kw.get("currency", "USD"),
        "financialCurrency": kw.get("financialCurrency", kw.get("currency", "USD")),
        "longName": kw.get("longName", symbol),
        "regularMarketPrice": kw.get("regularMarketPrice", 100.0),
        "averageDailyVolume3Month": kw.get("averageDailyVolume3Month", 1_000_000),
        "sharesOutstanding": kw.get("sharesOutstanding", 1_000_000_000),
        "marketCap": kw.get("marketCap"),
        "netAssets": kw.get("netAssets"),
    }


def _fx(majors: set[str]) -> dict[str, float]:
    missing = majors - set(_USD_PER_MAJOR)
    assert not missing, f"test has no rate for {sorted(missing)}"
    return {m: _USD_PER_MAJOR[m] for m in majors | {"USD"}}


def _source(
    quotes: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> YFinanceUniverseSource:
    screener = MagicMock()
    screener.screen.side_effect = [{"quotes": quotes}, {"quotes": []}]
    src = YFinanceUniverseSource(screener=screener, fx_resolver=_fx)
    monkeypatch.setattr(
        src, "_build_queries", lambda: [("Q", "STOCK", _STOCK_SORT_FIELD, _BIG_CAP)]
    )
    return src


class TestTablesAgree:
    """The venue tables agree with each other and with the original venues."""

    def test_original_venues_keep_their_exchange_names(self) -> None:
        """Widening the allowlist never re-routes an original venue."""
        for code, name in _ORIGINAL_CODE_TO_NAME.items():
            assert _CODE_TO_CONFIG_NAME[code] == name

    def test_the_incumbent_shield_covers_exactly_the_venues_added_later(self) -> None:
        """A line on a newcomer venue cannot displace an incumbent on an older one."""
        original = set(_ORIGINAL_CODE_TO_NAME)
        added = set(_CODE_TO_CONFIG_NAME) - original

        assert set(yfinance_source._ORIGINAL_VENUES) == original
        assert set(yfinance_source._NEWCOMER_VENUES) == added
        assert set(_NEW_VENUES) <= added

    def test_original_venues_keep_their_etf_regions(self) -> None:
        """An original venue keeps the region its ETFs are queried by."""
        for code, region in _ORIGINAL_REGION.items():
            assert _CODE_TO_REGION[code] == region

    def test_original_suffixes_are_unchanged(self) -> None:
        """The Yahoo suffix of an original venue is unchanged."""
        suffixes = UniverseBuilderConfig().yahoo_suffix_map
        for name, suffix in _ORIGINAL_SUFFIX.items():
            assert suffixes[name] == suffix

    def test_every_code_has_a_region_and_a_tiebreak_rank(self) -> None:
        """Every allowlisted code is also an ETF region key and has a tiebreak rank."""
        codes = set(_CODE_TO_CONFIG_NAME)
        assert set(_CODE_TO_REGION) == codes
        assert set(_EXCHANGE_PREF) == codes
        assert len(_EXCHANGE_PREF) == len(set(_EXCHANGE_PREF))

    def test_every_exchange_name_is_discoverable_by_the_builder(self) -> None:
        """The builder admits an exchange only when it has a Yahoo suffix."""
        # The builder admits an exchange only when it has a Yahoo suffix.
        allowed = UniverseBuilderConfig().get_allowed_exchanges()
        assert set(_CODE_TO_CONFIG_NAME.values()) <= allowed

    def test_every_discoverable_exchange_has_a_trading_calendar(self) -> None:
        """A venue without a calendar silently loses the truncated-history guard."""
        # A venue missing here silently loses the truncated-history guard.
        allowed = UniverseBuilderConfig().get_allowed_exchanges()
        assert allowed <= set(EXCHANGE_NAME_TO_MIC)

    def test_every_etf_region_is_a_valid_screener_region(self) -> None:
        """Each ETF region is accepted by the screener query builder."""
        # Building the query validates the region against Yahoo's own table.
        src = YFinanceUniverseSource(screener=MagicMock())
        built = src._build_queries()
        etf_queries = [q for q, kind, _, _ in built if kind == "ETF"]
        assert len(etf_queries) == len(set(_CODE_TO_REGION.values()))

    def test_every_stock_code_is_a_valid_screener_exchange(self) -> None:
        """Each stock exchange code is accepted by the screener query builder."""
        src = YFinanceUniverseSource(screener=MagicMock())
        built = src._build_queries()
        assert sum(1 for _, kind, _, _ in built if kind == "STOCK") == len(
            _CODE_TO_CONFIG_NAME
        )


class TestNewVenues:
    """The venues added beyond the original ones flow through the transform."""

    @pytest.mark.parametrize(("code", "venue"), list(_NEW_VENUES.items()))
    def test_code_maps_to_its_exchange_region_and_suffix(
        self, code: str, venue: tuple[str, str, str, str]
    ) -> None:
        """Each new code resolves to its exchange name, ETF region and Yahoo suffix."""
        name, suffix, region, _ = venue
        assert _CODE_TO_CONFIG_NAME[code] == name
        assert _CODE_TO_REGION[code] == region
        assert UniverseBuilderConfig().get_yahoo_suffix(name) == suffix

    @pytest.mark.parametrize("name", sorted({v[0] for v in _NEW_VENUES.values()}))
    def test_new_exchange_has_a_working_trading_calendar(self, name: str) -> None:
        """The new exchange resolves a real session calendar over five years."""
        sessions = get_expected_trading_sessions(name, "5y")
        assert sessions is not None
        assert sessions > 800

    def test_a_listing_on_every_new_venue_is_admitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One liquid listing per new venue is admitted and its exchange recorded."""
        quotes = [
            _quote(f"{code}1", code, currency=ccy)
            for code, (_, _, _, ccy) in _NEW_VENUES.items()
        ]
        src = _source(quotes, monkeypatch)

        stored = {i["ticker"] for i in src.get_instruments()}
        names = {e["name"] for e in src.get_exchanges()}

        assert stored == {f"{code}1" for code in _NEW_VENUES}
        assert names == {v[0] for v in _NEW_VENUES.values()}

    def test_the_builder_persists_instruments_from_the_new_venues(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The builder, not only the source, accepts the new exchange names."""
        quotes = [
            _quote("7203.T", "JPX", currency="JPY", longName="Toyota Motor"),
            _quote("BHP.AX", "ASX", currency="AUD", longName="BHP Group"),
            _quote("TCEHY", "PNK", longName="Tencent Holdings ADR"),
        ]
        src = _source(quotes, monkeypatch)
        repo = MagicMock()
        repo.save_instruments_batch.side_effect = lambda rows, _id: len(rows)
        repo.get_active_tickers.return_value = set()
        repo.get_pinned_tickers.return_value = set()

        result = UniverseBuilder(
            config=UniverseBuilderConfig(),
            api_client=src,
            ticker_mapper=PassThroughTickerMapper(),
            repository=repo,
            max_workers=1,
        ).build()

        assert result.instruments_saved == 3
        assert result.exchanges_saved == 3


class TestFxNormalisation:
    """Each new currency must reach the USD anti-junk floor correctly scaled."""

    @staticmethod
    def _sized(code: str, currency: str, usd_mcap: float, symbol: str) -> dict:
        rate = _USD_PER_MAJOR[_major(currency)]
        shares = 1_000_000.0
        price_major = usd_mcap / (shares * rate)
        price_quoted = price_major * (100.0 if currency == "ILA" else 1.0)
        adv_usd = 200_000.0
        return _quote(
            symbol,
            code,
            currency=currency,
            financialCurrency=_major(currency),
            regularMarketPrice=price_quoted,
            sharesOutstanding=shares,
            averageDailyVolume3Month=adv_usd / (price_major * rate),
        )

    @pytest.mark.parametrize(
        ("code", "currency"),
        [(code, v[3]) for code, v in _NEW_VENUES.items() if v[3] != "USD"],
    )
    def test_floor_is_applied_in_usd_for_the_currency(
        self, code: str, currency: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A listing above the USD floor is kept and one below it is dropped."""
        above = self._sized(code, currency, 40_000_000.0, "ABOVE")
        below = self._sized(code, currency, 15_000_000.0, "BELOW")
        src = _source([above, below], monkeypatch)

        assert {i["ticker"] for i in src.get_instruments()} == {"ABOVE"}

    def test_every_new_currency_is_requested_from_the_fx_resolver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The source asks for a rate for every major currency it meets."""
        requested: list[set[str]] = []

        def resolver(majors: set[str]) -> dict[str, float]:
            """Record the requested majors and answer with the fixed rates."""
            requested.append(set(majors))
            return _fx(majors)

        quotes = [
            _quote(f"{code}1", code, currency=ccy, financialCurrency=_major(ccy))
            for code, (_, _, _, ccy) in _NEW_VENUES.items()
        ]
        screener = MagicMock()
        screener.screen.side_effect = [{"quotes": quotes}, {"quotes": []}]
        src = YFinanceUniverseSource(screener=screener, fx_resolver=resolver)
        monkeypatch.setattr(
            src, "_build_queries", lambda: [("Q", "STOCK", _STOCK_SORT_FIELD, _BIG_CAP)]
        )
        src.get_instruments()

        assert requested == [{_major(v[3]) for v in _NEW_VENUES.values()}]

    @pytest.mark.parametrize(
        "major",
        [
            "JPY",
            "AUD",
            "MXN",
            "BRL",
            "HKD",
            "CNY",
            "ILS",
            "DKK",
            "SEK",
            "NOK",
            "KRW",
            "TWD",
            "SGD",
        ],
    )
    def test_live_resolver_asks_yahoo_for_the_usd_cross(
        self, major: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The live resolver reads the ``{MAJOR}USD=X`` series for the currency."""
        import pandas as pd

        asked: list[str] = []

        class _Ticker:
            def __init__(self, symbol: str) -> None:
                asked.append(symbol)

            def history(self, period: str) -> pd.DataFrame:
                """Return a two-row close series whose last value is the spot."""
                return pd.DataFrame({"Close": [0.5, 0.25]})

        monkeypatch.setattr(yfinance_source.yf, "Ticker", _Ticker)

        rates = yfinance_source._live_usd_per_major({major})

        assert asked == [f"{major}USD=X"]
        assert rates == {"USD": 1.0, major: 0.25}


class TestCrossListingsAcrossTheNewVenues:
    """Canonical-listing selection across home venues, depositary lines and OTC."""

    def test_home_listing_beats_a_thin_otc_depositary_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A liquid home listing outranks a thin OTC depositary line."""
        home = _quote(
            "NESN.SW",
            "EBS",
            currency="CHF",
            financialCurrency="CHF",
            longName="Nestlé S.A.",
            regularMarketPrice=90.0,
            averageDailyVolume3Month=5_000_000,
        )
        adr = _quote(
            "NSRGY",
            "PNK",
            currency="USD",
            financialCurrency="CHF",
            longName="Nestle SA",
            regularMarketPrice=100.0,
            averageDailyVolume3Month=300_000,
        )
        src = _source([home, adr], monkeypatch)

        assert [i["ticker"] for i in src.get_instruments()] == ["NESN.SW"]

    def test_tokyo_primary_replaces_a_thin_xetra_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Tokyo primary line replaces a thin Xetra cross-listing."""
        tokyo = _quote(
            "7203.T",
            "JPX",
            currency="JPY",
            financialCurrency="JPY",
            longName="Toyota Motor Corporation",
            regularMarketPrice=2500.0,
            averageDailyVolume3Month=20_000_000,
            sharesOutstanding=15_000_000_000,
        )
        xetra = _quote(
            "TOY.DE",
            "GER",
            currency="EUR",
            financialCurrency="JPY",
            longName="Toyota Motor Corporation",
            regularMarketPrice=15.0,
            averageDailyVolume3Month=5_000,
            sharesOutstanding=15_000_000_000,
        )
        src = _source([tokyo, xetra], monkeypatch)

        assert [i["ticker"] for i in src.get_instruments()] == ["7203.T"]

    def test_exact_liquidity_tie_prefers_the_real_venue_over_otc(self) -> None:
        """On an exact liquidity tie a real venue ranks ahead of OTC."""
        for code in ("PNK", "OQB", "OQX"):
            assert _EXCHANGE_PREF.index("NMS") < _EXCHANGE_PREF.index(code)
            assert _EXCHANGE_PREF.index("TOR") < _EXCHANGE_PREF.index(code)
