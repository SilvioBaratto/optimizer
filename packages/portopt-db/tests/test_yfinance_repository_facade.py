"""The split ``YFinanceRepository`` keeps the public surface consumers call.

``_DEVELOPMENT_SIGNATURES`` holds ``str(inspect.signature(...))`` of the constructor
and of each public method as they were on ``development`` at ed851bbc, before the
class was split into one repository per aggregate. Ingestion and fund call the
facade, so a dropped, renamed or re-typed method is a consumer break even while the
aggregate repositories' own tests pass.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from portopt_db.repositories import market_data
from portopt_db.repositories.market_data import yfinance_repository
from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository

_SEQ = "collections.abc.Sequence"
_YF = "portopt_db.models.market_data.yfinance_data."
_UNI = "portopt_db.models.universe.universe."
_PORTS = "portopt_db.ports."

_DEVELOPMENT_SIGNATURES: dict[str, str] = {
    "__init__": "(self, session: sqlalchemy.orm.session.Session)",
    "get_benchmark_coverage": (
        "(self, tickers: list[str]) -> dict[str, tuple[int, datetime.date | None]]"
    ),
    "get_dividends": f"(self, instrument_id: uuid.UUID) -> {_SEQ}[{_YF}Dividend]",
    "get_financial_statements": (
        "(self, instrument_id: uuid.UUID, statement_type: str | None = None, "
        f"period_type: str | None = None) -> {_SEQ}[{_YF}FinancialStatement]"
    ),
    "get_insider_transactions": (
        f"(self, instrument_id: uuid.UUID) -> {_SEQ}[{_YF}InsiderTransaction]"
    ),
    "get_institutional_holders": (
        f"(self, instrument_id: uuid.UUID) -> {_SEQ}[{_YF}InstitutionalHolder]"
    ),
    "get_instrument_by_yfinance_ticker": (
        f"(self, yfinance_ticker: str) -> {_UNI}Instrument | None"
    ),
    "get_instruments_with_yfinance_ticker": (
        f"(self, *, today: datetime.date | None = None) -> {_SEQ}[{_UNI}Instrument]"
    ),
    "get_mutualfund_holders": (
        f"(self, instrument_id: uuid.UUID) -> {_SEQ}[{_YF}MutualFundHolder]"
    ),
    "get_news": f"(self, instrument_id: uuid.UUID) -> {_SEQ}[{_YF}TickerNews]",
    "get_oldest_price_date": "(self, instrument_id: uuid.UUID) -> datetime.date | None",
    "get_options_as_of": "(self, instrument_id: uuid.UUID) -> datetime.date | None",
    "get_options_as_of_bulk": (
        f"(self, instrument_ids: {_SEQ}[uuid.UUID]) -> dict[uuid.UUID, datetime.date]"
    ),
    "get_price_history": (
        "(self, instrument_id: uuid.UUID, "
        "start_date: datetime.date | None = None, "
        "end_date: datetime.date | None = None, limit: int | None = None, "
        f"ascending: bool = False) -> {_SEQ}[{_YF}PriceHistory]"
    ),
    "get_price_targets": (
        f"(self, instrument_id: uuid.UUID) -> {_YF}AnalystPriceTarget | None"
    ),
    "get_profile": f"(self, instrument_id: uuid.UUID) -> {_YF}TickerProfile | None",
    "get_recommendations": (
        f"(self, instrument_id: uuid.UUID) -> {_SEQ}[{_YF}AnalystRecommendation]"
    ),
    "get_scraped_news_uuids": (
        f"(self, instrument_id: uuid.UUID, news_uuids: {_SEQ}[str]) -> set[str]"
    ),
    "get_sectors_by_yfinance_ticker": f"(self, tickers: {_SEQ}[str]) -> dict[str, str]",
    "get_splits": f"(self, instrument_id: uuid.UUID) -> {_SEQ}[{_YF}StockSplit]",
    "get_staleness_info": "(self, instrument_id: uuid.UUID) -> dict[str, typing.Any]",
    "get_staleness_info_bulk": (
        f"(self, instrument_ids: {_SEQ}[uuid.UUID])"
        " -> dict[uuid.UUID, dict[str, typing.Any]]"
    ),
    "instrument_for": (
        f"(self, yfinance_ticker: str) -> {_PORTS}InstrumentSummary | None"
    ),
    "newest_close_per_instrument": (
        "(self, start: datetime.date, end: datetime.date) -> list[datetime.date]"
    ),
    "price_bars_for": (
        "(self, instrument_id: uuid.UUID, "
        "start_date: datetime.date | None = None, "
        "end_date: datetime.date | None = None, limit: int | None = None, "
        f"ascending: bool = False) -> list[{_PORTS}PriceBar]"
    ),
    "profile_for": f"(self, instrument_id: uuid.UUID) -> {_PORTS}ProfileSummary | None",
    "prune_option_snapshots": "(self, instrument_id: uuid.UUID, *, keep: int) -> int",
    "prune_ticker_news": (
        "(self, older_than: datetime.datetime | datetime.date, *, "
        "batch_size: int = 1000) -> int"
    ),
    "upsert_analyst_actions": (
        "(self, instrument_id: uuid.UUID, df: pandas.DataFrame) -> int"
    ),
    "upsert_dividends": (
        "(self, instrument_id: uuid.UUID, "
        "dividends: pandas.Series | pandas.DataFrame, "
        "price_currency: str | None = None) -> int"
    ),
    "upsert_earnings_dates": (
        "(self, instrument_id: uuid.UUID, df: pandas.DataFrame) -> int"
    ),
    "upsert_earnings_estimate": (
        "(self, instrument_id: uuid.UUID, df: pandas.DataFrame) -> int"
    ),
    "upsert_earnings_history": (
        "(self, instrument_id: uuid.UUID, df: pandas.DataFrame) -> int"
    ),
    "upsert_eps_revisions": (
        "(self, instrument_id: uuid.UUID, df: pandas.DataFrame) -> int"
    ),
    "upsert_eps_trend": "(self, instrument_id: uuid.UUID, df: pandas.DataFrame) -> int",
    "upsert_financial_statements": (
        "(self, instrument_id: uuid.UUID, df: pandas.DataFrame, "
        "statement_type: str, period_type: str, "
        "currency_code: str | None = None, as_of: datetime.date | None = None)"
        " -> int"
    ),
    "upsert_growth_estimates": (
        "(self, instrument_id: uuid.UUID, df: pandas.DataFrame) -> int"
    ),
    "upsert_insider_purchases": (
        "(self, instrument_id: uuid.UUID, df: pandas.DataFrame) -> int"
    ),
    "upsert_insider_roster": (
        "(self, instrument_id: uuid.UUID, df: pandas.DataFrame) -> int"
    ),
    "upsert_insider_transactions": (
        "(self, instrument_id: uuid.UUID, insiders_df: pandas.DataFrame) -> int"
    ),
    "upsert_institutional_holders": (
        "(self, instrument_id: uuid.UUID, holders_df: pandas.DataFrame) -> int"
    ),
    "upsert_major_holders": (
        "(self, instrument_id: uuid.UUID, df: pandas.DataFrame) -> int"
    ),
    "upsert_mutualfund_holders": (
        "(self, instrument_id: uuid.UUID, holders_df: pandas.DataFrame) -> int"
    ),
    "upsert_news": (
        "(self, instrument_id: uuid.UUID, articles: list[dict[str, typing.Any]]) -> int"
    ),
    "upsert_option_chain": (
        "(self, instrument_id: uuid.UUID, rows: list[dict[str, typing.Any]]) -> int"
    ),
    "upsert_price_history": (
        "(self, instrument_id: uuid.UUID, history_df: pandas.DataFrame, "
        "price_unit: str | None = None) -> int"
    ),
    "upsert_price_targets": (
        "(self, instrument_id: uuid.UUID, targets: dict[str, typing.Any]) -> int"
    ),
    "upsert_profile": (
        "(self, instrument_id: uuid.UUID, info: dict[str, typing.Any]) -> int"
    ),
    "upsert_profile_extras": (
        "(self, instrument_id: uuid.UUID, info: dict[str, typing.Any]) -> int"
    ),
    "upsert_recommendations": (
        "(self, instrument_id: uuid.UUID, rec_df: pandas.DataFrame) -> int"
    ),
    "upsert_revenue_estimate": (
        "(self, instrument_id: uuid.UUID, df: pandas.DataFrame) -> int"
    ),
    "upsert_sec_filings": (
        "(self, instrument_id: uuid.UUID, filings: list[dict[str, typing.Any]]) -> int"
    ),
    "upsert_shares_outstanding": "(self, instrument_id: uuid.UUID, shares: Any) -> int",
    "upsert_splits": "(self, instrument_id: uuid.UUID, splits: pandas.Series) -> int",
}

# The public names ``yfinance_repository`` defined on ``development``, except
# ``logger``, and the private helpers this package's tests import from it.
_MODULE_NAMES_TO_KEEP = (
    "DORMANT_AFTER_DAYS",
    "FINANCIAL_STATEMENT_GROUPS",
    "FUNDAMENTAL_CATEGORIES",
    "VALUATION_MEASURES_PERIOD_TYPE",
    "VALUATION_MEASURES_STATEMENT_TYPE",
    "YFinanceRepository",
    "_SENTINEL_DATE",
    "_safe_date",
    "_safe_float",
    "_safe_int",
    "_safe_str",
    "_safe_val",
    "statement_staleness_key",
)


_MARKET_DATA = "portopt_db.repositories.market_data."

# Where each moved public method is defined; the facade reaches it by inheritance.
_HOME_MODULES: dict[str, str] = {
    "get_price_targets": "analyst_repository",
    "get_recommendations": "analyst_repository",
    "upsert_analyst_actions": "analyst_repository",
    "upsert_earnings_dates": "analyst_repository",
    "upsert_earnings_estimate": "analyst_repository",
    "upsert_earnings_history": "analyst_repository",
    "upsert_eps_revisions": "analyst_repository",
    "upsert_eps_trend": "analyst_repository",
    "upsert_growth_estimates": "analyst_repository",
    "upsert_price_targets": "analyst_repository",
    "upsert_recommendations": "analyst_repository",
    "upsert_revenue_estimate": "analyst_repository",
    "get_financial_statements": "fundamentals_repository",
    "get_profile": "fundamentals_repository",
    "get_sectors_by_yfinance_ticker": "fundamentals_repository",
    "upsert_financial_statements": "fundamentals_repository",
    "upsert_profile": "fundamentals_repository",
    "upsert_profile_extras": "fundamentals_repository",
    "upsert_sec_filings": "fundamentals_repository",
    "upsert_shares_outstanding": "fundamentals_repository",
    "get_news": "news_repository",
    "get_scraped_news_uuids": "news_repository",
    "prune_ticker_news": "news_repository",
    "upsert_news": "news_repository",
    "get_insider_transactions": "ownership_repository",
    "get_institutional_holders": "ownership_repository",
    "get_mutualfund_holders": "ownership_repository",
    "upsert_insider_purchases": "ownership_repository",
    "upsert_insider_roster": "ownership_repository",
    "upsert_insider_transactions": "ownership_repository",
    "upsert_institutional_holders": "ownership_repository",
    "upsert_major_holders": "ownership_repository",
    "upsert_mutualfund_holders": "ownership_repository",
    "get_dividends": "prices_repository",
    "get_oldest_price_date": "prices_repository",
    "get_price_history": "prices_repository",
    "get_splits": "prices_repository",
    "upsert_dividends": "prices_repository",
    "upsert_price_history": "prices_repository",
    "upsert_splits": "prices_repository",
}


def _public_names() -> set[str]:
    return {name for name in dir(YFinanceRepository) if not name.startswith("_")}


def test_facade_delegates_every_public_method() -> None:
    """The facade's public names are the snapshot's, none added and none lost."""
    assert _public_names() == set(_DEVELOPMENT_SIGNATURES) - {"__init__"}


@pytest.mark.parametrize("name", sorted(_DEVELOPMENT_SIGNATURES))
def test_signature_matches_the_development_snapshot(name: str) -> None:
    """Parameters, defaults and annotations render as they did before the split."""
    signature = inspect.signature(getattr(YFinanceRepository, name))

    assert str(signature) == _DEVELOPMENT_SIGNATURES[name]


@pytest.mark.parametrize(("name", "module"), sorted(_HOME_MODULES.items()))
def test_method_is_defined_in_its_aggregate_module(name: str, module: str) -> None:
    """The facade's method is the one its aggregate repository module defines."""
    assert getattr(YFinanceRepository, name).__module__ == _MARKET_DATA + module


def test_constructor_keeps_the_session_attribute() -> None:
    """``YFinanceRepository(session).session`` is the session it was built with."""
    session = MagicMock(name="session")

    assert YFinanceRepository(session).session is session


@pytest.mark.parametrize("name", _MODULE_NAMES_TO_KEEP)
def test_module_names_consumers_import_still_resolve(name: str) -> None:
    """A name ``yfinance_repository`` offered before the split still resolves there."""
    assert hasattr(yfinance_repository, name)


def test_package_export_is_the_facade() -> None:
    """``portopt_db.repositories.market_data`` exports the facade class."""
    assert market_data.YFinanceRepository is YFinanceRepository
