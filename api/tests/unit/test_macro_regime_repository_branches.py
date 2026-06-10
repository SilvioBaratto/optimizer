"""Branch-coverage tests for MacroRegimeRepository.

Goal: close as many of the 90 branches as possible using the real ``db_session``
SQLite fixture (SAVEPOINT, no second session, no live DB).

IMPORTANT — Postgres-only methods
----------------------------------
All ``_upsert``/``pg_insert … ON CONFLICT`` paths use
``sqlalchemy.dialects.postgresql.insert`` which is **not** executable on
SQLite.  The following public methods are therefore *skipped* for their main
execution path; only their SQLite-safe early-return guard branches are driven:

  - ``upsert_economic_indicator``          (pg ON CONFLICT)
  - ``upsert_economic_indicator_observation`` (pg ON CONFLICT)
  - ``upsert_te_indicators``               (pg ON CONFLICT)
  - ``upsert_bond_yields``                 (pg ON CONFLICT)
  - ``upsert_te_observations``             (pg ON CONFLICT)
  - ``upsert_bond_yield_observations``     (pg ON CONFLICT)
  - ``upsert_fred_observations``           (pg ON CONFLICT)
  - ``upsert_macro_news``                  (pg ON CONFLICT)
  - ``upsert_macro_news_summary``          (pg ON CONFLICT)
  - ``upsert_macro_calibration``           (pg ON CONFLICT)

All SELECT / plain-INSERT / DELETE methods run fine on SQLite and are fully
exercised below.
"""

from __future__ import annotations

import datetime

from sqlalchemy.orm import Session

from app.models.macro.macro_regime import (
    BondYield,
    BondYieldObservation,
    EconomicIndicator,
    EconomicIndicatorObservation,
    FredObservation,
    MacroCalibration,
    MacroNews,
    MacroNewsSummary,
    MacroNewsTheme,
    TradingEconomicsIndicator,
    TradingEconomicsObservation,
)
from app.repositories.macro.macro_regime_repository import MacroRegimeRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_D = datetime.date
_DT = datetime.datetime


def _flush(session: Session, *objs: object) -> None:
    """Add objects and flush within the SAVEPOINT transaction."""
    for obj in objs:
        session.add(obj)
    session.flush()


def _ei(country: str, ref: _D | None = None) -> EconomicIndicator:
    return EconomicIndicator(country=country, last_inflation=1.5, reference_date=ref)


def _eio(country: str, date: _D) -> EconomicIndicatorObservation:
    return EconomicIndicatorObservation(
        country=country, date=date, last_inflation=2.0, gdp_growth_6m=1.0
    )


def _tei(country: str, key: str, value: float | None = 3.5) -> TradingEconomicsIndicator:
    return TradingEconomicsIndicator(country=country, indicator_key=key, value=value)


def _by(country: str, maturity: str, yv: float | None = 4.1) -> BondYield:
    return BondYield(country=country, maturity=maturity, yield_value=yv)


def _teo(
    country: str, key: str, date: _D, value: float | None = 5.0
) -> TradingEconomicsObservation:
    return TradingEconomicsObservation(
        country=country, indicator_key=key, date=date, value=value
    )


def _byo(country: str, maturity: str, date: _D, yv: float | None = 3.0) -> BondYieldObservation:
    return BondYieldObservation(country=country, maturity=maturity, date=date, yield_value=yv)


def _fred(series: str, date: _D, value: float | None = 0.5) -> FredObservation:
    return FredObservation(series_id=series, date=date, value=value)


def _news(news_id: str, pub: _DT | None = None, ticker: str | None = None) -> MacroNews:
    return MacroNews(
        news_id=news_id,
        title=f"Title {news_id}",
        publish_time=pub,
        source_ticker=ticker,
    )


def _summary(
    country: str,
    date: _D,
    summary: str = "ok",
    sentiment: str = "neutral",
) -> MacroNewsSummary:
    return MacroNewsSummary(
        country=country,
        summary_date=date,
        summary=summary,
        sentiment=sentiment,
        sentiment_score=0.0,
        article_count=1,
    )


def _calib(
    country: str,
    phase: str = "expansion",
    delta: float = 0.1,
    tau: float = 0.5,
    confidence: float = 0.8,
    regime: str | None = None,
) -> MacroCalibration:
    return MacroCalibration(
        country=country,
        phase=phase,
        delta=delta,
        tau=tau,
        confidence=confidence,
        regime_classification=regime,
    )


# ===========================================================================
# Postgres-only early-return guards (empty-input branch = False branch)
# ===========================================================================


class TestUpsertEmptyInputGuards:
    """Each ``upsert_*`` returns 0 immediately when called with an empty payload.

    This drives the ``if not data / if not indicators_dict / if not rows``
    guard branch (the 'True' side) without executing the pg ON CONFLICT stmt.
    # Postgres-only ON CONFLICT — not executable on SQLite
    """

    def test_upsert_economic_indicator_empty_data_returns_zero(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.upsert_economic_indicator("USA", {}) == 0

    def test_upsert_economic_indicator_observation_empty_data_returns_zero(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.upsert_economic_indicator_observation("USA", _D(2026, 1, 1), {}) == 0

    def test_upsert_te_indicators_empty_dict_returns_zero(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.upsert_te_indicators("USA", {}) == 0

    def test_upsert_bond_yields_empty_dict_returns_zero(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.upsert_bond_yields("USA", {}) == 0

    def test_upsert_te_observations_empty_dict_returns_zero(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.upsert_te_observations("USA", _D(2026, 1, 1), {}) == 0

    def test_upsert_bond_yield_observations_empty_dict_returns_zero(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.upsert_bond_yield_observations("USA", _D(2026, 1, 1), {}) == 0

    def test_upsert_fred_observations_empty_list_returns_zero(
        self, db_session: Session
    ) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.upsert_fred_observations("T10Y2Y", []) == 0

    def test_upsert_macro_news_empty_list_returns_zero(
        self, db_session: Session
    ) -> None:
        # upsert_macro_news hits a PG upsert internally — but when rows=[] the
        # early return in _upsert fires BEFORE pg_insert is reached.
        # Postgres-only ON CONFLICT — not executable on SQLite (main path)
        repo = MacroRegimeRepository(db_session)
        assert repo.upsert_macro_news([]) == 0


# ===========================================================================
# upsert_te_observations — value=None skip branch
# ===========================================================================


class TestUpsertTeObservationsNoneValue:
    """Drive the ``if value is None: continue`` branch without hitting PG upsert.

    When ALL indicator entries have value=None the ``rows`` list stays empty,
    so ``_upsert`` returns 0 via its own empty-rows guard — still no PG stmt.
    # Postgres-only ON CONFLICT — not executable on SQLite (main path)
    """

    def test_all_none_values_skipped_returns_zero(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        result = repo.upsert_te_observations(
            "USA", _D(2026, 1, 1), {"cpi": {"value": None}, "gdp": {"value": None}}
        )
        assert result == 0


# ===========================================================================
# upsert_bond_yield_observations — yield_val=None skip branch
# ===========================================================================


class TestUpsertBondYieldObservationsNoneYield:
    """Drive the ``if yield_val is None: continue`` branch similarly.
    # Postgres-only ON CONFLICT — not executable on SQLite (main path)
    """

    def test_all_none_yields_skipped_returns_zero(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        result = repo.upsert_bond_yield_observations(
            "USA", _D(2026, 1, 1), {"10Y": {"yield": None}, "2Y": {"yield": None}}
        )
        assert result == 0


# ===========================================================================
# get_economic_indicators
# ===========================================================================


class TestGetEconomicIndicators:
    def test_when_empty_returns_empty_sequence(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_economic_indicators()) == []

    def test_when_no_country_filter_returns_all(self, db_session: Session) -> None:
        _flush(db_session, _ei("USA"), _ei("Germany"))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_economic_indicators()
        countries = {r.country for r in results}
        assert countries == {"USA", "Germany"}

    def test_when_country_filter_returns_only_matching(self, db_session: Session) -> None:
        _flush(db_session, _ei("USA"), _ei("Germany"))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_economic_indicators(country="USA")
        assert all(r.country == "USA" for r in results)
        assert len(results) == 1

    def test_when_country_filter_no_match_returns_empty(self, db_session: Session) -> None:
        _flush(db_session, _ei("USA"))
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_economic_indicators(country="France")) == []

    def test_orders_by_country(self, db_session: Session) -> None:
        _flush(db_session, _ei("USA"), _ei("Germany"), _ei("Australia"))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_economic_indicators()
        countries = [r.country for r in results]
        assert countries == sorted(countries)


# ===========================================================================
# get_economic_indicator_observations
# ===========================================================================


class TestGetEconomicIndicatorObservations:
    _D1 = _D(2026, 1, 1)
    _D2 = _D(2026, 2, 1)
    _D3 = _D(2026, 3, 1)

    def test_when_empty_returns_empty(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_economic_indicator_observations()) == []

    def test_no_filters_returns_all(self, db_session: Session) -> None:
        _flush(db_session, _eio("USA", self._D1), _eio("USA", self._D2), _eio("EU", self._D1))
        repo = MacroRegimeRepository(db_session)
        assert len(repo.get_economic_indicator_observations()) == 3

    def test_country_filter(self, db_session: Session) -> None:
        _flush(db_session, _eio("USA", self._D1), _eio("EU", self._D1))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_economic_indicator_observations(country="EU")
        assert all(r.country == "EU" for r in results)

    def test_start_date_filter(self, db_session: Session) -> None:
        _flush(db_session, _eio("USA", self._D1), _eio("USA", self._D2), _eio("USA", self._D3))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_economic_indicator_observations(start_date=self._D2)
        assert all(r.date >= self._D2 for r in results)
        assert len(results) == 2

    def test_end_date_filter(self, db_session: Session) -> None:
        _flush(db_session, _eio("USA", self._D1), _eio("USA", self._D2), _eio("USA", self._D3))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_economic_indicator_observations(end_date=self._D2)
        assert all(r.date <= self._D2 for r in results)
        assert len(results) == 2

    def test_start_and_end_date_filter(self, db_session: Session) -> None:
        _flush(db_session, _eio("USA", self._D1), _eio("USA", self._D2), _eio("USA", self._D3))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_economic_indicator_observations(
            start_date=self._D1, end_date=self._D2
        )
        assert len(results) == 2

    def test_country_and_date_combined(self, db_session: Session) -> None:
        _flush(db_session, _eio("USA", self._D1), _eio("EU", self._D2))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_economic_indicator_observations(
            country="USA", start_date=self._D2
        )
        assert results == []


# ===========================================================================
# get_te_indicators
# ===========================================================================


class TestGetTeIndicators:
    def test_when_empty_returns_empty(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_te_indicators()) == []

    def test_no_filter_returns_all(self, db_session: Session) -> None:
        _flush(db_session, _tei("USA", "cpi"), _tei("EU", "gdp"))
        repo = MacroRegimeRepository(db_session)
        assert len(repo.get_te_indicators()) == 2

    def test_country_filter_returns_matching(self, db_session: Session) -> None:
        _flush(db_session, _tei("USA", "cpi"), _tei("EU", "gdp"))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_te_indicators(country="USA")
        assert all(r.country == "USA" for r in results)

    def test_country_filter_no_match_returns_empty(self, db_session: Session) -> None:
        _flush(db_session, _tei("USA", "cpi"))
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_te_indicators(country="JP")) == []

    def test_orders_by_country_then_key(self, db_session: Session) -> None:
        _flush(db_session, _tei("USA", "gdp"), _tei("USA", "cpi"), _tei("EU", "cpi"))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_te_indicators()
        pairs = [(r.country, r.indicator_key) for r in results]
        assert pairs == sorted(pairs)


# ===========================================================================
# get_bond_yields
# ===========================================================================


class TestGetBondYields:
    def test_when_empty_returns_empty(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_bond_yields()) == []

    def test_no_filter_returns_all(self, db_session: Session) -> None:
        _flush(db_session, _by("USA", "10Y"), _by("EU", "2Y"))
        repo = MacroRegimeRepository(db_session)
        assert len(repo.get_bond_yields()) == 2

    def test_country_filter(self, db_session: Session) -> None:
        _flush(db_session, _by("USA", "10Y"), _by("EU", "2Y"))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_bond_yields(country="EU")
        assert all(r.country == "EU" for r in results)

    def test_country_filter_no_match(self, db_session: Session) -> None:
        _flush(db_session, _by("USA", "10Y"))
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_bond_yields(country="JP")) == []

    def test_orders_by_country_then_maturity(self, db_session: Session) -> None:
        _flush(db_session, _by("USA", "30Y"), _by("USA", "2Y"), _by("EU", "10Y"))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_bond_yields()
        pairs = [(r.country, r.maturity) for r in results]
        assert pairs == sorted(pairs)


# ===========================================================================
# get_te_observations
# ===========================================================================


class TestGetTeObservations:
    _D1 = _D(2026, 1, 1)
    _D2 = _D(2026, 2, 1)
    _D3 = _D(2026, 3, 1)

    def test_when_empty_returns_empty(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_te_observations()) == []

    def test_no_filters_returns_all(self, db_session: Session) -> None:
        _flush(
            db_session,
            _teo("USA", "cpi", self._D1),
            _teo("EU", "gdp", self._D2),
            _teo("USA", "gdp", self._D3),
        )
        repo = MacroRegimeRepository(db_session)
        assert len(repo.get_te_observations()) == 3

    def test_country_filter(self, db_session: Session) -> None:
        _flush(db_session, _teo("USA", "cpi", self._D1), _teo("EU", "gdp", self._D1))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_te_observations(country="EU")
        assert all(r.country == "EU" for r in results)

    def test_indicator_keys_filter_single(self, db_session: Session) -> None:
        _flush(
            db_session,
            _teo("USA", "cpi", self._D1),
            _teo("USA", "gdp", self._D1),
            _teo("USA", "unemployment", self._D1),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_te_observations(indicator_keys=["cpi"])
        assert all(r.indicator_key == "cpi" for r in results)
        assert len(results) == 1

    def test_indicator_keys_filter_multiple(self, db_session: Session) -> None:
        _flush(
            db_session,
            _teo("USA", "cpi", self._D1),
            _teo("USA", "gdp", self._D1),
            _teo("USA", "unemployment", self._D1),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_te_observations(indicator_keys=["cpi", "gdp"])
        keys = {r.indicator_key for r in results}
        assert keys == {"cpi", "gdp"}

    def test_indicator_keys_none_returns_all(self, db_session: Session) -> None:
        _flush(db_session, _teo("USA", "cpi", self._D1), _teo("USA", "gdp", self._D1))
        repo = MacroRegimeRepository(db_session)
        assert len(repo.get_te_observations(indicator_keys=None)) == 2

    def test_start_date_filter(self, db_session: Session) -> None:
        _flush(
            db_session,
            _teo("USA", "cpi", self._D1),
            _teo("USA", "cpi", self._D2),
            _teo("USA", "cpi", self._D3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_te_observations(start_date=self._D2)
        assert all(r.date >= self._D2 for r in results)
        assert len(results) == 2

    def test_end_date_filter(self, db_session: Session) -> None:
        _flush(
            db_session,
            _teo("USA", "cpi", self._D1),
            _teo("USA", "cpi", self._D2),
            _teo("USA", "cpi", self._D3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_te_observations(end_date=self._D2)
        assert all(r.date <= self._D2 for r in results)
        assert len(results) == 2

    def test_all_four_filters_combined(self, db_session: Session) -> None:
        _flush(
            db_session,
            _teo("USA", "cpi", self._D1),
            _teo("USA", "gdp", self._D2),
            _teo("EU", "cpi", self._D3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_te_observations(
            country="USA",
            indicator_keys=["cpi"],
            start_date=self._D1,
            end_date=self._D2,
        )
        assert len(results) == 1
        assert results[0].indicator_key == "cpi"

    def test_no_match_returns_empty(self, db_session: Session) -> None:
        _flush(db_session, _teo("USA", "cpi", self._D1))
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_te_observations(country="JP")) == []


# ===========================================================================
# get_bond_yield_observations
# ===========================================================================


class TestGetBondYieldObservations:
    _D1 = _D(2026, 1, 1)
    _D2 = _D(2026, 2, 1)
    _D3 = _D(2026, 3, 1)

    def test_when_empty_returns_empty(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_bond_yield_observations()) == []

    def test_no_filters_returns_all(self, db_session: Session) -> None:
        _flush(
            db_session,
            _byo("USA", "10Y", self._D1),
            _byo("EU", "2Y", self._D2),
        )
        repo = MacroRegimeRepository(db_session)
        assert len(repo.get_bond_yield_observations()) == 2

    def test_country_filter(self, db_session: Session) -> None:
        _flush(db_session, _byo("USA", "10Y", self._D1), _byo("EU", "2Y", self._D1))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_bond_yield_observations(country="USA")
        assert all(r.country == "USA" for r in results)

    def test_maturities_filter_single(self, db_session: Session) -> None:
        _flush(
            db_session,
            _byo("USA", "10Y", self._D1),
            _byo("USA", "2Y", self._D1),
            _byo("USA", "30Y", self._D1),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_bond_yield_observations(maturities=["10Y"])
        assert all(r.maturity == "10Y" for r in results)
        assert len(results) == 1

    def test_maturities_filter_multiple(self, db_session: Session) -> None:
        _flush(
            db_session,
            _byo("USA", "10Y", self._D1),
            _byo("USA", "2Y", self._D1),
            _byo("USA", "30Y", self._D1),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_bond_yield_observations(maturities=["10Y", "2Y"])
        mats = {r.maturity for r in results}
        assert mats == {"10Y", "2Y"}

    def test_maturities_none_returns_all(self, db_session: Session) -> None:
        _flush(db_session, _byo("USA", "10Y", self._D1), _byo("USA", "2Y", self._D1))
        repo = MacroRegimeRepository(db_session)
        assert len(repo.get_bond_yield_observations(maturities=None)) == 2

    def test_start_date_filter(self, db_session: Session) -> None:
        _flush(
            db_session,
            _byo("USA", "10Y", self._D1),
            _byo("USA", "10Y", self._D2),
            _byo("USA", "10Y", self._D3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_bond_yield_observations(start_date=self._D2)
        assert all(r.date >= self._D2 for r in results)
        assert len(results) == 2

    def test_end_date_filter(self, db_session: Session) -> None:
        _flush(
            db_session,
            _byo("USA", "10Y", self._D1),
            _byo("USA", "10Y", self._D2),
            _byo("USA", "10Y", self._D3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_bond_yield_observations(end_date=self._D2)
        assert all(r.date <= self._D2 for r in results)
        assert len(results) == 2

    def test_all_four_filters_combined(self, db_session: Session) -> None:
        _flush(
            db_session,
            _byo("USA", "10Y", self._D1),
            _byo("USA", "2Y", self._D2),
            _byo("EU", "10Y", self._D3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_bond_yield_observations(
            country="USA",
            maturities=["10Y"],
            start_date=self._D1,
            end_date=self._D2,
        )
        assert len(results) == 1
        assert results[0].maturity == "10Y"

    def test_no_match_returns_empty(self, db_session: Session) -> None:
        _flush(db_session, _byo("USA", "10Y", self._D1))
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_bond_yield_observations(country="JP")) == []


# ===========================================================================
# get_fred_observations
# ===========================================================================


class TestGetFredObservations:
    _D1 = _D(2026, 1, 1)
    _D2 = _D(2026, 2, 1)
    _D3 = _D(2026, 3, 1)
    _D4 = _D(2026, 4, 1)

    def test_when_empty_returns_empty(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_fred_observations()) == []

    def test_no_filters_returns_all(self, db_session: Session) -> None:
        _flush(
            db_session,
            _fred("T10Y2Y", self._D1),
            _fred("T10Y2Y", self._D2),
            _fred("DGS10", self._D1),
        )
        repo = MacroRegimeRepository(db_session)
        assert len(repo.get_fred_observations()) == 3

    def test_series_id_filter(self, db_session: Session) -> None:
        _flush(db_session, _fred("T10Y2Y", self._D1), _fred("DGS10", self._D1))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(series_id="DGS10")
        assert all(r.series_id == "DGS10" for r in results)
        assert len(results) == 1

    def test_series_id_filter_no_match(self, db_session: Session) -> None:
        _flush(db_session, _fred("T10Y2Y", self._D1))
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_fred_observations(series_id="UNKNOWN")) == []

    def test_start_date_filter(self, db_session: Session) -> None:
        _flush(
            db_session,
            _fred("S", self._D1),
            _fred("S", self._D2),
            _fred("S", self._D3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(start_date=self._D2)
        assert all(r.date >= self._D2 for r in results)
        assert len(results) == 2

    def test_end_date_filter(self, db_session: Session) -> None:
        _flush(
            db_session,
            _fred("S", self._D1),
            _fred("S", self._D2),
            _fred("S", self._D3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(end_date=self._D2)
        assert all(r.date <= self._D2 for r in results)
        assert len(results) == 2

    def test_limit_returns_most_recent_in_asc_order(self, db_session: Session) -> None:
        _flush(
            db_session,
            _fred("S", self._D1, 1.0),
            _fred("S", self._D2, 2.0),
            _fred("S", self._D3, 3.0),
            _fred("S", self._D4, 4.0),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(series_id="S", limit=2)
        # Most-recent 2 rows returned in ascending date order
        assert len(results) == 2
        assert results[0].date < results[1].date
        assert results[0].date >= self._D3  # D3 and D4 are the two most recent

    def test_limit_none_uses_ascending_order(self, db_session: Session) -> None:
        _flush(
            db_session,
            _fred("S", self._D3),
            _fred("S", self._D1),
            _fred("S", self._D2),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(series_id="S", limit=None)
        dates = [r.date for r in results]
        assert dates == sorted(dates)

    def test_limit_with_series_and_date_filters(self, db_session: Session) -> None:
        _flush(
            db_session,
            _fred("A", self._D1),
            _fred("A", self._D2),
            _fred("A", self._D3),
            _fred("B", self._D1),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(
            series_id="A", start_date=self._D2, limit=10
        )
        assert all(r.series_id == "A" for r in results)
        assert all(r.date >= self._D2 for r in results)

    def test_all_filters_no_limit(self, db_session: Session) -> None:
        _flush(db_session, _fred("X", self._D1), _fred("X", self._D2))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_fred_observations(
            series_id="X", start_date=self._D1, end_date=self._D1
        )
        assert len(results) == 1


# ===========================================================================
# get_fred_latest_date
# ===========================================================================


class TestGetFredLatestDate:
    _D1 = _D(2026, 1, 1)
    _D2 = _D(2026, 2, 1)

    def test_when_no_rows_returns_none(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.get_fred_latest_date("NONEXISTENT") is None

    def test_returns_maximum_date_for_series(self, db_session: Session) -> None:
        _flush(db_session, _fred("S", self._D1), _fred("S", self._D2))
        repo = MacroRegimeRepository(db_session)
        assert repo.get_fred_latest_date("S") == self._D2

    def test_isolates_by_series_id(self, db_session: Session) -> None:
        _flush(db_session, _fred("A", self._D1), _fred("B", self._D2))
        repo = MacroRegimeRepository(db_session)
        assert repo.get_fred_latest_date("A") == self._D1
        assert repo.get_fred_latest_date("B") == self._D2


# ===========================================================================
# get_macro_news
# ===========================================================================


class TestGetMacroNews:
    _PUB1 = _DT(2026, 1, 1, 12, 0, 0)
    _PUB2 = _DT(2026, 2, 1, 12, 0, 0)
    _PUB3 = _DT(2026, 3, 1, 12, 0, 0)

    def test_when_empty_returns_empty(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_macro_news()) == []

    def test_no_filters_returns_up_to_limit(self, db_session: Session) -> None:
        for i in range(5):
            _flush(db_session, _news(f"id{i}", self._PUB1))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(limit=3)
        assert len(results) == 3

    def test_orders_by_publish_time_desc(self, db_session: Session) -> None:
        _flush(
            db_session,
            _news("n1", self._PUB1),
            _news("n2", self._PUB2),
            _news("n3", self._PUB3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(limit=50)
        dates = [r.publish_time for r in results]
        assert dates == sorted(dates, reverse=True)

    def test_theme_filter_returns_matching(self, db_session: Session) -> None:
        article = _news("themed_001", self._PUB1)
        db_session.add(article)
        db_session.flush()
        db_session.add(MacroNewsTheme(news_id=article.id, theme="inflation"))
        db_session.flush()

        unrelated = _news("other_002", self._PUB2)
        db_session.add(unrelated)
        db_session.flush()

        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(theme="inflation")
        ids = [r.news_id for r in results]
        assert "themed_001" in ids
        assert "other_002" not in ids

    def test_theme_filter_no_match_returns_empty(self, db_session: Session) -> None:
        _flush(db_session, _news("n1", self._PUB1))
        repo = MacroRegimeRepository(db_session)
        assert list(repo.get_macro_news(theme="nonexistent")) == []

    def test_start_date_filter(self, db_session: Session) -> None:
        _flush(
            db_session,
            _news("early", self._PUB1),
            _news("late", self._PUB3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(start_date=self._PUB2, limit=50)
        ids = {r.news_id for r in results}
        assert "late" in ids
        assert "early" not in ids

    def test_end_date_filter(self, db_session: Session) -> None:
        _flush(
            db_session,
            _news("early", self._PUB1),
            _news("late", self._PUB3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(end_date=self._PUB2, limit=50)
        ids = {r.news_id for r in results}
        assert "early" in ids
        assert "late" not in ids

    def test_start_and_end_date_combined(self, db_session: Session) -> None:
        _flush(
            db_session,
            _news("jan", self._PUB1),
            _news("feb", self._PUB2),
            _news("mar", self._PUB3),
        )
        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(
            start_date=self._PUB2, end_date=self._PUB2, limit=50
        )
        assert len(results) == 1
        assert results[0].news_id == "feb"

    def test_no_theme_filter_skips_join(self, db_session: Session) -> None:
        _flush(db_session, _news("plain", self._PUB1))
        repo = MacroRegimeRepository(db_session)
        results = repo.get_macro_news(limit=10)
        assert any(r.news_id == "plain" for r in results)


# ===========================================================================
# delete_old_macro_news
# ===========================================================================


class TestDeleteOldMacroNews:
    _PUB1 = _DT(2025, 6, 1, 0, 0, 0)
    _PUB2 = _DT(2026, 1, 1, 0, 0, 0)
    _PUB3 = _DT(2026, 6, 1, 0, 0, 0)

    def test_when_empty_returns_zero(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.delete_old_macro_news(_DT(2026, 1, 1)) == 0

    def test_deletes_rows_before_cutoff(self, db_session: Session) -> None:
        _flush(db_session, _news("old", self._PUB1), _news("new", self._PUB3))
        repo = MacroRegimeRepository(db_session)
        count = repo.delete_old_macro_news(self._PUB2)
        db_session.flush()
        assert count == 1
        remaining = repo.get_macro_news(limit=100)
        ids = {r.news_id for r in remaining}
        assert "old" not in ids
        assert "new" in ids

    def test_preserves_rows_on_cutoff_boundary(self, db_session: Session) -> None:
        _flush(db_session, _news("boundary", self._PUB2))
        repo = MacroRegimeRepository(db_session)
        count = repo.delete_old_macro_news(self._PUB2)
        assert count == 0

    def test_deletes_multiple_rows(self, db_session: Session) -> None:
        for i in range(4):
            _flush(db_session, _news(f"old{i}", self._PUB1))
        _flush(db_session, _news("keep", self._PUB3))
        repo = MacroRegimeRepository(db_session)
        assert repo.delete_old_macro_news(self._PUB2) == 4


# ===========================================================================
# get_macro_calibration
# ===========================================================================


class TestGetMacroCalibration:
    def test_when_no_row_returns_none(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.get_macro_calibration("USA") is None

    def test_returns_matching_country_row(self, db_session: Session) -> None:
        _flush(db_session, _calib("USA", regime="expansion"))
        repo = MacroRegimeRepository(db_session)
        row = repo.get_macro_calibration("USA")
        assert row is not None
        assert row.country == "USA"
        assert row.regime_classification == "expansion"

    def test_isolates_by_country(self, db_session: Session) -> None:
        _flush(db_session, _calib("USA"), _calib("EU"))
        repo = MacroRegimeRepository(db_session)
        eu = repo.get_macro_calibration("EU")
        assert eu is not None and eu.country == "EU"

    def test_unknown_country_returns_none(self, db_session: Session) -> None:
        _flush(db_session, _calib("USA"))
        repo = MacroRegimeRepository(db_session)
        assert repo.get_macro_calibration("JP") is None


# ===========================================================================
# upsert_regime_classification (SQLite-safe: uses plain session.add)
# ===========================================================================


class TestUpsertRegimeClassification:
    def test_inserts_new_row_when_missing(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_regime_classification("USA", "expansion")
        db_session.flush()
        row = repo.get_macro_calibration("USA")
        assert row is not None
        assert row.regime_classification == "expansion"

    def test_inserted_row_has_placeholder_baml_values(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_regime_classification("USA", "slowdown")
        db_session.flush()
        row = repo.get_macro_calibration("USA")
        assert row is not None
        assert row.phase == ""
        assert row.delta == 0.0
        assert row.tau == 0.0
        assert row.confidence == 0.0

    def test_updates_existing_row_preserving_baml_fields(
        self, db_session: Session
    ) -> None:
        existing = _calib("USA", phase="mid", delta=1.5, tau=0.3, confidence=0.9)
        _flush(db_session, existing)
        repo = MacroRegimeRepository(db_session)
        repo.upsert_regime_classification("USA", "recession")
        db_session.flush()
        db_session.refresh(existing)
        assert existing.regime_classification == "recession"
        assert existing.phase == "mid"
        assert existing.delta == 1.5

    def test_overwrites_classification_on_second_call(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_regime_classification("USA", "recovery")
        db_session.flush()
        repo.upsert_regime_classification("USA", "expansion")
        db_session.flush()
        row = repo.get_macro_calibration("USA")
        assert row is not None
        assert row.regime_classification == "expansion"

    def test_two_countries_independent(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        repo.upsert_regime_classification("USA", "expansion")
        repo.upsert_regime_classification("EU", "recession")
        db_session.flush()
        assert repo.get_macro_calibration("USA").regime_classification == "expansion"
        assert repo.get_macro_calibration("EU").regime_classification == "recession"


# ===========================================================================
# get_country_summary
# ===========================================================================


class TestGetCountrySummary:
    def test_when_no_data_returns_empty_lists(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        summary = repo.get_country_summary("USA")
        assert summary["economic_indicators"] == []
        assert summary["te_indicators"] == []
        assert summary["bond_yields"] == []

    def test_returns_only_matching_country_data(self, db_session: Session) -> None:
        _flush(
            db_session,
            _ei("USA"),
            _ei("EU"),
            _tei("USA", "cpi"),
            _tei("EU", "cpi"),
            _by("USA", "10Y"),
        )
        repo = MacroRegimeRepository(db_session)
        summary = repo.get_country_summary("USA")
        assert all(r.country == "USA" for r in summary["economic_indicators"])
        assert all(r.country == "USA" for r in summary["te_indicators"])
        assert all(r.country == "USA" for r in summary["bond_yields"])

    def test_has_all_three_keys(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        summary = repo.get_country_summary("X")
        assert set(summary.keys()) == {"economic_indicators", "te_indicators", "bond_yields"}


# ===========================================================================
# get_distinct_countries
# ===========================================================================


class TestGetDistinctCountries:
    def test_when_empty_returns_empty_list(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.get_distinct_countries() == []

    def test_deduplicates_across_three_tables(self, db_session: Session) -> None:
        _flush(
            db_session,
            _ei("USA"),
            _tei("USA", "cpi"),
            _tei("EU", "gdp"),
            _by("JP", "10Y"),
        )
        repo = MacroRegimeRepository(db_session)
        countries = repo.get_distinct_countries()
        assert sorted(countries) == ["EU", "JP", "USA"]
        assert len(countries) == len(set(countries))  # no duplicates

    def test_returns_sorted_order(self, db_session: Session) -> None:
        _flush(db_session, _by("USA", "10Y"), _by("AU", "2Y"), _by("EU", "30Y"))
        repo = MacroRegimeRepository(db_session)
        countries = repo.get_distinct_countries()
        assert countries == sorted(countries)

    def test_country_only_in_ei_appears(self, db_session: Session) -> None:
        _flush(db_session, _ei("OnlyInEI"))
        repo = MacroRegimeRepository(db_session)
        assert "OnlyInEI" in repo.get_distinct_countries()

    def test_country_only_in_te_appears(self, db_session: Session) -> None:
        _flush(db_session, _tei("OnlyInTE", "cpi"))
        repo = MacroRegimeRepository(db_session)
        assert "OnlyInTE" in repo.get_distinct_countries()

    def test_country_only_in_bond_yields_appears(self, db_session: Session) -> None:
        _flush(db_session, _by("OnlyInBond", "10Y"))
        repo = MacroRegimeRepository(db_session)
        assert "OnlyInBond" in repo.get_distinct_countries()


# ===========================================================================
# MacroNewsSummary — get_macro_news_summary branches
# (complement to test_macro_news_summary_repository.py)
# ===========================================================================


class TestGetMacroNewsSummaryBranches:
    """Drive branches not yet covered by the existing summary test file."""

    def test_with_exact_date_bypasses_order_by(self, db_session: Session) -> None:
        d = _D(2026, 3, 1)
        _flush(db_session, _summary("USA", d, summary="exact"))
        repo = MacroRegimeRepository(db_session)
        row = repo.get_macro_news_summary("USA", summary_date=d)
        assert row is not None
        assert row.summary == "exact"

    def test_without_date_uses_desc_order(self, db_session: Session) -> None:
        _flush(
            db_session,
            _summary("USA", _D(2026, 1, 1), summary="jan"),
            _summary("USA", _D(2026, 3, 1), summary="mar"),
        )
        repo = MacroRegimeRepository(db_session)
        row = repo.get_macro_news_summary("USA")
        assert row is not None
        assert row.summary == "mar"

    def test_date_present_but_no_row_returns_none(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.get_macro_news_summary("USA", summary_date=_D(2026, 1, 1)) is None


# ===========================================================================
# delete_old_news_summaries (complementary edge cases)
# ===========================================================================


class TestDeleteOldNewsSummariesBranches:
    def test_empty_table_returns_zero(self, db_session: Session) -> None:
        repo = MacroRegimeRepository(db_session)
        assert repo.delete_old_news_summaries(_D(2026, 1, 1)) == 0

    def test_all_rows_newer_returns_zero(self, db_session: Session) -> None:
        _flush(db_session, _summary("USA", _D(2026, 6, 1)))
        repo = MacroRegimeRepository(db_session)
        assert repo.delete_old_news_summaries(_D(2026, 1, 1)) == 0

    def test_mixed_rows_only_old_deleted(self, db_session: Session) -> None:
        _flush(
            db_session,
            _summary("USA", _D(2025, 1, 1)),
            _summary("USA", _D(2026, 6, 1)),
        )
        repo = MacroRegimeRepository(db_session)
        deleted = repo.delete_old_news_summaries(_D(2026, 1, 1))
        db_session.flush()
        assert deleted == 1
        remaining = repo.get_all_news_summaries()
        assert all(r.summary_date >= _D(2026, 1, 1) for r in remaining)
