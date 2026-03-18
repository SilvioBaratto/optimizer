"""Repository for macro regime data access with PostgreSQL upsert support."""

import datetime
import logging
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.macro_regime import (
    BondYield,
    BondYieldObservation,
    EconomicIndicator,
    EconomicIndicatorObservation,
    FredObservation,
    MacroCalibration,
    MacroNews,
    MacroNewsTheme,
    MacroNewsSummary,
    TradingEconomicsIndicator,
    TradingEconomicsObservation,
)
from app.repositories.base import RepositoryBase
from app.utils.date_parsing import parse_reference_date

logger = logging.getLogger(__name__)


class MacroRegimeRepository(RepositoryBase):
    """Sync repository for macro regime data. Uses PostgreSQL ON CONFLICT upsert."""

    def __init__(self, session: Session):
        super().__init__(session)

    # ------------------------------------------------------------------
    # Economic Indicators (IlSole24Ore)
    # ------------------------------------------------------------------

    def upsert_economic_indicator(
        self,
        country: str,
        data: dict[str, Any],
    ) -> int:
        """
        Upsert a single economic indicator (forecast) row for a country.

        Args:
            country: Country name (e.g. "USA", "Germany")
            data: Dict of forecast column values from the scraper

        Returns:
            Number of rows processed (always 1 on success, 0 if data is empty).
        """
        if not data:
            return 0

        row: dict[str, Any] = {
            "id": uuid.uuid4(),
            "country": country,
            "last_inflation": data.get("last_inflation"),
            "inflation_6m": data.get("inflation_6m"),
            "inflation_10y_avg": data.get("inflation_10y_avg"),
            "gdp_growth_6m": data.get("gdp_growth_6m"),
            "earnings_12m": data.get("earnings_12m"),
            "eps_expected_12m": data.get("eps_expected_12m"),
            "peg_ratio": data.get("peg_ratio"),
            "lt_rate_forecast": data.get("lt_rate_forecast"),
            "reference_date": parse_reference_date(data.get("reference_date")),
        }

        return self._upsert(
            EconomicIndicator,
            [row],
            constraint_name="uq_economic_indicator_country",
        )

    def get_economic_indicators(
        self, country: str | None = None
    ) -> Sequence[EconomicIndicator]:
        """Query economic indicators with optional country filter."""
        stmt = select(EconomicIndicator)
        if country:
            stmt = stmt.where(EconomicIndicator.country == country)
        stmt = stmt.order_by(EconomicIndicator.country)
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Economic Indicator Observations (time-series)
    # ------------------------------------------------------------------

    _ECON_OBS_COLUMNS = [
        "last_inflation", "inflation_6m", "inflation_10y_avg",
        "gdp_growth_6m", "earnings_12m", "eps_expected_12m",
        "peg_ratio", "lt_rate_forecast", "reference_date",
    ]

    def upsert_economic_indicator_observation(
        self,
        country: str,
        snapshot_date: datetime.date,
        data: dict[str, Any],
    ) -> int:
        """Upsert an IlSole forecast observation row for a country on a given date.

        Args:
            country: Country name.
            snapshot_date: The date this snapshot was taken.
            data: Dict of forecast column values from the scraper.

        Returns:
            Number of rows processed (1 on success, 0 if data is empty).
        """
        if not data:
            return 0

        row: dict[str, Any] = {
            "id": uuid.uuid4(),
            "country": country,
            "date": snapshot_date,
            "last_inflation": data.get("last_inflation"),
            "inflation_6m": data.get("inflation_6m"),
            "inflation_10y_avg": data.get("inflation_10y_avg"),
            "gdp_growth_6m": data.get("gdp_growth_6m"),
            "earnings_12m": data.get("earnings_12m"),
            "eps_expected_12m": data.get("eps_expected_12m"),
            "peg_ratio": data.get("peg_ratio"),
            "lt_rate_forecast": data.get("lt_rate_forecast"),
            "reference_date": parse_reference_date(data.get("reference_date")),
        }

        return self._upsert(
            EconomicIndicatorObservation,
            [row],
            constraint_name="uq_econ_obs_country_date",
            update_columns=self._ECON_OBS_COLUMNS,
        )

    def get_economic_indicator_observations(
        self,
        country: str | None = None,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ) -> Sequence[EconomicIndicatorObservation]:
        """Query IlSole forecast time-series observations."""
        stmt = select(EconomicIndicatorObservation)
        if country:
            stmt = stmt.where(EconomicIndicatorObservation.country == country)
        if start_date:
            stmt = stmt.where(EconomicIndicatorObservation.date >= start_date)
        if end_date:
            stmt = stmt.where(EconomicIndicatorObservation.date <= end_date)
        stmt = stmt.order_by(
            EconomicIndicatorObservation.country,
            EconomicIndicatorObservation.date,
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Trading Economics Indicators
    # ------------------------------------------------------------------

    def upsert_te_indicators(
        self,
        country: str,
        indicators_dict: dict[str, dict[str, Any]],
    ) -> int:
        """
        Bulk upsert Trading Economics indicator rows for a country.

        Args:
            country: Country name (e.g. "USA")
            indicators_dict: Dict of indicator_key -> {value, previous, unit, reference, raw_name}

        Returns:
            Number of rows processed.
        """
        if not indicators_dict:
            return 0

        rows: list[dict[str, Any]] = []

        for indicator_key, indicator_data in indicators_dict.items():
            rows.append(
                {
                    "id": uuid.uuid4(),
                    "country": country,
                    "indicator_key": indicator_key,
                    "value": indicator_data.get("value"),
                    "previous": indicator_data.get("previous"),
                    "unit": indicator_data.get("unit", ""),
                    "reference": indicator_data.get("reference", ""),
                    "raw_name": indicator_data.get("raw_name", ""),
                }
            )

        return self._upsert(
            TradingEconomicsIndicator,
            rows,
            constraint_name="uq_te_indicator_country_key",
        )

    def get_te_indicators(
        self, country: str | None = None
    ) -> Sequence[TradingEconomicsIndicator]:
        """Query Trading Economics indicators with optional country filter."""
        stmt = select(TradingEconomicsIndicator)
        if country:
            stmt = stmt.where(TradingEconomicsIndicator.country == country)
        stmt = stmt.order_by(
            TradingEconomicsIndicator.country,
            TradingEconomicsIndicator.indicator_key,
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Bond Yields
    # ------------------------------------------------------------------

    def upsert_bond_yields(
        self,
        country: str,
        yields_dict: dict[str, dict[str, Any]],
    ) -> int:
        """
        Bulk upsert bond yield rows for a country.

        Args:
            country: Country name (e.g. "USA")
            yields_dict: Dict of maturity -> {yield, day_change, month_change, year_change, date, raw_name}

        Returns:
            Number of rows processed.
        """
        if not yields_dict:
            return 0

        rows: list[dict[str, Any]] = []

        for maturity, yield_data in yields_dict.items():
            rows.append(
                {
                    "id": uuid.uuid4(),
                    "country": country,
                    "maturity": maturity,
                    "yield_value": yield_data.get("yield"),
                    "day_change": yield_data.get("day_change"),
                    "month_change": yield_data.get("month_change"),
                    "year_change": yield_data.get("year_change"),
                    "reference_date": parse_reference_date(yield_data.get("date", "")),
                }
            )

        return self._upsert(
            BondYield,
            rows,
            constraint_name="uq_bond_yield_country_maturity",
        )

    def get_bond_yields(self, country: str | None = None) -> Sequence[BondYield]:
        """Query bond yields with optional country filter."""
        stmt = select(BondYield)
        if country:
            stmt = stmt.where(BondYield.country == country)
        stmt = stmt.order_by(BondYield.country, BondYield.maturity)
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Trading Economics Observations (time-series)
    # ------------------------------------------------------------------

    def upsert_te_observations(
        self,
        country: str,
        snapshot_date: datetime.date,
        indicators_dict: dict[str, dict[str, Any]],
    ) -> int:
        """Bulk upsert TE observation rows for a country on a given date.

        Args:
            country: Country name (e.g. "USA").
            snapshot_date: The date this snapshot was taken.
            indicators_dict: Dict of indicator_key -> {value, ...}.

        Returns:
            Number of rows processed.
        """
        if not indicators_dict:
            return 0

        rows: list[dict[str, Any]] = []
        for indicator_key, indicator_data in indicators_dict.items():
            value = indicator_data.get("value")
            if value is None:
                continue
            rows.append(
                {
                    "id": uuid.uuid4(),
                    "country": country,
                    "indicator_key": indicator_key,
                    "date": snapshot_date,
                    "value": value,
                }
            )

        return self._upsert(
            TradingEconomicsObservation,
            rows,
            constraint_name="uq_te_obs_country_key_date",
            update_columns=["value"],
        )

    def get_te_observations(
        self,
        country: str | None = None,
        indicator_keys: list[str] | None = None,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ) -> Sequence[TradingEconomicsObservation]:
        """Query TE time-series observations with optional filters."""
        stmt = select(TradingEconomicsObservation)
        if country:
            stmt = stmt.where(TradingEconomicsObservation.country == country)
        if indicator_keys:
            stmt = stmt.where(
                TradingEconomicsObservation.indicator_key.in_(indicator_keys)
            )
        if start_date:
            stmt = stmt.where(TradingEconomicsObservation.date >= start_date)
        if end_date:
            stmt = stmt.where(TradingEconomicsObservation.date <= end_date)
        stmt = stmt.order_by(
            TradingEconomicsObservation.country,
            TradingEconomicsObservation.indicator_key,
            TradingEconomicsObservation.date,
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Bond Yield Observations (time-series)
    # ------------------------------------------------------------------

    def upsert_bond_yield_observations(
        self,
        country: str,
        snapshot_date: datetime.date,
        yields_dict: dict[str, dict[str, Any]],
    ) -> int:
        """Bulk upsert bond yield observation rows for a country on a given date.

        Args:
            country: Country name (e.g. "USA").
            snapshot_date: The date this snapshot was taken.
            yields_dict: Dict of maturity -> {yield, ...}.

        Returns:
            Number of rows processed.
        """
        if not yields_dict:
            return 0

        rows: list[dict[str, Any]] = []
        for maturity, yield_data in yields_dict.items():
            yield_val = yield_data.get("yield")
            if yield_val is None:
                continue
            rows.append(
                {
                    "id": uuid.uuid4(),
                    "country": country,
                    "maturity": maturity,
                    "date": snapshot_date,
                    "yield_value": yield_val,
                }
            )

        return self._upsert(
            BondYieldObservation,
            rows,
            constraint_name="uq_bond_obs_country_mat_date",
            update_columns=["yield_value"],
        )

    def get_bond_yield_observations(
        self,
        country: str | None = None,
        maturities: list[str] | None = None,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ) -> Sequence[BondYieldObservation]:
        """Query bond yield time-series observations with optional filters."""
        stmt = select(BondYieldObservation)
        if country:
            stmt = stmt.where(BondYieldObservation.country == country)
        if maturities:
            stmt = stmt.where(BondYieldObservation.maturity.in_(maturities))
        if start_date:
            stmt = stmt.where(BondYieldObservation.date >= start_date)
        if end_date:
            stmt = stmt.where(BondYieldObservation.date <= end_date)
        stmt = stmt.order_by(
            BondYieldObservation.country,
            BondYieldObservation.maturity,
            BondYieldObservation.date,
        )
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # FRED Observations
    # ------------------------------------------------------------------

    def upsert_fred_observations(
        self,
        series_id: str,
        observations: list[dict[str, Any]],
    ) -> int:
        """Bulk upsert FRED observations for a single series.

        Args:
            series_id: FRED series identifier (e.g. ``"BAMLH0A0HYM2"``).
            observations: List of ``{"date": datetime.date, "value": float|None}``.

        Returns:
            Number of rows processed.
        """
        if not observations:
            return 0

        rows: list[dict[str, Any]] = [
            {
                "id": uuid.uuid4(),
                "series_id": series_id,
                "date": obs["date"],
                "value": obs.get("value"),
            }
            for obs in observations
        ]

        return self._upsert(
            FredObservation,
            rows,
            constraint_name="uq_fred_observation_series_date",
            update_columns=["value"],
        )

    def get_fred_observations(
        self,
        series_id: str | None = None,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        limit: int | None = None,
    ) -> Sequence[FredObservation]:
        """Query FRED observations with optional filters.

        When *limit* is supplied the most-recent rows are returned in
        ascending date order, so callers can read ``result[-1]`` for the
        latest value.
        """
        stmt = select(FredObservation)
        if series_id:
            stmt = stmt.where(FredObservation.series_id == series_id)
        if start_date:
            stmt = stmt.where(FredObservation.date >= start_date)
        if end_date:
            stmt = stmt.where(FredObservation.date <= end_date)

        if limit is not None:
            stmt = stmt.order_by(
                FredObservation.series_id,
                FredObservation.date.desc(),
            ).limit(limit)
            rows = list(self.session.execute(stmt).scalars().all())
            rows.sort(key=lambda r: (r.series_id, r.date))
            return rows

        stmt = stmt.order_by(FredObservation.series_id, FredObservation.date)
        return self.session.execute(stmt).scalars().all()

    def get_fred_latest_date(self, series_id: str) -> datetime.date | None:
        """Return the most recent stored observation date for a series."""
        result = self.session.execute(
            select(sa_func.max(FredObservation.date)).where(
                FredObservation.series_id == series_id
            )
        ).scalar()
        return result

    # ------------------------------------------------------------------
    # Macro News
    # ------------------------------------------------------------------

    def upsert_macro_news(self, rows: list[dict[str, Any]]) -> int:
        """Bulk upsert macro news rows. Deduplicates on ``news_id``.

        Extracts ``themes`` from each row dict, upserts the MacroNews rows,
        then manages MacroNewsTheme child rows for each article.
        """
        # Extract themes before upsert (themes is no longer a column)
        themes_by_news_id: dict[str, list[str]] = {}
        clean_rows = []
        for row in rows:
            row_copy = dict(row)
            raw_themes = row_copy.pop("themes", None)
            if raw_themes and isinstance(raw_themes, str):
                themes_by_news_id[row_copy["news_id"]] = [
                    t.strip() for t in raw_themes.split(",") if t.strip()
                ]
            clean_rows.append(row_copy)

        count = self._upsert(
            MacroNews,
            clean_rows,
            constraint_name="uq_macro_news_id",
        )

        # Now manage theme child rows
        if themes_by_news_id:
            news_ids = list(themes_by_news_id.keys())
            stmt = select(MacroNews).where(MacroNews.news_id.in_(news_ids))
            news_rows = self.session.execute(stmt).scalars().all()
            for news_row in news_rows:
                theme_list = themes_by_news_id.get(news_row.news_id, [])
                if not theme_list:
                    continue
                # Clear existing themes and re-add
                news_row.theme_entries.clear()
                self.session.flush()
                for theme_str in theme_list:
                    self.session.add(MacroNewsTheme(
                        news_id=news_row.id, theme=theme_str,
                    ))
            self.session.flush()

        return count

    def get_macro_news(
        self,
        theme: str | None = None,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
        limit: int = 50,
    ) -> Sequence[MacroNews]:
        """Query stored macro news with optional theme/date filters."""
        stmt = select(MacroNews)
        if theme:
            stmt = stmt.join(MacroNewsTheme).where(MacroNewsTheme.theme == theme)
        if start_date:
            stmt = stmt.where(MacroNews.publish_time >= start_date)
        if end_date:
            stmt = stmt.where(MacroNews.publish_time <= end_date)
        stmt = stmt.order_by(MacroNews.publish_time.desc()).limit(limit)
        return self.session.execute(stmt).scalars().all()

    def delete_old_macro_news(self, before_date: datetime.datetime) -> int:
        """Delete macro news older than the given date. Returns count deleted."""
        stmt = select(MacroNews).where(MacroNews.publish_time < before_date)
        rows = self.session.execute(stmt).scalars().all()
        count = len(rows)
        for row in rows:
            self.session.delete(row)
        return count

    # ------------------------------------------------------------------
    # Macro News Summaries (AI-generated daily country summaries)
    # ------------------------------------------------------------------

    def upsert_macro_news_summary(
        self,
        country: str,
        summary_date: datetime.date,
        data: dict[str, Any],
    ) -> int:
        """Insert or update the daily news summary for a country on a given date.

        Args:
            country: Country name (e.g. "USA").
            summary_date: The date this summary covers.
            data: Dict of summary column values (summary, key_themes, etc.).

        Returns:
            Number of rows processed (always 1).
        """
        row: dict[str, Any] = {
            "id": uuid.uuid4(),
            "country": country,
            "summary_date": summary_date,
            "updated_at": datetime.datetime.now(datetime.timezone.utc),
            **data,
        }
        return self._upsert(
            MacroNewsSummary,
            [row],
            constraint_name="uq_macro_news_summary_country_date",
            update_columns=[
                "summary",
                "sentiment",
                "sentiment_score",
                "article_count",
                "news_summary",
                "updated_at",
            ],
        )

    def get_macro_news_summary(
        self,
        country: str,
        summary_date: datetime.date | None = None,
    ) -> MacroNewsSummary | None:
        """Return the latest (or date-specific) news summary for a country, or None."""
        stmt = select(MacroNewsSummary).where(MacroNewsSummary.country == country)
        if summary_date is not None:
            stmt = stmt.where(MacroNewsSummary.summary_date == summary_date)
        else:
            stmt = stmt.order_by(MacroNewsSummary.summary_date.desc())
        return self.session.execute(stmt).scalars().first()

    def get_all_news_summaries(
        self,
        summary_date: datetime.date | None = None,
    ) -> Sequence[MacroNewsSummary]:
        """Return news summaries for all countries, optionally filtered to a single date."""
        stmt = select(MacroNewsSummary)
        if summary_date is not None:
            stmt = stmt.where(MacroNewsSummary.summary_date == summary_date)
        stmt = stmt.order_by(
            MacroNewsSummary.country,
            MacroNewsSummary.summary_date.desc(),
        )
        return self.session.execute(stmt).scalars().all()

    def delete_old_news_summaries(self, before_date: datetime.date) -> int:
        """Delete news summaries with summary_date before the cutoff. Returns count deleted."""
        stmt = select(MacroNewsSummary).where(
            MacroNewsSummary.summary_date < before_date
        )
        rows = self.session.execute(stmt).scalars().all()
        for row in rows:
            self.session.delete(row)
        return len(rows)

    # ------------------------------------------------------------------
    # Country Summary
    # ------------------------------------------------------------------

    def get_country_summary(self, country: str) -> dict[str, Any]:
        """
        Get all three data types for a single country.

        Returns:
            Dict with keys: economic_indicators, te_indicators, bond_yields
        """
        return {
            "economic_indicators": self.get_economic_indicators(country=country),
            "te_indicators": self.get_te_indicators(country=country),
            "bond_yields": self.get_bond_yields(country=country),
        }

    # ------------------------------------------------------------------
    # Macro Calibrations (cached LLM results)
    # ------------------------------------------------------------------

    def upsert_macro_calibration(
        self,
        country: str,
        data: dict[str, Any],
    ) -> int:
        """Insert or update the cached macro calibration for a country.

        Args:
            country: Country name (e.g. "USA").
            data: Dict of calibration column values (phase, delta, tau, etc.).

        Returns:
            Number of rows processed (always 1).
        """
        row: dict[str, Any] = {
            "id": uuid.uuid4(),
            "country": country,
            **data,
        }
        return self._upsert(
            MacroCalibration,
            [row],
            constraint_name="uq_macro_calibration_country",
        )

    def get_macro_calibration(self, country: str) -> MacroCalibration | None:
        """Return the cached calibration for a country, or None."""
        stmt = select(MacroCalibration).where(MacroCalibration.country == country)
        return self.session.execute(stmt).scalar_one_or_none()

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def get_distinct_countries(self) -> list[str]:
        """Return a deduplicated sorted list of all countries with stored data."""
        from sqlalchemy import distinct, union_all

        q1 = select(EconomicIndicator.country.label("country"))
        q2 = select(TradingEconomicsIndicator.country.label("country"))
        q3 = select(BondYield.country.label("country"))

        subq = union_all(q1, q2, q3).subquery()
        stmt = select(distinct(subq.c.country)).order_by(subq.c.country)
        return list(self.session.execute(stmt).scalars().all())
