"""Repository for macro regime data access with PostgreSQL upsert support."""

import logging
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models.macro_regime import (
    BondYield,
    EconomicIndicator,
    TradingEconomicsIndicator,
)
from app.utils.date_parsing import parse_reference_date

logger = logging.getLogger(__name__)


class MacroRegimeRepository:
    """Sync repository for macro regime data. Uses PostgreSQL ON CONFLICT upsert."""

    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------------
    # Generic upsert helper (mirrors YFinanceRepository._upsert)
    # ------------------------------------------------------------------

    def _upsert(
        self,
        model: type,
        rows: list[dict[str, Any]],
        constraint_name: str,
        update_columns: list[str] | None = None,
    ) -> int:
        """Insert rows with ON CONFLICT DO UPDATE. Returns count of rows processed."""
        if not rows:
            return 0

        stmt = pg_insert(model.__table__).values(rows)

        if update_columns:
            update_dict = {col: stmt.excluded[col] for col in update_columns}
        else:
            # Update all columns except the primary key and created_at
            exclude = {"id", "created_at"}
            update_dict = {
                col.name: stmt.excluded[col.name]
                for col in model.__table__.columns
                if col.name not in exclude
            }

        stmt = stmt.on_conflict_do_update(
            constraint=constraint_name,
            set_=update_dict,
        )

        self.session.execute(stmt)
        return len(rows)

    # ------------------------------------------------------------------
    # Economic Indicators (IlSole24Ore)
    # ------------------------------------------------------------------

    def upsert_economic_indicator(
        self,
        country: str,
        source: str,
        data: dict[str, Any],
    ) -> int:
        """
        Upsert a single economic indicator row for a country+source pair.

        Args:
            country: Country name (e.g. "USA", "Germany")
            source: "ilsole_real" or "ilsole_forecast"
            data: Dict of column values from the scraper

        Returns:
            Number of rows processed (always 1 on success, 0 if data is empty).
        """
        if not data:
            return 0

        row: dict[str, Any] = {
            "id": uuid.uuid4(),
            "country": country,
            "source": source,
        }

        if source == "ilsole_real":
            row.update(
                {
                    "gdp_growth_qq": data.get("gdp_growth_qq"),
                    "industrial_production": data.get("industrial_production"),
                    "unemployment": data.get("unemployment"),
                    "consumer_prices": data.get("consumer_prices"),
                    "deficit": data.get("deficit"),
                    "debt": data.get("debt"),
                    "st_rate": data.get("st_rate"),
                    "lt_rate": data.get("lt_rate"),
                }
            )
        elif source == "ilsole_forecast":
            row.update(
                {
                    "last_inflation": data.get("last_inflation"),
                    "inflation_6m": data.get("inflation_6m"),
                    "inflation_10y_avg": data.get("inflation_10y_avg"),
                    "gdp_growth_6m": data.get("gdp_growth_6m"),
                    "earnings_12m": data.get("earnings_12m"),
                    "eps_expected_12m": data.get("eps_expected_12m"),
                    "peg_ratio": data.get("peg_ratio"),
                    "st_rate_forecast": data.get("st_rate_forecast"),
                    "lt_rate_forecast": data.get("lt_rate_forecast"),
                    "reference_date": parse_reference_date(data.get("reference_date")),
                }
            )

        return self._upsert(
            EconomicIndicator,
            [row],
            constraint_name="uq_economic_indicator_country_source",
        )

    def get_economic_indicators(
        self, country: str | None = None
    ) -> Sequence[EconomicIndicator]:
        """Query economic indicators with optional country filter."""
        stmt = select(EconomicIndicator)
        if country:
            stmt = stmt.where(EconomicIndicator.country == country)
        stmt = stmt.order_by(EconomicIndicator.country, EconomicIndicator.source)
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
