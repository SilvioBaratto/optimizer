"""Repository for macro regime data access with PostgreSQL upsert support."""

import logging
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models.macro_regime import (
    BondYield,
    EconomicIndicator,
    TradingEconomicsIndicator,
)

logger = logging.getLogger(__name__)

# Month abbreviation mapping for parsing reference dates like "Dec 2024"
_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_reference_date(value: Any) -> Optional[date]:
    """Parse a reference date string like 'Dec 2024' into date(2024, 12, 1).

    Also handles ISO-format strings and date objects passed through.
    Returns None if parsing fails.
    """
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()

    # Try "Mon YYYY" format (e.g. "Dec 2024", "Jan 2025")
    parts = text.split()
    if len(parts) == 2:
        month_str, year_str = parts
        month = _MONTH_ABBR.get(month_str[:3].lower())
        if month is not None:
            try:
                return date(int(year_str), month, 1)
            except (ValueError, TypeError):
                pass

    # Try ISO format (e.g. "2024-12-01")
    try:
        return date.fromisoformat(text)
    except (ValueError, TypeError):
        pass

    # Try parsing via datetime for other formats (e.g. "1/15/2025")
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except (ValueError, TypeError):
            continue

    logger.debug("Could not parse reference_date: %r", value)
    return None


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
        rows: List[Dict[str, Any]],
        constraint_name: str,
        update_columns: Optional[List[str]] = None,
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
        data: Dict[str, Any],
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

        row: Dict[str, Any] = {
            "id": uuid.uuid4(),
            "country": country,
            "source": source,
        }

        if source == "ilsole_real":
            row.update({
                "gdp_growth_qq": data.get("gdp_growth_qq"),
                "industrial_production": data.get("industrial_production"),
                "unemployment": data.get("unemployment"),
                "consumer_prices": data.get("consumer_prices"),
                "deficit": data.get("deficit"),
                "debt": data.get("debt"),
                "st_rate": data.get("st_rate"),
                "lt_rate": data.get("lt_rate"),
            })
        elif source == "ilsole_forecast":
            row.update({
                "last_inflation": data.get("last_inflation"),
                "inflation_6m": data.get("inflation_6m"),
                "inflation_10y_avg": data.get("inflation_10y_avg"),
                "gdp_growth_6m": data.get("gdp_growth_6m"),
                "earnings_12m": data.get("earnings_12m"),
                "eps_expected_12m": data.get("eps_expected_12m"),
                "peg_ratio": data.get("peg_ratio"),
                "st_rate_forecast": data.get("st_rate_forecast"),
                "lt_rate_forecast": data.get("lt_rate_forecast"),
                "reference_date": _parse_reference_date(data.get("reference_date")),
            })

        return self._upsert(
            EconomicIndicator,
            [row],
            constraint_name="uq_economic_indicator_country_source",
        )

    def get_economic_indicators(
        self, country: Optional[str] = None
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
        indicators_dict: Dict[str, Dict[str, Any]],
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

        rows: List[Dict[str, Any]] = []

        for indicator_key, indicator_data in indicators_dict.items():
            rows.append({
                "id": uuid.uuid4(),
                "country": country,
                "indicator_key": indicator_key,
                "value": indicator_data.get("value"),
                "previous": indicator_data.get("previous"),
                "unit": indicator_data.get("unit", ""),
                "reference": indicator_data.get("reference", ""),
                "raw_name": indicator_data.get("raw_name", ""),
            })

        return self._upsert(
            TradingEconomicsIndicator,
            rows,
            constraint_name="uq_te_indicator_country_key",
        )

    def get_te_indicators(
        self, country: Optional[str] = None
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
        yields_dict: Dict[str, Dict[str, Any]],
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

        rows: List[Dict[str, Any]] = []

        for maturity, yield_data in yields_dict.items():
            rows.append({
                "id": uuid.uuid4(),
                "country": country,
                "maturity": maturity,
                "yield_value": yield_data.get("yield"),
                "day_change": yield_data.get("day_change"),
                "month_change": yield_data.get("month_change"),
                "year_change": yield_data.get("year_change"),
                "reference_date": _parse_reference_date(yield_data.get("date", "")),
            })

        return self._upsert(
            BondYield,
            rows,
            constraint_name="uq_bond_yield_country_maturity",
        )

    def get_bond_yields(
        self, country: Optional[str] = None
    ) -> Sequence[BondYield]:
        """Query bond yields with optional country filter."""
        stmt = select(BondYield)
        if country:
            stmt = stmt.where(BondYield.country == country)
        stmt = stmt.order_by(BondYield.country, BondYield.maturity)
        return self.session.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Country Summary
    # ------------------------------------------------------------------

    def get_country_summary(
        self, country: str
    ) -> Dict[str, Any]:
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
