"""Service layer orchestrating macro regime data fetching and storage."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.repositories.macro_regime_repository import MacroRegimeRepository
from app.services.scrapers.ilsole_scraper import IlSoleScraper, PORTFOLIO_COUNTRIES
from app.services.scrapers.tradingeconomics_scraper import TradingEconomicsIndicatorsScraper

logger = logging.getLogger(__name__)


class MacroRegimeService:
    """Fetches macroeconomic data from scrapers and stores via repository."""

    def __init__(self, session: Session):
        self.session = session
        self.repo = MacroRegimeRepository(session)
        self.ilsole_scraper = IlSoleScraper()
        self.te_scraper = TradingEconomicsIndicatorsScraper()

    def fetch_and_store(
        self,
        countries: Optional[List[str]] = None,
        include_bonds: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch macro data for all specified countries and store in database.

        Args:
            countries: List of countries to fetch. None means PORTFOLIO_COUNTRIES.
            include_bonds: Whether to fetch bond yield data.

        Returns:
            Dict with "counts" (per-category totals) and "errors" (list of error strings).
        """
        if countries is None:
            countries = list(PORTFOLIO_COUNTRIES)

        total_counts: Dict[str, int] = {
            "ilsole_real": 0,
            "ilsole_forecast": 0,
            "te_indicators": 0,
            "bond_yields": 0,
        }
        all_errors: List[str] = []

        for country in countries:
            try:
                result = self.fetch_country(country, include_bonds=include_bonds)

                # Accumulate counts
                for key, count in result["counts"].items():
                    total_counts[key] = total_counts.get(key, 0) + count

                # Accumulate errors with country prefix
                for err in result["errors"]:
                    all_errors.append(f"{country}: {err}")

                # Commit per country
                self.session.commit()

            except Exception as e:
                logger.error("Failed to process country %s: %s", country, e)
                all_errors.append(f"{country}: {e}")
                self.session.rollback()

        return {"counts": total_counts, "errors": all_errors}

    def fetch_country(
        self,
        country: str,
        include_bonds: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch and store macro data for a single country.

        Returns:
            Dict with "counts" and "errors" for this country.
        """
        counts: Dict[str, int] = {}
        errors: List[str] = []

        # 1. IlSole real indicators
        try:
            real_data = self.ilsole_scraper.get_real_indicators(country)
            if real_data:
                counts["ilsole_real"] = self.repo.upsert_economic_indicator(
                    country=country,
                    source="ilsole_real",
                    data=real_data,
                )
            else:
                counts["ilsole_real"] = 0
                logger.info("No IlSole real indicators for %s", country)
        except Exception as e:
            errors.append(f"ilsole_real: {e}")
            logger.warning("Failed IlSole real indicators for %s: %s", country, e)

        # 2. IlSole forecasts
        try:
            forecast_data = self.ilsole_scraper.get_forecasts(country)
            if forecast_data:
                counts["ilsole_forecast"] = self.repo.upsert_economic_indicator(
                    country=country,
                    source="ilsole_forecast",
                    data=forecast_data,
                )
            else:
                counts["ilsole_forecast"] = 0
                logger.info("No IlSole forecasts for %s", country)
        except Exception as e:
            errors.append(f"ilsole_forecast: {e}")
            logger.warning("Failed IlSole forecasts for %s: %s", country, e)

        # 3. Trading Economics indicators (+ bonds)
        try:
            te_data = self.te_scraper.get_country_indicators(
                country, include_bonds=include_bonds
            )

            if te_data.get("status") == "success":
                # Store indicators
                indicators = te_data.get("indicators", {})
                if indicators:
                    counts["te_indicators"] = self.repo.upsert_te_indicators(
                        country=country,
                        indicators_dict=indicators,
                    )
                else:
                    counts["te_indicators"] = 0

                # Store bond yields
                if include_bonds:
                    bond_yields = te_data.get("bond_yields", {})
                    if bond_yields:
                        counts["bond_yields"] = self.repo.upsert_bond_yields(
                            country=country,
                            yields_dict=bond_yields,
                        )
                    else:
                        counts["bond_yields"] = 0
                else:
                    counts["bond_yields"] = 0
            else:
                counts["te_indicators"] = 0
                counts["bond_yields"] = 0
                te_error = te_data.get("error", "Unknown error")
                errors.append(f"trading_economics: {te_error}")
                logger.warning(
                    "Trading Economics failed for %s: %s", country, te_error
                )

        except Exception as e:
            errors.append(f"trading_economics: {e}")
            counts.setdefault("te_indicators", 0)
            counts.setdefault("bond_yields", 0)
            logger.warning("Failed Trading Economics for %s: %s", country, e)

        return {"counts": counts, "errors": errors}
