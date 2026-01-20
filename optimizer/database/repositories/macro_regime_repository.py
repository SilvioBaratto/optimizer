import logging
from datetime import date, datetime
from typing import List, Dict, Optional, Union, Any

from sqlalchemy import select, desc

from optimizer.database.database import DatabaseManager
from optimizer.database.models.macro_regime import CountryRegimeAssessment, MarketIndicators
from optimizer.database.repositories.base import BaseRepository
from optimizer.domain.models.view import MacroRegimeDTO, MacroRegime

logger = logging.getLogger(__name__)


class MacroRegimeRepositoryImpl(BaseRepository[CountryRegimeAssessment]):
    """
    SQLAlchemy implementation of the MacroRegimeRepository protocol.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize macro regime repository.
        """
        super().__init__(db_manager, CountryRegimeAssessment)

    def _to_dto(
        self, regime: CountryRegimeAssessment, indicators: Optional[MarketIndicators] = None
    ) -> MacroRegimeDTO:
        """
        Convert SQLAlchemy model to domain DTO.
        """
        # Parse regime enum - regime is stored as string in DB
        regime_value = str(regime.regime)

        try:
            macro_regime = MacroRegime(regime_value.lower())
        except ValueError:
            macro_regime = MacroRegime.UNCERTAIN

        # Parse sector tilts (stored as JSON)
        sector_tilts = {}
        if regime.sector_tilts:
            if isinstance(regime.sector_tilts, dict):
                sector_tilts = regime.sector_tilts
            elif isinstance(regime.sector_tilts, str):
                import json

                try:
                    sector_tilts = json.loads(regime.sector_tilts)
                except json.JSONDecodeError:
                    pass

        # Parse recommendations (stored as JSON or list)
        def parse_list(value) -> List[str]:
            if value is None:
                return []
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                import json

                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return []
            return []

        overweights = parse_list(regime.recommended_overweights)
        underweights = parse_list(regime.recommended_underweights)
        primary_risks = parse_list(getattr(regime, "primary_risks", None))
        conflicting_signals = parse_list(getattr(regime, "conflicting_signals", None))

        # Get market indicators if provided
        vix = None
        hy_spread = None
        yield_curve = None
        ism_pmi = None

        if indicators:
            vix = float(indicators.vix) if indicators.vix else None
            hy_spread = float(indicators.hy_spread) if indicators.hy_spread else None
            yield_curve = (
                float(indicators.yield_curve_2s10s) if indicators.yield_curve_2s10s else None
            )
            ism_pmi = (
                float(indicators.ism_pmi)
                if hasattr(indicators, "ism_pmi") and indicators.ism_pmi
                else None
            )

        return MacroRegimeDTO(
            country=regime.country,
            regime=macro_regime,
            confidence=float(regime.confidence) if regime.confidence else 0.5,
            assessment_timestamp=regime.assessment_timestamp,
            recession_risk_6m=float(regime.recession_risk_6m) if regime.recession_risk_6m else None,
            recession_risk_12m=(
                float(regime.recession_risk_12m) if regime.recession_risk_12m else None
            ),
            transition_probability=None,  # Not stored in CountryRegimeAssessment model
            vix=vix,
            hy_spread=hy_spread,
            yield_curve_2s10s=yield_curve,
            ism_pmi=ism_pmi,
            sector_tilts=sector_tilts,
            factor_exposure=regime.factor_exposure if hasattr(regime, "factor_exposure") else None,
            recommended_overweights=overweights,
            recommended_underweights=underweights,
            rationale=regime.rationale if hasattr(regime, "rationale") else None,
            primary_risks=primary_risks,
            conflicting_signals=conflicting_signals,
        )

    def get_latest_by_country(self, country: str) -> Optional[MacroRegimeDTO]:
        """
        Fetch the most recent regime assessment for a country.
        """
        with self._get_session() as session:
            # Get latest regime
            regime_query = (
                select(CountryRegimeAssessment)
                .where(CountryRegimeAssessment.country == country)
                .order_by(desc(CountryRegimeAssessment.assessment_timestamp))
                .limit(1)
            )
            regime = session.execute(regime_query).scalar_one_or_none()

            if not regime:
                return None

            # Get latest market indicators
            indicators_query = (
                select(MarketIndicators).order_by(desc(MarketIndicators.data_timestamp)).limit(1)
            )
            indicators = session.execute(indicators_query).scalar_one_or_none()

            return self._to_dto(regime, indicators)

    def get_by_date_range(
        self,
        country: str,
        start_date: date,
        end_date: date,
    ) -> List[MacroRegimeDTO]:
        """
        Fetch regime assessments for a date range.
        """
        with self._get_session() as session:
            # Convert dates to datetime for comparison
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())

            query = (
                select(CountryRegimeAssessment)
                .where(CountryRegimeAssessment.country == country)
                .where(CountryRegimeAssessment.assessment_timestamp >= start_dt)
                .where(CountryRegimeAssessment.assessment_timestamp <= end_dt)
                .order_by(desc(CountryRegimeAssessment.assessment_timestamp))
            )

            results = session.execute(query).scalars().all()
            return [self._to_dto(r) for r in results]

    def get_all_countries_latest(self) -> Dict[str, MacroRegimeDTO]:
        """
        Fetch the most recent assessment for all countries.

        Returns:
            Dictionary mapping country -> MacroRegimeDTO
        """
        with self._get_session() as session:
            # Get distinct countries
            countries_query = select(CountryRegimeAssessment.country.distinct())
            countries = session.execute(countries_query).scalars().all()

            # Get latest for each
            result = {}
            for country in countries:
                regime_query = (
                    select(CountryRegimeAssessment)
                    .where(CountryRegimeAssessment.country == country)
                    .order_by(desc(CountryRegimeAssessment.assessment_timestamp))
                    .limit(1)
                )
                regime = session.execute(regime_query).scalar_one_or_none()
                if regime:
                    result[country] = self._to_dto(regime)

            return result

    def get_market_indicators_latest(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the most recent market indicators.
        """
        with self._get_session() as session:
            query = (
                select(MarketIndicators).order_by(desc(MarketIndicators.data_timestamp)).limit(1)
            )
            indicators = session.execute(query).scalar_one_or_none()

            if not indicators:
                return None

            return {
                "vix": float(indicators.vix) if indicators.vix else None,
                "hy_spread": float(indicators.hy_spread) if indicators.hy_spread else None,
                "yield_curve_2s10s": (
                    float(indicators.yield_curve_2s10s) if indicators.yield_curve_2s10s else None
                ),
                "data_timestamp": (
                    indicators.data_timestamp.isoformat() if indicators.data_timestamp else None
                ),
            }
