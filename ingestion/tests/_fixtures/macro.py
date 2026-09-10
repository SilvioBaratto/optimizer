"""DB seed builder for the macro domain.

Seeds one ``EconomicIndicator`` and one ``FredObservation`` for a single
country.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

from portopt_db.models.macro.macro_regime import (
    EconomicIndicator,
    FredObservation,
)
from sqlalchemy.orm import Session

from tests._fixtures._helpers import add_and_flush

_SEED_DATE = date(2024, 1, 1)


class MacroSeed(NamedTuple):
    indicator: EconomicIndicator
    fred: FredObservation


def _make_indicator(country: str) -> EconomicIndicator:
    return EconomicIndicator(
        country=country,
        last_inflation=2.0,
        gdp_growth_6m=1.5,
        reference_date=_SEED_DATE,
    )


def seed_macro(session: Session, *, country: str = "Italy") -> MacroSeed:
    indicator = add_and_flush(session, _make_indicator(country))
    fred = add_and_flush(
        session, FredObservation(series_id="T10Y2Y", date=_SEED_DATE, value=0.45)
    )
    return MacroSeed(indicator=indicator, fred=fred)
