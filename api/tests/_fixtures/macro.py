"""DB seed builder for the macro domain.

Seeds one ``EconomicIndicator``, one ``MacroCalibration`` and one
``FredObservation`` for a single country.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

from sqlalchemy.orm import Session

from app.models.macro.macro_regime import (
    EconomicIndicator,
    FredObservation,
    MacroCalibration,
)
from tests._fixtures._helpers import add_and_flush

_SEED_DATE = date(2024, 1, 1)


class MacroSeed(NamedTuple):
    indicator: EconomicIndicator
    calibration: MacroCalibration
    fred: FredObservation


def _make_indicator(country: str) -> EconomicIndicator:
    return EconomicIndicator(
        country=country,
        last_inflation=2.0,
        gdp_growth_6m=1.5,
        reference_date=_SEED_DATE,
    )


def _make_calibration(country: str) -> MacroCalibration:
    return MacroCalibration(
        country=country,
        phase="expansion",
        delta=0.1,
        tau=0.5,
        confidence=0.8,
        regime_classification="expansion",
    )


def seed_macro(session: Session, *, country: str = "Italy") -> MacroSeed:
    indicator = add_and_flush(session, _make_indicator(country))
    calibration = add_and_flush(session, _make_calibration(country))
    fred = add_and_flush(
        session, FredObservation(series_id="T10Y2Y", date=_SEED_DATE, value=0.45)
    )
    return MacroSeed(indicator=indicator, calibration=calibration, fred=fred)
