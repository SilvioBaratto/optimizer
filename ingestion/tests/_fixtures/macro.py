"""DB seed builder for the macro domain.

Seeds one ``FredObservation``.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

from portopt_db.models.macro.macro_regime import FredObservation
from sqlalchemy.orm import Session

from tests._fixtures._helpers import add_and_flush

_SEED_DATE = date(2024, 1, 1)


class MacroSeed(NamedTuple):
    """The rows ``seed_macro`` inserted.

    Attributes:
        fred: The seeded FRED series observation.
    """

    fred: FredObservation


def seed_macro(session: Session) -> MacroSeed:
    """Insert one FRED observation and flush it, without committing.

    Args:
        session: Session whose transaction the caller owns.

    Returns:
        The inserted rows.
    """
    fred = add_and_flush(
        session, FredObservation(series_id="T10Y2Y", date=_SEED_DATE, value=0.45)
    )
    return MacroSeed(fred=fred)
