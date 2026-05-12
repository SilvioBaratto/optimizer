"""Import-time guard against drift between preflight requirements and FRED catalog.

If anyone removes a required series from ``FRED_SERIES`` (or adds one to
``_REQUIRED_FRED_SERIES`` without seeding the scraper registry), this test
fails with a list of the missing IDs.
"""

from __future__ import annotations

from app.services.macro.scrapers.fred_scraper import FRED_SERIES
from research.preflight import _REQUIRED_FRED_SERIES


def test_required_fred_series_subset_of_catalog() -> None:
    missing = set(_REQUIRED_FRED_SERIES) - FRED_SERIES.keys()
    assert not missing, (
        f"FRED_SERIES missing required IDs: {sorted(missing)}. "
        f"Add them to a sub-dict in app/services/scrapers/fred_scraper.py "
        f"or remove them from _REQUIRED_FRED_SERIES in research/_preflight.py."
    )
