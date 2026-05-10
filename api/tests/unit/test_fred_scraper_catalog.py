"""Catalog membership tests for FRED_SERIES sub-dicts.

Guards against drift between research/_preflight._REQUIRED_FRED_SERIES
and the scraper registry.
"""

from __future__ import annotations

from app.services.scrapers.fred_scraper import (
    FRED_GDP_SERIES,
    FRED_RATE_SERIES,
    FRED_RECESSION_SERIES,
    FRED_SERIES,
)


class TestRateSeriesIncludesShortAndLongTreasuries:
    def test_when_DGS2_added_it_is_in_rate_series(self) -> None:
        assert "DGS2" in FRED_RATE_SERIES
        assert FRED_RATE_SERIES["DGS2"]

    def test_when_DGS10_added_it_is_in_rate_series(self) -> None:
        assert "DGS10" in FRED_RATE_SERIES
        assert FRED_RATE_SERIES["DGS10"]

    def test_when_DGS3MO_preserved_it_remains_in_rate_series(self) -> None:
        assert "DGS3MO" in FRED_RATE_SERIES


class TestRecessionSeriesIncludesUSRECDM:
    def test_when_USRECDM_added_it_is_in_recession_series(self) -> None:
        assert "USRECDM" in FRED_RECESSION_SERIES
        assert FRED_RECESSION_SERIES["USRECDM"]

    def test_when_existing_recession_ids_preserved_they_remain(self) -> None:
        for sid in ("RECPROUSM156N", "JHGDPBRINDX", "USREC"):
            assert sid in FRED_RECESSION_SERIES


class TestGdpSeriesDict:
    def test_when_FRED_GDP_SERIES_defined_it_is_a_dict(self) -> None:
        assert isinstance(FRED_GDP_SERIES, dict)

    def test_when_GDP_growth_id_added_it_is_in_gdp_series(self) -> None:
        assert "A191RL1Q225SBEA" in FRED_GDP_SERIES
        assert FRED_GDP_SERIES["A191RL1Q225SBEA"]


class TestFredSeriesUnion:
    def test_when_required_ids_added_FRED_SERIES_contains_all(self) -> None:
        required = {"DGS2", "DGS10", "USRECDM", "A191RL1Q225SBEA"}
        assert required <= FRED_SERIES.keys()

    def test_when_FRED_SERIES_built_each_value_is_non_empty_label(self) -> None:
        for sid, label in FRED_SERIES.items():
            assert isinstance(label, str) and label, sid
