"""Tests for REGION_MAP, FRED_SERIES_IDS additions, and assembly_hash (issue #518)."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

pytest.importorskip("typer")

from research.data_assembly import (
    FRED_SERIES_IDS,
    REGION_MAP,
    DataAssembly,
    _compute_assembly_hash,
)


class TestRegionMap:
    def test_when_imported_then_is_non_empty_dict(self) -> None:
        assert isinstance(REGION_MAP, dict)
        assert len(REGION_MAP) > 0

    def test_when_imported_then_other_is_not_a_key(self) -> None:
        assert "Other" not in REGION_MAP

    def test_when_country_us_then_americas_is_returned(self) -> None:
        assert REGION_MAP["United States"] == "Americas"

    def test_when_country_germany_then_europe_is_returned(self) -> None:
        assert REGION_MAP["Germany"] == "Europe"

    def test_when_country_japan_then_asia_pacific_is_returned(self) -> None:
        assert REGION_MAP["Japan"] == "Asia-Pacific"

    def test_when_country_israel_then_middle_east_africa_is_returned(self) -> None:
        assert REGION_MAP["Israel"] == "Middle East & Africa"

    def test_when_iterated_then_all_required_countries_are_covered(self) -> None:
        required = {
            "United States",
            "Canada",
            "Mexico",
            "Brazil",
            "United Kingdom",
            "France",
            "Germany",
            "Italy",
            "Spain",
            "Netherlands",
            "Switzerland",
            "Sweden",
            "Norway",
            "Denmark",
            "Finland",
            "Belgium",
            "Austria",
            "Ireland",
            "Portugal",
            "Japan",
            "China",
            "Hong Kong",
            "South Korea",
            "Taiwan",
            "Singapore",
            "Australia",
            "New Zealand",
            "India",
            "United Arab Emirates",
            "Saudi Arabia",
            "Israel",
            "South Africa",
            "Qatar",
        }
        missing = required - set(REGION_MAP)
        assert not missing, f"REGION_MAP missing countries: {sorted(missing)}"

    def test_when_iterated_then_all_values_are_canonical_regions(self) -> None:
        canonical = {"Americas", "Europe", "Asia-Pacific", "Middle East & Africa"}
        assert set(REGION_MAP.values()) <= canonical


class TestFredSeriesIds:
    def test_when_imported_then_dgs2_present(self) -> None:
        assert "DGS2" in FRED_SERIES_IDS

    def test_when_imported_then_dgs10_present(self) -> None:
        assert "DGS10" in FRED_SERIES_IDS

    def test_when_imported_then_usrecdm_present(self) -> None:
        assert "USRECDM" in FRED_SERIES_IDS

    def test_when_imported_then_existing_series_preserved(self) -> None:
        for sid in ("BAMLH0A0HYM2", "T10Y2Y", "VIXCLS", "USREC", "DGS3MO"):
            assert sid in FRED_SERIES_IDS


def _make_synthetic_assembly(
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> DataAssembly:
    prices = pd.DataFrame(1.0, index=dates, columns=tickers)
    volumes = pd.DataFrame(1.0, index=dates, columns=tickers)
    fundamentals = pd.DataFrame(index=tickers)
    return DataAssembly(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        sector_mapping={},
        financial_statements=pd.DataFrame(),
        analyst_data=pd.DataFrame(),
        insider_data=pd.DataFrame(),
        macro_data=pd.DataFrame(),
    )


class TestAssemblyHashField:
    def test_when_constructed_then_default_is_empty_string(self) -> None:
        assembly = _make_synthetic_assembly(
            pd.date_range("2024-01-02", periods=3, freq="B"),
            ["AAA"],
        )
        assert assembly.assembly_hash == ""

    def test_when_constructed_with_hash_then_attribute_is_set(self) -> None:
        assembly = _make_synthetic_assembly(
            pd.date_range("2024-01-02", periods=3, freq="B"),
            ["AAA"],
        )
        assembly.assembly_hash = "abcdef0123456789"
        assert assembly.assembly_hash == "abcdef0123456789"

    def test_when_summary_then_includes_assembly_hash(self) -> None:
        assembly = _make_synthetic_assembly(
            pd.date_range("2024-01-02", periods=3, freq="B"),
            ["AAA"],
        )
        assembly.assembly_hash = "deadbeefcafef00d"
        summary = assembly.summary()
        assert summary["assembly_hash"] == "deadbeefcafef00d"


class TestComputeAssemblyHash:
    def test_when_called_with_fixed_timestamp_then_returns_16_char_hex(self) -> None:
        assembly = _make_synthetic_assembly(
            pd.date_range("2024-01-02", periods=5, freq="B"),
            ["AAA", "BBB"],
        )
        ts = datetime.datetime(2024, 1, 10, 12, 0, 0, tzinfo=datetime.UTC)
        result = _compute_assembly_hash(assembly, timestamp=ts)
        assert isinstance(result, str)
        assert len(result) == 16
        int(result, 16)  # raises ValueError if non-hex

    def test_when_same_inputs_then_hash_is_deterministic(self) -> None:
        assembly = _make_synthetic_assembly(
            pd.date_range("2024-01-02", periods=5, freq="B"),
            ["AAA", "BBB"],
        )
        ts = datetime.datetime(2024, 1, 10, 12, 0, 0, tzinfo=datetime.UTC)
        h1 = _compute_assembly_hash(assembly, timestamp=ts)
        h2 = _compute_assembly_hash(assembly, timestamp=ts)
        assert h1 == h2

    def test_when_different_timestamps_then_hashes_differ(self) -> None:
        assembly = _make_synthetic_assembly(
            pd.date_range("2024-01-02", periods=5, freq="B"),
            ["AAA"],
        )
        ts1 = datetime.datetime(2024, 1, 10, 12, 0, 0, tzinfo=datetime.UTC)
        ts2 = datetime.datetime(2024, 1, 10, 12, 0, 1, tzinfo=datetime.UTC)
        assert _compute_assembly_hash(
            assembly, timestamp=ts1
        ) != _compute_assembly_hash(assembly, timestamp=ts2)

    def test_when_n_tickers_changes_then_hash_changes(self) -> None:
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        a1 = _make_synthetic_assembly(dates, ["AAA"])
        a2 = _make_synthetic_assembly(dates, ["AAA", "BBB"])
        ts = datetime.datetime(2024, 1, 10, tzinfo=datetime.UTC)
        assert _compute_assembly_hash(a1, timestamp=ts) != _compute_assembly_hash(
            a2, timestamp=ts
        )

    def test_when_last_date_changes_then_hash_changes(self) -> None:
        a1 = _make_synthetic_assembly(
            pd.date_range("2024-01-02", periods=3, freq="B"), ["AAA"]
        )
        a2 = _make_synthetic_assembly(
            pd.date_range("2024-01-02", periods=10, freq="B"), ["AAA"]
        )
        ts = datetime.datetime(2024, 1, 10, tzinfo=datetime.UTC)
        assert _compute_assembly_hash(a1, timestamp=ts) != _compute_assembly_hash(
            a2, timestamp=ts
        )

    def test_when_timestamp_default_then_hash_returned(self) -> None:
        assembly = _make_synthetic_assembly(
            pd.date_range("2024-01-02", periods=3, freq="B"),
            ["AAA"],
        )
        result = _compute_assembly_hash(assembly)
        assert isinstance(result, str)
        assert len(result) == 16
