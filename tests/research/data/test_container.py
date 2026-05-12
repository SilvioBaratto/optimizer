"""Tests for research/data/_container.py — DataAssembly frozen value object."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from research.data._container import DataAssembly, _compute_assembly_hash


def _make_assembly(**overrides: object) -> DataAssembly:
    defaults: dict[str, object] = {
        "prices": pd.DataFrame(
            {"A": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2)
        ),
        "volumes": pd.DataFrame(
            {"A": [100.0, 200.0]}, index=pd.date_range("2024-01-01", periods=2)
        ),
        "fundamentals": pd.DataFrame({"market_cap": [1e9]}, index=["A"]),
        "sector_mapping": {"A": "Tech"},
        "financial_statements": pd.DataFrame(),
        "analyst_data": pd.DataFrame(),
        "insider_data": pd.DataFrame(),
        "macro_data": pd.DataFrame(),
    }
    defaults.update(overrides)
    return DataAssembly(**defaults)  # type: ignore[arg-type]


class TestDataAssemblyFrozen:
    def test_when_prices_mutated_then_raises_frozen_instance_error(self) -> None:
        assembly = _make_assembly()
        with pytest.raises(dataclasses.FrozenInstanceError):
            assembly.prices = pd.DataFrame()  # type: ignore[misc]

    def test_when_assembly_hash_mutated_then_raises_frozen_instance_error(self) -> None:
        assembly = _make_assembly()
        with pytest.raises(dataclasses.FrozenInstanceError):
            assembly.assembly_hash = "new_hash"  # type: ignore[misc]

    def test_when_replaced_then_original_unchanged(self) -> None:
        assembly = _make_assembly()
        updated = dataclasses.replace(assembly, assembly_hash="abc123")
        assert assembly.assembly_hash == ""
        assert updated.assembly_hash == "abc123"

    def test_when_replaced_then_returns_new_instance(self) -> None:
        assembly = _make_assembly()
        updated = dataclasses.replace(assembly, assembly_hash="abc123")
        assert assembly is not updated


class TestDataAssemblyEquality:
    def test_two_distinct_instances_are_not_equal(self) -> None:
        a1 = _make_assembly()
        a2 = _make_assembly()
        assert a1 != a2

    def test_same_instance_is_equal_to_itself(self) -> None:
        a = _make_assembly()
        assert a == a

    def test_hash_based_on_assembly_hash_field(self) -> None:
        a = dataclasses.replace(_make_assembly(), assembly_hash="abc123")
        assert hash(a) == hash("abc123")

    def test_empty_assembly_hash_uses_id_for_hash(self) -> None:
        a = _make_assembly()
        assert hash(a) == id(a)


class TestDataAssemblyDefaults:
    def test_fred_data_defaults_to_empty_dataframe(self) -> None:
        assembly = _make_assembly()
        assert isinstance(assembly.fred_data, pd.DataFrame)
        assert assembly.fred_data.empty

    def test_assembly_hash_defaults_to_empty_string(self) -> None:
        assembly = _make_assembly()
        assert assembly.assembly_hash == ""

    def test_include_delisted_defaults_to_true(self) -> None:
        assembly = _make_assembly()
        assert assembly.include_delisted is True

    def test_delisting_returns_defaults_to_empty_mapping(self) -> None:
        assembly = _make_assembly()
        assert len(assembly.delisting_returns) == 0

    def test_currency_map_defaults_to_empty_mapping(self) -> None:
        assembly = _make_assembly()
        assert len(assembly.currency_map) == 0


class TestDataAssemblyProperties:
    def test_n_tickers_property(self) -> None:
        assembly = _make_assembly()
        assert assembly.n_tickers == 1

    def test_n_trading_days_property(self) -> None:
        assembly = _make_assembly()
        assert assembly.n_trading_days == 2

    def test_risk_free_rate_series_empty_when_no_fred_data(self) -> None:
        assembly = _make_assembly()
        series = assembly.risk_free_rate_series
        assert isinstance(series, pd.Series)
        assert series.empty

    def test_risk_free_rate_returns_zero_when_no_fred_data(self) -> None:
        assembly = _make_assembly()
        assert assembly.risk_free_rate == 0.0

    def test_risk_free_rate_series_when_dgs3mo_present(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3)
        fred = pd.DataFrame({"DGS3MO": [5.0, 5.1, 5.2]}, index=dates)
        assembly = _make_assembly(fred_data=fred)
        series = assembly.risk_free_rate_series
        assert not series.empty
        assert series.name == "risk_free_rate"
        assert all(series > 0)

    def test_risk_free_rate_scalar_when_dgs3mo_present(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3)
        fred = pd.DataFrame({"DGS3MO": [5.0, 5.1, 5.2]}, index=dates)
        assembly = _make_assembly(fred_data=fred)
        rf = assembly.risk_free_rate
        assert isinstance(rf, float)
        assert rf > 0


class TestDataAssemblySummary:
    def test_summary_returns_dict(self) -> None:
        assembly = _make_assembly()
        result = assembly.summary()
        assert isinstance(result, dict)

    def test_summary_includes_tickers(self) -> None:
        assembly = _make_assembly()
        assert assembly.summary()["tickers"] == 1

    def test_summary_includes_assembly_hash(self) -> None:
        assembly = dataclasses.replace(
            _make_assembly(), assembly_hash="deadbeef12345678"
        )
        assert assembly.summary()["assembly_hash"] == "deadbeef12345678"

    def test_summary_risk_free_rate_pct_is_none_without_fred(self) -> None:
        assembly = _make_assembly()
        assert assembly.summary()["risk_free_rate_pct"] is None


class TestComputeAssemblyHash:
    def test_returns_16_char_hex_string(self) -> None:
        import datetime

        assembly = _make_assembly()
        ts = datetime.datetime(2024, 1, 10, 12, 0, 0, tzinfo=datetime.timezone.utc)
        result = _compute_assembly_hash(assembly, timestamp=ts)
        assert isinstance(result, str)
        assert len(result) == 16
        int(result, 16)  # raises ValueError if not hex

    def test_is_deterministic_for_same_inputs(self) -> None:
        import datetime

        assembly = _make_assembly()
        ts = datetime.datetime(2024, 1, 10, tzinfo=datetime.timezone.utc)
        assert _compute_assembly_hash(assembly, timestamp=ts) == _compute_assembly_hash(
            assembly, timestamp=ts
        )

    def test_changes_with_different_timestamp(self) -> None:
        import datetime

        assembly = _make_assembly()
        ts1 = datetime.datetime(2024, 1, 10, 12, 0, 0, tzinfo=datetime.timezone.utc)
        ts2 = datetime.datetime(2024, 1, 10, 12, 0, 1, tzinfo=datetime.timezone.utc)
        h1 = _compute_assembly_hash(assembly, timestamp=ts1)
        h2 = _compute_assembly_hash(assembly, timestamp=ts2)
        assert h1 != h2
