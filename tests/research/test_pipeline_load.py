"""Tests for research.pipeline._load — Step 1 data loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from research.data._container import DataAssembly

_RNG = np.random.default_rng(0)


def _make_assembly(n_tickers: int = 2, n_trading_days: int = 3) -> DataAssembly:
    dates = pd.date_range("2024-01-02", periods=n_trading_days, freq="B")
    named = ["AAA", "BBB"]
    extra = [f"T{i:04d}" for i in range(max(0, n_tickers - 2))]
    tickers = (named + extra)[:n_tickers]
    data = _RNG.standard_normal((n_trading_days, n_tickers)) + 100.0
    prices = pd.DataFrame(data, index=dates, columns=tickers)
    volumes = pd.DataFrame(
        np.ones((n_trading_days, n_tickers)) * 1000.0, index=dates, columns=tickers
    )
    fx_rates = pd.DataFrame(
        {"EUR": np.ones(n_trading_days), "USD": np.full(n_trading_days, 1.1)},
        index=dates,
    )
    return DataAssembly(
        prices=prices,
        volumes=volumes,
        fundamentals=pd.DataFrame({"market_cap": [1e9, 5e8]}, index=["AAA", "BBB"]),
        sector_mapping=dict.fromkeys(tickers, "Tech"),
        financial_statements=pd.DataFrame(),
        analyst_data=pd.DataFrame(),
        insider_data=pd.DataFrame(),
        macro_data=pd.DataFrame(),
        currency_map={t: ("USD" if t == "AAA" else "EUR") for t in tickers},
        fx_rates=fx_rates,
        assembly_hash="test_hash_123",
    )


class TestAssertAssemblySize:
    def test_when_above_floor_then_no_error(self) -> None:
        from research.pipeline._load import _assert_assembly_size

        assembly = _make_assembly(n_tickers=3000, n_trading_days=2000)
        _assert_assembly_size(assembly)

    def test_when_n_tickers_below_floor_then_raises_runtime_error(self) -> None:
        from research.pipeline._load import _assert_assembly_size

        assembly = _make_assembly(n_tickers=1500, n_trading_days=2000)
        with pytest.raises(RuntimeError, match="n_tickers"):
            _assert_assembly_size(assembly)

    def test_when_n_trading_days_below_floor_then_raises_runtime_error(self) -> None:
        from research.pipeline._load import _assert_assembly_size

        assembly = _make_assembly(n_tickers=3000, n_trading_days=500)
        with pytest.raises(RuntimeError, match="n_trading_days"):
            _assert_assembly_size(assembly)


class TestMaterialiseCleanReturns:
    def test_when_valid_data_then_returns_clean_dataframe(self) -> None:
        from research.pipeline._load import _materialise_clean_returns

        assembly = _make_assembly(n_tickers=3000, n_trading_days=2000)
        investable = pd.Index(["AAA", "BBB"])
        result = _materialise_clean_returns(assembly, investable)
        assert isinstance(result, pd.DataFrame)
        assert "AAA" in result.columns
        assert not result.isna().any().any()
        assert not np.isinf(result.to_numpy()).any()

    def test_when_single_ticker_then_still_works(self) -> None:
        from research.pipeline._load import _materialise_clean_returns

        assembly = _make_assembly(n_tickers=3000, n_trading_days=2000)
        investable = pd.Index(["AAA"])
        result = _materialise_clean_returns(assembly, investable)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1


class TestReviewDatePersistence:
    def test_when_file_exists_then_reads_date(self, tmp_path) -> None:
        from research.pipeline._load import (
            _read_last_review_date,
            _write_last_review_date,
        )

        review_file = tmp_path / "last_review_date.txt"
        with patch("research.pipeline._load._LAST_REVIEW_DATE_FILE", review_file):
            test_date = pd.Timestamp("2024-03-15")
            _write_last_review_date(test_date)
            result = _read_last_review_date(pd.Timestamp("2024-06-01"))
            assert result == test_date.normalize()

    def test_when_file_missing_then_returns_prior_quarter_end(self) -> None:
        from research.pipeline._load import _read_last_review_date

        review_file = (
            __import__("pathlib").Path(__file__).parent / "nonexistent_review_date.txt"
        )
        with patch("research.pipeline._load._LAST_REVIEW_DATE_FILE", review_file):
            today = pd.Timestamp("2024-06-15")
            result = _read_last_review_date(today)
            expected = (today.to_period("Q") - 1).end_time.normalize()
            assert result == expected


class TestLoadData:
    def test_when_called_then_returns_assembly_country_map_and_db(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research.pipeline._load import load_data

        assembly = _make_assembly()
        db_manager = MagicMock()
        _patch_load_data(monkeypatch, assembly)

        result_assembly, country_map, _db = load_data(db_manager)
        assert isinstance(result_assembly, DataAssembly)
        assert isinstance(country_map, dict)

    def test_when_called_then_prices_are_fx_converted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from research.pipeline._load import load_data

        assembly = _make_assembly()
        db_manager = MagicMock()
        _patch_load_data(monkeypatch, assembly)

        result, _, _db = load_data(db_manager)
        assert result.prices is not None


def _patch_load_data(
    monkeypatch: pytest.MonkeyPatch,
    assembly: DataAssembly,
    *,
    preflight_raises: bool = False,
) -> None:
    from research.pipeline import _load

    if preflight_raises:
        _load._run_db_preflight = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("synthetic preflight")
        )
    else:
        monkeypatch.setattr(_load, "_run_db_preflight", MagicMock())
    monkeypatch.setattr(_load, "assemble_all", MagicMock(return_value=assembly))
    monkeypatch.setattr(_load, "_assert_assembly_size", MagicMock())
    monkeypatch.setattr(
        _load, "apply_fx_to_prices", MagicMock(return_value=assembly.prices)
    )
    monkeypatch.setattr(
        _load, "_build_country_map", MagicMock(return_value={"AAA": "USA"})
    )
