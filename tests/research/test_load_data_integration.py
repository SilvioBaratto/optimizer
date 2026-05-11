"""Integration smoke tests for load_data() Cycle 1 wiring (issue #522)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("typer")

import research.pipeline._load as _load_module
from research import stock_selection_pipeline as ssp
from research.data_assembly import DataAssembly

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_assembly(
    n_tickers: int = 2_500,
    n_days: int = 1_300,
    *,
    foreign_ticker: str | None = "AAA",
) -> DataAssembly:
    """Build a synthetic DataAssembly large enough to pass §0/§1/§2 gates."""
    rng = np.random.default_rng(0)
    tickers = [f"T{i:05d}" for i in range(n_tickers)]
    if foreign_ticker is not None and foreign_ticker not in tickers:
        tickers[0] = foreign_ticker
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    increments = rng.normal(0, 0.005, size=(n_days, n_tickers))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(increments, axis=0)),
        index=dates,
        columns=tickers,
    )
    volumes = pd.DataFrame(1.0, index=dates, columns=tickers)
    fundamentals = pd.DataFrame(index=tickers)
    sector_mapping = dict.fromkeys(tickers, "Tech")

    currency_map = dict.fromkeys(tickers, "EUR")
    if foreign_ticker is not None and foreign_ticker in tickers:
        currency_map[foreign_ticker] = "USD"

    fx_rates = pd.DataFrame({"USD": np.full(n_days, 0.9)}, index=dates)

    return DataAssembly(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        sector_mapping=sector_mapping,
        financial_statements=pd.DataFrame(),
        analyst_data=pd.DataFrame(),
        insider_data=pd.DataFrame(),
        macro_data=pd.DataFrame(),
        currency_map=currency_map,
        fx_rates=fx_rates,
    )


class _StubMgr:
    def initialize(self) -> None:
        return


def _patch_load_data(
    monkeypatch: pytest.MonkeyPatch,
    assembly: DataAssembly,
    *,
    preflight_raises: bool = False,
) -> None:
    def _fake_preflight(_db: object) -> None:
        if preflight_raises:
            raise RuntimeError("synthetic preflight failure")

    monkeypatch.setattr(_load_module, "_run_db_preflight", _fake_preflight)
    monkeypatch.setattr(_load_module, "assemble_all", lambda *_a, **_kw: assembly)
    monkeypatch.setattr(
        _load_module, "_build_country_map", lambda _db: {"AAA": "United States"}
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadDataIntegration:
    def test_when_called_then_returns_assembly_and_country_map(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assembly = _make_assembly()
        _patch_load_data(monkeypatch, assembly)
        result_assembly, country_map, _db = ssp.load_data(_StubMgr())
        assert isinstance(result_assembly, DataAssembly)
        assert isinstance(country_map, dict)
        assert "AAA" in country_map

    def test_when_called_then_prices_are_fx_converted_to_eur(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assembly = _make_assembly(foreign_ticker="AAA")
        original_aaa = assembly.prices["AAA"].copy()
        _patch_load_data(monkeypatch, assembly)
        result, _, _db = ssp.load_data(_StubMgr())
        np.testing.assert_allclose(
            result.prices["AAA"].to_numpy(),
            (original_aaa * 0.9).to_numpy(),
        )

    def test_when_called_then_assembly_hash_is_non_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Hash is set by assemble_all in production; the stub bypasses it,
        # so seed it explicitly to mirror the production contract.
        assembly = _make_assembly()
        assembly.assembly_hash = "abcdef1234567890"
        _patch_load_data(monkeypatch, assembly)
        result, _, _db = ssp.load_data(_StubMgr())
        assert result.assembly_hash != ""

    def test_when_preflight_raises_then_runtime_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assembly = _make_assembly()
        _patch_load_data(monkeypatch, assembly, preflight_raises=True)
        with pytest.raises(RuntimeError, match="synthetic preflight"):
            ssp.load_data(_StubMgr())

    def test_when_n_tickers_below_2000_then_runtime_error_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assembly = _make_assembly(n_tickers=1_500)
        _patch_load_data(monkeypatch, assembly)
        with pytest.raises(RuntimeError, match="n_tickers"):
            ssp.load_data(_StubMgr())

    def test_when_n_trading_days_below_1260_then_runtime_error_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assembly = _make_assembly(n_days=1_000)
        _patch_load_data(monkeypatch, assembly)
        with pytest.raises(RuntimeError, match="n_trading_days"):
            ssp.load_data(_StubMgr())


class TestAssertAssemblySize:
    def test_when_below_floors_then_raises(self) -> None:
        small = _make_assembly(n_tickers=100, n_days=100)
        with pytest.raises(RuntimeError):
            ssp._assert_assembly_size(small)

    def test_when_meets_floors_then_no_raise(self) -> None:
        ok = _make_assembly()
        ssp._assert_assembly_size(ok)


class TestMaterialiseCleanReturns:
    """Verify Cycle 1 §3 hand-off: prices_to_returns + preprocessing pipeline."""

    def test_when_called_then_returns_dataframe_no_nan_no_inf(self) -> None:
        assembly = _make_assembly(n_tickers=10, n_days=200, foreign_ticker=None)
        investable = pd.Index(assembly.prices.columns[:5])
        clean = ssp._materialise_clean_returns(assembly, investable)
        assert isinstance(clean, pd.DataFrame)
        arr = clean.to_numpy()
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()

    def test_when_called_then_subset_to_investable_columns(self) -> None:
        assembly = _make_assembly(n_tickers=10, n_days=200, foreign_ticker=None)
        investable = pd.Index(assembly.prices.columns[:3])
        clean = ssp._materialise_clean_returns(assembly, investable)
        assert list(clean.columns) == list(investable)
