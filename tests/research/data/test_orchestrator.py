"""Integration tests for research/data/_orchestrator.py — assemble_all."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from research.data._container import DataAssembly
from research.data._orchestrator import assemble_all

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_MOD = "research.data._orchestrator"
_DATES = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])


class _MockDbManager:
    """Minimal stub satisfying DatabaseManager.get_session() interface."""

    @contextmanager
    def get_session(self) -> Generator[MagicMock, None, None]:
        yield MagicMock()


def _run(db_mgr: Any, **kwargs: Any) -> DataAssembly:
    """Thin wrapper so tests pass _MockDbManager without Pyright complaints."""
    return assemble_all(db_mgr, **kwargs)  # pyright: ignore[reportArgumentType]


def _prices_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0], "MSFT": [200.0, 201.0, 202.0]},
        index=_DATES,
    )


def _patch_sub_assemblers(**overrides: Any) -> Any:
    """Patch all sub-assemblers in _orchestrator to return empty/minimal data."""
    empty = pd.DataFrame()
    defaults: dict[str, Any] = {
        "assemble_fundamentals": MagicMock(return_value=(empty, {}, {})),
        "assemble_prices": MagicMock(return_value=empty),
        "assemble_volumes": MagicMock(return_value=empty),
        "assemble_financial_statements": MagicMock(return_value=empty),
        "assemble_analyst_data": MagicMock(return_value=empty),
        "assemble_insider_data": MagicMock(return_value=empty),
        "assemble_macro_data": MagicMock(return_value=empty),
        "assemble_fred_series": MagicMock(return_value=empty),
        "assemble_te_observations": MagicMock(return_value=empty),
        "assemble_bond_observations": MagicMock(return_value=empty),
        "assemble_sentiment": MagicMock(return_value=empty),
        "assemble_fundamental_history": MagicMock(return_value=empty),
        "assemble_delisting_returns": MagicMock(return_value={}),
        "assemble_regime_data": MagicMock(return_value=empty),
        "assemble_fx_rates": MagicMock(return_value=empty),
    }
    defaults.update(overrides)
    return patch.multiple(_MOD, **defaults)


# ---------------------------------------------------------------------------
# TestAssembleAll
# ---------------------------------------------------------------------------


class TestAssembleAll:
    def test_when_all_loaders_empty_returns_data_assembly(self) -> None:
        """when all loaders return empty data, assemble_all returns DataAssembly."""
        with _patch_sub_assemblers():
            result = _run(_MockDbManager())
        assert isinstance(result, DataAssembly)

    def test_when_called_assembly_hash_is_non_empty(self) -> None:
        """when assemble_all completes, assembly_hash is set."""
        with _patch_sub_assemblers():
            result = _run(_MockDbManager())
        assert result.assembly_hash != ""

    def test_when_prices_empty_fx_rates_not_fetched(self) -> None:
        """when prices are empty, assemble_fx_rates is not called."""
        mock_fx = MagicMock(return_value=pd.DataFrame())
        with _patch_sub_assemblers(assemble_fx_rates=mock_fx):
            _run(_MockDbManager())
        mock_fx.assert_not_called()

    def test_when_prices_present_fx_rates_fetched(self) -> None:
        """when prices are non-empty, assemble_fx_rates is called."""
        mock_fx = MagicMock(return_value=pd.DataFrame())
        prices = _prices_df()
        with _patch_sub_assemblers(
            assemble_prices=MagicMock(return_value=prices),
            assemble_fx_rates=mock_fx,
        ):
            _run(_MockDbManager())
        mock_fx.assert_called_once()

    def test_when_prices_present_n_tickers_correct(self) -> None:
        """when prices have 2 tickers, result.n_tickers == 2."""
        prices = _prices_df()
        with _patch_sub_assemblers(
            assemble_prices=MagicMock(return_value=prices),
        ):
            result = _run(_MockDbManager())
        assert result.n_tickers == 2

    def test_when_include_delisted_false_passed_through(self) -> None:
        """when include_delisted=False, volumes assembler receives False."""
        mock_volumes = MagicMock(return_value=pd.DataFrame())
        with _patch_sub_assemblers(assemble_volumes=mock_volumes):
            _run(_MockDbManager(), include_delisted=False)
        _, kwargs = mock_volumes.call_args
        assert kwargs.get("include_delisted") is False

    def test_when_macro_country_set_passed_to_macro_assembler(self) -> None:
        """when macro_country='GBR', assemble_macro_data receives 'GBR'."""
        mock_macro = MagicMock(return_value=pd.DataFrame())
        with _patch_sub_assemblers(assemble_macro_data=mock_macro):
            _run(_MockDbManager(), macro_country="GBR")
        _, kwargs = mock_macro.call_args
        assert kwargs.get("country") == "GBR"

    def test_when_all_loaders_return_data_assembly_fields_typed(self) -> None:
        """when loaders provide data, DataAssembly fields have expected types."""
        prices = _prices_df()
        fundamentals = pd.DataFrame({"market_cap": [1e9]}, index=["AAPL"])
        sector_map = {"AAPL": "Tech", "MSFT": "Tech"}
        ccy_map = {"AAPL": "USD", "MSFT": "USD"}
        with _patch_sub_assemblers(
            assemble_prices=MagicMock(return_value=prices),
            assemble_fundamentals=MagicMock(
                return_value=(fundamentals, sector_map, ccy_map)
            ),
        ):
            result = _run(_MockDbManager())
        assert isinstance(result.prices, pd.DataFrame)
        assert isinstance(result.sector_mapping, dict)
        assert isinstance(result.currency_map, dict)
        assert isinstance(result.delisting_returns, dict)


# ---------------------------------------------------------------------------
# TestImportClean
# ---------------------------------------------------------------------------


class TestImportClean:
    def test_when_imported_no_circular_imports(self) -> None:
        """assemble_all importable with no circular import errors."""
        import importlib

        mod = importlib.import_module("research.data._orchestrator")
        assert hasattr(mod, "assemble_all")

    def test_when_imported_no_cross_layer_imports(self) -> None:
        """_orchestrator does not import from pipeline/optimization/factors."""
        import importlib
        import inspect

        mod = importlib.import_module("research.data._orchestrator")
        src = inspect.getsource(mod)
        for forbidden in (
            "from research.pipeline",
            "from optimizer.factors",
            "from optimizer.optimization",
            "from research.reporting",
        ):
            assert forbidden not in src, f"Cross-layer import detected: {forbidden}"
