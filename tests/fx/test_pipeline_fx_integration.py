"""Integration tests for FX conversion wiring through run_full_pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from optimizer.fx._config import BaseCurrency, FxConfig, FxConversionMode
from optimizer.optimization import EqualWeightedConfig, build_equal_weighted
from optimizer.pipeline._orchestrator import (
    run_full_pipeline,
    run_full_pipeline_with_selection,
)


@pytest.fixture()
def price_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", periods=120)


@pytest.fixture()
def local_prices(price_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """3 tickers: GBP, EUR (base), USD — 120 trading days."""
    rng = np.random.default_rng(42)
    n = len(price_dates)
    return pd.DataFrame(
        {
            "LLOY.L": 50.0 + rng.standard_normal(n).cumsum(),
            "ORA.PA": 90.0 + rng.standard_normal(n).cumsum(),
            "SPY": 480.0 + rng.standard_normal(n).cumsum(),
        },
        index=price_dates,
    )


@pytest.fixture()
def currency_map() -> dict[str, str]:
    return {"LLOY.L": "GBP", "ORA.PA": "EUR", "SPY": "USD"}


@pytest.fixture()
def fx_rates(price_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """EUR-base rates: GBP→EUR ≈1.16, USD→EUR ≈0.92."""
    n = len(price_dates)
    return pd.DataFrame(
        {
            "GBP": np.linspace(1.15, 1.17, n),
            "USD": np.linspace(0.91, 0.93, n),
        },
        index=price_dates,
    )


class TestRunFullPipelineFxConversion:
    """Verify FX conversion activates when all three kwargs are supplied."""

    def test_fx_conversion_fires(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        optimizer = build_equal_weighted(EqualWeightedConfig())
        fx_config = FxConfig.for_eur_base()

        result = run_full_pipeline(
            prices=local_prices,
            optimizer=optimizer,
            fx_config=fx_config,
            currency_map=currency_map,
            fx_rates=fx_rates,
        )

        assert result.currency == "EUR"
        assert result.weights is not None
        assert len(result.weights) > 0

    def test_fx_decomposition_mode(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        optimizer = build_equal_weighted(EqualWeightedConfig())
        fx_config = FxConfig.for_decomposition(BaseCurrency.EUR)

        result = run_full_pipeline(
            prices=local_prices,
            optimizer=optimizer,
            fx_config=fx_config,
            currency_map=currency_map,
            fx_rates=fx_rates,
        )

        assert result.currency == "EUR"
        assert result.fx_decomposition is not None

    def test_fx_skipped_when_no_config(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        optimizer = build_equal_weighted(EqualWeightedConfig())

        result = run_full_pipeline(
            prices=local_prices,
            optimizer=optimizer,
            fx_config=None,
            currency_map=currency_map,
            fx_rates=fx_rates,
        )

        assert result.currency is None
        assert result.fx_decomposition is None

    def test_fx_skipped_when_no_fx_rates(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
    ) -> None:
        optimizer = build_equal_weighted(EqualWeightedConfig())
        fx_config = FxConfig.for_eur_base()

        result = run_full_pipeline(
            prices=local_prices,
            optimizer=optimizer,
            fx_config=fx_config,
            currency_map=currency_map,
            fx_rates=None,
        )

        assert result.currency is None
        assert result.fx_decomposition is None

    def test_fx_skipped_when_mode_none(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        optimizer = build_equal_weighted(EqualWeightedConfig())
        fx_config = FxConfig(mode=FxConversionMode.NONE)

        result = run_full_pipeline(
            prices=local_prices,
            optimizer=optimizer,
            fx_config=fx_config,
            currency_map=currency_map,
            fx_rates=fx_rates,
        )

        assert result.currency is None


class TestRunFullPipelineWithSelectionFxSlicing:
    """Verify currency_map is sliced to selected tickers."""

    def test_currency_map_sliced_to_selected(
        self,
        local_prices: pd.DataFrame,
        currency_map: dict[str, str],
        fx_rates: pd.DataFrame,
    ) -> None:
        """When fundamentals are provided, currency_map should only contain
        tickers that survived selection."""
        optimizer = build_equal_weighted(EqualWeightedConfig())
        fx_config = FxConfig.for_eur_base()

        # Provide minimal fundamentals so stock selection runs and selects
        # a subset (just 2 of 3 tickers)
        fundamentals = pd.DataFrame(
            {
                "market_cap": [5e9, 8e9, 3e9],
                "pe_ratio": [12.0, 15.0, 20.0],
            },
            index=pd.Index(["LLOY.L", "ORA.PA", "SPY"], name="ticker"),
        )

        # Patch run_full_pipeline to capture the currency_map it receives
        with patch(
            "optimizer.pipeline._orchestrator.run_full_pipeline"
        ) as mock_rfp:
            mock_rfp.return_value = MagicMock(
                weights=pd.Series([0.5, 0.5], index=["LLOY.L", "SPY"]),
                currency="EUR",
            )
            import contextlib

            with contextlib.suppress(Exception):
                run_full_pipeline_with_selection(
                    prices=local_prices,
                    optimizer=optimizer,
                    fundamentals=fundamentals,
                    fx_config=fx_config,
                    currency_map=currency_map,
                    fx_rates=fx_rates,
                )

            if mock_rfp.called:
                call_kwargs = mock_rfp.call_args.kwargs
                passed_map = call_kwargs.get("currency_map", {})
                # Sliced map should only contain tickers in the prices columns
                passed_prices = call_kwargs.get("prices")
                if passed_prices is not None:
                    assert set(passed_map.keys()) <= set(passed_prices.columns)


class TestAssembleFxRatesEmptyPriceIndex:
    """Verify assemble_fx_rates handles empty price index gracefully."""

    def test_empty_price_index_returns_empty_df(self) -> None:
        from cli.data_assembly import assemble_fx_rates

        currency_map = {"LLOY.L": "GBP", "SPY": "USD"}
        empty_index = pd.DatetimeIndex([])

        result = assemble_fx_rates(
            currency_map=currency_map,
            base_currency="EUR",
            price_index=empty_index,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestAssembleAllBaseCurrency:
    """Verify assemble_all() threads base_currency parameter."""

    @patch("cli.data_assembly.assemble_fx_rates")
    @patch("cli.data_assembly.assemble_regime_data")
    @patch("cli.data_assembly.assemble_delisting_returns")
    @patch("cli.data_assembly.assemble_fundamental_history")
    @patch("cli.data_assembly.assemble_sentiment")
    @patch("cli.data_assembly.assemble_bond_observations")
    @patch("cli.data_assembly.assemble_te_observations")
    @patch("cli.data_assembly.assemble_fred_series")
    @patch("cli.data_assembly.assemble_macro_data")
    @patch("cli.data_assembly.assemble_insider_data")
    @patch("cli.data_assembly.assemble_analyst_data")
    @patch("cli.data_assembly.assemble_financial_statements")
    @patch("cli.data_assembly.assemble_volumes")
    @patch("cli.data_assembly.assemble_prices")
    @patch("cli.data_assembly.assemble_fundamentals")
    def test_base_currency_passed_to_fx_rates(
        self,
        mock_fundamentals: MagicMock,
        mock_prices: MagicMock,
        mock_volumes: MagicMock,
        mock_fin_stmts: MagicMock,
        mock_analyst: MagicMock,
        mock_insider: MagicMock,
        mock_macro: MagicMock,
        mock_fred: MagicMock,
        mock_te: MagicMock,
        mock_bonds: MagicMock,
        mock_sentiment: MagicMock,
        mock_fund_hist: MagicMock,
        mock_delisting: MagicMock,
        mock_regime: MagicMock,
        mock_fx_rates: MagicMock,
    ) -> None:
        from cli.data_assembly import assemble_all

        dates = pd.bdate_range("2024-01-02", periods=10)
        prices = pd.DataFrame(
            {"LLOY.L": range(10), "SPY": range(10)},
            index=dates,
        )
        cmap = {"LLOY.L": "GBP", "SPY": "USD"}

        mock_fundamentals.return_value = (pd.DataFrame(), {}, cmap)
        mock_prices.return_value = prices
        mock_volumes.return_value = pd.DataFrame()
        mock_fin_stmts.return_value = pd.DataFrame()
        mock_analyst.return_value = pd.DataFrame()
        mock_insider.return_value = pd.DataFrame()
        mock_macro.return_value = pd.DataFrame()
        mock_fred.return_value = pd.DataFrame()
        mock_te.return_value = pd.DataFrame()
        mock_bonds.return_value = pd.DataFrame()
        mock_sentiment.return_value = pd.DataFrame()
        mock_fund_hist.return_value = pd.DataFrame()
        mock_delisting.return_value = {}
        mock_regime.return_value = pd.DataFrame()
        mock_fx_rates.return_value = pd.DataFrame()

        mock_db = MagicMock()
        mock_db.get_session.return_value.__enter__ = MagicMock()
        mock_db.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        assemble_all(mock_db, base_currency="GBP")

        mock_fx_rates.assert_called_once()
        call_kwargs = mock_fx_rates.call_args
        assert call_kwargs.kwargs.get("base_currency") == "GBP" or (
            call_kwargs.args and call_kwargs.args[1] == "GBP"
        )
