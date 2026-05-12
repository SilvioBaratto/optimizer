"""Tests for research/data/_regime.py."""

from __future__ import annotations

import pandas as pd
import pytest

from research.data._regime import (
    _FRED_REGIME_MAP,
    _REQUIRED_REGIME_COLUMNS,
    _TE_REGIME_MAP,
    assemble_regime_data,
)


def _dated(data: dict, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(data, index=pd.to_datetime(dates))


class TestAssembleRegimeData:
    def test_when_all_inputs_empty_empty_dataframe_returned(self) -> None:
        """when all inputs are empty, empty DataFrame returned."""
        result = assemble_regime_data(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_fred_has_gdp_series_gdp_growth_column_present(self) -> None:
        """when FRED has A191RL1Q225SBEA, gdp_growth column present."""
        fred = _dated({"A191RL1Q225SBEA": [2.5]}, ["2024-01-01"])
        result = assemble_regime_data(pd.DataFrame(), fred, pd.DataFrame())
        assert "gdp_growth" in result.columns

    def test_when_fred_has_t10y2y_spread_2s10s_column_present(self) -> None:
        """when FRED has T10Y2Y, spread_2s10s column present."""
        fred = _dated({"T10Y2Y": [0.5]}, ["2024-01-01"])
        result = assemble_regime_data(pd.DataFrame(), fred, pd.DataFrame())
        assert "spread_2s10s" in result.columns

    def test_when_fred_has_hy_oas_series_hy_oas_column_present(self) -> None:
        """when FRED has BAMLH0A0HYM2, hy_oas column present."""
        fred = _dated({"BAMLH0A0HYM2": [3.0]}, ["2024-01-01"])
        result = assemble_regime_data(pd.DataFrame(), fred, pd.DataFrame())
        assert "hy_oas" in result.columns

    def test_when_te_has_manufacturing_pmi_pmi_column_present(self) -> None:
        """when TE has manufacturing_pmi, pmi column present."""
        te = _dated({"manufacturing_pmi": [52.0]}, ["2024-01-01"])
        result = assemble_regime_data(pd.DataFrame(), pd.DataFrame(), te)
        assert "pmi" in result.columns

    def test_when_macro_has_gdp_growth_gdp_growth_column_present(self) -> None:
        """when macro_data has gdp_growth, gdp_growth column present."""
        macro = _dated({"gdp_growth": [2.0]}, ["2024-01-01"])
        result = assemble_regime_data(macro, pd.DataFrame(), pd.DataFrame())
        assert "gdp_growth" in result.columns

    def test_when_sentiment_present_sentiment_column_included(self) -> None:
        """when sentiment_data with matching country, sentiment column present."""
        macro = _dated({"gdp_growth": [2.0]}, ["2024-01-01"])
        sentiment = _dated({"USA": [0.5]}, ["2024-01-01"])
        result = assemble_regime_data(
            macro, pd.DataFrame(), pd.DataFrame(), sentiment_data=sentiment
        )
        assert "sentiment" in result.columns

    def test_when_sentiment_country_absent_sentiment_column_not_added(self) -> None:
        """when sentiment_data lacks requested country, sentiment column absent."""
        macro = _dated({"gdp_growth": [2.0]}, ["2024-01-01"])
        sentiment = _dated({"Germany": [0.3]}, ["2024-01-01"])
        result = assemble_regime_data(
            macro,
            pd.DataFrame(),
            pd.DataFrame(),
            sentiment_data=sentiment,
            sentiment_country="USA",
        )
        assert "sentiment" not in result.columns

    def test_when_all_sources_provided_all_regime_columns_present(self) -> None:
        """when all sources provided, all five regime columns present."""
        dates = ["2024-01-01"]
        fred = _dated(
            {"T10Y2Y": [0.5], "BAMLH0A0HYM2": [3.0], "A191RL1Q225SBEA": [2.5]},
            dates,
        )
        te = _dated({"manufacturing_pmi": [52.0]}, dates)
        sentiment = _dated({"USA": [0.3]}, dates)
        result = assemble_regime_data(
            pd.DataFrame(), fred, te, sentiment_data=sentiment
        )
        for col in ("gdp_growth", "spread_2s10s", "hy_oas", "pmi", "sentiment"):
            assert col in result.columns

    def test_when_required_column_all_nan_value_error_raised(self) -> None:
        """when gdp_growth present but all NaN, ValueError raised."""
        # yield_spread keeps the row alive through dropna(how='all')
        macro = _dated(
            {"gdp_growth": [float("nan")], "yield_spread": [1.0]}, ["2024-01-01"]
        )
        with pytest.raises(ValueError, match="gdp_growth"):
            assemble_regime_data(macro, pd.DataFrame(), pd.DataFrame())

    def test_when_data_stale_beyond_fill_limit_user_warning_emitted(self) -> None:
        """when trailing NaN exceeds fill_limit, UserWarning emitted."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        fred = pd.DataFrame({"T10Y2Y": [0.5] + [float("nan")] * 99}, index=dates)
        macro = pd.DataFrame({"gdp_growth": [2.0] * 100}, index=dates)
        with pytest.warns(UserWarning, match="spread_2s10s"):
            assemble_regime_data(macro, fred, pd.DataFrame(), fill_limit=10)

    def test_when_fill_limit_not_exceeded_no_trailing_nan(self) -> None:
        """when gap fits within fill_limit, last value is not NaN."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        fred = pd.DataFrame({"T10Y2Y": [0.5] + [float("nan")] * 4}, index=dates)
        macro = pd.DataFrame({"gdp_growth": [2.0] * 5}, index=dates)
        result = assemble_regime_data(macro, fred, pd.DataFrame(), fill_limit=10)
        assert not pd.isna(result["spread_2s10s"].iloc[-1])

    def test_when_result_has_datetime_index(self) -> None:
        """when data provided, result index is DatetimeIndex."""
        macro = _dated({"gdp_growth": [2.0]}, ["2024-01-01"])
        result = assemble_regime_data(macro, pd.DataFrame(), pd.DataFrame())
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_when_non_datetime_index_user_warning_emitted(self) -> None:
        """when a part has non-DatetimeIndex, UserWarning about coercion emitted."""
        te = pd.DataFrame({"manufacturing_pmi": [52.0]}, index=[20240101])
        macro = _dated({"gdp_growth": [2.0]}, ["2024-01-01"])
        with pytest.warns(UserWarning, match="coercion to DatetimeIndex"):
            assemble_regime_data(macro, pd.DataFrame(), te)


class TestRegimeConstants:
    def test_fred_regime_map_contains_required_keys(self) -> None:
        """FRED map covers gdp, spread, and HY OAS series."""
        assert "T10Y2Y" in _FRED_REGIME_MAP
        assert "BAMLH0A0HYM2" in _FRED_REGIME_MAP
        assert "A191RL1Q225SBEA" in _FRED_REGIME_MAP

    def test_te_regime_map_contains_pmi(self) -> None:
        """TE map maps manufacturing_pmi to pmi."""
        assert _TE_REGIME_MAP.get("manufacturing_pmi") == "pmi"

    def test_gdp_growth_is_required_regime_column(self) -> None:
        """gdp_growth is in the required columns set."""
        assert "gdp_growth" in _REQUIRED_REGIME_COLUMNS
