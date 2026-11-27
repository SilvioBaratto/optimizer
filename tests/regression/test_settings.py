"""Tests for the validation of the numeric settings of every public function.

A count must be a whole number and a rate a real number. The checks below close the
gap where a comparison such as ``value < 1`` lets NaN, a float that happens to be
whole, or a bool through, for each entry point that takes the setting.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.regression import (
    CapmResult,
    FactorRegressionResult,
    OlsResult,
    SmlResult,
    capm_regression,
    fama_french_regression,
    multifactor_regression,
    ols,
    rolling_beta,
    security_market_line,
)

_N_OBS = 80
_INDEX = pd.date_range("2022-01-03", periods=_N_OBS, freq="B")
_RNG = np.random.default_rng(41)
_MARKET = pd.Series(_RNG.normal(0.0004, 0.01, _N_OBS), index=_INDEX)
_ASSET = pd.Series(
    0.0001 + 1.1 * _MARKET.to_numpy() + _RNG.normal(0.0, 0.003, _N_OBS), index=_INDEX
)
_ASSETS = pd.DataFrame(
    {
        "a": _ASSET,
        "b": 0.5 * _MARKET.to_numpy() + _RNG.normal(0.0, 0.003, _N_OBS),
        "c": 1.5 * _MARKET.to_numpy() + _RNG.normal(0.0, 0.003, _N_OBS),
    },
    index=_INDEX,
)
_FACTORS = pd.DataFrame(
    {
        "Mkt-RF": _MARKET,
        "SMB": _RNG.normal(0.0, 0.005, _N_OBS),
        "HML": _RNG.normal(0.0, 0.005, _N_OBS),
    },
    index=_INDEX,
)

_Result: TypeAlias = CapmResult | OlsResult | SmlResult | FactorRegressionResult

_NOT_A_COUNT = [True, False, 12.0, 12.5, float("nan"), float("inf"), "12", None]
_NOT_A_RATE = [True, False, float("nan"), float("inf"), -float("inf"), "0.01", None]


def _ols_call(**settings: Any) -> _Result:
    """Call ``ols`` on the shared data with the given settings."""
    return ols(_ASSET, _MARKET, **settings)


def _capm_call(**settings: Any) -> _Result:
    """Call ``capm_regression`` on the shared data with the given settings."""
    return capm_regression(_ASSET, _MARKET, **settings)


def _sml_call(**settings: Any) -> _Result:
    """Call ``security_market_line`` on the shared data with the given settings."""
    return security_market_line(_ASSETS, _MARKET, **settings)


def _multifactor_call(**settings: Any) -> _Result:
    """Call ``multifactor_regression`` on the shared data with the given settings."""
    return multifactor_regression(_ASSET, _FACTORS, **settings)


def _fama_french_call(**settings: Any) -> _Result:
    """Call ``fama_french_regression`` on the shared data with the given settings."""
    return fama_french_regression(_ASSET, _FACTORS, **settings)


def _numbers(result: _Result) -> list[float]:
    """Flatten the figures of any result so two results can be compared."""
    if isinstance(result, OlsResult):
        return [*result.coefficients, *result.std_errors, *result.p_values]
    if isinstance(result, SmlResult):
        return [
            result.slope,
            result.intercept,
            result.market_premium,
            *result.table.to_numpy().ravel(),
        ]
    if isinstance(result, FactorRegressionResult):
        return [
            result.alpha,
            result.annualized_alpha,
            result.alpha_t_statistic,
            result.residual_volatility,
            *result.loadings,
            *result.loading_t_statistics,
        ]
    return list(dataclasses.asdict(result).values())


_PERIODS_CALLS = [_capm_call, _sml_call, _multifactor_call, _fama_french_call]
_HAC_CALLS = [_ols_call, _capm_call, _multifactor_call, _fama_french_call]
_RATE_CALLS = [_capm_call, _sml_call, _multifactor_call, _fama_french_call]


class TestPeriodsPerYear:
    """``periods_per_year`` is a positive whole number."""

    @pytest.mark.parametrize("call", _PERIODS_CALLS, ids=lambda call: call.__name__)
    @pytest.mark.parametrize("value", [*_NOT_A_COUNT, 0, -1], ids=repr)
    def test_anything_else_is_rejected(
        self, call: Callable[..., _Result], value: object
    ) -> None:
        """NaN, floats, bools, strings and non-positive numbers raise."""
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            call(periods_per_year=value)

    @pytest.mark.parametrize("call", _PERIODS_CALLS, ids=lambda call: call.__name__)
    @pytest.mark.parametrize("whole", [np.int64(12), np.int32(12)], ids=repr)
    def test_numpy_integers_are_accepted(
        self, call: Callable[..., _Result], whole: object
    ) -> None:
        """A numpy integer gives the same figures as the equal Python int."""
        assert _numbers(call(periods_per_year=whole)) == pytest.approx(
            _numbers(call(periods_per_year=12))
        )


class TestHacLags:
    """``hac_lags`` is ``None`` or a non-negative whole number."""

    @pytest.mark.parametrize("call", _HAC_CALLS, ids=lambda call: call.__name__)
    @pytest.mark.parametrize(
        "value", [True, False, 2.0, 1.5, float("nan"), float("inf"), "2", -1], ids=repr
    )
    def test_anything_else_is_rejected(
        self, call: Callable[..., _Result], value: object
    ) -> None:
        """NaN, floats, bools, strings and negative numbers raise."""
        with pytest.raises(ConfigurationError, match="hac_lags"):
            call(hac_lags=value)

    @pytest.mark.parametrize("call", _HAC_CALLS, ids=lambda call: call.__name__)
    @pytest.mark.parametrize("whole", [np.int64(2), np.int32(2)], ids=repr)
    def test_numpy_integers_are_accepted(
        self, call: Callable[..., _Result], whole: object
    ) -> None:
        """A numpy integer gives the same figures as the equal Python int."""
        assert _numbers(call(hac_lags=whole)) == pytest.approx(
            _numbers(call(hac_lags=2))
        )

    @pytest.mark.parametrize("call", _HAC_CALLS, ids=lambda call: call.__name__)
    def test_zero_lags_are_accepted_as_a_numpy_integer(
        self, call: Callable[..., _Result]
    ) -> None:
        """Zero is a valid lag count in either integer type."""
        assert _numbers(call(hac_lags=np.int64(0))) == pytest.approx(
            _numbers(call(hac_lags=0))
        )


class TestWindow:
    """``window`` is a whole number of at least two."""

    @pytest.mark.parametrize("value", [*_NOT_A_COUNT, 3.0, 0, 1, -1], ids=repr)
    def test_anything_else_is_rejected(self, value: object) -> None:
        """NaN, floats, bools, strings and windows below two raise."""
        settings: dict[str, Any] = {"window": value}
        with pytest.raises(ConfigurationError, match="window"):
            rolling_beta(_ASSET, _MARKET, **settings)

    @pytest.mark.parametrize("whole", [np.int64(5), np.int32(5)], ids=repr)
    def test_numpy_integers_are_accepted(self, whole: object) -> None:
        """A numpy integer gives the same betas as the equal Python int."""
        settings: dict[str, Any] = {"window": whole}
        expected = rolling_beta(_ASSET, _MARKET, 5)
        pd.testing.assert_series_equal(
            rolling_beta(_ASSET, _MARKET, **settings), expected
        )


class TestRiskFreeRate:
    """A scalar ``risk_free_rate`` is a finite real number, not a bool."""

    @pytest.mark.parametrize("call", _RATE_CALLS, ids=lambda call: call.__name__)
    @pytest.mark.parametrize("value", _NOT_A_RATE, ids=repr)
    def test_anything_else_is_rejected(
        self, call: Callable[..., _Result], value: object
    ) -> None:
        """NaN, infinities, bools, strings and None raise."""
        with pytest.raises(ConfigurationError, match="risk_free_rate"):
            call(risk_free_rate=value)

    @pytest.mark.parametrize("call", _RATE_CALLS, ids=lambda call: call.__name__)
    @pytest.mark.parametrize(
        "rate", [0, np.int64(0), np.float64(0.0005), np.float32(0.0)], ids=repr
    )
    def test_python_and_numpy_numbers_are_accepted(
        self, call: Callable[..., _Result], rate: Any
    ) -> None:
        """Ints and numpy numbers give the same figures as the equal float."""
        assert _numbers(call(risk_free_rate=rate)) == pytest.approx(
            _numbers(call(risk_free_rate=float(rate)))
        )
