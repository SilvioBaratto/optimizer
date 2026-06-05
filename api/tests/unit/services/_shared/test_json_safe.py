"""Unit tests for ``_shared/_json_safe.safe_float`` (issue #826)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.services._shared._json_safe import safe_float


class TestSafeFloat:
    def test_when_none_then_returns_none(self) -> None:
        assert safe_float(None) is None

    @pytest.mark.parametrize("value", [math.nan, float("nan")])
    def test_when_nan_then_returns_none(self, value: float) -> None:
        assert safe_float(value) is None

    @pytest.mark.parametrize("value", [math.inf, -math.inf])
    def test_when_inf_then_returns_none(self, value: float) -> None:
        assert safe_float(value) is None

    @pytest.mark.parametrize("value", ["abc", object(), [1, 2]])
    def test_when_non_castable_then_returns_none(self, value: object) -> None:
        assert safe_float(value) is None

    def test_when_numpy_float_then_unwrapped(self) -> None:
        assert safe_float(np.float64(1.5)) == 1.5

    def test_when_numpy_nan_then_returns_none(self) -> None:
        assert safe_float(np.float64("nan")) is None

    def test_when_multi_element_array_then_returns_none(self) -> None:
        # safe_float does NOT unwrap arrays — float() raises → None.
        assert safe_float(np.array([1.0, 2.0])) is None

    def test_when_ndigits_then_rounds(self) -> None:
        assert safe_float(1.23456, ndigits=2) == 1.23

    def test_when_bare_float_then_passthrough(self) -> None:
        assert safe_float(1.5) == 1.5

    def test_when_int_then_cast_to_float(self) -> None:
        result = safe_float(3)
        assert result == 3.0
        assert isinstance(result, float)

    def test_when_numeric_string_then_parsed(self) -> None:
        assert safe_float("1.5") == 1.5
