"""Tests for the checks shared by the simulators.

Each check is exercised through the public functions elsewhere; these tests pin the
rule the checks share: settings raise ``ConfigurationError``, data raises
``DataError``, and an input numpy cannot read raises the same class as a bad value.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from optimizer.exceptions import ConfigurationError, DataError, OptimizerError
from optimizer.simulation._validation import (
    as_float_array,
    as_returns,
    as_vector,
    require_no_overflow,
    require_non_negative,
    require_open_unit_interval,
)


class TestAsFloatArray:
    """``as_float_array`` raises the class it is told to for unreadable input."""

    @pytest.mark.parametrize("error", [ConfigurationError, DataError])
    @pytest.mark.parametrize("values", [[[1.0], [2.0, 3.0]], ["a", "b"], [object()]])
    def test_unreadable_input_raises_the_requested_error(
        self, error: type[OptimizerError], values: Any
    ) -> None:
        """Ragged and non-numeric input never leaks numpy's own exception."""
        with pytest.raises(error, match="values"):
            as_float_array("values", values, error)

    def test_numbers_pass_through_as_float64(self) -> None:
        """Integers and nested lists become a float array of the same shape."""
        array = as_float_array("values", [[1, 2], [3, 4]], DataError)
        assert array.dtype == np.float64
        assert array.shape == (2, 2)


class TestAsVector:
    """``as_vector`` accepts finite one-dimensional input of enough length."""

    def test_minimum_size_is_enforced(self) -> None:
        """A vector shorter than ``min_size`` is rejected with the requested error."""
        with pytest.raises(DataError, match="at least 3"):
            as_vector("prices", [1.0, 2.0], error=DataError, min_size=3)

    def test_scalar_is_not_a_vector(self) -> None:
        """A zero-dimensional input has no length."""
        with pytest.raises(ConfigurationError, match="one-dimensional"):
            as_vector("mu", 0.05, error=ConfigurationError)

    def test_missing_values_are_named(self) -> None:
        """NaN and infinity are rejected and the message says how to fix it."""
        with pytest.raises(DataError, match="drop or fill"):
            as_vector("prices", [1.0, math.nan], error=DataError)


class TestAsReturns:
    """``as_returns`` is a data check: it always raises ``DataError``."""

    def test_a_loss_beyond_total_loss_is_rejected(self) -> None:
        """A simple return below -1 cannot happen without leverage."""
        with pytest.raises(DataError, match="at least -1"):
            as_returns([0.1, -1.01], 1)

    def test_a_total_loss_is_allowed(self) -> None:
        """Exactly -100% is a legitimate outcome."""
        assert as_returns([0.1, -1.0], 1).tolist() == [0.1, -1.0]


class TestScalarChecks:
    """Checks on single settings."""

    @pytest.mark.parametrize("value", [-0.1, math.nan, math.inf, "0", None, True])
    def test_require_non_negative_rejects(self, value: Any) -> None:
        """A negative, non-finite or non-numeric value is not a non-negative number."""
        with pytest.raises(ConfigurationError, match="sigma"):
            require_non_negative("sigma", value)

    def test_require_non_negative_accepts_zero(self) -> None:
        """Zero is allowed, unlike for a strictly positive setting."""
        require_non_negative("sigma", 0.0)

    @pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5, math.nan, "0.5", True])
    def test_require_open_unit_interval_rejects(self, value: Any) -> None:
        """Only numbers strictly between 0 and 1 pass."""
        with pytest.raises(ConfigurationError, match="level"):
            require_open_unit_interval("level", value)

    def test_require_open_unit_interval_accepts_the_interior(self) -> None:
        """A value inside the interval passes."""
        require_open_unit_interval("level", 0.5)


class TestRequireNoOverflow:
    """``require_no_overflow`` names what the caller can change."""

    def test_infinity_and_nan_raise_with_the_remedy(self) -> None:
        """An overflowed array is a configuration problem, with advice."""
        with pytest.raises(ConfigurationError, match="shrink n_steps"):
            require_no_overflow(np.array([1.0, np.inf]), "shrink n_steps")
        with pytest.raises(ConfigurationError, match="overflow"):
            require_no_overflow(np.array([1.0, np.nan]), "shrink n_steps")

    def test_finite_values_pass(self) -> None:
        """Ordinary values are left alone."""
        require_no_overflow(np.array([0.0, 1.0, 1e300]), "unused")
