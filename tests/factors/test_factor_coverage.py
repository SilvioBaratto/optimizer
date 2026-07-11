"""Tests for factor coverage error handling (issue #249)."""

from __future__ import annotations

import pytest

from optimizer.exceptions import FactorCoverageError, OptimizerError
from optimizer.factors import FactorBuildHealth

# ---------------------------------------------------------------------------
# FactorBuildHealth
# ---------------------------------------------------------------------------


class TestFactorBuildHealth:
    """Unit tests for the FactorBuildHealth diagnostic dataclass."""

    def test_healthy_when_all_succeed(self) -> None:
        h = FactorBuildHealth(
            total_dates=10,
            succeeded_dates=10,
            failed_dates=0,
            failures={},
            min_success_fraction=0.5,
        )
        assert h.is_healthy
        assert h.success_fraction == 1.0

    def test_unhealthy_below_threshold(self) -> None:
        h = FactorBuildHealth(
            total_dates=10,
            succeeded_dates=3,
            failed_dates=7,
            failures={f"2024-01-{i:02d}": "err" for i in range(1, 8)},
            min_success_fraction=0.5,
        )
        assert not h.is_healthy
        assert h.success_fraction == pytest.approx(0.3)

    def test_healthy_exactly_at_threshold(self) -> None:
        h = FactorBuildHealth(
            total_dates=10,
            succeeded_dates=5,
            failed_dates=5,
            failures={f"2024-01-{i:02d}": "err" for i in range(1, 6)},
            min_success_fraction=0.5,
        )
        assert h.is_healthy
        assert h.success_fraction == pytest.approx(0.5)

    def test_zero_total_dates_is_healthy(self) -> None:
        h = FactorBuildHealth(
            total_dates=0,
            succeeded_dates=0,
            failed_dates=0,
            failures={},
            min_success_fraction=0.5,
        )
        assert h.success_fraction == 1.0
        assert h.is_healthy

    def test_failures_dict_populated(self) -> None:
        failures = {"2024-01-01": "ValueError: bad data", "2024-02-01": "KeyError"}
        h = FactorBuildHealth(
            total_dates=5,
            succeeded_dates=3,
            failed_dates=2,
            failures=failures,
            min_success_fraction=0.5,
        )
        assert len(h.failures) == 2
        assert "2024-01-01" in h.failures


# ---------------------------------------------------------------------------
# FactorCoverageError
# ---------------------------------------------------------------------------


class TestFactorCoverageError:
    """Verify FactorCoverageError is part of the exception hierarchy."""

    def test_is_optimizer_error(self) -> None:
        assert issubclass(FactorCoverageError, OptimizerError)

    def test_message_preserved(self) -> None:
        err = FactorCoverageError("only 2/10 dates succeeded")
        assert "only 2/10 dates succeeded" in str(err)


# ---------------------------------------------------------------------------
# build_factor_scores_history exception handling
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# validate_factors VIF exception handling
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _validator.py boundary tests
# ---------------------------------------------------------------------------
