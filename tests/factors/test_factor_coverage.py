"""Tests for factor coverage error handling (issue #249)."""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
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


def _make_prices_volumes(
    n_dates: int = 300, n_tickers: int = 5, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Build synthetic prices, volumes, fundamentals, sector_mapping."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]

    prices = pd.DataFrame(
        rng.uniform(10, 100, (n_dates, n_tickers)),
        index=dates,
        columns=tickers,
    ).cumsum()

    volumes = pd.DataFrame(
        rng.uniform(1e5, 1e7, (n_dates, n_tickers)),
        index=dates,
        columns=tickers,
    )

    fundamentals = pd.DataFrame(
        {
            "market_cap": rng.uniform(1e9, 1e11, n_tickers),
            "enterprise_value": rng.uniform(1e9, 1e11, n_tickers),
            "total_equity": rng.uniform(1e8, 1e10, n_tickers),
            "net_income": rng.uniform(1e7, 1e9, n_tickers),
            "total_assets": rng.uniform(1e9, 5e10, n_tickers),
            "gross_profit": rng.uniform(1e8, 5e9, n_tickers),
            "operating_income": rng.uniform(1e7, 1e9, n_tickers),
            "total_revenue": rng.uniform(1e9, 1e11, n_tickers),
            "operating_cash_flow": rng.uniform(1e7, 1e9, n_tickers),
            "ebitda": rng.uniform(1e8, 5e9, n_tickers),
            "dividend_per_share": rng.uniform(0, 5, n_tickers),
            "current_price": rng.uniform(50, 200, n_tickers),
        },
        index=tickers,
    )

    sector_mapping = dict.fromkeys(tickers, "Technology")
    return prices, volumes, fundamentals, sector_mapping


class _MockAssembly:
    analyst_data = pd.DataFrame()
    insider_data = pd.DataFrame()


class TestBuildFactorScoresHistoryExceptionHandling:
    """Verify logging, health tracking, and coverage gate."""

    def test_logger_warning_emitted_on_date_failure(self) -> None:
        """When compute_all_factors raises, logger.warning is called."""
        from optimizer.factors import (
            FactorConstructionConfig,
            StandardizationConfig,
        )
        from research._factors import build_factor_scores_history

        prices, volumes, fundamentals, sector_mapping = _make_prices_volumes()

        call_count = 0

        def _failing_first_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("synthetic failure")
            # Import the real function and call it
            from optimizer.factors._construction import (
                compute_all_factors as real_fn,
            )
            return real_fn(*args, **kwargs)

        with patch(
            "research._factors.compute_all_factors",
            side_effect=_failing_first_call,
        ), patch("research._factors.logger") as mock_logger:
            result = build_factor_scores_history(
                investable_prices=prices,
                investable_volumes=volumes,
                investable_fundamentals=fundamentals,
                assembly=_MockAssembly(),  # type: ignore[arg-type]
                factor_config=FactorConstructionConfig(),
                std_config=StandardizationConfig(),
                sector_mapping=sector_mapping,
                rebalance_freq=63,
                fundamental_history=None,
            )
            assert mock_logger.warning.call_count >= 1
            warning_msg = mock_logger.warning.call_args_list[0][0][0]
            assert "skipping" in warning_msg

            # Health should report the failure
            health = result[2]
            assert health.failed_dates >= 1
            assert len(health.failures) >= 1

    def test_coverage_error_raised_when_all_fail(self) -> None:
        """When all dates fail, FactorCoverageError is raised."""
        from optimizer.factors import (
            FactorConstructionConfig,
            StandardizationConfig,
        )
        from research._factors import build_factor_scores_history

        prices, volumes, fundamentals, sector_mapping = _make_prices_volumes()

        with patch(
            "research._factors.compute_all_factors",
            side_effect=ValueError("always fails"),
        ), pytest.raises(FactorCoverageError, match="succeeded"):
            build_factor_scores_history(
                investable_prices=prices,
                investable_volumes=volumes,
                investable_fundamentals=fundamentals,
                assembly=_MockAssembly(),  # type: ignore[arg-type]
                factor_config=FactorConstructionConfig(),
                std_config=StandardizationConfig(),
                sector_mapping=sector_mapping,
                rebalance_freq=63,
                fundamental_history=None,
            )

    def test_no_coverage_error_when_enough_succeed(self) -> None:
        """When only the first call fails, no error is raised."""
        from optimizer.factors import (
            FactorConstructionConfig,
            StandardizationConfig,
        )
        from research._factors import build_factor_scores_history

        prices, volumes, fundamentals, sector_mapping = _make_prices_volumes()

        call_count = 0

        def _failing_first_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("synthetic failure")
            from optimizer.factors._construction import (
                compute_all_factors as real_fn,
            )
            return real_fn(*args, **kwargs)

        with patch(
            "research._factors.compute_all_factors",
            side_effect=_failing_first_call,
        ):
            result = build_factor_scores_history(
                investable_prices=prices,
                investable_volumes=volumes,
                investable_fundamentals=fundamentals,
                assembly=_MockAssembly(),  # type: ignore[arg-type]
                factor_config=FactorConstructionConfig(),
                std_config=StandardizationConfig(),
                sector_mapping=sector_mapping,
                rebalance_freq=63,
                fundamental_history=None,
            )
            health = result[2]
            assert isinstance(health, FactorBuildHealth)
            assert health.is_healthy

    def test_health_returned_as_third_element(self) -> None:
        """Happy path: health is returned as the third tuple element."""
        from optimizer.factors import (
            FactorConstructionConfig,
            StandardizationConfig,
        )
        from research._factors import build_factor_scores_history

        prices, volumes, fundamentals, sector_mapping = _make_prices_volumes()

        result = build_factor_scores_history(
            investable_prices=prices,
            investable_volumes=volumes,
            investable_fundamentals=fundamentals,
            assembly=_MockAssembly(),  # type: ignore[arg-type]
            factor_config=FactorConstructionConfig(),
            std_config=StandardizationConfig(),
            sector_mapping=sector_mapping,
            rebalance_freq=63,
            fundamental_history=None,
        )
        assert len(result) == 3
        health = result[2]
        assert isinstance(health, FactorBuildHealth)
        assert health.succeeded_dates == health.total_dates
        assert health.failed_dates == 0

    def test_custom_min_success_fraction(self) -> None:
        """A high threshold raises FactorCoverageError on partial failure."""
        from optimizer.factors import (
            FactorConstructionConfig,
            StandardizationConfig,
        )
        from research._factors import build_factor_scores_history

        prices, volumes, fundamentals, sector_mapping = _make_prices_volumes()

        call_count = 0

        def _failing_first_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("synthetic failure")
            from optimizer.factors._construction import (
                compute_all_factors as real_fn,
            )
            return real_fn(*args, **kwargs)

        with patch(
            "research._factors.compute_all_factors",
            side_effect=_failing_first_call,
        ), pytest.raises(FactorCoverageError):
            build_factor_scores_history(
                investable_prices=prices,
                investable_volumes=volumes,
                investable_fundamentals=fundamentals,
                assembly=_MockAssembly(),  # type: ignore[arg-type]
                factor_config=FactorConstructionConfig(),
                std_config=StandardizationConfig(),
                sector_mapping=sector_mapping,
                rebalance_freq=63,
                fundamental_history=None,
                min_success_fraction=1.0,
            )


# ---------------------------------------------------------------------------
# validate_factors VIF exception handling
# ---------------------------------------------------------------------------


class TestValidateFactorsVIFExceptionHandling:
    """Verify VIF exception narrowing in validate_factors."""

    @pytest.fixture()
    def _validation_inputs(self) -> tuple:
        """Build minimal inputs for validate_factors."""
        rng = np.random.default_rng(99)
        n_dates, n_tickers, n_factors = 20, 30, 4
        dates = pd.bdate_range("2024-01-01", periods=n_dates)
        tickers = [f"S{i}" for i in range(n_tickers)]
        factor_names = ["momentum_12_1", "volatility", "book_to_price", "roe"]

        factor_scores_history = {}
        for fn in factor_names:
            factor_scores_history[fn] = pd.DataFrame(
                rng.standard_normal((n_dates, n_tickers)),
                index=dates,
                columns=tickers,
            )

        returns_history = pd.DataFrame(
            rng.standard_normal((n_dates, n_tickers)),
            index=dates,
            columns=tickers,
        )

        standardized = pd.DataFrame(
            rng.standard_normal((n_tickers, n_factors)),
            index=tickers,
            columns=factor_names,
        )

        return factor_scores_history, returns_history, standardized

    def test_linalg_error_does_not_propagate(
        self, _validation_inputs: tuple
    ) -> None:
        from research._factors import validate_factors

        fsh, rh, std = _validation_inputs
        with patch(
            "research._factors.compute_vif",
            side_effect=np.linalg.LinAlgError("singular"),
        ):
            report = validate_factors(fsh, rh, std)
            assert report.vif_scores is None

    def test_value_error_does_not_propagate(
        self, _validation_inputs: tuple
    ) -> None:
        from research._factors import validate_factors

        fsh, rh, std = _validation_inputs
        with patch(
            "research._factors.compute_vif",
            side_effect=ValueError("bad shape"),
        ):
            report = validate_factors(fsh, rh, std)
            assert report.vif_scores is None

    def test_other_exceptions_propagate(
        self, _validation_inputs: tuple
    ) -> None:
        from research._factors import validate_factors

        fsh, rh, std = _validation_inputs
        with patch(
            "research._factors.compute_vif",
            side_effect=TypeError("wrong type"),
        ), pytest.raises(TypeError, match="wrong type"):
            validate_factors(fsh, rh, std)

    def test_warning_logged_on_linalg_error(
        self, _validation_inputs: tuple, caplog: pytest.LogCaptureFixture
    ) -> None:
        from research._factors import validate_factors

        fsh, rh, std = _validation_inputs
        with patch(
            "research._factors.compute_vif",
            side_effect=np.linalg.LinAlgError("singular"),
        ), caplog.at_level(logging.WARNING, logger="research._factors"):
            validate_factors(fsh, rh, std)
            assert any("VIF" in r.message for r in caplog.records)
