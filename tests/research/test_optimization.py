"""Cycle-3 §7.3 Top-4 retighten loop tests (issue #537)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from optimizer.optimization import MeanRiskConfig
from research._optimization import (
    _SHRINK_FACTOR,
    _TOP4_THRESHOLD,
    _solve_with_retighten,
)


class _FakeMeanRisk:
    """Stand-in for ``skfolio.MeanRisk`` used by the retighten loop."""

    def __init__(self, max_weights: float, weights: np.ndarray) -> None:
        self.max_weights = max_weights
        self._weights = np.asarray(weights, dtype=float)

    def fit(self, X: pd.DataFrame) -> _FakeMeanRisk:
        self.weights_ = self._weights
        return self


def _config(max_weights: float = 0.10) -> MeanRiskConfig:
    return MeanRiskConfig(max_weights=max_weights)


def _max_w(config: MeanRiskConfig) -> float:
    return cast(float, config.max_weights)


def _builder_from_sequence(
    weight_vectors: list[np.ndarray],
) -> tuple[Any, list[float]]:
    """Builder that yields a fresh ``_FakeMeanRisk`` per call."""
    pending = list(weight_vectors)
    seen_max: list[float] = []

    def builder(config: MeanRiskConfig) -> _FakeMeanRisk:
        seen_max.append(_max_w(config))
        return _FakeMeanRisk(_max_w(config), pending.pop(0))

    return builder, seen_max


@pytest.fixture()
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(60, 8)))


@pytest.fixture()
def wide_returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(60, 20)))


class TestRetightenConstants:
    def test_when_top4_threshold_imported_then_equals_030(self) -> None:
        assert pytest.approx(0.30) == _TOP4_THRESHOLD

    def test_when_shrink_factor_imported_then_equals_095(self) -> None:
        assert pytest.approx(0.95) == _SHRINK_FACTOR


class TestSolveWithRetightenConvergence:
    def test_when_first_attempt_below_threshold_then_no_rebuild(
        self, wide_returns: pd.DataFrame
    ) -> None:
        # 20 assets uniform → top-4 = 4 * (1/20) = 0.20 < 0.30
        diversified = np.full(20, 1.0 / 20)
        opt = _FakeMeanRisk(0.10, diversified)
        builder, seen_max = _builder_from_sequence([])

        result, trace = _solve_with_retighten(
            opt, wide_returns, config=_config(0.10), builder=builder
        )

        assert result is opt
        assert seen_max == []  # builder never invoked
        assert len(trace) == 1
        assert trace[0]["attempt"] == 1
        assert trace[0]["top4"] == pytest.approx(0.20)
        assert trace[0]["max_weights"] == pytest.approx(0.10)

    def test_when_iterative_convergence_then_trace_records_each_attempt(
        self, wide_returns: pd.DataFrame
    ) -> None:
        concentrated = np.zeros(20)
        concentrated[:4] = 0.10  # top-4 = 0.40
        concentrated[4:] = 0.60 / 16  # remaining = 0.0375 each
        diversified = np.full(20, 1.0 / 20)  # top-4 = 0.20
        opt = _FakeMeanRisk(0.10, concentrated)
        builder, seen_max = _builder_from_sequence([concentrated, diversified])

        result, trace = _solve_with_retighten(
            opt, wide_returns, config=_config(0.10), builder=builder
        )

        # 3 attempts: initial (0.10), shrunk (0.095), shrunk (0.09025)
        assert len(trace) == 3
        assert trace[0]["max_weights"] == pytest.approx(0.10)
        assert trace[1]["max_weights"] == pytest.approx(0.10 * 0.95)
        assert trace[2]["max_weights"] == pytest.approx(0.10 * 0.95 * 0.95)
        assert trace[2]["top4"] < _TOP4_THRESHOLD
        assert isinstance(result, _FakeMeanRisk)
        assert seen_max == [
            pytest.approx(0.10 * 0.95),
            pytest.approx(0.10 * 0.95 * 0.95),
        ]


class TestSolveWithRetightenDivergence:
    def test_when_always_concentrated_then_runtime_error_after_5_retries(
        self, returns: pd.DataFrame
    ) -> None:
        concentrated = np.array([0.10, 0.10, 0.10, 0.10, 0.20, 0.20, 0.10, 0.10])
        opt = _FakeMeanRisk(0.10, concentrated)
        # 5 retries → builder invoked 5 times
        builder, _ = _builder_from_sequence([concentrated.copy() for _ in range(5)])

        with pytest.raises(RuntimeError, match="retighten"):
            _solve_with_retighten(opt, returns, config=_config(0.10), builder=builder)

    def test_when_runtime_error_message_names_top4_and_max_weights(
        self, returns: pd.DataFrame
    ) -> None:
        concentrated = np.array([0.10, 0.10, 0.10, 0.10, 0.15, 0.15, 0.15, 0.15])
        opt = _FakeMeanRisk(0.10, concentrated)
        builder, _ = _builder_from_sequence([concentrated.copy() for _ in range(5)])
        with pytest.raises(RuntimeError) as exc_info:
            _solve_with_retighten(opt, returns, config=_config(0.10), builder=builder)
        msg = str(exc_info.value).lower()
        assert "top4" in msg or "top-4" in msg
        assert "max_weights" in msg

    def test_when_max_retries_custom_then_attempts_are_bounded(
        self, returns: pd.DataFrame
    ) -> None:
        concentrated = np.array([0.20, 0.20, 0.20, 0.20, 0.05, 0.05, 0.05, 0.05])
        opt = _FakeMeanRisk(0.10, concentrated)
        # max_retries=2 → 2 builder calls
        builder, seen_max = _builder_from_sequence(
            [concentrated.copy() for _ in range(2)]
        )
        with pytest.raises(RuntimeError):
            _solve_with_retighten(
                opt,
                returns,
                config=_config(0.10),
                builder=builder,
                max_retries=2,
            )
        assert len(seen_max) == 2


class TestSolveWithRetightenSignature:
    def test_when_called_then_returns_estimator_and_list_trace(
        self, wide_returns: pd.DataFrame
    ) -> None:
        diversified = np.full(20, 1.0 / 20)
        opt = _FakeMeanRisk(0.10, diversified)
        builder, _ = _builder_from_sequence([])
        result, trace = _solve_with_retighten(
            opt, wide_returns, config=_config(0.10), builder=builder
        )
        assert hasattr(result, "weights_")
        assert isinstance(trace, list)
        assert all({"attempt", "top4", "max_weights"} <= set(e) for e in trace)
