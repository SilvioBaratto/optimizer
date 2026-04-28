"""Property-based tests for the optimizer library using Hypothesis.

Each test encodes a mathematical invariant that must hold for all valid
inputs.  Fast tests use max_examples=100; slower tests use max_examples=5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from optimizer.factors._standardization import (
    winsorize_cross_section,
    z_score_standardize,
)
from optimizer.rebalancing._rebalancer import (
    compute_drifted_weights,
    compute_rebalancing_cost,
    compute_turnover,
)

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_N_ASSETS = st.integers(min_value=2, max_value=8)


def _weight_array(n: int) -> st.SearchStrategy[np.ndarray]:
    """Return a strategy producing a normalized long-only weight vector of length n."""
    raw = st.lists(
        st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=n,
        max_size=n,
    )

    def _normalize(vals: list[float]) -> np.ndarray:
        arr = np.array(vals, dtype=np.float64)
        total = arr.sum()
        if total == 0.0:
            arr = np.ones(n, dtype=np.float64)
        return arr / arr.sum()

    return raw.map(_normalize)


def _returns_array(n: int) -> st.SearchStrategy[np.ndarray]:
    """Return strategy for single-period returns > -1 (no total loss)."""
    return st.lists(
        st.floats(
            min_value=-0.99,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=n,
        max_size=n,
    ).map(lambda v: np.array(v, dtype=np.float64))


# ---------------------------------------------------------------------------
# Test 1: drifted weights sum to 1
# ---------------------------------------------------------------------------


@given(
    n=_N_ASSETS.flatmap(
        lambda n: st.tuples(
            st.just(n),
            _weight_array(n),
            _returns_array(n),
        )
    )
)
@settings(max_examples=100)
def test_drifted_weights_sum_to_one(
    n: tuple[int, np.ndarray, np.ndarray],
) -> None:
    """compute_drifted_weights preserves weight normalization when total > 0.

    For any non-negative weight vector summing to 1 and any returns > -1,
    the drifted weights must also sum to 1 because the grown portfolio
    values are re-normalized by their total.
    """
    _, weights, returns = n
    drifted = compute_drifted_weights(weights, returns)
    total = drifted.sum()
    # The function returns unnormalized grown values only when total == 0,
    # which cannot happen here because weights >= 0, returns > -1, and
    # at least one weight > 0 (ensured by normalization).
    assert np.isclose(total, 1.0, atol=1e-10), (
        "Drifted weights must sum to 1 when total portfolio value > 0"
    )


# ---------------------------------------------------------------------------
# Test 2: drifted weights non-negative
# ---------------------------------------------------------------------------


@given(
    n=_N_ASSETS.flatmap(
        lambda n: st.tuples(
            st.just(n),
            _weight_array(n),
            _returns_array(n),
        )
    )
)
@settings(max_examples=100)
def test_drifted_weights_non_negative(
    n: tuple[int, np.ndarray, np.ndarray],
) -> None:
    """compute_drifted_weights produces non-negative weights.

    When all input weights are non-negative and all returns satisfy
    r > -1, the grown values weights * (1 + returns) are non-negative
    and so are the drifted weights after normalization.
    """
    _, weights, returns = n
    drifted = compute_drifted_weights(weights, returns)
    assert np.all(drifted >= -1e-14), (
        "Drifted weights must be non-negative for non-negative inputs with returns > -1"
    )


# ---------------------------------------------------------------------------
# Test 3: turnover bounded [0, 1]
# ---------------------------------------------------------------------------


@given(
    n=_N_ASSETS.flatmap(
        lambda n: st.tuples(
            st.just(n),
            _weight_array(n),
            _weight_array(n),
        )
    )
)
@settings(max_examples=100)
def test_turnover_bounded_zero_to_one(
    n: tuple[int, np.ndarray, np.ndarray],
) -> None:
    """One-way turnover is in [0, 1] for normalized long-only portfolios.

    The one-way turnover is sum(|w_current - w_target|) / 2.  Because both
    weight vectors sum to 1 and all entries are in [0, 1], the maximum
    possible sum of absolute differences is 2 (complete portfolio rotation),
    giving a one-way turnover of at most 1.  The lower bound is 0.
    """
    _, current, target = n
    turnover = compute_turnover(current, target)
    assert 0.0 <= turnover <= 1.0 + 1e-12, (
        "One-way turnover must lie in [0, 1] for normalized long-only portfolios"
    )


# ---------------------------------------------------------------------------
# Test 4: rebalancing cost non-negative
# ---------------------------------------------------------------------------


@given(
    n=_N_ASSETS.flatmap(
        lambda n: st.tuples(
            st.just(n),
            _weight_array(n),
            _weight_array(n),
            st.floats(
                min_value=0.0,
                max_value=0.05,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
)
@settings(max_examples=100)
def test_rebalancing_cost_non_negative(
    n: tuple[int, np.ndarray, np.ndarray, float],
) -> None:
    """Total rebalancing cost is non-negative with non-negative cost rates.

    The cost is computed as sum(costs * |target - current|).  With
    non-negative cost rates and absolute trade sizes, the result is always
    >= 0.
    """
    _, current, target, cost_rate = n
    cost = compute_rebalancing_cost(current, target, cost_rate)
    assert cost >= -1e-14, (
        "Rebalancing cost must be non-negative with non-negative cost rates"
    )


# ---------------------------------------------------------------------------
# Test 5: z-score produces zero mean and unit variance
# ---------------------------------------------------------------------------


@given(
    raw=st.lists(
        st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=10,
        max_size=200,
    ).filter(lambda v: len(set(v)) >= 3)
)
@settings(max_examples=100)
def test_z_score_zero_mean_unit_variance(raw: list[float]) -> None:
    """z_score_standardize produces output with mean ~0 and std ~1.

    For any input with at least 3 distinct finite values (ensuring non-zero
    standard deviation), the z-scored output must have mean close to zero
    and sample standard deviation close to one.
    """
    scores = pd.Series(raw, dtype=float)
    result = z_score_standardize(scores)
    assert np.isclose(result.mean(), 0.0, atol=1e-10), (
        "Z-scored output must have mean close to zero"
    )
    assert np.isclose(result.std(), 1.0, atol=1e-10), (
        "Z-scored output must have sample std close to one"
    )


# ---------------------------------------------------------------------------
# Test 6: sample covariance is positive semi-definite
# ---------------------------------------------------------------------------


@given(
    n_cols=st.integers(min_value=2, max_value=5).flatmap(
        lambda nc: st.tuples(
            st.just(nc),
            st.lists(
                st.floats(
                    min_value=-0.2,
                    max_value=0.2,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=(nc + 1) * nc,
                max_size=200 * nc,
            ),
        )
    )
)
@settings(max_examples=100)
def test_sample_covariance_psd(n_cols: tuple[int, list[float]]) -> None:
    """Sample covariance matrices computed by np.cov are positive semi-definite.

    A sample covariance matrix is always PSD by construction.  All
    eigenvalues must be >= 0 up to floating-point rounding errors.
    """
    nc, flat = n_cols
    n_rows = len(flat) // nc
    arr = np.array(flat[: n_rows * nc], dtype=np.float64).reshape(n_rows, nc)
    cov = np.cov(arr, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-10), (
        "All eigenvalues of the sample covariance must be >= -1e-10 (PSD)"
    )


# ---------------------------------------------------------------------------
# Test 7: winsorized scores bounded by percentile clip points
# ---------------------------------------------------------------------------


@given(
    raw=st.lists(
        st.floats(
            min_value=-1e9,
            max_value=1e9,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=5,
        max_size=300,
    ),
    lower_pct=st.floats(min_value=0.01, max_value=0.1),
    upper_pct=st.floats(min_value=0.9, max_value=0.99),
)
@settings(max_examples=100)
def test_winsorized_scores_bounded(
    raw: list[float],
    lower_pct: float,
    upper_pct: float,
) -> None:
    """winsorize_cross_section clips all values to the specified percentile range.

    After winsorization, no value in the result may exceed the upper
    percentile boundary of the original distribution, and none may fall
    below the lower percentile boundary.
    """
    scores = pd.Series(raw, dtype=float)
    lower_bound = scores.quantile(lower_pct)
    upper_bound = scores.quantile(upper_pct)
    result = winsorize_cross_section(scores, lower_pct, upper_pct)
    assert (result >= lower_bound - 1e-10).all(), (
        "Winsorized scores must not fall below the lower percentile bound"
    )
    assert (result <= upper_bound + 1e-10).all(), (
        "Winsorized scores must not exceed the upper percentile bound"
    )


# ---------------------------------------------------------------------------
# Test 8: rank-percentile composite stays in [0, 1]
# ---------------------------------------------------------------------------


@given(
    raw=st.lists(
        st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=2,
        max_size=500,
    )
)
@settings(max_examples=100)
def test_rank_percentile_bounded_zero_to_one(raw: list[float]) -> None:
    """Rank transform scores / N stays in the unit interval [0, 1].

    Given any finite scores, ranking them (1-based) and dividing by the
    number of elements N produces values in [1/N, 1].  This is the
    foundation of the rank-percentile composite scoring step.
    """
    scores = pd.Series(raw, dtype=float)
    n = len(scores)
    rank_scores = scores.rank() / n
    assert (rank_scores >= 0.0).all(), "Rank-percentile values must be >= 0"
    assert (rank_scores <= 1.0 + 1e-12).all(), "Rank-percentile values must be <= 1"
