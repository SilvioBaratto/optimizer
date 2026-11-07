"""A per-asset weight cap that cannot sum to the budget says so before the solver does.

``max_weights=0.10`` with a fully invested budget needs at least ten assets. With fewer
the problem is infeasible and the solver reports a generic convex-optimisation failure
after the fit; the presets that default to that cap (``for_max_sharpe_diversified``,
``for_concentrated_sharpe``, ``for_long_only_capped``, ...) hit it on any small
universe.
The estimators now warn at fit time, naming the cap, the asset count and the largest
reachable total, and otherwise behave exactly like the skfolio classes they extend.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import MaximumDiversification, MeanRisk
from sklearn.base import clone

from optimizer.optimization import (
    MaxDiversificationConfig,
    MeanRiskConfig,
    build_max_diversification,
    build_mean_risk,
)

_CAP_WARNING = r"cannot reach the budget"


def _returns(n_assets: int, n_obs: int = 250, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0004, 0.01, (n_obs, n_assets)),
        columns=[f"A{i}" for i in range(n_assets)],
        index=pd.bdate_range("2024-01-01", periods=n_obs),
    )


def _capacity_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message) for w in caught if "cannot reach the budget" in str(w.message)
    ]


def _fit_collecting(est, X: pd.DataFrame) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Infeasibility itself is the solver's business here, not this test's.
        with contextlib.suppress(Exception):
            est.fit(X)
    return _capacity_warnings(caught)


def test_a_cap_that_cannot_fill_the_budget_warns_for_mean_risk() -> None:
    """Eight assets at 0.10 each can only reach 0.8 of a budget of 1."""
    est = build_mean_risk(MeanRiskConfig(max_weights=0.10))

    (message,) = _fit_collecting(est, _returns(8))

    assert "max_weights=0.1" in message
    assert "8 assets" in message
    assert "0.8" in message


def test_a_cap_that_can_fill_the_budget_is_silent() -> None:
    """Twelve assets at 0.10 reach 1.2, which covers the budget."""
    est = build_mean_risk(MeanRiskConfig(max_weights=0.10))

    assert _fit_collecting(est, _returns(12)) == []


@pytest.mark.parametrize(
    "config",
    [
        MeanRiskConfig(),
        MeanRiskConfig(max_weights=None),
        MeanRiskConfig(max_weights=0.10, budget=None),
    ],
)
def test_nothing_to_check_means_no_warning(config: MeanRiskConfig) -> None:
    """No cap, or no budget to fill, gives the check nothing to compare."""
    assert _fit_collecting(build_mean_risk(config), _returns(8)) == []


def test_the_check_does_not_change_what_the_fit_does() -> None:
    """With ``raise_on_failure=False`` an infeasible fit still returns, never raises."""
    est = build_mean_risk(MeanRiskConfig(max_weights=0.10, raise_on_failure=False))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(_returns(8))  # must not raise


def test_the_estimator_is_still_a_mean_risk_and_clones() -> None:
    """Callers keep getting a `MeanRisk`, and a clone keeps the check."""
    est = build_mean_risk(MeanRiskConfig(max_weights=0.10))

    assert isinstance(est, MeanRisk)
    twin = clone(est)
    assert isinstance(twin, MeanRisk)
    assert twin.get_params()["max_weights"] == 0.10
    assert len(_fit_collecting(twin, _returns(8))) == 1


def test_the_capped_long_only_max_diversification_preset_warns_too() -> None:
    """``for_long_only_capped`` is the fourth preset that defaults to a 0.10 cap."""
    est = build_max_diversification(MaxDiversificationConfig.for_long_only_capped())

    (message,) = _fit_collecting(est, _returns(8))

    assert isinstance(est, MaximumDiversification)
    assert "8 assets" in message


def test_max_diversification_with_enough_assets_is_silent() -> None:
    """The capped preset fits quietly once the universe is large enough."""
    est = build_max_diversification(MaxDiversificationConfig.for_long_only_capped())

    assert _fit_collecting(est, _returns(12)) == []
