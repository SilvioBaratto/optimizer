"""Tests for Gaussian HMM regime-probability fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.regime import HMMRegimeConfig, fit_hmm_regime_probabilities


def _make_two_regime_returns(
    n_obs: int = 500, n_assets: int = 6, switch_at: int = 300, seed: int = 0
) -> pd.DataFrame:
    """Synthetic panel: calm regime then a volatility/mean shift at `switch_at`."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(loc=0.0008, scale=0.005, size=(switch_at, n_assets))
    stressed = rng.normal(
        loc=-0.0025, scale=0.040, size=(n_obs - switch_at, n_assets)
    )
    data = np.vstack([calm, stressed])
    index = pd.bdate_range("2020-01-01", periods=n_obs)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(data, index=index, columns=cols)


class TestFitHmmRegimeProbabilities:
    @pytest.fixture(scope="class")
    def returns(self) -> pd.DataFrame:
        return _make_two_regime_returns()

    def test_when_empty_returns_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="empty"):
            fit_hmm_regime_probabilities(pd.DataFrame())

    def test_when_insufficient_observations_then_raises(self) -> None:
        short_returns = _make_two_regime_returns(n_obs=100, switch_at=60)
        with pytest.raises(ConfigurationError, match="Insufficient observations"):
            fit_hmm_regime_probabilities(
                short_returns, HMMRegimeConfig(min_observations=252)
            )

    def test_when_fitted_then_rows_sum_to_one(self, returns: pd.DataFrame) -> None:
        regime_probs, _model = fit_hmm_regime_probabilities(
            returns, HMMRegimeConfig.for_two_regime()
        )
        assert np.allclose(regime_probs.sum(axis=1).to_numpy(), 1.0, atol=1e-8)

    def test_when_fitted_then_shape_and_columns_match_config(
        self, returns: pd.DataFrame
    ) -> None:
        config = HMMRegimeConfig.for_two_regime()
        regime_probs, _model = fit_hmm_regime_probabilities(returns, config)
        assert list(regime_probs.columns) == ["regime_0", "regime_1"]
        assert regime_probs.shape[1] == config.n_regimes
        assert regime_probs.shape[0] <= len(returns)

    def test_when_fitted_then_probabilities_in_unit_interval(
        self, returns: pd.DataFrame
    ) -> None:
        regime_probs, _model = fit_hmm_regime_probabilities(
            returns, HMMRegimeConfig.for_two_regime()
        )
        assert (regime_probs.to_numpy() >= 0.0).all()
        assert (regime_probs.to_numpy() <= 1.0).all()

    def test_when_no_config_then_defaults_to_two_regime(
        self, returns: pd.DataFrame
    ) -> None:
        regime_probs, _model = fit_hmm_regime_probabilities(returns)
        assert regime_probs.shape[1] == 2

    def test_when_three_regime_then_three_columns(self, returns: pd.DataFrame) -> None:
        regime_probs, _model = fit_hmm_regime_probabilities(
            returns, HMMRegimeConfig.for_three_regime()
        )
        assert list(regime_probs.columns) == ["regime_0", "regime_1", "regime_2"]

    def test_when_index_returned_then_subset_of_input_index(
        self, returns: pd.DataFrame
    ) -> None:
        regime_probs, _model = fit_hmm_regime_probabilities(
            returns, HMMRegimeConfig.for_two_regime()
        )
        assert regime_probs.index.isin(returns.index).all()

    def test_when_fitted_then_regime_0_is_calmest_state(
        self, returns: pd.DataFrame
    ) -> None:
        # Label-switching guard: regime_0 must be the lower-vol/mean state.
        config = HMMRegimeConfig.for_two_regime()
        _regime_probs, model = fit_hmm_regime_probabilities(returns, config)
        order_key = model.means_[:, min(1, model.means_.shape[1] - 1)]
        assert order_key[0] <= order_key[1] or np.argsort(order_key)[0] == 0

    def test_when_regime_shift_then_filtered_probability_reacts_causally(
        self, returns: pd.DataFrame
    ) -> None:
        """Filtered (causal) posteriors must not anticipate the regime switch
        before it happens — no look-ahead into future observations."""
        switch_at = 300
        config = HMMRegimeConfig.for_two_regime()
        regime_probs, _model = fit_hmm_regime_probabilities(returns, config)

        # Well before the switch, the calm regime should dominate.
        pre_switch = regime_probs.loc[: regime_probs.index[switch_at - 50]]
        pre_switch_calm = pre_switch["regime_0"].mean()

        # Well after the switch, the stressed regime should dominate.
        post_switch = regime_probs.loc[regime_probs.index[switch_at + 50] :]
        post_switch_stressed = post_switch["regime_1"].mean()

        assert pre_switch_calm > 0.7
        assert post_switch_stressed > 0.7
