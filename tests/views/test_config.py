"""Tests for view integration configs and enums."""

from __future__ import annotations

import pytest

from optimizer.moments import MomentEstimationConfig, MuEstimatorType
from optimizer.views import (
    BlackLittermanConfig,
    EntropyPoolingConfig,
    OpinionPoolingConfig,
    ViewUncertaintyMethod,
)


class TestViewUncertaintyMethod:
    def test_members(self) -> None:
        assert set(ViewUncertaintyMethod) == {
            ViewUncertaintyMethod.HE_LITTERMAN,
            ViewUncertaintyMethod.IDZOREK,
            ViewUncertaintyMethod.EMPIRICAL_TRACK_RECORD,
        }

    def test_str_serialization(self) -> None:
        assert ViewUncertaintyMethod.HE_LITTERMAN.value == "he_litterman"
        assert ViewUncertaintyMethod.IDZOREK.value == "idzorek"
        assert (
            ViewUncertaintyMethod.EMPIRICAL_TRACK_RECORD.value
            == "empirical_track_record"
        )


class TestBlackLittermanConfig:
    def test_default_values(self) -> None:
        views = ("AAPL == 0.05",)
        cfg = BlackLittermanConfig(views=views)
        assert cfg.views == views
        assert cfg.tau == 0.05
        assert cfg.risk_free_rate == 0.0
        assert cfg.uncertainty_method == ViewUncertaintyMethod.HE_LITTERMAN
        assert cfg.view_confidences is None
        assert cfg.groups is None
        assert cfg.prior_config is None
        assert cfg.use_factor_model is False
        assert cfg.residual_variance is True

    def test_frozen(self) -> None:
        cfg = BlackLittermanConfig(views=("AAPL == 0.05",))
        with pytest.raises(AttributeError):
            cfg.tau = 0.1  # type: ignore[misc]

    def test_custom_values(self) -> None:
        views = ("AAPL == 0.05", "MSFT == 0.03")
        groups = {"tech": ["AAPL", "MSFT"]}
        cfg = BlackLittermanConfig(
            views=views,
            tau=0.1,
            risk_free_rate=0.02,
            uncertainty_method=ViewUncertaintyMethod.IDZOREK,
            view_confidences=(0.8, 0.6),
            groups=groups,
            use_factor_model=True,
            residual_variance=False,
        )
        assert cfg.views == views
        assert cfg.tau == 0.1
        assert cfg.risk_free_rate == 0.02
        assert cfg.uncertainty_method == ViewUncertaintyMethod.IDZOREK
        assert cfg.view_confidences == (0.8, 0.6)
        assert cfg.groups == groups
        assert cfg.use_factor_model is True
        assert cfg.residual_variance is False

    def test_for_equilibrium(self) -> None:
        views = ("AAPL == 0.05",)
        cfg = BlackLittermanConfig.for_equilibrium(views)
        assert cfg.views == views
        assert cfg.prior_config is not None
        assert cfg.prior_config.mu_estimator == MuEstimatorType.EQUILIBRIUM
        assert cfg.use_factor_model is False

    def test_for_factor_model(self) -> None:
        views = ("AAPL == 0.05",)
        cfg = BlackLittermanConfig.for_factor_model(views)
        assert cfg.views == views
        assert cfg.prior_config is not None
        assert cfg.prior_config.mu_estimator == MuEstimatorType.EQUILIBRIUM
        assert cfg.use_factor_model is True

    def test_tau_zero_raises(self) -> None:
        """tau=0 raises ValueError (issue #68)."""
        with pytest.raises(ValueError, match="tau must be strictly positive"):
            BlackLittermanConfig(views=("AAPL == 0.05",), tau=0.0)

    def test_tau_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="tau must be strictly positive"):
            BlackLittermanConfig(views=("AAPL == 0.05",), tau=-0.01)


class TestEntropyPoolingConfig:
    def test_default_values(self) -> None:
        cfg = EntropyPoolingConfig()
        assert cfg.mean_views is None
        assert cfg.mean_inequality_views is None
        assert cfg.variance_views is None
        assert cfg.correlation_views is None
        assert cfg.skew_views is None
        assert cfg.kurtosis_views is None
        assert cfg.cvar_views is None
        assert cfg.cvar_beta == 0.95
        assert cfg.groups is None
        assert cfg.solver == "TNC"
        assert cfg.solver_params is None
        assert cfg.prior_config is None

    def test_frozen(self) -> None:
        cfg = EntropyPoolingConfig()
        with pytest.raises(AttributeError):
            cfg.solver = "SLSQP"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        mean_views = ("AAPL == 0.05",)
        variance_views = ("AAPL == 0.04",)
        correlation_views = ("AAPL; MSFT == 0.5",)
        cfg = EntropyPoolingConfig(
            mean_views=mean_views,
            variance_views=variance_views,
            correlation_views=correlation_views,
            cvar_beta=0.99,
            solver="SLSQP",
            solver_params={"maxiter": 500},
        )
        assert cfg.mean_views == mean_views
        assert cfg.variance_views == variance_views
        assert cfg.correlation_views == correlation_views
        assert cfg.cvar_beta == 0.99
        assert cfg.solver == "SLSQP"
        assert cfg.solver_params == {"maxiter": 500}

    def test_for_mean_views(self) -> None:
        mean_views = ("AAPL == 0.05", "MSFT == 0.03")
        cfg = EntropyPoolingConfig.for_mean_views(mean_views)
        assert cfg.mean_views == mean_views
        assert cfg.variance_views is None
        assert cfg.correlation_views is None

    def test_for_stress_test(self) -> None:
        var_views = ("AAPL == 0.04",)
        corr_views = ("AAPL; MSFT == 0.8",)
        cfg = EntropyPoolingConfig.for_stress_test(var_views, corr_views)
        assert cfg.variance_views == var_views
        assert cfg.correlation_views == corr_views
        assert cfg.mean_views is None

    def test_mean_inequality_views_custom(self) -> None:
        """mean_inequality_views field stores inequality views (issue #69)."""
        cfg = EntropyPoolingConfig(
            mean_inequality_views=("AAPL >= 0.03",),
        )
        assert cfg.mean_inequality_views == ("AAPL >= 0.03",)

    def test_mean_inequality_views_default_none(self) -> None:
        cfg = EntropyPoolingConfig()
        assert cfg.mean_inequality_views is None


class TestOpinionPoolingConfig:
    def test_default_values(self) -> None:
        cfg = OpinionPoolingConfig()
        assert cfg.opinion_probabilities is None
        assert cfg.is_linear_pooling is True
        assert cfg.divergence_penalty == 0.0
        assert cfg.n_jobs is None
        assert cfg.prior_config is None

    def test_frozen(self) -> None:
        cfg = OpinionPoolingConfig()
        with pytest.raises(AttributeError):
            cfg.is_linear_pooling = False  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = OpinionPoolingConfig(
            opinion_probabilities=(0.6, 0.4),
            is_linear_pooling=False,
            divergence_penalty=0.1,
            n_jobs=2,
            prior_config=MomentEstimationConfig(),
        )
        assert cfg.opinion_probabilities == (0.6, 0.4)
        assert cfg.is_linear_pooling is False
        assert cfg.divergence_penalty == 0.1
        assert cfg.n_jobs == 2
        assert cfg.prior_config is not None

    def test_probabilities_sum_above_one_raises(self) -> None:
        """Sum > 1.0 raises ValueError (issue #70)."""
        with pytest.raises(ValueError, match=r"sum to at most 1\.0"):
            OpinionPoolingConfig(opinion_probabilities=(0.6, 0.5))

    def test_probabilities_sum_one_accepted(self) -> None:
        cfg = OpinionPoolingConfig(opinion_probabilities=(0.6, 0.4))
        assert cfg.opinion_probabilities == (0.6, 0.4)

    def test_negative_probability_raises(self) -> None:
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            OpinionPoolingConfig(opinion_probabilities=(-0.1, 0.5))

    def test_probability_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            OpinionPoolingConfig(opinion_probabilities=(1.5, 0.0))

    def test_none_probabilities_accepted(self) -> None:
        cfg = OpinionPoolingConfig(opinion_probabilities=None)
        assert cfg.opinion_probabilities is None
