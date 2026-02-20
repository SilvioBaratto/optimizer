"""Tests for VIF and PCA multicollinearity diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    FactorPCAResult,
    compute_factor_pca,
    compute_vif,
    flag_redundant_factors,
)

N_TICKERS = 200
N_FACTORS = 5
TICKERS = [f"T{i:03d}" for i in range(N_TICKERS)]
FACTORS = ["value", "momentum", "quality", "growth", "low_vol"]


@pytest.fixture()
def orthogonal_scores() -> pd.DataFrame:
    """Independent (orthogonal) factor scores — all VIFs ≈ 1."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((N_TICKERS, N_FACTORS)),
        index=TICKERS,
        columns=FACTORS,
    )


@pytest.fixture()
def collinear_scores(orthogonal_scores: pd.DataFrame) -> pd.DataFrame:
    """One factor is nearly a linear combination of the others → high VIF."""
    rng = np.random.default_rng(0)
    df = orthogonal_scores.copy()
    # redundant = sum of value + momentum + tiny noise
    df["redundant"] = (
        df["value"] + df["momentum"] + rng.normal(0, 0.01, N_TICKERS)
    )
    return df


# ---------------------------------------------------------------------------
# compute_vif
# ---------------------------------------------------------------------------


class TestComputeVIF:
    def test_returns_series(self, orthogonal_scores: pd.DataFrame) -> None:
        result = compute_vif(orthogonal_scores)
        assert isinstance(result, pd.Series)

    def test_index_matches_factor_names(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        result = compute_vif(orthogonal_scores)
        assert list(result.index) == FACTORS

    def test_orthogonal_factors_vif_near_one(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        result = compute_vif(orthogonal_scores)
        assert (result < 2.0).all(), (
            f"Expected VIF ≈ 1 for orthogonal factors, got:\n{result}"
        )

    def test_collinear_factor_has_high_vif(
        self, collinear_scores: pd.DataFrame
    ) -> None:
        result = compute_vif(collinear_scores)
        assert result["redundant"] > 10.0

    def test_vif_all_geq_one(self, orthogonal_scores: pd.DataFrame) -> None:
        result = compute_vif(orthogonal_scores)
        assert (result >= 1.0 - 1e-9).all()

    def test_perfectly_collinear_factor_gives_inf(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        df = orthogonal_scores.copy()
        # Exact linear combination — R² = 1 → VIF = ∞
        df["exact_combo"] = df["value"] + 2.0 * df["momentum"]
        result = compute_vif(df)
        assert np.isinf(result["exact_combo"]) or result["exact_combo"] > 1e6

    def test_single_factor_raises(self) -> None:
        single = pd.DataFrame({"value": np.random.default_rng(0).standard_normal(50)})
        with pytest.raises(ValueError, match="at least 2 factor columns"):
            compute_vif(single)

    def test_two_factors_works(self) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            rng.standard_normal((100, 2)), columns=["a", "b"]
        )
        result = compute_vif(df)
        assert len(result) == 2
        assert (result >= 1.0 - 1e-9).all()


# ---------------------------------------------------------------------------
# compute_factor_pca
# ---------------------------------------------------------------------------


class TestComputeFactorPCA:
    def test_returns_factor_pca_result(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        result = compute_factor_pca(orthogonal_scores)
        assert isinstance(result, FactorPCAResult)

    def test_loadings_shape_factors_x_components(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        result = compute_factor_pca(orthogonal_scores)
        n_factors = orthogonal_scores.shape[1]
        n_comps = result.loadings.shape[1]
        assert result.loadings.shape == (n_factors, n_comps)

    def test_loadings_row_index_matches_factor_names(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        result = compute_factor_pca(orthogonal_scores)
        assert list(result.loadings.index) == FACTORS

    def test_loadings_column_names_are_pc_labels(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        result = compute_factor_pca(orthogonal_scores)
        for i, col in enumerate(result.loadings.columns):
            assert col == f"PC{i + 1}"

    def test_explained_variance_ratio_sums_to_one(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        result = compute_factor_pca(orthogonal_scores)
        assert pytest.approx(result.explained_variance_ratio.sum(), abs=1e-6) == 1.0

    def test_explained_variance_ratio_descending(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        result = compute_factor_pca(orthogonal_scores)
        evr = result.explained_variance_ratio
        assert (np.diff(evr) <= 1e-10).all()

    def test_n_components_95pct_satisfies_threshold(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        result = compute_factor_pca(orthogonal_scores)
        n = result.n_components_95pct
        cumsum = np.cumsum(result.explained_variance_ratio)
        assert cumsum[n - 1] >= 0.95

    def test_n_components_95pct_is_minimal(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        result = compute_factor_pca(orthogonal_scores)
        n = result.n_components_95pct
        cumsum = np.cumsum(result.explained_variance_ratio)
        # No smaller n satisfies the threshold
        if n > 1:
            assert cumsum[n - 2] < 0.95

    def test_n_components_argument_limits_output(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        result = compute_factor_pca(orthogonal_scores, n_components=2)
        assert result.loadings.shape[1] == 2
        assert len(result.explained_variance_ratio) == 2

    def test_single_dominant_component_when_highly_correlated(self) -> None:
        """A single latent factor → 1 PC explains most variance."""
        rng = np.random.default_rng(5)
        latent = rng.standard_normal(N_TICKERS)
        df = pd.DataFrame(
            {f"f{i}": latent + rng.normal(0, 0.05, N_TICKERS) for i in range(5)},
            index=TICKERS,
        )
        result = compute_factor_pca(df)
        assert result.explained_variance_ratio[0] > 0.90
        assert result.n_components_95pct == 1

    def test_single_factor_raises(self) -> None:
        single = pd.DataFrame(
            {"value": np.random.default_rng(0).standard_normal(50)}
        )
        with pytest.raises(ValueError, match="at least 2 factor columns"):
            compute_factor_pca(single)

    def test_too_few_observations_raises(self) -> None:
        df = pd.DataFrame(
            np.random.default_rng(0).standard_normal((1, 3)),
            columns=["a", "b", "c"],
        )
        with pytest.raises(ValueError, match="at least 2 observations"):
            compute_factor_pca(df)


# ---------------------------------------------------------------------------
# flag_redundant_factors
# ---------------------------------------------------------------------------


class TestFlagRedundantFactors:
    def test_returns_list(self, orthogonal_scores: pd.DataFrame) -> None:
        result = flag_redundant_factors(orthogonal_scores)
        assert isinstance(result, list)

    def test_orthogonal_factors_not_flagged(
        self, orthogonal_scores: pd.DataFrame
    ) -> None:
        result = flag_redundant_factors(orthogonal_scores, vif_threshold=10.0)
        assert result == []

    def test_collinear_factor_flagged(
        self, collinear_scores: pd.DataFrame
    ) -> None:
        result = flag_redundant_factors(collinear_scores, vif_threshold=10.0)
        assert "redundant" in result

    def test_unrelated_factors_not_flagged(self) -> None:
        """Factors with no shared variance should have low VIF and not be flagged."""
        rng = np.random.default_rng(99)
        # Truly independent factors: no collinearity → VIF ≈ 1 → none flagged
        df = pd.DataFrame(
            rng.standard_normal((200, 4)),
            columns=["a", "b", "c", "d"],
        )
        result = flag_redundant_factors(df, vif_threshold=10.0)
        assert result == []

    def test_strict_threshold_flags_more(
        self, collinear_scores: pd.DataFrame
    ) -> None:
        loose = flag_redundant_factors(collinear_scores, vif_threshold=100.0)
        strict = flag_redundant_factors(collinear_scores, vif_threshold=5.0)
        assert len(strict) >= len(loose)

    def test_single_factor_raises(self) -> None:
        single = pd.DataFrame(
            {"value": np.random.default_rng(0).standard_normal(50)}
        )
        with pytest.raises(ValueError, match="at least 2 factor columns"):
            flag_redundant_factors(single)

    def test_result_preserves_column_order(
        self, collinear_scores: pd.DataFrame
    ) -> None:
        result = flag_redundant_factors(collinear_scores, vif_threshold=10.0)
        col_order = list(collinear_scores.columns)
        positions = [col_order.index(f) for f in result]
        assert positions == sorted(positions)
