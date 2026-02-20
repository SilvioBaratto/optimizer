"""Factor construction, scoring, and selection for stock pre-selection."""

from optimizer.factors._alpha import factor_scores_to_expected_returns
from optimizer.factors._config import (
    FACTOR_GROUP_MAPPING,
    GROUP_WEIGHT_TIER,
    CompositeMethod,
    CompositeScoringConfig,
    FactorConstructionConfig,
    FactorGroupType,
    FactorIntegrationConfig,
    FactorType,
    FactorValidationConfig,
    GroupWeight,
    MacroRegime,
    PublicationLagConfig,
    RegimeTiltConfig,
    SelectionConfig,
    SelectionMethod,
    StandardizationConfig,
    StandardizationMethod,
)
from optimizer.factors._construction import (
    align_to_pit,
    compute_all_factors,
    compute_factor,
)
from optimizer.factors._diagnostics import (
    FactorPCAResult,
    compute_factor_pca,
    flag_redundant_factors,
)
from optimizer.factors._integration import (
    build_factor_bl_views,
    build_factor_exposure_constraints,
    estimate_factor_premia,
)
from optimizer.factors._mimicking import (
    QuintileSpreadResult,
    build_factor_mimicking_portfolios,
    compute_cross_factor_correlation,
    compute_quintile_spread,
)
from optimizer.factors._oos_validation import (
    FactorOOSConfig,
    FactorOOSResult,
    run_factor_oos_validation,
)
from optimizer.factors._regime import (
    apply_regime_tilts,
    classify_regime,
    get_regime_tilts,
)
from optimizer.factors._ml_scoring import (
    FittedMLModel,
    fit_gbt_composite,
    fit_ridge_composite,
    predict_composite_scores,
)
from optimizer.factors._scoring import (
    compute_composite_score,
    compute_equal_weight_composite,
    compute_group_scores,
    compute_ic_weighted_composite,
    compute_icir_weighted_composite,
    compute_ml_composite,
)
from optimizer.factors._selection import (
    apply_sector_balance,
    select_fixed_count,
    select_quantile,
    select_stocks,
)
from optimizer.factors._standardization import (
    neutralize_sector,
    rank_normal_standardize,
    standardize_all_factors,
    standardize_factor,
    winsorize_cross_section,
    z_score_standardize,
)
from optimizer.factors._validation import (
    CorrectedPValues,
    FactorValidationReport,
    ICResult,
    ICStats,
    QuantileSpreadResult,
    benjamini_hochberg,
    compute_ic_series,
    compute_ic_stats,
    compute_icir,
    compute_monthly_ic,
    compute_newey_west_tstat,
    compute_quantile_spread,
    compute_vif,
    correct_pvalues,
    run_factor_validation,
    validate_factor_universe,
)

__all__ = [
    # Config enums
    "CompositeMethod",
    "FactorGroupType",
    "FactorType",
    "GroupWeight",
    "MacroRegime",
    "SelectionMethod",
    "StandardizationMethod",
    # Config dataclasses
    "CompositeScoringConfig",
    "FactorConstructionConfig",
    "FactorIntegrationConfig",
    "PublicationLagConfig",
    "FactorValidationConfig",
    "RegimeTiltConfig",
    "SelectionConfig",
    "StandardizationConfig",
    # Mapping constants
    "FACTOR_GROUP_MAPPING",
    "GROUP_WEIGHT_TIER",
    # Construction
    "align_to_pit",
    "compute_all_factors",
    "compute_factor",
    # Standardization
    "neutralize_sector",
    "rank_normal_standardize",
    "standardize_all_factors",
    "standardize_factor",
    "winsorize_cross_section",
    "z_score_standardize",
    # OOS validation
    "FactorOOSConfig",
    "FactorOOSResult",
    "run_factor_oos_validation",
    # Scoring
    "compute_composite_score",
    "compute_equal_weight_composite",
    "compute_group_scores",
    "compute_ic_weighted_composite",
    "compute_icir_weighted_composite",
    "compute_ml_composite",
    # ML scoring
    "FittedMLModel",
    "fit_gbt_composite",
    "fit_ridge_composite",
    "predict_composite_scores",
    # Selection
    "apply_sector_balance",
    "select_fixed_count",
    "select_quantile",
    "select_stocks",
    # Regime
    "apply_regime_tilts",
    "classify_regime",
    "get_regime_tilts",
    # Validation
    "CorrectedPValues",
    "FactorValidationReport",
    "ICResult",
    "ICStats",
    "QuantileSpreadResult",
    "benjamini_hochberg",
    "compute_ic_series",
    "compute_ic_stats",
    "compute_icir",
    "compute_monthly_ic",
    "compute_newey_west_tstat",
    "compute_quantile_spread",
    "compute_vif",
    "correct_pvalues",
    "run_factor_validation",
    "validate_factor_universe",
    # Integration
    "build_factor_bl_views",
    "build_factor_exposure_constraints",
    "estimate_factor_premia",
    "factor_scores_to_expected_returns",
    # Mimicking portfolios and quintile analysis
    "QuintileSpreadResult",
    "build_factor_mimicking_portfolios",
    "compute_cross_factor_correlation",
    "compute_quintile_spread",
    # Diagnostics
    "FactorPCAResult",
    "compute_factor_pca",
    "flag_redundant_factors",
]
