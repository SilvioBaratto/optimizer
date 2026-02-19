"""Configuration for factor construction, scoring, and selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FactorGroupType(str, Enum):
    """Factor group taxonomy."""

    VALUE = "value"
    PROFITABILITY = "profitability"
    INVESTMENT = "investment"
    MOMENTUM = "momentum"
    LOW_RISK = "low_risk"
    LIQUIDITY = "liquidity"
    DIVIDEND = "dividend"
    SENTIMENT = "sentiment"
    OWNERSHIP = "ownership"


class FactorType(str, Enum):
    """Individual factor identifiers."""

    # Value
    BOOK_TO_PRICE = "book_to_price"
    EARNINGS_YIELD = "earnings_yield"
    CASH_FLOW_YIELD = "cash_flow_yield"
    SALES_TO_PRICE = "sales_to_price"
    EBITDA_TO_EV = "ebitda_to_ev"
    # Profitability
    GROSS_PROFITABILITY = "gross_profitability"
    ROE = "roe"
    OPERATING_MARGIN = "operating_margin"
    PROFIT_MARGIN = "profit_margin"
    # Investment
    ASSET_GROWTH = "asset_growth"
    # Momentum
    MOMENTUM_12_1 = "momentum_12_1"
    # Low risk
    VOLATILITY = "volatility"
    BETA = "beta"
    # Liquidity
    AMIHUD_ILLIQUIDITY = "amihud_illiquidity"
    # Dividend
    DIVIDEND_YIELD = "dividend_yield"
    # Sentiment
    RECOMMENDATION_CHANGE = "recommendation_change"
    # Ownership
    NET_INSIDER_BUYING = "net_insider_buying"


class StandardizationMethod(str, Enum):
    """Cross-sectional standardization method."""

    Z_SCORE = "z_score"
    RANK_NORMAL = "rank_normal"


class CompositeMethod(str, Enum):
    """Composite scoring method."""

    EQUAL_WEIGHT = "equal_weight"
    IC_WEIGHTED = "ic_weighted"


class SelectionMethod(str, Enum):
    """Stock selection method."""

    FIXED_COUNT = "fixed_count"
    QUANTILE = "quantile"


class MacroRegime(str, Enum):
    """Macro-economic regime classification."""

    EXPANSION = "expansion"
    SLOWDOWN = "slowdown"
    RECESSION = "recession"
    RECOVERY = "recovery"


class GroupWeight(str, Enum):
    """Weight tier for factor groups."""

    CORE = "core"
    SUPPLEMENTARY = "supplementary"


# ---------------------------------------------------------------------------
# Mapping constants
# ---------------------------------------------------------------------------

FACTOR_GROUP_MAPPING: dict[FactorType, FactorGroupType] = {
    FactorType.BOOK_TO_PRICE: FactorGroupType.VALUE,
    FactorType.EARNINGS_YIELD: FactorGroupType.VALUE,
    FactorType.CASH_FLOW_YIELD: FactorGroupType.VALUE,
    FactorType.SALES_TO_PRICE: FactorGroupType.VALUE,
    FactorType.EBITDA_TO_EV: FactorGroupType.VALUE,
    FactorType.GROSS_PROFITABILITY: FactorGroupType.PROFITABILITY,
    FactorType.ROE: FactorGroupType.PROFITABILITY,
    FactorType.OPERATING_MARGIN: FactorGroupType.PROFITABILITY,
    FactorType.PROFIT_MARGIN: FactorGroupType.PROFITABILITY,
    FactorType.ASSET_GROWTH: FactorGroupType.INVESTMENT,
    FactorType.MOMENTUM_12_1: FactorGroupType.MOMENTUM,
    FactorType.VOLATILITY: FactorGroupType.LOW_RISK,
    FactorType.BETA: FactorGroupType.LOW_RISK,
    FactorType.AMIHUD_ILLIQUIDITY: FactorGroupType.LIQUIDITY,
    FactorType.DIVIDEND_YIELD: FactorGroupType.DIVIDEND,
    FactorType.RECOMMENDATION_CHANGE: FactorGroupType.SENTIMENT,
    FactorType.NET_INSIDER_BUYING: FactorGroupType.OWNERSHIP,
}

GROUP_WEIGHT_TIER: dict[FactorGroupType, GroupWeight] = {
    FactorGroupType.VALUE: GroupWeight.CORE,
    FactorGroupType.PROFITABILITY: GroupWeight.CORE,
    FactorGroupType.MOMENTUM: GroupWeight.CORE,
    FactorGroupType.LOW_RISK: GroupWeight.CORE,
    FactorGroupType.INVESTMENT: GroupWeight.SUPPLEMENTARY,
    FactorGroupType.LIQUIDITY: GroupWeight.SUPPLEMENTARY,
    FactorGroupType.DIVIDEND: GroupWeight.SUPPLEMENTARY,
    FactorGroupType.SENTIMENT: GroupWeight.SUPPLEMENTARY,
    FactorGroupType.OWNERSHIP: GroupWeight.SUPPLEMENTARY,
}


# ---------------------------------------------------------------------------
# Frozen dataclass configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorConstructionConfig:
    """Configuration for factor computation.

    Parameters
    ----------
    factors : tuple[FactorType, ...]
        Which factors to compute.
    momentum_lookback : int
        Lookback window for momentum in trading days.
    momentum_skip : int
        Recent days to skip for momentum (reversal avoidance).
    volatility_lookback : int
        Lookback window for volatility in trading days.
    beta_lookback : int
        Lookback window for beta estimation in trading days.
    amihud_lookback : int
        Lookback window for Amihud illiquidity in trading days.
    publication_lag : int
        Days to lag fundamental data for point-in-time correctness.
    """

    factors: tuple[FactorType, ...] = (
        FactorType.BOOK_TO_PRICE,
        FactorType.EARNINGS_YIELD,
        FactorType.GROSS_PROFITABILITY,
        FactorType.ROE,
        FactorType.ASSET_GROWTH,
        FactorType.MOMENTUM_12_1,
        FactorType.VOLATILITY,
        FactorType.DIVIDEND_YIELD,
    )
    momentum_lookback: int = 252
    momentum_skip: int = 21
    volatility_lookback: int = 252
    beta_lookback: int = 252
    amihud_lookback: int = 252
    publication_lag: int = 63

    @classmethod
    def for_core_factors(cls) -> FactorConstructionConfig:
        """Core factors with strongest empirical support."""
        return cls()

    @classmethod
    def for_all_factors(cls) -> FactorConstructionConfig:
        """All 17 factors."""
        return cls(factors=tuple(FactorType))


@dataclass(frozen=True)
class StandardizationConfig:
    """Configuration for cross-sectional factor standardization.

    Parameters
    ----------
    method : StandardizationMethod
        Z-score or rank-normal standardization.
    winsorize_lower : float
        Lower percentile for winsorization (0-1).
    winsorize_upper : float
        Upper percentile for winsorization (0-1).
    neutralize_sector : bool
        Whether to sector-neutralize scores.
    neutralize_country : bool
        Whether to country-neutralize scores.
    """

    method: StandardizationMethod = StandardizationMethod.Z_SCORE
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    neutralize_sector: bool = True
    neutralize_country: bool = False

    @classmethod
    def for_heavy_tailed(cls) -> StandardizationConfig:
        """Rank-normal for heavy-tailed distributions (e.g. value ratios)."""
        return cls(method=StandardizationMethod.RANK_NORMAL)

    @classmethod
    def for_normal(cls) -> StandardizationConfig:
        """Z-score for approximately normal factors (e.g. momentum)."""
        return cls(method=StandardizationMethod.Z_SCORE)


@dataclass(frozen=True)
class CompositeScoringConfig:
    """Configuration for composite score construction.

    Parameters
    ----------
    method : CompositeMethod
        Equal-weight or IC-weighted composite.
    ic_lookback : int
        Number of periods for IC estimation when using IC weighting.
    core_weight : float
        Relative weight for core factor groups.
    supplementary_weight : float
        Relative weight for supplementary factor groups.
    """

    method: CompositeMethod = CompositeMethod.EQUAL_WEIGHT
    ic_lookback: int = 36
    core_weight: float = 1.0
    supplementary_weight: float = 0.5

    @classmethod
    def for_equal_weight(cls) -> CompositeScoringConfig:
        """Equal-weight composite scoring."""
        return cls()

    @classmethod
    def for_ic_weighted(cls) -> CompositeScoringConfig:
        """IC-weighted composite scoring."""
        return cls(method=CompositeMethod.IC_WEIGHTED)


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for stock selection from scored universe.

    Parameters
    ----------
    method : SelectionMethod
        Fixed-count or quantile-based selection.
    target_count : int
        Number of stocks to select (for FIXED_COUNT).
    target_quantile : float
        Quantile threshold for selection (for QUANTILE, 0-1).
    exit_quantile : float
        Exit quantile for hysteresis (for QUANTILE).
    buffer_fraction : float
        Buffer zone fraction around selection boundary.
    sector_balance : bool
        Whether to enforce sector-proportional representation.
    sector_tolerance : float
        Maximum deviation from parent universe sector weights.
    """

    method: SelectionMethod = SelectionMethod.FIXED_COUNT
    target_count: int = 100
    target_quantile: float = 0.8
    exit_quantile: float = 0.7
    buffer_fraction: float = 0.1
    sector_balance: bool = True
    sector_tolerance: float = 0.03

    @classmethod
    def for_top_100(cls) -> SelectionConfig:
        """Select top 100 stocks by composite score."""
        return cls()

    @classmethod
    def for_top_quintile(cls) -> SelectionConfig:
        """Select top quintile by composite score."""
        return cls(
            method=SelectionMethod.QUANTILE,
            target_quantile=0.8,
            exit_quantile=0.7,
        )

    @classmethod
    def for_concentrated(cls) -> SelectionConfig:
        """Concentrated portfolio of top 30 stocks."""
        return cls(target_count=30, buffer_fraction=0.15)


@dataclass(frozen=True)
class RegimeTiltConfig:
    """Configuration for macro regime factor tilts.

    Per-regime multiplicative tilts stored as tuples of
    ``(group_name, tilt_factor)`` for frozen-dataclass compatibility.

    Parameters
    ----------
    enable : bool
        Whether to apply regime tilts.
    expansion_tilts : tuple[tuple[str, float], ...]
        Group tilts during expansion.
    slowdown_tilts : tuple[tuple[str, float], ...]
        Group tilts during slowdown.
    recession_tilts : tuple[tuple[str, float], ...]
        Group tilts during recession.
    recovery_tilts : tuple[tuple[str, float], ...]
        Group tilts during recovery.
    """

    enable: bool = False
    expansion_tilts: tuple[tuple[str, float], ...] = (
        ("momentum", 1.2),
        ("value", 0.8),
        ("low_risk", 0.8),
    )
    slowdown_tilts: tuple[tuple[str, float], ...] = (
        ("low_risk", 1.3),
        ("dividend", 1.2),
        ("momentum", 0.7),
    )
    recession_tilts: tuple[tuple[str, float], ...] = (
        ("low_risk", 1.5),
        ("profitability", 1.3),
        ("momentum", 0.5),
        ("value", 1.2),
    )
    recovery_tilts: tuple[tuple[str, float], ...] = (
        ("value", 1.3),
        ("momentum", 1.2),
        ("low_risk", 0.7),
    )

    @classmethod
    def for_moderate_tilts(cls) -> RegimeTiltConfig:
        """Enable moderate regime-conditional tilts."""
        return cls(enable=True)

    @classmethod
    def for_no_tilts(cls) -> RegimeTiltConfig:
        """Disable regime tilts (default)."""
        return cls(enable=False)


@dataclass(frozen=True)
class FactorValidationConfig:
    """Configuration for factor validation and statistical testing.

    Parameters
    ----------
    newey_west_lags : int
        Number of lags for Newey-West t-statistic.
    t_stat_threshold : float
        Minimum absolute t-statistic for significance.
    fdr_alpha : float
        False discovery rate alpha level.
    n_quantiles : int
        Number of quantiles for spread analysis.
    fmp_top_pct : float
        Top percentile for factor-mimicking portfolios.
    fmp_bottom_pct : float
        Bottom percentile for factor-mimicking portfolios.
    """

    newey_west_lags: int = 6
    t_stat_threshold: float = 2.0
    fdr_alpha: float = 0.05
    n_quantiles: int = 5
    fmp_top_pct: float = 0.2
    fmp_bottom_pct: float = 0.2

    @classmethod
    def for_strict(cls) -> FactorValidationConfig:
        """Strict validation thresholds."""
        return cls(t_stat_threshold=3.0, fdr_alpha=0.01)

    @classmethod
    def for_standard(cls) -> FactorValidationConfig:
        """Standard validation thresholds."""
        return cls()


@dataclass(frozen=True)
class FactorIntegrationConfig:
    """Configuration for bridging factor scores to optimization.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate for expected return mapping.
    market_risk_premium : float
        Annual equity risk premium.
    use_black_litterman : bool
        Whether to generate Black-Litterman views from factor scores.
    exposure_lower_bound : float
        Lower bound for factor exposure constraints.
    exposure_upper_bound : float
        Upper bound for factor exposure constraints.
    """

    risk_free_rate: float = 0.04
    market_risk_premium: float = 0.05
    use_black_litterman: bool = False
    exposure_lower_bound: float = -0.5
    exposure_upper_bound: float = 0.5

    @classmethod
    def for_linear_mapping(cls) -> FactorIntegrationConfig:
        """Direct factor score to expected return mapping."""
        return cls()

    @classmethod
    def for_black_litterman(cls) -> FactorIntegrationConfig:
        """Factor-based Black-Litterman views."""
        return cls(use_black_litterman=True)
