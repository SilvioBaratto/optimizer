"""Configuration for factor construction, scoring, and selection."""

from __future__ import annotations

from dataclasses import dataclass, field
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


class WinsorizeMethod(str, Enum):
    """Winsorization method for outlier treatment."""

    PERCENTILE = "percentile"
    MAD = "mad"


class CompositeMethod(str, Enum):
    """Composite scoring method."""

    EQUAL_WEIGHT = "equal_weight"
    IC_WEIGHTED = "ic_weighted"
    ICIR_WEIGHTED = "icir_weighted"
    RIDGE_WEIGHTED = "ridge_weighted"
    GBT_WEIGHTED = "gbt_weighted"


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
    UNKNOWN = "unknown"


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

HEAVY_TAILED_FACTORS: frozenset[str] = frozenset(
    {
        "book_to_price",
        "earnings_yield",
        "cash_flow_yield",
        "sales_to_price",
        "ebitda_to_ev",
        "asset_growth",
        "dividend_yield",
        "amihud_illiquidity",
        "accruals",
    }
)


# ---------------------------------------------------------------------------
# Frozen dataclass configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicationLagConfig:
    """Differentiated publication lags by data source type.

    Each source has an independent delay between the period end date and
    the date the data is reliably available for use in factor construction.
    Using source-specific lags avoids look-ahead bias when aligning
    fundamental data to price dates.

    Parameters
    ----------
    annual_days : int
        Lag for annual financial statements (days after fiscal year end).
        Default: 90 days (~3 months for 10-K filing).
    quarterly_days : int
        Lag for quarterly financial statements (days after quarter end).
        Default: 45 days (~6 weeks for 10-Q filing).
    analyst_days : int
        Lag for analyst estimates and recommendations.
        Default: 5 days (short dissemination buffer).
    macro_days : int
        Lag for macroeconomic indicators (release lag + revision lag).
        Default: 63 days (~2 months).
    """

    annual_days: int = 90
    quarterly_days: int = 45
    analyst_days: int = 5
    macro_days: int = 63

    @classmethod
    def uniform(cls, days: int) -> PublicationLagConfig:
        """Create a config with the same lag applied to all sources."""
        return cls(
            annual_days=days,
            quarterly_days=days,
            analyst_days=days,
            macro_days=days,
        )


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
    publication_lag : PublicationLagConfig
        Per-source publication lags for point-in-time correctness.
        Pass a plain ``int`` for a uniform lag across all sources
        (backward-compatible; converted to :class:`PublicationLagConfig`
        automatically).
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
    publication_lag: PublicationLagConfig = field(default_factory=PublicationLagConfig)

    def __post_init__(self) -> None:
        # Runtime backward-compat: accept plain int for uniform lag
        if not isinstance(self.publication_lag, PublicationLagConfig):
            object.__setattr__(
                self,
                "publication_lag",
                PublicationLagConfig.uniform(int(self.publication_lag)),
            )

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
        Z-score or rank-normal standardization.  Default is ``RANK_NORMAL``
        following MSCI Barra USE4 and Gu/Kelly/Xiu (2020) best practice for
        heavy-tailed financial factor distributions.
    winsorize_method : WinsorizeMethod
        Outlier treatment method.  ``PERCENTILE`` clips at fixed quantiles;
        ``MAD`` clips at median +/- k * 1.4826 * MAD.
    winsorize_lower : float
        Lower percentile for winsorization (0-1, used with PERCENTILE).
    winsorize_upper : float
        Upper percentile for winsorization (0-1, used with PERCENTILE).
    neutralize_sector : bool
        Whether to sector-neutralize scores.
    neutralize_country : bool
        Whether to country-neutralize scores.
    factor_method_overrides : tuple[tuple[str, str], ...]
        Per-factor standardization method overrides as
        ``(factor_name, method_value)`` pairs.  When non-empty, each factor
        is standardized with its assigned method; factors not in the map
        fall back to ``method``.
    """

    method: StandardizationMethod = StandardizationMethod.RANK_NORMAL
    winsorize_method: WinsorizeMethod = WinsorizeMethod.PERCENTILE
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    neutralize_sector: bool = True
    neutralize_country: bool = False
    re_standardize_after_neutralization: bool = False
    factor_method_overrides: tuple[tuple[str, str], ...] = ()

    @classmethod
    def for_heavy_tailed(cls) -> StandardizationConfig:
        """Rank-normal for heavy-tailed distributions (e.g. value ratios)."""
        return cls(method=StandardizationMethod.RANK_NORMAL)

    @classmethod
    def for_normal(cls) -> StandardizationConfig:
        """Z-score for approximately normal factors (e.g. momentum)."""
        return cls(method=StandardizationMethod.Z_SCORE)

    @classmethod
    def for_z_score(cls) -> StandardizationConfig:
        """Z-score standardization (backward-compatibility alias)."""
        return cls(method=StandardizationMethod.Z_SCORE)

    @classmethod
    def for_per_factor(cls) -> StandardizationConfig:
        """Per-factor method: RANK_NORMAL for heavy-tailed, Z_SCORE for normal.

        Based on MSCI Barra USE4 and Gu/Kelly/Xiu (2020) classification.
        Heavy-tailed: value ratios, illiquidity, dividend yield, accruals,
        asset growth.  Approximately normal: momentum, volatility, beta.
        """
        approximately_normal = frozenset(
            {
                FactorType.MOMENTUM_12_1.value,
                FactorType.VOLATILITY.value,
                FactorType.BETA.value,
            }
        )
        overrides: list[tuple[str, str]] = []
        for ft in FactorType:
            if ft.value in approximately_normal:
                overrides.append((ft.value, StandardizationMethod.Z_SCORE.value))
            else:
                overrides.append((ft.value, StandardizationMethod.RANK_NORMAL.value))
        return cls(
            method=StandardizationMethod.RANK_NORMAL,
            factor_method_overrides=tuple(sorted(overrides)),
        )

    @classmethod
    def for_mad_winsorize(cls) -> StandardizationConfig:
        """MAD-based winsorization (MSCI Barra +/-3 MAD convention)."""
        return cls(winsorize_method=WinsorizeMethod.MAD)


@dataclass(frozen=True)
class CompositeScoringConfig:
    """Configuration for composite score construction.

    Parameters
    ----------
    method : CompositeMethod
        Equal-weight, IC-weighted, ICIR-weighted, ridge, or GBT composite.
    ic_lookback : int
        Number of periods for IC estimation when using IC weighting.
    core_weight : float
        Relative weight for core factor groups.
    supplementary_weight : float
        Relative weight for supplementary factor groups.
    ridge_alpha : float
        L2 regularisation strength for ``RIDGE_WEIGHTED``.  Passed as the
        single candidate to ``RidgeCV``; increase for more shrinkage.
    gbt_max_depth : int
        Maximum tree depth for ``GBT_WEIGHTED``.
    gbt_n_estimators : int
        Number of boosting rounds for ``GBT_WEIGHTED``.
    min_coverage_groups : int
        Minimum number of non-NaN group scores required.  Tickers with
        fewer available groups receive NaN composite and are excluded from
        selection.  0 disables the threshold (default).
    return_coverage : bool
        When True, ``compute_composite_score`` returns a DataFrame with
        columns ``["composite", "coverage_ratio"]`` instead of a Series.
    """

    method: CompositeMethod = CompositeMethod.EQUAL_WEIGHT
    ic_lookback: int = 36
    core_weight: float = 1.0
    supplementary_weight: float = 0.5
    ridge_alpha: float = 1.0
    gbt_max_depth: int = 3
    gbt_n_estimators: int = 50
    min_coverage_groups: int = 0
    return_coverage: bool = False

    @classmethod
    def for_equal_weight(cls) -> CompositeScoringConfig:
        """Equal-weight composite scoring."""
        return cls()

    @classmethod
    def for_ic_weighted(cls) -> CompositeScoringConfig:
        """IC-weighted composite scoring (raw IC magnitude)."""
        return cls(method=CompositeMethod.IC_WEIGHTED)

    @classmethod
    def for_icir_weighted(cls) -> CompositeScoringConfig:
        """ICIR-weighted composite scoring (mean IC / std IC).

        Penalises inconsistent predictors by dividing mean IC by IC
        volatility before normalising weights.
        """
        return cls(method=CompositeMethod.ICIR_WEIGHTED)

    @classmethod
    def for_ridge_weighted(cls) -> CompositeScoringConfig:
        """Ridge regression composite scoring.

        Learns optimal linear factor weights from historical data with
        L2 regularisation, avoiding the need for IC proxies.
        """
        return cls(method=CompositeMethod.RIDGE_WEIGHTED)

    @classmethod
    def for_gbt_weighted(cls) -> CompositeScoringConfig:
        """Gradient-boosted tree composite scoring.

        Captures non-linear factor interactions (e.g. high value +
        improving momentum = stronger combined signal).
        """
        return cls(method=CompositeMethod.GBT_WEIGHTED)

    @classmethod
    def for_ic_weighted_robust(cls) -> CompositeScoringConfig:
        """IC-weighted scoring with minimum coverage of 3 groups."""
        return cls(method=CompositeMethod.IC_WEIGHTED, min_coverage_groups=3)

    @classmethod
    def for_sparse_universe(cls) -> CompositeScoringConfig:
        """Equal-weight scoring with minimum coverage of 2 groups."""
        return cls(min_coverage_groups=2)

    @classmethod
    def for_coverage_diagnostics(cls) -> CompositeScoringConfig:
        """Equal-weight scoring returning coverage_ratio alongside composite."""
        return cls(return_coverage=True)


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
        Maximum deviation from parent universe sector weights (fraction,
        0–1).  Default 0.05 (5 pp) matches MSCI, S&P DJI, and FTSE Russell
        factor-index methodology.  Use ``for_low_tracking_error()`` for a
        tighter 3% band suited to institutional low-active-risk mandates.
    """

    method: SelectionMethod = SelectionMethod.FIXED_COUNT
    target_count: int = 100
    target_quantile: float = 0.8
    exit_quantile: float = 0.7
    buffer_fraction: float = 0.1
    sector_balance: bool = True
    sector_tolerance: float = 0.05

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
    def for_top_20(cls) -> SelectionConfig:
        """Select top 20 stocks — concentrated diversified portfolio.

        Uses relaxed sector tolerance (10%) because at 20 stocks each
        addition/removal changes sector weight by ~5%.  Buffer of 3 stocks
        (15%) reduces unnecessary turnover.
        """
        return cls(
            target_count=20,
            buffer_fraction=0.15,
            sector_balance=True,
            sector_tolerance=0.10,
        )

    @classmethod
    def for_concentrated(cls) -> SelectionConfig:
        """Concentrated portfolio of top 30 stocks."""
        return cls(target_count=30, buffer_fraction=0.15)

    @classmethod
    def for_low_tracking_error(cls) -> SelectionConfig:
        """Top 100 stocks with tighter sector tolerance for low tracking error.

        Uses a 3% sector deviation cap (vs. the standard 5%) to more closely
        replicate the sector composition of the parent benchmark, matching
        the tighter band used by institutional index providers (e.g., MSCI
        Minimum Volatility) when minimising active sector bets is a mandate.
        """
        return cls(sector_tolerance=0.03)


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
    unknown_tilts : tuple[tuple[str, float], ...]
        Group tilts when regime is unknown (neutral — all multipliers
        default to 1.0 via empty tuple).
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
    unknown_tilts: tuple[tuple[str, float], ...] = ()

    @classmethod
    def for_moderate_tilts(cls) -> RegimeTiltConfig:
        """Enable moderate regime-conditional tilts."""
        return cls(enable=True)

    @classmethod
    def for_no_tilts(cls) -> RegimeTiltConfig:
        """Disable regime tilts (default)."""
        return cls(enable=False)


@dataclass(frozen=True)
class RegimeThresholdConfig:
    """Classification thresholds for the composite macro regime scorer.

    All eight thresholds drive the {-1, 0, +1} component scores used by
    :func:`~optimizer.factors._regime.classify_regime_composite` and the
    research-layer scoring functions in ``research/_macro.py``.

    Parameters
    ----------
    hy_oas_risk_on : float
        HY OAS level (bps) below which credit conditions are benign (+1).
        Empirical basis: ~40th pctl of ICE BofA HY OAS 1997-2023.
    hy_oas_risk_off : float
        HY OAS level (bps) above which credit stress is elevated (-1).
        Empirical basis: ~75th pctl of ICE BofA HY OAS historically.
    pmi_expansion : float
        ISM Manufacturing PMI above which growth is accelerating (+1).
        2-point buffer above the 50 neutral line (Koenig 2002).
    pmi_contraction : float
        ISM Manufacturing PMI below which growth is contracting (-1).
        Symmetric 2-point band around 50.
    spread_2s10s_steep : float
        10Y-2Y spread (percentage points) above which the curve is steep (+1).
        100 bps historically associated with early-cycle acceleration.
    spread_2s10s_inversion : float
        10Y-2Y spread (percentage points) at/below which the curve is inverted (-1).
        Conventional inversion definition (Estrella & Mishkin 1998).
    sentiment_positive : float
        Normalized NLP sentiment score above which sentiment is positive (+1).
    sentiment_negative : float
        Normalized NLP sentiment score below which sentiment is negative (-1).
    """

    hy_oas_risk_on: float = 350.0
    hy_oas_risk_off: float = 500.0
    pmi_expansion: float = 52.0
    pmi_contraction: float = 48.0
    spread_2s10s_steep: float = 1.0
    spread_2s10s_inversion: float = 0.0
    sentiment_positive: float = 0.3
    sentiment_negative: float = -0.3

    @classmethod
    def for_empirical(cls) -> RegimeThresholdConfig:
        """Canonical empirical thresholds (Chapter 7 calibration)."""
        return cls()

    @classmethod
    def for_rolling_percentile(
        cls,
        hy_series: object | None = None,
        spread_series: object | None = None,
        pmi_series: object | None = None,
        sentiment_series: object | None = None,
        hy_risk_on_pct: float = 0.40,
        hy_risk_off_pct: float = 0.75,
        pmi_expansion_pct: float = 0.60,
        pmi_contraction_pct: float = 0.40,
        spread_steep_pct: float = 0.65,
        sentiment_positive_pct: float = 0.70,
    ) -> RegimeThresholdConfig:
        """Compute thresholds from trailing empirical distributions.

        Pass historical Series for each indicator; thresholds are set at
        the specified percentiles.  Any ``None`` series falls back to the
        hard-coded empirical default for that indicator.
        """
        import pandas as pd

        defaults = cls()

        def _pct(series: object | None, q: float, fallback: float) -> float:
            if series is None:
                return fallback
            s = pd.Series(series).dropna()
            return float(s.quantile(q)) if len(s) > 0 else fallback

        return cls(
            hy_oas_risk_on=_pct(hy_series, hy_risk_on_pct, defaults.hy_oas_risk_on),
            hy_oas_risk_off=_pct(hy_series, hy_risk_off_pct, defaults.hy_oas_risk_off),
            pmi_expansion=_pct(pmi_series, pmi_expansion_pct, defaults.pmi_expansion),
            pmi_contraction=_pct(
                pmi_series, pmi_contraction_pct, defaults.pmi_contraction
            ),
            spread_2s10s_steep=_pct(
                spread_series, spread_steep_pct, defaults.spread_2s10s_steep
            ),
            spread_2s10s_inversion=_pct(
                spread_series, 1.0 - spread_steep_pct, defaults.spread_2s10s_inversion
            ),
            sentiment_positive=_pct(
                sentiment_series,
                sentiment_positive_pct,
                defaults.sentiment_positive,
            ),
            sentiment_negative=_pct(
                sentiment_series,
                1.0 - sentiment_positive_pct,
                defaults.sentiment_negative,
            ),
        )


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
    score_premium : float
        Annualized premium per unit of composite z-score.
    use_black_litterman : bool
        Whether to generate Black-Litterman views from factor scores.
    view_confidence_cap : float
        Maximum Idzorek confidence for BL views (0–1).  At 1.0 the
        posterior equals the view exactly, causing extreme concentration.
        Values 0.25–0.50 blend the view with the equilibrium prior.
    max_weight : float
        Maximum per-asset weight enforced on the optimizer when the
        integration injects a BL prior.  0.0 disables the constraint.
    exposure_lower_bound : float
        Lower bound for factor exposure constraints.
    exposure_upper_bound : float
        Upper bound for factor exposure constraints.
    """

    risk_free_rate: float = 0.04
    market_risk_premium: float = 0.05
    score_premium: float = 0.02
    use_black_litterman: bool = False
    view_confidence_cap: float = 0.50
    max_weight: float = 0.10
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


# ---------------------------------------------------------------------------
# Result containers (mutable dataclasses)
# ---------------------------------------------------------------------------


@dataclass
class FactorBuildHealth:
    """Diagnostic report from build_factor_scores_history().

    Parameters
    ----------
    total_dates : int
        Number of rebalancing dates attempted.
    succeeded_dates : int
        Number of dates for which factor computation succeeded.
    failed_dates : int
        Number of dates skipped due to errors.
    failures : dict[str, str]
        Mapping of ISO-date string to exception message for each failure.
    min_success_fraction : float
        Minimum fraction of succeeded/total required before
        FactorCoverageError is raised.
    """

    total_dates: int
    succeeded_dates: int
    failed_dates: int
    failures: dict[str, str]
    min_success_fraction: float

    @property
    def success_fraction(self) -> float:
        """Fraction of dates that succeeded (1.0 if total_dates == 0)."""
        if self.total_dates == 0:
            return 1.0
        return self.succeeded_dates / self.total_dates

    @property
    def is_healthy(self) -> bool:
        """True when success_fraction >= min_success_fraction."""
        return self.success_fraction >= self.min_success_fraction
