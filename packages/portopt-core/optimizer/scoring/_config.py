"""Configuration for performance scoring functions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from optimizer.optimization._config import RatioMeasureType, RiskMeasureType


class PerfMeasureType(str, Enum):
    """Performance (return) measure selection for scoring.

    Maps to :class:`skfolio.measures.PerfMeasure`.  Higher is better, so
    scorers built from a performance measure use ``greater_is_better=True``.
    """

    MEAN = "mean"
    ANNUALIZED_MEAN = "annualized_mean"


@dataclass(frozen=True)
class ScorerConfig:
    """Immutable configuration for building a scoring function.

    Exactly one *measure family* is active at a time:

    * ``ratio_measure`` set (default): score by a ratio measure such as the
      Sharpe, Sortino or Calmar ratio (higher is better).
    * ``perf_measure`` set: score by a performance measure such as the mean
      or annualized mean return (higher is better).  Requires
      ``ratio_measure=None``.
    * ``risk_measure`` set: score by a risk measure such as variance, CVaR
      or maximum drawdown (lower is better -- the scorer sign-flips so that
      higher is always better for model selection).  Requires
      ``ratio_measure=None``.
    * all three ``None``: a custom callable ``score_func`` must be passed to
      :func:`~optimizer.scoring.build_scorer`.

    Parameters
    ----------
    ratio_measure : RatioMeasureType or None
        Built-in ratio measure.  ``None`` indicates that another measure
        family or a custom scorer is used.
    perf_measure : PerfMeasureType or None
        Performance measure.  Requires ``ratio_measure=None``.
    risk_measure : RiskMeasureType or None
        Risk measure (minimised).  Requires ``ratio_measure=None``.
    greater_is_better : bool or None
        Whether higher scores are better.  ``None`` auto-detects
        from the measure family (ratio/perf -> True, risk -> False).
    risk_free_rate : float
        Per-period risk-free rate applied to the predicted portfolio before
        the (ratio) measure is read.  Defaults to 0.0.  Only affects ratio
        measures whose definition depends on the risk-free rate (Sharpe,
        Sortino, ...); ignored for perf/risk measures.
    annualization_factor : float or None
        Number of periods per year applied to the predicted portfolio before
        the measure is read (affects annualized measures) and used to
        annualise the custom Information Ratio scorer.  ``None`` keeps the
        skfolio default (252 trading days).
    """

    ratio_measure: RatioMeasureType | None = RatioMeasureType.SHARPE_RATIO
    perf_measure: PerfMeasureType | None = None
    risk_measure: RiskMeasureType | None = None
    greater_is_better: bool | None = None
    risk_free_rate: float = 0.0
    annualization_factor: float | None = None

    def __post_init__(self) -> None:
        active = [
            m
            for m in (self.ratio_measure, self.perf_measure, self.risk_measure)
            if m is not None
        ]
        if len(active) > 1:
            msg = (
                "at most one of ratio_measure / perf_measure / risk_measure "
                "may be set; pass ratio_measure=None to use perf_measure or "
                f"risk_measure (got ratio_measure={self.ratio_measure!r}, "
                f"perf_measure={self.perf_measure!r}, "
                f"risk_measure={self.risk_measure!r})"
            )
            raise ValueError(msg)
        if self.annualization_factor is not None and self.annualization_factor <= 0:
            msg = (
                "annualization_factor must be strictly positive, got "
                f"{self.annualization_factor!r}"
            )
            raise ValueError(msg)

    @classmethod
    def for_sharpe(cls) -> ScorerConfig:
        """Sharpe ratio scorer."""
        return cls(ratio_measure=RatioMeasureType.SHARPE_RATIO)

    @classmethod
    def for_sortino(cls) -> ScorerConfig:
        """Sortino ratio scorer."""
        return cls(ratio_measure=RatioMeasureType.SORTINO_RATIO)

    @classmethod
    def for_calmar(cls) -> ScorerConfig:
        """Calmar ratio scorer."""
        return cls(ratio_measure=RatioMeasureType.CALMAR_RATIO)

    @classmethod
    def for_cvar_ratio(cls) -> ScorerConfig:
        """CVaR ratio scorer."""
        return cls(ratio_measure=RatioMeasureType.CVAR_RATIO)

    @classmethod
    def for_information_ratio(cls) -> ScorerConfig:
        """Information Ratio scorer (active return / tracking error).

        Requires ``benchmark_returns`` to be passed to
        :func:`~optimizer.scoring.build_scorer`.
        """
        return cls(ratio_measure=RatioMeasureType.INFORMATION_RATIO)

    @classmethod
    def for_sharpe_with_rf(cls, rf_daily: float) -> ScorerConfig:
        """Sharpe ratio scorer with explicit daily risk-free rate."""
        return cls(
            ratio_measure=RatioMeasureType.SHARPE_RATIO,
            risk_free_rate=rf_daily,
        )

    @classmethod
    def for_perf_measure(
        cls, perf_measure: PerfMeasureType = PerfMeasureType.MEAN
    ) -> ScorerConfig:
        """Performance-measure scorer (higher is better)."""
        return cls(ratio_measure=None, perf_measure=perf_measure)

    @classmethod
    def for_risk_measure(
        cls, risk_measure: RiskMeasureType = RiskMeasureType.VARIANCE
    ) -> ScorerConfig:
        """Risk-measure scorer (lower risk -> higher score)."""
        return cls(ratio_measure=None, risk_measure=risk_measure)

    @classmethod
    def for_custom(cls) -> ScorerConfig:
        """Custom scoring function (callable passed to factory)."""
        return cls(ratio_measure=None)
