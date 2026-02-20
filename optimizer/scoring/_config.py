"""Configuration for performance scoring functions."""

from __future__ import annotations

from dataclasses import dataclass

from optimizer.optimization._config import RatioMeasureType


@dataclass(frozen=True)
class ScorerConfig:
    """Immutable configuration for building a scoring function.

    When ``ratio_measure`` is set, the scorer evaluates portfolios
    using the corresponding built-in ratio measure (Sharpe, Sortino,
    Calmar, etc.).  When ``ratio_measure`` is ``None``, a custom
    callable must be passed to the factory function.

    Parameters
    ----------
    ratio_measure : RatioMeasureType or None
        Built-in ratio measure.  ``None`` indicates a custom scorer.
    greater_is_better : bool or None
        Whether higher scores are better.  ``None`` auto-detects
        from the ratio measure.
    """

    ratio_measure: RatioMeasureType | None = RatioMeasureType.SHARPE_RATIO
    greater_is_better: bool | None = None

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
    def for_custom(cls) -> ScorerConfig:
        """Custom scoring function (callable passed to factory)."""
        return cls(ratio_measure=None)
