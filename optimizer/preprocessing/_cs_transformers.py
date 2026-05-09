"""Cross-sectional preprocessing transformers (skfolio 0.20+).

These operate ACROSS ASSETS PER PERIOD (axis=1), in contrast to the
time-series transformers DataValidator, OutlierTreater, SectorImputer,
and RegressionImputer (which operate axis=0). Use them as feature
preprocessors for CSLinearRegression or factor signals. They preserve
(T, N) shape and skip NaN per row.
"""

from __future__ import annotations

from skfolio.preprocessing import (
    CSGaussianRankScaler,
    CSPercentileRankScaler,
    CSStandardScaler,
    CSTanhShrinker,
    CSWinsorizer,
)

__all__ = [
    "CSGaussianRankScaler",
    "CSPercentileRankScaler",
    "CSStandardScaler",
    "CSTanhShrinker",
    "CSWinsorizer",
]
