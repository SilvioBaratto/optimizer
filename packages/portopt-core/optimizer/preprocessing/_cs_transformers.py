"""Cross-sectional preprocessing transformers (skfolio 0.20+ / 1.0).

These operate ACROSS ASSETS PER PERIOD (axis=1), in contrast to the
time-series transformers DataValidator, OutlierTreater, SectorImputer,
and RegressionImputer (which operate axis=0). Use them as feature
preprocessors for CSLinearRegression or factor signals. They preserve
(T, N) shape and skip NaN per row.

In addition to the raw skfolio re-exports this module ships the project's
frozen-config + factory convention (:class:`CSTransformerConfig` +
:func:`make_cs_transformer`) so a cross-sectional transformer can be selected
and configured from serialisable primitives — grid-searchable and round-trippable
like every other optimizer sub-module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from skfolio.preprocessing import (
    BaseCSTransformer,
    CSGaussianRankScaler,
    CSPercentileRankScaler,
    CSStandardScaler,
    CSTanhShrinker,
    CSWinsorizer,
)

__all__ = [
    "BaseCSTransformer",
    "CSGaussianRankScaler",
    "CSPercentileRankScaler",
    "CSStandardScaler",
    "CSTanhShrinker",
    "CSTransformerConfig",
    "CSTransformerType",
    "CSWinsorizer",
    "make_cs_transformer",
]


class CSTransformerType(str, Enum):
    """Cross-sectional transformer selection."""

    STANDARD = "standard"
    WINSORIZER = "winsorizer"
    GAUSSIAN_RANK = "gaussian_rank"
    PERCENTILE_RANK = "percentile_rank"
    TANH_SHRINKER = "tanh_shrinker"


@dataclass(frozen=True)
class CSTransformerConfig:
    """Frozen, serialisable config selecting one cross-sectional transformer.

    Only the fields relevant to the chosen ``transformer`` are consumed by
    :func:`make_cs_transformer`; the rest are ignored, so a single config can
    be reused across a grid of transformer types.

    Parameters
    ----------
    transformer : CSTransformerType, default=STANDARD
        Which cross-sectional transformer to build.
    min_group_size : int, default=8
        Minimum ``cs_groups`` size before falling back to the global
        cross-section.  Used by STANDARD, GAUSSIAN_RANK, PERCENTILE_RANK.
        Lower it for narrow universes (< ~30 assets).
    atol : float, default=1e-12
        Absolute tolerance for near-constant rows.  Used by STANDARD,
        GAUSSIAN_RANK, TANH_SHRINKER.
    scale : bool, default=True
        GAUSSIAN_RANK only — rescale to unit std after the Gaussianising
        transform.  Set ``False`` when a downstream model standardises.
    low : float, default=0.01
        WINSORIZER only — lower clip quantile (``0 <= low < high <= 1``).
    high : float, default=0.99
        WINSORIZER only — upper clip quantile.
    knee : float, default=3.0
        TANH_SHRINKER only — knee in robust-scale units; smaller compresses
        more aggressively.
    """

    transformer: CSTransformerType = CSTransformerType.STANDARD
    min_group_size: int = 8
    atol: float = 1e-12
    scale: bool = True
    low: float = 0.01
    high: float = 0.99
    knee: float = 3.0


def make_cs_transformer(config: CSTransformerConfig | None = None) -> BaseCSTransformer:
    """Build a skfolio cross-sectional transformer from a serialisable config.

    Parameters
    ----------
    config : CSTransformerConfig or None, default=None
        Selection + hyper-parameters.  ``None`` builds a default
        :class:`CSStandardScaler`.

    Returns
    -------
    BaseCSTransformer
        A fitted-ready cross-sectional transformer.
    """
    cfg = config or CSTransformerConfig()

    if cfg.transformer == CSTransformerType.STANDARD:
        return CSStandardScaler(min_group_size=cfg.min_group_size, atol=cfg.atol)
    if cfg.transformer == CSTransformerType.WINSORIZER:
        return CSWinsorizer(low=cfg.low, high=cfg.high)
    if cfg.transformer == CSTransformerType.GAUSSIAN_RANK:
        return CSGaussianRankScaler(
            min_group_size=cfg.min_group_size, scale=cfg.scale, atol=cfg.atol
        )
    if cfg.transformer == CSTransformerType.PERCENTILE_RANK:
        return CSPercentileRankScaler(min_group_size=cfg.min_group_size)
    if cfg.transformer == CSTransformerType.TANH_SHRINKER:
        return CSTanhShrinker(knee=cfg.knee, atol=cfg.atol)

    raise ValueError(f"Unknown cross-sectional transformer: {cfg.transformer!r}")
