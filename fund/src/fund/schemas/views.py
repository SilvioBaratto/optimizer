"""Black-Litterman view schema (Task 4) → optimizer BL config.

``View`` is a single analyst opinion: an absolute expected-return view on one
asset/factor (``AAPL == 0.012300``) or a relative one (``AAPL - MSFT ==
0.020000``), carrying an Idzorek confidence in ``[0, 1]``. ``ViewSet`` bundles the
views plus an optional prior selector. Both are pure, serialisable, frozen
(hashable) pydantic-v2 data — constructing one imports **no** ``optimizer`` code.

``to_black_litterman_config`` is the bridge: it renders each view to a skfolio
view-string, aligns the per-view confidences (Idzorek), maps the fund-side prior
selector onto a ``MomentEstimationConfig`` (always ``EquilibriumMu`` — the
BL-standard prior — paired with the chosen covariance flavour), and returns an
``optimizer.views.BlackLittermanConfig``. The optimizer is imported **lazily
inside the method** so schema construction stays cheap and optimizer-free. The
``_COV_ESTIMATOR_MAP`` holds plain enum-value strings (not optimizer enum
objects) for the same reason and must be total (every ``MomentsEstimator`` member
maps; an unmapped member is a hard ``KeyError``, never a silent default).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from fund.schemas.enums import MomentsEstimator

if TYPE_CHECKING:  # import only for the annotation; runtime import is method-local
    from optimizer.views import BlackLittermanConfig

__all__ = ["View", "ViewKind", "ViewSet"]

# Fixed-point decimals used when rendering a view's expected return into a
# skfolio view-string. Fixed-point (never scientific notation) keeps skfolio's
# string parser happy — mirrors optimizer.views._builder's precision convention.
_RETURN_PRECISION = 6


class ViewKind(str, Enum):
    """Whether a view is expressed on an asset (ticker) or a factor name.

    Absolute/relative factor views reference factor names when the BL prior is
    wrapped in a ``TimeSeriesFactorModel`` (skfolio 1.0); asset views reference
    tickers. Metadata only — it does not change the rendered view-string.
    """

    ASSET = "asset"
    FACTOR = "factor"


class View(BaseModel):
    """A single Black-Litterman view — absolute or relative, with confidence.

    ``relative_to is None`` renders an absolute view (``"<target> == <r>"``);
    otherwise a relative view (``"<target> - <relative_to> == <r>"``).
    ``expected_return`` is a per-period figure (matching the returns passed to
    ``fit``) and may be negative for a bearish view.
    """

    model_config = ConfigDict(frozen=True)

    kind: ViewKind = ViewKind.ASSET  # asset ticker vs factor name
    target: str  # the asset ticker / factor name the view is expressed on
    expected_return: float  # per-period expected return (the view RHS)
    relative_to: str | None = None  # None → absolute; else a relative view
    confidence: float = Field(ge=0.0, le=1.0)  # Idzorek confidence in [0, 1]

    def to_view_string(self) -> str:
        """Render to a skfolio view-string (fixed-point, no sci-notation)."""
        rhs = f"{self.expected_return:.{_RETURN_PRECISION}f}"
        if self.relative_to is None:
            return f"{self.target} == {rhs}"  # absolute view
        return f"{self.target} - {self.relative_to} == {rhs}"  # relative view


class ViewSet(BaseModel):
    """A bundle of Black-Litterman views + an optional prior selector.

    ``moments_estimator`` selects the covariance flavour of the inner prior;
    ``None`` uses the BL-standard ``EquilibriumMu`` + ``LedoitWolf`` prior.
    """

    model_config = ConfigDict(frozen=True)

    views: tuple[View, ...]
    moments_estimator: MomentsEstimator | None = None  # None → BL-standard prior

    def to_black_litterman_config(self) -> BlackLittermanConfig:
        """Map onto ``BlackLittermanConfig``. fund is the bridge → the import is OK."""
        from optimizer.moments import (
            CovEstimatorType,
            MomentEstimationConfig,
            MuEstimatorType,
        )
        from optimizer.views import BlackLittermanConfig, ViewUncertaintyMethod

        if self.moments_estimator is None:
            prior = MomentEstimationConfig.for_equilibrium_ledoitwolf()
        else:
            prior = MomentEstimationConfig(
                mu_estimator=MuEstimatorType.EQUILIBRIUM,  # BL needs equilibrium mu
                cov_estimator=CovEstimatorType(
                    _COV_ESTIMATOR_MAP[self.moments_estimator]
                ),
            )
        rendered = tuple(v.to_view_string() for v in self.views)
        confidences = tuple(v.confidence for v in self.views)
        return BlackLittermanConfig(
            views=rendered,
            uncertainty_method=ViewUncertaintyMethod.IDZOREK,
            view_confidences=confidences,  # aligned per-view (Idzorek)
            prior_config=prior,
        )


# ---------------------------------------------------------------------------
# fund MomentsEstimator → optimizer CovEstimatorType value (total; plain strings
# keep construction optimizer-free). BL always pairs these with EquilibriumMu, so
# only the covariance flavour varies here.
# ---------------------------------------------------------------------------
_COV_ESTIMATOR_MAP: dict[MomentsEstimator, str] = {
    MomentsEstimator.LEDOIT_WOLF: "ledoit_wolf",
    MomentsEstimator.EMPIRICAL: "empirical",
    MomentsEstimator.EW: "ew",
}
