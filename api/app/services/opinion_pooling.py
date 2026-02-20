"""Multi-LLM Opinion Pooling service.

Architecture:
  1. Run N LLM expert prompts (value, momentum, macro) — each produces independent views.
  2. Convert each expert's views into a skfolio BlackLitterman prior estimator.
  3. Compute IC-calibrated credibility weights from historical IC series.
  4. Pass experts + weights to ``build_opinion_pooling()`` from the optimizer layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from skfolio.prior import BlackLitterman
from skfolio.prior._base import BasePrior

from app.services.view_generation import _validate_idzorek_alphas, _views_to_arrays
from baml_client import b
from baml_client.types import AssetFactorData, ExpertPersona, ViewOutput
from optimizer.views._config import BlackLittermanConfig
from optimizer.views._factory import build_opinion_pooling
from optimizer.views._config import OpinionPoolingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expert persona registry
# ---------------------------------------------------------------------------

# Maps persona → (name, BAML callable)
_PERSONA_CALLABLES: dict[ExpertPersona, tuple[str, Callable[[list[AssetFactorData]], ViewOutput]]] = {
    ExpertPersona.VALUE_INVESTOR: (
        "value_investor",
        lambda assets: b.GenerateValueView(assets=assets),
    ),
    ExpertPersona.MOMENTUM_TRADER: (
        "momentum_trader",
        lambda assets: b.GenerateMomentumView(assets=assets),
    ),
    ExpertPersona.MACRO_ANALYST: (
        "macro_analyst",
        lambda assets: b.GenerateMacroView(assets=assets),
    ),
}

# Default set of personas when caller requests all experts
ALL_PERSONAS: list[ExpertPersona] = list(_PERSONA_CALLABLES)


# ---------------------------------------------------------------------------
# IC weight computation
# ---------------------------------------------------------------------------


def compute_ic_weights(
    ic_histories: list[pd.Series],
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute IC-calibrated credibility weights, normalised to sum to 1.

    Weight for expert *i* is proportional to ``max(ICIR_i, 0)``, where
    ICIR = mean(IC) / std(IC).  Experts with zero or negative ICIR receive a
    near-zero (not hard-zero) weight set to ``eps`` before normalisation, so
    they remain in the pool with negligible influence.

    Args:
        ic_histories: One ``pd.Series`` of IC values per expert.  Each series
            should have at least 3 observations for a meaningful ICIR estimate.
            Pass an empty series for an expert with no history.
        eps: Floor applied before normalisation (prevents exact-zero weights).

    Returns:
        1-D ``np.ndarray`` of length ``len(ic_histories)``, summing to 1.0.
    """
    raw_weights: list[float] = []
    for ic_series in ic_histories:
        if len(ic_series) < 3:
            raw_weights.append(eps)
            continue

        mean_ic = float(ic_series.mean())
        std_ic = float(ic_series.std(ddof=1))
        icir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
        raw_weights.append(max(icir, 0.0) + eps)

    arr = np.array(raw_weights, dtype=np.float64)
    total = arr.sum()
    if total <= 0.0:
        return np.full(len(raw_weights), 1.0 / len(raw_weights))
    return arr / total


# ---------------------------------------------------------------------------
# Per-expert result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExpertViewResult:
    """Structured output for a single LLM expert."""

    persona: ExpertPersona
    name: str
    view_output: ViewOutput
    view_strings: list[str]
    idzorek_alphas: dict[str, float]
    prior_estimator: BlackLitterman  # ready for Opinion Pooling


@dataclass
class OpinionPoolResult:
    """Full output of the multi-LLM opinion pooling pipeline."""

    expert_results: list[ExpertViewResult]
    ic_weights: np.ndarray            # shape (n_experts,), sums to 1
    opinion_pool: object   # fitted-ready skfolio.prior.OpinionPooling
    tickers: list[str]


# ---------------------------------------------------------------------------
# Single-expert runner
# ---------------------------------------------------------------------------


def _run_expert(
    persona: ExpertPersona,
    assets: list[AssetFactorData],
    tickers: list[str],
    tau: float = 0.05,
) -> ExpertViewResult:
    """Call the BAML function for one expert persona and build a BL prior."""
    name, baml_fn = _PERSONA_CALLABLES[persona]
    raw: ViewOutput = baml_fn(assets)

    # Filter to known tickers only
    valid_views = [v for v in raw.views if v.asset in set(tickers)]

    if not valid_views:
        # Expert produced no actionable views — return a no-view BL prior
        view_strings: list[str] = []
        idzorek_alphas: dict[str, float] = {}
        prior = BlackLitterman(views=[], tau=tau)
    else:
        view_strings_raw, _, _, confidences = _views_to_arrays(valid_views, tickers)
        view_assets = [v.asset for v in valid_views]
        idzorek_alphas = _validate_idzorek_alphas(raw.idzorek_alphas, view_assets)
        view_strings = view_strings_raw
        prior = BlackLitterman(
            views=view_strings,
            tau=tau,
            view_confidences=list(confidences),
        )

    return ExpertViewResult(
        persona=persona,
        name=name,
        view_output=raw,
        view_strings=view_strings,
        idzorek_alphas=idzorek_alphas,
        prior_estimator=prior,
    )


# ---------------------------------------------------------------------------
# Main service entry points
# ---------------------------------------------------------------------------


def run_llm_experts(
    assets: list[AssetFactorData],
    tickers: list[str],
    personas: list[ExpertPersona] | None = None,
    tau: float = 0.05,
) -> list[ExpertViewResult]:
    """Run each LLM expert persona and return their view results.

    Args:
        assets: Per-asset factor data (from ``fetch_factor_data``).
        tickers: Ordered universe tickers.
        personas: Personas to run.  Defaults to all three (value, momentum, macro).
        tau: BL uncertainty scaling applied to each expert's prior.

    Returns:
        List of :class:`ExpertViewResult` — one per persona, in input order.
    """
    if personas is None:
        personas = ALL_PERSONAS

    results: list[ExpertViewResult] = []
    for persona in personas:
        try:
            result = _run_expert(persona, assets, tickers, tau=tau)
            results.append(result)
        except Exception as exc:
            logger.error("Expert %s failed: %s", persona.value, exc)
            raise

    return results


def build_llm_opinion_pool(
    assets: list[AssetFactorData],
    tickers: list[str],
    ic_histories: list[pd.Series] | None = None,
    personas: list[ExpertPersona] | None = None,
    tau: float = 0.05,
    is_linear_pooling: bool = True,
    divergence_penalty: float = 0.0,
) -> OpinionPoolResult:
    """Run all LLM experts and combine via IC-calibrated Opinion Pooling.

    Args:
        assets: Per-asset factor data.
        tickers: Ordered universe tickers.
        ic_histories: One ``pd.Series`` per expert giving their historical IC.
            If ``None``, equal weights are used.
        personas: Subset of personas to run.  Defaults to all three.
        tau: BL uncertainty scaling for each expert's prior.
        is_linear_pooling: True = arithmetic pooling; False = geometric pooling.
        divergence_penalty: KL penalty for robust pooling.

    Returns:
        :class:`OpinionPoolResult` with experts, IC weights, and fitted-ready
        ``OpinionPooling`` estimator.
    """
    if personas is None:
        personas = ALL_PERSONAS

    expert_results = run_llm_experts(assets, tickers, personas=personas, tau=tau)

    n_experts = len(expert_results)

    # IC weights
    if ic_histories is None:
        ic_weights = np.full(n_experts, 1.0 / n_experts)
    else:
        if len(ic_histories) != n_experts:
            raise ValueError(
                f"ic_histories length ({len(ic_histories)}) must match "
                f"number of experts ({n_experts})."
            )
        ic_weights = compute_ic_weights(ic_histories)

    # Build estimators list for skfolio
    estimators: list[tuple[str, BasePrior]] = [
        (er.name, er.prior_estimator) for er in expert_results
    ]

    config = OpinionPoolingConfig(
        opinion_probabilities=tuple(float(w) for w in ic_weights),
        is_linear_pooling=is_linear_pooling,
        divergence_penalty=divergence_penalty,
    )
    opinion_pool = build_opinion_pooling(estimators=estimators, config=config)

    return OpinionPoolResult(
        expert_results=expert_results,
        ic_weights=ic_weights,
        opinion_pool=opinion_pool,
        tickers=tickers,
    )


