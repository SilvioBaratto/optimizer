"""Foundational primitives for ``pipeline_builder`` step handlers (issue #696).

This module deliberately holds only the small, dependency-free helpers
that every step handler will need:

* a module-level ``sys.path`` shim so that ``research.*`` is importable
  from inside the ``api`` package without a circular dependency on
  ``research.cli._main`` (which imports app models);
* :func:`_to_json` — a JSON-safe serializer for the mixed
  pandas/numpy/datetime payloads the optimization pipeline emits;
* :func:`_require` — a precondition validator that raises a
  :class:`ValueError` the moment a step is invoked before its inputs
  were populated;
* :func:`_load_market_proxy_from_db` — the DB helper used by the
  ``build_history`` step. Its body is copied verbatim from
  ``research/cli/_main._load_market_proxy`` to avoid a circular import
  back into the research CLI.

Step handlers proper land in follow-up issues (#697-#710); each will
import its private helpers from this module.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import tempfile
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.database import database_manager
from app.services._shared._json_safe import safe_float
from app.services.pipeline_builder.session_store import (
    _PipelineSession,
    update_session,
)

__all__ = [
    "_load_market_proxy_from_db",
    "_require",
    "_to_json",
    "step_build_history",
    "step_clean_returns",
    "step_cost",
    "step_coverage_gate",
    "step_load",
    "step_optimize",
    "step_persist",
    "step_rebalance_decision",
    "step_regime",
    "step_report",
    "step_screen",
    "step_slice",
    "step_validate_is",
    "step_validate_oos",
]

# Authoritative n_selected band for the wizard. Cycle 2 deliberately
# narrows this from research/pipeline/_factors.py (which still carries a
# stale [25, 50] band).
_N_SELECTED_MIN: int = 15
_N_SELECTED_MAX: int = 30

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# sys.path shim — make ``research.*`` importable from the api package.
# ---------------------------------------------------------------------------
# ``api/app/services/pipeline_builder/steps.py`` -> repo root is
# ``parents[4]``. ``parents[5]`` would resolve one level above the repo,
# so we deliberately use ``parents[4]`` despite the issue body listing
# ``parents[5]`` (recorded in the follow-up issue comment).
_repo_root = Path(__file__).resolve().parents[4]
if not (_repo_root / "research" / "__init__.py").exists():
    raise ImportError(
        f"pipeline_builder.steps: research package not found at {_repo_root}"
    )
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


# Import after the shim is installed so research.* is reachable. Bound at
# module level so tests can patch ``steps.<symbol>`` directly.
from optimizer.factors._config import SectorRegimeBandsConfig
from optimizer.factors._sector_bands import resolve_sector_bands
from optimizer.universe._config import InvestabilityScreenConfig
from optimizer.universe._factory import screen_universe
from research.optimization._rebalance import _decide_rebalance
from research.pipeline._factors import (
    _check_factor_coverage,
    _missing_gics_sectors,
    build_history,
    validate_is,
    validate_oos,
)
from research.pipeline._load import (
    _materialise_clean_returns,
    _read_last_review_date,
    _write_last_review_date,
    load_data,
)
from research.pipeline._optimize import (
    _DEFAULT_COSTS,
    COUNTRY_COSTS_BPS,
    optimize_portfolio,
)
from research.pipeline._regime import classify_and_tilt
from research.pipeline._report import (
    _persist_research_snapshot,
    _render_research_report,
    report_performance,
)
from research.pipeline._screen import (
    _UNIVERSE_BAND,
    _assert_universe_size,
    screen_investable,
)

_PRESET_FACTORIES = {
    "developed_markets": InvestabilityScreenConfig.for_developed_markets,
    "broad_universe": InvestabilityScreenConfig.for_broad_universe,
    "small_cap": InvestabilityScreenConfig.for_small_cap,
    "large_cap": InvestabilityScreenConfig.for_large_cap,
}


# ---------------------------------------------------------------------------
# JSON-safe serializer
# ---------------------------------------------------------------------------


def _to_json(obj: Any) -> Any:
    """Return ``obj`` recursively coerced to a JSON-serializable structure.

    Conversion table:

    * ``pd.DataFrame`` → list of records (index reset, included as a
      column);
    * ``pd.Series`` → ``{"index": [...], "values": [...]}``;
    * ``np.ndarray`` → ``list``;
    * numpy scalars → native ``int`` / ``float``;
    * ``pd.Timestamp`` / ``datetime`` / ``date`` → ``str``;
    * ``Enum`` → ``.value``;
    * ``dict`` / ``list`` / ``tuple`` → recurse element-wise;
    * leaf ``float`` (and numpy floats) routed through
      :func:`safe_float` so ``NaN`` / ``Inf`` become ``None``;
    * everything else → ``str(obj)``.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, pd.DataFrame):
        return _to_json(obj.reset_index().to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return {
            "index": _to_json(list(obj.index)),
            "values": _to_json(list(obj.values)),
        }
    if isinstance(obj, np.ndarray):
        return _to_json(obj.tolist())
    if isinstance(obj, np.floating):
        return safe_float(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float):
        return safe_float(obj)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {
            (k.value if isinstance(k, Enum) else str(k)): _to_json(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# Precondition validator
# ---------------------------------------------------------------------------


def _require(session: _PipelineSession, *fields: str) -> None:
    """Raise ``ValueError`` for the first ``fields`` entry that is ``None``.

    Step handlers call this at entry to refuse to run before their
    upstream producers have populated their inputs on the session
    snapshot.
    """
    for field in fields:
        if getattr(session, field) is None:
            raise ValueError(f"Step precondition failed: {field} not set")


# ---------------------------------------------------------------------------
# Market-proxy DB helper (copied verbatim from research/cli/_main.py)
# ---------------------------------------------------------------------------


def _load_market_proxy_from_db(
    db_manager: Any,
    tickers: tuple[str, ...] = ("URTH", "SPY"),
) -> pd.DataFrame | None:
    """Return a single-column close-price ``DataFrame`` for the market proxy.

    Tries each ticker in order; returns the first that has data. The DB
    session is closed (via the ``db_manager.get_session()`` context
    manager) before the function returns.
    """
    from sqlalchemy import select

    from app.models.market_data.yfinance_data import PriceHistory
    from app.models.universe.universe import Instrument

    with db_manager.get_session() as session:
        for ticker in tickers:
            stmt = (
                select(PriceHistory.date, PriceHistory.close)
                .join(Instrument)
                .where(Instrument.yfinance_ticker == ticker)
                .where(PriceHistory.close.isnot(None))
                .order_by(PriceHistory.date)
            )
            rows = session.execute(stmt).all()
            if not rows:
                continue
            prices = pd.DataFrame(
                {"close": [float(close) for _, close in rows]},
                index=pd.DatetimeIndex([pd.Timestamp(d) for d, _ in rows]),
            )
            logger.info("Loaded market proxy %s (%d rows).", ticker, len(prices))
            return prices

    logger.warning(
        "No market proxy found for tickers %s — beta will use EW fallback.",
        tickers,
    )
    return None


# ---------------------------------------------------------------------------
# Step 1 — load
# ---------------------------------------------------------------------------


def _validate_n_selected(n_selected: int) -> None:
    """Enforce the wizard's authoritative [15, 30] band for n_selected."""
    if not (_N_SELECTED_MIN <= n_selected <= _N_SELECTED_MAX):
        raise RuntimeError(
            f"n_selected={n_selected} outside [{_N_SELECTED_MIN}, {_N_SELECTED_MAX}]"
        )


def step_load(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 1 — load and assemble all DB data for the wizard.

    Validates ``run_config["n_selected"]`` upfront (default 20), then
    delegates to :func:`research.pipeline._load.load_data` which runs
    the DB preflight, ``assemble_all``, FX conversion and the assembly
    size assertion (``RuntimeError`` on either floor breach).

    Persists ``assembly`` on the session as a typed field; ``country_map``
    is stashed under ``session.run_config["country_map"]`` because
    ``_PipelineSession`` carries no dedicated field for it yet
    (deferred to a later cycle).

    Returns a JSON-safe summary dict for the API response.
    """
    n_selected = session.run_config.get("n_selected", 20)
    _validate_n_selected(n_selected)

    base_currency = session.run_config.get("base_currency", "EUR")

    on_progress(phase="db_preflight")
    on_progress(phase="assembling")
    assembly, country_map, _db_manager = load_data(
        database_manager, base_currency=base_currency
    )
    on_progress(phase="done")

    update_session(
        session_id,
        assembly=assembly,
        run_config={**session.run_config, "country_map": country_map},
    )

    return _to_json(
        {
            "n_tickers": assembly.n_tickers,
            "n_trading_days": assembly.n_trading_days,
            "assembly_hash": assembly.assembly_hash,
            "base_currency": base_currency,
            "price_start": assembly.prices.index[0],
            "price_end": assembly.prices.index[-1],
        }
    )


# ---------------------------------------------------------------------------
# Step 2 — slice
# ---------------------------------------------------------------------------


def _resolve_slice_dates(
    params: dict, run_config: dict
) -> tuple[str | None, str | None]:
    """Return ``(start, end)`` where ``params`` overrides ``run_config``."""
    start = params.get("start_date", run_config.get("start_date"))
    end = params.get("end_date", run_config.get("end_date"))
    return start, end


def _slice_summary(prices: pd.DataFrame, sliced: bool) -> dict:
    """Return the ``step_slice`` payload from a prices frame."""
    return {
        "sliced": sliced,
        "price_start": str(prices.index[0].date()),
        "price_end": str(prices.index[-1].date()),
        "n_trading_days": len(prices),
    }


def step_slice(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 2 — slice ``assembly.prices`` to ``[start_date, end_date]``.

    ``params`` overrides ``run_config`` per-run; either key may be ``None``
    for an open-ended slice. Returns a no-op payload (``sliced=False``) when
    neither date is set. Mirrors ``research/cli/_main.py:150-158``.
    """
    _require(session, "assembly")
    assembly = session.assembly

    start, end = _resolve_slice_dates(params, session.run_config)
    if start is None and end is None:
        return _slice_summary(assembly.prices, sliced=False)

    on_progress(phase="slicing")
    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    sliced_prices = assembly.prices.loc[start_ts:end_ts]
    new_assembly = dataclasses.replace(assembly, prices=sliced_prices)
    update_session(session_id, assembly=new_assembly)
    on_progress(phase="done")

    return _slice_summary(sliced_prices, sliced=True)


# ---------------------------------------------------------------------------
# Step 3 — screen
# ---------------------------------------------------------------------------


def _screen_for_preset(preset: str, assembly: Any) -> pd.Index:
    """Dispatch to the matching screening routine for ``preset``.

    The default ``developed_markets`` path goes through
    ``research.pipeline._screen.screen_investable`` (which hardcodes
    ``InvestabilityScreenConfig.for_developed_markets()``). Other presets
    bypass it and call ``optimizer.universe._factory.screen_universe``
    directly with the matching ``InvestabilityScreenConfig.for_*()``.
    """
    if preset == "developed_markets":
        return screen_investable(assembly)
    config_factory = _PRESET_FACTORIES[preset]
    return screen_universe(
        fundamentals=assembly.fundamentals,
        price_history=assembly.prices,
        volume_history=assembly.volumes,
        financial_statements=assembly.financial_statements,
        config=config_factory(),
    )


def step_screen(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 3 — screen the investable universe per preset.

    Enforces the floor of 200 tickers (``RuntimeError`` from
    ``_assert_universe_size``). Sets ``band_warning=True`` on the payload
    when the resulting count is outside ``[300, 1500]`` without raising.
    Persists the passing ticker index on the session as ``investable``.
    """
    _require(session, "assembly")
    preset = params.get("preset", "developed_markets")
    if preset not in _PRESET_FACTORIES:
        raise ValueError(
            f"unknown preset {preset!r}; choose from {sorted(_PRESET_FACTORIES)}"
        )

    on_progress(phase="screening")
    passing = _screen_for_preset(preset, session.assembly)
    _assert_universe_size(passing)
    on_progress(phase="done")

    n = len(passing)
    low, high = _UNIVERSE_BAND
    update_session(session_id, investable=passing)
    return {
        "n_investable": n,
        "preset": preset,
        "band_warning": not (low <= n <= high),
        "band_low": low,
        "band_high": high,
    }


# ---------------------------------------------------------------------------
# Step 4 — clean_returns
# ---------------------------------------------------------------------------

_PREPROCESSING_STEPS: tuple[str, ...] = (
    "DataValidator",
    "OutlierTreater",
    "SectorImputer",
    "RegressionImputer",
)


def step_clean_returns(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 4 — materialise the cleaned linear-return panel.

    Delegates to ``research.pipeline._load._materialise_clean_returns``
    which slices prices to ``investable``, converts to linear returns,
    runs the fixed preprocessing pipeline
    (``DataValidator → OutlierTreater → SectorImputer →
    RegressionImputer``), and raises ``RuntimeError`` if NaN/inf survives.
    """
    _require(session, "assembly", "investable")

    on_progress(phase="preprocessing")
    clean_df = _materialise_clean_returns(session.assembly, session.investable)
    on_progress(phase="done")

    update_session(session_id, clean_returns=clean_df)
    return {
        "n_days": len(clean_df),
        "n_tickers": clean_df.shape[1],
        "return_start": str(clean_df.index[0].date()),
        "return_end": str(clean_df.index[-1].date()),
        "preprocessing_steps": list(_PREPROCESSING_STEPS),
    }


# ---------------------------------------------------------------------------
# Step 5 — build_history
# ---------------------------------------------------------------------------


_DEFAULT_MARKET_PROXY_TICKERS: tuple[str, ...] = ("URTH", "SPY")


def _resolve_proxy_tickers(params: dict) -> tuple[str, ...]:
    """Return the market-proxy ticker tuple, honouring per-call override."""
    override = params.get("market_proxy_ticker")
    if override is None:
        return _DEFAULT_MARKET_PROXY_TICKERS
    return (override,)


def step_build_history(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 5 — build rolling factor scores + forward-return history.

    Loads the market proxy (URTH → SPY by default; overridable via
    ``params["market_proxy_ticker"]``) inside a short DB session that is
    fully closed before ``build_history`` runs — so no connection is
    held across the multi-minute factor compute (db-pool memory rule).
    """
    _require(session, "assembly", "investable")

    tickers = _resolve_proxy_tickers(params)
    rebalance_freq = session.run_config.get("rebalance_freq", 63)

    on_progress(phase="loading_market_proxy")
    market_prices = _load_market_proxy_from_db(database_manager, tickers=tickers)

    on_progress(phase="building_factor_history")
    factor_scores_dict, returns_history, health = build_history(
        session.assembly,
        session.investable,
        rebalance_freq=rebalance_freq,
        market_prices=market_prices,
    )
    on_progress(phase="done")

    update_session(
        session_id,
        factor_scores_dict=factor_scores_dict,
        returns_history=returns_history,
    )
    return {
        "succeeded_dates": health.succeeded_dates,
        "total_dates": health.total_dates,
        "failed_dates": health.failed_dates,
        "n_factors": len(factor_scores_dict),
        "rebalance_freq": rebalance_freq,
        "market_proxy_loaded": market_prices is not None,
    }


# ---------------------------------------------------------------------------
# Step 6 — validate_is
# ---------------------------------------------------------------------------


_IS_VALIDATION_PAYLOAD_CONFIG: dict[str, float] = {
    "newey_west_lags": 4,
    "fdr_alpha": 0.10,
    "t_stat_threshold": 1.645,
}
_VIF_THRESHOLD: float = 5.0


def _high_vif_factors(vif_scores: pd.Series | None) -> list[str]:
    """Return factor names whose VIF exceeds the threshold."""
    if vif_scores is None or vif_scores.empty:
        return []
    above = vif_scores[vif_scores > _VIF_THRESHOLD]
    return [str(name) for name in above.index]


def _ic_records(ic_results: list) -> list[dict]:
    """Serialize ``ICResult`` dataclasses to JSON-safe dicts."""
    return _to_json(
        [
            {
                "factor_name": r.factor_name,
                "mean_ic": r.mean_ic,
                "ic_std": r.ic_std,
                "t_stat": r.t_stat,
                "p_value": r.p_value,
                "significant": r.significant,
            }
            for r in ic_results
        ]
    )


def step_validate_is(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 6 — in-sample factor validation report.

    Delegates to ``research.pipeline._factors.validate_is`` which uses
    the module-level ``_IS_VALIDATION_CONFIG`` (newey_west_lags=4,
    fdr_alpha=0.10, t_stat_threshold=1.645). Returns a
    ``FactorValidationReport`` exposing ``ic_results`` (list of
    ``ICResult``), ``significant_factors`` (list[str]), and ``vif_scores``
    (Series | None). ``high_vif_factors`` is derived here as the index
    of ``vif_scores > 5.0``.
    """
    _require(session, "factor_scores_dict", "returns_history")

    on_progress(phase="validating_is")
    report = validate_is(session.factor_scores_dict, session.returns_history)
    on_progress(phase="done")

    update_session(session_id, is_result=report)

    significant = list(report.significant_factors)
    return {
        "ic_results": _ic_records(list(report.ic_results)),
        "n_significant": len(significant),
        "significant_factors": significant,
        "high_vif_factors": _high_vif_factors(report.vif_scores),
        "config": dict(_IS_VALIDATION_PAYLOAD_CONFIG),
    }


# ---------------------------------------------------------------------------
# Step 7 — validate_oos
# ---------------------------------------------------------------------------


_OOS_VALIDATION_PAYLOAD_CONFIG: dict[str, int] = {
    "train_periods": 8,
    "val_periods": 4,
    "step_periods": 2,
}


def _oos_records(result: Any) -> list[dict]:
    """Sort factors by OOS mean IC desc and serialize alongside ICIR."""
    icir = result.mean_oos_icir
    sorted_ic = result.mean_oos_ic.sort_values(ascending=False)
    return _to_json(
        [
            {
                "factor_name": str(factor),
                "oos_mean_ic": float(mean_ic),
                "oos_icir": (float(icir[factor]) if factor in icir.index else None),
            }
            for factor, mean_ic in sorted_ic.items()
        ]
    )


def step_validate_oos(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 7 — out-of-sample factor validation (rolling walk-forward).

    Delegates to ``research.pipeline._factors.validate_oos`` which uses
    the module-level ``OOS_CONFIG(train_periods=8, val_periods=4,
    step_periods=2)`` and raises ``RuntimeError`` when no fold could be
    produced — that abort gate propagates verbatim (do NOT wrap).
    """
    _require(session, "factor_scores_dict", "returns_history")

    on_progress(phase="validating_oos")
    result = validate_oos(session.factor_scores_dict, session.returns_history)
    on_progress(phase="done")

    update_session(session_id, oos_result=result)
    return {
        "n_folds": result.n_folds,
        "oos_results": _oos_records(result),
        "config": dict(_OOS_VALIDATION_PAYLOAD_CONFIG),
    }


# ---------------------------------------------------------------------------
# Step 8 — coverage_gate
# ---------------------------------------------------------------------------


_COVERAGE_DEFAULT_MIN_FACTORS: int = 2


def _coverage_membership(
    is_result: Any, oos_result: Any
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(passing, is_only, oos_only)`` factor names.

    Passing = IS-significant ∩ OOS ICIR > 0. Membership is computed
    locally so the success payload can carry the breakdown without
    re-deriving it from the raised exception.
    """
    is_sig = set(is_result.significant_factors)
    icir = oos_result.mean_oos_icir
    oos_pos = {f for f in icir.index if icir[f] > 0}
    passing = sorted(is_sig & oos_pos)
    is_only = sorted(is_sig - oos_pos)
    oos_only = sorted(oos_pos - is_sig)
    return passing, is_only, oos_only


def step_coverage_gate(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 8 — IS ∩ OOS coverage gate (factor-quality firewall).

    Raises ``optimizer.exceptions.FactorCoverageError`` verbatim when
    fewer than ``min_factors`` factors satisfy both NW p<0.10 (IS BH)
    and OOS ICIR>0. ``min_factors`` is overridable via ``params``
    (default 2).
    """
    _require(session, "is_result", "oos_result")
    min_factors = int(params.get("min_factors", _COVERAGE_DEFAULT_MIN_FACTORS))

    on_progress(phase="evaluating_coverage")
    passing, is_only, oos_only = _coverage_membership(
        session.is_result, session.oos_result
    )
    _check_factor_coverage(
        session.is_result, session.oos_result, min_factors=min_factors
    )
    on_progress(phase="done")

    payload = {
        "passing_factors": passing,
        "is_only_factors": is_only,
        "oos_only_factors": oos_only,
        "n_passing": len(passing),
        "min_factors": min_factors,
    }
    update_session(session_id, coverage_result=payload)
    return payload


# ---------------------------------------------------------------------------
# Step 9 — regime
# ---------------------------------------------------------------------------


_REGIME_PREVIEW_NOTE = (
    "Regime Preview (inspection only) — actual tilts applied inside Step 7"
)


class _RegimeRepoPersistence:
    """Adapter satisfying ``RegimePersistenceProtocol`` via the repo.

    Body copied verbatim from ``research/cli/_main.py`` to avoid coupling
    the api layer onto the research CLI (which already imports app).
    """

    def __init__(self, db_manager: Any, repo_class: Any) -> None:
        self._db_manager = db_manager
        self._repo_class = repo_class

    def upsert_regime_classification(self, country: str, regime: str) -> None:
        with self._db_manager.get_session() as session:
            self._repo_class(session).upsert_regime_classification(
                country=country, regime=regime
            )
            session.commit()


def _build_regime_persistence() -> _RegimeRepoPersistence:
    """Construct the persistence adapter with a lazy repo import.

    The lazy import avoids pulling repository modules at api-package load
    time (kept in line with the ``persist_regime=False`` default path).
    """
    from app.repositories.macro.macro_regime_repository import (
        MacroRegimeRepository,
    )

    return _RegimeRepoPersistence(database_manager, MacroRegimeRepository)


def step_regime(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 9 — macro regime classification (Regime Preview).

    ``classify_and_tilt`` returns ``(MacroRegime, dict[FactorGroupType,
    float])``. The function is logging-only for tilt application
    (issue #529); the actual tilts are re-computed inside Step 7's
    ``run_full_pipeline_with_selection``. Persistence is opt-in via
    ``params["persist_regime"]``.
    """
    _require(session, "assembly")
    persist_regime = bool(params.get("persist_regime", False))
    persistence = _build_regime_persistence() if persist_regime else None

    on_progress(phase="classifying_regime")
    regime, tilts = classify_and_tilt(session.assembly, regime_persistence=persistence)
    on_progress(phase="done")

    tilts_payload = _to_json(tilts)
    regime_value = _to_json(regime)
    update_session(
        session_id,
        regime_result={"regime": regime_value, "tilts": tilts_payload},
    )
    return {
        "regime": regime_value,
        "tilts": tilts_payload,
        "persisted": persist_regime,
        "note": _REGIME_PREVIEW_NOTE,
    }


# ---------------------------------------------------------------------------
# Step 10 — optimize
# ---------------------------------------------------------------------------


_HOCKEY_STICK_LOGGER = "research.optimization._rebalance"
_OPTIMIZE_PHASES: tuple[str, ...] = (
    "scoring",
    "selecting",
    "walk-forward",
    "optimizing",
)


class _HockeyStickWatcher(logging.Handler):
    """Capture whether a 'hockey-stick' warning fires on the rebalance logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.triggered: bool = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover — defensive
            # A logging.Handler.emit must never raise: a malformed log record
            # is dropped rather than propagating into the logging machinery.
            return
        if "hockey-stick" in msg:
            self.triggered = True


def _breakdown(weights: pd.Series, mapping: dict[str, str]) -> dict[str, float]:
    """Aggregate ``weights`` by ``mapping[ticker]``. Unknowns dropped."""
    bucket: dict[str, float] = {}
    for ticker, w in weights.items():
        key = mapping.get(str(ticker))
        if key is None:
            continue
        bucket[key] = bucket.get(key, 0.0) + float(w)
    return bucket


def _weights_records(weights: pd.Series) -> list[dict]:
    return _to_json(
        [{"ticker": str(t), "weight": float(w)} for t, w in weights.items()]
    )


def step_optimize(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 10 — factor-based stock selection + MeanRisk optimization.

    Delegates to ``research.pipeline._optimize.optimize_portfolio`` which
    drives one monolithic ``run_full_pipeline_with_selection`` call (a
    single sklearn ``Pipeline.fit``). Progress is therefore coarse:
    phases ``scoring`` → ``selecting`` → ``walk-forward`` → ``optimizing``
    are emitted in advance for UI ordering; the call itself is a single
    blocking compute.

    Hockey-stick detection uses a logging handler attached to
    ``research.optimization._rebalance`` (where ``_hockey_stick_warn``
    logs at WARNING level) — the handler is removed in ``finally`` so
    state never leaks across runs.
    """
    _require(session, "assembly", "investable", "oos_result")

    run_config = session.run_config
    n_selected = run_config["n_selected"]
    cost_bps = run_config["cost_bps"]
    country_map = run_config.get("country_map", {}) or {}
    robust = bool(params.get("robust", False))
    seed = params.get("seed")
    uncertainty_level = 0.95

    for phase in _OPTIMIZE_PHASES:
        on_progress(phase=phase)

    # Regime-conditional per-sector bands (mirrors research/cli/_main.py).
    # Default matrix; the optimizer overrides static max_sector_weight with
    # per-sector (floor, cap) rows resolved from the classified regime.
    sector_bands_cfg = SectorRegimeBandsConfig.for_default()

    watcher = _HockeyStickWatcher()
    rebalance_logger = logging.getLogger(_HOCKEY_STICK_LOGGER)
    rebalance_logger.addHandler(watcher)
    try:
        result = optimize_portfolio(
            session.assembly,
            session.investable,
            session.oos_result.per_fold_ic,
            n_selected=n_selected,
            cost_bps=cost_bps,
            country_map=country_map,
            robust=robust,
            uncertainty_level=uncertainty_level,
            seed=seed,
            sector_bands_config=sector_bands_cfg,
        )
    finally:
        rebalance_logger.removeHandler(watcher)

    on_progress(phase="done")

    weights = result.weights
    update_session(session_id, optimize_result=result)
    return {
        "weights": _weights_records(weights),
        "n_selected": len(weights),
        "is_sharpe": safe_float(result.summary.get("sharpe_ratio")),
        "net_sharpe": safe_float(getattr(result, "net_sharpe_ratio", None)),
        "hockey_stick_warning": watcher.triggered,
        "sector_breakdown": _breakdown(weights, dict(session.assembly.sector_mapping)),
        "country_breakdown": _breakdown(weights, dict(country_map)),
    }


# ---------------------------------------------------------------------------
# Step 11 — rebalance_decision
# ---------------------------------------------------------------------------


def _resolve_prev_weights(params: dict) -> np.ndarray | None:
    """Return ``prev_weights`` honouring ``force_cold_start`` override."""
    if params.get("force_cold_start"):
        return None
    raw = params.get("prev_weights")
    if raw is None:
        return None
    return np.asarray(raw, dtype=float)


def _resolve_last_review_date(params: dict, current_date: pd.Timestamp) -> pd.Timestamp:
    """Return the supplied override or read from disk fallback."""
    override = params.get("last_review_date")
    if override is not None:
        return pd.Timestamp(override).normalize()
    return _read_last_review_date(current_date)


def step_rebalance_decision(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 11 — hybrid rebalance decision + diagnostic warnings.

    Cold start: ``params["force_cold_start"]`` OR ``prev_weights`` not
    supplied → ``_decide_rebalance`` returns ``(False, "cold_start")`` and
    the review-date file is bumped (matching CLI semantics).

    Weight-count warning flags when ``n_weights ∉ [15, 30]`` — this band
    is the wizard's authoritative one; the stale ``_factors.py:44-45``
    constants are deliberately NOT used.
    """
    _require(session, "optimize_result")
    result = session.optimize_result
    target_weights = np.asarray(result.weights.to_numpy(), dtype=float)

    prev_weights = _resolve_prev_weights(params)
    current_date = pd.Timestamp.today().normalize()
    last_review_date = _resolve_last_review_date(params, current_date)

    on_progress(phase="deciding_rebalance")
    decision, reason = _decide_rebalance(
        prev_weights=prev_weights,
        target_weights=target_weights,
        current_date=current_date,
        last_review_date=last_review_date,
    )
    if decision or reason == "cold_start":
        _write_last_review_date(current_date)
    on_progress(phase="done")

    sector_mapping = (
        dict(session.assembly.sector_mapping) if session.assembly is not None else {}
    )
    n_weights = len(result.weights)
    payload = {
        "decision": bool(decision),
        "reason": reason,
        "n_weights": n_weights,
        "weight_count_warning": not (_N_SELECTED_MIN <= n_weights <= _N_SELECTED_MAX),
        "missing_sectors": _missing_gics_sectors(result.weights, sector_mapping),
        "current_date": str(current_date.date()),
        "last_review_date": str(last_review_date.date()),
    }
    update_session(session_id, rebalance_result=payload)
    return payload


# ---------------------------------------------------------------------------
# Step 12 — cost
# ---------------------------------------------------------------------------


def _merge_country_costs(
    overrides: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Deep-merge ``overrides`` onto a copy of ``COUNTRY_COSTS_BPS``.

    The module-level ``COUNTRY_COSTS_BPS`` is NEVER mutated; callers
    pass per-run override dicts and receive a freshly-built map.
    """
    merged: dict[str, dict[str, float]] = {
        country: dict(costs) for country, costs in COUNTRY_COSTS_BPS.items()
    }
    for country, patch in (overrides or {}).items():
        base = dict(merged.get(country, _DEFAULT_COSTS))
        base.update(patch)
        merged[country] = base
    return merged


def _weighted_cost_bps_from(
    weights: pd.Series,
    country_map: dict[str, str],
    costs: dict[str, dict[str, float]],
) -> float:
    """Inline weighted cost using ``costs`` (matches ``compute_weighted_cost_bps``)."""
    clean = weights.dropna()
    if clean.empty:
        return 0.0
    totals = [
        sum(costs.get(country_map.get(str(t), ""), _DEFAULT_COSTS).values())
        for t in clean.index
    ]
    return float((clean.to_numpy() * np.asarray(totals, dtype=float)).sum())


def _per_country_breakdown(
    weights: pd.Series,
    country_map: dict[str, str],
    costs: dict[str, dict[str, float]],
) -> list[dict]:
    """Return cost components for each country present in ``weights``."""
    seen: dict[str, dict[str, float]] = {}
    for ticker in weights.dropna().index:
        country = country_map.get(str(ticker), "")
        if country in seen:
            continue
        components = costs.get(country, _DEFAULT_COSTS)
        seen[country] = {
            "stamp": float(components.get("stamp", 0.0)),
            "spread": float(components.get("spread", 0.0)),
            "fx": float(components.get("fx", 0.0)),
        }
    return [
        {
            "country": country,
            "stamp": v["stamp"],
            "spread": v["spread"],
            "fx": v["fx"],
            "total": v["stamp"] + v["spread"] + v["fx"],
        }
        for country, v in seen.items()
    ]


def step_cost(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 12 — round-trip transaction-cost (bps) against assumed cost.

    Mirrors ``compute_weighted_cost_bps`` but rebuilds the calculation
    against a private merged map so per-run ``country_cost_overrides``
    never mutate ``COUNTRY_COSTS_BPS``.
    """
    _require(session, "optimize_result")
    weights = session.optimize_result.weights
    country_map: dict[str, str] = dict(session.run_config.get("country_map", {}) or {})
    overrides = params.get("country_cost_overrides") or {}
    costs = _merge_country_costs(overrides)

    on_progress(phase="computing_cost")
    cost_actual = _weighted_cost_bps_from(weights, country_map, costs)
    cost_assumed = float(session.run_config.get("cost_bps", 10.0))
    on_progress(phase="done")

    payload = {
        "cost_bps_actual": cost_actual,
        "cost_bps_assumed": cost_assumed,
        "exceeds_assumed": cost_actual > cost_assumed,
        "per_country": _per_country_breakdown(weights, country_map, costs),
    }
    update_session(session_id, cost_result=payload)
    return payload


# ---------------------------------------------------------------------------
# Step 13 — report
# ---------------------------------------------------------------------------


_CHECKLIST_TOTAL: int = 17


def _make_session_tempdir(session_id: str) -> Path:
    """Create an isolated tempdir for one session's report artefacts."""
    return Path(tempfile.mkdtemp(prefix=f"pipeline_{session_id[:8]}_"))


def _artifact_paths(output_dir: Path) -> dict[str, str]:
    """Absolute paths to the four canonical report artefacts."""
    return {
        "report_md": str(output_dir / "report.md"),
        "weights_csv": str(output_dir / "weights.csv"),
        "metrics_json": str(output_dir / "metrics.json"),
        "checklist_json": str(output_dir / "checklist.json"),
        "weights_diagnostic": str(output_dir / "weights_diagnostic.csv"),
    }


def step_report(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 13 — performance report + checklist + report.md render.

    Writes ``report.md``, ``weights.csv``, ``metrics.json``,
    ``checklist.json`` into a per-session tempdir. The CLI's
    ``_apply_terminal_gate`` is deliberately NOT invoked — it calls
    ``sys.exit`` and is unsuitable for an API path. Instead derives a
    boolean ``checklist_passed`` from ``pass_count == _CHECKLIST_TOTAL``.
    """
    _require(session, "assembly", "optimize_result", "cost_result")

    result = session.optimize_result
    assembly = session.assembly
    cost_result = session.cost_result
    is_result = session.is_result
    oos_result = session.oos_result
    regime_result = session.regime_result or {}

    # The research report renderer expects ``result.rebalance_decision``
    # as a ``tuple[bool, str]`` (the CLI attaches it via ``_decide_rebalance``
    # before rendering — research/cli/_main.py). The API path keeps the
    # decision in ``session.rebalance_result``; mirror the CLI by attaching
    # the same tuple onto the PortfolioResult.
    rebalance_result = session.rebalance_result or {}
    result.rebalance_decision = (
        bool(rebalance_result.get("decision", False)),
        str(rebalance_result.get("reason", "")),
    )
    country_map: dict[str, str] = dict(session.run_config.get("country_map", {}) or {})
    cost_bps = float(session.run_config.get("cost_bps", 10.0))
    tax_rate = float(session.run_config.get("tax_rate", 0.26))

    output_dir = _make_session_tempdir(session_id)

    # Resolve regime-conditional sector bands so the checklist uses the same
    # thresholds as the optimizer (mirrors research/cli/_main.py). The default
    # matrix is deterministic, so rebuild it here rather than threading the
    # config through the session. ``None`` when the optimizer did not classify
    # a regime — the checklist then keeps its static thresholds.
    resolved_bands: dict[str, tuple[float, float]] | None = None
    classified_regime = getattr(result, "classified_regime", None)
    if classified_regime is not None:
        resolved_bands = resolve_sector_bands(
            classified_regime, SectorRegimeBandsConfig.for_default()
        )

    on_progress(phase="computing_metrics")
    pass_count, checklist_rules, metrics, chart_paths = report_performance(
        result,
        assembly,
        country_map,
        cost_bps_actual=cost_result.get("cost_bps_actual"),
        cost_bps=cost_bps,
        tax_rate=tax_rate,
        validation_report=is_result,
        oos_per_fold_ic=oos_result.per_fold_ic if oos_result is not None else None,
        output_dir=output_dir,
        sector_bands=resolved_bands,
    )

    on_progress(phase="rendering_report")
    _render_research_report(
        output_dir=output_dir,
        assembly_hash=assembly.assembly_hash,
        current_regime=regime_result.get("regime"),
        tilts=regime_result.get("tilts", {}),
        validation_report=is_result,
        oos_per_fold_ic=oos_result.per_fold_ic if oos_result is not None else None,
        result=result,
        country_map=country_map,
        checklist_rules=checklist_rules,
        metrics=metrics,
        chart_paths=chart_paths,
    )
    on_progress(phase="done")

    checklist_passed = pass_count == _CHECKLIST_TOTAL and bool(checklist_rules)

    # Diagnostic-only weights dump on a near-miss (mirrors research
    # ``_apply_terminal_gate``). The blessed ``weights.csv`` is gate-passed;
    # this distinct file is written only when the checklist fails so it can
    # never be mistaken for the accepted artefact.
    if not checklist_passed:
        sorted_w = result.weights.sort_values(ascending=False)
        pd.DataFrame(
            {
                "ticker": list(sorted_w.index),
                "weight": sorted_w.to_numpy(dtype=float),
            }
        ).to_csv(output_dir / "weights_diagnostic.csv", index=False)

    payload = {
        "pass_count": int(pass_count),
        "checklist_total": _CHECKLIST_TOTAL,
        "checklist_passed": checklist_passed,
        "checklist_rules": _to_json(checklist_rules),
        "metrics": _to_json(metrics),
        "artifact_paths": _artifact_paths(output_dir),
        "chart_paths": [str(p) for p in chart_paths],
        "output_dir": str(output_dir),
    }
    update_session(session_id, report_result=payload)
    return payload


# ---------------------------------------------------------------------------
# Step 14 (terminal) — persist
# ---------------------------------------------------------------------------
#
# TODO(Cycle 3): immediate session eviction (`delete_session(session_id)`) once
# the routing layer ties session lifetime to this terminal step.


def _resolve_persist_flag(params: dict, run_config: dict) -> bool:
    """Return whether persistence is requested (param override OR run_config)."""
    return bool(params.get("force_persist", False) or run_config.get("persist", False))


def step_persist(
    session_id: str,
    session: _PipelineSession,
    params: dict,
    on_progress: Any,
) -> dict:
    """Step 14 (terminal) — persist research snapshot, gated on 17/17 PASS.

    Calls ``_persist_research_snapshot`` only when ``persist`` AND
    ``checklist_passed`` are BOTH true. ``params["force_persist"]`` may
    bypass ``run_config["persist"]`` (testing aid) but it does NOT
    bypass the checklist gate — 17/17 is non-negotiable.
    """
    _require(session, "optimize_result", "assembly", "report_result")

    persist_flag = _resolve_persist_flag(params, session.run_config)
    report = session.report_result
    checklist_passed = bool(report.get("checklist_passed", False))
    pass_count = int(report.get("pass_count", 0))

    on_progress(phase="evaluating_persist_gate")
    if not persist_flag:
        reason = "persist_flag_off"
        persisted = False
    elif not checklist_passed:
        reason = "checklist_failed"
        persisted = False
    else:
        _persist_research_snapshot(
            result=session.optimize_result,
            assembly=session.assembly,
            metrics=report["metrics"],
            cost_bps=float(session.run_config.get("cost_bps", 10.0)),
        )
        reason = "persisted"
        persisted = True
    on_progress(phase="done")

    return {
        "persisted": persisted,
        "reason": reason,
        "pass_count": pass_count,
        "checklist_passed": checklist_passed,
    }
