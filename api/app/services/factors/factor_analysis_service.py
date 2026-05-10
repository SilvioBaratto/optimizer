"""Factor analysis service — stock selection, exposure constraints, quintile
spread analysis, and macro regime tilts.

Responsibilities:
  - Select stocks from composite factor scores with hysteresis support.
  - Build linear factor exposure constraint matrices.
  - Compute quintile spread diagnostics for a single factor.
  - Classify macro regime and apply multiplicative tilts to group weights.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.repositories.factors.factor_repository import FactorRepository
from app.repositories.macro.macro_regime_repository import MacroRegimeRepository
from app.services.factors._factor_helpers import (
    FactorDataError,
    _build_returns_from_price_rows,
    _build_standardized_scores_df,
    _fetch_price_rows,
)
from optimizer.factors import (
    FactorGroupType,
    RegimeTiltConfig,
    SelectionConfig,
    SelectionMethod,
    apply_regime_tilts,
    build_factor_exposure_constraints,
    classify_regime,
    compute_quintile_spread,
    get_regime_tilts,
    select_stocks,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stock selection
# ---------------------------------------------------------------------------


def select_stocks_for_tickers(
    session: Session,
    tickers: list[str],
    start_date: datetime.date,
    end_date: datetime.date,
    current_members: list[str] | None = None,
    method: str | None = None,
    target_count: int | None = None,
    target_quantile: float | None = None,
    buffer_fraction: float | None = None,
    sector_balance: bool = False,
) -> dict[str, Any]:
    """Select stocks from composite scores in the DB for the given tickers/dates.

    Args:
        session: SQLAlchemy session.
        tickers: Candidate universe tickers.
        start_date: Start of scoring window.
        end_date: End of scoring window.
        current_members: Currently selected tickers for buffer/hysteresis.
        method: ``fixed_count`` or ``quantile``.
        target_count: Target number of stocks (fixed_count method).
        target_quantile: Top quantile threshold (quantile method).
        buffer_fraction: Buffer fraction for fixed_count hysteresis.
        sector_balance: Apply sector-proportional balancing.

    Returns:
        Dict with ``selected_tickers``, ``count``, ``turnover``, ``buffer_zone``.

    Raises:
        FactorDataError: When no factor scores are found.
    """
    raw_scores = _fetch_and_validate_scores(session, tickers, start_date, end_date)
    composite_scores = _build_composite_scores_series(raw_scores)
    config = _build_selection_config(
        method=method,
        target_count=target_count,
        target_quantile=target_quantile,
        buffer_fraction=buffer_fraction,
        sector_balance=sector_balance,
    )
    return _run_selection(composite_scores, config, current_members)


def _fetch_and_validate_scores(
    session: Session,
    tickers: list[str],
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[Any]:
    """Fetch factor scores and raise FactorDataError when empty."""
    repo = FactorRepository(session)
    raw_scores = repo.get_scores_by_tickers_and_date_range(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    if not raw_scores:
        raise FactorDataError(
            f"No factor scores found for tickers={tickers}, {start_date}–{end_date}"
        )
    return raw_scores


def _run_selection(
    composite_scores: pd.Series,
    config: SelectionConfig,
    current_members: list[str] | None,
) -> dict[str, Any]:
    """Execute select_stocks and build the response dict."""
    prev_members = pd.Index(current_members) if current_members else None
    selected, turnover = select_stocks(
        scores=composite_scores,
        config=config,
        current_members=prev_members,
        return_turnover=True,
    )
    buffer_entered, buffer_exited = _compute_buffer_zone(
        selected=selected,
        current_members=prev_members,
    )
    return {
        "selected_tickers": list(selected),
        "count": len(selected),
        "turnover": turnover
        if prev_members is not None and len(prev_members) > 0
        else None,
        "buffer_zone": {"entered": buffer_entered, "exited": buffer_exited},
    }


def _build_composite_scores_series(scores: list[Any]) -> pd.Series:
    """Average standardized scores per ticker across all factor types."""
    rows = [
        {"ticker": s.ticker, "score": s.standardized_score or s.raw_score}
        for s in scores
    ]
    df = pd.DataFrame(rows)
    return pd.Series(df.groupby("ticker")["score"].mean())


def _build_selection_config(
    method: str | None,
    target_count: int | None,
    target_quantile: float | None,
    buffer_fraction: float | None,
    sector_balance: bool,
) -> SelectionConfig:
    """Construct SelectionConfig from optional request parameters."""
    sel_method = (
        SelectionMethod.QUANTILE
        if method == "quantile"
        else SelectionMethod.FIXED_COUNT
    )
    kwargs: dict[str, Any] = {"method": sel_method, "sector_balance": sector_balance}
    if target_count is not None:
        kwargs["target_count"] = target_count
    if target_quantile is not None:
        kwargs["target_quantile"] = target_quantile
    if buffer_fraction is not None:
        kwargs["buffer_fraction"] = buffer_fraction
    return SelectionConfig(**kwargs)


def _compute_buffer_zone(
    selected: pd.Index,
    current_members: pd.Index | None,
) -> tuple[list[str], list[str]]:
    """Compute tickers entering or exiting relative to previous members."""
    if current_members is None or len(current_members) == 0:
        return list(selected), []
    return list(selected.difference(current_members)), list(
        current_members.difference(selected)
    )


# ---------------------------------------------------------------------------
# Exposure constraints
# ---------------------------------------------------------------------------


def build_exposure_constraints_for_tickers(
    session: Session,
    tickers: list[str],
    start_date: datetime.date,
    end_date: datetime.date,
    bounds: list[float] | dict[str, list[float]],
) -> dict[str, Any]:
    """Build factor exposure constraint matrices from DB scores.

    Args:
        session: SQLAlchemy session.
        tickers: Asset tickers.
        start_date: Start of scoring window.
        end_date: End of scoring window.
        bounds: Uniform [lb, ub] list or per-factor dict.

    Returns:
        Dict with ``left_inequality`` (nested list) and ``right_inequality``
        (list).

    Raises:
        FactorDataError: When no standardized scores are found.
    """
    repo = FactorRepository(session)
    raw_scores = repo.get_scores_by_tickers_and_date_range(
        tickers=tickers, start_date=start_date, end_date=end_date
    )
    standardized = [s for s in raw_scores if s.standardized_score is not None]
    if not standardized:
        raise FactorDataError(
            f"No standardized factor scores found for tickers={tickers}, "
            f"{start_date}–{end_date}"
        )

    factor_scores_df, _ = _build_standardized_scores_df(standardized)
    resolved_bounds = _resolve_bounds(bounds)
    constraints = build_factor_exposure_constraints(
        factor_scores=factor_scores_df,
        bounds=resolved_bounds,
    )
    return {
        "left_inequality": constraints.left_inequality.tolist(),
        "right_inequality": constraints.right_inequality.tolist(),
    }


def _resolve_bounds(
    bounds: list[float] | dict[str, list[float]],
) -> tuple[float, float] | dict[str, tuple[float, float]]:
    """Convert JSON-compatible bounds to library-expected tuple/dict form."""
    if isinstance(bounds, list):
        return (float(bounds[0]), float(bounds[1]))
    return {k: (float(v[0]), float(v[1])) for k, v in bounds.items()}


# ---------------------------------------------------------------------------
# Quintile spread
# ---------------------------------------------------------------------------


def compute_quintile_spread_for_tickers(
    session: Session,
    tickers: list[str],
    factor_name: str,
    start_date: datetime.date,
    end_date: datetime.date,
    n_quantiles: int = 5,
) -> dict[str, Any]:
    """Compute quintile spread analysis from DB scores and price history.

    Args:
        session: SQLAlchemy session.
        tickers: Asset tickers.
        factor_name: Factor type string (e.g. ``momentum_12_1``).
        start_date: Start of analysis window.
        end_date: End of analysis window.
        n_quantiles: Number of quantile buckets (default 5).

    Returns:
        Dict with ``quintile_cumulative_returns``, ``spread_cumulative_return``,
        ``annualized_spread``.

    Raises:
        FactorDataError: When no factor scores or prices are found.
    """
    repo = FactorRepository(session)
    raw_scores = repo.get_scores_by_tickers_and_date_range(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        factor_type=factor_name,
    )
    if not raw_scores:
        raise FactorDataError(
            f"No factor scores found for factor_type={factor_name!r}, "
            f"tickers={tickers}, {start_date}–{end_date}"
        )

    scores_df = _build_scores_wide(raw_scores)
    price_rows = _fetch_price_rows(session, tickers, start_date, end_date)
    returns_df = _build_returns_from_price_rows(price_rows)

    if returns_df.empty:
        raise FactorDataError(
            f"No price history found for tickers={tickers}, {start_date}–{end_date}"
        )

    result = compute_quintile_spread(
        scores=scores_df, returns=returns_df, n_quantiles=n_quantiles
    )
    return {
        "quintile_cumulative_returns": _cumulative_returns_dict(
            result.quintile_returns
        ),
        "spread_cumulative_return": _cumulative_series(result.spread_returns),
        "annualized_spread": result.annualised_mean,
    }


def _build_scores_wide(scores: list[Any]) -> pd.DataFrame:
    """Pivot factor scores to a dates x tickers DataFrame."""
    rows = [
        {"date": pd.Timestamp(s.score_date), "ticker": s.ticker, "value": s.raw_score}
        for s in scores
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.pivot_table(index="date", columns="ticker", values="value")


def _cumulative_returns_dict(quintile_returns: pd.DataFrame) -> dict[str, list[float]]:
    """Convert quintile_returns DataFrame to per-column cumulative return lists."""
    return {
        str(col): [
            round(float(v), 8)
            for v in (1 + quintile_returns[col].fillna(0)).cumprod() - 1
        ]
        for col in quintile_returns.columns
    }


def _cumulative_series(spread_returns: pd.Series) -> list[float]:
    """Convert spread returns Series to a cumulative return list."""
    cum = (1 + spread_returns.fillna(0)).cumprod() - 1
    return [round(float(v), 8) for v in cum]


# ---------------------------------------------------------------------------
# Regime tilt
# ---------------------------------------------------------------------------


def compute_regime_tilt(
    session: Session,
    group_weights: dict[str, float],
    enable: bool = True,
    max_tilt_multiplier: float | None = None,
    min_post_tilt_weight: float | None = None,
) -> dict[str, Any]:
    """Classify macro regime from DB and apply tilts to group weights.

    Args:
        session: SQLAlchemy session.
        group_weights: Base group weights keyed by FactorGroupType string values.
        enable: Whether to apply tilts.
        max_tilt_multiplier: Cap on any single tilt multiplier.
        min_post_tilt_weight: Floor for post-tilt group weight fraction.

    Returns:
        Dict with ``regime``, ``tilted_weights``, ``tilt_multipliers``.

    Raises:
        FactorDataError: When no macro data is available.
    """
    macro_df = _fetch_macro_dataframe(session)
    if macro_df.empty:
        raise FactorDataError(
            "No macro data found in database for regime classification"
        )

    regime = classify_regime(macro_df)
    typed_weights = _parse_group_weights(group_weights)
    tilt_config = _build_tilt_config(enable, max_tilt_multiplier, min_post_tilt_weight)
    return _apply_tilts(typed_weights, regime, tilt_config)


def _fetch_macro_dataframe(session: Session) -> pd.DataFrame:
    """Build a macro indicators DataFrame from Trading Economics observations."""
    repo = MacroRegimeRepository(session)
    observations = repo.get_te_observations()
    if not observations:
        return pd.DataFrame()

    rows = [
        {"date": obs.date, "indicator_key": obs.indicator_key, "value": obs.value}
        for obs in observations
        if obs.value is not None
    ]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="date", columns="indicator_key", values="value", aggfunc="last"
    )
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.sort_index()


def _parse_group_weights(
    group_weights: dict[str, float],
) -> dict[FactorGroupType, float]:
    """Convert string-keyed weights to FactorGroupType-keyed dict."""
    result: dict[FactorGroupType, float] = {}
    for key, weight in group_weights.items():
        try:
            result[FactorGroupType(key)] = float(weight)
        except ValueError:
            logger.warning("Unknown FactorGroupType %r — skipping", key)
    return result


def _build_tilt_config(
    enable: bool,
    max_tilt_multiplier: float | None,
    min_post_tilt_weight: float | None,
) -> RegimeTiltConfig:
    """Build RegimeTiltConfig from optional override parameters."""
    kwargs: dict[str, Any] = {"enable": enable}
    if max_tilt_multiplier is not None:
        kwargs["max_tilt_multiplier"] = max_tilt_multiplier
    if min_post_tilt_weight is not None:
        kwargs["min_post_tilt_weight"] = min_post_tilt_weight
    return RegimeTiltConfig(**kwargs)


def _apply_tilts(
    typed_weights: dict[FactorGroupType, float],
    regime: Any,
    tilt_config: RegimeTiltConfig,
) -> dict[str, Any]:
    """Apply regime tilts and return the serializable result dict."""
    tilted = apply_regime_tilts(
        group_weights=typed_weights, regime=regime, config=tilt_config
    )
    raw_tilts = get_regime_tilts(regime, tilt_config)
    return {
        "regime": regime.value,
        "tilted_weights": {g.value: float(w) for g, w in tilted.items()},
        "tilt_multipliers": {
            g.value: float(raw_tilts.get(g, 1.0)) for g in typed_weights
        },
    }
