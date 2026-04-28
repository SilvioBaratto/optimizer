"""Factor validation and composite scoring service.

Responsibilities:
  - Run in-sample and out-of-sample factor validation.
  - Compute composite factor scores using configurable weighting methods.
  - Persist validation reports via FactorRepository.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.repositories.factor_repository import FactorRepository
from app.services._factor_helpers import (
    FactorDataError,
    _build_factor_scores_dict,
    _build_multiindex_factor_df,
    _build_multiindex_returns,
    _build_returns_from_price_rows,
    _build_standardized_scores_df,
    _fetch_price_rows,
)
from optimizer.factors import (
    compute_composite_score,
    run_factor_oos_validation,
    run_factor_validation,
)

logger = logging.getLogger(__name__)

# Methods that require IC history from validation reports
_IC_METHODS: frozenset[str] = frozenset({"ic_weighted", "icir_weighted"})
# Methods that require ML training data
_ML_METHODS: frozenset[str] = frozenset({"ridge_weighted", "gbt_weighted"})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_factors(
    session: Session,
    tickers: list[str],
    start_date: datetime.date,
    end_date: datetime.date,
    factor_type: str,
    validation_type: str = "in_sample",
) -> dict[str, Any]:
    """Run factor validation and persist the report.

    Args:
        session: Active SQLAlchemy session.
        tickers: Asset tickers to include in validation.
        start_date: Start of validation window.
        end_date: End of validation window.
        factor_type: Factor type string (e.g. ``book_to_price``).
        validation_type: ``in_sample`` or ``out_of_sample``.

    Returns:
        Dict with IC stats, t-stat, VIF, and metadata fields.

    Raises:
        FactorDataError: When no factor scores are found for the given window.
    """
    repo = FactorRepository(session)
    scores = repo.get_scores_by_tickers_and_date_range(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        factor_type=factor_type,
    )
    if not scores:
        raise FactorDataError(
            f"No factor scores found for factor_type={factor_type!r}, "
            f"tickers={tickers}, {start_date}–{end_date}"
        )

    price_rows = _fetch_price_rows(session, tickers, start_date, end_date)
    returns_history = _build_returns_from_price_rows(price_rows)

    if validation_type == "out_of_sample":
        report_data = _run_oos_validation(scores, returns_history, factor_type, end_date)
    else:
        report_data = _run_in_sample_validation(scores, returns_history, factor_type, end_date)

    report_data["validation_type"] = validation_type
    repo.save_validation_report(report_data)
    session.commit()

    return _serialize_report(report_data)


def _run_in_sample_validation(
    scores: list[Any],
    returns_history: pd.DataFrame,
    factor_type: str,
    report_date: datetime.date,
) -> dict[str, Any]:
    """Reshape scores, call run_factor_validation, return serializable dict.

    Args:
        scores: ORM score rows.
        returns_history: Wide returns DataFrame.
        factor_type: Factor type string.
        report_date: Date to attach to the report.

    Returns:
        Flat dict of validation stats.
    """
    factor_scores_history = _build_factor_scores_dict(scores)
    report = run_factor_validation(
        factor_scores_history=factor_scores_history,
        returns_history=returns_history,
    )
    return _extract_in_sample_stats(report, factor_type, report_date)


def _run_oos_validation(
    scores: list[Any],
    returns_history: pd.DataFrame,
    factor_type: str,
    report_date: datetime.date,
) -> dict[str, Any]:
    """Reshape scores into MultiIndex, call run_factor_oos_validation.

    Args:
        scores: ORM score rows.
        returns_history: Wide returns DataFrame.
        factor_type: Factor type string.
        report_date: Date to attach to the report.

    Returns:
        Flat dict of OOS validation stats.
    """
    scores_mi = _build_multiindex_factor_df(scores)
    returns_mi = _build_multiindex_returns(returns_history)
    oos_result = run_factor_oos_validation(scores=scores_mi, returns=returns_mi)
    return _extract_oos_stats(oos_result, factor_type, report_date)


def _extract_in_sample_stats(
    report: Any,
    factor_type: str,
    report_date: datetime.date,
) -> dict[str, Any]:
    """Extract scalar stats from FactorValidationReport into a flat dict.

    Args:
        report: Library FactorValidationReport object.
        factor_type: Factor type string.
        report_date: Date for the report record.

    Returns:
        Flat dict with ic_mean, ic_std, icir, t_stat, p_value, vif, details.
    """
    ic_result = _find_ic_result(report.ic_results, factor_type)
    vif = _extract_vif(report.vif_scores, factor_type)
    return {
        "report_date": report_date,
        "factor_type": factor_type,
        "ic_mean": ic_result.mean_ic if ic_result else None,
        "ic_std": ic_result.ic_std if ic_result else None,
        "icir": _safe_icir(ic_result),
        "t_stat": ic_result.t_stat if ic_result else None,
        "p_value": ic_result.p_value if ic_result else None,
        "vif": vif,
        "details": {"significant_factors": list(report.significant_factors)},
    }


def _extract_oos_stats(
    result: Any,
    factor_type: str,
    report_date: datetime.date,
) -> dict[str, Any]:
    """Extract scalar stats from FactorOOSResult into a flat dict.

    Args:
        result: Library FactorOOSResult object.
        factor_type: Factor type string.
        report_date: Date for the report record.

    Returns:
        Flat dict with ic_mean, icir, n_folds, and placeholder nulls.
    """
    return {
        "report_date": report_date,
        "factor_type": factor_type,
        "ic_mean": _series_get(result.mean_oos_ic, factor_type),
        "ic_std": None,
        "icir": _series_get(result.mean_oos_icir, factor_type),
        "t_stat": None,
        "p_value": None,
        "vif": None,
        "details": {"n_folds": result.n_folds},
    }


def _serialize_report(report_data: dict[str, Any]) -> dict[str, Any]:
    """Return the report data dict as-is (already serializable).

    Args:
        report_data: Flat dict of validation stats.

    Returns:
        The same dict unchanged.
    """
    return report_data


def _find_ic_result(ic_results: list[Any], factor_type: str) -> Any | None:
    """Return the ICResult matching factor_type, or None.

    Args:
        ic_results: List of library ICResult objects.
        factor_type: Factor type string to search for.

    Returns:
        Matching ICResult or None.
    """
    for r in ic_results:
        if r.factor_name == factor_type:
            return r
    return None


def _safe_icir(ic_result: Any | None) -> float | None:
    """Compute ICIR = mean_ic / ic_std, guarding division by zero.

    Args:
        ic_result: Library ICResult or None.

    Returns:
        ICIR float or None when result is absent or ic_std is zero.
    """
    if ic_result is None or ic_result.ic_std == 0:
        return None
    return ic_result.mean_ic / ic_result.ic_std


def _extract_vif(vif_scores: Any, factor_type: str) -> float | None:
    """Extract VIF for factor_type from a Series, returning None if absent.

    Args:
        vif_scores: pandas Series or None.
        factor_type: Factor type string key.

    Returns:
        VIF float or None.
    """
    if vif_scores is None:
        return None
    try:
        return float(vif_scores[factor_type])
    except (KeyError, TypeError):
        return None


def _series_get(series: Any, key: str) -> float | None:
    """Safe .get on a pd.Series, returning None if key is absent.

    Args:
        series: pandas Series or None.
        key: Lookup key.

    Returns:
        Float value or None.
    """
    if series is None:
        return None
    try:
        return float(series[key])
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def compute_factor_scores(
    session: Session,
    tickers: list[str],
    score_date: datetime.date,
    composite_method: str,
    training_start_date: datetime.date | None = None,
    training_end_date: datetime.date | None = None,
    group_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute composite factor scores for tickers on a given date.

    Args:
        session: Active SQLAlchemy session.
        tickers: Asset tickers to score.
        score_date: Date at which to look up standardized factor scores.
        composite_method: One of equal_weight / ic_weighted / icir_weighted /
            ridge_weighted / gbt_weighted.
        training_start_date: Required for ridge_weighted / gbt_weighted.
        training_end_date: Required for ridge_weighted / gbt_weighted.
        group_weights: Optional per-group weight overrides.

    Returns:
        Dict with ``score_date``, ``scores`` (ticker->float), and
        ``group_contributions`` (group->float).

    Raises:
        FactorDataError: When no standardized scores are found.
    """
    repo = FactorRepository(session)
    raw_scores = repo.get_scores_by_tickers_at_date(tickers=tickers, score_date=score_date)

    standardized = [s for s in raw_scores if s.standardized_score is not None]
    if not standardized:
        raise FactorDataError(
            f"No standardized factor scores found for tickers={tickers} on {score_date}"
        )

    std_df, coverage_df = _build_standardized_scores_df(standardized)
    extra_kwargs = _build_extra_kwargs(
        repo,
        composite_method,
        group_weights,
        tickers,
        training_start_date,
        training_end_date,
        session,
    )

    result = compute_composite_score(
        standardized_factors=std_df,
        coverage=coverage_df,
        **extra_kwargs,
    )

    scores, contributions = _extract_score_results(result)
    return {
        "score_date": score_date,
        "scores": scores,
        "group_contributions": contributions,
    }


def _build_extra_kwargs(
    repo: FactorRepository,
    composite_method: str,
    group_weights: dict[str, float] | None,
    tickers: list[str],
    training_start_date: datetime.date | None,
    training_end_date: datetime.date | None,
    session: Session,
) -> dict[str, Any]:
    """Build extra keyword arguments for compute_composite_score.

    Extracts IC history for IC-weighted methods and training data for ML
    methods, keeping ``compute_factor_scores`` under 30 lines.

    Args:
        repo: FactorRepository bound to the active session.
        composite_method: Composite scoring method string.
        group_weights: Optional per-group weight overrides.
        tickers: Asset tickers for training data retrieval.
        training_start_date: Training window start for ML methods.
        training_end_date: Training window end for ML methods.
        session: SQLAlchemy session for price fetching.

    Returns:
        Dict of extra kwargs to spread into ``compute_composite_score``.
    """
    extra: dict[str, Any] = {}
    if group_weights is not None:
        extra["group_weights"] = group_weights
    if composite_method in _IC_METHODS:
        extra["ic_history"] = _build_ic_history(repo)
    elif composite_method in _ML_METHODS:
        extra.update(
            _build_training_data(repo, tickers, training_start_date, training_end_date, session)
        )
    return extra


def _build_ic_history(repo: FactorRepository) -> pd.DataFrame:
    """Fetch IC history from validation reports for IC-weighted methods.

    Args:
        repo: FactorRepository bound to the active session.

    Returns:
        Pivoted DataFrame (report_date x factor_type) of ic_mean values;
        empty DataFrame when no reports exist.
    """
    reports = repo.get_validation_reports(validation_type="in_sample")
    if not reports:
        return pd.DataFrame()

    rows = [
        {
            "factor_type": r.factor_type,
            "ic_mean": r.ic_mean,
            "icir": r.icir,
            "report_date": r.report_date,
        }
        for r in reports
        if r.factor_type is not None and r.ic_mean is not None
    ]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.pivot_table(index="report_date", columns="factor_type", values="ic_mean")


def _build_training_data(
    repo: FactorRepository,
    tickers: list[str],
    training_start_date: datetime.date | None,
    training_end_date: datetime.date | None,
    session: Session,
) -> dict[str, Any]:
    """Build training_scores and training_returns for ML-based methods.

    Args:
        repo: FactorRepository bound to the active session.
        tickers: Asset tickers.
        training_start_date: Training window start.
        training_end_date: Training window end.
        session: SQLAlchemy session for price fetching.

    Returns:
        Dict with ``training_scores`` and ``training_returns`` keys; empty
        dict when dates are missing or no data is found.
    """
    if training_start_date is None or training_end_date is None:
        return {}

    training_raw = repo.get_scores_by_tickers_and_date_range(
        tickers=tickers,
        start_date=training_start_date,
        end_date=training_end_date,
    )
    standardized_training = [s for s in training_raw if s.standardized_score is not None]
    if not standardized_training:
        return {}

    training_df, _ = _build_standardized_scores_df(standardized_training)
    price_rows = _fetch_price_rows(session, tickers, training_start_date, training_end_date)
    returns_df = _build_returns_from_price_rows(price_rows)
    training_returns = (
        returns_df.mean(axis=1) if not returns_df.empty else pd.Series(dtype=float)
    )
    return {"training_scores": training_df, "training_returns": training_returns}


def _extract_score_results(
    result: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    """Unpack compute_composite_score result into (scores, group_contributions).

    Args:
        result: Output from ``compute_composite_score`` — either a Series or
            a DataFrame.

    Returns:
        Tuple of (scores dict, group_contributions dict).
    """
    if isinstance(result, pd.Series):
        scores = {str(k): float(v) for k, v in result.items() if pd.notna(v)}
        return scores, {}
    if isinstance(result, pd.DataFrame):
        first_col = result.columns[0]
        scores = {str(idx): float(result.at[idx, first_col]) for idx in result.index}
        contributions: dict[str, float] = {
            str(col): float(result[col].mean()) for col in result.columns[1:]
        }
        return scores, contributions
    return {}, {}
