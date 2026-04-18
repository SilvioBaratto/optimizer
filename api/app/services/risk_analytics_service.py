"""Risk analytics service: VaR/CVaR, correlation, factor exposure, concentration, and liquidity.

Each public method delegates to focused private helpers (< 10 lines each).
Single Responsibility: service only computes analytics, does not handle HTTP.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.stats import norm
from sqlalchemy.orm import Session

from app.repositories.factor_repository import FactorRepository
from app.repositories.yfinance_repository import YFinanceRepository

logger = logging.getLogger(__name__)

# Confidence levels used for VaR/CVaR (stored as string keys in response)
_CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)
_CONFIDENCE_KEYS = ("90", "95", "99")


class RiskAnalyticsService:
    """Compute risk analytics from stored price and factor data.

    Accepts a SQLAlchemy session; all DB access is delegated to repositories.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_var(
        self,
        portfolio_id: str,
        weights: dict[str, float],
        lookback: int,
        method: str,
    ) -> dict[str, Any]:
        """Compute VaR and CVaR at three confidence levels.

        Args:
            portfolio_id: Portfolio UUID string (used for fetching prices).
            weights: Dict of ticker → portfolio weight.
            lookback: Number of trading days to use.
            method: ``"historical"`` or ``"parametric"``.

        Returns:
            Dict with ``var``, ``cvar``, ``method``, ``lookback``, ``n_observations``.

        Raises:
            ValueError: When fewer observations than requested are available.
        """
        logger.debug("compute_var portfolio_id=%s method=%s lookback=%d", portfolio_id, method, lookback)
        returns = self._fetch_weighted_returns(weights, lookback)
        _assert_sufficient_data(returns, lookback)

        if method == "historical":
            var_dict, cvar_dict = _historical_var_cvar(returns)
        else:
            var_dict, cvar_dict = _parametric_var_cvar(returns)

        return {
            "var": var_dict,
            "cvar": cvar_dict,
            "method": method,
            "lookback": lookback,
            "n_observations": len(returns),
        }

    def compute_correlation(
        self,
        portfolio_id: str,
        weights: dict[str, float],
        lookback: int,
    ) -> dict[str, Any]:
        """Compute block-clustered correlation matrix.

        Args:
            portfolio_id: Portfolio UUID string (used for fetching prices).
            weights: Dict of ticker → portfolio weight.
            lookback: Number of trading days to use.

        Returns:
            Dict with ``assets``, ``matrix``, ``cluster_labels``.

        Raises:
            ValueError: When fewer than 2 assets have usable price data.
        """
        logger.debug("compute_correlation portfolio_id=%s lookback=%d", portfolio_id, lookback)
        prices = self._fetch_prices(weights, lookback)
        available = _surviving_tickers(weights, prices)
        _assert_at_least_two_assets(available)

        prices = prices[available].dropna()
        _assert_at_least_two_assets(list(prices.columns))

        corr_matrix = _compute_corr_matrix(prices)
        ordered_assets, reordered_corr, cluster_labels = _cluster_and_reorder(
            available, corr_matrix
        )

        return {
            "assets": ordered_assets,
            "matrix": reordered_corr.tolist(),
            "cluster_labels": cluster_labels,
        }

    def compute_factor_exposure(
        self,
        portfolio_id: str,
        weights: dict[str, float],
    ) -> dict[str, Any]:
        """Compute portfolio-weighted factor exposures.

        Returns empty ``exposures`` and ``asset_exposures`` dicts when no factor
        scores have been persisted for any ticker yet (empty-state semantics,
        issue #424 — the portfolio exists; the computation simply has no input).

        Args:
            portfolio_id: Portfolio UUID string.
            weights: Dict of ticker → portfolio weight.

        Returns:
            Dict with ``exposures`` and ``asset_exposures``. Both are empty
            dicts when no scores are available.
        """
        logger.debug("compute_factor_exposure portfolio_id=%s", portfolio_id)
        scores = self._fetch_latest_factor_scores(list(weights.keys()))
        asset_exposures = _build_asset_exposures(scores)
        exposures = _compute_weighted_exposures(weights, asset_exposures)
        return {"exposures": exposures, "asset_exposures": asset_exposures}

    def compute_concentration(
        self,
        weights: dict[str, float],
        n: int,
    ) -> dict[str, Any]:
        """Compute per-asset concentration breakdown and portfolio-level aggregates.

        Args:
            weights: Dict of ticker → portfolio weight.
            n: Top-N threshold for ``top_n_ratio`` computation.

        Returns:
            Dict with ``assets`` (list) and ``summary`` (hhi, effective_n, top_n_ratio).
        """
        logger.debug("compute_concentration n=%d assets=%d", n, len(weights))
        names = self._fetch_asset_names(list(weights.keys()))
        assets = _build_concentration_assets(weights, names)
        summary = _compute_concentration_summary(weights, n)
        return {"assets": assets, "summary": summary}

    def compute_liquidity(
        self,
        weights: dict[str, float],
        lookback_days: int,
        participation_rate: float,
    ) -> dict[str, Any]:
        """Compute ADDV-based liquidity metrics per asset.

        Args:
            weights: Dict of ticker → portfolio weight.
            lookback_days: Trading days window for ADDV computation.
            participation_rate: Daily participation rate for liquidation capacity.

        Returns:
            Dict with ``assets`` (list) and ``summary`` (weighted_avg_days_to_liquidate).
        """
        logger.debug(
            "compute_liquidity lookback=%d participation=%.2f assets=%d",
            lookback_days, participation_rate, len(weights),
        )
        names = self._fetch_asset_names(list(weights.keys()))
        assets = [
            _build_liquidity_asset(ticker, weight, names, self._fetch_addv(ticker, lookback_days), participation_rate)
            for ticker, weight in weights.items()
        ]
        summary = _compute_liquidity_summary(weights, assets)
        return {"assets": assets, "summary": summary}

    # ------------------------------------------------------------------
    # Private fetch helpers (can be patched in tests)
    # ------------------------------------------------------------------

    def _fetch_weighted_returns(
        self, weights: dict[str, float], lookback: int
    ) -> pd.Series:
        """Fetch close prices and compute portfolio-weighted daily returns.

        Tickers without any price data are silently skipped so a single missing
        instrument (e.g. newly delisted or not yet ingested) does not collapse
        the entire weighted-return series. Remaining assets keep their original
        weights; the missing asset contributes 0 to the weighted risk measure.
        """
        prices = self._fetch_prices(weights, lookback)
        available = _surviving_tickers(weights, prices)
        if not available:
            return pd.Series(dtype=float)

        prices = prices[available].dropna()
        returns = prices.pct_change().dropna()
        weight_array = np.array([weights[t] for t in returns.columns])
        weighted = returns.values @ weight_array
        return pd.Series(weighted, index=returns.index)

    def _fetch_prices(
        self, weights: dict[str, float], lookback: int
    ) -> pd.DataFrame:
        """Fetch close prices from price_history for the given tickers."""
        repo = YFinanceRepository(self._session)
        data: dict[str, dict] = {}
        for ticker in weights:
            instrument = repo.get_instrument_by_yfinance_ticker(ticker)
            if instrument is None:
                continue
            rows = repo.get_price_history(instrument.id, limit=lookback + 10)
            if rows:
                data[ticker] = {r.date: float(r.close) for r in rows if r.close}
        return pd.DataFrame(data).sort_index()

    def _fetch_latest_factor_scores(self, tickers: list[str]) -> list[Any]:
        """Fetch the most recent factor scores for each ticker."""
        repo = FactorRepository(self._session)
        today = datetime.date.today()
        ninety_days_ago = today - datetime.timedelta(days=90)
        return repo.get_scores_by_tickers_and_date_range(
            tickers,
            start_date=ninety_days_ago,
            end_date=today,
        )

    def _fetch_asset_names(self, tickers: list[str]) -> dict[str, str]:
        """Return {ticker: display_name} for each ticker; fallback to ticker if not found."""
        repo = YFinanceRepository(self._session)
        names: dict[str, str] = {}
        for ticker in tickers:
            instrument = repo.get_instrument_by_yfinance_ticker(ticker)
            if instrument is None:
                names[ticker] = ticker
            else:
                names[ticker] = instrument.name or instrument.short_name or ticker
        return names

    def _fetch_addv(self, ticker: str, lookback_days: int) -> float | None:
        """Return avg daily dollar volume (volume × close) over lookback_days; None if unavailable."""
        repo = YFinanceRepository(self._session)
        instrument = repo.get_instrument_by_yfinance_ticker(ticker)
        if instrument is None:
            return None
        rows = repo.get_price_history(instrument.id, limit=lookback_days)
        dollar_volumes = [
            float(r.volume) * float(r.close)
            for r in rows
            if r.volume is not None and r.close is not None
        ]
        if not dollar_volumes:
            return None
        return float(np.mean(dollar_volumes))


# ---------------------------------------------------------------------------
# Pure computation helpers (stateless, easily unit-testable)
# ---------------------------------------------------------------------------


def _assert_sufficient_data(returns: pd.Series, lookback: int) -> None:
    if returns.empty or len(returns) < 2:
        raise ValueError(
            f"Insufficient data: requested lookback={lookback} but only "
            f"{len(returns)} observations available."
        )


def _assert_at_least_two_assets(tickers: list[str]) -> None:
    if len(tickers) < 2:
        raise ValueError(
            f"At least 2 assets required for correlation; got {len(tickers)}."
        )


def _surviving_tickers(
    weights: dict[str, float],
    prices: pd.DataFrame,
) -> list[str]:
    """Return weight tickers that actually have a column of prices (preserving weight order)."""
    if prices.empty:
        return []
    return [t for t in weights if t in prices.columns]


def _historical_var_cvar(
    returns: pd.Series,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute empirical VaR and CVaR at three confidence levels."""
    losses = -returns.values
    var_dict: dict[str, float] = {}
    cvar_dict: dict[str, float] = {}
    for key, level in zip(_CONFIDENCE_KEYS, _CONFIDENCE_LEVELS):
        var_val = float(np.quantile(losses, level))
        cvar_val = float(losses[losses >= var_val].mean())
        var_dict[key] = round(var_val, 8)
        cvar_dict[key] = round(cvar_val, 8)
    return var_dict, cvar_dict


def _parametric_var_cvar(
    returns: pd.Series,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute normal-distribution VaR and CVaR at three confidence levels."""
    arr = returns.to_numpy(dtype=float)
    mu = float(arr.mean())
    sigma = float(arr.std())
    var_dict: dict[str, float] = {}
    cvar_dict: dict[str, float] = {}
    for key, level in zip(_CONFIDENCE_KEYS, _CONFIDENCE_LEVELS):
        z = norm.ppf(level)
        var_val = -(mu - z * sigma)
        cvar_val = -(mu - sigma * norm.pdf(z) / (1 - level))
        var_dict[key] = round(float(var_val), 8)
        cvar_dict[key] = round(float(cvar_val), 8)
    return var_dict, cvar_dict


def _compute_corr_matrix(prices: pd.DataFrame) -> np.ndarray:
    """Return Pearson correlation matrix as numpy array."""
    return prices.pct_change().dropna().corr().values


def _cluster_and_reorder(
    tickers: list[str], corr_matrix: np.ndarray
) -> tuple[list[str], np.ndarray, list[int]]:
    """Hierarchically cluster assets and reorder matrix by cluster leaves."""
    distance = np.sqrt(np.maximum(0.0, (1 - corr_matrix) / 2))
    np.fill_diagonal(distance, 0.0)
    condensed = distance[np.triu_indices(len(tickers), k=1)]
    linkage_matrix = linkage(condensed, method="ward")
    leaf_order = leaves_list(linkage_matrix)
    cluster_ids = fcluster(linkage_matrix, t=2, criterion="maxclust")

    ordered_assets = [tickers[i] for i in leaf_order]
    reordered = corr_matrix[np.ix_(leaf_order, leaf_order)]
    reordered_labels = [int(cluster_ids[i]) for i in leaf_order]

    return ordered_assets, reordered, reordered_labels


def _build_asset_exposures(scores: list[Any]) -> dict[str, dict[str, float]]:
    """Build per-asset factor exposure dict from factor score rows."""
    exposures: dict[str, dict[str, float]] = {}
    for score in scores:
        ticker = score.ticker
        factor = score.factor_type
        raw = score.standardized_score
        if raw is None:
            continue
        exposures.setdefault(ticker, {})[factor] = float(raw)
    return exposures


def _compute_weighted_exposures(
    weights: dict[str, float],
    asset_exposures: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute portfolio-weighted exposure per factor type."""
    portfolio_exp: dict[str, float] = {}
    for ticker, weight in weights.items():
        for factor, score in asset_exposures.get(ticker, {}).items():
            portfolio_exp[factor] = portfolio_exp.get(factor, 0.0) + weight * score
    return {k: round(v, 8) for k, v in portfolio_exp.items()}


# ---------------------------------------------------------------------------
# Concentration helpers (issue #369)
# ---------------------------------------------------------------------------


def _build_concentration_assets(
    weights: dict[str, float],
    names: dict[str, str],
) -> list[dict[str, Any]]:
    """Build per-asset concentration list sorted by descending weight."""
    entries = [
        {"ticker": t, "name": names.get(t, t), "weight": w}
        for t, w in weights.items()
    ]
    return sorted(entries, key=lambda x: x["weight"], reverse=True)


def _compute_hhi(weights: dict[str, float]) -> float:
    """Herfindahl-Hirschman Index = Σ(wᵢ²)."""
    return sum(w**2 for w in weights.values())


def _compute_top_n_ratio(weights: dict[str, float], n: int) -> float:
    """Sum of top-N weights sorted descending."""
    sorted_weights = sorted(weights.values(), reverse=True)
    return sum(sorted_weights[:n])


def _compute_concentration_summary(
    weights: dict[str, float],
    n: int,
) -> dict[str, float]:
    """Return hhi, effective_n, and top_n_ratio for given weights."""
    if not weights:
        return {"hhi": 0.0, "effective_n": 0.0, "top_n_ratio": 0.0}
    hhi = _compute_hhi(weights)
    effective_n = 1.0 / hhi if hhi > 0 else 0.0
    top_n_ratio = _compute_top_n_ratio(weights, n)
    return {"hhi": round(hhi, 8), "effective_n": round(effective_n, 8), "top_n_ratio": round(top_n_ratio, 8)}


# ---------------------------------------------------------------------------
# Liquidity helpers (issue #369)
# ---------------------------------------------------------------------------

_MARKET_IMPACT_CONSTANT = 0.0005


def _compute_days_to_liquidate(
    weight: float,
    addv: float,
    participation_rate: float,
) -> float:
    """days = weight / (ADDV × participation_rate)."""
    return weight / (addv * participation_rate)


def _build_liquidity_asset(
    ticker: str,
    weight: float,
    names: dict[str, str],
    addv: float | None,
    participation_rate: float,
) -> dict[str, Any]:
    """Build a single liquidity asset dict; null fields when ADDV is unavailable."""
    name = names.get(ticker, ticker)
    if addv is None or addv <= 0:
        return {
            "ticker": ticker,
            "name": name,
            "weight": weight,
            "avg_daily_volume": None,
            "days_to_liquidate": None,
            "liquidity_cost": None,
        }
    days = _compute_days_to_liquidate(weight, addv, participation_rate)
    return {
        "ticker": ticker,
        "name": name,
        "weight": weight,
        "avg_daily_volume": addv,
        "days_to_liquidate": days,
        "liquidity_cost": days * _MARKET_IMPACT_CONSTANT,
    }


def _compute_liquidity_summary(
    weights: dict[str, float],
    assets: list[dict[str, Any]],
) -> dict[str, float]:
    """Weighted-average days to liquidate, excluding assets with null ADDV."""
    valid = [(a["weight"], a["days_to_liquidate"]) for a in assets if a["days_to_liquidate"] is not None]
    if not valid:
        return {"weighted_avg_days_to_liquidate": 0.0}
    total_weight = sum(w for w, _ in valid)
    if total_weight == 0:
        return {"weighted_avg_days_to_liquidate": 0.0}
    wavg = sum(w * d for w, d in valid) / total_weight
    return {"weighted_avg_days_to_liquidate": wavg}
