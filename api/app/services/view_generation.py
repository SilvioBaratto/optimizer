"""LLM-driven Black-Litterman view generation service.

Workflow:
  1. Fetch per-asset factor data from the database (TickerProfile + PriceHistory).
  2. Call the BAML ``GenerateViews`` function.
  3. Convert the structured output to P (n_views × n_assets), Q (n_views,),
     and Idzorek alphas compatible with ``build_black_litterman()``.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta

import numpy as np
from baml_client import b
from baml_client.types import AssetFactorData, AssetView, ViewOutput
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.universe import Instrument
from app.models.yfinance_data import PriceHistory, TickerProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result dataclass (plain dict for JSON-serialisability)
# ---------------------------------------------------------------------------


class GeneratedViews:
    """Container for the BL-ready view components returned by the service."""

    __slots__ = (
        "P",
        "Q",
        "asset_views",
        "idzorek_alphas",
        "rationale",
        "view_confidences",
        "view_strings",
    )

    def __init__(
        self,
        view_strings: list[str],
        P: np.ndarray,
        Q: np.ndarray,
        view_confidences: list[float],
        idzorek_alphas: dict[str, float],
        asset_views: list[AssetView],
        rationale: str,
    ) -> None:
        self.view_strings = view_strings  # skfolio-compatible strings
        self.P = P  # (n_views, n_assets) float64
        self.Q = Q  # (n_views,) float64
        self.view_confidences = view_confidences  # ordered by view
        self.idzorek_alphas = idzorek_alphas  # ticker → alpha_k
        self.asset_views = asset_views  # raw BAML output
        self.rationale = rationale


# ---------------------------------------------------------------------------
# DB fetch helpers
# ---------------------------------------------------------------------------


def _get_instrument_by_ticker(session: Session, ticker: str) -> Instrument | None:
    return session.execute(
        select(Instrument).where(Instrument.ticker == ticker)
    ).scalar_one_or_none()


def _fetch_price_history_closes(
    session: Session,
    instrument_id,
    lookback_days: int = 260,
) -> list[float]:
    """Return close prices in ascending date order."""
    cutoff = date.today() - timedelta(days=lookback_days)
    rows = (
        session.execute(
            select(PriceHistory.close)
            .where(
                PriceHistory.instrument_id == instrument_id,
                PriceHistory.date >= cutoff,
                PriceHistory.close.isnot(None),
            )
            .order_by(PriceHistory.date.asc())
        )
        .scalars()
        .all()
    )
    return [float(c) for c in rows if c is not None]


def _compute_momentum(
    closes: list[float], short_months: int = 1, long_months: int = 12
) -> tuple[float | None, float | None]:
    """Return (mom_12_1m, mom_1m) as decimals.  Returns (None, None) on insufficient data."""
    days_per_month = 21
    short_n = short_months * days_per_month
    long_n = long_months * days_per_month

    if len(closes) < long_n + 1:
        return None, None

    p_now = closes[-1]
    p_1m = closes[-(short_n + 1)]
    p_12m = closes[-(long_n + 1)]

    mom_1m = (p_now / p_1m - 1.0) if p_1m else None
    # 12-1 mom: performance over [12m ago, 1m ago]
    mom_12_1m = (p_1m / p_12m - 1.0) if p_12m else None

    return mom_12_1m, mom_1m


def _compute_rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    recent = closes[-(period + 1) :]
    gains, losses = [], []
    for i in range(1, len(recent)):
        diff = recent[i] - recent[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return None


def _build_factor_data(
    session: Session,
    ticker: str,
) -> AssetFactorData | None:
    """Assemble per-asset factor data from DB for a single ticker."""
    instrument = _get_instrument_by_ticker(session, ticker)
    if instrument is None:
        logger.warning("Ticker %s not found in instruments table", ticker)
        return None

    profile: TickerProfile | None = session.execute(
        select(TickerProfile).where(TickerProfile.instrument_id == instrument.id)
    ).scalar_one_or_none()

    closes = _fetch_price_history_closes(session, instrument.id)
    mom_12_1m, mom_1m = _compute_momentum(closes)
    rsi = _compute_rsi(closes)

    # 52-week high/low from price history
    pct_from_high: float | None = None
    pct_from_low: float | None = None
    if closes:
        recent_closes = closes[-252:] if len(closes) >= 252 else closes
        w52_high = max(recent_closes)
        w52_low = min(recent_closes)
        current = closes[-1]
        pct_from_high = (current / w52_high - 1.0) if w52_high else None
        pct_from_low = (current / w52_low - 1.0) if w52_low else None

    # Analyst target upside from profile
    target_upside: float | None = None
    if profile and profile.target_mean_price and profile.current_price:
        target_upside = _safe_float(
            (profile.target_mean_price - profile.current_price) / profile.current_price
        )

    return AssetFactorData(
        ticker=ticker,
        trailing_pe=_safe_float(profile.trailing_pe) if profile else None,
        price_to_book=_safe_float(profile.price_to_book) if profile else None,
        ev_to_ebitda=_safe_float(profile.enterprise_to_ebitda) if profile else None,
        momentum_12_1m=mom_12_1m,
        momentum_1m=mom_1m,
        rsi_14=rsi,
        return_on_equity=_safe_float(profile.return_on_equity) if profile else None,
        debt_to_equity=_safe_float(profile.debt_to_equity) if profile else None,
        profit_margins=_safe_float(profile.profit_margins) if profile else None,
        revenue_growth_yoy=_safe_float(profile.revenue_growth) if profile else None,
        earnings_growth_yoy=_safe_float(profile.earnings_growth) if profile else None,
        pct_from_52w_high=pct_from_high,
        pct_from_52w_low=pct_from_low,
        recommendation_mean=_safe_float(profile.recommendation_mean)
        if profile
        else None,
        target_upside=target_upside,
        analyst_count=profile.number_of_analyst_opinions if profile else None,
    )


# ---------------------------------------------------------------------------
# View conversion helpers
# ---------------------------------------------------------------------------


def _views_to_arrays(
    views: list[AssetView],
    tickers: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray, list[float]]:
    """Convert BAML AssetView list to BL-compatible arrays.

    Returns
    -------
    view_strings : list[str]
        skfolio-compatible view expressions (e.g. ``"AAPL == 0.02"``).
    P : np.ndarray, shape (n_views, n_assets)
        Pick matrix.  Row i is all zeros except ``+1`` at the column for the
        view's asset (absolute view).
    Q : np.ndarray, shape (n_views,)
        Expected excess returns in decimal (bps / 10_000).
    view_confidences : list[float]
        Idzorek α_k per view, ordered identically to views.
    """
    ticker_index = {t: i for i, t in enumerate(tickers)}
    n_assets = len(tickers)

    view_strings: list[str] = []
    rows: list[list[float]] = []
    q_vals: list[float] = []
    confidences: list[float] = []

    for view in views:
        if view.asset not in ticker_index:
            logger.warning(
                "View asset %s not in requested tickers — skipped", view.asset
            )
            continue

        signed_return = view.direction * view.magnitude_bps / 10_000.0
        view_strings.append(f"{view.asset} == {signed_return:.6f}")

        row = [0.0] * n_assets
        row[ticker_index[view.asset]] = 1.0
        rows.append(row)
        q_vals.append(signed_return)
        confidences.append(float(view.confidence))

    P = np.array(rows, dtype=np.float64).reshape(-1, n_assets)
    Q = np.array(q_vals, dtype=np.float64)

    return view_strings, P, Q, confidences


def _validate_idzorek_alphas(
    alphas: dict[str, float],
    view_assets: list[str],
) -> dict[str, float]:
    """Clamp all alphas to (0, 1) and fill missing keys with 0.5."""
    result: dict[str, float] = {}
    for asset in view_assets:
        raw = alphas.get(asset, 0.5)
        result[asset] = max(1e-6, min(1.0 - 1e-6, float(raw)))
    return result


# ---------------------------------------------------------------------------
# Main service entry points
# ---------------------------------------------------------------------------


def fetch_factor_data(
    session: Session,
    tickers: list[str],
) -> list[AssetFactorData]:
    """Fetch per-asset factor data from the DB for each ticker.

    Assets with no DB record are silently skipped.
    """
    result: list[AssetFactorData] = []
    for ticker in tickers:
        fd = _build_factor_data(session, ticker)
        if fd is not None:
            result.append(fd)
    return result


def adjust_view_confidences(
    views: list[AssetView],
    sentiment_map: dict[str, float],
) -> list[AssetView]:
    """Return a new list of views with Idzorek confidence adjusted by sentiment.

    For each view, the sentiment signal for that asset is looked up in
    ``sentiment_map`` (ticker → signed decayed signal from
    :func:`app.services.sentiment.compute_sentiment_signal`).  The
    confidence is then adjusted via
    :func:`app.services.sentiment.adjust_idzorek_alpha` and clamped to
    ``(0.01, 0.99)``.  Assets not present in ``sentiment_map`` are left
    unchanged.

    Args:
        views: BAML-generated :class:`AssetView` list.
        sentiment_map: Mapping of ticker → sentiment signal (signed float).
            Zero or absent → confidence unchanged.

    Returns:
        New list of :class:`AssetView` instances with updated confidence.
    """
    from app.services.sentiment import adjust_idzorek_alpha  # avoid circular import

    adjusted: list[AssetView] = []
    for view in views:
        signal = sentiment_map.get(view.asset, 0.0)
        new_conf = adjust_idzorek_alpha(view.confidence, signal, view.direction)
        adjusted.append(view.model_copy(update={"confidence": new_conf}))
    return adjusted


def generate_views(
    tickers: list[str],
    factor_data: list[AssetFactorData],
) -> GeneratedViews:
    """Call the BAML LLM and convert the output to BL-ready arrays.

    Args:
        tickers: Full ordered list of asset tickers in the universe.
        factor_data: Per-asset factor data (from :func:`fetch_factor_data`).

    Returns:
        :class:`GeneratedViews` with P, Q, view_confidences, view strings,
        idzorek_alphas, and rationale.

    Raises:
        ValueError: If ``factor_data`` is empty.
    """
    if not factor_data:
        raise ValueError("factor_data is empty — no assets to generate views for.")

    raw: ViewOutput = b.GenerateViews(assets=factor_data)

    # Filter to views on requested tickers only
    valid_views = [v for v in raw.views if v.asset in set(tickers)]

    view_strings, P, Q, view_confidences = _views_to_arrays(valid_views, tickers)

    view_assets = [v.asset for v in valid_views]
    idzorek_alphas = _validate_idzorek_alphas(raw.idzorek_alphas, view_assets)

    return GeneratedViews(
        view_strings=view_strings,
        P=P,
        Q=Q,
        view_confidences=view_confidences,
        idzorek_alphas=idzorek_alphas,
        asset_views=valid_views,
        rationale=raw.rationale,
    )
