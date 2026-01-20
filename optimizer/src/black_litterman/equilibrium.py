#!/usr/bin/env python3
import logging
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

from optimizer.database.database import database_manager
from optimizer.database.models.universe import Instrument

logger = logging.getLogger(__name__)


def fetch_market_caps_from_db(
    tickers: Optional[list] = None, exchange_name: Optional[str] = None
) -> pd.Series:
    """
    Fetch real-time market capitalizations from yfinance.
    """
    from src.yfinance import YFinanceClient

    # Get yfinance client
    client = YFinanceClient.get_instance()

    # If no tickers provided, fetch all active instruments from DB
    if tickers is None:
        with database_manager.get_session() as session:
            query = session.query(Instrument).filter(Instrument.is_active == True)

            if exchange_name:
                from optimizer.database.models.universe import Exchange

                query = query.join(Exchange).filter(Exchange.exchange_name == exchange_name)

            instruments = query.all()
            tickers = [inst.yfinance_ticker for inst in instruments if inst.yfinance_ticker]

    if not tickers:
        logger.warning("No tickers provided for market cap fetch")
        return pd.Series(dtype=float)

    logger.info(f"Fetching market caps for {len(tickers)} tickers from yfinance")

    # Fetch market caps from yfinance
    market_caps = {}
    failed_tickers = []

    for ticker in tickers:
        try:
            info = client.fetch_info(ticker)
            if info and "marketCap" in info and info["marketCap"]:
                market_caps[ticker] = float(info["marketCap"])
            else:
                failed_tickers.append(ticker)
        except Exception as e:
            logger.debug(f"Failed to fetch market cap for {ticker}: {e}")
            failed_tickers.append(ticker)

    logger.info(f"✓ Fetched market caps for {len(market_caps)}/{len(tickers)} tickers")

    if failed_tickers and len(failed_tickers) < 10:
        logger.warning(f"Missing market caps for: {failed_tickers}")
    elif failed_tickers:
        logger.warning(f"Missing market caps for {len(failed_tickers)} tickers")

    if not market_caps:
        logger.error("No market caps fetched from yfinance - cannot calculate equilibrium")
        return pd.Series(dtype=float)

    return pd.Series(market_caps)


def estimate_risk_aversion(market_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Estimate market-implied risk aversion coefficient (δ).
    """
    # Annualize returns and volatility
    annual_return = float(market_returns.mean()) * 252
    annual_variance = float(market_returns.var()) * 252  # type: ignore

    if annual_variance == 0:
        logger.warning("Market variance is zero, using default delta=2.5")
        return 2.5

    delta = float((annual_return - risk_free_rate) / annual_variance)

    # Sanity check: delta should be positive and reasonable (1-4 range)
    if delta < 0 or delta > 10:
        logger.warning(
            f"Estimated delta={delta:.2f} is outside normal range [1, 4]. "
            f"Using default delta=2.5"
        )
        delta = 2.5

    logger.info(
        f"Estimated risk aversion: δ={delta:.2f} "
        f"(market return={annual_return:.2%}, variance={annual_variance:.3f})"
    )

    return delta


def calculate_equilibrium_prior(
    market_caps: Union[Dict[str, float], pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    risk_aversion: float,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    Calculate market-implied equilibrium returns (π).
    """
    # Convert market caps to Series if dict
    if isinstance(market_caps, dict):
        market_caps = pd.Series(market_caps)

    # Convert covariance to DataFrame if ndarray
    if isinstance(cov_matrix, np.ndarray):
        if hasattr(market_caps, "index"):
            tickers = market_caps.index
        else:
            raise ValueError("If cov_matrix is ndarray, market_caps must have index")
        cov_matrix = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)

    # Align tickers
    tickers = market_caps.index
    cov_tickers = cov_matrix.index

    # Check alignment
    if not all(ticker in cov_tickers for ticker in tickers):
        missing = [t for t in tickers if t not in cov_tickers]
        logger.warning(f"Some tickers in market_caps missing from cov_matrix: {missing}")

        # Filter to common tickers
        common_tickers = tickers.intersection(cov_tickers)
        market_caps = market_caps[common_tickers]
        cov_matrix = cov_matrix.loc[common_tickers, common_tickers]
        logger.info(f"Using {len(common_tickers)} common tickers")

    # Calculate market weights
    w_mkt = market_caps / market_caps.sum()

    # Calculate equilibrium returns
    # π = δ * Σ * w_mkt
    pi = risk_aversion * cov_matrix.dot(w_mkt)

    # Add risk-free rate (π is excess returns, so add rf for absolute returns)
    pi = pi + risk_free_rate

    logger.info(
        f"Calculated equilibrium prior: "
        f"mean={pi.mean():.2%}, min={pi.min():.2%}, max={pi.max():.2%}"
    )

    return pi


def adjust_risk_aversion_for_regime(
    base_risk_aversion: float, regime: str, recession_risk: Optional[float] = None
) -> float:
    """
    Adjust risk aversion coefficient based on macro regime.
    """
    # Chapter 4 (lines 182-188): Use ABSOLUTE regime-based values
    regime_delta = {
        "EARLY_CYCLE": 2.0,  # Risk-on: growth opportunities
        "MID_CYCLE": 2.5,  # Neutral: balanced positioning
        "LATE_CYCLE": 3.5,  # Risk-off: defensive tilt
        "RECESSION": 5.0,  # Defensive: capital preservation
        "UNCERTAIN": 3.0,  # Moderately defensive when regime unclear
    }

    # Use regime-specific delta (REPLACES market-implied, per theory)
    adjusted_delta = regime_delta.get(regime, base_risk_aversion)

    # Fine-tune based on recession risk probability
    if recession_risk is not None:
        # Gradual adjustment: increase delta as recession risk rises
        # At 25% risk: +0.25, at 50% risk: +0.50, at 75% risk: +0.75
        if recession_risk > 0.15:
            extra = (recession_risk - 0.15) * 1.0
            adjusted_delta += extra

    recession_risk_str = f"{recession_risk:.1%}" if recession_risk is not None else "N/A"
    logger.info(
        f"Risk aversion adjusted for regime: "
        f"market-implied={base_risk_aversion:.2f} -> regime-adjusted={adjusted_delta:.2f} "
        f"(regime={regime}, recession_risk={recession_risk_str})"
    )

    return adjusted_delta


def calculate_implied_returns_from_weights(
    weights: Union[Dict[str, float], pd.Series],
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    risk_aversion: float,
) -> pd.Series:
    """
    Reverse optimization: calculate implied returns given portfolio weights.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)

    if isinstance(cov_matrix, np.ndarray):
        tickers = weights.index
        cov_matrix = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)

    # Align
    common_tickers = weights.index.intersection(cov_matrix.index)
    weights = weights[common_tickers]
    cov_matrix = cov_matrix.loc[common_tickers, common_tickers]

    # Reverse optimization
    implied_returns = risk_aversion * cov_matrix.dot(weights)

    logger.info(f"Calculated implied returns from weights: mean={implied_returns.mean():.2%}")

    return implied_returns


if __name__ == "__main__":
    # Example usage
    import logging

    logging.basicConfig(level=logging.INFO)

    # Fetch market caps from database
    market_caps = fetch_market_caps_from_db()

    if not market_caps.empty:
        print(f"\nFetched market caps for {len(market_caps)} tickers")
        print(f"Top 5 by market cap:")
        print(market_caps.nlargest(5))

        # Example: calculate equilibrium prior
        # (Would need actual covariance matrix in practice)
        print("\nTo calculate equilibrium prior, you need:")
        print("1. Covariance matrix (from risk_models.ledoit_wolf_shrinkage)")
        print("2. Risk aversion (from estimate_risk_aversion or regime adjustment)")
        print("3. Then call: pi = calculate_equilibrium_prior(market_caps, cov, delta)")
