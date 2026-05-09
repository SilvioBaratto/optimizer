"""Cross-sectional factor-score preprocessing pipeline.

Pipes raw factor signals through ``CSWinsorizer → CSGaussianRankScaler``
to produce calibrated cross-sectional scores. Uses synthetic factor
scores derived from the bundled SP500 dataset (per-period z-scored
returns).

Run:
    python examples/cs_preprocessing.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from sklearn.pipeline import Pipeline

from optimizer.preprocessing import CSGaussianRankScaler, CSWinsorizer


def _build_factor_scores(returns: pd.DataFrame) -> pd.DataFrame:
    """Use 21-day rolling Sharpe as a momentum-style raw factor."""
    rolling_mean = returns.rolling(21).mean()
    rolling_std = returns.rolling(21).std().replace(0.0, np.nan)
    return (rolling_mean / rolling_std).dropna(how="all")


def main() -> None:
    prices = load_sp500_dataset()
    returns = prices_to_returns(prices).tail(500)
    raw_scores = _build_factor_scores(returns)

    pipeline = Pipeline(
        steps=[
            ("winsorize", CSWinsorizer(low=0.05, high=0.95)),
            ("rank_scale", CSGaussianRankScaler()),
        ]
    )

    transformed = pipeline.fit_transform(raw_scores)
    transformed_df = pd.DataFrame(
        transformed,
        index=raw_scores.index,
        columns=raw_scores.columns,
    )

    print("Raw rolling-Sharpe factor (last 5 dates):")
    print(raw_scores.tail(5).round(4).to_string())
    print()
    print("After CSWinsorizer → CSGaussianRankScaler (last 5 dates):")
    print(transformed_df.tail(5).round(4).to_string())


if __name__ == "__main__":
    main()
