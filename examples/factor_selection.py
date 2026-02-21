"""Factor-based stock selection pipeline.

Uses ticker names from the skfolio S&P 500 dataset with synthetic
factor scores, then demonstrates standardization, composite scoring,
and stock selection.

Note: skfolio provides factor ETF *prices* (MTUM, QUAL, etc.), not
cross-sectional factor *scores*. Factor scores require fundamental
data (book value, ROE, etc.) which is not bundled with skfolio, so
we generate synthetic scores for the 20 S&P 500 tickers.
"""

import numpy as np
import pandas as pd
from skfolio.datasets import load_sp500_dataset

from optimizer.factors import (
    CompositeScoringConfig,
    SelectionConfig,
    StandardizationConfig,
    compute_composite_score,
    select_stocks,
    standardize_all_factors,
)

# --- Use real tickers from skfolio dataset ---
prices = load_sp500_dataset()
tickers = list(prices.columns)

# Assign sectors (approximate real GICS sectors)
sector_map = {
    "AAPL": "Tech",
    "AMD": "Tech",
    "MSFT": "Tech",
    "BAC": "Finance",
    "JPM": "Finance",
    "JNJ": "Health",
    "LLY": "Health",
    "MRK": "Health",
    "PFE": "Health",
    "UNH": "Health",
    "CVX": "Energy",
    "XOM": "Energy",
    "RRC": "Energy",
    "HD": "Consumer",
    "BBY": "Consumer",
    "KO": "Consumer",
    "PEP": "Consumer",
    "PG": "Consumer",
    "WMT": "Consumer",
    "GE": "Industrial",
}
sector_labels = pd.Series(sector_map)

# Simulate factor scores using actual FactorType enum values
rng = np.random.default_rng(42)
factor_names = [
    "book_to_price",
    "roe",
    "momentum_12_1",
    "volatility",
    "dividend_yield",
]
factor_data = rng.standard_normal((len(tickers), len(factor_names)))
raw_factors = pd.DataFrame(factor_data, index=tickers, columns=factor_names)

# --- Standardize ---
std_config = StandardizationConfig()
standardized, coverage = standardize_all_factors(
    raw_factors, std_config, sector_labels=sector_labels
)

# --- Composite score ---
score_config = CompositeScoringConfig()
composite = compute_composite_score(standardized, coverage, score_config)

print("Top 10 stocks by composite score:")
print(composite.sort_values(ascending=False).head(10).round(4))

# --- Select stocks ---
sel_config = SelectionConfig(target_count=10)
selected = select_stocks(composite, sel_config, sector_labels=sector_labels)

print(f"\nSelected {len(selected)} stocks:")
print(sorted(selected))
