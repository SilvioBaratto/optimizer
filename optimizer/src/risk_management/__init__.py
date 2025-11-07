"""
Risk Management Module - Stock Selection for Portfolio Optimization
====================================================================

This module provides institutional-grade stock selection tools for concentrated
momentum strategies. Portfolio weighting is handled separately by Black-Litterman
optimizer.

Key Components:
--------------

1. **ConcentratedPortfolioBuilder**: Stock selector for portfolio optimization
   - Fetches LARGE_GAIN momentum signals
   - Applies quality filters (Sharpe, volatility, drawdown)
   - Enforces correlation constraints (max 0.7 pairwise)
   - Applies sector diversification (max 15% per sector)
   - Applies country allocation (max 60% per country)
   - Returns 20 selected stocks (NO WEIGHTS - that's Black-Litterman's job)

2. **QualityFilter**: Quality screens for momentum signals
   - Sharpe ratio filter (≥0.5)
   - Volatility filter (≤40%)
   - Drawdown filter (≥-30%)
   - Price filter (≥$5)

3. **CorrelationAnalyzer**: Correlation clustering and diversification
   - Builds correlation matrices
   - Identifies correlation clusters
   - Enforces maximum pairwise correlation (0.7)
   - Limits positions per cluster (2 stocks max)

4. **SectorAllocator**: Sector diversification constraints
   - Maximum 15% per sector
   - Minimum 8 sectors required
   - Minimum 10% in defensive sectors
   - Maximum 2 stocks per industry

5. **PortfolioAnalytics**: Stock ranking utilities
   - Composite scoring for stock ranking
   - Country mapping from exchange names

Usage Example:
-------------

```python
from src.risk_management import ConcentratedPortfolioBuilder

# Select 20 stocks for Black-Litterman optimization
builder = ConcentratedPortfolioBuilder(
    target_positions=20,
    max_sector_weight=0.15,
    max_country_weight=0.60,
    max_correlation=0.7,
    capital=1500.0  # Trading212 constraint
)

# Execute selection pipeline (returns stocks WITHOUT weights)
selected_stocks = builder.build_portfolio()

# Pass to Black-Litterman optimizer for weight calculation
# (Black-Litterman handles ALL portfolio optimization)
```

Expected Output:
---------------

From ~500 LARGE_GAIN signals → 20 selected stocks with:
- Quality filtering (Sharpe≥0.5, Vol≤40%, Drawdown≥-30%)
- Sector diversification (8+ sectors, max 3 stocks per sector)
- Geographic diversity (max 60% per country, 40% non-US target)
- Correlation constraints (max 0.7 pairwise, max 2 per cluster)
- Affordability (Trading212: €1-€75 per stock for €1500 capital)

Note: Stock weighting is handled by Black-Litterman optimizer, not here!

Author: Portfolio Optimization System
Version: 2.0.0 (Simplified - Stock Selection Only)
"""

from .concentrated_portfolio_builder import (
    ConcentratedPortfolioBuilder
)

from .quality_filter import (
    QualityFilter,
    QualityMetrics
)

from .correlation_analyzer import (
    CorrelationAnalyzer
)

from .sector_allocator import (
    SectorAllocator,
    DEFENSIVE_SECTORS,
    CYCLICAL_SECTORS,
    ALL_SECTORS
)

from .portfolio_analytics import (
    PortfolioAnalytics
)

__all__ = [
    # Main stock selector
    "ConcentratedPortfolioBuilder",

    # Quality filtering
    "QualityFilter",
    "QualityMetrics",

    # Correlation analysis
    "CorrelationAnalyzer",

    # Sector allocation
    "SectorAllocator",
    "DEFENSIVE_SECTORS",
    "CYCLICAL_SECTORS",
    "ALL_SECTORS",

    # Utilities
    "PortfolioAnalytics",
]

__version__ = "2.0.0"
