"""Pull 5y daily closes for 30 tickers, build a covariance matrix."""

import yfinance as yf

from skfolio.moments import LedoitWolf
from skfolio.preprocessing import prices_to_returns

# Swap for your own 30 tickers.
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "UNH", "HD", "MA", "XOM", "CVX", "KO", "PEP", "ABBV",
    "MRK", "COST", "AVGO", "ADBE", "CRM", "NFLX", "INTC", "CSCO", "MCD", "NKE",
]

# auto_adjust=True (default) → "Close" is split/div-adjusted. actions=False by default.
data = yf.download(TICKERS, period="5y", interval="1d", auto_adjust=True)

# List of tickers → multi-level columns. Grab adjusted close, keep ticker order.
closes = data["Close"][TICKERS]

# Linear returns (skfolio rule — never log returns). Drops leading NaN row.
X = prices_to_returns(closes)

# Covariance: LedoitWolf shrinkage (robust). Swap for EmpiricalCovariance if you want raw sample.
cov = LedoitWolf().fit(X)
cov_matrix = cov.covariance_  # numpy (N, N), ticker order = X.columns

print(f"closes shape: {closes.shape}")   # ~(1257, 30)
print(f"returns shape: {X.shape}")
print(f"cov matrix shape: {cov_matrix.shape}")  # (30, 30)
print(X.columns.tolist())
