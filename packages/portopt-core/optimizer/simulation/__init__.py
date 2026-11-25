"""Monte Carlo price paths and a Kelly-criterion simulator.

Three tools for looking at the range of outcomes a strategy or an asset could have
produced, rather than the single history that happened:

* geometric Brownian motion paths for one asset or for several correlated assets,
  and a maximum-likelihood fit of its drift and volatility to a price series;
* bootstrap wealth paths that resample the observed returns, in single returns or
  in blocks that keep short-range autocorrelation, and a summary of the
  distribution of the final values and drawdowns;
* the Kelly fraction, from the normal approximation or from observed returns, and a
  simulation of how wealth grows at fractions around it.

Simulated paths are a model of what could happen, not a forecast: the geometric
Brownian motion assumes constant volatility and normal log returns, the bootstrap
assumes the sample is representative of the future.
"""

from optimizer.simulation._bootstrap import (
    PathSummary,
    simulate_bootstrap,
    summarize_paths,
)
from optimizer.simulation._config import BootstrapConfig, GbmConfig
from optimizer.simulation._gbm import (
    GbmFit,
    calibrate_gbm,
    simulate_correlated_gbm,
    simulate_gbm,
)
from optimizer.simulation._kelly import (
    kelly_fraction_empirical,
    kelly_fraction_gaussian,
    simulate_kelly_growth,
)

__all__ = [
    "BootstrapConfig",
    "GbmConfig",
    "GbmFit",
    "PathSummary",
    "calibrate_gbm",
    "kelly_fraction_empirical",
    "kelly_fraction_gaussian",
    "simulate_bootstrap",
    "simulate_correlated_gbm",
    "simulate_gbm",
    "simulate_kelly_growth",
    "summarize_paths",
]
