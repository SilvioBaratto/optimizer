# Distance Estimators

## When to use

Distance estimators turn a return panel into a square ticker-by-ticker
distance matrix that hierarchical optimizers (HRP, HERC, NCO,
SchurComplementary) consume to grow their dendrograms. Pick the
estimator whose dependence assumption matches your data: linear
correlations for daily equity returns, rank correlations when fat
tails distort linearity, mutual information when the relationship is
non-monotone.

The full skfolio reference for distance choices and their statistical
properties is in
`~/.claude/skills/skfolio/references/distance_clustering.md`.

## Available estimators

The `DistanceEstimatorType` enum exposes six skfolio estimators:

| Member | skfolio class | Use it for |
|--------|---------------|------------|
| `PEARSON` | `PearsonDistance` | Default linear correlation distance. |
| `KENDALL` | `KendallDistance` | Rank-based; robust to outliers. |
| `SPEARMAN` | `SpearmanDistance` | Rank-based; broader use than Kendall. |
| `COVARIANCE` | `CovarianceDistance` | Volatility-aware metric. |
| `DISTANCE_CORRELATION` | `DistanceCorrelation` | Detects non-linear dependence. |
| `MUTUAL_INFORMATION` | `MutualInformation` | Non-parametric; needs `n_bins`. |

All instances expose `.fit(X)` and store a `(n_assets, n_assets)`
`distance_` matrix.

## Composition pattern

`DistanceConfig` is a frozen dataclass — pass it as the
`distance_config` field on hierarchical optimizer Configs (HRP, HERC,
NCO, SchurComplementary). The factory composes the estimator at build
time, so the Config stays serialisable.

```python
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

from optimizer.cluster import HierarchicalClusteringConfig, LinkageMethodType
from optimizer.distance import DistanceConfig
from optimizer.optimization import HRPConfig, build_hrp


prices = load_sp500_dataset()
returns = prices_to_returns(prices)

cfg = HRPConfig(
    distance_config=DistanceConfig.for_mutual_information(n_bins=10),
    clustering_config=HierarchicalClusteringConfig(
        linkage_method=LinkageMethodType.WARD
    ),
)
optimizer = build_hrp(cfg)
optimizer.fit(returns)
print(optimizer.weights_)
```

The full snippet lives in `examples/hrp_clustering.py`.

## Validation

`DistanceConfig.__post_init__` rejects MI-only fields (`n_bins`,
`bandwidth`) when the estimator is not `MUTUAL_INFORMATION`. The
`bandwidth` field is reserved for forward compatibility — skfolio
0.20.1 `MutualInformation` does not currently expose it.

## Presets

Each estimator has a matching `for_<name>` classmethod returning a
ready-to-use Config. Use them when you do not need to tune secondary
parameters:

```python
DistanceConfig.for_pearson()
DistanceConfig.for_kendall()
DistanceConfig.for_spearman()
DistanceConfig.for_covariance()
DistanceConfig.for_distance_correlation()
DistanceConfig.for_mutual_information(n_bins=10)
```

## See also

- [Cluster](cluster.md) — pair with a clustering estimator to drive
  HRP / HERC / NCO / SchurComplementary.
- skfolio reference:
  `~/.claude/skills/skfolio/references/distance_clustering.md`.
