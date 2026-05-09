# Hierarchical Clustering

## When to use

Hierarchical clustering groups assets into a dendrogram so that
HRP / HERC / NCO / SchurComplementary can allocate within and across
clusters instead of inverting the full covariance matrix. Use it
whenever covariance noise is the dominant source of optimizer
instability — small samples, long histories with regime shifts, or
universes with strong sector blocks.

The full skfolio reference for hierarchical clustering and linkage
choice is in
`~/.claude/skills/skfolio/references/distance_clustering.md`.

## API surface

`HierarchicalClusteringConfig` wraps
`skfolio.cluster.HierarchicalClustering` with three primitives:

| Field | Default | Purpose |
|-------|---------|---------|
| `linkage_method` | `LinkageMethodType.WARD` | Agglomerative join rule. |
| `max_clusters` | `None` | Cap on the number of clusters. ``None`` lets skfolio choose via `compute_optimal_n_clusters`. |
| `min_cluster_size` | `1` | Reserved field — skfolio 0.20.1 ignores it. |

`LinkageMethodType` mirrors `skfolio.cluster.LinkageMethod` exactly:
seven members `SINGLE`, `COMPLETE`, `AVERAGE`, `WEIGHTED`, `CENTROID`,
`MEDIAN`, `WARD`.

## Composition pattern

The Config feeds hierarchical optimizers via the `clustering_config`
field on `HRPConfig` / `HERCConfig` / `NCOConfig` /
`SchurComplementaryConfig`. The factory builds a fresh
`HierarchicalClustering` instance per call, keeping the Config
serialisable.

```python
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

from optimizer.cluster import HierarchicalClusteringConfig, LinkageMethodType
from optimizer.distance import DistanceConfig
from optimizer.optimization import HRPConfig, build_hrp


prices = load_sp500_dataset()
returns = prices_to_returns(prices)

cluster_cfg = HierarchicalClusteringConfig(
    linkage_method=LinkageMethodType.WARD,
    max_clusters=4,
)
cfg = HRPConfig(
    distance_config=DistanceConfig.for_pearson(),
    clustering_config=cluster_cfg,
)
optimizer = build_hrp(cfg)
optimizer.fit(returns)
print(optimizer.weights_)
```

A complete runnable script is at `examples/hrp_clustering.py`.

## Choosing a linkage method

* `WARD` — minimises within-cluster variance; default and usually a
  safe choice on equity universes.
* `SINGLE` — chaining; classic HRP literature default.
* `AVERAGE` / `COMPLETE` — middle-ground; tighter than single, looser
  than Ward.
* `WEIGHTED` / `CENTROID` / `MEDIAN` — niche; document why if you
  reach for them.

## Direct estimator access

The factory also returns the unfitted estimator if you need it
standalone (for debugging or custom pipelines):

```python
from optimizer.cluster import (
    HierarchicalClusteringConfig,
    LinkageMethodType,
    build_hierarchical_clustering,
)
from optimizer.distance import DistanceConfig, build_distance


distance = build_distance(DistanceConfig.for_pearson())
distance.fit(returns)

clustering = build_hierarchical_clustering(
    HierarchicalClusteringConfig(linkage_method=LinkageMethodType.SINGLE)
)
clustering.fit(distance.distance_)
print(clustering.labels_)
```

## See also

- [Distance](distance.md) — supplies the matrix that clustering
  consumes.
- skfolio reference:
  `~/.claude/skills/skfolio/references/distance_clustering.md`.
