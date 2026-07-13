# Distance, Clustering, Pre-Selection, Uncertainty Sets

## Distance Estimators

Produce `codependence_` and `distance_` after `fit(X)`. Plug into hierarchical optimizers (HRP, HERC, NCO, SchurComplementary).

```python
from skfolio.distance import (
    PearsonDistance, KendallDistance, SpearmanDistance,
    CovarianceDistance, DistanceCorrelation, MutualInformation,
)
```

| Estimator | Measures |
|---|---|
| `PearsonDistance` | Linear correlation |
| `KendallDistance` | Rank correlation (Kendall tau) |
| `SpearmanDistance` | Rank correlation (Spearman rho) |
| `CovarianceDistance` | Covariance-based |
| `DistanceCorrelation` | Non-linear dependence |
| `MutualInformation` | Information-theoretic |

## HierarchicalClustering

```python
from skfolio.cluster import HierarchicalClustering, LinkageMethod

clustering = HierarchicalClustering(
    linkage_method=LinkageMethod.WARD,
    max_clusters=None,     # int to fix cluster count
)
```

## Pre-Selection Transformers

scikit-learn transformers that filter assets before optimization. Compose in a `Pipeline`.

```python
from skfolio.pre_selection import (
    DropCorrelated, DropZeroVariance,
    SelectKExtremes, SelectNonDominated,
    SelectComplete, SelectNonExpiring,
)
```

| Transformer | Purpose | Key parameter |
|---|---|---|
| `DropCorrelated` | Remove highly correlated assets | `threshold=0.95` |
| `DropZeroVariance` | Remove near-zero variance | — |
| `SelectKExtremes` | Top/bottom k performers | `k`, `highest=True` |
| `SelectNonDominated` | Pareto-optimal assets | — |
| `SelectComplete` | Assets with full history | — |
| `SelectNonExpiring` | Exclude soon-expiring | `expiration_lookahead` |

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("pre", DropCorrelated(threshold=0.90)),
    ("opt", MeanRisk()),
])
pipe.fit(X)
```

## Uncertainty Sets

Used with `MeanRisk` for robust optimization — the optimizer minimizes the worst case over the set.

```python
from skfolio.uncertainty_set import (
    EmpiricalMuUncertaintySet,
    EmpiricalCovarianceUncertaintySet,
    BootstrapMuUncertaintySet,
    BootstrapCovarianceUncertaintySet,
)

model = MeanRisk(
    mu_uncertainty_set_estimator=BootstrapMuUncertaintySet(confidence_level=0.95),
    covariance_uncertainty_set_estimator=BootstrapCovarianceUncertaintySet(confidence_level=0.95),
)
```

Use `Empirical*UncertaintySet` for analytical (Gaussian-based) sets and `Bootstrap*UncertaintySet` when data is non-normal.
