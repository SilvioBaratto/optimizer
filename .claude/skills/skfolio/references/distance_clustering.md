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

`fit()` takes a **distance matrix**, not raw returns — feed it a distance estimator's `distance_`.

```python
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.distance import PearsonDistance

dist = PearsonDistance().fit(X)
clustering = HierarchicalClustering(
    linkage_method=LinkageMethod.WARD,
    max_clusters=None,     # int to fix cluster count
)
clustering.fit(dist.distance_)     # → linkage_matrix_
```

## Pre-Selection Transformers

scikit-learn transformers that filter assets before optimization. Compose in a `Pipeline`. Set `sklearn.set_config(transform_output="pandas")` so column/ticker labels survive.

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
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(transform_output="pandas")
pipe = Pipeline([
    ("pre", DropCorrelated(threshold=0.90)),
    ("opt", MeanRisk()),
])
pipe.fit(X)
```

## Uncertainty Sets

Used with `MeanRisk` for robust optimization — the optimizer minimizes the worst case over the set. Fitted result in `uncertainty_set_`.

```python
from skfolio.uncertainty_set import (
    EmpiricalMuUncertaintySet,
    EmpiricalCovarianceUncertaintySet,
    BootstrapMuUncertaintySet,
    BootstrapCovarianceUncertaintySet,
    OrthogonalMuUncertaintySet,          # NEW 1.0
    OrthogonalCovarianceUncertaintySet,  # NEW 1.0
    CompactCovarianceUncertaintySet,     # NEW 1.0
    UncertaintySet,                      # dataclass
)

model = MeanRisk(
    mu_uncertainty_set_estimator=BootstrapMuUncertaintySet(confidence_level=0.95),
    covariance_uncertainty_set_estimator=EmpiricalCovarianceUncertaintySet(confidence_level=0.95),
)
```

| Estimator | Use |
|---|---|
| `EmpiricalMuUncertaintySet` / `EmpiricalCovarianceUncertaintySet` | Analytical (chi-squared) sets; size via `confidence_level` |
| `BootstrapMuUncertaintySet` / `BootstrapCovarianceUncertaintySet` | Non-normal data; size via `confidence_level` |
| `OrthogonalMuUncertaintySet` / `OrthogonalCovarianceUncertaintySet` | **Require a factor-model `prior_estimator`** (e.g. `CharacteristicsFactorModel`); confine uncertainty to the space orthogonal to the loading matrix. Covariance variant parameterized by `radius` (not `confidence_level`) |
| `CompactCovarianceUncertaintySet` | Reduced quadratic-form representation added directly to the variance term (avoids the lifted SDP) |

Higher `confidence_level` → wider set → larger worst-case penalty. Covariance uncertainty applies only when `risk_measure=RiskMeasure.VARIANCE` (or `max_variance` is set).

### ⚠️ 1.0 `UncertaintySet` dataclass renames

`UncertaintySet` is `@dataclass(frozen=True)` with fields:

| 1.0 field | was (0.20.x) | meaning |
|---|---|---|
| `radius` | `k` | ball size κ |
| `geometry` | `sigma` | linear map L (may be low-rank) |
| `norm` | *(new)* | shape: `1`=diamond, `2`=ellipsoid (default), `inf`=box |

Also a `dual_norm` property. Update any `.k` / `.sigma` attribute access to `.radius` / `.geometry`.
