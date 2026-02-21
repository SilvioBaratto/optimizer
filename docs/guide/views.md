# View Integration

Frameworks for incorporating investor views into portfolio construction. The views
module provides three complementary approaches -- Black-Litterman, Entropy Pooling,
and Opinion Pooling -- each offering different tradeoffs between expressiveness,
computational cost, and theoretical grounding.

All three frameworks produce skfolio `BasePrior` objects that plug directly into
any skfolio optimiser via the `prior_estimator` parameter. The module follows the
standard project convention: **frozen `@dataclass` config** + **factory function**.
Configs hold only serialisable primitives; non-serialisable objects (estimator
instances, numpy arrays) are passed as factory keyword arguments.

**Important**: Views use `tuple[str, ...]` in configs (hashable for frozen
dataclasses). Factory functions convert these to `list[str]` before passing to
skfolio.

---

## Module Overview

| Framework | Config | Factory | Use Case |
|-----------|--------|---------|----------|
| Black-Litterman | `BlackLittermanConfig` | `build_black_litterman()` | Equilibrium-based return tilting with absolute/relative views |
| Entropy Pooling | `EntropyPoolingConfig` | `build_entropy_pooling()` | Non-parametric views on any distributional moment |
| Opinion Pooling | `OpinionPoolingConfig` | `build_opinion_pooling()` | Combining multiple expert prior estimators |
| Omega Calibration | -- | `calibrate_omega_from_track_record()` | Empirical uncertainty from forecast track records |

```python
from optimizer.views import (
    BlackLittermanConfig,
    EntropyPoolingConfig,
    OpinionPoolingConfig,
    ViewUncertaintyMethod,
    build_black_litterman,
    build_entropy_pooling,
    build_opinion_pooling,
    calibrate_omega_from_track_record,
)
```

---

## 1. Black-Litterman

The Black-Litterman (BL) model starts from a market equilibrium prior and
tilts expected returns toward investor views. The posterior blends equilibrium
returns \( \mu_{eq} \) with view-implied returns through a Bayesian update.

### Posterior Formula

Given:

- \( \mu_{eq} \) -- equilibrium expected returns (from the prior)
- \( \Sigma \) -- covariance matrix
- \( \tau \) -- uncertainty scaling parameter (typically 0.01--0.10)
- \( P \) -- picking matrix (maps views to assets)
- \( Q \) -- view return vector
- \( \Omega \) -- view uncertainty matrix (diagonal)
- \( r_f \) -- risk-free rate

The BL posterior expected returns are:

\[
\mu_{BL} = \mu_{eq} + \tau \Sigma P^{\top} \left( P \tau \Sigma P^{\top} + \Omega \right)^{-1} \left( Q - P \mu_{eq} \right) + r_f
\]

The BL posterior covariance is:

\[
\Sigma_{BL} = \Sigma + \tau \Sigma - \tau \Sigma P^{\top} \left( P \tau \Sigma P^{\top} + \Omega \right)^{-1} P \tau \Sigma
\]

### View Syntax

Views are expressed as strings that skfolio parses into the picking matrix \( P \)
and view vector \( Q \).

**Absolute views** -- a single asset will achieve a specific return:

```python
"AAPL == 0.05"    # AAPL expected return is 5%
"JPM == 0.03"     # JPM expected return is 3%
```

**Relative views** -- the difference in returns between two assets:

```python
"AAPL - MSFT == 0.02"   # AAPL outperforms MSFT by 2%
```

### Configuration

`BlackLittermanConfig` is a frozen dataclass with the following fields:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `views` | `tuple[str, ...]` | *(required)* | View expressions (absolute or relative) |
| `tau` | `float` | `0.05` | Uncertainty scaling; must be strictly positive |
| `risk_free_rate` | `float` | `0.0` | Risk-free rate added to posterior returns |
| `uncertainty_method` | `ViewUncertaintyMethod` | `HE_LITTERMAN` | How to calibrate the \( \Omega \) matrix |
| `view_confidences` | `tuple[float, ...] \| None` | `None` | Per-view confidence levels in \([0, 1]\); required for Idzorek |
| `groups` | `dict[str, list[str]] \| None` | `None` | Asset group mapping for group-relative views |
| `prior_config` | `MomentEstimationConfig \| None` | `None` | Inner prior config; defaults to `EquilibriumMu` + `LedoitWolf` |
| `use_factor_model` | `bool` | `False` | Wrap BL in a `FactorModel` |
| `residual_variance` | `bool` | `True` | Include residual variance in `FactorModel` |

**Validation**: `tau` must be strictly positive. Setting `tau=0` or a negative
value raises `ValueError`.

### Uncertainty Methods

The `ViewUncertaintyMethod` enum controls how the diagonal uncertainty matrix
\( \Omega \) is constructed:

| Method | Enum Value | Description |
|--------|------------|-------------|
| He-Litterman | `HE_LITTERMAN` | \( \Omega = \text{diag}(P \cdot \tau\Sigma \cdot P^{\top}) \); proportional to the variance of each view portfolio |
| Idzorek | `IDZOREK` | Per-view confidence levels in \([0, 1]\) that interpolate between 0% and 100% tilt |
| Empirical Track Record | `EMPIRICAL_TRACK_RECORD` | \( \Omega_{kk} = \text{Var}(Q_{k,t} - r_{k,t}) \); calibrated from historical forecast errors |

### Presets

Three factory methods provide common configurations:

```python
# Standard BL with EquilibriumMu + LedoitWolf prior
cfg = BlackLittermanConfig.for_equilibrium(
    views=("AAPL == 0.05", "JPM == 0.03"),
)

# BL wrapped in a FactorModel
cfg = BlackLittermanConfig.for_factor_model(
    views=("MTUM == 0.05",),
)

# Idzorek method with per-view confidence levels
cfg = BlackLittermanConfig.for_idzorek(
    views=("AAPL == 0.05", "MSFT == 0.03"),
    view_confidences=(0.9, 0.6),
)
```

### Factory Function

```python
def build_black_litterman(
    config: BlackLittermanConfig,
    view_history: pd.DataFrame | None = None,
    return_history: pd.DataFrame | None = None,
    omega: npt.NDArray[np.float64] | None = None,
) -> BasePrior:
```

**Parameters**:

- `config` -- the `BlackLittermanConfig` instance
- `view_history` -- historical forecasted \( Q \) values (dates x views); required for `EMPIRICAL_TRACK_RECORD` unless `omega` is pre-supplied
- `return_history` -- realised returns aligned to each view (dates x views); required together with `view_history`
- `omega` -- pre-computed diagonal \( \Omega \) matrix; when provided with `EMPIRICAL_TRACK_RECORD`, used directly (skips history computation)

**Returns**: a `BlackLitterman` instance (or `FactorModel` wrapping one if `use_factor_model=True`).

### Examples

#### Absolute and Relative Views

```python
from skfolio.preprocessing import prices_to_returns
from optimizer.views import BlackLittermanConfig, build_black_litterman

returns = prices_to_returns(prices)

cfg = BlackLittermanConfig.for_equilibrium(
    views=("AAPL == 0.05", "AAPL - MSFT == 0.02", "JPM == 0.03"),
)
prior = build_black_litterman(cfg)
prior.fit(returns)

mu_posterior = prior.return_distribution_.mu
cov_posterior = prior.return_distribution_.covariance
```

#### Idzorek Confidence Levels

Higher confidence values pull the posterior closer to the view target:

```python
cfg = BlackLittermanConfig.for_idzorek(
    views=("AAPL == 0.10", "MSFT == 0.03"),
    view_confidences=(0.9, 0.5),   # 90% confident on AAPL, 50% on MSFT
)
prior = build_black_litterman(cfg)
prior.fit(returns)
```

#### Empirical Track Record

When historical forecast data is available, the uncertainty matrix can be
calibrated from realised forecast errors:

```python
import pandas as pd
import numpy as np

cfg = BlackLittermanConfig(
    views=("AAPL == 0.05",),
    uncertainty_method=ViewUncertaintyMethod.EMPIRICAL_TRACK_RECORD,
)

# Option A: supply view/return history, let the factory calibrate omega
prior = build_black_litterman(
    cfg,
    view_history=view_history_df,     # shape (n_dates, n_views)
    return_history=return_history_df,  # same shape
)

# Option B: supply a pre-computed omega directly
omega = np.diag([1e-4])
prior = build_black_litterman(cfg, omega=omega)

prior.fit(returns)
```

#### Factor Model Variant

When using BL inside a `FactorModel`, views must reference **factor names**
(not asset names):

```python
from skfolio.preprocessing import prices_to_returns

asset_returns = prices_to_returns(asset_prices)
factor_returns = prices_to_returns(factor_prices)

cfg = BlackLittermanConfig.for_factor_model(
    views=("MTUM == 0.05", "QUAL == 0.03"),
)
prior = build_black_litterman(cfg)
prior.fit(asset_returns, y=factor_returns)
```

#### Composing with MeanRisk

```python
from skfolio.optimization import MeanRisk

cfg = BlackLittermanConfig.for_equilibrium(views=("AAPL == 0.05",))
prior = build_black_litterman(cfg)

model = MeanRisk(prior_estimator=prior)
model.fit(returns)
portfolio = model.predict(returns)
```

---

## 2. Entropy Pooling

Entropy Pooling (Meucci, 2008) is a non-parametric framework that finds the
probability distribution closest to an empirical prior (in the Kullback-Leibler
divergence sense) subject to moment constraints derived from investor views.

### Mathematical Formulation

Given a prior probability vector \( \mathbf{p}_0 \) over historical scenarios,
Entropy Pooling solves:

\[
\min_{\mathbf{p}} \; \text{KL}(\mathbf{p} \| \mathbf{p}_0) = \sum_{s=1}^{S} p_s \ln \frac{p_s}{p_{0,s}}
\]

subject to:

\[
\sum_{s=1}^{S} p_s \, f_k(r_s) = v_k, \quad k = 1, \ldots, K
\]

where \( f_k \) encodes the view constraint (mean, variance, correlation,
skewness, kurtosis, or CVaR) and \( v_k \) is the target value. The solution
re-weights scenarios to satisfy the views while staying as close as possible
to the original distribution.

This approach is strictly more general than Black-Litterman: it supports views
on any distributional moment, not just expected returns.

### Supported View Types

| View Type | Config Field | Example | Description |
|-----------|-------------|---------|-------------|
| Mean equality | `mean_views` | `"AAPL == 0.05"` | Expected return equals 5% |
| Mean inequality | `mean_inequality_views` | `"AAPL >= 0.03"` | Expected return at least 3% |
| Variance | `variance_views` | `"AAPL == 0.04"` | Variance equals 0.04 |
| Correlation | `correlation_views` | `"(AAPL, JPM) == 0.5"` | Pairwise correlation equals 0.5 |
| Skewness | `skew_views` | `"AAPL == -0.5"` | Skewness equals -0.5 |
| Kurtosis | `kurtosis_views` | `"AAPL == 5.0"` | Kurtosis equals 5.0 |
| CVaR | `cvar_views` | `"AAPL <= -0.05"` | CVaR at `cvar_beta` level |
| Relative mean | `relative_mean_views` | `("AAPL", 0.01)` | Shift mean by +1% from prior |
| Relative variance | `relative_variance_views` | `("AAPL", 2.0)` | Scale variance by 2x from prior |

**Note on correlation view syntax**: In the config, correlation views use
parenthesised pairs: `"(AAPL, JPM) == 0.5"`. In skfolio, these are passed with
semicolon separators: `"AAPL; JPM == 0.5"`. The factory handles this conversion.

**Note on inequality operators**: Both `mean_views` (equality, `==`) and
`mean_inequality_views` (inequality, `>=` / `<=`) are merged into a single
list before being passed to skfolio's `EntropyPooling`, which handles all three
operators.

### Configuration

`EntropyPoolingConfig` is a frozen dataclass:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mean_views` | `tuple[str, ...] \| None` | `None` | Mean equality view expressions |
| `mean_inequality_views` | `tuple[str, ...] \| None` | `None` | Mean inequality view expressions |
| `variance_views` | `tuple[str, ...] \| None` | `None` | Variance view expressions |
| `relative_mean_views` | `tuple[tuple[str, float], ...] \| None` | `None` | Relative mean shifts from prior |
| `relative_variance_views` | `tuple[tuple[str, float], ...] \| None` | `None` | Relative variance multipliers from prior |
| `correlation_views` | `tuple[str, ...] \| None` | `None` | Correlation view expressions |
| `skew_views` | `tuple[str, ...] \| None` | `None` | Skewness view expressions |
| `kurtosis_views` | `tuple[str, ...] \| None` | `None` | Kurtosis view expressions |
| `cvar_views` | `tuple[str, ...] \| None` | `None` | CVaR view expressions |
| `cvar_beta` | `float` | `0.95` | Confidence level for CVaR views |
| `groups` | `dict[str, list[str]] \| None` | `None` | Asset group mapping |
| `solver` | `str` | `"TNC"` | Scipy solver for the dual optimisation |
| `solver_params` | `dict[str, object] \| None` | `None` | Additional solver parameters |
| `prior_config` | `MomentEstimationConfig \| None` | `None` | Inner prior; defaults to `EmpiricalPrior()` |

### Presets

```python
# Mean-only views
cfg = EntropyPoolingConfig.for_mean_views(
    mean_views=("AAPL == 0.05", "JPM == 0.03"),
)

# Stress testing with variance and correlation views
cfg = EntropyPoolingConfig.for_stress_test(
    variance_views=("AAPL == 0.04",),
    correlation_views=("(AAPL, JPM) == 0.5",),
)

# Group-relative views
cfg = EntropyPoolingConfig.for_group_views(
    mean_views=("tech == 0.05",),
    groups={"tech": ["AAPL", "MSFT", "GOOGL"]},
)
```

### Factory Function

```python
def build_entropy_pooling(
    config: EntropyPoolingConfig,
    prior_moments: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    asset_names: list[str] | None = None,
) -> EntropyPooling:
```

**Parameters**:

- `config` -- the `EntropyPoolingConfig` instance
- `prior_moments` -- `(mu, cov)` arrays from a fitted prior; required when `relative_mean_views` or `relative_variance_views` are set
- `asset_names` -- asset names corresponding to rows/columns of `prior_moments`; required together with `prior_moments`

**Returns**: an `EntropyPooling` instance.

**Raises**: `ConfigurationError` if relative views are specified without
providing `prior_moments` and `asset_names`.

### Examples

#### Mean Equality and Inequality Views

```python
from optimizer.views import EntropyPoolingConfig, build_entropy_pooling

cfg = EntropyPoolingConfig(
    mean_views=("AAPL == 0.05",),
    mean_inequality_views=("JPM >= 0.02",),
)
prior = build_entropy_pooling(cfg)
prior.fit(returns)
```

Both equality and inequality mean views are merged into a single list internally.

#### Higher-Moment Views

```python
cfg = EntropyPoolingConfig(
    skew_views=("AAPL == -0.5",),
    kurtosis_views=("AAPL == 5.0",),
    cvar_views=("AAPL <= -0.05",),
    cvar_beta=0.99,
)
prior = build_entropy_pooling(cfg)
prior.fit(returns)
```

#### Relative Views

Relative views express shifts from the fitted prior rather than absolute targets.
This requires passing the fitted prior moments:

```python
import numpy as np

# Fit a prior first to obtain moments
from skfolio.prior import EmpiricalPrior
base_prior = EmpiricalPrior()
base_prior.fit(returns)
mu = base_prior.return_distribution_.mu
cov = base_prior.return_distribution_.covariance
asset_names = list(returns.columns)

# Relative mean: shift AAPL's expected return up by 1%
# Relative variance: double MSFT's variance
cfg = EntropyPoolingConfig(
    relative_mean_views=(("AAPL", 0.01),),
    relative_variance_views=(("MSFT", 2.0),),
)
prior = build_entropy_pooling(
    cfg,
    prior_moments=(mu, cov),
    asset_names=asset_names,
)
prior.fit(returns)
```

Internally, relative mean views are converted to absolute views by adding the
shift to the prior mean: `AAPL == {mu[i] + 0.01}`. Relative variance views
multiply the prior diagonal variance: `MSFT == {cov[j,j] * 2.0}`.

#### Stress Testing

```python
cfg = EntropyPoolingConfig.for_stress_test(
    variance_views=("AAPL == 0.04", "JPM == 0.06"),
    correlation_views=("(AAPL, JPM) == 0.8",),
)
prior = build_entropy_pooling(cfg)
prior.fit(returns)
```

---

## 3. Opinion Pooling

Opinion Pooling combines forecasts from multiple expert prior estimators into a
single posterior distribution. Each expert independently produces a return
distribution, and the pooling operator aggregates them.

### Pooling Methods

**Linear (arithmetic) pooling**:

\[
p_{\text{pool}}(r) = \sum_{k=1}^{K} w_k \, p_k(r)
\]

**Logarithmic (geometric) pooling**:

\[
p_{\text{pool}}(r) \propto \prod_{k=1}^{K} p_k(r)^{w_k}
\]

where \( w_k \) are the expert weights (`opinion_probabilities`) and \( p_k \)
is the distribution from expert \( k \).

### Configuration

`OpinionPoolingConfig` is a frozen dataclass:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `opinion_probabilities` | `tuple[float, ...] \| None` | `None` | Per-expert weights; each in \([0, 1]\), sum at most 1.0 |
| `is_linear_pooling` | `bool` | `True` | `True` for linear pooling, `False` for logarithmic |
| `divergence_penalty` | `float` | `0.0` | KL-divergence penalty for robust pooling |
| `n_jobs` | `int \| None` | `None` | Number of parallel jobs for expert fitting |
| `prior_config` | `MomentEstimationConfig \| None` | `None` | Common prior configuration |

**Validation**:
- Each probability must be in \([0, 1]\).
- The sum of `opinion_probabilities` must be at most 1.0 (with numerical tolerance of \(10^{-10}\)).
- Setting `opinion_probabilities=None` gives equal weight to all experts.

### Factory Function

```python
def build_opinion_pooling(
    estimators: Sequence[tuple[str, BasePrior]],
    config: OpinionPoolingConfig | None = None,
) -> OpinionPooling:
```

**Parameters**:

- `estimators` -- named expert prior estimators as a sequence of `(name, estimator)` tuples. These are passed directly because estimator objects are not serialisable in a frozen dataclass.
- `config` -- optional `OpinionPoolingConfig`; defaults to `OpinionPoolingConfig()` (linear pooling, equal weights, no penalty).

**Returns**: an `OpinionPooling` instance.

### Examples

#### Combining Expert Forecasts

```python
from skfolio.prior import EntropyPooling
from optimizer.views import OpinionPoolingConfig, build_opinion_pooling

expert_1 = EntropyPooling(mean_views=["AAPL == 0.05"])
expert_2 = EntropyPooling(mean_views=["JPM == 0.03"])

estimators = [
    ("fundamental_analyst", expert_1),
    ("quant_model", expert_2),
]
cfg = OpinionPoolingConfig(opinion_probabilities=(0.6, 0.4))
prior = build_opinion_pooling(estimators, cfg)
prior.fit(returns)
```

#### Logarithmic Pooling

```python
cfg = OpinionPoolingConfig(is_linear_pooling=False)
prior = build_opinion_pooling(estimators, cfg)
prior.fit(returns)
```

#### Anchoring to a Base Prior

When expert weights are small, the posterior is anchored to the common prior.
This is useful for blending mild expert views with a strong empirical baseline:

```python
cfg = OpinionPoolingConfig(opinion_probabilities=(0.01, 0.01))
prior = build_opinion_pooling(estimators, cfg)
prior.fit(returns)
# Posterior will be very close to the empirical prior
```

---

## 4. Omega Calibration from Track Record

The `calibrate_omega_from_track_record()` function computes an empirical diagonal
\( \Omega \) matrix from a history of forecast errors.

### Formula

For each view \( k \):

\[
\Omega_{kk} = \text{Var}(Q_{k,t} - r_{k,t})
\]

where \( Q_{k,t} \) is the analyst's forecast for view \( k \) at time \( t \)
and \( r_{k,t} \) is the corresponding realised return. The off-diagonal entries
are zero (diagonal matrix). The sample variance uses Bessel's correction (`ddof=1`).

### Function Signature

```python
def calibrate_omega_from_track_record(
    view_history: pd.DataFrame,
    return_history: pd.DataFrame,
) -> npt.NDArray[np.float64]:
```

**Parameters**:

- `view_history` -- DataFrame of shape `(n_dates, n_views)` with historical forecasted \( Q \) values
- `return_history` -- DataFrame of same shape with realised returns aligned to each view

**Returns**: diagonal \( \Omega \) matrix of shape `(n_views, n_views)`.

**Raises**: `DataError` if:
- The two DataFrames have different shapes
- The column names do not match
- Fewer than 5 aligned observations remain after dropping NaN rows

### Example

```python
import pandas as pd
import numpy as np
from optimizer.views import calibrate_omega_from_track_record

# view_history: 30 dates, 2 views
view_history = pd.DataFrame({
    "view_aapl": np.random.normal(0.001, 0.005, 30),
    "view_jpm": np.random.normal(0.002, 0.003, 30),
})
return_history = pd.DataFrame({
    "view_aapl": np.random.normal(0.001, 0.02, 30),
    "view_jpm": np.random.normal(0.002, 0.015, 30),
})

omega = calibrate_omega_from_track_record(view_history, return_history)
# omega is a (2, 2) diagonal matrix
print(omega)
```

This \( \Omega \) can then be passed directly to `build_black_litterman()`:

```python
cfg = BlackLittermanConfig(
    views=("AAPL == 0.05", "JPM == 0.03"),
    uncertainty_method=ViewUncertaintyMethod.EMPIRICAL_TRACK_RECORD,
)
prior = build_black_litterman(cfg, omega=omega)
```

---

## Gotchas and Common Pitfalls

### Factor Model Views Must Reference Factor Names

When `use_factor_model=True`, the BL prior is wrapped in a `FactorModel`.
In this case, views must reference **factor names** (e.g., `"MTUM"`, `"QUAL"`),
not asset names. Using asset names will cause a fitting error because the
picking matrix is constructed over the factor return space.

```python
# CORRECT: views reference factor names
cfg = BlackLittermanConfig.for_factor_model(views=("MTUM == 0.05",))
prior = build_black_litterman(cfg)
prior.fit(asset_returns, y=factor_returns)

# WRONG: views reference asset names inside a FactorModel
cfg = BlackLittermanConfig.for_factor_model(views=("AAPL == 0.05",))
```

### Tau Must Be Strictly Positive

The uncertainty scaling parameter \( \tau \) must be strictly greater than zero.
Setting `tau=0.0` or a negative value raises `ValueError` at config creation time.
Typical values range from 0.01 to 0.10; the default is 0.05.

### Relative Views Require Prior Moments

When using `relative_mean_views` or `relative_variance_views` in
`EntropyPoolingConfig`, you must supply `prior_moments` and `asset_names`
to `build_entropy_pooling()`. Without them, the factory raises
`ConfigurationError`.

### Empirical Track Record Requires History or Pre-Computed Omega

When `uncertainty_method=EMPIRICAL_TRACK_RECORD`, one of the following must hold:

1. Both `view_history` and `return_history` are supplied (at least 5 aligned non-NaN observations).
2. A pre-computed `omega` array is supplied directly.

Supplying neither raises `ConfigurationError`.

### Opinion Probabilities Must Sum to At Most 1.0

The `opinion_probabilities` tuple in `OpinionPoolingConfig` is validated:

- Each value must be in \([0, 1]\).
- The sum must not exceed 1.0.

Violating either constraint raises `ValueError`.

### Estimators Are Not Stored in OpinionPoolingConfig

Because skfolio estimator objects are not serialisable in a frozen dataclass,
expert estimators for Opinion Pooling are passed as a factory argument, not
stored in the config. This preserves the config-is-serialisable invariant.

### Config Tuples vs. skfolio Lists

All view fields in configs use `tuple` types for hashability (required by
frozen dataclasses). The factory functions convert these to `list` before
passing to skfolio. You do not need to handle this conversion manually.

### Mean Equality and Inequality Views Are Merged

`EntropyPoolingConfig` has separate fields for `mean_views` (equality, `==`)
and `mean_inequality_views` (inequality, `>=` / `<=`). The factory merges
both into a single list for skfolio's `EntropyPooling.mean_views` parameter,
which handles all three operators natively.

---

## Choosing a Framework

| Criterion | Black-Litterman | Entropy Pooling | Opinion Pooling |
|-----------|-----------------|-----------------|-----------------|
| View types | Mean (absolute/relative) | Mean, variance, correlation, skew, kurtosis, CVaR | Any (via expert estimators) |
| Equilibrium anchor | Yes (built-in) | No (empirical prior) | Configurable |
| Factor model support | Yes (`use_factor_model`) | No | No |
| Closed-form solution | Yes | No (numerical optimisation) | No |
| Number of experts | Single analyst | Single analyst | Multiple experts |
| Computational cost | Low | Medium | Depends on experts |

**Use Black-Litterman** when you have return views and want to tilt away from
an equilibrium baseline, especially if you want closed-form updates and factor
model integration.

**Use Entropy Pooling** when you have views on distributional moments beyond
the mean (variance, correlation, tail risk) or need inequality constraints.

**Use Opinion Pooling** when you want to combine multiple independent expert
forecasts (each represented as a prior estimator) into a single distribution.
