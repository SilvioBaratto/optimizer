# Moment Estimation

The `optimizer.moments` module provides expected return estimation, covariance
estimation, Hidden Markov Model regime blending, Deep Markov Models, and
multi-period log-normal scaling. Every component follows the library-wide
pattern of **frozen dataclass config + factory function**, and all estimators
conform to the skfolio `BaseMu` / `BaseCovariance` API so they compose
directly inside sklearn pipelines.

---

## Module Layout

| File | Contents |
|---|---|
| `optimizer/moments/_config.py` | `MuEstimatorType`, `CovEstimatorType`, `ShrinkageMethod` enums; `MomentEstimationConfig` frozen dataclass |
| `optimizer/moments/_factory.py` | `build_mu_estimator()`, `build_cov_estimator()`, `build_prior()` factories |
| `optimizer/moments/_hmm.py` | `HMMConfig`, `HMMResult`, `fit_hmm()`, `select_hmm_n_states()`, `blend_moments_by_regime()`, `HMMBlendedMu`, `HMMBlendedCovariance` |
| `optimizer/moments/_dmm.py` | `DMMConfig`, `DMMResult`, `fit_dmm()`, `blend_moments_dmm()` (optional; requires `torch` + `pyro-ppl`) |
| `optimizer/moments/_scaling.py` | `apply_lognormal_correction()`, `scale_moments_to_horizon()` |

---

## Expected Return Estimators

The `MuEstimatorType` enum selects which skfolio `BaseMu` estimator
`build_mu_estimator()` instantiates.

| Enum value | skfolio class | Key parameter(s) | Description |
|---|---|---|---|
| `EMPIRICAL` | `EmpiricalMu` | -- | Sample mean of historical returns |
| `SHRUNK` | `ShrunkMu` | `shrinkage_method` | Shrinkage toward a structured target (see table below) |
| `EW` | `EWMu` | `ew_mu_alpha` (default 0.2) | Exponentially weighted mean; higher alpha puts more weight on recent observations |
| `EQUILIBRIUM` | `EquilibriumMu` | `risk_aversion` (default 1.0) | Implied equilibrium returns from market-cap weights; Black-Litterman starting point |
| `HMM_BLENDED` | `HMMBlendedMu` | `hmm_config` | Regime-probability-weighted blend of per-state means (see [HMM section](#hidden-markov-model-regime-blending)) |

### Shrinkage methods

When `mu_estimator = MuEstimatorType.SHRUNK`, the `ShrinkageMethod` enum
controls which shrinkage flavour is used:

| Enum value | skfolio method | Reference |
|---|---|---|
| `JAMES_STEIN` | `ShrunkMuMethods.JAMES_STEIN` | James & Stein (1961) |
| `BAYES_STEIN` | `ShrunkMuMethods.BAYES_STEIN` | Jorion (1986) |
| `BODNAR_OKHRIN` | `ShrunkMuMethods.BODNAR_OKHRIN` | Bodnar & Okhrin (2011) |

---

## Covariance Estimators

The `CovEstimatorType` enum selects which skfolio `BaseCovariance` estimator
`build_cov_estimator()` instantiates.

| Enum value | skfolio class | Key parameter(s) | Description |
|---|---|---|---|
| `EMPIRICAL` | `EmpiricalCovariance` | -- | Sample covariance matrix |
| `LEDOIT_WOLF` | `LedoitWolf` | -- | Analytical shrinkage (Ledoit & Wolf, 2004); optimal bias-variance trade-off without cross-validation |
| `OAS` | `OAS` | -- | Oracle Approximating Shrinkage (Chen et al., 2010); similar to Ledoit-Wolf but with a different analytical formula |
| `SHRUNK` | `ShrunkCovariance` | `shrunk_cov_shrinkage` (default 0.1) | Fixed shrinkage intensity toward a diagonal target |
| `EW` | `EWCovariance` | `ew_cov_alpha` (default 0.2) | Exponentially weighted covariance; recent observations receive higher weight |
| `GERBER` | `GerberCovariance` | `gerber_threshold` (default 0.5) | Gerber statistic-based covariance; only co-movements that exceed the threshold contribute |
| `GRAPHICAL_LASSO_CV` | `GraphicalLassoCV` | -- | Sparse inverse covariance via L1-penalised MLE with cross-validated penalty |
| `DENOISE` | `DenoiseCovariance` | inner: `EmpiricalCovariance` | Random matrix theory denoising; filters eigenvalues below the Marchenko-Pastur threshold |
| `DETONE` | `DetoneCovariance` | inner: `EmpiricalCovariance` | Market factor removal; strips the largest eigenvalue (market mode) from the covariance |
| `IMPLIED` | `ImpliedCovariance` | -- | Implied covariance from option-market data |
| `HMM_BLENDED` | `HMMBlendedCovariance` | `hmm_config` | Full law-of-total-variance blend of regime covariances (see [HMM section](#hidden-markov-model-regime-blending)) |

---

## MomentEstimationConfig

`MomentEstimationConfig` is a frozen dataclass that bundles all moment
estimation parameters into a single serialisable object. Non-serialisable
objects (estimator instances, numpy arrays) are never stored in the config;
they are constructed by the factory functions.

### Fields

```python
@dataclass(frozen=True)
class MomentEstimationConfig:
    # Expected return estimator
    mu_estimator: MuEstimatorType = MuEstimatorType.EMPIRICAL
    shrinkage_method: ShrinkageMethod = ShrinkageMethod.JAMES_STEIN
    ew_mu_alpha: float = 0.2
    risk_aversion: float = 1.0

    # Covariance estimator
    cov_estimator: CovEstimatorType = CovEstimatorType.LEDOIT_WOLF
    ew_cov_alpha: float = 0.2
    shrunk_cov_shrinkage: float = 0.1
    gerber_threshold: float = 0.5

    # Prior assembly
    is_log_normal: bool = False
    investment_horizon: float | None = None

    # HMM blended estimators
    hmm_config: HMMConfig = field(default_factory=HMMConfig)

    # Factor model
    use_factor_model: bool = False
    residual_variance: bool = True
```

### Presets

| Preset method | mu estimator | cov estimator | Use case |
|---|---|---|---|
| `for_equilibrium_ledoitwolf()` | `EquilibriumMu` | `LedoitWolf` | Black-Litterman-ready prior; equilibrium returns serve as the neutral starting point |
| `for_shrunk_denoised()` | `ShrunkMu` (James-Stein) | `DenoiseCovariance` | Conservative prior; shrinks expected returns and removes noise from the covariance spectrum |
| `for_adaptive()` | `EWMu` | `EWCovariance` | Responsive prior; exponentially weighted moments adapt quickly to regime changes |
| `for_hmm_blended(n_states=2)` | `HMMBlendedMu` | `HMMBlendedCovariance` | Regime-aware prior; probability-weighted blend of per-regime moments |

### Usage

```python
from optimizer.moments import MomentEstimationConfig

# Use a preset
config = MomentEstimationConfig.for_equilibrium_ledoitwolf()

# Or build from scratch
config = MomentEstimationConfig(
    mu_estimator=MuEstimatorType.SHRUNK,
    shrinkage_method=ShrinkageMethod.BAYES_STEIN,
    cov_estimator=CovEstimatorType.DENOISE,
)
```

---

## Factory Functions

### build_mu_estimator

Maps the `mu_estimator` field of a `MomentEstimationConfig` to a concrete
skfolio `BaseMu` instance.

```python
from optimizer.moments import MomentEstimationConfig, build_mu_estimator

config = MomentEstimationConfig(mu_estimator=MuEstimatorType.EW, ew_mu_alpha=0.3)
mu_est = build_mu_estimator(config)
# Returns an EWMu(alpha=0.3) instance ready for .fit(X)
```

### build_cov_estimator

Maps the `cov_estimator` field to a concrete skfolio `BaseCovariance` instance.

```python
from optimizer.moments import MomentEstimationConfig, build_cov_estimator

config = MomentEstimationConfig(cov_estimator=CovEstimatorType.GERBER, gerber_threshold=0.4)
cov_est = build_cov_estimator(config)
# Returns a GerberCovariance(threshold=0.4) instance
```

### build_prior

Composes `build_mu_estimator` and `build_cov_estimator` into an
`EmpiricalPrior`, and optionally wraps it in a `FactorModel`.

```python
from optimizer.moments import MomentEstimationConfig, build_prior

# Default prior: EmpiricalMu + LedoitWolf
prior = build_prior()

# Prior from a preset
config = MomentEstimationConfig.for_shrunk_denoised()
prior = build_prior(config)

# Factor model prior
config = MomentEstimationConfig(
    mu_estimator=MuEstimatorType.EMPIRICAL,
    cov_estimator=CovEstimatorType.LEDOIT_WOLF,
    use_factor_model=True,
    residual_variance=True,
)
prior = build_prior(config)
# Returns a FactorModel wrapping EmpiricalPrior
```

When `use_factor_model=True`, the resulting `FactorModel` expects factor
returns as the `y` argument during `fit(X, y)`. The fitted prior attribute is
`return_distribution_` (not `prior_model_`), containing `mu`, `covariance`,
`returns`, `sample_weight`, and `cholesky`.

---

## Hidden Markov Model Regime Blending

The HMM subsystem fits a Gaussian Hidden Markov Model to a panel of asset
returns, extracts regime-conditional moments, and produces
probability-weighted blended estimates suitable for portfolio optimization.

### HMMConfig

```python
@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 2          # Number of latent regimes
    n_iter: int = 100          # Max Baum-Welch EM iterations
    tol: float = 1e-4          # Convergence tolerance on log-likelihood
    covariance_type: str = "full"  # "full", "diag", "tied", or "spherical"
    random_state: int | None = None
```

### HMMResult

After fitting, `fit_hmm()` returns an `HMMResult` dataclass containing:

| Attribute | Shape | Description |
|---|---|---|
| `transition_matrix` | `(n_states, n_states)` | Row-stochastic matrix \( A_{ij} = P(z_t = j \mid z_{t-1} = i) \) |
| `regime_means` | `DataFrame (n_states, n_assets)` | Per-regime expected return vectors \( \mu_s \) |
| `regime_covariances` | `(n_states, n_assets, n_assets)` | Per-regime covariance matrices \( \Sigma_s \) |
| `filtered_probs` | `DataFrame (n_dates, n_states)` | Forward-only causal probabilities \( \alpha_t(s) \) |
| `smoothed_probs` | `DataFrame (n_dates, n_states)` | Full-sequence posterior probabilities \( \gamma_t(s) \) |
| `log_likelihood` | `float` | Log-likelihood of the data under the fitted model |

### Fitting an HMM

```python
from optimizer.moments import HMMConfig, fit_hmm

config = HMMConfig(n_states=2, random_state=42)
result = fit_hmm(returns, config)

print(result.transition_matrix)
print(result.regime_means)
print(result.filtered_probs.tail())
```

The function drops NaN rows before fitting and raises `DataError` if fewer
than `n_states + 1` observations remain.

### Filtered vs. Smoothed Probabilities

The HMM produces two sets of state probabilities, and the distinction between
them is critical for correct usage:

**Filtered probabilities** \( \alpha_t(s) \propto P(r_t \mid z_t = s) \sum_{s'} A_{s',s} \cdot \alpha_{t-1}(s') \) are computed via the forward algorithm and are conditioned only on past and current observations \( r_{1:t} \). They are **causal** -- they do not use future data -- and are therefore the correct choice for online blending in backtests and live trading.

**Smoothed probabilities** \( \gamma_t(s) = P(z_t = s \mid r_{1:T}) \) are computed via the Baum-Welch forward-backward algorithm and are conditioned on the **entire** observation sequence. They provide the best point estimate of the regime at each time step but introduce **look-ahead bias** and must only be used for diagnostics, regime labelling, and parameter estimation -- never for causal blending in backtests.

| Property | Filtered | Smoothed |
|---|---|---|
| Conditioning | \( r_{1:t} \) (past + present) | \( r_{1:T} \) (full sequence) |
| Causal | Yes | No |
| Look-ahead bias | None | Yes |
| Use for backtests | Yes | No |
| Use for diagnostics | Yes | Yes |

### Model Selection: AIC / BIC

`select_hmm_n_states()` evaluates multiple candidate state counts and returns
the one that minimises the chosen information criterion.

Free parameters for a model with \( S \) states and \( d \) assets:

\[
k = S(S - 1) + S \cdot d + S \cdot \frac{d(d+1)}{2}
\]

where the three terms count the transition matrix rows (each sums to 1, so
\( S - 1 \) free per row), per-regime means, and per-regime full covariance
(lower triangle).

The criteria are:

\[
\text{AIC} = -2 \ln L + 2k
\]
\[
\text{BIC} = -2 \ln L + k \ln T
\]

```python
from optimizer.moments import select_hmm_n_states

best_n = select_hmm_n_states(
    returns,
    candidate_n_states=(2, 3, 4),
    criterion="bic",
)
print(f"Optimal number of regimes: {best_n}")
```

### Blending Moments by Regime

#### Simple blend: `blend_moments_by_regime()`

Computes a probability-weighted blend using the filtered probabilities at the
final time step:

\[
\mu = \sum_s p_s \cdot \mu_s
\]
\[
\Sigma = \sum_s p_s \cdot \Sigma_s
\]

where \( p_s = \alpha_T(s) \) are the last-period filtered regime
probabilities.

```python
from optimizer.moments import fit_hmm, blend_moments_by_regime

result = fit_hmm(returns, HMMConfig(n_states=2, random_state=42))
mu_blended, cov_blended = blend_moments_by_regime(result)
```

> **Gotcha**: This function computes only the **within-regime** weighted
> covariance and **omits** the between-regime mean-dispersion term. The
> blended covariance will underestimate total uncertainty when regime means
> differ materially. For optimizer inputs, use `HMMBlendedCovariance` instead.

#### Full blend: `HMMBlendedCovariance`

The `HMMBlendedCovariance` class implements the full law of total variance:

\[
\Sigma = \sum_s p_s \left[ \Sigma_s + (\mu_s - \mu)(\mu_s - \mu)^\top \right]
\]

The second term \( (\mu_s - \mu)(\mu_s - \mu)^\top \) captures the
**between-regime mean dispersion** -- the additional uncertainty that arises
because the true mean itself is uncertain across regimes. This term can be
substantial when regime means differ (e.g., bull vs. bear markets) and
omitting it leads to systematically underestimated risk.

### skfolio-Compatible Estimator Classes

Both `HMMBlendedMu` and `HMMBlendedCovariance` conform to the skfolio
`BaseMu` / `BaseCovariance` API, which means they expose the standard
`mu_` and `covariance_` attributes after `.fit(X)` and can be plugged
directly into `EmpiricalPrior` or any skfolio pipeline.

#### HMMBlendedMu

```python
from optimizer.moments import HMMBlendedMu, HMMConfig

mu_est = HMMBlendedMu(hmm_config=HMMConfig(n_states=2, random_state=42))
mu_est.fit(X_returns)

print(mu_est.mu_)              # ndarray of shape (n_assets,)
print(mu_est.hmm_result_)      # Full HMMResult for inspection
```

After fitting, `mu_` contains the probability-weighted blended expected
return vector:

\[
\mu = \sum_s p(z_T = s \mid r_{1:T}) \cdot \mu_s
\]

#### HMMBlendedCovariance

```python
from optimizer.moments import HMMBlendedCovariance, HMMConfig

cov_est = HMMBlendedCovariance(
    hmm_config=HMMConfig(n_states=2, random_state=42),
    nearest=True,      # project to nearest PSD if needed
    higham=False,       # use eigenvalue clipping (not Higham)
)
cov_est.fit(X_returns)

print(cov_est.covariance_)     # ndarray of shape (n_assets, n_assets)
print(cov_est.hmm_result_)     # Full HMMResult for inspection
```

The `nearest` parameter controls whether the blended covariance is projected
to the nearest positive semi-definite matrix (via eigenvalue clipping or
the Higham algorithm). This is enabled by default because the law-of-total-
variance blend is not guaranteed to be PSD in finite samples.

### Using HMM Blending in the Prior

The recommended way to use HMM blending is through the
`MomentEstimationConfig` preset, which wires everything together:

```python
from optimizer.moments import MomentEstimationConfig, build_prior

config = MomentEstimationConfig.for_hmm_blended(n_states=2)
prior = build_prior(config)

# Use in a MeanRisk optimizer
from skfolio.optimization import MeanRisk
model = MeanRisk(prior_estimator=prior)
model.fit(X_returns)
```

---

## Deep Markov Model (Optional)

The DMM module implements the architecture from Krishnan et al. (2016),
"Structured Inference Networks for Nonlinear State Space Models," using
Pyro's stochastic variational inference (SVI) with KL annealing.

> **Dependency note**: The DMM requires `torch` and `pyro-ppl`, which are
> **not** declared in `pyproject.toml`. The module is effectively optional.
> Import it with:
> ```python
> pip install torch pyro-ppl
> ```
> If the dependencies are missing, importing `DMMConfig` or `fit_dmm` from
> `optimizer.moments` will silently be suppressed (via `contextlib.suppress`
> in `__init__.py`).

### Architecture

The generative model factorises as:

\[
p(x_{1:T}, z_{1:T}) = p(z_1) \prod_{t=2}^{T} p(z_t \mid z_{t-1}) \prod_{t=1}^{T} p(x_t \mid z_t)
\]

The variational guide uses a backward-RNN inference network:

\[
q(z_{1:T} \mid x_{1:T}) = \prod_{t=1}^{T} q(z_t \mid z_{t-1}, h_t^{\text{rnn}})
\]

where \( h_t^{\text{rnn}} \) encodes the future context \( x_t, \ldots, x_T \)
via a GRU running backward over the sequence.

| Component | Class | Role |
|---|---|---|
| Emitter | `Emitter` | Maps \( z_t \to (\text{loc}, \text{scale}) \) of emission distribution \( p(x_t \mid z_t) \) |
| Transition | `GatedTransition` | Gated residual MLP for \( p(z_t \mid z_{t-1}) \); gate interpolates between linear identity and nonlinear proposal |
| Combiner | `Combiner` | Fuses \( z_{t-1} \) with backward-RNN context for variational posterior \( q(z_t \mid z_{t-1}, h_t) \) |
| Inference RNN | `nn.GRU` | Backward-running GRU encoding \( x_t, \ldots, x_T \) into context vectors |

### DMMConfig

```python
@dataclass(frozen=True)
class DMMConfig:
    z_dim: int = 16                    # Latent state dimension
    emission_dim: int = 64             # Emitter hidden layer size
    transition_dim: int = 64           # Transition hidden layer size
    rnn_dim: int = 128                 # GRU hidden state size
    num_epochs: int = 1000             # SVI training epochs
    learning_rate: float = 3e-4        # ClippedAdam learning rate
    annealing_epochs: int = 50         # KL annealing ramp length
    minimum_annealing_factor: float = 0.2  # Starting KL weight
    random_state: int | None = None
```

### DMMResult

| Attribute | Shape | Description |
|---|---|---|
| `latent_means` | `DataFrame (T, z_dim)` | Variational posterior means for each time step |
| `latent_stds` | `DataFrame (T, z_dim)` | Variational posterior standard deviations |
| `elbo_history` | `list[float]` | ELBO value per training epoch (for convergence monitoring) |
| `model` | `DMM` | Trained PyTorch module instance |
| `tickers` | `list[str]` | Asset names in training order |
| `input_mean` | `ndarray (n_assets,)` | Per-asset mean used for input standardisation |
| `input_std` | `ndarray (n_assets,)` | Per-asset std used for input standardisation |

### Fitting and Blending

```python
from optimizer.moments import DMMConfig, fit_dmm, blend_moments_dmm

config = DMMConfig(z_dim=16, num_epochs=500, random_state=42)
result = fit_dmm(returns, config)

# Check convergence
import matplotlib.pyplot as plt
plt.plot(result.elbo_history)
plt.xlabel("Epoch")
plt.ylabel("ELBO")
plt.title("DMM Training Convergence")
plt.show()

# Produce blended moments via Monte Carlo posterior-predictive sampling
mu, cov = blend_moments_dmm(result, n_mc_samples=500, seed=42)
```

`blend_moments_dmm()` works by:

1. Sampling \( z_T \sim q(z_T \mid x_{1:T}) \) from the variational posterior
   at the last time step.
2. Propagating through the transition: \( z_{T+1} \sim p(z_{T+1} \mid z_T) \).
3. Emitting: \( x_{T+1} \sim p(x_{T+1} \mid z_{T+1}) \).
4. Applying the law of total variance across Monte Carlo samples.
5. Un-standardising back to the original return scale.

> **Critical limitation**: The DMM produces **diagonal covariance only**.
> The law-of-total-variance computation yields
> \( \text{diag}(\mathbb{E}[\text{Var}[X \mid Z]] + \text{Var}[\mathbb{E}[X \mid Z]]) \),
> so all off-diagonal entries are zero. This makes the DMM unsuitable as a
> standalone covariance estimator for portfolio optimization -- it should be
> combined with a separate cross-sectional covariance estimate.

---

## Log-Normal Multi-Period Scaling

When working with multi-period investment horizons, daily log-return moments
must be scaled to the target horizon. The `_scaling` module provides two
methods: an exact log-normal formula and a linear (delta-method)
approximation.

### The Scaling Problem

If daily log-returns \( r_t \sim N(\mu, \Sigma) \) are i.i.d., then the
cumulative simple return over \( T \) days is:

\[
R_T = \exp\left(\sum_{t=1}^{T} r_t\right) - 1
\]

Because the sum of normals is normal, \( \sum r_t \sim N(\mu T, \Sigma T) \),
but \( R_T \) is **not** normal -- it is log-normally distributed. The
scaling functions convert log-return parameters to simple-return space.

### Expected Return (Both Methods)

Jensen's inequality correction gives the expected simple return:

\[
\mathbb{E}[R_T^i] = \exp\!\left(\mu_i T + \frac{1}{2} \sigma_i^2 T\right) - 1
\]

where \( \sigma_i^2 = \Sigma_{ii} \) is the daily variance of asset \( i \).
This formula is identical for both the exact and linear methods.

### Covariance: Exact Method

The exact log-normal covariance is:

\[
\text{Cov}[R_T^i, R_T^j] = \exp\!\left((\mu_i + \mu_j) T + \frac{1}{2}(\sigma_i^2 + \sigma_j^2) T\right) \cdot \left(\exp(\sigma_{ij} \cdot T) - 1\right)
\]

where \( \sigma_{ij} = \Sigma_{ij} \) is the daily covariance between assets
\( i \) and \( j \).

### Covariance: Linear Method

The delta-method (first-order Taylor) approximation:

\[
\Sigma_T \approx \Sigma \cdot T
\]

This is accurate for short horizons and small variances but increasingly
biased as \( T \) grows or volatility increases.

### Function Signatures

#### `apply_lognormal_correction`

```python
def apply_lognormal_correction(
    mu: pd.Series,        # Daily log-return expected values
    cov: pd.DataFrame,    # Daily log-return covariance matrix
    horizon: int,         # Trading days (21=monthly, 63=quarterly, 252=annual)
    method: str = "exact" # "exact" or "linear"
) -> tuple[pd.Series, pd.DataFrame]:
    ...
```

#### `scale_moments_to_horizon`

A higher-level wrapper that validates inputs (square covariance, aligned
indices, non-negative diagonal) before delegating to
`apply_lognormal_correction`.

```python
def scale_moments_to_horizon(
    mu: pd.Series,
    cov: pd.DataFrame,
    daily_horizon: int,
    method: str = "exact"
) -> tuple[pd.Series, pd.DataFrame]:
    ...
```

### Usage

```python
import pandas as pd
from optimizer.moments import apply_lognormal_correction, scale_moments_to_horizon

# Daily log-return moments
mu_daily = pd.Series({"AAPL": 0.0005, "MSFT": 0.0004, "GOOG": 0.0003})
cov_daily = pd.DataFrame(
    [[0.0004, 0.0001, 0.0001],
     [0.0001, 0.0003, 0.0001],
     [0.0001, 0.0001, 0.0005]],
    index=mu_daily.index,
    columns=mu_daily.index,
)

# Scale to quarterly horizon (63 trading days), exact method
mu_q, cov_q = apply_lognormal_correction(mu_daily, cov_daily, horizon=63, method="exact")

# Scale to annual horizon (252 trading days), linear approximation
mu_a, cov_a = scale_moments_to_horizon(mu_daily, cov_daily, daily_horizon=252, method="linear")
```

> **Important**: Inputs must be **log-return** parameters (mean and covariance
> of log-returns). The outputs are in **simple-return** space
> (\( \mathbb{E}[R_T] \) and \( \text{Cov}[R_T] \)). Feeding simple-return
> moments into these functions will produce incorrect results.

---

## Common Gotchas

### 1. `blend_moments_by_regime()` vs. `HMMBlendedCovariance`

`blend_moments_by_regime()` computes only the within-regime weighted
covariance:

\[
\Sigma_{\text{simple}} = \sum_s p_s \cdot \Sigma_s
\]

`HMMBlendedCovariance` adds the between-regime mean-dispersion term:

\[
\Sigma_{\text{full}} = \sum_s p_s \left[ \Sigma_s + (\mu_s - \mu)(\mu_s - \mu)^\top \right]
\]

The difference \( \Sigma_{\text{full}} - \Sigma_{\text{simple}} = \sum_s p_s (\mu_s - \mu)(\mu_s - \mu)^\top \) is a positive semi-definite matrix. When regime means differ materially (e.g., 10% annualised spread between bull and bear), this term can be a significant fraction of total variance.

**Rule**: For optimizer inputs, always use `HMMBlendedCovariance`. Reserve
`blend_moments_by_regime()` for quick diagnostics or situations where you
explicitly want to ignore between-regime uncertainty.

### 2. Filtered vs. Smoothed Probabilities for Backtests

The `HMMBlendedMu` and `HMMBlendedCovariance` classes use **filtered**
(forward-only) probabilities from the last time step. This is the correct
causal choice for backtesting. If you manually call
`blend_moments_by_regime()`, it also uses the filtered probabilities from
`result.filtered_probs.iloc[-1]`.

Never use `result.smoothed_probs` for weight computation in a backtest -- it
conditions on the entire sequence and introduces look-ahead bias.

### 3. Log-Return vs. Simple-Return Inputs for Scaling

The `apply_lognormal_correction` and `scale_moments_to_horizon` functions
expect **log-return** (continuously compounded) parameters as input. The
output is in **simple-return** space. If you accidentally pass simple-return
moments as input, the resulting expected returns and covariances will be
biased upward.

### 4. DMM Produces Diagonal Covariance

The Deep Markov Model's `blend_moments_dmm()` returns a diagonal covariance
matrix. All off-diagonal covariances are zero. This means the DMM cannot
capture cross-asset dependencies and should not be used as a standalone
covariance estimator. Consider combining the DMM's variance estimates with a
separate cross-sectional covariance model.

### 5. The Fitted Prior Attribute

After fitting a prior with `build_prior()`, the estimated distribution is
stored in the `return_distribution_` attribute (not `prior_model_`). It
contains `mu`, `covariance`, `returns`, `sample_weight`, and `cholesky`.

```python
prior = build_prior(config)
prior.fit(X_returns)
print(prior.return_distribution_.mu)
print(prior.return_distribution_.covariance)
```

### 6. Factor Model Views

When the prior is wrapped in a `FactorModel` (via `use_factor_model=True`),
any downstream views (e.g., Black-Litterman) must reference **factor names**
(e.g., `MTUM`, `QUAL`), not asset names.

---

## Complete Example

```python
import pandas as pd
from optimizer.moments import (
    MomentEstimationConfig,
    MuEstimatorType,
    CovEstimatorType,
    ShrinkageMethod,
    HMMConfig,
    build_prior,
    build_mu_estimator,
    build_cov_estimator,
    fit_hmm,
    select_hmm_n_states,
    apply_lognormal_correction,
)

# --- 1. Basic prior construction ---
config = MomentEstimationConfig.for_shrunk_denoised()
prior = build_prior(config)
prior.fit(X_returns)
print("Expected returns:", prior.return_distribution_.mu)

# --- 2. HMM regime analysis ---
hmm_cfg = HMMConfig(n_states=2, random_state=42)
result = fit_hmm(returns, hmm_cfg)

# Transition probabilities
print("Transition matrix:\n", result.transition_matrix)

# Current regime belief (causal)
print("Filtered probs (last):", result.filtered_probs.iloc[-1].to_dict())

# --- 3. Model selection ---
best_n = select_hmm_n_states(returns, candidate_n_states=(2, 3, 4), criterion="bic")
print(f"BIC-optimal states: {best_n}")

# --- 4. HMM-blended prior in optimizer ---
config_hmm = MomentEstimationConfig.for_hmm_blended(n_states=best_n)
prior_hmm = build_prior(config_hmm)

from skfolio.optimization import MeanRisk
model = MeanRisk(prior_estimator=prior_hmm)
model.fit(X_returns)

# --- 5. Multi-period scaling ---
mu_daily = pd.Series({"AAPL": 0.0005, "MSFT": 0.0004})
cov_daily = pd.DataFrame(
    [[0.0004, 0.0001], [0.0001, 0.0003]],
    index=mu_daily.index,
    columns=mu_daily.index,
)
mu_annual, cov_annual = apply_lognormal_correction(
    mu_daily, cov_daily, horizon=252, method="exact"
)
print("Annual expected return:", mu_annual)
print("Annual covariance:\n", cov_annual)
```
