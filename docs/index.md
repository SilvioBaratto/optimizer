# Portfolio Optimizer

Quantitative portfolio construction and optimization platform built on [skfolio](https://skfolio.org/) and scikit-learn. Every component follows the frozen-config + factory pattern and composes in standard sklearn pipelines.

## Features

- **[Pipeline](guide/pipeline.md)** -- End-to-end orchestration from prices to validated, rebalanced weights in a single function call
- **[Preprocessing](guide/preprocessing.md)** -- Data validation, three-group outlier treatment, sector imputation, OLS regression imputation
- **[Pre-selection](guide/pre-selection.md)** -- Asset filtering pipeline: completeness, variance, correlation, dominance, expiry
- **[Moments](guide/moments.md)** -- 5 expected return + 11 covariance estimators with HMM regime blending, DMM, and multi-period scaling
- **[Views](guide/views.md)** -- Black-Litterman, Entropy Pooling (9 view types), Opinion Pooling with omega calibration
- **[Optimization](guide/optimization.md)** -- 10+ models: Mean-Risk, Risk Budgeting, HRP/HERC/NCO, robust ellipsoidal, DR-CVaR, regime-conditional
- **[Validation](guide/validation.md)** -- Walk-Forward, Combinatorial Purged CV, Multiple Randomized CV
- **[Scoring](guide/scoring.md)** -- 19 ratio measures for model selection (Sharpe, Sortino, Calmar, CVaR ratio, ...)
- **[Tuning](guide/tuning.md)** -- Grid and randomized search with temporal CV defaults
- **[Rebalancing](guide/rebalancing.md)** -- Calendar-based, threshold-based, and hybrid rebalancing with turnover/cost utilities
- **[Factors](guide/factors.md)** -- 17 factors across 9 groups: construction, standardization, scoring, selection, regime tilts, validation
- **[Synthetic Data](guide/synthetic.md)** -- Vine copula models for scenario generation and conditional stress testing
- **[Universe Screening](guide/universe.md)** -- 8 investability screens with hysteresis entry/exit thresholds

## Design Principles

**Config + Factory pattern**: Every module uses frozen `@dataclass` configs that hold only serializable primitives/enums. Factory functions create the actual estimator objects. This separation keeps configs serializable for storage, logging, and hyperparameter sweeps.

**sklearn compatibility**: All transformers follow the `BaseEstimator + TransformerMixin` API and compose in `sklearn.pipeline.Pipeline`. This means the full pre-selection + optimization chain can be cross-validated, tuned, and serialized as a single sklearn object.

**skfolio foundation**: Optimization models wrap [skfolio](https://skfolio.org/) estimators — a mature library for portfolio optimization with the sklearn API. The optimizer library adds regime blending, robust uncertainty sets, factor research, and rebalancing on top.

## Quick Start

```bash
pip install -e ".[dev]"
```

```python
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline
from optimizer.validation import WalkForwardConfig

optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
result = run_full_pipeline(
    prices=price_df,
    optimizer=optimizer,
    cv_config=WalkForwardConfig.for_quarterly_rolling(),
)

print(result.weights)          # pd.Series of asset weights
print(result.summary)          # dict with Sharpe, max drawdown, etc.
print(result.backtest)         # out-of-sample MultiPeriodPortfolio
```

See the [Quickstart guide](getting-started/quickstart.md) for more examples.

## Pipeline Data Flow

```
prices → returns → [preprocess → pre-select → optimize] → backtest → weights
                    └──── sklearn Pipeline ────┘
```

The pipeline follows a linear data flow. Prices are converted to returns **outside** the pipeline (semantic change), then everything inside is a single sklearn `Pipeline` that can be cross-validated and tuned.

See the [Pipeline Overview](guide/pipeline.md) for architectural details.
