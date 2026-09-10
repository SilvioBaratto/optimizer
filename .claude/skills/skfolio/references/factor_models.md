# Factor Models (Time-Series + Characteristics)

skfolio 1.0 ships **two** factor-model estimators plus the modules that feed the cross-sectional one.

| Estimator | Style | Input | Import |
|---|---|---|---|
| `TimeSeriesFactorModel` | Time-series (regress assets on observed factor returns) | asset returns `X` + `factors=` | `skfolio.prior` |
| `CharacteristicsFactorModel` | Cross-sectional / BARRA-style (build exposures from descriptors) | an `AssetPanel` via `characteristics=` | `skfolio.prior` |

Both produce a fitted `FactorModel` **container** (loading matrix, factor moments, idiosyncratic covariance, exposures, factor/idio returns) and a `return_distribution_`.

> ⚠️ `FactorModel` in 1.0 is the **result container**, not an estimator. Instantiate `TimeSeriesFactorModel()` or `CharacteristicsFactorModel()` as the `prior_estimator`. Factor returns for the time-series model are keyword-only: `fit(X, factors=...)`.

## TimeSeriesFactorModel

See `priors.md`. `fit(X, factors=factor_returns)`; `factors` from `prices_to_returns(prices, factor_prices)`.

---

## CharacteristicsFactorModel (NEW in 1.0)

Cross-sectional factor model: per-date WLS regression of asset returns on lagged characteristic exposures. Needs an `AssetPanel` of characteristics, not a returns DataFrame.

```python
CharacteristicsFactorModel(
    *,
    factors,                              # list[(name, BaseFactorExposure)]
    currency_factor=None,
    exposure_lag=1,                       # returns at t regress on exposures at t-1
    cs_regressor=None,                    # default CSLinearRegression
    neutralize_against=None,              # {"non_linear_size": ["size"], ...}
    constrained_families=None,            # [("industry", None)] zero-sum constraint
    benchmark_mcap_power=1.0,
    regression_mcap_power=0.5,
    inv_idio_variance_weight_shrinkage=0.0,
    inv_idio_variance_max_weight_ratio=20.0,
    factor_prior_estimator=None,          # e.g. EmpiricalPrior(...) on the factors
    alpha_estimator=None,
    spanned_alpha_shrinkage=1.0,
    orthogonal_alpha_confidence=1.0,
    idio_variance_estimator=None,
    idio_corr_estimator=None,
    idio_corr_threshold=0.0,              # sparse idio covariance via corr thresholding
    max_history=None,
    min_regression_assets=None,           # default ~ max(2K, 30), K = #factors
    n_jobs=1,
)
```

Fit contract (characteristics is **keyword-only**):

```python
model.fit(characteristics=panel)                     # X/y are None
model.partial_fit(characteristics=new_panel)         # incremental / online
# priors expose no predict(); read model.return_distribution_ / model.factor_model_,
# or feed the fitted prior into an optimizer via prior_estimator=
```

Fitted: `return_distribution_` (ReturnDistribution), `factor_model_` (`FactorModel` container with `summary()`, `exposures_df()`, `factor_returns_df()`, `cs_regression_scores()`, `exposure_vif()`, and many `plot_*` diagnostics), plus fitted sub-estimators `cs_regressor_`, `factor_prior_estimator_`, `idio_variance_estimator_`.

### Worked example

```python
from skfolio.containers import AssetPanel
from skfolio.datasets import make_synthetic_characteristics
from skfolio.descriptor import (
    BookToPrice, SalesToPrice, CashFlowToPrice, EWMarketBeta, EWMomentum, LogMarketCap,
)
from skfolio.factor_exposure import (
    GlobalFactor, OneHotCategoricalFactors, FixedWeightedFactor, DerivedFactor,
)
from skfolio.moments import EWMu, RegimeAdjustedEWCovariance
from skfolio.prior import CharacteristicsFactorModel, EmpiricalPrior

panel = make_synthetic_characteristics(n_assets=200, n_observations=1000)

month, half_year, year = 21, 126, 252

market   = GlobalFactor(family="market")                     # unit exposure = regression intercept
industry = OneHotCategoricalFactors(category="industry", family="industry")
beta     = FixedWeightedFactor(descriptors=[("beta", EWMarketBeta(half_life=year))],
                               transform_by_group="industry")
momentum = FixedWeightedFactor(descriptors=[("mom", EWMomentum(half_life=half_year, skip=month))],
                               transform_by_group="industry")
size     = FixedWeightedFactor(descriptors=[("log_mcap", LogMarketCap())],
                               transform_by_group="industry")
nl_size  = DerivedFactor(source="size", func=lambda x: x ** 3, transform_by_group="industry")
value    = FixedWeightedFactor(
    descriptors=[("btp", BookToPrice()), ("stp", SalesToPrice()), ("cfp", CashFlowToPrice())],
    weights=[0.8, 0.1, 0.1], transform_by_group="industry",
)

model = CharacteristicsFactorModel(
    factors=[
        ("market", market), ("industry", industry), ("beta", beta),
        ("momentum", momentum), ("size", size), ("non_linear_size", nl_size), ("value", value),
    ],
    neutralize_against={"non_linear_size": ["size"]},
    constrained_families=[("industry", None)],
    exposure_lag=1,
    factor_prior_estimator=EmpiricalPrior(
        mu_estimator=EWMu(half_life=year),
        covariance_estimator=RegimeAdjustedEWCovariance(half_life=half_year, corr_half_life=year),
    ),
    n_jobs=-1,
)
model.fit(characteristics=panel)
```

---

## `skfolio.factor_exposure` — exposure estimators

Build the cross-sectional exposure of one factor. Base class `BaseFactorExposure`.

| Class | Purpose | Key params |
|---|---|---|
| `GlobalFactor` | Unit exposure for all assets (market / intercept) | `family="market"` |
| `OneHotCategoricalFactors` | Binary membership from a categorical panel field | `category=`, `family=` |
| `FixedWeightedFactor` | Fixed weighted combo of descriptors | `descriptors=[(name, Descriptor())]`, `weights=`, `transform_by_group=`, `outlier_transformer=CSWinsorizer()`, `scoring_transformer=CSStandardScaler()`, `min_coverage=0.0` |
| `DerivedFactor` | Function of another factor's exposure | `source=<factor name>`, `func=`, `transform_by_group=` |

`transform_by_group="industry"` standardizes within a categorical field (which must exist on the panel). `DerivedFactor.source`, `neutralize_against`, and `constrained_families` reference **factor names** declared in `factors=`, not descriptor or asset names.

## `skfolio.descriptor` — fundamental / price descriptors

Base class `BaseDescriptor`; ~46 concrete descriptors consuming an `AssetPanel`, returning `(n_observations, n_assets)`. Common ones:

`BookToPrice`, `SalesToPrice`, `CashFlowToPrice`, `EarningsToPrice`, `ForwardEarningsToPrice`, `DividendToPrice`, `EbitdaToEnterpriseValue`, `LogMarketCap`, `EWMarketBeta`, `EWMomentum`, `EWVolatility`, `EWResidualVolatility`, `EWDownsideBeta`, `EWAmihudIlliquidity`, `EWShareTurnover`, `ReturnOnEquity`, `ReturnOnAssets`, `GrossProfitability`, `GrossMargin`, `AssetTurnover`, `DebtToAssets`, `BookLeverage`, `MarketLeverage`, `SalesGrowthRate`, `AssetsGrowthRate`, `IssuanceGrowthRate`, `EarningsChangeToPrice`, `ShareholderYield`, `Reversal`, `RollingMomentum`, `MaxReturn`, `ShortInterest`, `DaysToCover`, `AnalystDispersionToPrice`, `Passthrough`, ... (see API reference for the full 46).

- EW/time-based descriptors take `half_life`, `skip`, `min_periods` (e.g. `EWMomentum(half_life=126, skip=21)`, `EWResidualVolatility(beta_half_life=...)`).
- Fundamental/growth descriptors take a `lag` (e.g. `SalesGrowthRate(lag=252)`).

## `skfolio.containers.AssetPanel`

Aligned wide-format `(observations × assets)` multi-field container — the input to `CharacteristicsFactorModel`. See `data_representation.md` for the full contract.

```python
from skfolio.containers import AssetPanel

AssetPanel(fields, observations, asset_names, active_mask=None, estimation_mask=None)
# ergonomic construction:
panel.add_2d_field(name, values)
panel.add_categorical_field(name, values, levels=[...])
panel.save(path); AssetPanel.load(path)
# props: n_observations, n_assets, n_fields, shape, ndim
```

Generate a synthetic one for prototyping (testing/examples only — stale, not investable):

```python
from skfolio.datasets import make_synthetic_characteristics
panel = make_synthetic_characteristics(n_assets=500, n_observations=2520, random_state=42)
```

## Orthogonal uncertainty sets

`OrthogonalMuUncertaintySet` / `OrthogonalCovarianceUncertaintySet` require a factor-model `prior_estimator` (e.g. `CharacteristicsFactorModel`) — they confine estimation-error uncertainty to the space orthogonal to the loading matrix. See `distance_clustering.md`.
