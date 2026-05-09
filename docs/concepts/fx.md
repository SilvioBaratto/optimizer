# Multi-Currency FX

## When to use

`optimizer.fx` converts local-currency prices to a single base
currency before they enter the optimization pipeline, and optionally
decomposes total returns into stock-only and FX components. Use it
whenever your universe spans currencies and you need either
consistent base-currency comparisons or hedged-vs-unhedged
attribution.

The submodule is sklearn-pipeline-friendly: `FxPriceConverter`
implements `BaseEstimator + TransformerMixin` and slots into
`sklearn.pipeline.Pipeline` like any other preprocessing step.

## API surface

| Symbol | Purpose |
|--------|---------|
| `FxConfig` | Immutable configuration. |
| `FxConversionMode` | `NONE`, `TO_BASE`, `DECOMPOSE`. |
| `BaseCurrency` | `EUR`, `GBP`, `USD`. |
| `FxDataSource` | `YFINANCE`, `FRED`. |
| `FxPriceConverter` | sklearn transformer. |
| `build_fx_converter(config, **kwargs)` | Factory. |
| `decompose_fx_returns(...)` | Stock vs FX return decomposition. |
| `FxReturnDecomposition` | Decomposition result container. |
| `required_fx_currencies(...)` | List required currency pairs for a universe. |
| `build_fx_pair_ticker(base, quote)` | yfinance pair-symbol helper. |
| `compute_cross_rate(...)` | Cross-rate via USD when direct pair missing. |
| `align_fx_rates(...)` | Forward-fill FX rates onto a price index. |

## Modes

* `NONE` — pass-through. Default; preserves backward compatibility.
* `TO_BASE` — converts every ticker priced in a non-base currency to
  the base currency. Optimizer sees a homogeneous price panel.
* `DECOMPOSE` — converts and additionally returns an
  `FxReturnDecomposition` so callers can attribute total return to
  stock and FX components.

## Composition pattern

```python
import pandas as pd

from optimizer.fx import (
    BaseCurrency,
    FxConfig,
    FxConversionMode,
    FxDataSource,
    build_fx_converter,
)


cfg = FxConfig(
    base_currency=BaseCurrency.USD,
    mode=FxConversionMode.TO_BASE,
    data_source=FxDataSource.YFINANCE,
    fill_limit=5,
)

prices = pd.DataFrame(...)               # local-currency prices
currency_map = {"VOD.L": "GBP", "SAP.DE": "EUR", "AAPL": "USD"}
fx_rates = pd.DataFrame(...)             # quote-per-base columns

converter = build_fx_converter(
    cfg,
    currency_map=currency_map,
    fx_rates=fx_rates,
)
prices_in_base = converter.fit_transform(prices)
```

`currency_map` and `fx_rates` are non-serialisable runtime kwargs and
therefore live on the factory call, not the Config.

## Cross-rates via USD

yfinance direct cross pairs (e.g. GBPEUR=X) are sparse. Set
`FxConfig.cross_via_usd=True` to compute crosses through USD when the
direct pair is unavailable.

## Strict vs lenient coverage

* `require_full_coverage=True` — raise `DataError` if any required FX
  pair lacks sufficient data.
* `require_full_coverage=False` — log a warning and leave the
  affected tickers in their original currency.

## See also

- `optimizer.fx._decomposition` — `decompose_fx_returns` for hedged-
  vs-unhedged attribution analysis.
- skfolio does not natively model FX, so this submodule is an
  optimizer-library extension.
