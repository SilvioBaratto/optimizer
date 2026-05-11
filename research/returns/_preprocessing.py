"""FX pre-conversion + sklearn return preprocessing pipeline (issue #521).

Two helpers, both small and composable:

* :func:`apply_fx_to_prices` converts a raw local-currency price panel to
  a single base currency using :class:`optimizer.fx.FxPriceConverter`.
  Operates on **prices** and must run BEFORE
  :func:`skfolio.preprocessing.prices_to_returns` (changing FX after
  ``prices_to_returns`` would mix linear stock returns with FX returns
  inconsistently).

* :func:`build_return_preprocessing_pipeline` returns an
  :class:`sklearn.pipeline.Pipeline` of the four time-series preprocessing
  transformers (validate → outlier-treat → sector-impute → regression-impute).
  Operates on **linear returns** only — log returns are not supported.

The two helpers are deliberately NOT fused into a single Pipeline because
``FxPriceConverter`` and the four return transformers operate on different
input domains (prices vs returns) separated by ``prices_to_returns``,
which changes data semantics and must stay outside any sklearn Pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from optimizer.fx import FxPriceConverter
from optimizer.preprocessing import (
    DataValidator,
    OutlierTreater,
    RegressionImputer,
    SectorImputer,
)


def _assert_fx_shape_unchanged(
    input_shape: tuple[int, int],
    output: pd.DataFrame,
) -> None:
    """Raise ``RuntimeError`` if FX conversion altered the panel shape."""
    if output.shape != input_shape:
        raise RuntimeError(
            f"FX conversion changed panel shape: {input_shape} -> {output.shape}."
        )
    if np.isinf(output.to_numpy()).any():
        raise RuntimeError("FX conversion produced inf values.")


def apply_fx_to_prices(
    prices: pd.DataFrame,
    currency_map: dict[str, str],
    fx_rates: pd.DataFrame,
    base_currency: str = "EUR",
) -> pd.DataFrame:
    """Convert ``prices`` to ``base_currency`` via :class:`FxPriceConverter`.

    Parameters
    ----------
    prices : pd.DataFrame
        Dates × tickers price panel in local currencies.
    currency_map : dict[str, str]
        Ticker → ISO currency code.
    fx_rates : pd.DataFrame
        Dates × currency-codes panel of base-per-foreign rates.
    base_currency : str, default="EUR"
        Target base currency.

    Returns
    -------
    pd.DataFrame
        Same shape as ``prices``, all values expressed in ``base_currency``.
    """
    converter = FxPriceConverter(
        base_currency=base_currency,
        currency_map=currency_map,
        fx_rates=fx_rates,
    )
    converted: pd.DataFrame = converter.fit_transform(prices)
    _assert_fx_shape_unchanged(prices.shape, converted)
    return converted


def build_return_preprocessing_pipeline(
    sector_mapping: dict[str, str],
) -> Pipeline:
    """Build the four-step linear-return preprocessing pipeline.

    Steps (in order): ``validate`` → ``outlier`` → ``sector_impute`` →
    ``regression_impute``.  Pipeline expects linear returns as input
    (NOT prices, NOT log returns).

    Parameters
    ----------
    sector_mapping : dict[str, str]
        Ticker → sector mapping consumed by :class:`SectorImputer`.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline(
        steps=[
            ("validate", DataValidator()),
            (
                "outlier",
                OutlierTreater(
                    winsorize_threshold=4.0,
                    remove_threshold=8.0,
                ),
            ),
            ("sector_impute", SectorImputer(sector_mapping=sector_mapping)),
            (
                "regression_impute",
                RegressionImputer(
                    n_neighbors=5,
                    min_train_periods=60,
                ),
            ),
        ]
    )
