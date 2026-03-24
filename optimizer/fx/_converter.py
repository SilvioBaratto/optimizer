"""FX price conversion transformer."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from optimizer.exceptions import DataError
from optimizer.fx._rates import align_fx_rates

logger = logging.getLogger(__name__)


class FxPriceConverter(BaseEstimator, TransformerMixin):
    """Convert local-currency prices to base-currency prices.

    Multiplies each ticker's price series by the appropriate FX rate
    to express all prices in a single base currency.  Tickers already
    denominated in the base currency are passed through unchanged.

    This transformer operates on *prices* (not returns) and must be
    applied **before** ``prices_to_returns()``.

    Parameters
    ----------
    base_currency : str
        Target base currency ISO code (e.g. ``"EUR"``).
    currency_map : dict[str, str]
        Mapping of ticker → ISO currency code.
    fx_rates : pd.DataFrame
        Pre-loaded FX rate DataFrame indexed by date, with one column
        per foreign currency.  Each column holds the rate expressed as
        units-of-base per one unit-of-foreign.  For example, if
        base is EUR and column is ``"GBP"``, values are EUR per 1 GBP
        (≈ 1.16).
    fill_limit : int
        Forward-fill limit for aligning FX rates to the price index.
    require_full_coverage : bool
        If ``True``, raise ``DataError`` when any non-base currency
        lacks FX rate data.
    """

    base_currency: str
    currency_map: dict[str, str]
    fx_rates: pd.DataFrame
    fill_limit: int
    require_full_coverage: bool

    def __init__(
        self,
        base_currency: str = "EUR",
        currency_map: dict[str, str] | None = None,
        fx_rates: pd.DataFrame | None = None,
        fill_limit: int = 5,
        require_full_coverage: bool = False,
    ) -> None:
        self.base_currency = base_currency.upper()
        self.currency_map = currency_map or {}
        self.fx_rates = fx_rates if fx_rates is not None else pd.DataFrame()
        self.fill_limit = fill_limit
        self.require_full_coverage = require_full_coverage

    def fit(self, X: pd.DataFrame, y: object = None) -> FxPriceConverter:
        """Validate FX rate coverage and align rates to the price index.

        Parameters
        ----------
        X : pd.DataFrame
            Price matrix (dates x tickers).
        y : ignored
            Not used; present for sklearn API compatibility.

        Returns
        -------
        self
        """
        self._validate_input(X)
        self.n_features_in_: int = X.shape[1]
        self.feature_names_in_: np.ndarray = np.asarray(X.columns)

        # Identify currencies that need conversion
        foreign_tickers: dict[str, str] = {}
        for ticker in X.columns:
            ccy = self.currency_map.get(ticker, self.base_currency).upper()
            if ccy != self.base_currency:
                foreign_tickers[ticker] = ccy

        self.foreign_tickers_: dict[str, str] = foreign_tickers

        # Determine which FX columns we need
        needed = set(foreign_tickers.values())
        available = set(self.fx_rates.columns) if not self.fx_rates.empty else set()
        self.missing_currencies_: set[str] = needed - available

        if self.missing_currencies_:
            msg = (
                f"Missing FX rates for currencies: {self.missing_currencies_}. "
                f"Available: {available}."
            )
            if self.require_full_coverage:
                raise DataError(msg)
            logger.warning(msg + " Affected tickers will not be converted.")

        # Align FX rates to the price index
        if not self.fx_rates.empty:
            self.fx_aligned_: pd.DataFrame = align_fx_rates(
                self.fx_rates, X.index, fill_limit=self.fill_limit
            )
        else:
            self.fx_aligned_ = pd.DataFrame(index=X.index)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert prices from local currencies to the base currency.

        Parameters
        ----------
        X : pd.DataFrame
            Price matrix (dates x tickers) in local currencies.

        Returns
        -------
        pd.DataFrame
            Price matrix with all values expressed in the base currency.
        """
        check_is_fitted(self)
        self._validate_input(X)

        out = X.copy()

        for ticker, ccy in self.foreign_tickers_.items():
            if ticker not in out.columns:
                continue
            if ccy in self.missing_currencies_:
                continue
            if ccy not in self.fx_aligned_.columns:
                continue

            rate = self.fx_aligned_[ccy].reindex(out.index)
            out[ticker] = out[ticker] * rate

        # Warn about tickers with NaN prices due to fill_limit exhaustion
        nan_mask = out.isnull() & ~X.isnull()
        if nan_mask.any().any():
            affected = nan_mask.any(axis=0)
            affected_tickers = list(affected[affected].index)
            total_nan_dates = int(nan_mask.sum().sum())
            logger.warning(
                "FxPriceConverter: %d tickers have %d NaN prices introduced by FX "
                "conversion (fill_limit=%d). Affected tickers: %s. "
                "Consider increasing fill_limit or using a fallback rate strategy.",
                len(affected_tickers),
                total_nan_dates,
                self.fill_limit,
                affected_tickers,
            )

        n_converted = sum(
            1
            for t, c in self.foreign_tickers_.items()
            if t in out.columns and c not in self.missing_currencies_
        )
        logger.info(
            "Converted %d/%d tickers to %s.",
            n_converted,
            len(out.columns),
            self.base_currency,
        )

        return out

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return feature names (pass-through)."""
        check_is_fitted(self)
        return self.feature_names_in_

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise DataError(
                f"FxPriceConverter requires a pandas DataFrame, "
                f"got {type(X).__name__}"
            )
