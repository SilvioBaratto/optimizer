"""FX price conversion transformer."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from optimizer.exceptions import DataError
from optimizer.fx._minor_units import normalize_currency_code
from optimizer.fx._rates import align_fx_rates

logger = logging.getLogger(__name__)


class FxPriceConverter(BaseEstimator, TransformerMixin):
    """Convert local-currency prices to base-currency prices.

    For each ticker this (1) divides the quoted price by its minor-unit
    scale so sub-unit listings (e.g. ``GBp`` pence, ``ZAc`` cents, ``ILA``
    agorot) are expressed in the major unit, then (2) multiplies by the
    appropriate FX rate to express all prices in a single base currency.
    Tickers already denominated in the base *major* currency are passed
    through unchanged (subject only to minor-unit rescaling).

    The minor-unit step is essential for this DB: ``price_history.price_unit``
    is the listing currency as-is and yfinance keeps pence/cents/agorot in
    their sub-unit, so a code-only FX conversion would be a 100x error for
    London / Johannesburg / Tel Aviv listings.  See
    :mod:`optimizer.fx._minor_units`.

    This transformer operates on *prices* (not returns) and must be
    applied **before** ``prices_to_returns()`` — scale differences make
    converting returns instead of prices incorrect.

    Parameters
    ----------
    base_currency : str
        Target base currency ISO code (e.g. ``"EUR"``).
    currency_map : dict[str, str]
        Mapping of ticker → currency / price-unit code as stored in the DB
        (``price_history.price_unit``).  Minor-unit codes (``GBp``, ``ZAc``,
        ``ILA``, ...) are recognised and rescaled; do **not** pre-normalise
        them to the major code, or the sub-unit scale would be lost.
    fx_rates : pd.DataFrame
        Pre-loaded FX rate DataFrame indexed by date, with one column
        per foreign **major** currency.  Each column holds the rate expressed
        as units-of-base per one unit-of-foreign.  For example, if
        base is EUR and column is ``"GBP"``, values are EUR per 1 GBP
        (≈ 1.16).  (Pence tickers are rescaled to GBP first, so only the
        major-unit ``GBP`` rate is needed — never a ``GBp`` column.)
    fill_limit : int
        Forward-fill limit for aligning FX rates to the price index.
    require_full_coverage : bool
        If ``True``, raise ``DataError`` when any non-base currency
        lacks FX rate data.
    """

    base_currency: str
    currency_map: dict[str, str] | None
    fx_rates: pd.DataFrame | None
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
        # sklearn convention: __init__ stores constructor arguments
        # verbatim (no mutation / normalisation) so that ``clone`` and
        # ``get_params`` round-trip exactly.  Normalisation happens in fit.
        self.base_currency = base_currency
        self.currency_map = currency_map
        self.fx_rates = fx_rates
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

        # Normalise constructor arguments (kept verbatim on the instance).
        # The base currency is resolved to its major unit; a base is expected
        # to be a major code (EUR/GBP/USD) so its scale is 1.
        base_ccy, _base_scale = normalize_currency_code(self.base_currency)
        self.base_currency_: str = base_ccy
        currency_map = self.currency_map or {}
        raw_fx_rates = self.fx_rates if self.fx_rates is not None else pd.DataFrame()

        # Resolve every ticker to (major currency, minor-unit scale).  The
        # scale rescales sub-unit prices (pence/cents/agorot) to the major
        # unit and applies to base-currency tickers too; the FX step only
        # applies to tickers whose *major* currency differs from the base.
        foreign_tickers: dict[str, str] = {}
        ticker_scale: dict[str, int] = {}
        minor_unit_tickers: dict[str, int] = {}
        for ticker in X.columns:
            raw_code = currency_map.get(ticker, base_ccy)
            major, scale = normalize_currency_code(raw_code)
            ticker_scale[ticker] = scale
            if scale != 1:
                minor_unit_tickers[ticker] = scale
            if major != base_ccy:
                foreign_tickers[ticker] = major

        self.foreign_tickers_: dict[str, str] = foreign_tickers
        self.ticker_scale_: dict[str, int] = ticker_scale
        self.minor_unit_tickers_: dict[str, int] = minor_unit_tickers

        if minor_unit_tickers:
            logger.info(
                "Rescaling %d minor-unit ticker(s) to their major unit "
                "(e.g. GBp/ZAc/ILA -> GBP/ZAR/ILS, /100): %s.",
                len(minor_unit_tickers),
                sorted(minor_unit_tickers),
            )

        # Normalise FX rate columns to upper-case currency codes so that
        # currency matching is case-insensitive (e.g. "gbp" vs "GBP").
        if not raw_fx_rates.empty:
            fx_rates = raw_fx_rates.rename(columns=lambda c: str(c).upper())
        else:
            fx_rates = raw_fx_rates

        # Determine which FX columns we need
        needed = set(foreign_tickers.values())
        available = set(fx_rates.columns) if not fx_rates.empty else set()
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
        if not fx_rates.empty:
            self.fx_aligned_: pd.DataFrame = align_fx_rates(
                fx_rates, X.index, fill_limit=self.fill_limit
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

        # Cast to float up front: DB Numeric columns arrive as Python Decimal
        # (object dtype), which does not multiply cleanly against float FX
        # rates.  astype yields a new frame and, under pandas copy-on-write,
        # the subsequent per-column assignments never mutate ``X``.
        out = X.astype("float64")

        # 1. Minor-unit rescaling (pence/cents/agorot -> major unit).  Applied
        #    to every ticker, including base-currency sub-unit listings.
        for ticker, scale in self.ticker_scale_.items():
            if scale != 1 and ticker in out.columns:
                out[ticker] = out[ticker] / scale

        # 2. FX conversion (major local currency -> base currency).
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
            self.base_currency_,
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
                f"FxPriceConverter requires a pandas DataFrame, got {type(X).__name__}"
            )
