"""FX return decomposition utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from optimizer.exceptions import DataError
from optimizer.fx._minor_units import normalize_currency_code


@dataclass
class FxReturnDecomposition:
    """Decomposition of portfolio returns into stock and FX components.

    The total return for a foreign-currency asset is::

        r_total = r_local + r_fx + r_local * r_fx

    where ``r_local`` is the return in the asset's local currency and
    ``r_fx`` is the return of the foreign currency vs the base currency.

    For base-currency assets, ``r_fx = 0`` and ``r_total = r_local``.

    Attributes
    ----------
    total_returns : pd.DataFrame
        Base-currency total returns (dates x tickers).
    local_returns : pd.DataFrame
        Local-currency stock returns (dates x tickers).
    fx_returns : pd.DataFrame
        FX contribution (dates x tickers).  Zero for base-currency
        tickers.
    cross_terms : pd.DataFrame
        Interaction term ``r_local * r_fx`` (dates x tickers).
    currency_map : dict[str, str]
        Ticker → ISO currency code mapping used.
    base_currency : str
        Base currency for the decomposition.
    """

    total_returns: pd.DataFrame
    local_returns: pd.DataFrame
    fx_returns: pd.DataFrame
    cross_terms: pd.DataFrame
    currency_map: dict[str, str]
    base_currency: str

    def hedged_returns(self, hedge_ratio: float = 1.0) -> pd.DataFrame:
        """Return currency-hedged total returns.

        A currency hedge removes a fraction ``hedge_ratio`` of the FX
        contribution (FX return plus the local/FX cross term) from the
        base-currency total return::

            r_hedged = r_local + (1 - hedge_ratio) * (r_fx + r_cross)

        - ``hedge_ratio=1.0`` (full hedge) returns the local-currency
          returns (FX risk fully removed).
        - ``hedge_ratio=0.0`` (unhedged) returns the base-currency total
          returns.

        Parameters
        ----------
        hedge_ratio : float
            Fraction of FX exposure hedged, in ``[0, 1]``.  Values
            outside the unit interval are allowed (over/under hedging)
            but emit no special handling.

        Returns
        -------
        pd.DataFrame
            Hedged returns (dates x tickers).

        Raises
        ------
        DataError
            If ``hedge_ratio`` is not finite.
        """
        if not pd.notna(hedge_ratio) or hedge_ratio in (float("inf"), float("-inf")):
            raise DataError(f"hedge_ratio must be finite, got {hedge_ratio!r}.")
        return self.local_returns + (1.0 - hedge_ratio) * (
            self.fx_returns + self.cross_terms
        )

    def cumulative_contributions(self) -> pd.DataFrame:
        """Aggregate compounded return contribution per ticker.

        Compounds each component series over the full window and returns
        a per-ticker summary.  The ``total`` column equals
        ``(1 + total_returns).prod() - 1`` and the local/fx/cross columns
        are the compounded component series; they sum only approximately
        to ``total`` because compounding is multiplicative.

        Returns
        -------
        pd.DataFrame
            Indexed by ticker with columns
            ``["local", "fx", "cross", "total"]``.
        """

        def _compound(frame: pd.DataFrame) -> pd.Series:
            return (1.0 + frame).prod(axis=0) - 1.0

        return pd.DataFrame(
            {
                "local": _compound(self.local_returns),
                "fx": _compound(self.fx_returns),
                "cross": _compound(self.cross_terms),
                "total": _compound(self.total_returns),
            }
        )


def decompose_fx_returns(
    local_prices: pd.DataFrame,
    base_prices: pd.DataFrame,
    fx_rates_aligned: pd.DataFrame,
    currency_map: dict[str, str],
    base_currency: str,
) -> FxReturnDecomposition:
    """Decompose total returns into local, FX, and cross components.

    Parameters
    ----------
    local_prices : pd.DataFrame
        Price matrix in local currencies (dates x tickers).
    base_prices : pd.DataFrame
        Price matrix in base currency (dates x tickers), as produced
        by :class:`FxPriceConverter`.
    fx_rates_aligned : pd.DataFrame
        FX rates aligned to the price index (from
        :func:`align_fx_rates`).  Columns are currency codes; values
        are units-of-base per one unit-of-foreign.
    currency_map : dict[str, str]
        Ticker → ISO currency code mapping.
    base_currency : str
        Base currency ISO code.

    Returns
    -------
    FxReturnDecomposition
        Decomposition with total, local, FX, and cross-term returns.
    """
    base_ccy, _base_scale = normalize_currency_code(base_currency)

    # Compute returns from prices.  Returns are scale-invariant, so the
    # minor-unit rescaling applied to prices cancels in pct_change and the
    # decomposition identity r_total = r_local + r_fx + r_local*r_fx still
    # holds exactly — the only sub-unit concern here is resolving each ticker
    # to its *major* currency so the correct FX column is found.
    local_returns = local_prices.pct_change().iloc[1:]
    total_returns = base_prices.pct_change().iloc[1:]

    # Case-insensitive currency-column lookup on the aligned FX frame.
    fx_col_by_ccy = {str(c).upper(): c for c in fx_rates_aligned.columns}

    # Build per-ticker FX return series
    fx_returns = pd.DataFrame(
        0.0, index=local_returns.index, columns=local_returns.columns
    )

    for ticker in local_returns.columns:
        ccy, _scale = normalize_currency_code(currency_map.get(ticker, base_ccy))
        if ccy == base_ccy:
            continue
        col = fx_col_by_ccy.get(ccy)
        if col is None:
            continue
        # Compute the FX return on the FULL aligned rate series *before*
        # slicing, so the first return date uses the prior (dropped) price
        # date's rate rather than being lost to a leading NaN.  With the
        # correct r_fx the identity r_total = r_local + r_fx + r_local*r_fx
        # holds exactly for foreign assets.
        fx_ret = fx_rates_aligned[col].pct_change()
        fx_returns[ticker] = fx_ret.reindex(local_returns.index).fillna(0.0)

    cross_terms = local_returns * fx_returns

    return FxReturnDecomposition(
        total_returns=total_returns,
        local_returns=local_returns,
        fx_returns=fx_returns,
        cross_terms=cross_terms,
        currency_map=currency_map,
        base_currency=base_ccy,
    )
