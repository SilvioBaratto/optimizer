"""FX return decomposition utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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
    base_ccy = base_currency.upper()

    # Compute returns from prices
    local_returns = local_prices.pct_change().iloc[1:]
    total_returns = base_prices.pct_change().iloc[1:]

    # Build per-ticker FX return series
    fx_returns = pd.DataFrame(
        0.0, index=local_returns.index, columns=local_returns.columns
    )

    for ticker in local_returns.columns:
        ccy = currency_map.get(ticker, base_ccy).upper()
        if ccy == base_ccy:
            continue
        if ccy not in fx_rates_aligned.columns:
            continue
        rate = fx_rates_aligned[ccy].reindex(local_returns.index)
        fx_ret = rate.pct_change()
        # The first row of fx_ret is NaN from pct_change on the aligned
        # rates — but local_returns already dropped the first price row,
        # so we align from the second rate observation onward.
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
