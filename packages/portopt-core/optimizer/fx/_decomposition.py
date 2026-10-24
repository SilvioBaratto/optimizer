"""FX return decomposition utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from optimizer.exceptions import DataError
from optimizer.fx._minor_units import normalize_currency_code


def _bridged_returns(prices: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Simple returns between consecutive observed prices, on observed dates.

    Args:
        prices: Price matrix (dates x tickers).
        observed: Boolean mask of the same shape; ``False`` cells are
            treated as missing.

    Returns:
        Returns that are ``NaN`` on unobserved dates and on each ticker's
        first observed date.  The first observation after a gap carries the
        full move since the last observation before it.
    """
    return prices.where(observed).ffill().pct_change().where(observed)


@dataclass
class FxReturnDecomposition:
    """Decomposition of portfolio returns into stock and FX components.

    The total return for a foreign-currency asset is::

        r_total = r_local + r_fx + r_local * r_fx

    where ``r_local`` is the return in the asset's local currency and
    ``r_fx`` is the return of the foreign currency vs the base currency.

    For base-currency assets, ``r_fx = 0`` and ``r_total = r_local``.

    Attributes:
        total_returns: Base-currency total returns (dates x tickers). All
            components are ``NaN`` on dates where any of them is unknown.
        local_returns: Local-currency stock returns (dates x tickers).
        fx_returns: FX contribution (dates x tickers). Zero for
            base-currency tickers.
        cross_terms: Interaction term ``r_local * r_fx`` (dates x tickers).
        currency_map: Ticker → ISO currency code mapping used.
        base_currency: Base currency for the decomposition.
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

        Args:
            hedge_ratio: Fraction of FX exposure hedged, in ``[0, 1]``.
                Values outside the unit interval are allowed
                (over/under hedging) but emit no special handling.

        Returns:
            Hedged returns (dates x tickers).

        Raises:
            DataError: If ``hedge_ratio`` is not finite.
        """
        if not pd.notna(hedge_ratio) or hedge_ratio in (float("inf"), float("-inf")):
            raise DataError(f"hedge_ratio must be finite, got {hedge_ratio!r}.")
        return self.local_returns + (1.0 - hedge_ratio) * (
            self.fx_returns + self.cross_terms
        )

    def cumulative_contributions(self) -> pd.DataFrame:
        """Aggregate compounded return contribution per ticker.

        Compounds each component series over the full window and returns
        a per-ticker summary. The ``total`` column equals
        ``(1 + total_returns).prod() - 1`` and the local/fx/cross columns
        are the compounded component series; they sum only approximately
        to ``total`` because compounding is multiplicative.

        Returns:
            DataFrame indexed by ticker with columns
            ``["local", "fx", "cross", "total"]``.  A component with no
            valid return for a ticker is ``NaN``, not ``0.0``.
        """

        def _compound(frame: pd.DataFrame) -> pd.Series:
            # min_count=1: prod() of an all-NaN column is 1.0, which would
            # report a ticker with no data as a 0% return.
            return (1.0 + frame).prod(axis=0, min_count=1) - 1.0

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

    Returns are taken between consecutive dates on which a ticker's local
    price, base price and FX rate are all known.  A gap therefore yields one
    multi-day return on the first date after it (``NaN`` on the gap dates),
    with local, FX and total measured over the same interval, so the identity
    ``r_total = r_local + r_fx + r_local * r_fx`` holds on every date.

    Args:
        local_prices: Price matrix in local currencies (dates x tickers).
        base_prices: Price matrix in base currency (dates x tickers), as
            produced by `FxPriceConverter`.
        fx_rates_aligned: FX rates aligned to the price index (from
            `align_fx_rates`). Columns are currency codes; values
            are units-of-base per one unit-of-foreign.
        currency_map: Ticker → ISO currency code mapping.
        base_currency: Base currency ISO code.

    Returns:
        Decomposition with total, local, FX, and cross-term returns.
    """
    base_ccy, _base_scale = normalize_currency_code(base_currency)

    # DB Numeric columns arrive as Python Decimal (object dtype), which cannot
    # be multiplied against the float FX returns below; FxPriceConverter casts
    # the same way at its own boundary.
    local_prices = local_prices.astype("float64")
    base_prices = base_prices.astype("float64").reindex_like(local_prices)

    # Resolve each foreign ticker to its aligned FX rate series.  Returns are
    # scale-invariant, so the minor-unit rescaling applied to prices cancels
    # and the decomposition identity r_total = r_local + r_fx + r_local*r_fx
    # still holds exactly -- the only sub-unit concern here is resolving each
    # ticker to its *major* currency so the correct FX column is found.
    fx_col_by_ccy = {str(c).upper(): c for c in fx_rates_aligned.columns}
    rates: dict[str, pd.Series] = {}
    for ticker in local_prices.columns:
        ccy, _scale = normalize_currency_code(currency_map.get(ticker, base_ccy))
        col = fx_col_by_ccy.get(ccy)
        if ccy == base_ccy or col is None:
            continue
        rates[ticker] = fx_rates_aligned[col].reindex(local_prices.index)
    rate_frame = pd.DataFrame(rates, index=local_prices.index)

    # A date is usable for a ticker only when every input to its decomposition
    # is known there.  Returns are then taken between consecutive usable dates,
    # so a missing price (a market holiday on a union calendar) or a missing
    # rate is bridged by one multi-day return for local, FX and total alike
    # instead of dropping that move or booking a zero for it.
    observed = local_prices.notna() & base_prices.notna()
    for ticker in rate_frame.columns:
        observed[ticker] &= rate_frame[ticker].notna()

    local_returns = _bridged_returns(local_prices, observed)
    total_returns = _bridged_returns(base_prices, observed)
    fx_returns = pd.DataFrame(
        0.0, index=local_prices.index, columns=local_prices.columns
    )
    fx_returns = fx_returns.where(local_returns.notna())
    if not rate_frame.empty:
        fx_returns[rate_frame.columns] = _bridged_returns(
            rate_frame, observed[rate_frame.columns]
        )

    known = local_returns.notna() & total_returns.notna() & fx_returns.notna()
    local_returns = local_returns.where(known).iloc[1:]
    total_returns = total_returns.where(known).iloc[1:]
    fx_returns = fx_returns.where(known).iloc[1:]

    cross_terms = local_returns * fx_returns

    return FxReturnDecomposition(
        total_returns=total_returns,
        local_returns=local_returns,
        fx_returns=fx_returns,
        cross_terms=cross_terms,
        currency_map=currency_map,
        base_currency=base_ccy,
    )
