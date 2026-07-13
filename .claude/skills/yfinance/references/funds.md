# Funds Data (ETFs & Mutual Funds)

When the ticker represents a fund, `Ticker` exposes an extra `fund_*` surface derived from Yahoo's fund pages. These return `dict` or `DataFrame` depending on the metric shape.

```python
ticker = yf.Ticker("SPY")

ticker.fund_overview             # dict — description, family, category
ticker.fund_top_holdings         # DataFrame — top holdings with % weight
ticker.fund_sector_weightings    # DataFrame — sector allocation
ticker.fund_asset_allocation     # dict — stocks / bonds / cash / other %
ticker.fund_performance          # dict — trailing & annual returns
ticker.fund_holding_info         # dict — turnover, inception date, AUM
ticker.fund_equity_holdings      # dict — P/E, P/B, price/sales of holdings
ticker.fund_bond_holdings        # dict — duration, credit quality, maturity
ticker.fund_bond_ratings         # dict — credit rating breakdown
```

## Detecting a fund

`ticker.quote_type` or `ticker.fast_info.quoteType` returns `"ETF"` or `"MUTUALFUND"` for funds; guard `fund_*` access with that check to avoid Yahoo returning empty structures for equities.

```python
if ticker.fast_info.quoteType in ("ETF", "MUTUALFUND"):
    holdings = ticker.fund_top_holdings
```

## Typical weights

`fund_top_holdings` is the most-used — DataFrame indexed by ticker with a `Holding Percent` column. Multiply by NAV to dollarize; sum should be close to but not exactly 1.0 (Yahoo typically only reports top 10).

## Related

For screening funds by size, performance, or category, see `screener.md` — `FundQuery` and (v1.3.0+) `ETFQuery`.
