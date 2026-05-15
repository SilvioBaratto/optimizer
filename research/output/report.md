# Research Run Report

**Assembly hash:** `18ee6b51913d2546`

## 1. Regime & Tilts

**Current regime:** `expansion`

| Factor | Tilt |
| --- | ---: |
| FactorGroupType.VALUE | 1.1739 |
| FactorGroupType.PROFITABILITY | 0.9783 |
| FactorGroupType.INVESTMENT | 0.9783 |
| FactorGroupType.MOMENTUM | 1.1739 |
| FactorGroupType.LOW_RISK | 0.7826 |
| FactorGroupType.LIQUIDITY | 0.9783 |
| FactorGroupType.DIVIDEND | 0.9783 |
| FactorGroupType.SENTIMENT | 0.9783 |
| FactorGroupType.OWNERSHIP | 0.9783 |


## 2. Factor IS / OOS IC

| Factor | IS Mean IC | IS t-stat | IS Significant | OOS Mean IC |
| --- | ---: | ---: | :---: | ---: |
| book_to_price | -0.1110 | -8.62 | yes | -0.1201 |
| earnings_yield | -0.1017 | -7.82 | yes | -0.1138 |
| cash_flow_yield | -0.0819 | -4.03 | yes | -0.0962 |
| gross_profitability | -0.0263 | -1.66 | yes | -0.0342 |
| roe | -0.0462 | -4.32 | yes | -0.0494 |
| accruals | 0.0602 | 2.04 | yes | 0.0393 |
| momentum_12_1 | 0.0161 | 1.04 | no | 0.0311 |
| volatility | -0.1029 | -4.84 | yes | -0.1021 |
| beta | -0.0629 | -2.46 | yes | -0.0608 |
| amihud_illiquidity | 0.0742 | 4.35 | yes | 0.0814 |
| dividend_yield | -0.0603 | -4.65 | yes | -0.0729 |


## 3. Optimizer Config Diff vs Default

_No deviations from defaults._

## 4. Binding Constraints

_None._

## 5. Top-4 Retighten Trace

_No retighten attempts (Top-4 below threshold on first fit)._

## 6. Hybrid Rebalance Decision

- **Rebalance:** no
- **Reason:** `cold_start`

## 7. 17-Rule Checklist

| # | Rule | Pass | Measured | Target |
| ---: | --- | :---: | --- | --- |
| 1 | No single region > 60% | yes | `35.5%` | `≤ 60%` |
| 2 | No single sector > 15% | yes | `15.0% (Industrials)` | `≤ 15%` |
| 3 | HHI < 0.12 | yes | `0.0558` | `< 0.12` |
| 4 | Top-4 holdings < 30% | yes | `25.8%` | `< 30%` |
| 5 | Health Care exposure ≥ 8% | yes | `12.9%` | `≥ 8%` |
| 6 | Information Technology exposure ≥ 10% | yes | `15.0%` | `≥ 10%` |
| 7 | At least 8/11 sectors present | yes | `8/11 (Consumer Defensive, Utilities, Real Estate)` | `≥ 8/11` |
| 8 | Single-stock cap ≤ 10% | yes | `6.5%` | `≤ 10%` |
| 9 | Min position ≥ 2% | yes | `3.6%` | `≥ 2%` |
| 10 | Max drawdown > -22% | yes | `-15.7%` | `> -22%` |
| 11 | Vol ≤ benchmark vol | yes | `16.3% vs 16.5%` | `≤ benchmark` |
| 12 | Sharpe ∈ (1.0, 2.0) | yes | `1.586` | `∈ (1.0, 2.0)` |
| 13 | Sortino > 1.5 | yes | `2.330` | `> 1.5` |
| 14 | Info Ratio > 0.5 | yes | `0.536` | `> 0.5` |
| 15 | Downside vol < 75% x total vol | yes | `11.1% vs 75% x 16.3% = 12.2%` | `< 75% total` |
| 16 | Total cost ≤ 100 bps | yes | `23.1 bps` | `≤ 100 bps` |
| 17 | OOS span ≥ 1.5 years | yes | `1.95 yrs` | `≥ 1.5 yrs` |


## 8. Metrics

### Portfolio (gross)

| KPI | Value |
| --- | ---: |
| Ann. Return | 0.2630 |
| Ann. Vol | 0.1633 |
| Sharpe (rf) | 1.6089 |
| Sortino | 2.3655 |
| Info Ratio | 0.5549 |
| Downside Vol | 0.1111 |
| Max Drawdown | -0.1567 |


### Portfolio

| KPI | Value |
| --- | ---: |
| Ann. Return | 0.2627 |
| Ann. Vol | 0.1634 |
| Sharpe (rf) | 1.6068 |
| Sortino | 2.3624 |
| Info Ratio | 0.5549 |
| Downside Vol | 0.1111 |
| Max Drawdown | -0.1567 |


### Portfolio (after-tax)

| KPI | Value |
| --- | ---: |
| Ann. Return | 0.2591 |
| Ann. Vol | 0.1633 |
| Sharpe (rf) | 1.5856 |
| Sortino | 2.3301 |
| Info Ratio | 0.5357 |
| Downside Vol | 0.1111 |
| Max Drawdown | -0.1567 |


### SPY (benchmark)

| KPI | Value |
| --- | ---: |
| Ann. Return | 0.1798 |
| Ann. Vol | 0.1650 |
| Sharpe (rf) | 1.0883 |
| Sortino | 1.3792 |
| Info Ratio | 0.0000 |
| Downside Vol | 0.1302 |
| Max Drawdown | -0.1876 |




## 9. Charts

- ![cumulative_returns](./cumulative_returns.png)
- ![drawdowns](./drawdowns.png)
- ![rolling_sharpe](./rolling_sharpe.png)
- ![sector_allocation](./sector_allocation.png)
- ![country_allocation](./country_allocation.png)
- ![factor_ic](./factor_ic.png)

