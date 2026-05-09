# Portfolio Construction Checklist

Essential criteria derived from the European Multi-Factor Portfolio analysis (Jan 2022 – Mar 2026).

---

## Diversification

- [ ] **Geographic diversification** — no single region above 60-70%; include US, EM, Asia-Pacific exposure
- [ ] **Sector concentration** — no single sector above 15%; target HHI < 0.12
- [ ] **Top-4 holdings** — combined weight below 30% to limit idiosyncratic risk
- [ ] **Healthcare exposure** — minimum 8-12% as a defensive/recession buffer
- [ ] **Technology exposure** — minimum 10-12% to avoid structural underperformance vs global indices
- [ ] **No absent sectors** — every major MSCI World sector should have some representation

## Risk Management

- [ ] **Max drawdown** — target below benchmark (SPY -22% historical reference)
- [ ] **Annualized volatility** — target below or equal to benchmark
- [ ] **Currency risk** — if >30% in a foreign currency, evaluate partial FX hedge (25-50%)
- [ ] **Single-stock cap** — max 10% per position
- [ ] **Minimum position size** — eliminate positions below 2% (cost/benefit is negative)

## Risk-Adjusted Performance

- [ ] **Sharpe ratio** — target > 1.0; if > 2.0, investigate for overfitting
- [ ] **Sortino ratio** — target > 1.5; favorable upside/downside asymmetry
- [ ] **Information ratio** — target > 0.5 vs benchmark
- [ ] **Downside volatility** — should be < 75% of total volatility (favorable skew)

## Factor Model

- [ ] **Academically validated factors** — Quality, Value, Momentum, Low Volatility, Illiquidity
- [ ] **No ad-hoc factors** — all factors backed by peer-reviewed literature
- [ ] **Publication lag** — align factor data to point-in-time to prevent look-ahead bias
- [ ] **IC validation** — run factor validation with Newey-West t-stats and Benjamini-Hochberg correction

## Validation

- [ ] **Out-of-sample only** — never evaluate on in-sample data
- [ ] **OOS length** — minimum 8-10 years across complete market cycles for robust validation
- [ ] **Walk-forward** — prefer walk-forward CV over single train/test split
- [ ] **Hockey stick check** — verify outperformance is distributed across time, not concentrated in one period
- [ ] **Document Rf assumption** — explicitly state risk-free rate used in Sharpe calculation

## Cost Awareness

- [ ] **Transaction taxes** — account for Stamp Duty (UK 0.5%), FTT (France 0.3%)
- [ ] **Bid-ask spread** — estimate weighted average across cap segments
- [ ] **FX conversion costs** — budget ~0.10-0.15% annually for multi-currency portfolios
- [ ] **Total cost budget** — target < 1.0% annualized with quarterly rebalancing
- [ ] **Net-of-tax return** — compute after local capital gains tax (e.g., 26% Italy)

## Rebalancing

- [ ] **Defined frequency** — quarterly rebalancing as baseline
- [ ] **Drift thresholds** — trigger rebalancing if any position drifts > 5% from target
- [ ] **Document methodology** — rebalancing frequency and criteria explicitly stated
- [ ] **Cost-aware** — factor transaction costs into rebalancing decisions
