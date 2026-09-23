---
type: concept
title: "Factor Timing and Factor Rotation"
description: Conditioning factor exposures on the state of the world — the result that the optimal factor-timing portfolio is the stochastic discount factor, the robust cross-sectional predictability of equity factors, and the skeptical counterpoint that this predictability is hard to monetize net of costs and estimation error, leaving relative valuation as the one robust signal.
tags: [factor-timing, factor-rotation, stochastic-discount-factor, predictability, valuation, near-arbitrage, smart-beta, regimes]
sources:
  - id: openwiki-source-7385803e2ce5b9d59764879f
    resource: repo://docs/27_factor_timing_and_factor_rotation.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Factor Timing and Factor Rotation

Factor performance is strongly time-varying — small-cap and value lagged badly in
the late 1990s tech run-up, then were rewarded after the bubble burst, while
quality and low-volatility led late in the sample. This chapter asks the normative
question that [factor models](./factor-models.md) leave open: can one *time* factor
exposures to beat a fixed-weight sleeve? Its answer is balanced — timing is possible
at the margin, but chasing recent performance is dangerous, and discipline lives in
valuation and the absence of near-arbitrage.

## Why factor premia vary — and why everyone is already a timer

Factor premia must vary over time by an *anthropic* argument: without such
variation the underlying phenomenon would be arbitraged away and the factor would
not exist at all. Each premium decomposes into (1) compensation for risk exposure,
(2) return arising from market-participant irrationality, and (3) the effects of
market frictions — each with its own dynamics and drivers. A corollary is that
diversification tends to disappoint exactly when needed most, in top-down-driven
environments where factor correlations rise. Crucially, even a nominal fixed-weight
manager is *implicitly* timing: as correlations fluctuate around their long-run
means, effective weights and exposures drift non-trivially — and do so less
informedly than an explicit forecasting approach could. Selecting funds or managers
on past performance is the self-defeating, performance-chasing form of the same
timing.

## The optimal factor-timing portfolio is the SDF

The economic stakes become clear once one recognizes that the optimal factor-timing
portfolio equals the **stochastic discount factor (SDF)**. Start from the
minimum-variance SDF in the span of the $N$ excess returns,
$m_{t+1}=1-b_t'(R_{t+1}-E_t[R_{t+1}])$, satisfying $0=E_t[m_{t+1}R_{t+1}]$. Assuming
cross-sectional heterogeneity in the risk prices $b_t$ is captured by $K\ll N$
observable characteristics, $b_t=C_t\delta_t$, rewrites the SDF in terms of
characteristic-managed factor portfolios $F_{t+1}=C_t'R_{t+1}$, so $\delta_t$ are
the **time-varying risk prices** on the factors. This yields a conditional factor
model $E_t[R_{j,t+1}]=\beta_{jt}'E_t[F_{t+1}]$ that generates risk-premium variation
*even under homoskedasticity* (constant $\Sigma_{F,t}$ and $\beta_{jt}$), driven
entirely by $\delta_t$. Because the SDF loadings equal the weights of the maximum
conditional-Sharpe-ratio portfolio $R^{opt}_{t+1}=E_t[F_{t+1}]'\Sigma_{F,t}^{-1}F_{t+1}$,
estimating the SDF and solving the factor-timing problem are the *same* exercise.
Factor timing thus unifies aggregate return predictability (market timing) with the
cross-sectional factor premia of [factor models](./factor-models.md).

## Near-arbitrage and left-hand-side dimension reduction

Measuring the value of timing empirically is hard because it means measuring the
predictability of many returns, inviting spurious findings; the "factor zoo"
inflates that risk in finite samples. Two economic restrictions discipline the
problem. First, when returns are uncorrelated the average maximum conditional
squared Sharpe ratio equals the expected variance of the minimum-variance SDF and
splits into an unconditional term plus a predictability term increasing in each
asset's maximal predictive $R^2$. Second, **no near-arbitrage** (Assumption 2) caps
the average conditional squared Sharpe ratios by a constant — equivalently, a bound
on the SDF variance, in the tradition of the APT bound. Applying the decomposition
to the factors' principal components shows **small PCs cannot contribute meaningful
predictability** without implying implausibly high Sharpe ratios, so the SDF (and
the optimal timing portfolio) is approximated by a handful of dominant components
$Z_{t+1}$. This is a **regularization of the left-hand side** — *which factors are
predictable?* — rather than the right-hand side — *which variables are useful
predictors?*.

## The robust predictability of the factor cross-section

Empirically, using fifty long-short "anomaly" portfolios (CRSP-COMPUSTAT, decile
10 minus decile 1, 1974–2017), the anomalies show a moderately strong factor
structure: the first PC explains 25.8% of variance and the first five nearly
two-thirds. Studying $Z=(\text{market}, PC_1,\dots,PC_5)$ and predicting each
component by its own net book-to-market ratio, the dominant components are strongly
predictable: PC1 and PC4 are unambiguously predictable (t-stats ≈ 4.31 and 3.74, in-
and out-of-sample $R^2$ ≈ 4% and 3.5%, about four times the market's), while the
market itself is insignificant (t = 1.24). The decisive ingredient is left-hand-side
dimension reduction: predicting each anomaly by its own $bm$ yields only half the
predictability, and running all 50 anomalies on all 50 valuations without
regularization gives an out-of-sample $R^2$ of $-134\%$. By contrast, **factor
momentum** (past performance as predictor) *fails* out of sample ($R^2$ of $-0.49\%$
and $-0.08\%$).

## Timing gains and the character of the SDF

The predictability translates into substantial gains. Building the optimal portfolio
with weights $\omega_t=\Sigma_{Z,t}^{-1}E_t[Z_{t+1}]$, **pure anomaly timing** (zero
weight on the market, zero average exposure to every factor) reaches a Sharpe ratio
of 0.71 in-sample and 0.77 out-of-sample despite taking *no* static bets; measured
by the conservative information ratio, factor/anomaly/pure-anomaly timing lift the
opportunity set to 0.42, 0.60, 0.59 out-of-sample. Expected utility nearly doubles,
from 1.66 to 2.96, of which 1.26 comes from pure anomaly timing alone, whereas
adding market timing to factor investing adds only 0.03. These gains raise the
SDF's average variance from 1.67 to 2.96 — far above standard models (≈0.85
annualized in long-run-risk) — and the SDF is strongly heteroskedastic (variation
mostly at business-cycle frequency; size and value procyclical, momentum
countercyclical). Reducing rebalancing frequency does not materially hurt (annual
rebalancing even lifts pure-anomaly-timing Sharpe to 0.79), suggesting the
strategies could be implementable pending direct transaction-cost measurement.

## Valuation-based timing: buy factors on sale

A distinct tradition times from relative valuation. Past performance splits into
*structural alpha* (net of any valuation change) and *revaluation alpha* (the part
from rising multiples — non-recurring and as likely to reverse as persist); rising
valuations create an illusion of alpha and encourage performance chasing. Relative
valuations, by contrast, predict future returns robustly and globally: the slope of
subsequent five-year return on aggregate valuation is negative (correlations $-0.19$
for the value blend, $-0.31$ for equal-weighted smart beta). A **trend chaser**
buying the three best-past-performing factors destroys value (factor Sharpe collapses
to 0.14), while a **contrarian** buying the three cheapest beats it (factor return
3.3%, Sharpe 0.39). In the US, the three cheapest minus three most expensive factors
earns 7.2% per year (t = 3.62) with a four-factor alpha of 7.7% (t = 4.05), *not*
explained by value exposure. But moderation is essential — aggressive single-factor
bets erode Sharpe ratios through lost diversification — and post-publication decay
is real: mean factor excess returns fall from 5.8% to 2.4% and roughly half of
factor alpha evaporates once academics discover factors when they are expensive.

## The failure of extrapolation

The complement is what does *not* predict returns. A factor's recent five-year
performance is *negatively* correlated with its subsequent five-year performance —
history is worse than useless. Across six alpha-forecast models, the trailing
five-year-return model forecasts worst (correlation $-0.18$) and the
since-inception model, despite better accuracy, still has negative correlation
$-0.39$ (t = $-2.85$): selecting factors on past performance, regardless of sample
length, points the wrong way. Only the valuation-dependent models (3–6) have
positive correlation with future performance; the clairvoyant, full-sample
look-ahead Model 6 reaches correlation 0.54 and cuts error 39%, and the realistic
shrunk Model 4 (no look-ahead) reaches 0.36 and cuts error 25% — about two-thirds of
clairvoyance. Two caveats attach: transaction costs (heaviest on momentum and
low-volatility) must be netted out, and the estimation error of the return forecasts
usually *exceeds the alpha forecast itself*.

## A taxonomy of predictors and the traps of timing

Timing signals fall into five categories — Financial Conditions (credit/TED spreads,
money-supply growth), Economic Cycle (GDP growth, capacity, confidence), Sentiment
(VIX, ISM PMI), Valuation (CAPE, dividend/earnings yield, book-to-price), and
Trend/Momentum — with the cardinal point that **different predictors matter at
different horizons**; few are strong at the one-month horizon (e.g. the
term-spread/profitability correlation moves from $-0.18$ at one month to $-0.46$ at
one year). Three traps threaten any timing model: (1) **time-varying causal links** —
value's market beta changes sign over time, so a credit-spread widening precedes good
value performance in 2000–2001 but bad in 2008–2009; (2) **hindsight cherry-picking
and data mining** — of predictors significant in 1972–1989, very few survive in
1990–2010 (e.g. 1 of 18 for size at the quarterly horizon); (3) **data revisions** —
GDP and unemployment are revised after first release, so backtests may not reflect
truly available information. The prescriptions are a parsimonious few-signal model
grounded in theory (Valuation and Trend are good candidates) or a multi-signal panel
regression that lifts the information ratio from 0.88 to 1.17.

## The skeptical counterpoint

A skeptical strand centers on the **value spread** — a factor's cheapness as the
valuation gap between high- and low-exposure assets — and finds the measure is not
unique (P/B vs P/E, percentiles vs z-scores change the reading). On today's levels
the answer is benign: HML and UMD in US large-caps are not markedly more expensive
than their post-1968 means, and BAB, though richer in 2007–2016, stayed well below
two standard deviations. On timing, contrarian valuation timing is "deceptively
difficult": initial correlations between value spread and subsequent factor returns
look modestly promising but the promise evaporates once realistic contrarian trading
strategies are simulated. Timing is harder for factors than markets because of
**turnover** — one forecasts using today's constituents, but the portfolio can look
very different three months later. Decisively, value timing works better on single
factors than on multi-style portfolios that already hold value, because value timing
is highly correlated with the plain value factor: if you want value, just add a
strategic value allocation.

## The balanced verdict

Two conclusions are shared across the evidence and connect to
[market regimes and views](../regimes/market-regimes-and-views.md) and
[macro signals and risk premia](../regimes/macro-signals-and-risk-premia.md). First,
**valuation is the one robust predictor** — the only variable generating positive
correlations and error reductions, surviving among the few strong long-horizon
signals, and informing the value spread. Second, **extrapolating past performance is
dangerous** — the five-year correlation between recent and future performance is
negative, most hindsight-selected predictor-factor links fail out of sample, and
factor momentum fails even in the disciplined SDF framework. Timing is possible at
the margin for a long-horizon investor with a horizon-matched model, but discipline
must come from valuation, model parsimony, and the no-near-arbitrage constraint.
