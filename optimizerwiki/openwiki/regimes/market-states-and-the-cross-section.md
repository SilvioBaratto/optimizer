---
type: "Reference"
title: "Market States and the Cross-Section"
openwiki_generated: true
sources:
  - id: openwiki-source-9a84c0297b2f3a9d5484b916
    resource: repo://docs/26_market_states_and_cross_section.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---


# Market States and the Cross-Section

Cross-sectional selection signals — momentum and value in particular — are often treated as
if their profitability were a stationary parameter summarized by an average information
coefficient. At the opposite extreme, the [market regimes and views](./market-regimes-and-views.md)
chapter introduced the idea that the market moves through distinct states governed by a latent
Markov chain. This chapter bridges the two: the payoff of a cross-sectional signal is not a
constant but a function of the market state, and the same regime apparatus that conditions
allocation across asset classes also conditions selection within them. The common thesis is
that the expected return of a zero-cost long-short signal portfolio is time-varying and tied
to state variables — for momentum, the market's past return and volatility; for value, the
business-cycle phase — with value and momentum sharing a common factor structure traceable to
a single funding-liquidity state. This feeds naturally into
[factor timing and rotation](../factor-models/factor-timing-and-rotation.md) and connects to
[turbulence and systemic risk](./turbulence-and-systemic-risk.md).

## Market states and momentum

A Jegadeesh–Titman relative momentum strategy ranks stocks each month by the prior six-month
cumulative return (months $t-6$ to $t-1$, skipping $t$), buys the winner decile and sells the
loser decile, held for six months; the momentum profit $\pi_t$ is the winner-minus-loser
return, measured raw and Fama–French three-factor risk-adjusted. The **market state** is
defined ex ante by the sign of the market's past cumulative return (value-weighted CRSP) over
a long horizon, typically 36 months (results hold at 12 and 24):
$$ \text{state}_t = \begin{cases} \text{UP} & \text{if } R^{m}_{t-1-J,\,t-1} \ge 0,\\ \text{DOWN} & \text{if } R^{m}_{t-1-J,\,t-1} < 0. \end{cases} $$
State-conditional mean profits are the coefficients $\alpha_{\text{UP}}$, $\alpha_{\text{DOWN}}$
of $\text{CAR}_t = \alpha_{\text{UP}}\mathbb{1}_{\{\text{UP}\}} + \alpha_{\text{DOWN}}\mathbb{1}_{\{\text{DOWN}\}} + \varepsilon_t$
with HAC standard errors for the overlapping windows. The empirical result is sharp and
asymmetric: **momentum is profitable only after up markets**. In the first holding semester
after an UP state the mean profit is strong and positive — about 0.93% per month raw
($t\approx 8.4$) and about 1.12% CAPM-adjusted — while after a DOWN state it is null or
slightly negative (about $-0.37\%$ raw, 0.01% adjusted, indistinguishable from zero), the
difference being significant ($t\approx 2.3$). The relation is also nonlinear: regressing the
profit on the lagged market return and its square gives a positive linear and negative
quadratic coefficient, so the profit rises from negative to moderately positive states but
attenuates at bullish extremes.

Over the long horizon the profit realized after UP states **reverses**: over holding months
13 to 60 the winner-minus-loser portfolio following a bull state returns about $-0.36\%$ per
month raw ($t\approx -3.2$). This short-run momentum followed by long-run reversal is the
empirical signature of behavioral overreaction theories — self-attribution/overconfidence and
gradual information diffusion — which predict stronger continuation and correction precisely
when aggregate sentiment, proxied by the past market state, is high. The state dependence
resists risk-based controls: comparing the lagged market return against a macro-factor model
(dividend yield, default spread, term spread, yield spread) as predictors of momentum profits,
the lagged market return is the best out-of-sample predictor while the macro variables do not
significantly reduce forecast error — so the UP-state momentum payoff is hard to rationalize as
compensation for systematic risk.

## Momentum crashes and short optionality

The state dependence has a dramatic counterpart in the tail. The winner-minus-loser (WML)
strategy, despite a high mean return (about 17.9% annual, Sharpe above the market's,
unconditional CAPM alpha about 22% annual), suffers infrequent but extreme **crashes**
concentrated in identifiable phases. Its monthly distribution is strongly left-skewed and its
worst returns cluster in a few episodes: of the fifteen worst months, fourteen fall when the
prior two-year market return is negative. Crashes occur in **panic states** — after strong
market declines (a bear market by negative two-year past return) and in high-volatility
regimes — and materialize alongside market **rebounds**. In July–August 1932 the loser
portfolio returned about $+232\%$ vs. $+32\%$ for winners; in March–May 2009 losers returned
about $+163\%$ vs. $+8\%$ for winners, with August 1932 the worst month at about $-74\%$,
because distressed past losers gained far more than winners in the rally.

The mechanism is the **state dependence of the WML portfolio's betas**. In up markets the beta
is modest; in down markets the short leg (losers) acquires a call-option-like exposure, since
distressed firms' residual equity reacts convexly to a recovery (equity as an option on firm
value). WML is thus **short this optionality**: a negative beta that becomes strongly negative
when the market rebounds from a bear state, formalized by the state-conditional beta regression
$$ \tilde{R}_{\text{WML},t} = [\alpha_0 + \alpha_B \mathbb{1}_{B,t-1}] + [\beta_0 + \mathbb{1}_{B,t-1}(\beta_B + \mathbb{1}_{U,t}\beta_{B,U})]\tilde{R}_{m,t} + \tilde{\varepsilon}_t, $$
with $\mathbb{1}_{B,t-1}$ the bear state and $\mathbb{1}_{U,t}$ a contemporaneous up market. The
estimate returns base beta $\beta_0 \approx -0.03$, bear-state worsening $\beta_B \approx -0.66$,
and a markedly negative interaction $\beta_{B,U} \approx -0.82$: in the bear state, when the
market rises, WML beta is strongly negative, so the up-market beta (about $-1.5$) is far more
negative than the down-market beta (about $-0.7$), and the strategy loses in rebounds. Because
crashes depend on ex-ante observable variables — the bear state and realized market variance
(daily returns over the prior 126 days) — they are **partly predictable** and hence manageable:
the bear-plus-high-volatility combination signals low expected momentum returns and a fat left
tail.

## Dynamic momentum weighting

Predictability translates into a **dynamic weighting** rule. For a mean-variance investor who
scales the WML position, the optimal weight is proportional to the conditional mean over
conditional variance,
$$ w^{*}_{t-1} = \frac{1}{2\lambda}\frac{\mu_{t-1}}{\sigma_{t-1}^{2}}, $$
with $\lambda$ a time-invariant risk scalar. Two ingredients feed it: the **volatility scale**
— estimating $\sigma_{t-1}^{2}$ from recent realized variance yields the constant-volatility
variant $w_{t-1} \propto \sigma_{\text{target}}/\sigma_{t-1}$, cutting exposure exactly in
panic states — and the **mean forecast** $\mu_{t-1}$, made a function of the bear indicator and
expected variance to further cut or reverse the position when the state signals negative
expected return. Historically the dynamic strategy strongly attenuates crashes and lifts the
Sharpe ratio from about 0.6 (constant weight) to about 1.0 (volatility scaling only) and about
1.2 (full dynamic), persisting across subperiods and asset classes. The lesson: the momentum
signal must be used with state-conditioned intensity — not a static alpha chased at fixed
weight, but one whose optimal exposure is a function of market state and volatility, in line
with dynamic signal-exposure management.

## The regime-dependent value premium

The value signal also has a regime-dependent payoff. The reference model is a two-state Markov
chain on book-to-market-sorted portfolio returns with time-varying transition probabilities,
estimated by maximum likelihood in the Hamilton-filter tradition. With latent state
$s_t\in\{1,2\}$, the excess return follows
$$ r_t = \beta_{0,s_t} + \beta_{1,s_t}TB_{t-1} + \beta_{2,s_t}DEF_{t-1} + \beta_{3,s_t}\Delta M_{t-2} + \beta_{4,s_t}DIV_{t-1} + \varepsilon_t, \quad \varepsilon_t\sim N(0,\sigma^2_{s_t}), $$
where $TB$ is the one-month T-bill rate, $DEF$ the default spread (Baa minus Aaa), $\Delta M$
money-base growth (lagged two months), and $DIV$ the market dividend yield; intercepts,
loadings, and variance all depend on the state, and transition probabilities depend on the
short rate, $p_t=\Phi(\pi_0+\pi_1 TB_{t-1})$, $q_t=\Phi(\pi_0+\pi_2 TB_{t-1})$. **State 1** is
the high-volatility regime (about twice the volatility of state 2, associated with recessions);
state 2 is the low-volatility expansion regime. The central result is a **sensitivity asymmetry
between value and growth concentrated in the high-volatility state**: in state 1 value's
expected returns load on macro conditions far more than growth's — on the default spread about
7.76 (value) vs. 4.60 (growth), on the short rate about $-10.76$ vs. $-6.74$ — while in state 2
both loadings are small and insignificant. The test that the loading change across states is
equal for value and growth is rejected. Since state 1 coincides with economic deterioration,
the **value premium is countercyclical and time-varying**: the expected value-minus-growth
differential is about 12.4% annual in the high-volatility state vs. about 0.6% in the
low-volatility state, with an average expected premium of about 0.39% monthly, positive in
about 73% of months.

The economic reading is a downside-risk story: in bad times value firms' fundamentals are hit
disproportionately (higher operating leverage, costly asset reversibility, tighter financial
constraints), so value's higher expected return is compensation for this aggravated bad-times
risk. The regime framework's nonlinearity is essential — a linear predictive regression does
not capture the expected-premium peaks the Markov model identifies — and justifies, on the value
side, the same conclusion momentum imposed: the signal's payoff is a state-conditioned quantity,
not a constant fed into the fundamental law of active management.

## The risk-based mechanism: countercyclical betas

The value premium's cycle dependence has a foundation in a conditional CAPM with time-varying
beta, where a portfolio's market beta is a linear function of cycle state variables,
$$ \beta_{i,t} = b_{i0} + b_{i1}\text{DIV}_t + b_{i2}\text{DEF}_t + b_{i3}\text{TERM}_t + b_{i4}\text{TB}_t, $$
and the same variables proxy the **expected market premium**
$\gamma_t=E_t[r_{m,t+1}-r_f]$, high in bad times and low in good times. The risk-based
signature is that **value betas covary positively with the expected market premium** (rising in
bad times when $\gamma_t$ is high) while **growth betas covary negatively** — value betas
countercyclical, growth betas procyclical — with the conditional HML beta going from negative at
cycle peaks to positive at troughs, a peak-to-trough difference of order 0.5–0.7. This is the
pattern a time-varying-price-of-risk pricing model requires for value to be riskier: a stock
whose beta is high exactly when the price of risk is high commands a higher expected return.
Taking the unconditional expectation of the conditional CAPM $E_t[r_{i,t+1}-r_f]=\beta_{i,t}\gamma_t$,
$$ E[r_i-r_f] = \bar\gamma\,\bar\beta_i + \operatorname{Cov}(\beta_{i,t},\gamma_t) = \bar\gamma\,\bar\beta_i + \operatorname{Var}[\gamma_t]\,\omega_i, $$
with beta-premium sensitivity $\omega_i=\operatorname{Cov}(\beta_{i,t},\gamma_t)/\operatorname{Var}[\gamma_t]$
widely positive for value (order 18–20) and negative for growth. But for the value-minus-growth
spread the **magnitude is modest**: the covariance term explains only part of the observed
premium (about 0.39% monthly for HML) and does not fully rationalize it, so conditional-CAPM
alphas remain positive and significant. The lasting contribution is the qualitative mechanism:
conditional betas are the state-dependent channel through which the cycle enters value's payoff.

## Common factor structure and funding-liquidity across asset classes

The generality emerges when value and momentum are studied simultaneously across eight markets
— US, UK, continental Europe, and Japan individual stocks, plus equity index futures,
currencies, government bonds, and commodities (48 test portfolios) — with positive pervasive
premia for both signals almost everywhere (in stocks value is book-to-market and momentum is
the 12-month return skipping the most recent; elsewhere value is the sign-reversed five-year
past return and momentum the 12-month return). The structural discovery is the **cross-market
correlation of signals**: value in one market is positively correlated with value elsewhere
(about 0.68 across equity markets) and momentum with momentum (about 0.65), while value and
momentum are **negatively correlated** with each other (about $-0.5$ within each class). A
principal-component analysis confirms a single latent factor loading with opposite signs on
value and momentum, so there exist global common value-everywhere and momentum-everywhere
factors crossing class boundaries. The negative correlation makes their combination powerful: a
50/50 combination diversified across all classes and rescaled to equal volatility reaches a
Sharpe of about 1.45, and a global three-factor model
$$ R^{p}_t-r_{f,t} = \alpha^p + \beta^p MKT_t + v^p VAL^{\text{everywhere}}_t + m^p MOM^{\text{everywhere}}_t + \varepsilon^p_t $$
explains about 71% of the cross-sectional variation of the 48 portfolios with average pricing
errors of order 18 basis points per month.

The commonality calls for a shared risk source, and the candidate is the **funding-liquidity**
state. Regressing the two signals' returns on funding-liquidity shocks (proxied by the TED
spread and other funding spreads, purged by an autoregression), **momentum loads positively on
liquidity risk while value loads negatively**: the funding-shock beta ($t$-statistic) is about
$-4.74$ for value and $+3.58$ for momentum. Momentum suffers in liquidity crises when funding
dries up and crowded positions are liquidated — consistent with funding-liquidity theory —
while value, with opposite exposure, gains in those phases. The two together partly **hedge**
funding-liquidity risk, explaining both the high Sharpe of the joint portfolio and the negative
correlation between the signals. The shared macro-financial state — arbitrageurs' funding
availability — is the common variable linking value's and momentum's payoffs coherently across
asset classes.

## The shared regime engine for allocation and selection

The final step closes the bridge with [market regimes and views](./market-regimes-and-views.md):
the state $s_t$ conditioning cross-sectional selection can be the very state estimated by the
regime-switching model driving allocation across asset classes. The multivariate model on
excess returns of large caps, small caps, and long-term bonds is $r_t = \mu_{s_t} + \varepsilon_t$,
$\varepsilon_t\sim N(0,\Omega_{s_t})$, with $s_t\in\{1,\dots,k\}$ a latent Markov chain of
transition matrix $P=[p_{ij}]$, so means, volatilities, and — crucially in the multivariate view
— **correlations** change with the regime. Maximum likelihood identifies **four regimes**: a
**crash** (strongly negative means, highest volatility, and negative stock-bond correlation, about
$-0.41$ between large caps and bonds, duration about two months); **slow growth** (means near
zero, low volatility, positive correlations, persistent, about seven months); **bull** (positive
means, higher for small caps, moderate volatility, most persistent, about eight months); and
**recovery** (high positive means, very high volatility, short, about three months), with
stationary probabilities of order (9%, 40%, 28%, 23%). The **sign-switching of correlations** is
the key trait — bonds diversify in the crash and lose hedging power elsewhere, and the large-vs-
small-cap correlation falls from about 0.82 (crash) to about 0.50 (recovery) — a time-varying
structure an i.i.d. Markowitz model cannot capture. The current state is inferred by the
recursive Hamilton filter updating
$$ \Pr(s_t=j\mid \mathcal{F}_t) \propto f(r_t\mid s_t=j)\sum_i p_{ij}\Pr(s_{t-1}=i\mid \mathcal{F}_{t-1}), $$
and with $\gamma=5$ power utility the optimal weights differ markedly by regime (maximum equity
in the bull, defensive in the crash) and generate intertemporal hedging demands from uncertainty
about future state transitions.

The same model shows the direct link to cross-sectional selection: the **size premium**
(small minus large) is strongly regime-dependent. It is positive and maximal in the crash state
(about $+100$ basis points per month, where small caps at $-4.10\%$ lose less than large at
$-5.10\%$), positive in the bull (about $+71$ bps per month), but negative in slow growth (about
$-61$ bps per month, where large caps outperform), a spread of about 161 bps between the
maximum- and minimum-premium regimes. The same state-probability vector $\pi_t$ that governs
allocation across asset classes can thus condition selection within them: the size premium is
not constant but concentrates in the bull and crash states and reverses in slow growth. The
unifying reading is that the same latent Markov chain $s_t$ that forms allocation views supplies
the state that conditions selection — momentum full-weighted in the UP/low-volatility state and
cut or reversed in the bear/high-volatility state to avoid its crashes, the value premium
widening in the adverse/high-volatility state, and the size premium positive and wide in the
crash and bull states. Cross-sectional selection inherits its missing premise from the regime
engine: signals' information coefficient is not a constant but a function of the market state,
and [factor timing and rotation](../factor-models/factor-timing-and-rotation.md) is its natural
operational continuation.
