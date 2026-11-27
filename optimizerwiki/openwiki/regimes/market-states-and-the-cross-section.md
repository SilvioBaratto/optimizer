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
allocation across asset classes can also condition selection within them. The common thesis is
that the expected return of a zero-cost long-short signal portfolio is time-varying and tied
to state variables — for momentum, the market's past return and volatility; for value, the
business-cycle phase — with value and momentum sharing a common factor structure of which
funding-liquidity risk is a partial source. This feeds naturally into
[factor timing and rotation](../factor-models/factor-timing-and-rotation.md) and connects to
[turbulence and systemic risk](./turbulence-and-systemic-risk.md).

## Market states and momentum

A Jegadeesh–Titman relative momentum strategy ranks stocks each month by the prior six-month
cumulative return (months $`t-6`$ to $`t-1`$, skipping $`t`$), buys the winner decile and sells the
loser decile, held for six months; the momentum profit $`\pi_t`$ is the winner-minus-loser
return, measured raw and risk-adjusted with the CAPM and the Fama–French three-factor model
(Cooper, Gutierrez and Hameed, 1929–1995). The **market state** is
defined ex ante by the sign of the market's past cumulative return (value-weighted CRSP) over
a long horizon of $`J`$ months before formation, typically 36 (results hold at 12 and 24):

$$\text{state}_t = \begin{cases} \text{UP} & \text{if } R^{m}_{t-1-J,\,t-1} \ge 0,\\ \text{DOWN} & \text{if } R^{m}_{t-1-J,\,t-1} < 0. \end{cases}$$

State-conditional mean profits are the coefficients $`\alpha_{\text{UP}}`$, $`\alpha_{\text{DOWN}}`$
of $`\text{CAR}_t = \alpha_{\text{UP}}\mathbb{1}_{\{\text{UP}\}} + \alpha_{\text{DOWN}}\mathbb{1}_{\{\text{DOWN}\}} + \varepsilon_t`$,
where $`\text{CAR}_t=\sum_{k=K_1}^{K_2} r_{t+k}`$ is the cumulative return over the holding
months, with HAC standard errors for the overlapping windows; the profits quoted are monthly
averages, the coefficients divided by $`K_2-K_1+1`$. The empirical result is sharp and
asymmetric: **momentum is profitable only after up markets**. In the first holding semester
after an UP state the mean profit is strong and positive — about 0.93% per month raw
($`t\approx 8.4`$) and about 1.12% CAPM-adjusted — while after a DOWN state it is null or
slightly negative (about $`-0.37\%`$ raw, 0.01% CAPM-adjusted, indistinguishable from zero),
the difference being significant for raw and CAPM-adjusted profits ($`t\approx 2.3`$ and
2.2); Fama–French-adjusted, the profits are about 1.28% and 0.64% and the difference
($`t\approx 1.4`$) is not significant with the 36-month state, though it is with the 12- and
24-month ones. The relation is also nonlinear: regressing the
profit on the lagged market return and its square gives a positive linear and negative
quadratic coefficient, so the profit rises from negative to moderately positive states but
attenuates at bullish extremes.

Over the long horizon the profit realized after UP states **reverses**: over holding months
13 to 60 the winner-minus-loser portfolio following a bull state returns about $`-0.36\%`$ per
month raw ($`t\approx -3.2`$). This short-run momentum followed by long-run reversal is
consistent with behavioral overreaction theories — self-attribution/overconfidence and
gradual information diffusion — which predict stronger continuation and correction after
market gains, when aggregate overconfidence is higher or momentum traders' risk aversion has
fallen with their wealth. The reversal is not specific to UP states, however: after DOWN
states, with no initial momentum, months 13 to 60 return about $`-0.67\%`$ per month
($`t\approx -3.2`$), the difference is not significant ($`t\approx 1.5`$), and long-run
reversals are not solely corrections of prior momentum. The state dependence
resists risk-based controls: comparing the lagged market return against a macro-factor model
(dividend yield, default spread, term spread, three-month T-bill yield) as predictors of
momentum profits, the lagged market return is the best out-of-sample predictor while the macro
variables do not significantly reduce forecast error — so the UP-state momentum payoff is hard
to rationalize as compensation for systematic risk. This holds for strategies that exclude
stocks priced under one dollar or skip a month between formation and holding; without these
adjustments Chordia and Shivakumar find that the macro model does explain momentum profits.

## Momentum crashes and short optionality

The state dependence has a dramatic counterpart in the tail. The winner-minus-loser (WML)
strategy of Daniel and Moskowitz (value-weighted extreme CRSP deciles ranked on returns from
$`t-12`$ to $`t-2`$, 1927–2013, with a two-year rather than three-year market window for the
state, so its figures do not transfer to the six-month strategy above), despite a high mean
return (about 17.9% annual, Sharpe above the market's, unconditional CAPM alpha about 22%
annual), suffers infrequent but extreme **crashes**
concentrated in identifiable phases. Its monthly distribution is strongly left-skewed and its
worst returns cluster in a few episodes: of the fifteen worst months, fourteen fall when the
prior two-year market return is negative. Crashes occur in **panic states** — after strong
market declines (a bear market by negative two-year past return) and in high-volatility
regimes — and materialize alongside market **rebounds**. In July–August 1932 the loser
portfolio returned about $`+232\%`$ vs. $`+32\%`$ for winners; in March–May 2009 losers returned
about $`+163\%`$ vs. $`+8\%`$ for winners, with August 1932 the worst month at about $`-74\%`$,
because distressed past losers gained far more than winners in the rally.

The mechanism is the **state dependence of the WML portfolio's betas**. In bull states the beta
is modest; in bear states (after a two-year market decline) the short leg (losers) acquires a
call-option-like exposure, whichever way the market moves that month, since
distressed firms' residual equity reacts convexly to a recovery (equity as an option on firm
value). WML is thus **short this optionality**: a negative beta that becomes strongly negative
when the market rebounds from a bear state, formalized by the state-conditional beta regression

$$\tilde{R}_{\text{WML},t} = [\alpha_0 + \alpha_B \mathbb{1}_{B,t-1}] + [\beta_0 + \mathbb{1}_{B,t-1}(\beta_B + \mathbb{1}_{U,t}\beta_{B,U})]\tilde{R}_{m,t} + \tilde{\varepsilon}_t,$$

with $`\mathbb{1}_{B,t-1}`$ the bear state and $`\mathbb{1}_{U,t}`$ a contemporaneous up market. The
estimate returns base beta $`\beta_0 \approx -0.03`$, bear-state worsening $`\beta_B \approx -0.66`$,
and a markedly negative interaction $`\beta_{B,U} \approx -0.82`$: in the bear state, when the
market rises, WML beta is strongly negative, so the up-market beta (about $`-1.5`$) is far more
negative than the down-market beta (about $`-0.7`$), and the strategy loses in rebounds. Because
crashes depend on ex-ante observable variables — the bear state and realized market variance
(daily returns over the prior 126 days) — they are **partly predictable** and hence manageable:
the bear-plus-high-volatility combination signals low expected momentum returns and a fat left
tail.

## Dynamic momentum weighting

Predictability translates into a **dynamic weighting** rule. For an investor who scales the
WML position to maximize the unconditional Sharpe ratio (the same form follows from a
per-period mean-variance utility), the optimal weight is proportional to the conditional mean
over conditional variance,

$$w^{*}_{t-1} = \frac{1}{2\lambda}\frac{\mu_{t-1}}{\sigma_{t-1}^{2}},$$

with $`\mu_{t-1}`$ and $`\sigma_{t-1}^{2}`$ the conditional mean and variance of next month's
WML return and $`\lambda`$ a time-invariant scaling constant set for a 19% in-sample annual
volatility. Two ingredients feed it: the **volatility scale**
— $`\sigma_{t-1}^{2}`$ is forecast from a GJR-GARCH model combined with WML's realized
volatility over the prior 126 days; with a constant mean the weight scales as
$`1/\sigma_{t-1}^{2}`$, and it reduces to the constant-volatility variant
$`w_{t-1} \propto \sigma_{\text{target}}/\sigma_{t-1}`$ only if $`\mu_{t-1}/\sigma_{t-1}`$ is
constant, which momentum violates (its mean falls when forecast volatility is high); either way
exposure is cut in panic states — and the **mean forecast** $`\mu_{t-1}`$, made a function of
the bear indicator and expected variance to further cut or reverse the position when the
state signals negative expected return. Historically (1927–2013) the dynamic strategy
strongly attenuates crashes and lifts the Sharpe ratio from about 0.6 (constant weight) to
about 1.0 (volatility scaling only) and about 1.2 (full dynamic), persisting across subperiods
and asset classes; these are
in-sample figures, with $`\lambda`$ and the forecast parameters estimated on the full sample,
unconstrained position size and no transaction costs. The lesson: the momentum
signal must be used with state-conditioned intensity — not a static alpha chased at fixed
weight, but one whose optimal exposure is a function of market state and volatility, in line
with dynamic signal-exposure management.

## The regime-dependent value premium

The value signal also has a regime-dependent payoff. The reference model (Gulen, Xing and
Zhang, 1954–2007) is a two-state Markov
chain on book-to-market-sorted portfolio returns with time-varying transition probabilities,
estimated by maximum likelihood in the Hamilton-filter tradition. With latent state
$`s_t\in\{1,2\}`$, the excess return follows

$$r_t = \beta_{0,s_t} + \beta_{1,s_t}TB_{t-1} + \beta_{2,s_t}DEF_{t-1} + \beta_{3,s_t}\Delta M_{t-2} + \beta_{4,s_t}DIV_{t-1} + \varepsilon_t, \quad \varepsilon_t\sim N(0,\sigma^2_{s_t}),$$

where $`TB`$ is the one-month T-bill rate, $`DEF`$ the default spread (Baa minus Aaa), $`\Delta M`$
money-base growth (lagged two months), and $`DIV`$ the market dividend yield; intercepts,
loadings, and variance all depend on the state, and transition probabilities depend on the
short rate, $`p_t=\Pr(s_t=1\mid s_{t-1}=1)=\Phi(\pi_0+\pi_1 TB_{t-1})`$,
$`q_t=\Pr(s_t=2\mid s_{t-1}=2)=\Phi(\pi_0+\pi_2 TB_{t-1})`$, with $`\Phi`$ the standard normal
CDF. **State 1** is
the high-volatility regime (about twice the volatility of state 2, associated with recessions);
state 2 is the low-volatility expansion regime. The central result is a **sensitivity asymmetry
between value and growth concentrated in the high-volatility state**: in state 1 value's
expected returns load on the default spread and the short rate far more than growth's — on the
default spread about 7.76 (value) vs. 4.60 (growth), on the short rate about $`-10.76`$ vs.
$`-6.74`$ — while in state 2 these loadings are smaller (the default-spread ones insignificant,
value's short-rate loading of about $`-2.97`$ still significant at the 5% level). The test that the
loading change across states is equal for value and growth is rejected, but only for the short
rate and the default spread is the change larger for value; for money-base growth and the
dividend yield it is larger for growth. Since state 1 coincides with economic deterioration,
in sample the **value premium is countercyclical and time-varying**: the expected
value-minus-growth differential is about 12.4% annual in the high-volatility state vs. about 0.6% in the
low-volatility state, with an average expected premium of about 0.39% monthly, positive in
about 73% of months.

The economic reading is a downside-risk story: in bad times value firms' fundamentals are hit
disproportionately (higher operating leverage, costly asset reversibility, tighter financial
constraints), so value's higher expected return is compensation for this aggravated bad-times
risk. The regime framework's nonlinearity is essential — a linear predictive regression does
not capture the expected-premium peaks the Markov model identifies — and in sample supports, on
the value side, the same conclusion momentum imposed: the signal's payoff is a state-conditioned
quantity, not a constant fed into the fundamental law of active management. Out of sample,
however, the predictability of the value premium is close to nonexistent and the in-sample
results may severely overstate it, so conditioning a value signal on this model is not
supported by that evidence.

## The risk-based mechanism: countercyclical betas

The value premium's cycle dependence has a foundation in a conditional CAPM with time-varying
beta (Petkova and Zhang, 1927–2001), where a portfolio's market beta is a linear function of
cycle state variables,

$$\beta_{i,t} = b_{i0} + b_{i1}\text{DIV}_t + b_{i2}\text{DEF}_t + b_{i3}\text{TERM}_t + b_{i4}\text{TB}_t,$$

with $`\text{TERM}`$ the term spread and the other variables as above, and the same variables
proxy the **expected market premium**
$`\gamma_t=E_t[r_{m,t+1}-r_f]`$, high in bad times and low in good times. The risk-based
signature is that **value betas covary positively with the expected market premium** (rising in
bad times when $`\gamma_t`$ is high) while **growth betas covary negatively** — value betas
countercyclical, growth betas procyclical — with the conditional HML beta going from negative at
cycle peaks to positive at troughs, a peak-to-trough difference of order 0.5–0.7. This is the
pattern a time-varying-price-of-risk pricing model requires for value to be riskier: a stock
whose beta is high exactly when the price of risk is high commands a higher expected return.
Taking the unconditional expectation of the conditional CAPM $`E_t[r_{i,t+1}-r_f]=\beta_{i,t}\gamma_t`$
(which rests on restrictive assumptions, such as quadratic utility without labor income or
exponential utility with normal returns),

$$E[r_i-r_f] = \bar\gamma\,\bar\beta_i + \operatorname{Cov}(\beta_{i,t},\gamma_t) = \bar\gamma\,\bar\beta_i + \operatorname{Var}[\gamma_t]\,\omega_i,$$

with beta-premium sensitivity $`\omega_i=\operatorname{Cov}(\beta_{i,t},\gamma_t)/\operatorname{Var}[\gamma_t]`$
positive for value (about 18.6 rolling and 29.4 fitted-beta for the value portfolio, 20.4 and
33.3 for HML, $`t\approx 2`$) and negative but indistinguishable from zero for growth (about
$`-1.9`$ and $`-4.0`$). The evidence is weaker without the Great Depression or in 1963–2001,
and among small stocks the value spread's sensitivity is weak and the growth portfolio's
positive. But for the value-minus-growth
spread the **magnitude is modest**: the covariance term explains only part of the observed
premium (about 0.39% monthly for HML) and does not fully rationalize it, so conditional-CAPM
alphas remain positive and mostly significant (HML about 0.29% monthly over the full sample,
an insignificant 0.07% in 1927–1962). The lasting contribution is the qualitative mechanism:
conditional betas are the state-dependent channel through which the cycle enters value's payoff.

## Common factor structure and funding-liquidity across asset classes

The generality emerges when value and momentum are studied simultaneously across eight markets
(Asness, Moskowitz and Pedersen, data from as early as 1972 to 2011)
— US, UK, continental Europe, and Japan individual stocks, plus equity index futures,
currencies, government bonds, and commodities (48 test portfolios) — with positive pervasive
premia for both signals almost everywhere (value is book-to-market for stocks and country
indices, the sign-reversed five-year past return for commodities, the five-year change in
purchasing-power parity for currencies and the five-year change in ten-year yields for bonds;
momentum is the 12-month return skipping the most recent month in every class). The structural
discovery is the **cross-market correlation of signals**: value in one stock market correlates
about 0.68 with the average value strategy of the other stock markets and momentum about 0.65
with the average momentum elsewhere (pairwise averages are weaker), while value and momentum
are **negatively correlated** with each other within each class (about $`-0.43`$ to $`-0.65`$
in stocks, $`-0.17`$ to $`-0.46`$ elsewhere). In a principal-component analysis the first
component (54% of the stock-strategy covariance, 23% of the all-asset covariance) loads with
opposite signs on value and momentum; the value-everywhere and momentum-everywhere factors
crossing class boundaries are equal-volatility averages across markets, tested through the
model below. The negative correlation makes their combination powerful: a
50/50 combination diversified across all classes and rescaled to equal volatility reaches a
Sharpe of about 1.45, and a global three-factor model

$$R^{p}_t-r_{f,t} = \alpha^p + \beta^p MKT_t + v^p VAL^{\text{everywhere}}_t + m^p MOM^{\text{everywhere}}_t + \varepsilon^p_t$$

explains about 71% of the cross-sectional variation in the average returns of the 48
portfolios with average absolute pricing errors of order 18 basis points per month, although
the factors share securities with the test portfolios (built from the other markets only, about
55% and 22.6 basis points) and the GRS test still rejects the model.

The commonality calls for a shared risk source, and the candidate is the **funding-liquidity**
state. Regressing the two signals' returns on funding-liquidity shocks (proxied by the TED
spread and other funding spreads, purged by an autoregression, 1987–2011), **momentum loads
positively on liquidity risk while value loads negatively**: on a principal-component-weighted
index of global funding shocks, with macro controls, the loading is about $`-0.0094`$ for value
($`t\approx -4.74`$) and $`+0.0112`$ for momentum ($`t\approx 3.58`$); the TED spread alone
gives weaker loadings. Momentum suffers in liquidity crises when funding
dries up and crowded positions are liquidated — consistent with funding-liquidity theory —
while value, with opposite exposure, gains in those phases. The two together **hedge**
funding-liquidity risk: the opposite exposures partly explain the negative correlation between
the signals, but the 50/50 combination is nearly immune to funding shocks and still earns large
returns, so liquidity risk cannot explain its high Sharpe. The shared macro-financial state —
arbitrageurs' funding availability — is thus a partial source of value's and momentum's
comovement across asset classes, explaining only a small fraction of their premia and
comovement.

## The shared regime engine for allocation and selection

The final step closes the bridge with [market regimes and views](./market-regimes-and-views.md):
the state $`s_t`$ conditioning cross-sectional selection can be the very state estimated by the
regime-switching model driving allocation across asset classes. This is a hypothesis, not a
result of the sources: each defines its own state (the sign of the 36- or 24-month market
return for momentum, a two-state volatility chain or the fitted expected premium for value),
and none shows that one Markov chain drives all these premia. The multivariate model
(Guidolin and Timmermann, 1954–1999) on
excess returns of large caps, small caps, and long-term bonds is $`r_t = \mu_{s_t} + \varepsilon_t`$,
$`\varepsilon_t\sim N(0,\Omega_{s_t})`$, with $`s_t\in\{1,\dots,k\}`$ a latent Markov chain of
transition matrix $`P=[p_{ij}]`$, $`p_{ij}=\Pr(s_t=j\mid s_{t-1}=i)`$, so means, volatilities,
and — crucially in the multivariate view — **correlations** change with the regime. Maximum likelihood identifies **four regimes**: a
**crash** (strongly negative means, highest volatility, and negative stock-bond correlation, about
$`-0.41`$ between large caps and bonds, duration about two months); **slow growth** (means near
zero, low volatility, positive correlations, persistent, about seven months); **bull** (positive
means, higher for small caps, moderate volatility, most persistent, about eight months); and
**recovery** (high positive means, very high volatility, short, about three months), with
stationary probabilities of order (9%, 40%, 28%, 23%). The **sign-switching of correlations** is
the key trait — bonds diversify in the crash and lose hedging power elsewhere, and the large-vs-
small-cap correlation falls from about 0.82 (crash) to about 0.50 (recovery) — a time-varying
structure an i.i.d. Markowitz model cannot capture. The current state is inferred by the
recursive Hamilton filter updating

$$\Pr(s_t=j\mid \mathcal{F}_t) \propto f(r_t\mid s_t=j)\sum_i p_{ij}\Pr(s_{t-1}=i\mid \mathcal{F}_{t-1}),$$

and with $`\gamma=5`$ power utility (no short sales or frictions, a constant 5.3% annual risk-free rate)
the optimal weights differ markedly by regime: at horizons beyond a few months equity is highest
in the bull and lowest in the always-defensive crash, while at the shortest horizons it is near
100% in the other three states (mainly small caps in bull and recovery, large caps in slow
growth). They generate intertemporal hedging demands from uncertainty about future state
transitions and from the updating of beliefs about the current state.

The same model shows the direct link to cross-sectional selection: the **size premium**
(small minus large) is strongly regime-dependent. Its highest point estimate is in the crash
state (about $`+100`$ basis points per month, where small caps at $`-4.10\%`$ lose less than
large at $`-5.10\%`$), but with standard errors of about 1.5 (large) and 2.2 (small)
percentage points on the two means it is far from significant; it is positive in the bull (about $`+71`$ bps per month)
and negative in slow growth (about $`-61`$ bps per month, where large caps outperform) and in
recovery (about $`-41`$ bps), a spread of about 161 bps between the highest and lowest point
estimates. The same state-probability vector $`\pi_t`$ that governs
allocation across asset classes can thus condition selection within them: the size premium is
not constant (its stationary-probability-weighted average is close to zero), positive in the
bull and reversing in slow growth and recovery. The unifying hypothesis is that the same latent
Markov chain $`s_t`$ that forms allocation views could supply the state that conditions
selection — momentum full-weighted in the UP/low-volatility state and cut or reversed in the
bear/high-volatility state to avoid its crashes, the value premium widening in the
adverse/high-volatility state, and the size premium positive in the bull state. Mapping these
states onto one another is the assumption to test (for example, by conditioning momentum and
value returns on the filtered state probabilities): the size premium is the only
cross-sectional quantity the four-state model estimates, and momentum crashes occur on
rebounds, closer to its recovery state than to its crash state. Cross-sectional selection
inherits from the regime engine a premise to verify: that signals' information coefficient is
not a constant but a function of the market state, and
[factor timing and rotation](../factor-models/factor-timing-and-rotation.md) is its natural
operational continuation.
