---
type: concept
title: "Market Regimes and View Formation"
description: How the market alternates between persistent regimes (bull/bear, high/low volatility), how the Markov regime-switching model and its nonlinear filter estimate regime probabilities from returns, how regimes reshape the optimal allocation and the cost of ignoring them, and how the current macro state is measured in real time (nowcasting, diffusion index) to form the views a Black-Litterman model then ingests.
tags: [regimes, markov-switching, hamilton-filter, regime-allocation, nowcasting, diffusion-index, black-litterman, views]
sources:
  - id: openwiki-source-95965781acff910b105abfa6
    resource: repo://docs/21_market_regimes_and_view_formation.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Market Regimes and View Formation

Financial markets change behavior abruptly, and the new behavior often persists: the mean,
volatility, and correlation structure of equity returns can shift and stay at the new level,
as at the onset of the 2008–2009 crisis. Some shifts are recurrent — recessions vs.
expansions, calm vs. turbulent markets, bull vs. bear phases — while others are unique,
tied to structural breaks in regulation, policy, or secular change (in interest rates, a
distinct regime is the 1979–1982 period of altered US central-bank operating procedure).
Recognizing the current regime is the first step in forming a view over the opportunity set
that the [Black-Litterman model](../foundations/black-litterman.md) then ingests, and the
same latent state connects to [factor timing and rotation](../factor-models/factor-timing-and-rotation.md)
and [market states and the cross-section](./market-states-and-the-cross-section.md).

## Why regime-switching models

Regime-switching models capture this alternation, and their appeal rests on several grounds.
The idea of a regime is natural and intuitive — the original application concerned business-
cycle recessions and expansions, with regimes tied to ex-post turning-point dates. The models
parsimoniously reproduce stylized facts of returns — fat tails, persistent turbulence
followed by calm (ARCH effects), skewness, and time-varying correlations — by mixing
conditionally normal distributions, and they contain jump models as a special case (a regime
abandoned the next period). And because conditional on the regime normality (or
log-normality) is recovered, pricing under regimes stays analytically tractable and often
closed-form. A regime is close to the familiar "good"/"bad" or low-risk/high-risk state, but
embedding regimes in an equilibrium model can contradict intuition: whereas linear pricing
models imply a positive monotone risk-return relation, discrete regime shifts with different
consumption growth rates can produce risk-return relations that are increasing, decreasing,
flat, or non-monotone.

## The Markov regime-switching model (Hamilton)

Hamilton's tractable approach treats the parameters of an autoregressive process as the
outcome of a latent state governed by a discrete Markov chain. A trend component $n_t$ follows
a *Markov trend in levels* if $n_t = \alpha_0 + \alpha_1 s_t + n_{t-1}$, with $s_t \in \{0,1\}$
the unobserved state, and transitions follow a first-order Markov process:
$$ \Pr[S_t=1 \mid S_{t-1}=1]=p, \qquad \Pr[S_t=0 \mid S_{t-1}=0]=q. $$
The state has an AR(1) representation $s_t = (1-q) + \lambda s_{t-1} + v_t$ with
$\lambda \equiv -1+p+q$, so the conditional probability converges to the unconditional limit
$\Pr[S_t=1] = \pi \equiv (1-q)/[(1-p)+(1-q)]$. The econometrician does not observe $s_t$ but
infers it: Hamilton maximizes the *marginal* likelihood of the observed series over population
parameters, then uses those parameters to infer the unobserved state — differing from earlier
work that maximized the joint likelihood of series and states assumed observable. Combining
trend with a cyclical ARIMA$(r,1,0)$ component and differencing gives
$$ y_t = \alpha_0 + \alpha_1 s_t + z_t, \qquad z_t = \phi_1 z_{t-1} + \cdots + \phi_r z_{t-r} + \varepsilon_t, $$
with $\varepsilon_t$ i.i.d. $N(0,\sigma^2)$; the growth rate changes only in response to
discrete occasional events, and the $\alpha_1 s_t$ term makes the observed series nonlinear.

## The nonlinear filter and maximum-likelihood estimation

The operational core is a filter that recursively estimates regime probabilities. Given the
joint conditional probability of past states, the filter returns the updated probability at
$t$ plus, as a by-product, the conditional likelihood $f(y_t \mid y_{t-1},\dots)$, in five
steps: (1) form the joint probability at $t$ conditional on the past using the transition
probabilities; (2) multiply by the Gaussian conditional density of $y_t$; (3) sum over states
to get the marginal likelihood $f(y_t \mid y_{t-1},\dots)$; (4) apply Bayes' rule to update
the state probability; (5) sum out $s_{t-r}$ to produce the input for the next iteration. By
construction all probabilities lie in $[0,1]$ and sum to one. The sample log-likelihood
$\sum_{t=1}^{T} \log f(y_t \mid y_{t-1},\dots)$ is maximized numerically over
$(\alpha_0,\alpha_1,p,q,\sigma,\phi_1,\dots,\phi_r)$; the model is identified only up to state
labeling, fixed by convention (e.g. $\alpha_1>0$). It generalizes immediately to more than two
states and higher-order dependence, and a smoother gives full-sample state inference.

Applied to quarterly US real GDP (100 times the log change, 1952:II–1984:IV), maximum
likelihood gives $\hat\alpha_1=1.522$, $\hat\alpha_0=-0.3577$, $\hat p=0.9049$, $\hat q=0.7550$,
$\hat\sigma=0.7690$: a negative growth rate of $-0.4\%$ per quarter in state 0 and $+1.2\%$
($\alpha_0+\alpha_1$) in state 1, coinciding with business-cycle phases rather than secular
change. The estimated state sequence matches official recession dates closely; the expected
recession duration is $(1-q)^{-1}\approx 4.1$ quarters (vs. 4.7 historical) and expected
expansion $(1-p)^{-1}\approx 10.5$ quarters, with the expansion-to-recession transition tied
to a permanent GDP fall of about 3%.

## Statistical properties of regimes: tails, skewness, correlations

In the canonical model $y_t = \mu_{s_t} + \phi_{s_t} y_{t-1} + \sigma_{s_t}\varepsilon_t$ with
$s_t$ a first-order Markov chain of transition matrix $\Pi_{[i,j]}=p_{ij}$, recurrent regimes
recur if $p_{ii}<1$, while a change-point process expands the regime set and never returns.
Without AR terms, $y_t=\mu_{s_t}+\sigma_{s_t}\varepsilon_t$ is a mixture of two normals, with
$$ \mathrm{Var}(y_t)=\pi_0(1-\pi_0)(\mu_0-\mu_1)^2+\pi_0\sigma_0^2+(1-\pi_0)\sigma_1^2, $$
$$ \mathrm{skew}(y_t)=\pi_0(1-\pi_0)(\mu_0-\mu_1)\big[(1-2\pi_0)(\mu_0-\mu_1)^2+3(\sigma_0^2-\sigma_1^2)\big]. $$
Mean differences between regimes enter the higher moments: variance is not the simple average
of variances (switching to a different-mean regime adds a risk source), and skewness arises
only if $\mu_0 \neq \mu_1$. For a mixture of $N(1,1)$ with probability 0.8 and $N(-2,2^2)$ with
probability 0.2, the mean is $-0.50$, the standard deviation 2.18 (larger than either
individual one), and skewness $-0.65$. The autocovariances
$\mathrm{cov}(y_t,y_{t-1})=\pi_0(1-\pi_0)(\mu_0-\mu_1)^2[p_{00}+p_{11}-1]$ show that
level persistence requires mean differences while volatility persistence can come from mean
or variance differences, both scaling with combined regime persistence $p_{00}+p_{11}-1$.
A stylized fact is that correlations rise during market downturns; regime models reproduce it
via the persistence of a low-mean/high-volatility/high-correlation regime. The diagnostic is
the *exceedance correlation* — the correlation of the subset
$\{(y_1,y_2) \mid y_1 \ge (1+\theta)\bar y_1, y_2 \ge (1+\theta)\bar y_2\}$ — which in the data
is asymmetric (higher for negative exceedances); a regime model matches it closely where a
normal distribution and an asymmetric GARCH fail.

## Regimes, equilibrium, and the risk-return relation

Embedded in a Lucas representative-agent economy, a regime model generates realistic
risk-return dynamics. With Euler equation
$P_t U'(C_t) = \beta E_t[U'(C_{t+1})(P_{t+1}+D_{t+1})]$, power utility
$U(C)=C^{1+\gamma}/(1+\gamma)$, and consumption equal to the dividend, a dividend process
switching in both mean and volatility yields a closed-form solution $P_t=\rho(s_t)D_t$: the
price-dividend ratio is constant within each regime but depends highly nonlinearly on the
parameters and takes a finite number of values equal to the number of regimes. With
persistent high- and low-growth regimes, two effects compete — an intertemporal relative-price
effect and a consumption-smoothing effect — and which dominates depends on utility concavity:
at $\gamma=-1$ (log utility) they cancel and the ratio is regime-independent; for $\gamma>-1$
the relative-price effect dominates, for $\gamma<-1$ the smoothing effect. The conditional
expected return is therefore regime-dependent and time-varying, and the usual monotone linear
premium-variance relation may fail: near $\gamma=-1$ mean return can peak in the high-growth
regime while return variance peaks in the low-growth regime. When regimes are not identifiable
in real time the state is latent and agents update beliefs by Bayes' rule; their attempt to
hedge regime uncertainty can generate over- and under-reaction and higher price volatility in
high-uncertainty periods around recessions, and filtered probability tracks the state well but
sometimes misses a shift or raises false alarms.

## Two-regime portfolio choice: the opportunity set changes with the state

To test whether rising bear-market correlations kill international diversification, a dynamic
US-investor problem rebalances over $N$ assets monthly on a $T$-month horizon,
$\max E_0[U(W_T)]$ subject to $\alpha_t'\mathbf{1}=1$, with CRRA utility and a regime-switching
opportunity set $y_{t+1}=\mu(s_{t+1})+\Sigma^{1/2}(s_{t+1})\epsilon_{t+1}$ with two states and
transition matrix entries $P,Q$. Estimated on US/UK/German monthly equity returns, the model
finds a "bear" regime 1 (lower mean, much higher volatility, higher correlations) and a
regime 2 (higher mean, lower volatility, lower correlations); volatility and correlations rise
together, the strongest differentiator being volatility, with expected durations of about 6.9
months (regime 1) and 4.25 years (regime 2). When returns are i.i.d. CRRA weights are constant
and the $T$-period problem reduces to the myopic one-period problem; with regimes the optimal
weights become state functions $\alpha_t^*(s_t)$ and persistence induces intertemporal hedging
demands. Because US volatility is lower in regime 1, US equity is the safer asset and
risk-averse investors hold more of it in both regimes, more so in the bear regime; the
estimated transition matrix is of the *momentum* type (more likely to stay than switch), under
which risk-averse investors *reduce* risky exposure as the horizon lengthens.

The economic cost of suboptimal strategies is measured as a certainty-equivalent change in
"cents per dollar" of wealth. The cost of ignoring regimes (using i.i.d. weights) is small for
all-equity portfolios — the high-volatility regime mainly shifts toward lower-volatility
assets, which i.i.d. weights approximate — but rises sharply when a conditionally risk-free
asset is available, since the high-volatility/high-correlation regime induces a dramatic
flight to cash; it then becomes the same order as the cost of not diversifying internationally,
about one cent per dollar at the annual horizon for risk aversion 5. The cost of myopia is
negligible. The main conclusion: a high-volatility bear regime does **not** eliminate the
benefits of international diversification.

## Four economically interpretable multi-asset regimes (Guidolin–Timmermann)

Extending to more asset classes (large caps, small caps, long-term bonds, cash as residual), a
regime-switching VAR lets mean, covariance, and serial correlation vary with the state,
$$ \binom{\mathbf r_t}{\mathbf z_t} = \binom{\boldsymbol\mu_{s_t}}{\boldsymbol\mu_{z s_t}} + \sum_{j=1}^{p} \mathbf A_{j,s_t}\binom{\mathbf r_{t-j}}{\mathbf z_{t-j}} + \binom{\boldsymbol\varepsilon_t}{\boldsymbol\varepsilon_{zt}}, $$
with $S_t$ a first-order Markov chain of constant transition probabilities, the latent state
estimated by the EM algorithm; information criteria select a four-state model, with fewer
states or constant volatility clearly misspecified. The four states are economically
interpretable: regime 1 is a *crash* (large negative mean excess returns, high volatility;
the 1970s oil shocks, October 1987, early 1990s, "Asian flu"); regime 2 is *slow growth* (low
volatility, small positive returns); regime 3 is a persistent *bull*; regime 4 is *recovery*
(strong rebounds, high small-cap and bond volatility). Correlations vary substantially — large-
vs. small-cap from 0.82 (crash) to 0.50 (recovery), stock-bond from $-0.40$ (crash) to
positive elsewhere — with stationary probabilities 9% (crash), 40% (slow growth), 28% (bull),
23% (recovery) and durations of about two, seven, eight, and three months respectively; smoothed
state probabilities correlate positively with recession dates for the high-volatility states.

The power-utility investor incorporates learning by optimally revising state beliefs each
instant, and optimal allocations vary considerably with the state probabilities. Because
equities are unattractive in the crash state, the short-horizon investor holds little there;
but since the crash is transient and likely to move to recovery, its equity allocation *rises*
with horizon, whereas in the persistent slow-growth and bull states short-horizon investors
hold large equity positions that *decrease* with horizon. The demand-horizon relation is
therefore non-monotone and state-dependent: the common advice to raise equity exposure as
horizon lengthens holds in only one of the four regimes. The cost of ignoring regimes reaches
3% at short horizons and about 130 basis points annually at long horizons, remaining
economically significant under parameter uncertainty (the confidence-band lower bound reaching
7–8% of wealth at long horizons), and specifying four regimes rather than two matters out of
sample.

## Measuring the current macro state: nowcasting and the diffusion index

Recognizing the regime requires a timely measure of the current macro state, but most data are
released with delay and later revised. *Nowcasting* formalizes updating the current-quarter
estimate as new data arrive within the month. The target is
$\mathrm{Proj}[GDP_t \mid \Omega^n_v]$ over $n$ monthly series (about 200 for the US) up to
month $v$; because series are released non-synchronously the end-of-sample panel is unbalanced.
The framework is a parametric dynamic factor model decomposing each stationary series into a
common and an idiosyncratic component,
$$ y_{it|v_j}=\mu_i+\lambda_i F_t + \xi_{it|v_j}, \qquad F_t = A F_{t-1} + B u_t, $$
with $B$ full rank $q$; empirically $r=10$ factors and $q=2$ pervasive shocks. Factors are
estimated by principal components, then the Kalman filter updates the signal on the unbalanced
panel, assigning zero weight to missing series by imposing infinite idiosyncratic variance.
Each release expands the information set, and the induced *news* is the revision
$NEWS[i,v_j]=\hat y_{it|v_j}-\hat y_{it|v_{j-1}}$; projection uncertainty decreases as the
dataset expands. Empirically, intra-monthly information matters (signal precision rises
monotonically through the quarter), business surveys (Philadelphia Fed) have large marginal
impact because they arrive early, and interest rates affect GDP precision but not inflation
while asset prices affect inflation but not GDP.

The factor representation compressing hundreds of series into a few state factors is
formalized as the *diffusion index*: with an approximate dynamic factor model
$y_{t+1}=\beta(L)f_t+\gamma(L)y_t+\epsilon_{t+1}$, $X_{it}=\lambda_i(L)f_t+e_{it}$, rewritten in
static form $y_{t+1}=\beta' F_t + \gamma(L)y_t + \epsilon_{t+1}$, $X_t = \Lambda F_t + e_t$, the
factors are estimated by principal components even for very large $N$, and the feasible forecast
is asymptotically first-order efficient as $N,T\to\infty$. On 215 US monthly series the first
six factors explain 39% of variance and the first twelve 53%: the first loads on production and
employment, the second on rate spreads and capacity utilization, the third on interest rates,
the fourth on equity returns, the fifth on inflation, the sixth on housing starts. Out of
sample, diffusion-index forecasts clearly beat univariate autoregressions, VARs, and
leading-indicator models. This connects to the [macro signals and risk premia](./macro-signals-and-risk-premia.md)
predictors, which apply the same factor compression to expected returns.

## From the estimated regime to the view: the Black-Litterman bridge

The elements form a chain from market state to portfolio view. The Markov regime-switching
filter yields, at each instant, a probabilistic estimate of the current regime
$\Pr(s_t \mid I_t)$ from observed returns, while nowcasting and the diffusion index give a
real-time macro-state measure updated at each data release; the two are complementary, and it
remains open whether high-frequency return-inferred regimes can predict low-frequency macro
regimes. The relevance to portfolio selection is direct: the opportunity set changes with the
state, and the investor who exploits a higher-Sharpe frontier in the low-volatility regime and
shifts to the risk-free asset in the bad regime beats one who sits on the unconditional
frontier — the average of the regime frontiers. This is where
[Black-Litterman](../foundations/black-litterman.md) plugs in: the estimated regime — whether
the market is in crash, slow-growth, bull, or recovery, and with what probability — is
precisely what forms the views on expected returns and covariances that the model combines with
market equilibrium. The filtered probability of a bear regime justifies a view of lower
expected returns and higher correlations, which the Black-Litterman update translates into a
tilt away from the market portfolio. Recognizing the market state is the first step — not the
last — in the chain to allocation.
