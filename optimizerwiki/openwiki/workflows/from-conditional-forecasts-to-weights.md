---
type: concept
title: "From Conditional Forecasts to Weights"
description: How a return forecast conditioned on macroeconomic variables becomes optimal portfolio weights — the plug-in two-stage approach (model conditional moments, then optimize) with its estimation-error pathologies and remedies, the decision-theoretic/Bayesian route with subjective predictive distributions, and the direct approach that models the weights themselves via a state index, parametric portfolio policies, or an augmented asset space — with the choice among routes governed by the trade-off against estimation error.
tags: [portfolio-choice, conditional-forecast, plug-in, parametric-portfolio-policies, state-index, augmented-asset-space, hedging-demand, market-timing, estimation-error, shrinkage, bayesian, macro-conditioning]
sources:
  - id: openwiki-source-e9d02bf56aceadfba6a69d60
    resource: repo://docs/30_from_conditional_forecasts_to_weights.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# From Conditional Forecasts to Weights

This chapter establishes how a return forecast conditioned on macroeconomic variables translates into
optimal portfolio weights. It distinguishes the two-stage plug-in approach — model the conditional
moments, then optimize — from the approach that models the weights directly: variable selection via a
state index, parametric policies as functions of characteristics, and augmenting the asset space. Each
route closes the pipeline carrying [macro signals](../regimes/macro-signals-and-risk-premia.md) into the
convex optimizer, and the choice among them is governed by the trade-off against estimation error. The
conditioning views connect to [Black-Litterman](../foundations/black-litterman.md) and
[market regimes and views](../regimes/market-regimes-and-views.md), the cross-sectional characteristics
to [quantitative selection and construction](../signals/quantitative-selection-and-construction.md), and
the intertemporal rebalancing to [portfolio revision](./portfolio-revision.md).

## Macro conditioning and the weight problem

Means, variances, covariances and higher moments of stock and bond returns vary over time and are partly
predictable from economic variables; this chapter's problem is to translate that predictability into
weights: given a forecast conditioned on macroeconomic information, how to obtain the optimal allocations
handed to the convex optimizer. An investor choosing at $t$ the weights $x_t$ on $N$ risky assets and one
riskless asset to maximize expected utility of wealth at $t+\tau$, with observable state vector $z_t$ and
$y_t\equiv[r_t,z_t]$ a first-order Markov process, solves
$$V(\tau,W_t,z_t)=\max_{\{x_s\}}\mathrm{E}_t[u(W_{t+\tau})], \qquad W_{s+1}=W_s(x_s'r_{s+1}+R_s^f),$$
with value function $V$; rewriting as a one-period state-dependent choice gives the Bellman equation
$V(\tau,W_t,z_t)=\max_{x_t}\mathrm{E}_t[V(\tau-1,W_t(x_t'r_{t+1}+R_t^f),z_{t+1})]$ and first-order
conditions $\mathrm{E}_t[V_2(\tau-1,\cdot,z_{t+1})r_{t+1}]=0$, generally solvable only numerically.
$\tau=1$ is the static one-period choice; $\tau>1$ is dynamic and multiperiod.

Under constant relative risk aversion (CRRA) $u(W)=W^{1-\gamma}/(1-\gamma)$, homotheticity lets one
normalize $W_t=1$ and the value function depends only on horizon and states. If excess returns
$r_{t+1}$ are contemporaneously independent of the state innovations $z_{t+1}$, the conditional
expectation factorizes and the multiperiod first-order conditions coincide with the one-period ones; if
not, the dynamic choice differs from the myopic one by the *hedging demand*, with which the investor
hedges against changes in investment opportunities. In continuous time, the Hamilton-Jacobi-Bellman
solution splits into
$$x_t^\star=\underbrace{-\frac{V_2}{W_t V_{22}}(\Sigma_t^p)^{-1}\mu_t^p}_{\text{myopic demand}}\;\underbrace{-\frac{V_2}{W_t V_{22}}\frac{V_{23}}{V_2}(\Sigma_t^p)^{-1}D_t^p\rho_t'D_t^{z\prime}}_{\text{hedging demand}},$$
the myopic term being a fraction $1/\gamma_t$ of the tangency portfolio of the instantaneous
mean-variance frontier — the *market-timing* channel — and the hedging term the intertemporal channel,
absent when $\tau=1$ or when states do not predict changes in future opportunities. Obtaining weights
from a conditional forecast admits two fundamentally different strategies: the traditional two-stage
procedure (model and estimate the conditional moments, then solve for weights) and the alternative that
skips the first stage and models or estimates the optimal weights directly.

## The plug-in approach: from estimated moments to weights

In the plug-in approach the expected-utility solution is a map from preference parameters $\phi$, the
state $z_t$ and data-generating-process parameters $\theta$ to weights, $x_t^\star=x(\phi,z_t,\theta)$;
one obtains a consistent estimate $\hat\theta$ and plugs it in, $\hat x_t^\star=x(\phi,z_t,\hat\theta)$,
with the delta method giving $\sqrt T(\hat x_t^\star-x_t^\star)\to N[0,x_3(\cdot)V_\theta x_3(\cdot)']$.
In the mean-variance problem with i.i.d. excess returns the optimal weights are
$x^\star=(1/\gamma)\Sigma^{-1}\mu$, estimated by sample analogs with the anomalous $T-N-2$ degrees of
freedom that make $\hat\Sigma^{-1}$ unbiased under normality, so $\mathrm{E}[\hat
x^\star]=(1/\gamma)\Sigma^{-1}\mu$. Britten-Jones shows the plug-in tangency weights come from an OLS
regression of a vector of ones on the excess returns without intercept, so standard theory allows
$t$- and $F$-inference on the weights.

The imprecision is severe. For one risky asset,
$$\mathrm{var}[\hat x^\star]=\frac{1}{\gamma^2}\Big(\frac{\mu}{\sigma^2}\Big)^2\Big(\frac{\mathrm{var}[\hat\mu]}{\mu^2}+\frac{\mathrm{var}[\hat\sigma^2]}{\sigma^4}\Big),$$
scaled by the optimal weight and driven by the imprecision of risk premium and volatility each relative
to its own size. With ten years of monthly data, $\mu=6\%$, $\sigma=15\%$, $\gamma=5$, the standard
error of $\hat x^\star$ is 14% against a true weight of 53.3%; under GARCH the sampling variance of the
sample variance inflates enormously and the standard error can exceed 100%. In finite samples precision
worsens drastically with the number of assets because the covariance matrix's distinct elements grow
quadratically; Jobson-Korkie simulations show extremely volatile plug-in frontiers well below the true
one, with extreme unstable weights, and Michaud describes mean-variance optimizers as "error
maximizers" that give large positive (negative) weights to assets with large positive (negative)
estimation errors in the risk premium. A economic measure of the error is the certainty-equivalent loss,
$$\mathrm{CE}-\mathrm{E}[\hat{\mathrm{CE}}]\simeq \frac{\gamma}{2}\,\mathrm{tr}[\mathrm{cov}[\hat x^\star]\,\Sigma],$$
arising from *parameter uncertainty* penalized like intrinsic return uncertainty, an order of magnitude
smaller than the weight's standard error since first-order deviations from optimal rules have
second-order consequences (Kan-Zhou seek a scalar $c$ in $c\,\hat\Sigma^{-1}\hat\mu$ minimizing this
loss).

Three complementary remedies improve the estimates. *Shrinkage*: the James-Stein estimate
$\mu_s=\delta\mu_0+(1-\delta)\bar\mu$ contracts sample means toward a common value with optimal
$\delta^\star=\min[1,(N-2)/T/((\bar\mu-\mu_0)'\Sigma^{-1}(\bar\mu-\mu_0))]$, and shrinkage applies to the
covariance too ($\hat\Sigma_s=\delta\hat S+(1-\delta)\hat\Sigma$), which toward a positive-definite target
also guarantees a positive-definite estimate when $N>T$. *Factor structure* reduces dimensionality
($\Sigma=\sigma_m^2\beta\beta'+\Sigma_\epsilon$ for one factor, $\Sigma=B\Sigma_f B'+\Sigma_\epsilon$ in
general). *Weight constraints* truncate extreme positions; Jagannathan-Ma show no-short-sale and
position limits equal a covariance shrinkage $\tilde\Sigma=\Sigma+(\delta\iota'+\iota\delta')-(\lambda
\iota'+\iota'\lambda)$ with the constraint multipliers, explaining why constraints help even when not
theoretically justified.

## The decision-theoretic approach: subjective distributions and economic beliefs

The second traditional route is decision theory: the econometrician takes the investor's role and
chooses weights optimal with respect to a *subjective* belief about the return distribution, which under
parameter (or parameterization) uncertainty can differ sharply from plugging in point estimates. Writing
the myopic maximization as $\max_{x_t}\int u(x_t'r_{t+1}+R^f)p(r_{t+1}\mid\theta)dr_{t+1}$, when $\theta$
is unknown there are three ways forward: naively use estimates (plug-in), take worst-case parameter
values within a set (robust control), or eliminate the unknown parameters by replacing the true
distribution with a subjective one depending only on data and priors. Bayes' theorem gives the posterior
$p(\theta\mid Y_T)\propto p(Y_T\mid\theta)p_0(\theta)$, which integrates out the parameter into the
subjective predictive distribution
$$p(r_{t+1}\mid Y_T)=\int p(r_{t+1}\mid\theta)p(\theta\mid Y_T)d\theta,$$
substituted into the maximization; with non-informative priors the posterior mean of the weights matches
the frequentist estimate up to degrees of freedom, but prior variability relative to data information
automatically determines how much estimates are shrunk toward the target — a Bayesian reading of
shrinkage.

Economic beliefs enter as informative priors. For return predictability, the regression
$r_{t+1}=a+b z_t+\varepsilon_{t+1}$ is very noisy ($R^2\approx1\%$), so a prior centered on market
efficiency ($b=0$) yields, in Connor's form,
$$\hat b_{\mathrm{Bayes}}=\Big[\frac{T}{T+1/\rho}\Big]\hat b_{ols},$$
with $\rho\simeq\mathrm{E}[R^2]$, contracting the OLS coefficient about halfway to zero with $1\%$
expected $R^2$ and five-to-ten years of data. Pastor centers a prior on the cross-sectional intercepts
$\alpha$ whose dispersion measures belief in the CAPM, so the posterior contracts intercepts toward zero,
$\mathrm{E}[\alpha\mid Y_T]=(1-\delta)\hat\alpha_{ols}$. Averaging over parameter values extends to *model
uncertainty*: with a finite model set, posterior probabilities $p(M_j\mid Y_T)$ build a model-averaged
predictive distribution $p(r_{t+1}\mid Y_T)=\sum_j p(r_{t+1}\mid Y_T,M_j)p(M_j\mid Y_T)$, whose
uncertainty contributes to the subjective return variance as much as, or more than, parameter
uncertainty.

## Modeling the weights directly

The traditional approach, plug-in or decision-theoretic, is inherently two-stage; an alternative skips
the first stage and infers directly on the optimal weights. Three reasons favor this beyond the weights
being the object of interest. First, return modeling is the traditional approach's weak point: there is
wide disagreement on how to model returns, the documented state-moment relations are usually tenuous,
with grave risk of misspecification and estimation error later amplified by the optimizer — the intuition
is that weights are easier to model than conditional distributions. Second, dimensionality reduction: an
unconditional 500-asset mean-variance problem involves over 125,000 return-model parameters but only 500
final weights, so focusing on weights shrinks the room for misspecification and estimation error. Third,
inferring on weights lends itself naturally to an expected-utility loss, avoiding the incoherent practice
of estimating the return model under quadratic loss then switching to expected utility for the weights.

The *nonparametric* method characterizes the optimal weights by the conditional Euler conditions
$\mathrm{E}_t[u'(x_t'r_{t+1}+R_t^f)r_{t+1}]=0$, replacing the conditional expectations with kernel
regressions,
$$\hat x(z)=\Big\{x:\ \frac{1}{Th_T^K}\sum_{t=1}^T\omega\Big(\frac{z_t-z}{h_T}\Big)u'(x'r_{t+1}+R_t^f)\,r_{t+1}=0\Big\},$$
consistent and asymptotically Gaussian but suffering the curse of dimensionality — in practice no more
than two predictors with postwar monthly or quarterly data. The other two direct methods — the state
index and augmenting the asset space — address exactly this dimensional limit.

## Variable selection via a state index

Which predictors to follow is not settled by forecasting the individual moments separately, because the
moments relevant to choice are *endogenous* to preferences. Even in the mean-variance case, where the
optimal allocation is proportional to the ratio of conditional mean to variance, one wants variables that
predict that *ratio* well, not the two moments separately: a variable may help the mean but hurt the
variance, and different objectives emphasize different parts of the distribution (a loss-averse investor
cares about the left tail), so different investors may select different predictors. This suggests
selecting variables to predict the optimal weights directly. The one-period mean-variance policy is
$$\alpha_t=\Sigma_t^{-1}\iota\,\frac{\gamma W_t-\iota'\Sigma_t^{-1}\mu_t}{\gamma W_t\,\iota'\Sigma_t^{-1}\iota}+\frac{\Sigma_t^{-1}\mu_t}{\gamma W_t},$$
highly nonlinear and non-monotone, reducing with a riskless asset to a fraction of the tangency portfolio
$\alpha_t^{tgc}=(1/\gamma W_t)\mathrm{E}[r_{t+1}^{tgc}\mid Z_t]/\mathrm{Var}[r_{t+1}^{tgc}]$, proportional
to the conditional mean-variance ratio of the tangency portfolio.

To dodge the curse of dimensionality one adopts a *semiparametric* approach: assume the optimal weights
depend on the predictors only through a single linear index $Z_t'\beta$, leaving the dependence on that
index free, $\alpha_t\equiv\alpha(Z_t'\beta;\beta)$. Collapsing the state vector into a univariate index
allows one-dimensional kernel regression with parametric convergence rate $\sqrt T$ independent of the
number of predictors, and gives a univariate summary of investment opportunities. The coefficients
$\beta$ come from the unrestricted conditional Euler conditions, giving moment conditions $\mathrm{E}
[m_{t+1}(\beta)\mid Z_t]=0$ solved as a standard GMM problem, or alternatively by maximizing unconditional
expected utility $\max_\beta\mathrm{E}[v(W_t(\alpha(Z_t'\beta;\beta)'R_{t+1}))]$ — the utility loss from
collapsing information into one index is empirically negligible in most cases. Applied to four popular
predictors (default spread, S&P 500 log dividend yield, term spread, and an index momentum/trend
variable), the term spread is by far the most important predictor of expected stock and bond returns, the
default spread predicts variances and covariance positively, and the log dividend yield predicts expected
equity returns at the annual horizon; the index reaches a *compromise* between predicting means and
variances, and with short-sale constraints relaxed the CRRA indices are identical across risk-aversion
levels. All investors but the very risk-averse do market timing (more at long horizons than short), while
hedging demands turn out surprisingly small — indeed *negative* — since the endogenously dominant state
variable is the term spread and returns are not much correlated with its changes.

## Parametric portfolio policies

The simplest way to estimate weights directly is to parameterize them as a function of observables —
state variables and/or firm characteristics — and solve for the parameters maximizing expected utility,
avoiding covariance estimation. With $N_t$ securities each with excess return $r_{i,t+1}$ and
characteristics $y_{i,t}$ (market beta, size, book-to-market, past twelve-month return), the optimal
weights are parameterized as
$$x_{i,t}=\bar x_{i,t}+\frac{1}{N_t}\,\theta'\hat y_{i,t},$$
where $\bar x_{i,t}$ is the benchmark weight, $\theta$ a coefficient vector, and $\hat y_{i,t}$ the
cross-sectionally standardized characteristics (zero mean, unit standard deviation each date). The
standardization makes the cross-sectional distribution stationary and makes the deviations $\theta'\hat
y_{i,t}$ sum to zero so the weights always sum to one, and $1/N_t$ normalizes for an arbitrary number of
securities. Crucially $\theta$ varies neither across assets (the policy cares only about
characteristics, not the securities themselves) nor over time (so it also maximizes unconditional
expected utility), letting one estimate it by maximizing the sample analog
$$\max_\theta\ \frac{1}{T}\sum_{t=0}^{T-1}u(r_{p,t+1})=\frac{1}{T}\sum_{t=0}^{T-1}u\Big(\sum_{i=1}^{N_t}\Big(\bar x_{i,t}+\frac{1}{N_t}\theta'\hat y_{i,t}\Big)r_{i,t+1}\Big),$$
with standard GMM standard errors. The method has three advantages: it optimizes portfolios with very
many securities as long as the parameter vector stays small; the optimal weights are far less exposed to
error maximization and overfitting (much less extreme than plug-in, stable in and out of sample), since
the whole portfolio is optimized by choosing few parameters; and it implicitly accounts for the
dependence of expected returns, variances, covariances and higher moments on the characteristics.
Combined with parametric market timing, letting the characteristics' impact vary over time via macro
predictors $z_t$,
$$x_{i,t}=\bar x_{i,t}+\frac{1}{N_t}\,\theta'(z_t\otimes\hat y_{i,t}),$$
the problem becomes a cross-sectional parameterized choice on an asset space augmented with naively
managed portfolios — the connection to the next section.

## Augmenting the asset space

The third direct method turns the dynamic conditional problem into a static Markowitz problem on an
augmented asset set. Assuming the optimal weights are linear in $K$ state variables, $x_t=\theta z_t$
with $\theta$ an $N\times K$ matrix, the conditional problem uses the identity
$(\theta z_t)'r_{t+1}=\mathrm{vec}(\theta)'(z_t\otimes r_{t+1})$; defining $\tilde x=\mathrm{vec}(\theta)$
and $\tilde r_{t+1}=z_t\otimes r_{t+1}$, the same $\tilde x$ maximizes the conditional mean-variance
trade-off at every date and therefore the *unconditional* one,
$$\max_{\tilde x}\ \mathrm{E}[\tilde x'\tilde r_{t+1}]-\frac{\gamma}{2}\mathrm{E}[(\tilde x'\tilde r_{t+1})^2],$$
the unconditional mean-variance problem over the augmented set of $N\times K$ assets. These augmented
assets are *conditional* (managed) portfolios, each investing in one base asset an amount proportional to
a state variable, and the solution
$$\tilde x^\star=\frac{1}{\gamma}\,\mathrm{E}[\tilde r_{t+1}\tilde r_{t+1}']^{-1}\mathrm{E}[\tilde r_{t+1}]$$
depends only on the *unconditional* moments of the augmented set — no assumption on the conditional joint
distribution, on how conditional moments depend on states, or on how states evolve. The idea extends to
the multiperiod case with *timing* portfolios,
$$\tilde r_{t\to t+H}=\Big\{\prod_{i=0,\,i\neq j}^{H-1}R^f_{t+i}\,r_{t+j+1}\Big\}_{j=0}^{H-1},$$
each investing in the risky assets in one period and the riskless asset in the others; combining
conditional and timing portfolios (substituting $z_{t+j}\otimes r_{t+j+1}$) gives the optimal allocations
to conditional portfolios at each date. Linearity in states is innocuous since $z_t$ can include
nonlinear transformations. The appeal is simplicity: all statistical techniques for the static
mean-variance problem apply directly to one- and multi-period market timing, so macro conditioning
becomes a simple set of timing portfolios (e.g. more equity when the term spread is high) solved by the
static optimizer.

## The methodological map and the trade-offs

The whole path from conditional moments to weights organizes along one axis. On one side the traditional
two-stage approach: model the return distribution — by plugging in point estimates (plug-in) or forming a
subjective belief (decision-theoretic/Bayesian) — then solve for weights. On the other the approach that
skips the first stage and infers directly on weights, with nonparametric (kernel regressions on
conditional Euler conditions), semiparametric (the state index $Z_t'\beta$) and parametric (weights as a
linear function of states and characteristics) methods. The selection criterion along this axis is the
trade-off against estimation error. Plug-in is unbiased or consistent but extremely imprecise in finite
samples — imprecision scaled by weight size, worsening quadratically with the number of assets, producing
extreme unstable weights with the optimizer as "error maximizer"; the first-stage remedies (shrinkage,
factor models, constraints) cut variability at the cost of bias and are unified since both Bayesian
shrinkage and constraints equal contractions. The decision-theoretic approach integrates out parameter
and model uncertainty, recognizing these contribute to subjective return variance as much as intrinsic
uncertainty.

The direct approach attacks the error at its root by dimensionality reduction — 500 assets need over
125,000 return-model parameters but only 500 final weights — so focusing on weights shrinks the room for
misspecification and estimation error. Variable selection via the index $Z_t'\beta$ avoids the curse of
dimensionality by collapsing the state vector into a univariate index with $\sqrt T$ convergence, at
negligible empirical utility loss; parametric policies reduce the whole choice to a few coefficients
$\theta$ constant across assets and time, estimated by maximizing sample utility without estimating the
covariance matrix, giving less extreme and stable weights; augmenting the asset space returns the dynamic
conditional choice to a static Markowitz problem on managed portfolios depending only on unconditional
moments. A common trait of the direct methods is coherence between the estimation loss and the economic
objective: inferring on weights lends itself to an expected-utility loss, avoiding the incoherence of
estimating returns under quadratic loss then switching to expected utility. With this the map closes:
the macro signals — term-structure slope, credit spread, dividend yield — enter as state vector $z_t$ or
index $Z_t'\beta$; through one of the routes they become optimal weights, as a timing allocation, a
benchmark deviation, or a convex-problem solution; and the constrained, non-extreme weights are precisely
what the convex optimizer is equipped to produce. It is the pipeline that carries raw data to weights.
