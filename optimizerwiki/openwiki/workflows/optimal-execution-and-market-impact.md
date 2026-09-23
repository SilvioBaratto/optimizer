---
type: concept
title: "Optimal Execution and Market Impact"
description: How a target portfolio is traded efficiently once turnover has been decided — the implementation shortfall as objective, Kyle's auction microfoundation of linear permanent impact, the Almgren-Chriss efficient trading frontier trading impact cost against timing risk, Bertsimas-Lo dynamic-programming execution with information and cross-impact, Obizhaeva-Wang transient impact and order-book resilience, the Bouchaud-Farmer-Lillo empirical square-root impact law with long memory, and power-law nonlinear scheduling.
tags: [execution, market-impact, implementation-shortfall, kyle-model, almgren-chriss, bertsimas-lo, obizhaeva-wang, square-root-law, order-book-resilience, propagator-model, transaction-costs, turnover]
sources:
  - id: openwiki-source-19f11a41efb7f8a3b3158898
    resource: repo://docs/22_optimal_execution_and_market_impact.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Optimal Execution and Market Impact

Trading a target portfolio has an endogenous cost — *market impact* — and that cost depends on how
the orders are split over time. This chapter microfounds linear permanent impact in Kyle's auction
model, derives the illiquidity coefficient $\lambda$, and builds the Almgren-Chriss efficient trading
frontier as a compromise between impact cost and timing risk. It then extends the problem to the
Bertsimas-Lo optimal control, to transient impact with order-book resilience (Obizhaeva-Wang), to the
Bouchaud-Farmer-Lillo empirical square-root impact law, and to scheduling under power-law impact. The
result closes the loop from the [turnover induced by a decaying signal](../signals/signal-decay-horizon-turnover.md)
to the problem of executing it at minimum cost — the step that follows [portfolio revision](./portfolio-revision.md).

## From target portfolio to orders: the implementation shortfall

Portfolio selection produces a vector of target weights; periodic revision turns the difference between
current and target weights into *turnover* — a quantity of shares to buy and sell. Up to this point the
portfolio is still *on paper*: its expected returns are computed at the observed market prices. This
chapter's problem is that the very act of trading that turnover moves prices against the trader, so the
price actually paid or received differs from the reference (decision) price.

This gap is the *implementation shortfall*, the difference between the value of the ideal portfolio
(executed instantaneously at the decision price) and the portfolio actually obtained, inclusive of the
cost of any unexecuted quantity. The implementation shortfall is the natural objective of the execution
problem: not an exogenous commission cost, but a loss that depends on the chosen trading trajectory. Two
forces pull against each other. Trading fast concentrates the order in little time and so pushes the
price a lot — high *impact cost*; trading slowly dilutes the order but exposes the still-open position to
price volatility for longer — high *timing risk*. The expected cost falls and its variance rises as one
slows down, and the optimum is a compromise between the two.

## The microfoundation of impact: Kyle's auction model

Price impact is not a mechanical friction: it arises from the inference the liquidity provider makes
observing the order flow. Kyle's model gives the canonical microfoundation through three agents — a
risk-neutral *insider* who observes the liquidation value $v$; the *noise traders*, submitting a random
aggregate order $u$; and a risk-neutral, competitive *market maker* who observes only the total order
$x+u$ and sets an efficient price.

**Single auction.** With $v\sim N(p_0,\Sigma_0)$ and $u\sim N(0,\sigma_u^2)$ independent, the insider
chooses $x$ and the market maker, unable to separate $x$ from $u$, sets $p=\mathbb{E}[v\mid x+u]$. One
seeks an equilibrium with a linear insider strategy $x=\beta(v-p_0)$ and a linear pricing rule
$p=p_0+\lambda(x+u)$. Taking $\lambda$ as given, the insider maximizes expected profit
$\mathbb{E}[(v-p_0-\lambda x)x]=(v-p_0)x-\lambda x^2$, whose first-order condition gives
$x=(v-p_0)/(2\lambda)$, i.e. $\beta=1/(2\lambda)$; the market maker's efficiency condition imposes
$\lambda=\operatorname{cov}(v,x+u)/\operatorname{var}(x+u)=\beta\Sigma_0/(\beta^2\Sigma_0+\sigma_u^2)$.
Solving the system yields the unique linear-strategy equilibrium
$$\beta=\frac{\sigma_u}{\sqrt{\Sigma_0}},\qquad \lambda=\frac{1}{2}\,\frac{\sqrt{\Sigma_0}}{\sigma_u},$$
with the residual value variance after observing the order exactly halved, $\operatorname{var}(v\mid
x+u)=\tfrac12\Sigma_0$ — half the private information is impounded into the price — and an ex ante
expected insider profit of $\tfrac12\sqrt{\Sigma_0}\,\sigma_u$.

**The coefficient $\lambda$ as illiquidity and depth.** The rule $p=p_0+\lambda(x+u)$ is a price impact
*linear* in quantity and *permanent* (once moved, the price does not revert). The parameter $\lambda$
measures market illiquidity: it grows with the uncertainty $\Sigma_0$ about fundamental value (more
asymmetric information makes the market maker more cautious) and falls with the noise-trading variance
$\sigma_u^2$ (more noise screens the insider). Its reciprocal $1/\lambda$ is the *depth* of the market —
the order size needed to move the price by one unit.

**Sequential auctions and continuous limit.** Splitting $[0,1]$ into $N$ auctions with the value a
Brownian motion, the equilibrium becomes recursive and, as $N\to\infty$, converges to a continuous one
in which depth is *constant in time*: $\lambda$ does not depend on $t$, while the residual variance
decays linearly, $\Sigma(t)=(1-t)\Sigma_0$, so information is impounded at a constant rate and at $t=1$
the price equals the value. The insider trades ever more aggressively ($\beta(t)\to\infty$ as $t\to1$),
but the market maker keeps depth unchanged because his growing knowledge of the value exactly offsets the
insider's acceleration. This is the theoretical basis for using, in execution models, a permanent-impact
coefficient $\lambda$ that is constant and linear in the order.

## The efficient trading frontier: Almgren-Chriss

Almgren-Chriss translate the impact-cost/timing-risk trade-off into an efficient frontier formally
analogous to the [mean-variance frontier](../foundations/mean-variance-selection.md), with the
implementation shortfall in place of return and its variance in place of risk.

**Setup.** One liquidates a block of $X$ units within horizon $T$, split into $N$ intervals of length
$\tau=T/N$. The *holdings* trajectory $x_k$ ($x_0=X$, $x_N=0$) gives the trade list $n_k=x_{k-1}-x_k$.
The price follows an arithmetic Brownian motion with permanent and temporary impact,
$$S_k=S_{k-1}+\sigma\sqrt{\tau}\,\zeta_k-\tau\,g(n_k/\tau),\qquad \widetilde S_k=S_{k-1}-h(n_k/\tau),$$
where $\widetilde S_k$ is the price actually received. Permanent impact $g$ shifts the equilibrium price
and persists; temporary impact $h$ is a misalignment that dies within the interval. In linear form
$g(v)=\gamma v$ and $h(v)=\varepsilon\operatorname{sgn}(v)+\eta v$, with $\gamma$ the permanent-impact
coefficient, $\eta$ the temporary one, and $\varepsilon$ a fixed cost tied to the half bid-ask spread.

**Expected cost and variance.** The implementation shortfall has expectation and variance
$$E(x)=\sum_k \tau\,x_k\,g(n_k/\tau)+\sum_k n_k\,h(n_k/\tau),\qquad V(x)=\sigma^2\sum_k \tau\,x_k^2,$$
and, with the linear forms and $\tilde\eta=\eta-\tfrac12\gamma\tau$,
$E(x)=\tfrac12\gamma X^2+\varepsilon\sum_k|n_k|+(\tilde\eta/\tau)\sum_k n_k^2$. The permanent term
$\tfrac12\gamma X^2$ is *independent of the trajectory*: it depends only on the total quantity, not on
how it is spread. The variance comes entirely from price volatility on the still-open position.

**The efficient frontier.** A trajectory is efficient if no other reduces both $E$ and $V$
simultaneously; one minimizes $E(x)+\lambda V(x)$, $\lambda\ge0$, where $\lambda$ is a risk-aversion
parameter — how many dollars of variance one accepts to cut a dollar of expected cost. The optimality
condition is a linear second-order difference equation
$(x_{j-1}-2x_j+x_{j+1})/\tau^2=\tilde\kappa^2 x_j$ with $\tilde\kappa^2=\lambda\sigma^2/\tilde\eta$, whose
solution under $x_0=X$, $x_N=0$ is hyperbolic,
$$x_j=\frac{\sinh(\kappa(T-t_j))}{\sinh(\kappa T)}\,X,$$
with $\kappa\simeq\sqrt{\lambda\sigma^2/\eta}$ for $\tau\to0$. The optimal trajectory decays
exponentially with a *characteristic time* (half-life) $1/\kappa$ that depends only on the asset's
liquidity and the risk aversion, not on the imposed horizon $T$.

**Limiting cases.** For $\lambda\to0$ (risk neutrality) one sells at a constant rate $n_k=X/N$ with
maximal variance $V=\tfrac13\sigma^2X^2T(1-1/N)(1-1/(2N))$ — the *naive* minimum-expected-cost strategy;
for $\lambda\to\infty$ one liquidates everything in the first interval, $n_1=X$, with zero variance and
maximal impact cost. As $\lambda$ ranges over $[0,\infty)$ it traces a smooth convex frontier in the
$(V,E)$ plane. At the naive point the frontier has a horizontal tangent, so a first reduction of variance
comes at nearly zero (second-order) expected cost.

**Liquidity-adjusted VaR, drift, and multi-asset.** The same frontier minimizes a cost quantile:
$\mathrm{VaR}_p(x)=E(x)+\lambda_u\sqrt{V(x)}$ with $\lambda_u$ the normal quantile, minimized by choosing
$\lambda=\lambda_u$ in the efficient problem, linking execution to
[coherent risk measures](../risk-measures/coherent-risk-measures.md). A directional view (price drift
$\alpha$) superimposes a target level $\bar x=\alpha/(2\lambda\sigma^2)$, and the per-period gain from a
serial correlation $\rho$ is $\rho^2\sigma^2\tau^2/(4\eta)$ — independent of portfolio size, hence
negligible for large blocks whose impact grows as $X^2$. With $m$ assets, diagonal permanent/temporary
impact matrices decouple the expected cost per asset, but the covariance $C$ still couples the whole
system through the risk term; the problem diagonalizes in the eigenbasis of $\tilde H^{-1/2}C\tilde
H^{-1/2}$ — not asset by asset unless $C$ too is diagonal.

## The optimal control of execution costs: Bertsimas-Lo

Bertsimas-Lo treat the same problem as stochastic optimal control, minimizing only the expected
execution cost by dynamic programming, and obtain the optimal strategy in recursive form, including with
information variables and multiple assets.

**Dynamic formulation.** One must acquire $\bar S$ units in $T$ periods minimizing
$\mathbb{E}[\sum_t P_t S_t]$ under $\sum_t S_t=\bar S$. The state is $(P_{t-1},W_t)$ with $W_t$ remaining
units, $W_{t+1}=W_t-S_t$; with linear permanent impact $P_t=P_{t-1}+\theta S_t+\epsilon_t$ ($\theta>0$),
the value function satisfies $V_t=\min_{S_t}\mathbb{E}_t[P_tS_t+V_{t+1}]$. At the last period one is
forced to buy the remainder, $S_T^\ast=W_T$; solving backward gives $S_{T-k}^\ast=W_{T-k}/(k+1)$ and,
propagating forward from $W_1=\bar S$, the equal-slice policy
$$S_t^\ast=\frac{\bar S}{T},\qquad t=1,\dots,T,$$
with expected cost $V_1=P_0\bar S+\tfrac12\theta\bar S^2(1+1/T)$, the impact term quadratic in total size.

**Information and state-dependent strategy.** Adding an $\mathrm{AR}(1)$ information variable,
$P_t=P_{t-1}+\theta S_t+\gamma X_t+\epsilon_t$, $X_t=\rho X_{t-1}+\eta_t$, the optimal strategy ceases to
be naive and becomes linear in the state, $S_{T-k}^\ast=\delta_{w,k}W_{T-k}+\delta_{x,k}X_{T-k}$: trading
anticipates the information — with $\rho>0$ (persistence) buy more when $X_t$ signals rising prices, with
$\rho<0$ (reversion) procrastinate. A *linear-percentage temporary* variant separates an impact-free
price $\bar P_t=\bar P_{t-1}\exp(Z_t)$ from a temporary-impact component, guaranteeing positive prices
and proportional impact.

**Multi-asset with cross-impact.** With $n$ assets, the vector dynamics $\mathbf P_t=\mathbf
P_{t-1}+\mathbf A\mathbf S_t+\mathbf B\mathbf X_t+\boldsymbol\epsilon_t$ introduce in the positive-definite
matrix $\mathbf A$ off-diagonal terms $A_{ij}$ quantifying *cross-impact*: trading asset $j$ moves the
price of asset $i$. The optimal strategy is still linear in the state, with matrices defined by
Riccati-type recursions; if $\mathbf A$ is diagonal the problem decomposes into $n$ independent
executions, whereas with material cross-impact each asset's order depends on all others' residuals. This
portfolio dimension connects to [risk attribution, budgets and limits](../risk-management/risk-attribution-budget-limits.md).

## Transient impact and order-book resilience: Obizhaeva-Wang

The previous models collapse the market reaction into just two components — permanent and instantaneous.
Obizhaeva-Wang show this dichotomy is incomplete: what governs the optimal strategy is the *dynamics*
with which supply and demand rebuild after a trade — the *resilience* of the order book — not its static
properties (spread and depth).

**Order book and transient impact.** The limit order book is a density $q$ of orders per unit price above
the ask $A_t$; buying $x$ units consumes the book and shifts the ask by $x/q$. After the trade new orders
flow in and the book rebuilds, the ask's deviation from steady state decaying exponentially. With
discrete trades the ask is
$$A_t=V_t+\frac{s}{2}+\sum_{t_i<t}x_i\,\kappa\,e^{-\rho(t-t_i)},\qquad V_t=F_t+\lambda\!\sum_{t_i\le t}x_i,$$
with $F_t$ the fundamental value, $V_t$ the mid-quote shifted by permanent impact $\lambda$, $s$ the
spread, $\kappa=1/q-\lambda$ the transient-impact amplitude and $\rho\ge0$ the *resilience rate*. Each
past trade leaves an exponential tail that sums linearly: $\rho=0$ makes impact permanent, $\rho\to\infty$
makes it vanish instantly, with recovery half-life $\ln2/\rho$.

**Optimal strategy: blocks and continuous flow.** Minimizing the expected cost of executing $X_0$ units
over $[0,T]$ under this dynamics, and passing to the continuous limit, gives a strategy radically
different from uniform slicing:
$$x_0=x_T=\frac{X_0}{\rho T+2},\qquad \dot X_t=\frac{\rho\,X_0}{\rho T+2}\quad(0<t<T).$$
One trades a *discrete initial block*, a *constant-rate continuous flow* in the middle, and a *discrete
final block of the same size* as the first. The initial block pushes the book away from steady state,
attracting new limit orders; the continuous flow harvests them as they arrive, keeping transient impact
in equilibrium; the final block closes the residual. The continuously executed fraction is $\rho
T/(\rho T+2)$.

**Limiting cases.** For $\rho\to0$ (no resilience) the strategy collapses to two equal blocks
$x_0=x_T=X_0/2$ with no intermediate trading; for $\rho\to\infty$ (instant rebuilding) the blocks vanish
and one recovers the uniform execution of Almgren-Chriss and Bertsimas-Lo. Relative to uniform, exploiting
resilience produces cost savings that are largest for intermediate $\rho T$ and can reach the order of
10% when permanent impact $\lambda$ is small. The decisive execution variable is dynamic — how fast the
market digests the trade — not static.

## The empirical impact law: Bouchaud-Farmer-Lillo

The empirical regularities of market impact constrain and partly contradict the linear models. Three
stylized facts emerge from high-frequency data.

**Concave impact and the square-root law.** A single trade's impact is *concave* in volume: the price
change conditional on volume $v$ scales as $\mathbb{E}[r\mid v]\propto v^\psi$ with a small exponent
$\psi\approx0.2$–$0.5$. More strikingly, the impact of a *metaorder* — a large order executed in many
pieces — follows approximately a *square-root law* in traded volume,
$$\Delta p\;\propto\;\sqrt{Q/V},$$
with $Q$ the metaorder size and $V$ a reference volume (the propagator model, with order-flow memory
exponent $\gamma\approx0.5$, predicts a power $N^{1-\beta}$ with $\beta=(1-\gamma)/2\approx0.25$, i.e.
about $N^{3/4}$ — of the same concave spirit but not identical to the square root). Splitting a large
order thus reduces its cost more than proportionally: impact non-linearity is the empirical foundation of
*scheduling*.

**Long memory of the order flow.** Order signs $\epsilon_t\in\{-1,+1\}$ are strongly autocorrelated, with
a power-law-decaying autocorrelation $C(\tau)=\mathbb{E}[\epsilon_t\epsilon_{t+\tau}]\sim\tau^{-\gamma}$,
$\gamma\approx0.5$ — a *long-memory* process (Hurst $H=1-\gamma/2>\tfrac12$) reflecting the practice of
fragmenting large orders: if metaorder sizes are power-law distributed, splitting automatically generates
slowly decaying sign autocorrelations.

**The paradox and the propagator model.** Order flow is thus strongly predictable, yet prices stay nearly
diffusive (martingale) — an apparent contradiction with efficiency. The resolution is that each order's
impact is *transient*, not permanent, and its decay exactly offsets the flow's persistence. The
*propagator model* writes the price as a sum of decaying past impacts,
$p_t=\sum_{t'<t}G(t-t')\epsilon_{t'}+\text{noise}$ with $G(\ell)\sim\ell^{-\beta}$,
$\beta=(1-\gamma)/2$ — the critical exponent that balances long sign memory against impact decay, making
price variance diffusive and returns nearly unpredictable despite predictable order flow. Obizhaeva-Wang's
transient decaying impact finds here its empirical counterpart.

## Nonlinear power-law impact in scheduling

Empirical concavity motivates generalizing the scheduling problem to a power-law impact function with an
arbitrary exponent, developed by Marzo-Ritelli-Zagaglia reworking Almgren's continuous-time formulation.

**Cost functional.** With residual position $x(t)$, $x(0)=X$, $x(T)=0$, and trading speed $\dot x$, one
minimizes the continuous-time implementation cost — power-law temporary-impact cost plus volatility risk
on the open position,
$$\min_{x(\cdot)}\int_0^T\big[\eta\,|\dot x|^{k+1}+\lambda\sigma^2 x^2\big]\,dt,$$
where $k>0$ is the power-law exponent ($k=1$ reproduces Almgren-Chriss quadratic impact) and permanent
impact contributes only a trajectory-independent constant $\tfrac12\gamma X^2$. Since the integrand has no
explicit time dependence, the Beltrami identity gives a first integral
$\lambda\sigma^2 x^2-k\eta(-\dot x)^{k+1}=-k\eta v_0^{k+1}$, with $v_0=-\dot x(T)$ evaluated at the
terminal point $x(T)=0$.

**Solution.** Separating variables gives $(-\dot x)=(v_0^{k+1}+(\lambda\sigma^2/(k\eta))x^2)^{1/(k+1)}$,
whose integration yields an implicit $x$-$t$ relation expressed through the Gaussian hypergeometric
function ${}_2F_1$. The map $x\mapsto x\,{}_2F_1(\cdots)$ is strictly decreasing, so the relation is
invertible for $x(t)$, and the Legendre condition
$\partial^2L/\partial\dot x^2=\eta k(k+1)(-\dot x)^{k-1}>0$ ensures a minimum.

**Linear case and boundary velocity.** For $k=1$ the Euler-Lagrange equation reduces to $\ddot
x=(\lambda\sigma^2/\eta)x$ and, via ${}_2F_1(\tfrac12,\tfrac12;\tfrac32;-z^2)=\operatorname{arsinh}(z)/z$,
the hypergeometric solution collapses to the Almgren-Chriss hyperbolic trajectory
$x(t)=X\sinh(\sqrt{\lambda\sigma^2/\eta}(T-t))/\sinh(\sqrt{\lambda\sigma^2/\eta}T)$. A distinctive result
concerns boundary velocity: imposing zero initial velocity ($v_0=0$, no trading at the initial instant,
as in Almgren) is compatible with the constraints only if $k>1$, in which case the zero-velocity starting
position is finite and determined by
$x(0)=((k-1)T/(k+1))^{(k+1)/(k-1)}(\lambda\sigma^2/(k\eta))^{1/(k-1)}$. For $k\le1$ no zero-velocity
solution exists: the trader must trade from the initial instant with $v_0>0$, and the general
hypergeometric formulation supplies the optimal trajectory for any $v_0$. The impact exponent $k$ thus
determines not only the shape of the trajectory but the very nature of the optimum at the horizon
endpoints.

## Synthesis: executing the turnover at minimum cost

These models form a coherent hierarchy. Kyle provides the microfoundation: linear permanent impact
$\lambda$ arises from the market maker's inference on order flow and measures illiquidity and depth. On
that basis, Almgren-Chriss and Bertsimas-Lo build the optimal execution trajectory as a compromise
between impact cost (trading fast) and timing risk (trading slowly) — the first variationally, with the
mean-variance efficient frontier of trading profiles and the hyperbolic solution; the second by dynamic
programming, with the $\bar S/T$ strategy and its extensions to information and cross-impact.
Obizhaeva-Wang enrich impact with order-book dynamics, replacing the permanent/temporary dichotomy with a
transient impact decaying at resilience rate $\rho$, whence the mixed block-and-flow strategy.
Bouchaud-Farmer-Lillo document the empirical regularities — concave impact, square-root law, long
order-flow memory, propagator model — that constrain these models, and Marzo-Ritelli-Zagaglia with
Almgren close the circle by embedding nonlinear power-law impact into optimal scheduling.

The thread joining them is the tie to portfolio construction. The signal generates, through its decay and
horizon, a [turnover](../signals/signal-decay-horizon-turnover.md); this chapter establishes how to
execute it. The implementation shortfall is the cost that erodes the portfolio's theoretical alpha:
trading fast pays it in impact, trading slowly pays it in price risk. The efficient trading frontier, the
block-and-flow strategy, and nonlinear scheduling are so many answers to the same problem — turning target
weights into executed orders while minimizing the distance between the paper portfolio and the realized
one.
