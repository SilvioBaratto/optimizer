---
title: "Optimal Execution and Market Impact"
chapter: 22
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-15
---

> [!abstract] Summary
> The chapter establishes that trading a target portfolio has an endogenous cost — market impact — and that this cost depends on how orders are split over time. It micro-founds permanent linear impact in Kyle's auction model, derives from it the illiquidity coefficient $`\lambda`$, and builds the Almgren-Chriss efficient trading frontier as a trade-off between impact cost and timing risk; it then extends the problem to Bertsimas-Lo's optimal control, to Obizhaeva-Wang's transient impact with order book resilience, to the empirical square-root impact law and the Bouchaud-Farmer-Lillo propagator model, and to scheduling with power-law impact. The result reconnects the turnover induced by the signal to the problem of executing it at minimum cost.

## From the Target Portfolio to Orders: Implementation Shortfall

Portfolio selection produces a target weight vector; periodic review (see [Chapter 11: Signal Decay, Horizon, and Turnover](11_signal_decay_horizon_and_turnover.md)) transforms the difference between current and target weights into a *turnover*, that is, a quantity of securities to buy and sell. Up to this point the portfolio is still *on paper*: its expected returns are computed at observed market prices. The problem addressed in this chapter is that the very act of trading that turnover moves prices against the trader, so that the price actually paid or received differs from the reference price at the time of decision.

This gap is the *implementation shortfall*, the difference between the value of the ideal portfolio (executed instantaneously at the decision price) and that of the portfolio actually obtained, including the cost of any quantity left unexecuted [Perold1988]. Implementation shortfall is the natural objective function of the execution problem: not an exogenous commission cost, but a loss that depends on the chosen trading trajectory.

Two forces emerge in tension. Executing quickly concentrates the order in a short time, and thus pushes the price a great deal: high *impact cost*. Executing slowly dilutes the order, but exposes the still-open position to price volatility for a longer time: high *timing risk*. Expected cost decreases and its variance increases as one slows down; the optimal choice is a trade-off between the two. The following sections build this trade-off starting from its micro-foundation — why the price moves when one trades — up to closed-form execution strategies and the empirical regularities of impact.

## The Micro-Foundation of Impact: Kyle's Auction Model

Price impact is not a mechanical friction: it arises from the inference that liquidity providers make by observing order flow. The model of [Kyle1985] provides the canonical micro-foundation through three agents: a risk-neutral *insider* who observes the liquidation value $`v`$ of the asset; *noise traders*, who submit a random aggregate order $`u`$; and a risk-neutral, competitive *market maker*, who observes only the overall order $`x+u`$ and sets an efficient price.

**Single auction.** Assume $`v\sim N(p_0,\Sigma_0)`$ and $`u\sim N(0,\sigma_u^2)`$, independent. The insider chooses the quantity $`x`$; the market maker, unable to distinguish $`x`$ from $`u`$, sets a price equal to the conditional expectation of the value given the observed flow, $`p=\mathbb{E}[v\mid x+u]`$. An equilibrium is sought in which the insider's strategy is linear, $`x=\beta(v-p_0)`$, and the pricing rule is linear, $`p=p_0+\lambda(x+u)`$. The insider, taking $`\lambda`$ as given, maximizes the expected profit

$$\mathbb{E}[(v-p)\,x]=\mathbb{E}\big[(v-p_0-\lambda x)\,x\big]=(v-p_0)x-\lambda x^2,$$

whose first-order condition gives $`x=\dfrac{v-p_0}{2\lambda}`$, that is, $`\beta=\dfrac{1}{2\lambda}`$. The market maker's efficiency condition requires, for the Gaussian linear projection,

$$\lambda=\frac{\operatorname{cov}(v,\,x+u)}{\operatorname{var}(x+u)}=\frac{\beta\Sigma_0}{\beta^2\Sigma_0+\sigma_u^2}.$$

Solving the system in the two unknowns $`\beta,\lambda`$ yields the unique equilibrium in linear strategies [Kyle1985]:

$$\beta=\frac{\sigma_u}{\sqrt{\Sigma_0}},\qquad \lambda=\frac{1}{2}\,\frac{\sqrt{\Sigma_0}}{\sigma_u}.$$

The residual variance of the value after observing the order is exactly halved,

$$\operatorname{var}(v\mid x+u)=\tfrac{1}{2}\Sigma_0,$$

which means that half of the private information is incorporated into the price; the insider's ex ante expected profit is $`\tfrac{1}{2}\sqrt{\Sigma_0}\,\sigma_u`$.

**The coefficient $`\lambda`$ as illiquidity and depth.** The rule $`p=p_0+\lambda(x+u)`$ is a price impact that is *linear* in the quantity and *permanent* (once moved, the price does not revert). The parameter $`\lambda`$ is the market's illiquidity measure: it increases with uncertainty $`\Sigma_0`$ about the fundamental value (more asymmetric information makes the market maker more cautious) and decreases with the variance $`\sigma_u^2`$ of noise trading (more noise shields the insider). Its reciprocal $`1/\lambda`$ is the market's *depth*: the amount of order needed to move the price by one unit.

**Sequential auctions and the continuous limit.** Splitting the interval $`[0,1]`$ into $`N`$ auctions, with the liquidation value $`v\sim N(p_0,\Sigma_0)`$ drawn once and revealed at $`t=1`$, and noise-trader orders $`u(t)`$ following a Brownian motion with instantaneous variance $`\sigma_u^2`$, the equilibrium becomes recursive: at each auction $`\Delta x_n=\beta_n(v-p_{n-1})\Delta t_n`$ and $`\Delta p_n=\lambda_n(\Delta x_n+\Delta u_n)`$, with $`\lambda_n=\beta_n\Sigma_n/\sigma_u^2`$ and $`\Sigma_n=(1-\beta_n\lambda_n\Delta t_n)\Sigma_{n-1}`$. In the limit $`N\to\infty`$ the equilibrium converges to the continuous one, in which depth is *constant over time*: $`\lambda`$ does not depend on $`t`$, while the residual variance decays linearly,

$$\Sigma(t)=(1-t)\,\Sigma_0,$$

so that information is incorporated at a constant rate and at $`t=1`$ the price coincides with the value. The insider trades ever more aggressively, $`\beta(t)\to\infty`$ as $`t\to1`$, the less time remains, but the market maker keeps depth unchanged because their growing knowledge of the value exactly offsets the insider's acceleration [Kyle1985]. This is the theoretical foundation that justifies the use, in execution models, of a constant permanent impact coefficient $`\lambda`$ linear in the order.

## The Efficient Trading Frontier: Almgren-Chriss

The model of [AlmgrenChriss2000] translates the trade-off between impact cost and timing risk into an efficient frontier formally analogous to the mean-variance frontier of [Chapter 2: Mean-Variance Selection](02_mean_variance_selection.md), with implementation shortfall in place of return and its variance in place of risk.

**Setup.** A block of $`X`$ units is liquidated over the horizon $`T`$, divided into $`N`$ intervals of duration $`\tau=T/N`$. The *holdings* trajectory $`x_k`$ (units still in the portfolio at $`t_k=k\tau`$) satisfies $`x_0=X`$, $`x_N=0`$; the trade list is $`n_k=x_{k-1}-x_k`$. The price evolves as an arithmetic Brownian motion with permanent and temporary impact:

$$S_k=S_{k-1}+\sigma\sqrt{\tau}\,\zeta_k-\tau\,g\!\Big(\frac{n_k}{\tau}\Big),\qquad \widetilde S_k=S_{k-1}-h\!\Big(\frac{n_k}{\tau}\Big),$$

where the $`\zeta_k`$ are independent standard normal shocks and $`\widetilde S_k`$ is the price actually received. The *permanent* impact $`g`$ shifts the equilibrium price and persists; the *temporary* impact $`h`$ is a misalignment that dissipates within the interval. In the linear forms,

$$g(v)=\gamma v,\qquad h(v)=\varepsilon\,\operatorname{sgn}(v)+\eta\,v,$$

with $`\gamma`$ the permanent impact coefficient, $`\eta`$ the temporary impact coefficient, $`\varepsilon`$ a fixed cost linked to half the bid-ask spread.

**Expected cost and variance.** The trajectory's implementation shortfall has expected value and variance

$$E(x)=\sum_{k=1}^N \tau\,x_k\,g\!\Big(\frac{n_k}{\tau}\Big)+\sum_{k=1}^N n_k\,h\!\Big(\frac{n_k}{\tau}\Big),\qquad V(x)=\sigma^2\sum_{k=1}^N \tau\,x_k^2 .$$

With the linear forms, and setting $`\tilde\eta=\eta-\tfrac{1}{2}\gamma\tau`$,

$$E(x)=\tfrac{1}{2}\gamma X^2+\varepsilon\sum_k|n_k|+\frac{\tilde\eta}{\tau}\sum_k n_k^2 .$$

The term $`\tfrac12\gamma X^2`$ of the permanent impact is independent of the trajectory: it depends only on the total quantity, not on how it is distributed. The variance comes entirely from the price volatility on the still-open position.

**The efficient frontier.** A trajectory is efficient if no other one simultaneously reduces both $`E`$ and $`V`$; the comparison is among static trajectories, fixed in advance, and strategies that adapt to realized prices can improve on the resulting frontier. The problem is solved by minimizing the combination

$$\min_x\;E(x)+\lambda\,V(x),\qquad \lambda\ge 0,$$

where $`\lambda`$ is a risk-aversion parameter, in inverse dollars: it is the reduction in expected cost, in dollars, that one requires before accepting one more square dollar of variance (along the frontier $`dE/dV=-\lambda`$). For trajectories whose trades $`n_k`$ never change sign, so that $`\varepsilon\sum_k|n_k|=\varepsilon X`$ is constant, and for $`\tilde\eta>0`$, which makes $`E`$ strictly convex, the optimality condition is a second-order linear difference equation,

$$\frac{x_{j-1}-2x_j+x_{j+1}}{\tau^2}=\tilde\kappa^2\,x_j,\qquad \tilde\kappa^2=\frac{\lambda\sigma^2}{\tilde\eta},$$

whose solution with boundary conditions $`x_0=X`$, $`x_N=0`$ is, in terms of hyperbolic functions,

$$x_j=\frac{\sinh\!\big(\kappa(T-t_j)\big)}{\sinh(\kappa T)}\,X,\qquad n_j=\frac{2\sinh(\tfrac12\kappa\tau)}{\sinh(\kappa T)}\,\cosh\!\big(\kappa(T-t_{j-1/2})\big)\,X,$$

with $`\kappa`$ solving $`\tfrac{2}{\tau^2}(\cosh(\kappa\tau)-1)=\tilde\kappa^2`$ and, as $`\tau\to0`$, $`\kappa\simeq\sqrt{\lambda\sigma^2/\eta}`$. The optimal trajectory decays exponentially with a *characteristic time* $`1/\kappa`$, the time over which the position falls by a factor $`e`$ (Almgren-Chriss call it the half-life; a half-life proper would be $`\ln 2/\kappa`$), which depends on the security's liquidity and volatility and on risk aversion, not on the imposed horizon $`T`$.

**Limiting cases.** As $`\lambda\to0`$ (risk neutrality) one obtains the linear strategy that sells at a constant rate, $`n_k=X/N`$, with maximum variance $`V=\tfrac13\sigma^2X^2T(1-\tfrac1N)(1-\tfrac1{2N})`$; this is the *naive* minimum-expected-cost strategy. As $`\lambda\to\infty`$ (maximum risk aversion) everything is liquidated in the first interval, $`n_1=X`$, with zero variance and maximum impact cost. As $`\lambda\in[0,\infty)`$ varies, a smooth and convex frontier is traced in the $`(V,E)`$ plane: this is the efficient trading frontier. At the naive strategy's point the frontier has a horizontal tangent, so that a first reduction in variance is obtained at nearly zero (second-order) expected cost.

**Value at Risk of liquidation.** The same frontier allows minimizing the cost quantile. For Gaussian price increments and static trajectories, defining $`\mathrm{VaR}_p(x)=E(x)+\lambda_v\sqrt{V(x)}`$, with $`\lambda_v=\Phi^{-1}(p)`$ the standard normal quantile at level $`p`$, the *liquidity-adjusted VaR* is the minimum of $`\mathrm{VaR}_p`$ over trajectories. Its minimizer $`x^\ast`$ lies on the efficient frontier, at the point where $`E`$ plotted against $`\sqrt V`$ has slope $`-\lambda_v`$, that is, at $`\lambda=\lambda_v/\big(2\sqrt{V(x^\ast)}\big)`$ in the efficient problem, not at $`\lambda=\lambda_v`$ [AlmgrenChriss2000]; it links execution to the risk measures of [Chapter 4: Coherent Risk Measures](04_coherent_risk_measures.md).

**Drift and serial correlation.** If the trader has a directional view, that is, a drift $`\alpha`$ in the price, the optimal trajectory is superimposed on a target level $`\bar x=\alpha/(2\lambda\sigma^2)`$, and the per-period gain from exploiting a serial correlation $`\rho`$, which requires a strategy that adapts to realized price moves, is roughly $`\rho^2\sigma^2\tau^2/(4\eta)`$, independent of portfolio size and hence negligible for large blocks relative to impact costs, which grow as $`X^2`$. **Multi-security portfolio.** With $`m`$ securities, covariance matrix $`C=\sigma\sigma^\top`$, and permanent and temporary impact matrices $`\Gamma`$ and $`H`$, if $`\Gamma`$ and $`H`$ are diagonal the expected cost decomposes into a sum over each security, but the covariance $`C`$ continues to couple the entire system through the risk term: the problem then diagonalizes in the eigenvector basis of $`\tilde H^{-1/2} C\, \tilde H^{-1/2}`$ — with the solution a combination of exponentials in the eigenvalues — not per individual security, and separation into $`m`$ independent liquidations also requires $`C`$ to be diagonal.

## Optimal Control of Execution Costs: Bertsimas-Lo

[BertsimasLo1998] address the same problem as stochastic optimal control, minimizing only the expected execution cost via dynamic programming, and obtain the optimal strategy in recursive form, even in the presence of information variables and multiple securities.

**Dynamic formulation.** One must acquire $`\bar S`$ units over $`T`$ periods minimizing $`\mathbb{E}\big[\sum_{t=1}^T P_t S_t\big]`$ subject to the constraint $`\sum_t S_t=\bar S`$. The state is $`(P_{t-1},W_t)`$ with $`W_t`$ the remaining units, $`W_1=\bar S`$, $`W_{t+1}=W_t-S_t`$; the control is $`S_t`$. With linear and permanent impact

$$P_t=P_{t-1}+\theta S_t+\epsilon_t,\qquad \theta>0,\quad \mathbb{E}[\epsilon_t\mid P_{t-1}]=0,$$

the value function satisfies the Bellman equation

$$V_t(P_{t-1},W_t)=\min_{S_t}\;\mathbb{E}_t\big[P_tS_t+V_{t+1}(P_t,W_{t+1})\big].$$

At the last period one is forced to buy the residual, $`S_T^\ast=W_T`$ and $`V_T=(P_{T-1}+\theta W_T)W_T`$. Solving backward, the first-order condition at each step gives $`S_{T-k}^\ast=W_{T-k}/(k+1)`$, from which, propagating forward with $`W_1=\bar S`$,

$$S_t^\ast=\frac{\bar S}{T},\qquad t=1,\dots,T .$$

The optimal strategy is thus to split the block into equal parts; the expected cost is $`V_1=P_0\bar S+\tfrac12\theta\bar S^2\big(1+\tfrac1T\big)`$, with the impact term quadratic in the total size.

**Information and state-dependent strategy.** Introducing an information variable following an $`\mathrm{AR}(1)`$,

$$P_t=P_{t-1}+\theta S_t+\gamma X_t+\epsilon_t,\qquad X_t=\rho X_{t-1}+\eta_t,$$

the optimal strategy ceases to be naive and becomes linear in the state,

$$S_{T-k}^\ast=\delta_{w,k}\,W_{T-k}+\delta_{x,k}\,X_{T-k},$$

with coefficients computed recursively. Trading anticipates the information: with $`\rho>0`$ (persistence) it is worth buying more when $`X_t`$ signals rising prices; with $`\rho<0`$ (reversion) it is worth delaying. A *linear-percentage temporary* variant separates an impact-free price $`\bar P_t=\bar P_{t-1}\exp(Z_t)`$ from a temporary-impact component $`\Delta_t=(\theta S_t+\gamma X_t)\bar P_t`$, which keeps the impact-free price positive and the impact proportional to it; the execution price $`\bar P_t(1+\theta S_t+\gamma X_t)`$ is non-negative only under mild restrictions on $`\Delta_t`$.

**Multi-asset extension with cross-impact.** With $`n`$ securities, the vector dynamics

$$\mathbf P_t=\mathbf P_{t-1}+\mathbf A\,\mathbf S_t+\mathbf B\,\mathbf X_t+\boldsymbol\epsilon_t,\qquad \mathbf X_t=\mathbf C\,\mathbf X_{t-1}+\boldsymbol\eta_t,$$

introduces into the (positive definite) matrix $`\mathbf A`$ the off-diagonal terms $`A_{ij}`$, which quantify the *cross-impact*: trading security $`j`$ moves the price of security $`i`$. The optimal strategy is again linear in the state,

$$\mathbf S_{T-k}^\ast=\Big(\mathbf I-\tfrac12\mathbf A_{k-1}^{-1}\mathbf A^\top\Big)\mathbf W_{T-k}+\tfrac12\mathbf A_{k-1}^{-1}\mathbf B_{k-1}^\top\mathbf C\,\mathbf X_{T-k},$$

with matrices $`\mathbf A_k,\mathbf B_k,\mathbf C_k`$ defined by Riccati-type recursions. If $`\mathbf A`$ is diagonal the problem decomposes into $`n`$ independent executions [BertsimasLo1998]. A symmetric cross-impact matrix ($`A_{ij}=A_{ji}`$) without information variables still leaves the equal-slice policy $`\bar{\mathbf S}/T`$ optimal; the order on each security depends on the residuals of all the others only when $`\mathbf A`$ is asymmetric. This portfolio dimension reconnects to the cost attribution of [Chapter 23: Risk Attribution, Budgets, and Limits](23_risk_attribution_budget_and_limits.md).

## Transient Impact and Order Book Resilience: Obizhaeva-Wang

The preceding models collapse the market's reaction into just two components — a permanent one and an instantaneous one. [ObizhaevaWang2013] show that this dichotomy is incomplete: what governs the optimal strategy is the *dynamics* with which supply and demand replenish after a trade, that is, the order book's *resilience*, not its static properties (spread and depth).

**Order book and transient impact.** The limit order book is described by an order density $`q`$ per unit of price above the ask price $`A_t`$; buying $`x`$ units consumes the book and shifts the ask by $`x/q`$. After the trade, new orders flow in and the book replenishes: the deviation of the ask from its steady state decays exponentially. With multiple discrete trades the ask is

$$A_t=V_t+\frac{s}{2}+\sum_{t_i<t} x_i\,\kappa\,e^{-\rho(t-t_i)},\qquad V_t=F_t+\lambda\!\!\sum_{t_i\le t}\! x_i,$$

where $`F_t`$ is the fundamental value, $`V_t`$ the mid-quote shifted by the permanent impact $`\lambda`$, $`s`$ the spread, $`\kappa=1/q-\lambda`$ the amplitude of the transient impact, and $`\rho\ge0`$ the *resilience rate*. Every past trade leaves an exponential tail that adds up linearly: $`\rho=0`$ makes the impact permanent, $`\rho\to\infty`$ makes it vanish instantaneously, with recovery half-life $`\ln 2/\rho`$.

**Optimal strategy: discrete blocks and continuous flow.** Minimizing the expected execution cost of $`X_0`$ units over $`[0,T]`$ under this dynamic, and passing to the continuous limit, yields a strategy radically different from uniform splitting:

$$x_0=x_T=\frac{X_0}{\rho T+2},\qquad \dot X_t=\frac{\rho\,X_0}{\rho T+2}\quad (0<t<T).$$

That is, one trades with an *initial discrete block*, a *continuous flow at constant rate* in the middle, and a *final discrete block of the same size* as the first. The initial block pushes the book away from steady state, attracting new limit orders; the continuous flow collects them as they arrive, keeping the transient impact in equilibrium; the final block closes the remaining position. The fraction executed continuously is $`\rho T/(\rho T+2)`$.

**Limiting cases and comparison.** As $`\rho\to0`$ (no resilience) the optimal strategy converges to two equal blocks, $`x_0=x_T=X_0/2`$, with no intermediate trading; as $`\rho\to\infty`$ (instantaneous replenishment) the blocks vanish and it converges to the uniform execution of Bertsimas-Lo (and of Almgren-Chriss for a risk-neutral trader). At the limit points themselves the expected cost does not depend on the strategy, so these schedules are optimal but not uniquely so. Relative to the uniform strategy, exploiting resilience produces cost savings that are largest for intermediate values of $`\rho T`$ and, in their calibration ($`q=5000`$, $`X_0=10^5`$), can reach the order of 10% when the permanent impact $`\lambda`$ is small [ObizhaevaWang2013]. The lesson is that the decisive variable of execution is dynamic — how quickly the market digests the trade — not static.

## The Empirical Impact Law: Bouchaud-Farmer-Lillo

The empirical regularities of market impact, synthesized by [BouchaudFarmerLillo2009], constrain and partly contradict linear models. Three stylized facts emerge from high-frequency data.

**Concave impact and the square-root law.** The impact of a single trade is *concave* in volume: the price change conditional on a volume $`v`$ scales as $`\mathbb{E}[r\mid v]\propto v^{\psi}`$ with a small exponent $`\psi`$, on the order of $`0{.}2`$–$`0{.}5`$. More notable is the impact of a *metaorder* — a large order executed in many pieces: its overall price change approximately follows a *square-root law* in the traded volume, fitted in the BARRA market-impact model of Torre and Ferrari [TorreFerrari1997] and documented for metaorders by [Toth2011],

$$\frac{\Delta p}{p}\;\simeq\;Y\,\sigma_D\sqrt{\frac{Q}{V_D}},$$

where $`Q`$ is the size of the metaorder, $`V_D`$ the daily traded volume, $`\sigma_D`$ the daily volatility and $`Y`$ a constant of order one (the propagator model of [BouchaudFarmerLillo2009], with order-flow memory exponent $`\gamma\approx 0{,}5`$, predicts that for a metaorder of $`N`$ trades at a fixed participation rate the impact grows as $`N^{1-\beta}`$ with $`\beta=(1-\gamma)/2\approx 0{,}25`$, i.e., about $`N^{3/4}`$, of the same concave spirit but not identical to the square root). Concavity alone does not make splitting pay: with permanent concave impact, $`n`$ pieces would move the price by $`n(Q/n)^{\psi}>Q^{\psi}`$. The gain from spreading a large order over time comes from the transient part of impact, which decays as the book replenishes, while under the square-root law the total cost of a metaorder grows as $`Q\sqrt{Q}=Q^{3/2}`$; the dynamics of impact are the empirical foundation of *scheduling*.

**Long memory of order flow.** Order signs $`\epsilon_t\in\{-1,+1\}`$ are strongly autocorrelated, with autocorrelation decaying as a power law,

$$C(\tau)=\mathbb{E}[\epsilon_t\epsilon_{t+\tau}]\sim \tau^{-\gamma},\qquad \gamma\in(0,1),\;\;\gamma\approx0{.}5,$$

a *long-memory* process (Hurst $`H=1-\gamma/2>\tfrac12`$) that reflects the practice of splitting large orders: if metaorder sizes $`L`$ have a Pareto tail $`P(L>x)\sim x^{-\alpha}`$ with $`1<\alpha<2`$, splitting generates sign autocorrelations decaying as $`\tau^{-(\alpha-1)}`$, that is, $`\gamma=\alpha-1`$ (the mechanism proposed by Lillo, Mike and Farmer in 2005, reviewed in [BouchaudFarmerLillo2009]); the long memory itself is documented in [LilloFarmer2004].

**The paradox and the propagator model.** Order flow is thus strongly predictable, yet prices remain nearly diffusive (a martingale): an apparent contradiction with efficiency. One resolution is that the impact of each order is *transient*, not permanent, and its decay exactly offsets the persistence of the flow; a mathematically equivalent one is that liquidity responds asymmetrically to predictable orders, shrinking their permanent impact [LilloFarmer2004]. The *propagator model* writes the price as a sum of past impacts that decay,

$$p_t=\sum_{t'<t}G(t-t')\,\epsilon_{t'}+\text{noise},\qquad G(\ell)\sim \ell^{-\beta},\;\;\beta=\frac{1-\gamma}{2},$$

where $`G`$ is the propagator function. The critical exponent $`\beta=(1-\gamma)/2`$ is precisely the one that balances the long memory of the signs with the decay of impact, making the price variance diffusive and returns nearly unpredictable despite the predictability of order flow [BouchaudFarmerLillo2009]. Obizhaeva-Wang's transient, decaying impact finds here its empirical counterpart, with the difference that the empirical propagator decays as a power law rather than exponentially.

## Non-Linear Power-Law Impact in Scheduling

The empirical concavity of impact motivates the generalization of the scheduling problem to a power-law impact function with arbitrary exponent, developed by [MarzoRitelliZagaglia2011] reworking the continuous-time formulation of [Almgren2003].

**The cost functional.** Letting $`x(t)`$ be the residual position, with $`x(0)=X`$, $`x(T)=0`$, and $`\dot x`$ the trading velocity, one minimizes the continuous-time implementation cost, the sum of the power-law temporary impact cost and the volatility risk on the open position,

$$\min_{x(\cdot)}\int_0^T\Big[\eta\,|\dot x|^{\,k+1}+\lambda\,\sigma^2 x^2\Big]\,dt,$$

where $`k>0`$ is the power-law exponent (the case $`k=1`$ reproduces the quadratic impact of Almgren-Chriss) and the permanent impact contributes only a constant term $`\tfrac12\gamma X^2`$, independent of the trajectory. Since the integrand does not depend explicitly on time, Beltrami's identity holds, $`L-\dot x\,\partial L/\partial\dot x=\text{const}`$, which provides a first integral:

$$\lambda\sigma^2 x^2-k\,\eta\,(-\dot x)^{\,k+1}=-k\,\eta\,v_0^{\,k+1},\qquad v_0=-\dot x(T),$$

having evaluated the constant at the terminal point $`x(T)=0`$.

**Solution.** Separating variables, the first integral gives

$$(-\dot x)=\Big(v_0^{\,k+1}+\frac{\lambda\sigma^2}{k\eta}\,x^2\Big)^{\!1/(k+1)},$$

whose integration leads to an implicit relation between $`x`$ and $`t`$ expressed by the Gaussian hypergeometric function,

$$\frac{1}{v_0}\Big[X\,{}_2F_1\!\Big(\tfrac12,\tfrac{1}{k+1};\tfrac32;-\tfrac{\lambda\sigma^2X^2}{k\eta v_0^{k+1}}\Big)-x\,{}_2F_1\!\Big(\tfrac12,\tfrac{1}{k+1};\tfrac32;-\tfrac{\lambda\sigma^2x^2}{k\eta v_0^{k+1}}\Big)\Big]=t.$$

The function $`x\mapsto x\,{}_2F_1(\cdots)`$ is strictly increasing (its derivative is $`\big(1+\tfrac{\lambda\sigma^2x^2}{k\eta v_0^{k+1}}\big)^{-1/(k+1)}>0`$), so $`t`$ is strictly decreasing in $`x`$ and the relation is invertible to give $`x(t)`$. The Legendre condition $`\partial^2L/\partial\dot x^2=\eta k(k+1)(-\dot x)^{k-1}>0`$, which holds wherever $`\dot x<0`$, is necessary for a minimum but not sufficient; the extremal is the global minimizer because the integrand $`\eta|\dot x|^{k+1}+\lambda\sigma^2x^2`$ is jointly convex in $`(x,\dot x)`$.

**Linear case and terminal velocity.** For $`k=1`$ the Euler-Lagrange equation reduces to $`\ddot x=(\lambda\sigma^2/\eta)\,x`$ and, via the identity $`{}_2F_1(\tfrac12,\tfrac12;\tfrac32;-z^2)=\operatorname{arsinh}(z)/z`$, the hypergeometric solution collapses into the hyperbolic trajectory of Almgren-Chriss,

$$x(t)=X\,\frac{\sinh\!\big(\sqrt{\lambda\sigma^2/\eta}\,(T-t)\big)}{\sinh\!\big(\sqrt{\lambda\sigma^2/\eta}\,T\big)}.$$

A distinctive result concerns the boundary velocity. Imposing zero terminal velocity ($`v_0=-\dot x(T)=0`$, so that trading fades out at $`t=T`$: the zero-speed case that [MarzoRitelliZagaglia2011] attribute to Almgren) is compatible with the constraints only if $`k>1`$, because only then does $`-\dot x\propto x^{2/(k+1)}`$ bring the position to zero in finite time; in that case the block that is liquidated exactly over $`[0,T]`$ with zero final velocity is not free but tied to the horizon by

$$x(0)=\Big(\frac{(k-1)T}{k+1}\Big)^{\!\frac{k+1}{k-1}}\Big(\frac{\lambda\sigma^2}{k\eta}\Big)^{\!\frac{1}{k-1}}.$$

A smaller block $`X`$ is optimally liquidated early, at $`T^\ast=T\,\big(X/x(0)\big)^{(k-1)/(k+1)}<T`$ with $`x(0)`$ the value above, and a larger one ends with positive velocity. For $`k\le1`$, on the other hand, no zero-velocity solution exists: the trader is still trading at $`t=T`$, with velocity $`v_0>0`$. Whenever $`v_0>0`$ it is not a free parameter: setting $`x=0`$, $`t=T`$ in the hypergeometric relation fixes it through $`X\,{}_2F_1\big(\tfrac12,\tfrac{1}{k+1};\tfrac32;-\tfrac{\lambda\sigma^2X^2}{k\eta v_0^{k+1}}\big)=v_0T`$ [MarzoRitelliZagaglia2011]. The impact exponent $`k`$ thus determines not only the shape of the trajectory, but the very nature of the optimum at the extremes of the horizon.

## Synthesis: Executing Turnover at Minimum Cost

The models in this chapter form a coherent hierarchy. [Kyle1985] provides the micro-foundation: the permanent linear impact $`\lambda`$ arises from the market maker's inference about order flow, and measures illiquidity and depth. On this basis, [AlmgrenChriss2000] and [BertsimasLo1998] build the optimal execution trajectory: the former via a variational approach, as a trade-off between impact cost (trading fast) and timing risk (trading slowly), with the mean-variance efficient frontier of trading profiles and the hyperbolic solution; the latter via dynamic programming, minimizing expected cost alone (a risk-neutral trader), with the $`\bar S/T`$ strategy and its extensions to information and cross-impact. [ObizhaevaWang2013] enrich impact with order book dynamics, replacing the permanent/temporary dichotomy with a transient impact that decays at the resilience rate $`\rho`$, from which the mixed discrete-block-and-continuous-flow strategy follows. [BouchaudFarmerLillo2009] document the empirical regularities — concave impact, long memory of order flow, the propagator model — that constrain these models, together with the square-root law of metaorder impact [Toth2011], and [MarzoRitelliZagaglia2011] with [Almgren2003] close the circle by incorporating non-linear power-law impact into optimal scheduling.

The thread that unites them is the link with portfolio construction. The signal generates, through its decay and horizon, a turnover (see [Chapter 11: Signal Decay, Horizon, and Turnover](11_signal_decay_horizon_and_turnover.md)); this chapter establishes how to execute that turnover. Implementation shortfall is the cost that erodes the portfolio's theoretical alpha: trading fast pays for it in impact, trading slowly pays for it in price risk. The efficient trading frontier, the block-and-flow strategy, and non-linear scheduling are so many answers to the same problem — turning target weights into executed orders while minimizing the distance between the paper portfolio and the realized one.

## References

- **[Almgren2003]** Almgren, R. (2003). Optimal Execution with Nonlinear Impact Functions and Trading-Enhanced Risk. Applied Mathematical Finance, 10(1), 1-18.
- **[AlmgrenChriss2000]** Almgren, R. & Chriss, N. (2000). Optimal Execution of Portfolio Transactions. Journal of Risk, 3(2), 5-39.
- **[BertsimasLo1998]** Bertsimas, D. & Lo, A. W. (1998). Optimal Control of Execution Costs. Journal of Financial Markets, 1(1), 1-50.
- **[BouchaudFarmerLillo2009]** Bouchaud, J.-P., Farmer, J. D. & Lillo, F. (2009). How Markets Slowly Digest Changes in Supply and Demand. In T. Hens & K. R. Schenk-Hoppé (Eds.), Handbook of Financial Markets: Dynamics and Evolution (pp. 57-160). North-Holland/Elsevier. (Preprint version: arXiv:0809.0822, 2008.)
- **[Kyle1985]** Kyle, A. S. (1985). Continuous Auctions and Insider Trading. Econometrica, 53(6), 1315-1335.
- **[LilloFarmer2004]** Lillo, F. & Farmer, J. D. (2004). The Long Memory of the Efficient Market. Studies in Nonlinear Dynamics & Econometrics, 8(3), Article 1.
- **[Markowitz1952]** Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.
- **[MarzoRitelliZagaglia2011]** Marzo, M., Ritelli, D. & Zagaglia, P. (2011). Optimal Trading Execution with Nonlinear Market Impact: An Alternative Solution Method. Working paper, arXiv:1111.6826 (SSRN 1964660).
- **[ObizhaevaWang2013]** Obizhaeva, A. A. & Wang, J. (2013). Optimal Trading Strategy and Supply/Demand Dynamics. Journal of Financial Markets, 16(1), 1-32. (Working paper version: NBER Working Paper 11444, 2005.)
- **[Perold1988]** Perold, A. F. (1988). The Implementation Shortfall: Paper versus Reality. Journal of Portfolio Management, 14(3), 4-9.
- **[TorreFerrari1997]** Torre, N. & Ferrari, M. (1997). Market Impact Model Handbook. BARRA Inc., Berkeley, CA.
- **[Toth2011]** Tóth, B., Lempérière, Y., Deremble, C., de Lataillade, J., Kockelkoren, J. & Bouchaud, J.-P. (2011). Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets. Physical Review X, 1(2), 021006.
