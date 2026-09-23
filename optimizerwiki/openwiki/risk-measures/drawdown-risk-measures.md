---
type: concept
title: "Drawdown-Based Risk Measures"
description: A family of risk measures defined on the wealth trajectory rather than its marginal distribution — the drawdown as a fall from the running maximum, and the measures derived from it (maximum drawdown, average drawdown, and Conditional Drawdown-at-Risk), with the CDaR as the transposition of Conditional Value-at-Risk to the drawdown distribution on a sample path, its convexity and interpolation between average and maximum drawdown, and the reduction of drawdown-constrained portfolio optimization to a linear program.
tags: [drawdown, maximum-drawdown, conditional-drawdown-at-risk, path-dependent, running-maximum, linear-programming, convexity, managed-accounts]
sources:
  - id: openwiki-source-a886f46fd9241311d5326cd8
    resource: repo://docs/20_drawdown_risk_measures.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Drawdown-Based Risk Measures

This chapter introduces a family of risk measures defined on the *trajectory* of wealth rather
than its marginal distribution: the drawdown as a fall from the running maximum, and the measures
derived from it. The CDaR is the transposition to the drawdown path of the Conditional
Value-at-Risk defined in [coherent risk measures](./coherent-risk-measures.md) and optimized in
[CVaR optimization](../optimization/cvar-optimization.md), sharing the same auxiliary-variable
linearization.

## From marginal risk to trajectory risk

The coherent measures built on the marginal return distribution — VaR, expected shortfall and
their derivatives in [coherent risk measures](./coherent-risk-measures.md) — summarize risk at a
single observation instant, ignoring the order in which losses occur over time. But in some
contexts the temporal sequence of outcomes determines the relevant risk. Chekhlov, Uryasev and
Zabarankin start from the managed-account manager, for whom clients are the sole income source
through management and incentive fees: losing a client ends the business, and the client's
decision to close the account likely depends on the magnitude and duration of the account's
drawdown. Operational thresholds make this concrete: a Commodity Trading Advisor rarely keeps a
client whose account has been in drawdown — even small — for over two years; a client is unlikely
to tolerate a 50% drawdown at a medium- or low-risk CTA; and a proprietary-system trader can be
suspended when a maximum-drawdown condition (typically ~20% of the collateral capital) is
breached, with a warning at a lower level (~15%).

The problem differs from earlier work. Grossman-Zhou obtain an exact analytic solution to a
one-dimensional maximum-drawdown problem under log-normal equity assumptions via dynamic
programming, and Cvitanić-Karatzas generalize to the multidimensional case; those seek
time-dependent allocation strategies under return-distribution assumptions. Here, instead, risk
is defined and *constant-in-time* optimal weights are found on a single sample path of portfolio
returns (historical or most-likely future), with no assumption on the underlying probability
distribution — risk is a function of the sample path, not a measure over a set of paths, an
approach akin to index replication with constant weights.

## The drawdown as a fall from the running maximum

Let $w(x,t)$ be the uncompounded portfolio return at time $t$, with $x=(x_1,\dots,x_m)$ the
weights of $m$ instruments. A drawdown is the fall in portfolio value relative to its past
maximum, so the drawdown function is
$$ D(x,t) = \max_{0\le\tau\le t}\{w(x,\tau)\} - w(x,t). \tag{1} $$
Two properties define its nature: $D(x,t)$ is non-negative by construction and vanishes exactly
when the portfolio hits a new maximum; and the running maximum $\max_{0\le\tau\le t}\{w(x,\tau)\}$
depends on the entire trajectory preceding $t$, making the drawdown **path-dependent** — it
cannot be reconstructed from the marginal return distribution alone. It can be expressed in
absolute or relative terms: a current value of 9 million against a past peak of 10 million is a
drawdown of 1 million absolute or 10% relative. On this trajectory function three risk functions
are built on a sample path: the maximum drawdown (MaxDD), the average drawdown (AvDD), and the
Conditional Drawdown-at-Risk (CDaR) — a family parametrized by $\alpha$ containing the other two
as limit cases.

## Maximum and average drawdown

The maximum drawdown over $[0,T]$ maximizes the drawdown function,
$M(x)=\max_{0\le t\le T}\{D(x,t)\}$, concentrating on a single event — the largest loss relative
to the previous high. The average drawdown is the time average along the path,
$A(x)=\tfrac1T\int_0^T D(x,t)\,dt$. They differ in sensitivity: MaxDD focuses on one episode,
while the CDaR — of which MaxDD and AvDD are limit cases — accounts for both the magnitude and
the duration of drawdowns.

## The Conditional Drawdown-at-Risk

The CDaR modifies the Conditional Value-at-Risk to the case where the loss function is defined by
drawdowns at discrete times with equal weights instead of scenario probabilities. Recall that the
$\alpha$-VaR is the minimal $\zeta_\alpha$ such that the probability the loss does not exceed it
is at least $\alpha$, and the $\alpha$-CVaR is the expectation on the $\alpha$-tail of losses,
coinciding for continuous distributions with the conditional expectation of losses above the VaR.
With $N$ sub-periods of $[0,T]$ and confidence $\alpha\in[0,1]$, the $\alpha$-CDaR is the average
of the worst $(1-\alpha)\cdot 100\%$ drawdowns on the path — e.g. the 95%-CDaR is the average of
the worst 5% of drawdowns. When $(1-\alpha)N$ is an integer, with $\zeta_\alpha(x)$ a threshold
exactly $(1-\alpha)\cdot 100\%$ of drawdowns exceed,
$\Delta_\alpha(x)=\tfrac1{(1-\alpha)T}\int_{\Omega_\alpha}D(x,t)\,dt$ over
$\Omega_\alpha=\{t:D(x,t)\ge\zeta_\alpha(x)\}$. In the general (non-integer) case the CDaR takes
the optimization form
$$ \Delta_\alpha(x) = \min_\zeta\left\{\zeta + \frac{1}{(1-\alpha)T}\int_0^T [D(x,t)-\zeta]^+\,dt\right\}, \tag{4} $$
with $[g]^+=\max\{0,g\}$; the optimal $\zeta$ equals $\zeta_\alpha(x)$ when $(1-\alpha)N$ is
integer and is otherwise $\ge\zeta_\alpha(x)$. The family interpolates between the two elementary
measures: as $\alpha\to 1$, $\Delta_1(x)=M(x)$ (maximum drawdown), and at $\alpha=0$,
$\Delta_0(x)=A(x)$ (average drawdown) — as confidence grows, the average concentrates on an ever
narrower fraction of the worst drawdowns, moving continuously from the whole underwater profile
to the single extreme event.

## Convexity and homogeneity

The uncompounded cumulative portfolio return is linear in the weights: with $y(t)$ the vector of
instruments' cumulative returns, $w(x,t)=y(t)\cdot x=\sum_i y_i(t)x_i$. From this linearity follow
the structural properties. The running maximum is a pointwise maximum of linear functions of $x$,
hence convex in $x$; since $D(x,t)$ is the difference between this convex function and the linear
$y(t)\cdot x$, the drawdown function is convex in $x$ for every $t$. Therefore MaxDD (a max over
$t$ of convex functions), AvDD (a non-negatively weighted integral of convex functions), and the
CDaR of (4) (a min over $\zeta$ of a function jointly convex in $(x,\zeta)$) are all convex,
making the return-CDaR problem a convex optimization with linear objective and piecewise-linear
convex constraint — tractable with the tools of
[convex optimization](../optimization/convex-optimization.md). Linearity also gives positive
homogeneity: $w(\lambda x,t)=\lambda w(x,t)$ gives $D(\lambda x,t)=\lambda D(x,t)$, so $M$, $A$
and $\Delta_\alpha$ are positively homogeneous of degree one. The CDaR is conceptually akin, as a
percentile function, to the CVaR (also known for continuous distributions as Mean Excess
Loss/Mean Shortfall and Tail Value-at-Risk).

## Discrete formulation and reduction to a linear program

For a sample path with instrument returns $(r_j(1),\dots,r_j(N))$, the cumulative return is
$y_j(t)=\sum_{k=1}^t r_j(k)$, and the discrete drawdown is
$D_k(x)=\max_{1\le j\le k}\{y_j\cdot x\}-y_k\cdot x$. The mean annualized return is
$R(x)=\tfrac1{dC}y_N\cdot x$ ($d$ years, $C$ available capital), and box "technological"
constraints $X=\{x:x_{\min}\le x_i\le x_{\max}\}$ bound the weights. The problem maximizes $R(x)$
under risk-function constraints; the key step, following the CVaR approach, introduces auxiliary
variables $u_k$ tracking the running maximum. The maximum-drawdown-constrained problem reduces to
the linear program
$$ \begin{aligned}\max_{x,u}\ & \tfrac1{dC}y_N\cdot x \\ \text{s.t.}\ & u_k - y_k\cdot x \le \nu_1 C,\quad u_k\ge y_k\cdot x,\quad u_k\ge u_{k-1},\quad u_0=0,\quad x_{\min}\le x_i\le x_{\max},\end{aligned} \tag{16} $$
where the constraints $u_k\ge y_k\cdot x$ and monotonicity $u_k\ge u_{k-1}$ force $u_k$ to the
running maximum, so $u_k-y_k\cdot x$ reconstructs $D_k(x)$. The average-drawdown problem replaces
the pointwise constraint with the arithmetic mean $\tfrac1N\sum_k(u_k-y_k\cdot x)\le\nu_2 C$
(17). The CDaR-constrained problem reduces, following the CVaR approach, to
$$ \begin{aligned}\max_{x,\zeta,u,z}\ & \tfrac1{dC}y_N\cdot x \\ \text{s.t.}\ & \zeta + \tfrac1{(1-\alpha)N}\sum_{k=1}^N z_k \le \nu_3 C,\quad z_k\ge u_k-y_k\cdot x-\zeta,\quad z_k\ge 0,\\ & u_k\ge y_k\cdot x,\quad u_k\ge u_{k-1},\quad u_0=0,\quad x_{\min}\le x_i\le x_{\max},\end{aligned} \tag{18} $$
where the $z_k$ linearize the positive part $[D_k(x)-\zeta]^+$ of (4). A key feature of (18) is
that it does not involve the threshold function $\zeta_\alpha(x)$: at the optimum, $x$ and $\zeta$
give an optimal portfolio and the corresponding threshold value. The constants $\nu_1,\nu_2,\nu_3
\in[0,1]$ define the portions of capital "one is willing to lose," and multiple constraints can be
combined; the LP reduction solves problems with many thousands of instruments.

## Drawdown constraints and objectives in portfolio construction

Symmetrically to Markowitz's mean-variance approach, the three measures give constrained
formulations maximizing return subject to $M(x)\le\nu_1 C$, $A(x)\le\nu_2 C$ or
$\Delta_\alpha(x)\le\nu_3 C$; dually, they serve as objectives through reward/risk ratios
$\mathrm{MaxDDRatio}=R(x)/M(x)$, $\mathrm{AvDDRatio}=R(x)/A(x)$,
$\mathrm{CDaRRatio}=R(x)/\Delta_\alpha(x)$, the maximum-ratio portfolio being the tangency of the
line through the origin with the efficient frontier. The numerical evidence uses equity curves
from a technical trading system on futures across $m=32$ markets (1/1/1988–1/9/1999, $20M
collateral, uncompounded curves); with $x_{\min}=0.2$, $x_{\max}=0.8$ (analogous to the
fully-invested condition, bounding leverage and making the frontier concave), the LPs (16), (17),
(18) were solved with CPLEX and independently verified via a genetic algorithm yielding identical
weights. The comparison has a practical reading: in the reward-MaxDD plane the frontier is
efficient only for $(1-\alpha)=0$ and in the reward-AvDD plane only for $(1-\alpha)=1$, each
measure being optimal in its own sense; the 5% CDaR constraint ($(1-\alpha)=0.05$) yields weights
significantly different from MaxDD, involving dozens of events in the average, producing a
portfolio more robust than both MaxDD and AvDD. Because MaxDD rests on a single observation of the
maximum loss, its solutions can carry substantial statistical error; the CDaR family's statistical
averaging of drawdowns gives better future-risk prediction and more stable weight allocation at an
appropriate confidence level (e.g. $\alpha=0.95$).
