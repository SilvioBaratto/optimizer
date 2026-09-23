---
title: "Drawdown-Based Risk Measures"
chapter: 20
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-11
---

> [!abstract] Summary
> The chapter introduces a family of risk measures defined on the wealth trajectory, rather than on its marginal distribution: drawdown as the fall from the current maximum, and the measures derived from it — maximum drawdown, average drawdown, and Conditional Drawdown-at-Risk (CDaR). It establishes that CDaR is the transposition of Conditional Value-at-Risk to the distribution of drawdowns on a sample path, that it is convex and interpolates between average drawdown and maximum drawdown as the confidence level varies, and that portfolio optimization with drawdown constraints reduces to a linear program via auxiliary variables that track the running maximum.

## From Marginal Risk to Path Risk

Coherent risk measures built on the marginal distribution of returns — Value-at-Risk, expected shortfall, and their derivatives discussed in [[04 Misure di rischio coerenti]] — summarize risk at a single observation instant, ignoring the order in which losses occur over time. There are, however, contexts in which it is precisely the temporal sequence of outcomes that determines the relevant risk. [ChekhlovUryasevZabarankin2003] starts from the perspective of a managed-accounts manager, for whom clients are the sole source of income through management and incentive fees: losing a client is equivalent to the end of the business, and the client's decision to close the account depends, in all likelihood, on the magnitude and duration of their account's drawdown.

The operational thresholds cited make this concern concrete. It is very rare for a Commodity Trading Advisor (CTA) to keep a client whose account has remained in drawdown, even a small one, for more than two years; it is unlikely that a client would tolerate a 50% drawdown with a medium- or low-risk CTA; in an investment banking context, a proprietary system trader is required to produce profits within a year at the latest, may be suspended when a maximum drawdown condition is violated — typically around 20% of the collateral capital — and receives a warning at a lower level (around 15%) beyond which their trading is reviewed. These constraints make account management practitioners very attentive to both the magnitude and the duration of clients' account drawdowns.

The problem addressed differs from that of previous works. [GrossmanZhou1993] obtains, under log-normality assumptions on equity statistics and with dynamic programming, an exact analytical solution to a maximum drawdown problem in the one-dimensional case; [CvitanicKaratzas1995] generalizes this result to the multi-dimensional case. In those works, time-dependent allocation strategies are sought under assumptions on the distribution of returns; here, instead, risk is defined and time-constant optimal weights are found on a single sample path of portfolio returns (historical or most likely future path), without any assumption on the underlying probability distribution. Risk is thus a function of the sample path, not a risk measure over a set of paths. The approach is akin to the index-tracking problem ([DemboKing1992]), in which the historical performance of an index is replicated by a constant-weight portfolio.

## Drawdown as a Fall from the Current Maximum

Let $w(x,t)$ denote the uncompounded return of the portfolio at time $t$, where the components of the vector $x = (x_1, x_2, \ldots, x_m)$ are the weights of $m$ instruments in the portfolio. By definition, a drawdown is the fall in the portfolio's value relative to the maximum reached in the past. The drawdown function at time $t$ is the difference between the maximum of the function $w(x,\tau)$ over the history preceding $t$ and the value of the function itself at $t$:
$$D(x,t) = \max_{0 \le \tau \le t} \{ w(x,\tau) \} - w(x,t) . \tag{1}$$

Two properties define the nature of this measure. First, $D(x,t)$ is by construction non-negative and vanishes exactly when the portfolio touches a new maximum. Second — and this is the essential point — the quantity $\max_{0 \le \tau \le t} \{ w(x,\tau) \}$, the running maximum of wealth, depends on the entire portion of the trajectory preceding time $t$: drawdown is thus a path-dependent quantity, which cannot be reconstructed from the marginal distribution of returns alone. Drawdown can be expressed in absolute or relative (percentage) terms: if the portfolio's current value is 9 million and the past maximum value was 10 million, the drawdown is 1 million in absolute terms and 10% in relative terms.

On this trajectory function [ChekhlovUryasevZabarankin2003] builds three risk functions defined on a sample path of portfolio returns: maximum drawdown (MaxDD), average drawdown (AvDD), and Conditional Drawdown-at-Risk (CDaR). CDaR is actually a family of risk functions parameterized by $\alpha$, which contains the other two as limiting cases.

## Maximum Drawdown and Average Drawdown

The maximum drawdown over the interval $[0,T]$ is obtained by maximizing the drawdown function:
$$M(x) = \max_{0 \le t \le T} \{ D(x,t) \} . \tag{2}$$
It concentrates attention on a single event: the portfolio's maximum loss relative to its previous highest value.

The average drawdown is the time average of the drawdown function along the path:
$$A(x) = \frac{1}{T} \int_0^T D(x,t)\, dt . \tag{3}$$

The two measures have different sensitivities to events. MaxDD focuses on a single episode — the maximum loss relative to the previous highest value — while CDaR, of which MaxDD and AvDD are the limiting cases, accounts for both the magnitude and the duration of drawdowns.

## Conditional Drawdown-at-Risk

CDaR arises as a modification of Conditional Value-at-Risk to the case where the loss function is defined by drawdowns at discrete time instants with equal weights, instead of by scenario probabilities. It is useful to recall the link with percentile measures. With a probability level $\alpha$, the $\alpha$-VaR of a portfolio is the minimum quantity $\zeta_\alpha$ such that the probability that the loss does not exceed $\zeta_\alpha$ is greater than or equal to $\alpha$ over a specified time $\tau$ ([Jorion1996]), while the $\alpha$-CVaR is the expectation over the $\alpha$-tail of the losses ([RockafellarUryasev2000], [RockafellarUryasev2002]); for continuous distributions, the expectation over the $\alpha$-tail coincides with the conditional expectation of losses above the VaR value $\zeta_\alpha$. The construction of CVaR is taken up in [[16 Ottimizzazione del CVaR - la formulazione di Rockafellar-Uryasev]].

Let $N$ be the number of sub-periods of the interval $[0,T]$ and $\alpha \in [0,1]$ the confidence level. For a given value of the tolerance parameter $\alpha$, the $\alpha$-CDaR is defined as the average of the worst $(1-\alpha)\cdot 100\%$ drawdowns experienced on the path over the period considered. For example, the 0.95-CDaR (or 95%-CDaR) is the average of the worst 5% of the drawdowns over the time interval considered.

In the case where $(1-\alpha)N$ is an integer — that is, when exactly $(1-\alpha)\cdot 100\%$ of the drawdowns can be counted — let $\zeta_\alpha(x)$ be a threshold such that exactly $(1-\alpha)\cdot 100\%$ of the drawdowns exceed it. The CDaR at confidence level $\alpha$ is then the average of that $(1-\alpha)\cdot 100\%$ of the drawdowns:
$$\Delta_\alpha(x) = \frac{1}{(1-\alpha)T} \int_{\Omega_\alpha} D(x,t)\, dt , \qquad \Omega_\alpha = \{ t \in [0,T] : D(x,t) \ge \zeta_\alpha(x) \} .$$

When instead $(1-\alpha)N$ is not an integer — the general case — the CDaR is expressed as a linear combination of $\zeta_\alpha(x)$ and the drawdowns that strictly exceed $\zeta_\alpha(x)$, in a manner analogous to what is done for CVaR ([RockafellarUryasev2002]). The optimization formula presenting the CDaR in the general case is
$$\Delta_\alpha(x) = \min_{\zeta} \left\{ \zeta + \frac{1}{(1-\alpha)T} \int_0^T [\, D(x,t) - \zeta \,]^{+}\, dt \right\}, \tag{4}$$
where $[g]^{+} = \max\{0, g\}$. If $(1-\alpha)N$ is an integer, the optimal value of $\zeta$ in (4) coincides with $\zeta_\alpha(x)$; otherwise the optimal $\zeta$ is greater than or equal to $\zeta_\alpha(x)$.

The family interpolates between the two elementary measures of the previous section. As $\alpha$ tends to $1$, the CDaR tends to the maximum drawdown, that is, $\Delta_1(x) = M(x)$; when $\alpha = 0$, the CDaR coincides with the average drawdown, that is, $\Delta_0(x) = A(x)$. As the confidence level increases, therefore, the average concentrates on a progressively narrower fraction of the worst drawdowns, moving continuously from the average of the entire underwater profile (AvDD) to the single extreme event (MaxDD).

## Convexity and Homogeneity of Drawdown Measures

The portfolio's cumulative uncompounded return is linear in the weights. Letting $y(t) = (y_1(t), y_2(t), \ldots, y_m(t))$ be the vector of cumulative returns of the individual instruments up to time $t$, the portfolio's cumulative return is
$$w(x,t) = y(t) \cdot x = \sum_{i=1}^m y_i(t)\, x_i .$$

From this linearity follow the structural properties of drawdown measures. The running maximum $\max_{0 \le \tau \le t} \{ y(\tau) \cdot x \}$ is a pointwise maximum of linear functions of $x$, hence convex in $x$; since $D(x,t)$ is the difference between this convex function and the linear function $y(t) \cdot x$, the drawdown function is convex in $x$ for every $t$. It follows that the maximum drawdown $M(x)$, being a maximum over $t$ of convex functions, is convex; that the average drawdown $A(x)$, being an integral with non-negative weights of convex functions, is convex; and that the CDaR $\Delta_\alpha(x)$ of (4), being the minimum over $\zeta$ of a function jointly convex in $(x,\zeta)$, is itself convex. Consistently, the return-CDaR optimization problem is a convex optimization problem with a linear objective function and a piece-wise linear convex constraint, in the sense of the definition of convexity of [Rockafellar1970]. It is this convex structure that makes the problem tractable with the tools of [[14 Ottimizzazione convessa - coni, dualità e KKT]].

A further consequence of the linearity of $w(x,t)$ in $x$ is positive homogeneity: since $w(\lambda x, t) = \lambda\, w(x,t)$ for $\lambda > 0$, we have $D(\lambda x, t) = \lambda\, D(x,t)$, and hence $M$, $A$, and $\Delta_\alpha$ are positively homogeneous of degree one in the weights. CDaR is conceptually akin, as a percentile function, to the CVaR of [RockafellarUryasev2000] and [RockafellarUryasev2002]; CVaR is in turn linked to the class of coherent measures of [ArtznerEtAl1999]; for continuous distributions it is also known as Mean Excess Loss or Mean Shortfall ([MausserRosen1999]) and as Tail Value-at-Risk ([ArtznerEtAl1999]).

## Discrete Formulation and Reduction to a Linear Program

Let $j$ be the instrument index, $1 \le j \le m$. For a particular sample path, suppose the returns $(r_j(1), r_j(2), \ldots, r_j(N))$ of the individual instruments are available; the cumulative uncompounded return of the $j$-th instrument up to time $t$ is $y_j(t) = \sum_{k=1}^t r_j(k)$. Denoting by $y_k = y(k)$ the vector of cumulative instrument returns up to time $k$, the discrete drawdown function is
$$D_k(x) = \max_{1 \le j \le k} \{ y_j \cdot x \} - y_k \cdot x , \tag{14}$$
and the average annualized return $R(x)$ over the period $[0,T]$, a linear function of $x$, is given by the inner product
$$R(x) = \frac{1}{dC}\, y_N \cdot x , \tag{15}$$
where $d$ is the number of years in the interval $[0,T]$ and $C$ is the available capital. So-called technological constraints are imposed on the weights via box inequalities
$$X = \{ x : x_{\min} \le x_i \le x_{\max},\ i = 1,\ldots,m \} . \tag{8}$$

The problem consists in maximizing the return $R(x)$ subject to constraints on the various risk functions. The key step, which follows the CVaR approach of [RockafellarUryasev2000] and [RockafellarUryasev2002], is the introduction of auxiliary variables $u_k$ that track the running maximum. The problem with a maximum drawdown constraint,
$$\max_{x \in X}\ \frac{1}{dC}\, y_N \cdot x \quad \text{s.t.}\quad \max_{1 \le k \le N} \Big\{ \max_{1 \le j \le k} \{ y_j \cdot x \} - y_k \cdot x \Big\} \le \nu_1 C ,$$
reduces to the linear program
$$\begin{aligned}
\max_{x, u}\ & \frac{1}{dC}\, y_N \cdot x \\
\text{s.t.}\ & u_k - y_k \cdot x \le \nu_1 C, && 1 \le k \le N, \\
& u_k \ge y_k \cdot x, && 1 \le k \le N, \\
& u_k \ge u_{k-1}, && 1 \le k \le N, \\
& u_0 = 0, \\
& x_{\min} \le x_i \le x_{\max}, && 1 \le i \le m,
\end{aligned} \tag{16}$$
where the $u_k$, $1 \le k \le N$, are the auxiliary variables. The constraints $u_k \ge y_k \cdot x$ and the monotonicity $u_k \ge u_{k-1}$ require that $u_k$ not fall below any $y_j \cdot x$ with $j \le k$; at the optimum $u_k$ achieves the running maximum and $u_k - y_k \cdot x$ reconstructs the drawdown $D_k(x)$.

The problem with an average drawdown constraint is written by replacing the pointwise constraint with the arithmetic mean of the discrete drawdowns:
$$\begin{aligned}
\max_{x, u}\ & \frac{1}{dC}\, y_N \cdot x \\
\text{s.t.}\ & \frac{1}{N} \sum_{k=1}^N ( u_k - y_k \cdot x ) \le \nu_2 C, \\
& u_k \ge y_k \cdot x, \quad u_k \ge u_{k-1}, \quad u_0 = 0, \quad x_{\min} \le x_i \le x_{\max}.
\end{aligned} \tag{17}$$

Following the CVaR approach of [RockafellarUryasev2002], it can be shown that the problem with a CDaR constraint can be formulated and reduced to the linear program
$$\begin{aligned}
\max_{x, \zeta, u, z}\ & \frac{1}{dC}\, y_N \cdot x \\
\text{s.t.}\ & \zeta + \frac{1}{(1-\alpha)N} \sum_{k=1}^N z_k \le \nu_3 C, \\
& z_k \ge u_k - y_k \cdot x - \zeta, && 1 \le k \le N, \\
& z_k \ge 0, && 1 \le k \le N, \\
& u_k \ge y_k \cdot x, && 1 \le k \le N, \\
& u_k \ge u_{k-1}, && 1 \le k \le N, \\
& u_0 = 0, \\
& x_{\min} \le x_i \le x_{\max}, && 1 \le i \le m.
\end{aligned} \tag{18}$$

The variables $z_k$ linearize the positive part $[\,D_k(x) - \zeta\,]^{+}$ of (4). An important feature of formulation (18) is that it does not involve the threshold function $\zeta_\alpha(x)$: at the optimal solution of the problem, the variables $x$ and $\zeta$ provide an optimal portfolio and the corresponding value of the threshold function. The constants $\nu_1$, $\nu_2$, $\nu_3$ define the portions of capital "one is willing to lose," with $0 \le \nu_1, \nu_2, \nu_3 \le 1$, and several constraints can be combined together. The reduction to a linear program allows solving problems with many thousands of instruments; linear programming approaches are in common use in portfolio optimization with various criteria, including mean absolute deviation ([KonnoYamazaki1991]), maximum deviation ([Young1998]), and mean regret ([DemboKing1992]).

## Drawdown-Based Constraints and Objectives in Portfolio Construction

The three measures give rise, symmetrically to the mean-variance approach of [Markowitz1952] in the sample-variance setting, to as many formulations with the return as the performance function and the risk measure as the constraint: $M(x) \le \nu_1 C$, $A(x) \le \nu_2 C$, or $\Delta_\alpha(x) \le \nu_3 C$. Dually with respect to the constraints, the same measures serve as alternative objectives through the reward/risk ratios, defined as
$$\text{MaxDDRatio} = \frac{R(x)}{M(x)}, \qquad \text{AvDDRatio} = \frac{R(x)}{A(x)}, \qquad \text{CDaRRatio} = \frac{R(x)}{\Delta_\alpha(x)} .$$
As in classical portfolio theory, the portfolio with maximum reward/risk ratio corresponds to the point of tangency between the line through the origin $(0,0)$ and the efficient frontier.

The numerical evidence reported concerns the equity curves generated by a technical trading system on futures in $m = 32$ different markets (currencies, currency crosses, short- and long-term U.S. treasuries, foreign treasuries, international equity indices, and metals), over the period from 1/1/1988 to 1/9/1999, on 20 million of collateral capital in a margin account, with uncompounded curves. The technological constraints were set at $x_{\min} = 0.2$ and $x_{\max} = 0.8$, analogous to the "fully-invested" condition of Sharpe-Markowitz theory: they bound the strategy's leverage and make the efficient frontier concave, whereas their absence would lead to infinite leverage and a straight-line frontier through the origin. The linear problems (16), (17), and (18) were solved with the CPLEX solver, with independent verification obtained by solving the corresponding non-linear optimization problems via a genetic algorithm, which produced the same sets of weights.

The comparison among the three measures has a practical reading. The case $(1-\alpha) = 0$ corresponds to the MaxDD problem and $(1-\alpha) = 1$ to the AvDD problem; in the reward-MaxDD plane the frontier is efficient only for the case $(1-\alpha) = 0$, and in the reward-AvDD plane only for $(1-\alpha) = 1$, with each measure being optimal in its own sense. Optimization with a 5% CDaR constraint (that is, $(1-\alpha) = 0.05$, minimizing the average of the worst 5% of the underwater curve) produces a set of weights significantly different from that of the MaxDD case and involves dozens of events in the average, resulting in a portfolio more robust than both MaxDD and AvDD: the CDaR solution accounts for many significant drawdowns, unlike MaxDD, which considers only the single largest drawdown, while at the same time not being influenced by many small drawdowns as the AvDD solution is. The conclusion is that the solutions produced by the MaxDD approach may contain a substantial amount of statistical error, being based on a single observation of the maximum loss; thanks to the statistical averaging of drawdowns, the CDaR family allows better prediction of future risk and a more stable weight allocation, with an appropriate confidence level (for example $\alpha = 0.95$).

## References

- **[ArtznerEtAl1999]** Artzner, P., Delbaen, F., Eber, J. M., & Heath, D. (1999). Coherent Measures of Risk. Mathematical Finance, 9, 203-228.
- **[ChekhlovUryasevZabarankin2003]** Chekhlov, A., Uryasev, S., & Zabarankin, M. (2003). Portfolio Optimization with Drawdown Constraints. Research Report, Risk Management and Financial Engineering Lab, Center for Applied Optimization, University of Florida, Gainesville, FL.
- **[CvitanicKaratzas1995]** Cvitanic, J., & Karatzas, I. (1995). On Portfolio Optimization Under "Drawdown" Constraints. IMA Lecture Notes in Mathematics & Applications, 65, 77-88.
- **[DemboKing1992]** Dembo, R. S., & King, A. J. (1992). Tracking Models and the Optimal Regret Distribution in Asset Allocation. Applied Stochastic Models and Data Analysis, 8, 151-157.
- **[GrossmanZhou1993]** Grossman, S. J., & Zhou, Z. (1993). Optimal Investment Strategies for Controlling Drawdowns. Mathematical Finance, 3(3), 241-276.
- **[Jorion1996]** Jorion, P. (1996). Value at Risk: A New Benchmark for Measuring Derivatives Risk. Irwin Professional Publisher.
- **[KonnoYamazaki1991]** Konno, H., & Yamazaki, H. (1991). Mean Absolute Deviation Portfolio Optimization Model and Its Application to Tokyo Stock Market. Management Science, 37, 519-531.
- **[KrokhmalEtAl2002]** Krokhmal, P., Palmquist, J., & Uryasev, S. (2002). Portfolio Optimization with Conditional Value-At-Risk Objective and Constraints. The Journal of Risk, 4(2).
- **[Markowitz1952]** Markowitz, H. M. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.
- **[MausserRosen1999]** Mausser, H., & Rosen, D. (1999). Beyond VaR: From Measuring Risk to Managing Risk. ALGO Research Quarterly, 1(2), 5-20.
- **[Rockafellar1970]** Rockafellar, R. T. (1970). Convex Analysis. Princeton University Press.
- **[RockafellarUryasev2000]** Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. The Journal of Risk, 2, 21-42.
- **[RockafellarUryasev2002]** Rockafellar, R. T., & Uryasev, S. (2002). Conditional Value-at-Risk for General Loss Distributions. Journal of Banking and Finance, 26, 1443-1471.
- **[Young1998]** Young, M. R. (1998). A Minimax Portfolio Selection Rule with Linear Programming Solution. Management Science, 44(5), 673-683.
