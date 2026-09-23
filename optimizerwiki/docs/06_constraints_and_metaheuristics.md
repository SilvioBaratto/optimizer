---
title: "Constraints and Metaheuristics"
chapter: 6
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-10
---

> [!abstract] Summary
> The chapter establishes that introducing mixed-integer constraints — minimum transaction lots, maximum number of assets, minimum number of lots — turns portfolio selection into a problem whose feasibility is NP-complete and whose solution is NP-hard, destroying the analytical properties of the efficient frontier. It shows how, alongside exact approaches available only for special cases and alternative risk measures (semi-variance, mean absolute deviation, skewness), the practicable route runs through penalty reformulation of the constraints and their solution via metaheuristics. It finally formalizes Particle Swarm Optimization and documents its numerical application to portfolio selection.

## The Three Categories of Mixed-Integer Constraints

The classical mean-variance model, discussed in the chapter [[02 Selezione media-varianza]], operates on continuous decision variables: the percentages of capital to be invested in each asset. Operational practice, however, imposes restrictions that cannot be represented by continuous variables alone. In mathematical programming problems for the static selection of equity portfolios, three main categories of mixed-integer constraints are distinguished [AndramonovCorazza2002]:

- **constraints on minimum transaction lots**: a security must be bought or sold only in an integer number of units (lots). For example, for a security whose lot consists of $250$ shares, only integer quantities of lots can be traded;
- **constraints on the maximum positive integer number of different equity securities** that can be bought and sold; for example, the portfolio to be selected must consist of at most $5$ securities;
- **constraints on the minimum positive integer number of minimum transaction lots** of a given security that must be bought; for example, at least $2$ minimum lots of a security must be bought or sold.

These constraints give mathematical programming problems for static portfolio selection greater operational relevance than that of the classical continuous-variable-only problems. Their introduction, however, carries at least two non-trivial implications [AndramonovCorazza2002]:

- **verifying the feasibility** of the constraint system of these problems is, in general, an **NP-complete** problem;
- **solving** these mathematical programming problems is, in general, an **NP-hard** problem.

Informally, NP-complete problems are hard to solve in terms of required computation time, since they admit no polynomial-time solution algorithms; NP-hard problems are at least as hard to solve as NP-complete problems, and can be even harder.

## Complexity of Feasibility

The intractable nature of these problems already manifests itself at the level of mere feasibility checking, even before optimization. Consider the following constraint system, which contains a single mixed-integer constraint, the cardinality constraint:

$$
\begin{cases}
Ax \le b \\
\#(\{i : x_i > 0\}) \le K \\
0 \le x_i \le u_i \quad \forall i = 1,\dots,N
\end{cases}
$$

where $A$ is a known $M\times N$ matrix, $x$ is the $N$-dimensional vector of decision variables, $b$ is a known $M$-dimensional vector, $K$ is a positive integer smaller than $N$, $\#(\cdot)$ denotes the cardinality of the argument set, and the $u_i$ are upper bounds. It can be shown that, under specified assumptions, verifying the feasibility of this system is an NP-complete problem already when

$$
M \ge 3,
$$

that is, already when matrix $A$ has a number of rows greater than or equal to three [AndramonovCorazza2002]. The intractability threshold is thus reached with an extremely small number of linear constraints: it is not the size $N$ of the portfolio that generates the difficulty, but the joint presence of the integer cardinality constraint and a few linear inequalities.

## Minimum Transaction Lots

Among the categories of mixed-integer constraints, that of minimum transaction lots is presumably the most widespread. Such constraints require that, given a security, it may be bought or sold only in an integer number of minimum lots. In this formulation the most natural decision variables turn out to be the **number of lots**, rather than the classical percentages of the initially available capital [AndramonovCorazza2002].

### The Andramonov and Corazza Model

The model of [AndramonovCorazza2002] is written as

$$
\begin{aligned}
\min \quad & (PLx)'V(PLx) \\
\text{s.t.} \quad &
\begin{cases}
(PLx)'r \ge \pi C \\
f_1(x) \le \alpha C \\
f_2(x) \le \beta C \\
(PLx)'e \ge (1-\alpha-\beta)C \\
x_i \ge 0 \quad \forall i \\
x_j \in \mathbb{N} \quad \forall j \in I
\end{cases}
\end{aligned}
$$

where $x$ is the $(N+1)$-dimensional vector of decision variables, $P$ is the known diagonal matrix of current prices, $L$ is the known diagonal matrix of positive integer numbers of units making up the corresponding lot, $V$ is the known variance-covariance matrix of returns, $r$ is the known vector of average returns, $\pi$ is the required return, $C$ the initially available capital, $f_1(\cdot)$ and $f_2(\cdot)$ suitable nonlinear functions relating respectively to transaction costs and taxation, $e$ the unit vector, with $\alpha,\beta\ge 0$ and $\alpha+\beta<1$. The objective function can be rewritten as

$$
(PLx)'V(PLx) = x'(L'P'VPL)x,
$$

exhibiting the quadratic form in the matrix $L'P'VPL$. To solve the problem, an iterative two-stage solution approach was proposed, based on **branch-and-bound** algorithmic techniques and on **cutting plane** and **sub-gradient** methodologies. The following termination result also holds: if at least one security is infinitely divisible, then in a finite number of iterations the proposed solution approach either finds an optimal solution or indicates that the feasible region is empty [AndramonovCorazza2002].

An illustrative application with $N=2$ clarifies the structure of the problem. Setting

$$
V = \begin{pmatrix} 0.60 & -0.50 \\ -0.50 & 1.00 \end{pmatrix}, \quad
P = \begin{pmatrix} 3 & 0 \\ 0 & 7 \end{pmatrix}, \quad
L = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix},
$$

$r'=(0.20,\,0.40)$, $\pi=0.25$, $C=100$, $\alpha=0.10$, $\beta=0.20$, $f_1(x)=\tfrac{200}{81}(\sqrt{x_1}+\sqrt{x_2})$, and $f_2(x)=2(x_1+x_2)$, the problem can be rewritten as

$$
\begin{aligned}
\min \quad & 0.6\, x_1^2 + x_2^2 - x_1 x_2 \\
\text{s.t.} \quad &
\begin{cases}
0.6\, x_1 + 2.8\, x_2 \ge 25 \\
3 x_1 + 7 x_2 \le 70 \\
\sqrt{x_1} + \sqrt{x_2} \le 4.05 \\
x_1 + x_2 \le 10 \\
x_1, x_2 \ge 0, \quad x_1, x_2 \in \mathbb{N}.
\end{cases}
\end{aligned}
$$

The iterative procedure starts from the feasible point $(0,10)$, passes through the feasible point $(0,9)$, explores the infeasible intermediate point $(2,9)$, and finally converges to the optimal solution $(1,9)$.

### The Corazza and Favaretto Existence Theorem

A second integer-lot model, due to [CorazzaFavaretto2007], comes with a complete characterization of the existence of feasible solutions. The problem is

$$
\begin{aligned}
\min \quad & (PLx)'V(PLx) \\
\text{s.t.} \quad &
\begin{cases}
(PLx)'r \ge \pi C \\
(PLx)'e \ge C \\
x_i \in \mathbb{N}, \quad i = 2,\dots,N+1.
\end{cases}
\end{aligned}
$$

Let $I=\{2,\dots,n+1\}$ be the set of indices of the equity securities. The problem admits a feasible solution **if and only if**

$$
(r_1 \ge \pi) \ \lor\ \big((r_i < \pi) \land (\exists\, \bar\imath \in I \ \text{such that}\ r_{\bar\imath} > r_1)\big).
$$

The condition provides a constructive feasibility test: either the reference asset (index $1$) already achieves at least the required return, or, when this is not the case, there must exist at least one security that dominates it in average return [CorazzaFavaretto2007].

## Maximum Number of Securities and Tracking Error Volatility

Constraints on the maximum number of tradable securities require that the number of different equity securities traded not exceed a prescribed positive integer value. Using this category of constraints makes it possible to limit, albeit indirectly and imprecisely, transaction costs and the drag from taxation.

### The Jansen and Van Dijk Model

In the context of index management, [JansenVanDijk2002] formulate the problem of replicating a benchmark with a portfolio composed of a fixed number $K$ of securities:

$$
\begin{aligned}
\min \quad & TEV(x) \\
\text{s.t.} \quad &
\begin{cases}
x'e = 1 \\
\#(\{i : x_i > 0\}) = K \\
x_i \ge 0, \quad i = 1,\dots,N,
\end{cases}
\end{aligned}
$$

where $TEV(\cdot)$ is a suitable **tracking error volatility** function. This measure quantifies the closeness of the portfolio's performance to that of the chosen benchmark, typically via the volatility of the difference between the portfolio's return and the benchmark's return:

$$
TEV(x) = Var(r_{\text{Portfolio}} - r_{\text{Benchmark}}).
$$

### An Approximate Solution Approach

Given the complexity of the problem, which contains a mixed-integer cardinality constraint, an approximate all-continuous-variable version of it is solved instead. The approach proceeds in three steps [JansenVanDijk2002]. First, the cardinality constraint is moved into the objective function via an additive term weighted by $c\ge 0$:

$$
\begin{aligned}
\min \quad & TEV(x) + c\cdot \#(\{i : x_i > 0\}) \\
\text{s.t.} \quad & x'e = 1, \quad x_i \ge 0, \ i=1,\dots,N.
\end{aligned}
$$

The following result is then used

$$
\#(\{i : x_i > 0\}) = \lim_{p \downarrow 0} \left(x_1^p, \dots, x_N^p\right)'e,
$$

which expresses the cardinality of the support as the limit of the sum of the $p$-th powers of the components. Substituting an approximation of this limit into the objective function, for a value of $p$ close to zero, finally yields the approximate all-continuous-variable version

$$
\begin{aligned}
\min \quad & TEV(x) + c \cdot \left(x_1^p, \dots, x_N^p\right)'e \\
\text{s.t.} \quad & x'e = 1, \quad x_i \ge 0, \ i=1,\dots,N.
\end{aligned}
$$

### Discontinuity of the Efficient Frontier

The presence of this category of constraints can make the **efficient frontier discontinuous**. In the example of [JobstEtAl2001] with $N=4$ securities and $K=2$, the frontier in the expected-return-variance plane turns out to be made up of two separate, disjoint arcs: the mixed-integer cardinality constraint breaks the efficient curve into disconnected segments rather than leaving it continuous. This is one of the most significant consequences of integer constraints: they destroy meaningful analytical properties of the efficient frontier, which in the classical model is instead a continuous curve.

## Minimum Number of Lots

The third category of constraints requires that the minimum number of minimum transaction lots of a given security to be bought not be less than a prescribed positive integer value. Of the three, it is one of the least widespread.

A model that simultaneously incorporates lower and upper bounds on quantities and a cardinality constraint is that of [JobstEtAl2001]:

$$
\begin{aligned}
\min \quad & \sum_{i=1}^{N}\sum_{j=1}^{N} \sigma_{ij}\, x_i x_j \\
\text{s.t.} \quad &
\begin{cases}
\displaystyle\sum_{i=1}^{N} x_i r_i = \pi \\[4pt]
\displaystyle\sum_{i=1}^{N} x_i = 1 \\[4pt]
l_i \cdot \delta_i \le x_i \le u_i \cdot \delta_i, \quad i = 1,\dots,N \\
\delta_i \in \{0,1\}, \quad \forall i \\[4pt]
\displaystyle\sum_{i=1}^{N} \delta_i = K,
\end{cases}
\end{aligned}
$$

where $l_i$ and $u_i$ are respectively the lower and upper bounds. The binary variables $\delta_i$ act as switches: when $\delta_i=0$ the double constraint $l_i\delta_i\le x_i\le u_i\delta_i$ forces $x_i=0$; when $\delta_i=1$ the share is constrained to lie between $l_i$ and $u_i$. In particular, a lower bound $l_i>0$ implements the minimum constraint, requiring that, if the security is selected, at least the quantity $l_i$ be bought. The last constraint, $\sum_i \delta_i = K$, also adds to the model a restriction on the number of different tradable securities. The solution approach used is based on **branch-and-bound tree search** techniques [JobstEtAl2001].

In summary, the following facts hold for all the mixed-integer constraints considered here: verifying the feasibility of the constraint system is NP-complete; solving the problem is NP-hard; there is a limited number of constructive theoretical results regarding their solution; and their presence destroys meaningful analytical properties of the efficient frontier.

## Alternative Risk Measures in the Constrained Model

The motivations for extending the classical model concern both the constraints and the adopted risk measure. The Markowitz model indeed has some limitations: returns in general do not follow a multivariate normal distribution, but a right-skewed distribution; the diversification effect tends to decrease as the number of included securities grows; variance is a variability index sensitive to outliers and can be adopted as a risk measure only if the return distribution is symmetric; and the underlying assumptions are not very realistic. The topic of desirable properties for a risk measure is taken up in the chapter [[04 Misure di rischio coerenti]].

### Three Alternative Measures

**Semi-variance** is a measure of so-called *downside risk*, that is, of risk associated with returns below a prescribed benchmark, usually zero or the mean value. Investors who do not engage in short selling are interested in minimizing the risk that portfolio returns fall below the mean, since returns above the mean are desirable; semi-variance is therefore appropriate when risk is perceived as the possibility of adverse outcomes, rather than as return dispersion. For the random return $r$ of an asset it is defined as

$$
Semi\text{-}Var(r) = \sum_{t=1;\,r_t<\bar r}^{T} \left(r_t-\bar r\right)^2 \Big/ T .
$$

Semi-variance is equivalent to variance when returns follow a symmetric distribution, but it is not easily tractable from an analytical point of view.

**Mean absolute deviation (MAD)** is defined as

$$
MAD(r) = \sum_{t=1}^{T} \left| r_t - \bar r \right| \Big/ T .
$$

It is a measure sensitive to outliers, though less so than variance, and it is not easy to handle analytically due to the presence of the absolute value operator.

**Skewness** is the first moment after variance, the third moment, and can contain useful information for investment decisions. Investors generally prefer positive skewness: it indicates that the distribution has a fatter right tail, that is, that the return takes values above the mean with higher probability.

### The Cardinality Constraint and the Mixed-Integer Formulation

A first extension of the classical model introduces a cardinality constraint limiting the number of securities actually held:

$$
\min_{x,z}\; Var(r) \quad \text{s.t.} \quad \sum_{i=1}^{N} x_i z_i = \pi, \ \ \sum_{i=1}^{N} x_i z_i = 1, \ \ \sum_{i=1}^{N} z_i = K, \ \ z_i\in\{0,1\},
$$

with $z_i=1$ if the $i$-th security is selected and $z_i=0$ otherwise, and $K$ the number of selectable securities. The purpose is a, albeit indirect, control of transaction costs. The constraint leads to the so-called **cardinality constrained efficient frontier**, which in general is discontinuous: imposing a fixed number $K$ of securities breaks the efficient frontier into disconnected segments rather than leaving it as a continuous curve.

The alternative measures are grafted onto this mixed-integer structure. The **mean-variance** model with mixed-integer constraints is

$$
\min_{x,z} \; \lambda\Big[\sum_{i,j} x_i x_j \sigma_{ij}\Big] - (1-\lambda)\Big[\sum_{i} x_i \mu_i\Big]
$$
$$
\text{s.t.}\quad \sum_i x_i = 1, \ \ \sum_i z_i = K, \ \ \varepsilon_i z_i \le x_i \le \delta_i z_i, \ \ z_i\in\{0,1\},
$$

with $\lambda\in[0,1]$, $\varepsilon_i$ and $\delta_i$ the minimum and maximum selectable percentages of security $i$. The case $\lambda=0$ corresponds to maximizing expected return, $\lambda=1$ to minimizing risk, and an intermediate value to a trade-off between risk and return. Substituting the objective function yields the **mean semi-variance** model, with risk term $\lambda\big[\sum_{t;\,r_t<\bar r}(r_t-\bar r)^2/T\big]-(1-\lambda)\bar r$, and the **mean absolute deviation** model, with term $\lambda\big[\sum_t |r_t-\bar r|/T\big]-(1-\lambda)\bar r$. Between these two, MAD and variance are rather similar variability measures, but they differ in one crucial respect: the problem associated with mean absolute deviation is **linear**, while that associated with variance is **quadratic**.

The **mean-variance with skewness** model adds the third moment:

$$
\min_{x,z} \; \lambda\Big[\sum_{t=1}^{T} \frac{(r_t-\bar r)^2}{T}\Big] - (1-\lambda)\bar r - \theta\,\frac{\sum_{t=1}^{T} (r_t-\bar r)^3/T}{\big(\sum_{t=1}^{T} (r_t-\bar r)^2/T\big)^{3/2}},
$$

where $\theta$ denotes the investor's aversion to or preference for skewness. With reference to the portfolio's return, the model maximizes its mean value, minimizes its variance, and maximizes its skewness.

### Equivalence Between Physical Quantities and Wealth Proportions

The preceding formulations can be written either in physical quantities $x_i$ (number of units held) or in wealth proportions $w_i$. Letting $v_{it}$ be the value of one unit of asset $i$ at time $t$, $C_{cash}$ the available cash, and setting $w_i = v_{iT}x_i/C_{cash}$, the constraint system in $x_i$

$$
\sum_i z_i = K, \quad \varepsilon_i z_i \le \frac{v_{iT}x_i}{C_{cash}} \le \delta_i z_i, \quad \sum_i v_{iT}x_i = C_{cash}, \quad x_i\ge 0, \quad z_i\in\{0,1\}
$$

is equivalent to the system in $w_i$

$$
\sum_i w_i = 1, \quad 0\le w_i\le 1, \quad \sum_i z_i = K, \quad \varepsilon_i z_i \le w_i \le \delta_i z_i, \quad z_i\in\{0,1\},
$$

with the continuous-time return over a single period given by $r_t = \log_e\big\{(\sum_i w_i v_{it}/v_{iT})/(\sum_i w_i v_{i,t-1}/v_{iT})\big\}$. The switch from physical quantities to wealth proportions preserves exactly the same feasible set, and it is the formulation in $w_i$ that is used in models with mixed-integer constraints.

## From Constrained Optimization to Metaheuristics

When optimal solutions cannot be obtained efficiently, the only option is to trade optimality for efficiency. The approximate algorithms called **heuristics** try to obtain near-optimal solutions at a relatively low computational cost: a heuristic is a technique designed to solve a problem more quickly when classical methods are too slow, or to find an approximate solution when classical methods fail to find an exact one, trading optimality, completeness, accuracy, or precision for speed [BlumRoli2003]. **Metaheuristics**, proposed starting in the 1980s to overcome the limitations of heuristics, are defined as an iterative generation process that guides a subordinate heuristic by intelligently combining different concepts to explore and exploit the search space [OsmanLaporte1996]. They balance **exploration**, the ability to visit different regions of the search space, and **exploitation**, the ability to concentrate the search around a promising area to refine a candidate solution [Engelbrecht2007]. Metaheuristics are not specific to a particular problem, are approximate and stochastic solution methods, can avoid the traps of local minima, and can incorporate some form of memory. They are classified, depending on the number of solutions used simultaneously, into *population-based* and *trajectory-based*, and, depending on the type of inspiration, into evolutionary, physics-based, human-based, and swarm-based [NassefEtAl2023].

### Penalty Functions and Exact Penalization

Metaheuristics such as genetic algorithms and Particle Swarm Optimization are tools for **unconstrained** global optimization, whereas portfolio selection models are **constrained** optimization problems. The bridge between the two is provided by penalty functions: the constrained problem is reformulated as unconstrained by adding to the objective function terms that measure constraint violation. Given the general problem

$$
\min_{x\in\mathbb{R}^d} f(x) \quad \text{s.t.} \quad g_r(x)=a_r\ (r=1,\dots,R), \quad h_s(x)\ge b_s\ (s=1,\dots,S),
$$

the **Exact Penalty Method** rewrites it as

$$
\min_{x\in\mathbb{R}^d} f(x) + \frac{1}{\varepsilon}\Big[\sum_{r=1}^{R} |g_r(x)-a_r| + \sum_{s=1}^{S} \max(0,\,b_s-h_s(x))\Big],
$$

where $\varepsilon$ is the **penalty factor**, the term $|g_r(x)-a_r|$ measures the violation of the equality constraints, and $\max(0,b_s-h_s(x))$ that of the inequality constraints. It can be shown that, for appropriate values of $\varepsilon$, the solutions of the constrained and unconstrained problems coincide. The qualifier "exact" distinguishes this method from penalties that would require $\varepsilon\to 0$: here there exists a finite threshold of $\varepsilon$ beyond which the two problems have the same optimum.

Constraints on binary variables, which are in themselves neither equalities nor inequalities in standard form, are handled with an ad hoc violation measure. For the constraint $z_i\in\{0,1\}$ one adopts

$$
z_i(1-z_i),
$$

which is zero if and only if $z_i\in\{0,1\}$ and positive for any other real value of $z_i$, and can therefore be incorporated into the penalty function. Applying this scheme to the mean-variance model with mixed-integer constraints, each constraint becomes a penalty term:

$$
\sum_i x_i = 1 \ \to\ \Big|\sum_i x_i - 1\Big|, \qquad \sum_i z_i = K \ \to\ \Big|\sum_i z_i - K\Big|,
$$
$$
\varepsilon_i z_i \le x_i \ \to\ \sum_i \max\{0;\varepsilon_i z_i - x_i\}, \qquad x_i \le \delta_i z_i \ \to\ \sum_i \max\{0; x_i-\delta_i z_i\},
$$
$$
z_i\in\{0,1\} \ \to\ \sum_i |z_i(1-z_i)|,
$$

so that the reformulated unconstrained model, with penalty weighted by $1/\varepsilon$, is

$$
\min_{x,z} \; \lambda\Big[\sum_{i,j} x_i x_j \sigma_{ij}\Big] - (1-\lambda)\Big[\sum_i x_i \mu_i\Big] + \frac{1}{\varepsilon}\Big[\Big|\sum_i x_i - 1\Big| + \Big|\sum_i z_i - K\Big|
$$
$$
+ \sum_i \max\{0;\varepsilon_i z_i - x_i\} + \sum_i \max\{0; x_i-\delta_i z_i\} + \sum_i |z_i(1-z_i)|\Big].
$$

The procedure for the semi-variance, MAD, and mean-variance-skewness models is analogous. The portfolio selection problem in its complex form is, moreover, strongly nonlinear, non-differentiable, and mixed-variable (integer and continuous), with a non-continuous objective function and constraints, so that ad hoc solution procedures must be developed; a complete treatment of a complex version with non-smooth penalty reformulation is given in [CorazzaFasanoGusso2013].

## Particle Swarm Optimization

**Particle Swarm Optimization (PSO)**, introduced by [KennedyEberhart1995], is a *nature-inspired*, iterative, *population-based* metaheuristic, using memory, evolutionary and *derivative-free*, for solving unconstrained global optimization problems. *Swarm intelligence* is the cooperation of individuals within a group to achieve a goal by exchanging locally available information [ErwinEngelbrecht2023]. The idea of PSO is to replicate the behavior of bird flocks when they cooperate to optimize their search for food: each particle in the swarm explores the search area while retaining memory of its own best position reached so far, information that is then exchanged with its neighbors; the entire swarm is thus supposed to converge toward the best global position. In the mathematical counterpart, each swarm member represents a possible solution, is initially positioned randomly within the feasible set, and receives a random velocity that determines its initial direction of movement.

### Formalization

Consider the global optimization problem

$$
\min_{x\in\mathbb{R}^d} f(x), \qquad f:\mathbb{R}^d \to \mathbb{R},
$$

and assume $M$ particles are used. At step $k$, three vectors are associated with particle $j$: the position $x_j^k\in\mathbb{R}^d$, the velocity $v_j^k\in\mathbb{R}^d$, and the best position visited so far $p_j\in\mathbb{R}^d$, with $pbest_j = f(p_j)$. In addition, $p_{g,nei}\in\mathbb{R}^d$ denotes the best position visited so far by the swarm members belonging to a given neighborhood $nei$. Which $p_{g,nei}$ applies to each particle depends on the **topology** of the neighborhood, that is, on the structure of the connections through which particles exchange information. Among the topologies one distinguishes the **fully connected neighborhood**, in which every particle is connected to all others, the **von Neumann neighborhood**, with partial cross-connections, and the **ring neighborhood**, in which particles are arranged in a ring and each is connected only to its adjacent neighbors.

### The Algorithm with Inertia Weight

In the version with *inertia weight*, the algorithm proceeds as follows. Set $k=1$, evaluate $f(x_j^k)$ for $j=1,\dots,M$, and initialize $pbest_j=+\infty$. Then, until a stopping criterion is satisfied, for each particle: if $f(x_j^k)<pbest_j$, update $p_j=x_j^k$ and $pbest_j=f(x_j^k)$; if $f(x_j^k)<gbest$, update $p_g=x_j^k$ and $gbest=f(x_j^k)$; and finally update velocity and position according to

$$
\begin{cases}
v_j^{k+1} = w^{k+1}v_j^k + U_{\phi_1}\otimes(p_j - x_j^k) + U_{\phi_2}\otimes(p_g - x_j^k) \\
x_j^{k+1} = x_j^k + v_j^{k+1},
\end{cases}
$$

where $U_{\phi_1}$ and $U_{\phi_2}$ are vectors uniformly distributed on $[0,\phi_1]$ and $[0,\phi_2]$ and $\otimes$ denotes the component-wise product. The updated velocity thus combines three contributions: the inertia term $w^{k+1}v_j^k$, the **cognitive** component $U_{\phi_1}\otimes(p_j-x_j^k)$, directed toward the individual best position, and the **social** component $U_{\phi_2}\otimes(p_g-x_j^k)$, directed toward the global best position. The values of $\phi_1$ and $\phi_2$ affect the algorithm's performance and, to achieve swarm convergence, must be set consistently with the value of the inertia weight $w^k$.

The inertia weight, introduced by [ShiEberhart1998], is generally linearly decreasing with the number of steps,

$$
w^k = w_{max} + \frac{w_{min}-w_{max}}{K}k,
$$

with $w_{min}$ and $w_{max}$ usually equal to $0.4$ and $0.9$ and $K$ the maximum number of allowed steps. Alternatively $w^k$ can be constant, $w^k = w = 0.7298$; in this case it is advisable to set an upper bound on $v_j^k$ so as to prevent divergent behavior during optimization.

### Convergence

The standard version of the algorithm (*Standard PSO*, SPSO) reaches the global optimum **in probability** [XuYu2018]. This is a probabilistic guarantee, consistent with the stochastic nature of metaheuristics, not a deterministic convergence in a finite number of steps.

### Handling Constraints and Binary Variables

Since PSO is a method for unconstrained optimization, the constraints of the selection problem are incorporated into the function to be minimized via the exact penalization described earlier; the binary variables $z_i$ are treated as real variables subject to the penalty term $z_i(1-z_i)$, which vanishes exactly at the admissible values $0$ and $1$. In this way the entire mixed-integer, constrained problem is reduced to the unconstrained minimization of a single fitness function, to which the PSO algorithm can be directly applied.

## Numerical Experiments on Portfolio Selection

PSO with exact penalization has been applied to two portfolio selection problems, documenting respectively its behavior on the classical model and on the model with mixed-integer constraints.

### The Basic Mean-Variance Problem

The first problem is the basic portfolio selection problem with $D=3$ risky assets, for which the exact closed-form solution is known (cf. [[02 Selezione media-varianza]]),

$$
\min_{x} x'Vx \quad \text{s.t.} \quad x'r=\pi, \ \ x'e=1,
$$

reformulated for PSO as

$$
\min_{x} x'Vx + \frac{1}{\varepsilon}\big(|x'r-\pi| + |x'e-1|\big).
$$

The algorithm setup uses a number of particles equal to $15,20,25,30$, a *fully connected* topology, $w^k=0.85$, $c_1=0.5$, $c_2=0.7$, a stopping criterion at $k_{max}=8000$ iterations, and initialization of positions and velocities with values uniform on $[0,1)$, suitably normalized for the positions. The exact frontier is computed at nine values of the expected return $\pi$, defining nine scenarios.

The experiments investigated the role of the parameters in sequence. Regarding the **penalty factor** $\varepsilon$, the values $1,\,0.1,\,0.01,\,0.001,\,0.0001$ were compared with twenty simulations each: as $\varepsilon$ decreases, the penalization of violations becomes stricter, but for $\varepsilon$ too small ($0.0001$) the algorithm struggles to balance variance minimization with constraint satisfaction, as shown by very high fitness values; for $\varepsilon=0.001$ the cloud of PSO portfolios turns out more concentrated around the Markowitz optimal solution. Regarding **initialization**, setting $x^i=[1,0,0]^T$ identical for all particles, the selected portfolios do not lie close to the frontier, due to slow convergence and poor uniform initialization. Regarding the **number of particles** $N$, as $N$ increases the cloud of points becomes denser around the optimal solution: increasing the number of particles improves both the speed and the quality of convergence.

With $N=30$ and normalized random initialization, the PSO points for each scenario cluster very close to the corresponding point on the Markowitz frontier. The **budget constraint** $x_{(PSO)}'e=1$ is satisfied exactly in $100\%$ of the simulations for every scenario, while the **return constraint** $x_{(PSO)}'r=R_P$ is satisfied in a variable percentage, from $0\%$ (scenario $9$) to $90\%$ (scenario $5$). Decomposing fitness into portfolio variance $VP$ and the residual share $\text{Fitness}-VP$ due to penalization, simulations are observed in which $\text{Fitness}-VP=0$: in these cases the particle satisfied almost exactly both constraints, and fitness coincides with variance alone. Constructing the frontier from the minimum-fitness PSO portfolio for each scenario, the PSO curve turns out nearly superimposed on the exact Markowitz one. A variant with the return constraint expressed as an inequality,

$$
x_{(PSO)}'R \ge R_P, \qquad \text{fitness} = x_{(PSO)}'V x_{(PSO)} + \frac{1}{\varepsilon}\big|x_{(PSO)}'\mathbf 1 - 1\big| + \frac{1}{\varepsilon}\big|\max(0,\,R_P - x_{(PSO)}'R)\big|,
$$

clusters the PSO points even more markedly along the frontier: relaxing the return constraint from equality to inequality makes it easier to find portfolios close to the frontier, since a particle can exceed the minimum required return without being penalized for it.


### The Mean-Variance Model with Mixed-Integer Constraints

The second problem is the mean-variance model with mixed-integer constraints in the penalty reformulation, applied to FTSE-MIB securities with data from $15.05.2007$ to $31.12.2008$. The algorithm setup is: $40$ particles, $4000$ iterations, penalty parameter $0{,}0001$, inertia weight $w=0{,}7298$, acceleration coefficients $c_1=c_2=1{,}49618$, minimum percentage $\varepsilon_i=1\%$, maximum percentage $\delta_i=50\%$, number of selectable securities $K=10,20$, and variance aversion $\lambda=0{,}5$. The minimized fitness function adds to the mean-variance part the five penalty terms for budget, cardinality, minimum share, maximum share, and binariness of $z$, weighted by $1/\varepsilon$.

As $K$ varies, the optimal value of the objective function and the computation time are:

| $K$ | Objective function value | Time taken |
|---|---|---|
| $10$ | $15{,}2432$ | $10{,}3341$ |
| $20$ | $11{,}5738$ | $11{,}9563$ |

Consistent with the cardinality constraint, for each value of $K$ exactly $K$ securities have selection variable $z_i\approx 1$ and share $x_i>0$, while the remaining ones have $z_i\approx 0$ and $x_i\approx 0$; slight deviations from the exact theoretical values are observed (for example $z_i$ slightly negative or above $1$), due to the heuristic nature of PSO, which does not guarantee exact constraint satisfaction but only its minimization via penalties. The convergence curve with $K=10$ starts from a value close to $2\times 10^5$, decreases rapidly in the first few hundred iterations, and stabilizes near zero after about $1200$–$1500$ iterations: the algorithm converges quickly and then settles on the minimum found. Over the out-of-sample period from $01.01.2009$ to $30.06.2009$, the portfolio selected with $K=10$ achieves, starting from a unit base value, a marked positive cumulative return, reaching a value of about $1{,}47$ by the end of the semester despite an intermediate downturn and sideways phase.

Overall, the two experiments show that PSO with exact penalization faithfully reproduces the exact solution in the classical case and, in the mixed-integer case, delivers portfolios that respect the cardinality constraint and converge rapidly to the minimum of the fitness function.

## References

- **[AndramonovCorazza2002]** Andramonov, M.Y. and Corazza, M. (2002), "Mixed-integer non-linear programming methods for mean-variance portfolio selection", Rendiconti per gli Studi Economici Quantitativi, vol. 2001, 21-34.
- **[BlumRoli2003]** Blum, C. and Roli, A. (2003), "Metaheuristics in combinatorial optimization: Overview and conceptual comparison", ACM Computing Surveys, vol. 35(3), 268-308.
- **[CorazzaFasanoGusso2013]** Corazza, M., Fasano, G. and Gusso, R. (2013), "Particle Swarm Optimization with non-smooth penalty reformulation, for a complex portfolio selection problem", Applied Mathematics and Computation, vol. 224, 611-624.
- **[CorazzaFavaretto2007]** Corazza, M. and Favaretto, D. (2007), "On the existence of solutions in the quadratic mixed-integer mean-variance portfolio selection problem", European Journal of Operational Research, vol. 176(3), 1947-1960.
- **[Engelbrecht2007]** Engelbrecht, A.P. (2007), Computational Intelligence: An Introduction, 2nd ed., John Wiley & Sons, Chichester.
- **[ErwinEngelbrecht2023]** Erwin, K. and Engelbrecht, A. (2023), "Meta-heuristics for portfolio optimization", Soft Computing, vol. 27, 19045-19073.
- **[JansenVanDijk2002]** Jansen, R. and Van Dijk, R. (2002), "Optimal Benchmark Tracking with Small Portfolios", The Journal of Portfolio Management, vol. 28(2), Winter, 33-39.
- **[JobstEtAl2001]** Jobst, N.J., Horniman, M.D., Lucas, C.A. and Mitra, G. (2001), "Computational aspects of alternative portfolio selection models in the presence of discrete asset choice constraints", Quantitative Finance, vol. 1(5), 1-13.
- **[KennedyEberhart1995]** Kennedy, J. and Eberhart, R. (1995), "Particle Swarm Optimization", Proceedings of ICNN'95 - International Conference on Neural Networks, vol. 4, IEEE, Perth, 1942-1948.
- **[Markowitz1952]** Markowitz, H. (1952), "Portfolio Selection", The Journal of Finance, vol. 7(1), 77-91.
- **[NassefEtAl2023]** Nassef, A.M., Abdelkareem, M.A., Maghrabie, H.M. and Baroutaji, A. (2023), "Review of Metaheuristic Optimization Algorithms for Power Systems Problems", Sustainability, vol. 15(12), 9434.
- **[OsmanLaporte1996]** Osman, I.H. and Laporte, G. (1996), "Metaheuristics: A bibliography", Annals of Operations Research, vol. 63, 511-623.
- **[ShiEberhart1998]** Shi, Y. and Eberhart, R. (1998), "A Modified Particle Swarm Optimizer", Proceedings of the IEEE International Conference on Evolutionary Computation, IEEE World Congress on Computational Intelligence, Anchorage, 69-73.
- **[XuYu2018]** Xu, G. and Yu, G. (2018), "On convergence analysis of particle swarm optimization algorithm", Journal of Computational and Applied Mathematics, vol. 333, 65-73.
