---
type: concept
title: "Constraints and Metaheuristics"
description: How real-world mixed-integer constraints — minimum trading lots, a cap on the number of holdings, a minimum number of lots — turn portfolio selection into an NP-complete feasibility / NP-hard optimization problem that destroys the analytic efficient frontier, and how exact-penalty reformulation plus metaheuristics (Particle Swarm Optimization) recover practical solutions, with numerical evidence on both the classical and mixed-integer models.
tags: [cardinality-constraint, minimum-lot, mixed-integer, np-hard, penalty-method, particle-swarm-optimization, metaheuristics, tracking-error, downside-risk]
sources:
  - id: openwiki-source-f57eadb25599aa3d5c206ed0
    resource: repo://docs/06_constraints_and_metaheuristics.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Constraints and Metaheuristics

The classical mean-variance model of
[mean-variance selection](../foundations/mean-variance-selection.md) works on
continuous decision variables — the fractions of capital invested in each asset.
Operational practice imposes restrictions that cannot be represented by continuous
variables alone, and these **mixed-integer** constraints change the problem's
character: feasibility becomes NP-complete, optimization NP-hard, and the analytic
efficient frontier is destroyed. This page traces that break and the practical route
around it — exact-penalty reformulation solved with metaheuristics — complementing the
smooth theory in [convex optimization](../optimization/convex-optimization.md) and the
applied build in
[quantitative selection and construction](../signals/quantitative-selection-and-construction.md).

## The three mixed-integer constraint categories

Static equity portfolio-selection problems distinguish three main mixed-integer
constraint families:

- **Minimum trading lots** — an asset can be bought or sold only in an integer number
  of lots (e.g. a stock whose lot is 250 shares trades only in whole lots). Here the
  natural decision variable is the *number of lots*, not the percentage of capital.
- **Maximum number of distinct holdings** — the portfolio may contain at most a fixed
  positive integer of different stocks (e.g. at most 5).
- **Minimum number of lots** — if a stock is bought, at least a fixed positive integer
  number of its minimum lots must be acquired (e.g. at least 2 lots).

These constraints give the problem far greater operational relevance than the
classical continuous version, but introduce two non-trivial implications: **checking
the feasibility** of the constraint system is in general **NP-complete**, and
**solving** the problem is in general **NP-hard** (at least as hard as NP-complete,
possibly harder). Informally, NP-complete problems admit no polynomial-time solution
algorithm.

## The complexity of feasibility

Intractability shows up already at feasibility, before any optimization. For the
system with a single mixed-integer (cardinality) constraint,

$$Ax\le b,\qquad \#\{i:x_i>0\}\le K,\qquad 0\le x_i\le u_i,$$

with $A$ an $M\times N$ matrix and $K<N$ a positive integer, verifying feasibility is
NP-complete already when $M\ge 3$ — i.e. with at least three linear inequality rows.
The difficulty is therefore driven not by the portfolio dimension $N$ but by the joint
presence of the integer cardinality constraint and a handful of linear inequalities.

## Minimum trading lots

Lot constraints are the most widespread family. Writing decisions as lot counts $x$
with price matrix $P$ and lot-size matrix $L$ (both diagonal), the objective
$(PLx)'V(PLx)=x'(L'P'VPL)x$ is a quadratic form in $L'P'VPL$. The **Andramonov-Corazza**
model adds nonlinear transaction-cost and tax functions and integrality
$x_j\in\mathbb N$; it is solved by a two-stage iterative scheme using branch-and-bound,
cutting-plane and sub-gradient techniques, with a termination guarantee: **if at least
one asset is infinitely divisible**, the scheme either finds the optimum or reports
the feasible region empty in finitely many iterations. A worked $N=2$ instance starts
from the feasible point $(0,10)$, passes through $(0,9)$, explores the infeasible
$(2,9)$, and converges to the integer optimum $(1,9)$.

A second integer-lot model, due to **Corazza-Favaretto**, comes with a complete
feasibility characterization: with reference asset indexed 1 and stock set
$I=\{2,\dots,N+1\}$, a feasible solution exists **iff** $r_1\ge\pi$, or else
$r_1<\pi$ and some asset $\bar\imath\in I$ has $r_{\bar\imath}>r_1$ — a constructive
test: either the reference asset already meets the required return, or some stock must
dominate it in mean return.

## Maximum holdings, tracking error, and the discontinuous frontier

Capping the number of distinct holdings indirectly (and imprecisely) limits
transaction costs and taxes. In index tracking, the **Jansen-Van Dijk** model
minimizes the **tracking error volatility** $TEV(x)=\mathrm{Var}(r_{\text{port}}-r_{\text{bench}})$
subject to $x'e=1$, an exact cardinality $\#\{i:x_i>0\}=K$, and $x\ge0$. Because that
integer constraint is hard, an all-continuous approximation moves cardinality into the
objective weighted by $c\ge0$ and uses
$\#\{i:x_i>0\}=\lim_{p\downarrow0}(x_1^p,\dots,x_N^p)'e$, substituting a small $p$ to
get a continuous surrogate.

The decisive qualitative consequence: cardinality constraints can make the **efficient
frontier discontinuous**. On a worked $N=4$, $K=2$ example the frontier in the
mean-variance plane is two separate, disconnected arcs — the integer constraint breaks
the efficient curve into unconnected segments rather than leaving it continuous. This
is the general fact: integer constraints destroy significant analytic properties of
the frontier, which in the classical model is a continuous curve.

## Minimum number of lots

The least common family fixes a minimum lot count when a stock is bought. The
**Jobst et al.** model combines lower and upper bounds with cardinality using binary
switches $\delta_i\in\{0,1\}$: the double bound $l_i\delta_i\le x_i\le u_i\delta_i$
forces $x_i=0$ when $\delta_i=0$ and confines $x_i\in[l_i,u_i]$ when $\delta_i=1$, so a
lower bound $l_i>0$ realizes the minimum constraint, and $\sum_i\delta_i=K$ adds the
holdings cap; it is solved by branch-and-bound tree search. Across all three families:
feasibility is NP-complete, solving is NP-hard, constructive theory is scarce, and the
analytic frontier is destroyed.

## Alternative risk measures in the constrained model

Extending the classical model is motivated by both constraints and the risk measure.
Markowitz's variance has limits — returns are generally right-skewed rather than
multivariate normal, diversification's benefit decays as holdings grow, and variance
is outlier-sensitive and valid as a risk measure only for symmetric distributions
(the desiderata of a proper risk measure are taken up in
[coherent risk measures](../risk-measures/coherent-risk-measures.md)). Three
alternatives:

- **Semi-variance** — a *downside-risk* measure summing only below-benchmark
  deviations $\sum_{t:r_t<\bar r}(r_t-\bar r)^2/T$; it equals variance for symmetric
  returns but is not analytically tractable.
- **Mean absolute deviation (MAD)** $\sum_t|r_t-\bar r|/T$ — outlier-sensitive (less
  so than variance), awkward analytically because of the absolute value, but its
  optimization problem is **linear** whereas variance's is **quadratic**.
- **Skewness** — the third moment; investors prefer positive skew (a fatter right
  tail, higher-than-mean returns more likely).

A cardinality constraint yields the **cardinality-constrained efficient frontier**,
generally discontinuous. The mixed-integer **mean-variance** model minimizes
$\lambda\sum_{ij}x_ix_j\sigma_{ij}-(1-\lambda)\sum_i x_i\mu_i$ subject to $x'e=1$,
$\sum_i z_i=K$, $\varepsilon_i z_i\le x_i\le\delta_i z_i$, $z_i\in\{0,1\}$; $\lambda=0$
maximizes return, $\lambda=1$ minimizes risk, intermediate values trade off. Swapping
the risk term gives **mean-semi-variance** and **MAD** models, and adding the third
moment gives **mean-variance-skewness** with skew-preference weight $\theta$. The
physical-quantity ($x_i$) and wealth-proportion ($w_i$) formulations describe exactly
the same feasible set; the $w_i$ form is the one used.

## From constrained optimization to metaheuristics

When exact optima cannot be found efficiently, one trades optimality for efficiency.
**Heuristics** seek near-optimal solutions cheaply — trading optimality, completeness,
accuracy or precision for speed. **Metaheuristics** are an iterative generation
process guiding a subordinate heuristic, balancing **exploration** (visiting diverse
regions) against **exploitation** (refining a promising area); they are
problem-agnostic, approximate, stochastic, can escape local minima, and may use
memory, and are classified as population- vs trajectory-based and by inspiration
(evolutionary, physics-based, human-based, swarm-based).

### Penalty functions and exact penalization

Genetic algorithms and Particle Swarm Optimization solve **unconstrained** global
optimization, whereas selection is **constrained** — bridged by penalty functions.
The **Exact Penalty Method** rewrites $\min f(x)$ s.t. equalities $g_r(x)=a_r$ and
inequalities $h_s(x)\ge b_s$ as

$$\min_x f(x)+\tfrac1\varepsilon\Big[\sum_r|g_r(x)-a_r|+\sum_s\max(0,b_s-h_s(x))\Big],$$

with penalty factor $\varepsilon$. It is "exact" because, unlike penalties needing
$\varepsilon\to0$, a **finite** threshold of $\varepsilon$ makes the constrained and
unconstrained optima coincide. Binary constraints $z_i\in\{0,1\}$ get the ad-hoc
violation term $z_i(1-z_i)$, which is zero exactly on $\{0,1\}$ and positive
otherwise. Applied to the mixed-integer mean-variance model, budget, cardinality,
minimum-share, maximum-share and binarity constraints each become a penalty term,
weighted by $1/\varepsilon$. The full complex selection problem is strongly nonlinear,
non-differentiable and mixed-variable with non-continuous objective and constraints,
requiring purpose-built solvers (a non-smooth penalty reformulation is one such
treatment).

## Particle Swarm Optimization

**Particle Swarm Optimization (PSO)** is a nature-inspired, iterative,
population-based, memory-using, derivative-free metaheuristic for unconstrained global
optimization. It mimics birds cooperating to forage: each particle explores the search
space, remembers its own best position, and shares it with neighbors so the swarm
converges toward the global best; each particle is a candidate solution, randomly
placed in the feasible set with a random initial velocity.

### Formalization

For $\min_{x\in\mathbb R^d}f(x)$ with $M$ particles, at step $k$ particle $j$ carries a
position $x_j^k$, a velocity $v_j^k$, and its best-visited position $p_j$ with
$pbest_j=f(p_j)$; $p_{g,nei}$ is the neighborhood best. Which best applies depends on
the **topology** — fully connected (every particle linked to all), von Neumann
(partial cross links), or ring (each linked only to adjacent neighbors).

### The inertia-weight algorithm

Initializing $pbest_j=+\infty$, then until a stopping criterion holds, each particle
updates its personal and global bests and then

$$v_j^{k+1}=w^{k+1}v_j^k+U_{\phi_1}\otimes(p_j-x_j^k)+U_{\phi_2}\otimes(p_g-x_j^k),\qquad x_j^{k+1}=x_j^k+v_j^{k+1},$$

where $U_{\phi_1},U_{\phi_2}$ are uniform on $[0,\phi_1],[0,\phi_2]$ and $\otimes$ is
componentwise product. The velocity combines an **inertia** term $w^{k+1}v_j^k$, a
**cognitive** term toward the personal best, and a **social** term toward the global
best; $\phi_1,\phi_2$ must be set consistently with the inertia weight for
convergence. The inertia weight is typically linearly decreasing,
$w^k=w_{max}+\frac{w_{min}-w_{max}}{K}k$ (usually $w_{min}=0.4$, $w_{max}=0.9$), or
constant $w=0.7298$ (in which case a velocity cap prevents divergence). Standard PSO
reaches the global optimum **in probability** — a probabilistic guarantee, not
deterministic finite-step convergence. Constraints and binaries are handled by the
exact penalization above, reducing the whole mixed-integer constrained problem to
minimizing one unconstrained fitness function.

## Numerical experiments

**Base mean-variance ($D=3$).** For the closed-form problem $\min_x x'Vx$ s.t.
$x'r=\pi$, $x'e=1$, reformulated as $\min_x x'Vx+\tfrac1\varepsilon(|x'r-\pi|+|x'e-1|)$
with fully-connected topology and $k_{max}=8000$: smaller $\varepsilon$ tightens
violation penalties, but $\varepsilon=0.0001$ is too small to balance variance against
constraints (high fitness), while $\varepsilon=0.001$ concentrates the PSO cloud around
the Markowitz optimum; more particles ($N=30$) densify and speed convergence. At
$N=30$ with normalized random initialization the budget constraint holds exactly in
100% of simulations while the equality return constraint holds 0%–90% across scenarios;
relaxing the return constraint to an inequality $x'R\ge R_P$ concentrates points even
more tightly, since a particle may exceed the required return without penalty. The
PSO-built frontier nearly overlaps the exact Markowitz frontier.

**Mixed-integer mean-variance (FTSE-MIB, 2007–2008).** With 40 particles, 4000
iterations, penalty $0.0001$, $w=0.7298$, $c_1=c_2=1.49618$, $\varepsilon_i=1\%$,
$\delta_i=50\%$, $\lambda=0.5$, and $K\in\{10,20\}$: exactly $K$ assets get
$z_i\approx1$, $x_i>0$ and the rest $z_i\approx0$, $x_i\approx0$, with slight deviations
(e.g. $z_i$ marginally outside $[0,1]$) reflecting PSO's heuristic nature. The
objective value falls with $K$ (15.2432 at $K=10$ vs 11.5738 at $K=20$); the
convergence curve starts near $2\times10^5$ and stabilizes near zero after ~1200–1500
iterations; out-of-sample (Jan–Jun 2009) the $K=10$ portfolio reaches ~1.47 from a unit
base. Together the experiments show exact-penalty PSO reproduces the classical exact
solution and, in the mixed-integer case, respects cardinality and converges quickly.
