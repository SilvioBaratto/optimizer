---
type: concept
title: "Convex Optimization: Cones, Duality and KKT"
description: The convex-programming backbone of portfolio optimization — the standard form and the local-equals-global property that makes convex problems tractable, the nested tractable classes LP ⊂ QP ⊂ QCQP ⊂ SOCP with the second-order cone, Markowitz selection as a QP that loss-risk or robustness constraints lift to an SOCP, Lagrangian duality (dual function, weak/strong duality, Slater's condition), and the KKT conditions that are sufficient for a convex optimum and that interior-point methods solve.
tags: [convex-optimization, quadratic-program, socp, second-order-cone, lagrangian-duality, kkt, slater-condition, interior-point, markowitz]
sources:
  - id: openwiki-source-972a6a94017bb35add48d051
    resource: repo://docs/14_convex_optimization_cones_duality_and_kkt.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Convex Optimization: Cones, Duality and KKT

Portfolio selection reduces, at bottom, to a convex program. This page lays out the
mathematical machine that solves it: the standard form, the property that makes convex
problems tractable (every local optimum is global), the nested tractable classes, how
Markowitz [mean-variance selection](../foundations/mean-variance-selection.md) sits in
that hierarchy, and the duality and KKT theory that interior-point solvers exploit. It
contrasts with the non-convex world of
[constraints and metaheuristics](../optimization/constraints-and-metaheuristics.md) and
underpins [CVaR optimization](../optimization/cvar-optimization.md) and
[robust optimization](../optimization/robust-optimization.md).

## Standard form

An optimization problem in standard form minimizes an objective $f_0(x)$ over
$x\in\mathbf R^n$ subject to inequality constraints $f_i(x)\le0$ and equality
constraints $h_i(x)=0$. The problem's domain is the intersection of all objective and
constraint-function domains; a point is **feasible** if it satisfies every constraint,
and the problem is feasible if a feasible point exists. The **optimal value** $p^\star$
is the infimum of $f_0$ over the feasible set, with the extended-value conventions
$p^\star=+\infty$ for an infeasible problem (infimum of the empty set) and
$p^\star=-\infty$ for one unbounded below. A point $x^\star$ is optimal if it is
feasible with $f_0(x^\star)=p^\star$. A point is **locally optimal** if it minimizes
$f_0$ among nearby feasible points (within some radius $R$). The special case of a
zero objective is the **feasibility problem** — decide whether the constraints are
consistent and, if so, exhibit a satisfying point. Maximization is handled by
minimizing $-f_0$.

## Convex problems and global optimality

A **convex optimization problem** minimizes a convex objective subject to convex
inequality constraints and affine equality constraints $a_i^{\mathsf T}x=b_i$. From
these three requirements the feasible set is convex (an intersection of the convex
domain with convex sublevel sets and hyperplanes), so a convex problem minimizes a
convex objective over a convex set; a strictly convex objective admits at most one
optimizer. The cardinal tractability property is that **every locally optimal point is
globally optimal**: if a nearby-better point existed, a convex combination of it with
the local optimum would be feasible, closer than $R$, and strictly better —
contradicting local optimality. When $f_0$ is differentiable, the first-order
condition gives an explicit test: $x\in X$ is optimal iff
$\nabla f_0(x)^{\mathsf T}(y-x)\ge0$ for all feasible $y$ (reducing to
$\nabla f_0(x)=0$ when unconstrained). This local-equals-global distinction is exactly
what separates convex problems from the non-convex constraints and metaheuristics of
[constraints and metaheuristics](../optimization/constraints-and-metaheuristics.md).

## The tractable classes: LP, QP, QCQP, SOCP

Convex problems form nested classes of increasing generality:

- **Linear program (LP)** — affine objective and constraints; the feasible set is a
  polyhedron.
- **Quadratic program (QP)** — convex quadratic objective
  $\tfrac12 x^{\mathsf T}Px+q^{\mathsf T}x+r$ with $P\in\mathbf S^n_+$ over a
  polyhedron; LP is the case $P=0$.
- **Quadratically constrained QP (QCQP)** — inequality constraints are also convex
  quadratic; with $P_i\succ0$ the feasible set is an intersection of ellipsoids. QCQP
  includes QP (hence LP).
- **Second-order cone program (SOCP)** — minimizes $f^{\mathsf T}x$ subject to
  second-order cone constraints $\|A_i x+b_i\|_2\le c_i^{\mathsf T}x+d_i$, which
  require the affine map $(A_i x+b_i, c_i^{\mathsf T}x+d_i)$ to lie in the second-order
  (Lorentz / ice-cream) cone $\mathcal C_k=\{(u,t):\|u\|_2\le t\}$. As the inverse
  image of a cone under an affine map, the feasible set is convex.

The inclusions read off the SOCP parameters: $A_i=0$ reduces it to an LP, and $c_i=0$
turns each constraint into $\|A_i x+b_i\|_2\le d_i$, equivalent by squaring to a
quadratic constraint, so the SOCP reduces to a QCQP. A QCQP with $P_i\succ0$ converts
to an SOCP by completing the square,
$x^{\mathsf T}P_i x+2q_i^{\mathsf T}x+r_i=\|P_i^{1/2}x+P_i^{-1/2}q_i\|_2^2+r_i-q_i^{\mathsf T}P_i^{-1}q_i$,
turning each quadratic constraint into a conic one. Because each class is a special
case of the next, an SOCP algorithm automatically solves QCQP, QP and LP.

## Markowitz as a QP, and its lift to SOCP

The classical portfolio problem sits exactly in this hierarchy. With portfolio vector
$x$, mean price change $\bar p$ and covariance $\Sigma$, the return $r=p^{\mathsf T}x$
has mean $\bar p^{\mathsf T}x$ and variance $x^{\mathsf T}\Sigma x$, and Markowitz's
problem minimizes $x^{\mathsf T}\Sigma x$ subject to $\bar p^{\mathsf T}x\ge r_{\min}$,
$\mathbf 1^{\mathsf T}x=1$, $x\succeq0$ — a **QP**, since the objective is convex
quadratic ($\Sigma\succeq0$) and the constraints affine; its efficient frontier is
built in [mean-variance selection](../foundations/mean-variance-selection.md). Several
extensions stay QP: short positions via $x=x_{\text{long}}-x_{\text{short}}$ with a cap
$\mathbf 1^{\mathsf T}x_{\text{short}}\le\eta\,\mathbf 1^{\mathsf T}x_{\text{long}}$,
and linear transaction costs with buy/sell variables and a self-financing constraint.

The **lift to SOCP** happens when a constraint bounds the *norm* of an affine
combination of the weights. The paradigmatic case is a **loss-risk (shortfall)
constraint**: with Gaussian price changes, $r$ is Gaussian with standard deviation
$\|\Sigma^{1/2}x\|_2$, and $\mathbf{prob}(r\le\alpha)\le\beta$ becomes
$\bar p^{\mathsf T}x+\Phi^{-1}(\beta)\|\Sigma^{1/2}x\|_2\ge\alpha$, which is a
second-order cone constraint whenever $\beta\le1/2$ (so $\Phi^{-1}(\beta)\le0$).
Maximizing expected return under this bound is thus an SOCP, extendable to several
loss levels $\mathbf{prob}(r\le\alpha_i)\le\beta_i$. The same structure appears in
**robust optimization**: if each $a_i$ is only known to lie in an ellipsoid
$\mathcal E_i=\{\bar a_i+P_i u:\|u\|_2\le1\}$, the robust constraint
$\sup\{a_i^{\mathsf T}x\}=\bar a_i^{\mathsf T}x+\|P_i^{\mathsf T}x\|_2\le b_i$ is again
second-order conic, and the norm term acts as **regularization**, discouraging $x$
from large values in directions of high uncertainty — developed in
[robust optimization](../optimization/robust-optimization.md).

## Lagrangian duality

The Lagrangian augments the objective with a weighted sum of the constraint functions,
$L(x,\lambda,\nu)=f_0(x)+\sum_i\lambda_i f_i(x)+\sum_i\nu_i h_i(x)$, with **Lagrange
multipliers** (dual variables) $\lambda,\nu$. The **Lagrange dual function**
$g(\lambda,\nu)=\inf_x L(x,\lambda,\nu)$ is the pointwise infimum of a family of affine
functions of $(\lambda,\nu)$, hence **concave even when the primal is not convex**
(taking $-\infty$ when the Lagrangian is unbounded below). Its cardinal property is
that it lower-bounds the optimal value: for every $\lambda\succeq0$ and every $\nu$,
$g(\lambda,\nu)\le p^\star$, because at any feasible point the multiplier terms are
$\le0$.

Seeking the best such bound gives the **Lagrange dual problem** — maximize
$g(\lambda,\nu)$ subject to $\lambda\succeq0$ — which is always convex (maximizing a
concave function over a convex constraint), regardless of the primal. Its optimal
value $d^\star$ satisfies **weak duality** $d^\star\le p^\star$ (always, even
nonconvex, even infinite); the nonnegative difference $p^\star-d^\star$ is the
**optimal duality gap**. When $d^\star=p^\star$ the gap is zero and **strong duality**
holds. Strong duality does not hold in general but usually holds for convex problems
under a **constraint qualification**. **Slater's condition** — the existence of a
strictly feasible point $x\in\mathbf{relint}\,\mathcal D$ with $f_i(x)<0$ and $Ax=b$ —
guarantees strong duality for a convex problem (and can be relaxed so affine
inequalities need not be strict); it also ensures the dual optimum is attained when
$d^\star>-\infty$. A feasible dual point certifies primal suboptimality via the
duality gap $f_0(x)-g(\lambda,\nu)$, localizing $p^\star\in[g(\lambda,\nu),f_0(x)]$ —
the basis of non-heuristic stopping criteria.

## The Karush-Kuhn-Tucker conditions

Suppose strong duality holds with primal optimum $x^\star$ and dual optimum
$(\lambda^\star,\nu^\star)$. Chaining the equalities forces two conclusions: $x^\star$
minimizes $L(\cdot,\lambda^\star,\nu^\star)$, and **complementary slackness**
$\lambda_i^\star f_i(x^\star)=0$ holds for every $i$ (each multiplier is zero unless
its constraint is active). With differentiable functions, stationarity of the
Lagrangian gives the **KKT conditions**: primal feasibility ($f_i(x^\star)\le0$,
$h_i(x^\star)=0$), dual feasibility ($\lambda_i^\star\ge0$), complementary slackness,
and stationarity
$\nabla f_0(x^\star)+\sum_i\lambda_i^\star\nabla f_i(x^\star)+\sum_i\nu_i^\star\nabla h_i(x^\star)=0$.
For *any* differentiable problem with strong duality, every primal-dual optimal pair
satisfies KKT. (Historically due to Kuhn and Tucker (1951), and earlier to Karush's
1939 thesis.)

The decisive fact for convex optimization: **when the primal is convex, the KKT
conditions are also sufficient** for optimality. If the $f_i$ are convex, the $h_i$
affine, and $\tilde x,\tilde\lambda,\tilde\nu$ satisfy KKT, then $\tilde x$ is primal
optimal and $(\tilde\lambda,\tilde\nu)$ dual optimal with zero gap — because
$\tilde\lambda_i\ge0$ makes the Lagrangian convex in $x$, stationarity makes
$\tilde x$ its minimizer, and complementary slackness plus $h_i(\tilde x)=0$ collapse
$g(\tilde\lambda,\tilde\nu)$ to $f_0(\tilde x)$. Under Slater's condition the KKT
conditions become necessary and sufficient. Occasionally they solve in closed form:
equality-constrained convex quadratic minimization reduces to the linear KKT system
$\begin{psmallmatrix}P&A^{\mathsf T}\\A&0\end{psmallmatrix}\begin{psmallmatrix}x^\star\\\nu^\star\end{psmallmatrix}=\begin{psmallmatrix}-q\\b\end{psmallmatrix}$;
in general they have no analytic solution, and many convex algorithms are methods for
solving the KKT system.

## From convexity to solution: interior-point methods

The thread closes here. Convexity guarantees (via local-equals-global) that a found
solution is *global*, and the same convexity, through Slater's condition and strong
duality, makes KKT necessary and sufficient — furnishing both the system an algorithm
must solve and a verifiable optimality certificate. **Interior-point methods** solve
this system for the SOCP class (hence QCQP, QP, LP). SOCP has its own duality theory:
the dual is again an SOCP, weak duality follows from a nonnegative Cauchy-Schwarz gap,
and strong duality holds if primal or dual is strictly feasible. The second-order cone
carries a smooth convex **barrier** $\phi(u,t)=-\log(t^2-u^{\mathsf T}u)$, diverging at
the cone boundary, on which a **primal-dual potential-reduction method** (a
specialization of Nesterov-Nemirovskii) is built: each iteration solves a linear
system for search directions and reduces the potential by a guaranteed amount.
Worst-case iteration count grows at most as the square root of problem size, while in
practice the typical count is **5 to 50, almost independent of problem size**. This
combination — convexity assuring globality, interior-point methods reaching it
reliably — is why optimization libraries return the global solution of LP, QP, QCQP and
SOCP portfolio problems.
