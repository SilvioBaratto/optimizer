---
title: "Convex Optimization - Cones, Duality, and KKT"
chapter: 14
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-11
---

> [!abstract] Summary
> The chapter presents the mathematical machinery that solves portfolio selection problems: the standard form of an optimization problem, the notion of a convex problem, and the pivotal property that every local optimum is global. It orders by increasing generality the tractable classes — LP, QP, QCQP, and SOCP with the second-order cone — showing that Markowitz's mean-variance selection is a QP and that constraints on the norm of returns or weights elevate it to an SOCP. It then develops Lagrangian duality (dual function, weak and strong duality, Slater's condition) and the Karush-Kuhn-Tucker conditions, which for a convex problem are sufficient for optimality and constitute the system that interior-point methods solve to find the global optimum.

## The Standard Form of an Optimization Problem

We adopt the notation

$$\begin{array}{ll}\text{minimize} & f_0(x)\\ \text{subject to} & f_i(x)\le 0,\quad i=1,\dots,m\\ & h_i(x)=0,\quad i=1,\dots,p,\end{array}$$

to describe the problem of finding an $x$ that minimizes $f_0(x)$ among all $x$ satisfying the conditions $f_i(x)\le 0$ and $h_i(x)=0$ [BoydVandenberghe2004]. We call $x\in\mathbf{R}^n$ the *optimization variable* and $f_0:\mathbf{R}^n\to\mathbf{R}$ the *objective function* (or cost function). The inequalities $f_i(x)\le 0$ are the *inequality constraints*, with the corresponding *inequality constraint functions* $f_i$; the equations $h_i(x)=0$ are the *equality constraints*, with functions $h_i$. If there are no constraints ($m=p=0$) the problem is called *unconstrained*.

The set of points at which the objective and constraint functions are all defined,

$$\mathcal{D}=\bigcap_{i=0}^{m}\mathbf{dom}\,f_i\ \cap\ \bigcap_{i=1}^{p}\mathbf{dom}\,h_i,$$

is the *domain* of the problem. A point $x\in\mathcal{D}$ is *feasible* if it satisfies all the constraints; the problem is feasible if at least one feasible point exists, and *infeasible* otherwise. The set of all feasible points is the *feasible set* (or constraint set).

The *optimal value* $p^\star$ is defined as

$$p^\star=\inf\{f_0(x)\mid f_i(x)\le 0,\ i=1,\dots,m,\ h_i(x)=0,\ i=1,\dots,p\}.$$

We allow $p^\star$ to take the extended values $\pm\infty$: if the problem is infeasible then $p^\star=\infty$ (the infimum of the empty set is $\infty$); if there are feasible points $x_k$ with $f_0(x_k)\to-\infty$ the problem is *unbounded below* and $p^\star=-\infty$. We say $x^\star$ is an *optimal point*, or that it solves the problem, if $x^\star$ is feasible and $f_0(x^\star)=p^\star$ [BoydVandenberghe2004].

A feasible point $x$ is *locally optimal* if there exists $R>0$ such that $x$ solves

$$\begin{array}{ll}\text{minimize} & f_0(z)\\ \text{subject to} & f_i(z)\le 0,\ i=1,\dots,m,\quad h_i(z)=0,\ i=1,\dots,p\\ & \|z-x\|_2\le R,\end{array}$$

that is, if $x$ minimizes $f_0$ among nearby feasible points. The term *globally optimal* distinguishes, when needed, optimality over the entire feasible set from merely local optimality.

An important special case is one in which the objective function is identically zero: the optimal value is then $0$ (if the feasible set is nonempty) or $\infty$ (if it is empty). This is the *feasibility problem*, written as

$$\begin{array}{ll}\text{find} & x\\ \text{subject to} & f_i(x)\le 0,\ i=1,\dots,m,\quad h_i(x)=0,\ i=1,\dots,p,\end{array}$$

and consists of determining whether the constraints are consistent and, if so, exhibiting a point that satisfies them [BoydVandenberghe2004]. A maximization problem, $\text{maximize}\ f_0(x)$ subject to the same constraints, is by convention handled by minimizing $-f_0$; for it the optimal value is $p^\star=\sup\{f_0(x)\mid x\ \text{feasible}\}$.

## The Convex Problem and the Globality of Optima

A *convex optimization problem* has the form

$$\begin{array}{ll}\text{minimize} & f_0(x)\\ \text{subject to} & f_i(x)\le 0,\quad i=1,\dots,m\\ & a_i^{\mathsf T}x=b_i,\quad i=1,\dots,p,\end{array}$$

where $f_0,\dots,f_m$ are convex functions. Relative to the general standard form, the convex problem imposes three additional requirements [BoydVandenberghe2004]:

- the objective function must be convex;
- the inequality constraint functions must be convex;
- the equality constraint functions $h_i(x)=a_i^{\mathsf T}x-b_i$ must be affine.

A fundamental property follows immediately from these requirements: the feasible set of a convex problem is convex, since it is the intersection of the domain $\mathcal{D}=\bigcap_{i=0}^m\mathbf{dom}\,f_i$ — a convex set — with the $m$ convex sublevel sets $\{x\mid f_i(x)\le 0\}$ and with the $p$ hyperplanes $\{x\mid a_i^{\mathsf T}x=b_i\}$. In a convex problem, therefore, a convex objective function is minimized over a convex set. If the objective is strictly convex, the optimal set contains at most one point [BoydVandenberghe2004].

The property that makes these problems tractable is that **every locally optimal point is also globally optimal**. The proof is direct. Suppose $x$ is locally optimal, that is, feasible and such that

$$f_0(x)=\inf\{f_0(z)\mid z\ \text{feasible},\ \|z-x\|_2\le R\}$$

for some $R>0$. If $x$ were not globally optimal, there would exist a feasible point $y$ with $f_0(y)<f_0(x)$; necessarily $\|y-x\|_2>R$, otherwise $f_0(x)\le f_0(y)$ would hold. Consider then

$$z=(1-\theta)x+\theta y,\qquad \theta=\frac{R}{2\|y-x\|_2}.$$

We have $\|z-x\|_2=R/2<R$ and, by convexity of the feasible set, $z$ is feasible. By convexity of $f_0$,

$$f_0(z)\le(1-\theta)f_0(x)+\theta f_0(y)<f_0(x),$$

which contradicts the local optimality of $x$. There is therefore no feasible $y$ with $f_0(y)<f_0(x)$, that is, $x$ is globally optimal [BoydVandenberghe2004].

When the objective $f_0$ is differentiable, the first-order inequality $f_0(y)\ge f_0(x)+\nabla f_0(x)^{\mathsf T}(y-x)$ provides an explicit optimality criterion. Letting $X$ be the feasible set, $x$ is optimal if and only if $x\in X$ and

$$\nabla f_0(x)^{\mathsf T}(y-x)\ge 0\quad\text{for every }y\in X.$$

Geometrically, if $\nabla f_0(x)\neq 0$, the vector $-\nabla f_0(x)$ defines a supporting hyperplane to the feasible set at $x$. In the unconstrained case the condition reduces to the well-known necessary and sufficient condition $\nabla f_0(x)=0$ [BoydVandenberghe2004]. This distinction between local and global optimum is precisely what separates convex problems from the non-convex-constrained problems or metaheuristics discussed in [[06 Vincoli e metaeuristiche]].

## The Tractable Classes: LP, QP, QCQP, and SOCP

Convex problems are organized into classes of increasing generality, identified by the form of the objective and of the constraints.

**Linear Programming (LP).** When the objective and constraints are all affine, the problem is called a *linear program*:

$$\begin{array}{ll}\text{minimize} & c^{\mathsf T}x+d\\ \text{subject to} & Gx\preceq h\\ & Ax=b,\end{array}$$

with $G\in\mathbf{R}^{m\times n}$ and $A\in\mathbf{R}^{p\times n}$. The feasible set is a polyhedron $\mathcal{P}$ and the problem consists in minimizing an affine function over $\mathcal{P}$; linear programs are, obviously, convex problems [BoydVandenberghe2004].

**Quadratic Programming (QP).** The convex problem is a *quadratic program* if the objective is (convex) quadratic and the constraints are affine:

$$\begin{array}{ll}\text{minimize} & \tfrac12 x^{\mathsf T}Px+q^{\mathsf T}x+r\\ \text{subject to} & Gx\preceq h\\ & Ax=b,\end{array}$$

with $P\in\mathbf{S}^n_+$. In a QP a convex quadratic function is minimized over a polyhedron. Linear programs are a special case, obtained by setting $P=0$ [BoydVandenberghe2004].

**Quadratically Constrained Quadratic Programming (QCQP).** If the inequality constraint functions are also (convex) quadratic,

$$\begin{array}{ll}\text{minimize} & \tfrac12 x^{\mathsf T}P_0x+q_0^{\mathsf T}x+r_0\\ \text{subject to} & \tfrac12 x^{\mathsf T}P_ix+q_i^{\mathsf T}x+r_i\le 0,\quad i=1,\dots,m\\ & Ax=b,\end{array}$$

with $P_i\in\mathbf{S}^n_+$, the problem is a *quadratically constrained quadratic program*. When $P_i\succ 0$ the feasible set is the intersection of ellipsoids. QCQPs include QPs (with $P_i=0$ for $i\ge 1$), and hence also LPs, as a special case [BoydVandenberghe2004].

**Second-Order Cone Programming (SOCP).** The *second-order cone program* has the form

$$\begin{array}{ll}\text{minimize} & f^{\mathsf T}x\\ \text{subject to} & \|A_ix+b_i\|_2\le c_i^{\mathsf T}x+d_i,\quad i=1,\dots,N\\ & Fx=g,\end{array}$$

with $x\in\mathbf{R}^n$, $A_i\in\mathbf{R}^{(n_i-1)\times n}$, and $F\in\mathbf{R}^{p\times n}$ [LoboEtAl1998; BoydVandenberghe2004]. A constraint of the form $\|A_ix+b_i\|_2\le c_i^{\mathsf T}x+d_i$ is called a *second-order cone constraint*, because it is equivalent to requiring that the affine function $(A_ix+b_i,\,c_i^{\mathsf T}x+d_i)$ belong to the second-order cone. The *second-order cone* (also called the quadratic cone, ice-cream cone, or Lorentz cone) of dimension $k$ is defined by

$$\mathcal{C}_k=\left\{\begin{bmatrix}u\\ t\end{bmatrix}\ \middle|\ u\in\mathbf{R}^{k-1},\ t\in\mathbf{R},\ \|u\|_2\le t\right\},$$

and, since

$$\|A_ix+b_i\|_2\le c_i^{\mathsf T}x+d_i\iff\begin{bmatrix}A_i\\ c_i^{\mathsf T}\end{bmatrix}x+\begin{bmatrix}b_i\\ d_i\end{bmatrix}\in\mathcal{C}_{n_i},$$

the set of points satisfying the constraint is the inverse image of the unit cone under an affine map, hence convex: the SOCP is a convex problem [LoboEtAl1998].

The inclusion relations among the classes can be read directly off the parameters of the SOCP. If $A_i=0$ for every $i$, the SOCP reduces to an LP; if $c_i=0$ for every $i$, each constraint becomes $\|A_ix+b_i\|_2\le d_i$, equivalent by squaring to the quadratic constraint $\|A_ix+b_i\|_2^2\le d_i^2$, and the SOCP reduces to a QCQP [LoboEtAl1998]. In general, second-order cone programs are more general than QCQPs (and hence than LPs). The explicit reduction of a QCQP to SOCP, when $P_i\succ 0$, exploits the completing-the-square identity — adopting here, as in [LoboEtAl1998], the normalization $x^{\mathsf T}P_ix+2q_i^{\mathsf T}x+r_i\le 0$ of the constraint parameters, with linear coefficient $2q_i$ and without the $\tfrac12$ factor used above —

$$x^{\mathsf T}P_ix+2q_i^{\mathsf T}x+r_i=\big\|P_i^{1/2}x+P_i^{-1/2}q_i\big\|_2^2+r_i-q_i^{\mathsf T}P_i^{-1}q_i,$$

so that such a quadratic constraint $x^{\mathsf T}P_ix+2q_i^{\mathsf T}x+r_i\le 0$ becomes the conic constraint

$$\big\|P_i^{1/2}x+P_i^{-1/2}q_i\big\|_2\le\big(q_i^{\mathsf T}P_i^{-1}q_i-r_i\big)^{1/2},$$

and the entire QCQP is rewritten as an SOCP by introducing an auxiliary variable $t$ and minimizing $t$ subject to $\|P_0^{1/2}x+P_0^{-1/2}q_0\|_2\le t$ and the constraints above [LoboEtAl1998]. Since each class is a special case of the next, an algorithm for SOCP automatically solves QCQP, QP, and LP.

## The Markowitz Problem as a QP and Its Elevation to SOCP

The classical portfolio selection problem sits exactly within this hierarchy. Consider $n$ assets held over a period, with portfolio vector $x\in\mathbf{R}^n$; the relative price change is modeled as a random vector $p$ with known mean $\bar p$ and covariance $\Sigma$, so that the return $r=p^{\mathsf T}x$ has mean $\bar p^{\mathsf T}x$ and variance $x^{\mathsf T}\Sigma x$. The choice of $x$ involves a trade-off between the mean return and its variance. The portfolio optimization problem introduced by Markowitz is the QP

$$\begin{array}{ll}\text{minimize} & x^{\mathsf T}\Sigma x\\ \text{subject to} & \bar p^{\mathsf T}x\ge r_{\min}\\ & \mathbf 1^{\mathsf T}x=1,\quad x\succeq 0,\end{array}$$

in which one seeks the portfolio that minimizes the variance of the return (associated with the portfolio's *risk*) while guaranteeing an acceptable mean return $r_{\min}$ and respecting the budget constraint and the absence of short positions [BoydVandenberghe2004; Markowitz1952]. The construction of the efficient frontier for this formulation is the subject of [[02 Selezione media-varianza]]. The objective is convex quadratic (since $\Sigma$ is a covariance matrix, hence positive semi-definite) and the constraints are affine: this is a quadratic program.

The formulation is enriched with extensions that remain within the QP class: allowing short positions is achieved by setting $x=x_{\text{long}}-x_{\text{short}}$, with $x_{\text{long}}\succeq 0$, $x_{\text{short}}\succeq 0$, and $\mathbf 1^{\mathsf T}x_{\text{short}}\le\eta\,\mathbf 1^{\mathsf T}x_{\text{long}}$, which limits the total short position to a fraction $\eta$ of the long position; the inclusion of linear transaction costs, with variables $u_{\text{buy}},u_{\text{sell}}\succeq 0$ and a self-financing constraint $(1-f_{\text{sell}})\mathbf 1^{\mathsf T}u_{\text{sell}}=(1+f_{\text{buy}})\mathbf 1^{\mathsf T}u_{\text{buy}}$, keeps the problem a QP in the variables $x,u_{\text{buy}},u_{\text{sell}}$ [BoydVandenberghe2004].

The elevation to SOCP instead occurs when constraints are imposed on the *norm* of an affine combination of the weights. The paradigmatic case is the loss-risk constraint. Assuming the price change $p$ is Gaussian, the return $r=p^{\mathsf T}x$ is a Gaussian random variable with mean $\bar p^{\mathsf T}x$ and variance $\sigma_r^2=x^{\mathsf T}\Sigma x$; a *loss-risk constraint* of the form

$$\mathbf{prob}(r\le\alpha)\le\beta,$$

with $\alpha$ an undesired return level and $\beta$ a maximum probability, is expressed via the standard Gaussian cumulative distribution function $\Phi$ as

$$\bar p^{\mathsf T}x+\Phi^{-1}(\beta)\,\|\Sigma^{1/2}x\|_2\ge\alpha.$$

When $\beta\le 1/2$, $\Phi^{-1}(\beta)\le 0$, and this is a second-order cone constraint [BoydVandenberghe2004; LoboEtAl1998]. The problem of maximizing expected return subject to a loss-risk limit can therefore be posed as an SOCP with a second-order cone constraint,

$$\begin{array}{ll}\text{maximize} & \bar p^{\mathsf T}x\\ \text{subject to} & \bar p^{\mathsf T}x+\Phi^{-1}(\beta)\,\|\Sigma^{1/2}x\|_2\ge\alpha\\ & x\succeq 0,\quad \mathbf 1^{\mathsf T}x=1,\end{array}$$

and admits the extension to multiple loss constraints $\mathbf{prob}(r\le\alpha_i)\le\beta_i$ (with $\beta_i\le 1/2$), which express the risks tolerated at different loss levels [LoboEtAl1998]. The norm term $\|\Sigma^{1/2}x\|_2$ is precisely the standard deviation of the return; imposing a limit on it, or more generally imposing a limit on a norm of the weights, produces second-order cone constraints. The same structure emerges in robust optimization, where uncertainty about the parameters $a_i$, assumed to belong to ellipsoids $\mathcal{E}_i=\{\bar a_i+P_iu\mid\|u\|_2\le 1\}$, transforms a linear constraint $a_i^{\mathsf T}x\le b_i$ into the robust constraint

$$\sup\{a_i^{\mathsf T}x\mid a_i\in\mathcal{E}_i\}=\bar a_i^{\mathsf T}x+\|P_i^{\mathsf T}x\|_2\le b_i,$$

also second-order conic: the robust linear program becomes an SOCP, in which the norm terms act as *regularization*, discouraging $x$ from taking large values in the directions of greatest uncertainty [LoboEtAl1998]. Robust formulations with uncertainty sets are developed in [[15 Ottimizzazione robusta e insiemi di incertezza]].

## Lagrangian Duality

The basic idea of Lagrangian duality is to account for the constraints by augmenting the objective function with a weighted sum of the constraint functions. To the problem in standard form one associates the *Lagrangian* $L:\mathbf{R}^n\times\mathbf{R}^m\times\mathbf{R}^p\to\mathbf{R}$,

$$L(x,\lambda,\nu)=f_0(x)+\sum_{i=1}^m\lambda_i f_i(x)+\sum_{i=1}^p\nu_i h_i(x),$$

where $\lambda_i$ is the *Lagrange multiplier* associated with the constraint $f_i(x)\le 0$ and $\nu_i$ the one associated with the constraint $h_i(x)=0$. The vectors $\lambda$ and $\nu$ are called *dual variables* [BoydVandenberghe2004].

The *Lagrange dual function* $g:\mathbf{R}^m\times\mathbf{R}^p\to\mathbf{R}$ is the minimum of the Lagrangian with respect to $x$:

$$g(\lambda,\nu)=\inf_{x\in\mathcal{D}}L(x,\lambda,\nu)=\inf_{x\in\mathcal{D}}\Big(f_0(x)+\sum_{i=1}^m\lambda_i f_i(x)+\sum_{i=1}^p\nu_i h_i(x)\Big).$$

When the Lagrangian is unbounded below in $x$, the dual function is $-\infty$. Being the pointwise infimum of a family of affine functions of $(\lambda,\nu)$, the dual function is *concave*, even when the original problem is not convex [BoydVandenberghe2004].

The pivotal property is that the dual function provides lower bounds on the optimal value $p^\star$: for every $\lambda\succeq 0$ and every $\nu$,

$$g(\lambda,\nu)\le p^\star.$$

The verification is immediate. If $\tilde x$ is feasible, that is, $f_i(\tilde x)\le 0$ and $h_i(\tilde x)=0$, and $\lambda\succeq 0$, then $\sum_i\lambda_i f_i(\tilde x)+\sum_i\nu_i h_i(\tilde x)\le 0$, since every term in the first sum is non-positive and those in the second are zero; hence

$$g(\lambda,\nu)=\inf_{x\in\mathcal{D}}L(x,\lambda,\nu)\le L(\tilde x,\lambda,\nu)\le f_0(\tilde x).$$

Since this holds for every feasible point $\tilde x$, it follows that $g(\lambda,\nu)\le p^\star$ [BoydVandenberghe2004].

It is natural to ask what the *best* lower bound obtainable from the dual function is. This leads to the *Lagrange dual problem*

$$\begin{array}{ll}\text{maximize} & g(\lambda,\nu)\\ \text{subject to} & \lambda\succeq 0,\end{array}$$

in whose context the original problem is called the *primal problem*. The dual problem is always a convex problem — a concave function is maximized over a convex constraint — regardless of whether the primal is convex [BoydVandenberghe2004].

Letting $d^\star$ be the optimal value of the dual problem, the inequality $g(\lambda,\nu)\le p^\star$ implies

$$d^\star\le p^\star,$$

called *weak duality*: it always holds, even if the primal is not convex, and even when the two values are infinite. The difference $p^\star-d^\star$, always non-negative, is the *optimal duality gap*: it is the gap between the optimal value of the primal and the best lower bound obtainable from the dual function [BoydVandenberghe2004].

When the equality

$$d^\star=p^\star$$

holds, that is, when the optimal duality gap is zero, *strong duality* is said to hold: the best bound obtainable from the dual function is exact. Strong duality does not hold in general, but if the primal problem is convex it usually does hold. There are numerous conditions on the problem, beyond convexity, that guarantee it, called *constraint qualifications*. A simple qualification is *Slater's condition*: there exists a point $x\in\mathbf{relint}\,\mathcal{D}$ such that

$$f_i(x)<0,\ i=1,\dots,m,\qquad Ax=b.$$

Such a point is called *strictly feasible*. Slater's theorem states that strong duality holds if Slater's condition is satisfied and the problem is convex [BoydVandenberghe2004; Slater1950]. The condition can be refined when some constraint functions $f_i$ are affine: if the first $k$ functions are affine, it suffices that there exist $x\in\mathbf{relint}\,\mathcal{D}$ with

$$f_i(x)\le 0,\ i=1,\dots,k,\qquad f_i(x)<0,\ i=k+1,\dots,m,\qquad Ax=b,$$

that is, the affine inequalities need not hold strictly. Slater's condition, besides implying strong duality for convex problems, guarantees that the dual optimum is attained when $d^\star>-\infty$, that is, that there exists a feasible dual pair $(\lambda^\star,\nu^\star)$ with $g(\lambda^\star,\nu^\star)=d^\star=p^\star$ [BoydVandenberghe2004].

A feasible dual point $(\lambda,\nu)$ moreover provides a *certificate* of the sub-optimality of a feasible primal point $x$: we have $f_0(x)-p^\star\le f_0(x)-g(\lambda,\nu)$, so that the quantity $f_0(x)-g(\lambda,\nu)$, called the *duality gap* associated with the pair $x,(\lambda,\nu)$, localizes the optimal value in the interval $p^\star\in[g(\lambda,\nu),f_0(x)]$. If the gap vanishes, $x$ is primal optimal and $(\lambda,\nu)$ is dual optimal [BoydVandenberghe2004]. This property underlies the non-heuristic stopping criteria of algorithms.

## The Karush-Kuhn-Tucker Conditions

Suppose the primal and dual values are attained and equal, that is, strong duality holds, and let $x^\star$ be a primal optimal point and $(\lambda^\star,\nu^\star)$ a dual optimal point. Then

$$f_0(x^\star)=g(\lambda^\star,\nu^\star)=\inf_x\Big(f_0(x)+\sum_{i=1}^m\lambda_i^\star f_i(x)+\sum_{i=1}^p\nu_i^\star h_i(x)\Big)\le f_0(x^\star)+\sum_{i=1}^m\lambda_i^\star f_i(x^\star)+\sum_{i=1}^p\nu_i^\star h_i(x^\star)\le f_0(x^\star).$$

The first equality expresses the vanishing of the duality gap; the second is the definition of the dual function; the second-to-last inequality follows because the infimum of the Lagrangian is less than or equal to its value at $x^\star$; the last follows from $\lambda_i^\star\ge 0$, $f_i(x^\star)\le 0$, and $h_i(x^\star)=0$. The two inequalities therefore hold with equality [BoydVandenberghe2004].

Two conclusions follow from this chain. First, $x^\star$ minimizes $L(x,\lambda^\star,\nu^\star)$ with respect to $x$. Second, $\sum_{i=1}^m\lambda_i^\star f_i(x^\star)=0$; since every term is non-positive, each of them must vanish:

$$\lambda_i^\star f_i(x^\star)=0,\qquad i=1,\dots,m.$$

This condition is called *complementary slackness*: it holds for every primal-dual optimal pair when strong duality holds. Equivalently, $\lambda_i^\star>0\Rightarrow f_i(x^\star)=0$ and $f_i(x^\star)<0\Rightarrow\lambda_i^\star=0$; the $i$-th optimal multiplier is zero unless the $i$-th constraint is active at the optimum [BoydVandenberghe2004].

Now suppose $f_0,\dots,f_m,h_1,\dots,h_p$ are differentiable (with no convexity assumption for now). Since $x^\star$ minimizes the Lagrangian with respect to $x$, its gradient vanishes at $x^\star$. Combining this with the preceding conditions yields the *Karush-Kuhn-Tucker* (KKT) conditions:

$$\begin{aligned} f_i(x^\star)&\le 0, & i&=1,\dots,m &&\text{(primal feasibility)}\\ h_i(x^\star)&=0, & i&=1,\dots,p &&\text{(primal feasibility)}\\ \lambda_i^\star&\ge 0, & i&=1,\dots,m &&\text{(dual feasibility)}\\ \lambda_i^\star f_i(x^\star)&=0, & i&=1,\dots,m &&\text{(complementary slackness)}\\ \nabla f_0(x^\star)+\sum_{i=1}^m\lambda_i^\star\nabla f_i(x^\star)+\sum_{i=1}^p\nu_i^\star\nabla h_i(x^\star)&=0 &&&&\text{(stationarity)}. \end{aligned}$$

For *any* optimization problem with differentiable objective and constraints for which strong duality holds, every primal-dual optimal pair must satisfy the KKT conditions [BoydVandenberghe2004]. Historically these conditions were presented by Kuhn and Tucker [KuhnTucker1951], and it was later discovered that they had already been obtained by Karush in his 1939 thesis [Karush1939].

The decisive fact for convex optimization is that, **when the primal problem is convex, the KKT conditions are also sufficient** for optimality. Precisely, if the $f_i$ are convex, the $h_i$ are affine, and $\tilde x,\tilde\lambda,\tilde\nu$ are any points satisfying

$$\begin{aligned} f_i(\tilde x)&\le 0, & h_i(\tilde x)&=0,\\ \tilde\lambda_i&\ge 0, & \tilde\lambda_i f_i(\tilde x)&=0,\\ \nabla f_0(\tilde x)+\sum_{i=1}^m\tilde\lambda_i\nabla f_i(\tilde x)&+\sum_{i=1}^p\tilde\nu_i\nabla h_i(\tilde x)=0, \end{aligned}$$

then $\tilde x$ and $(\tilde\lambda,\tilde\nu)$ are primal and dual optimal, with zero duality gap. The proof observes that the first two conditions ensure the primal feasibility of $\tilde x$; since $\tilde\lambda_i\ge 0$, the Lagrangian $L(x,\tilde\lambda,\tilde\nu)$ is convex in $x$, and the stationarity condition says its gradient vanishes at $\tilde x$, so $\tilde x$ minimizes $L(\cdot,\tilde\lambda,\tilde\nu)$. It follows that

$$g(\tilde\lambda,\tilde\nu)=L(\tilde x,\tilde\lambda,\tilde\nu)=f_0(\tilde x)+\sum_{i=1}^m\tilde\lambda_i f_i(\tilde x)+\sum_{i=1}^p\tilde\nu_i h_i(\tilde x)=f_0(\tilde x),$$

where in the last step $h_i(\tilde x)=0$ and $\tilde\lambda_i f_i(\tilde x)=0$ were used. The coincidence $g(\tilde\lambda,\tilde\nu)=f_0(\tilde x)$ means zero duality gap, and hence optimality [BoydVandenberghe2004]. If the convex problem also satisfies Slater's condition, the KKT conditions become necessary and sufficient: Slater guarantees a zero gap and attainment of the dual optimum, so that $x$ is optimal if and only if there exists $(\lambda,\nu)$ that together with it satisfies the KKT conditions [BoydVandenberghe2004].

In some cases the KKT conditions are solved in closed form. For convex quadratic minimization with equality constraints,

$$\begin{array}{ll}\text{minimize} & \tfrac12 x^{\mathsf T}Px+q^{\mathsf T}x+r\\ \text{subject to} & Ax=b,\end{array}\qquad P\in\mathbf{S}^n_+,$$

the KKT conditions are $Ax^\star=b$ and $Px^\star+q+A^{\mathsf T}\nu^\star=0$, that is, the linear system

$$\begin{bmatrix}P & A^{\mathsf T}\\ A & 0\end{bmatrix}\begin{bmatrix}x^\star\\ \nu^\star\end{bmatrix}=\begin{bmatrix}-q\\ b\end{bmatrix},$$

whose solution provides the optimal primal and dual variables [BoydVandenberghe2004]. In most cases, however, the KKT conditions do not admit an analytical solution, and many convex optimization algorithms are designed — or can be interpreted — as methods for solving the KKT conditions.

## From Convexity to Solution: Interior-Point Methods

The thread running through the chapter closes here. It is the convexity of the problem — convex objective, convex inequality constraints, affine equality constraints — that guarantees, through the globality property of local optima, that a solution found is a *global* solution; and it is this same convexity that, via Slater's condition and strong duality, makes the KKT conditions necessary and sufficient for optimality, thereby providing the system an algorithm must solve and a verifiable certificate of optimality.

The algorithms that solve this system for the SOCP class — and hence for QCQP, QP, and LP — are the *interior-point methods*. For SOCP a corresponding duality theory is available: the dual of the SOCP is itself a second-order cone problem, weak duality $p^\star\ge d^\star$ follows from the non-negativity of the duality gap $\eta(x,z,w)=\sum_i(z_i^{\mathsf T}u_i+w_i t_i)$ — each term non-negative by the Cauchy-Schwarz inequality — and strong duality $p^\star=d^\star$ holds if the primal or the dual is strictly feasible [LoboEtAl1998]. To the second-order cone $\mathcal{C}_m$ one associates the *barrier function*

$$\phi(u,t)=\begin{cases}-\log\!\big(t^2-u^{\mathsf T}u\big), & \|u\|_2<t\\ \infty, & \text{otherwise},\end{cases}$$

smooth and convex inside the cone and diverging at the boundary, on which a *primal-dual potential-reduction method* is built, a specialization to SOCP of the method of Nesterov and Nemirovskii [LoboEtAl1998; NesterovNemirovskii1994]. At each iteration the method computes primal and dual search directions by solving a linear system and updates the points by reducing the potential function by a guaranteed amount; the linear system solved is the direct analogue of those arising in interior-point methods for LP [LoboEtAl1998]. We do not develop the algorithm here, reporting only its documented performance: worst-case theoretical analysis shows that the number of iterations required grows at most as the square root of the problem's dimension, while numerical experiments indicate that the typical number of iterations lies between 5 and 50, almost independently of the problem's size [LoboEtAl1998]. It is by virtue of this combination — convexity ensuring globality and interior-point methods reliably achieving it — that optimization libraries solve portfolio selection problems in the LP, QP, QCQP, and SOCP classes, returning the global solution.

## References

- **[BoydVandenberghe2004]** Boyd, S. & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press, Cambridge. (Ch. 4, "Convex optimization problems," and Ch. 5, "Duality.")
- **[Karush1939]** Karush, W. (1939). Minima of Functions of Several Variables with Inequalities as Side Constraints. Master's thesis, Department of Mathematics, University of Chicago.
- **[KuhnTucker1951]** Kuhn, H. W. & Tucker, A. W. (1951). Nonlinear programming. In Proceedings of the Second Berkeley Symposium on Mathematical Statistics and Probability (pp. 481-492). University of California Press, Berkeley.
- **[LoboEtAl1998]** Lobo, M. S., Vandenberghe, L., Boyd, S. & Lebret, H. (1998). Applications of second-order cone programming. Linear Algebra and its Applications, 284(1-3), 193-228.
- **[Markowitz1952]** Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
- **[NesterovNemirovskii1994]** Nesterov, Y. & Nemirovskii, A. (1994). Interior-Point Polynomial Algorithms in Convex Programming. SIAM Studies in Applied Mathematics, vol. 13. SIAM, Philadelphia.
- **[Slater1950]** Slater, M. (1950). Lagrange Multipliers Revisited: A Contribution to Non-Linear Programming. Cowles Commission Discussion Paper, Math. 403, University of Chicago.
