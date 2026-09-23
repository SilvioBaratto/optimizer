---
type: concept
title: "CVaR Optimization: The Rockafellar-Uryasev Formulation"
description: How Rockafellar and Uryasev turn Conditional Value-at-Risk from a risk measure into a minimizable objective — a single auxiliary function, convex in the weights and a threshold variable, whose joint minimization returns VaR (the optimal threshold) and CVaR (the minimum value) at once, reduces to a linear program over scenarios for any loss distribution, and remains a coherent risk measure even for general, discontinuous losses.
tags: [cvar, value-at-risk, rockafellar-uryasev, linear-program, coherent-risk, expected-shortfall, tail-distribution, scenario-optimization, convexity]
sources:
  - id: openwiki-source-f093b35e056bbc41b00ba616
    resource: repo://docs/16_cvar_optimization_rockafellar_uryasev_formulation.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# CVaR Optimization: The Rockafellar-Uryasev Formulation

In [coherent risk measures](../risk-measures/coherent-risk-measures.md) the Conditional
Value-at-Risk (CVaR) is introduced as a risk measure; here it becomes the quantity to
minimize in portfolio choice. The starting point, due to Rockafellar and Uryasev, is
that Value-at-Risk (VaR) — despite its regulatory status — has undesirable mathematics:
it lacks subadditivity and convexity, is coherent only when based on the standard
deviation of normal distributions, and, computed from scenarios, is ill-behaved as a
function of positions with multiple local extrema, obstructing optimization. CVaR by
contrast is a coherent measure with translation-equivariance, positive homogeneity,
convexity and monotonicity with respect to stochastic dominance. This page builds on
[convex optimization](../optimization/convex-optimization.md) and connects to the
[drawdown risk measures](../risk-measures/drawdown-risk-measures.md) that share its
computational form.

## From CVaR as a measure to CVaR as an objective

For a decision (portfolio) $\mathbf w\in X\subset\mathbb R^n$ and a random vector
$\mathbf y$ with density $p(\mathbf y)$, each pair $(\mathbf w,\mathbf y)$ carries a
loss $f(\mathbf w,\mathbf y)$ (possibly negative, i.e. a gain). The distribution
function of the loss is $\Psi(\mathbf w,\alpha)=\int_{f(\mathbf w,\mathbf y)\le\alpha}p(\mathbf y)\,d\mathbf y$.
For a probability level $\beta\in(0,1)$ (typically 0.90, 0.95, 0.99), the **$\beta$-VaR**
is the smallest threshold $\alpha_\beta(\mathbf w)=\min\{\alpha:\Psi(\mathbf w,\alpha)\ge\beta\}$,
and the **$\beta$-CVaR** is the conditional expectation of the loss in the tail beyond
it,
$\phi_\beta(\mathbf w)=(1-\beta)^{-1}\int_{f\ge\alpha_\beta}f(\mathbf w,\mathbf y)p(\mathbf y)\,d\mathbf y$.
Under a first, simplifying continuity assumption on $\Psi$, the tail has probability
exactly $1-\beta$, and by construction $\beta$-VaR never exceeds $\beta$-CVaR — so
low-CVaR portfolios necessarily have low VaR too.

## The Rockafellar-Uryasev auxiliary function

Working directly with $\phi_\beta$ is hard because its definition depends on the
often ill-behaved $\alpha_\beta(\mathbf w)$. The central contribution is to characterize
both through a single auxiliary function on $X\times\mathbb R$,

$$F_\beta(\mathbf w,\alpha)=\alpha+(1-\beta)^{-1}\,\mathbb E\{[f(\mathbf w,\mathbf y)-\alpha]^+\},\qquad [t]^+=\max\{0,t\},$$

where the free threshold $\alpha$ accumulates, rescaled by $(1-\beta)^{-1}$, only the
loss excesses beyond it. The first theorem: as a function of $\alpha$, $F_\beta$ is
convex and continuously differentiable, and the CVaR is obtained by minimizing it,
$\phi_\beta(\mathbf w)=\min_\alpha F_\beta(\mathbf w,\alpha)$. The argmin set
$A_\beta(\mathbf w)$ is a non-empty closed bounded interval whose **left endpoint is
the $\beta$-VaR**, so $\alpha_\beta(\mathbf w)\in\arg\min_\alpha F_\beta$ and
$\phi_\beta(\mathbf w)=F_\beta(\mathbf w,\alpha_\beta(\mathbf w))$. Thus **CVaR is
computed without first computing VaR** — the VaR emerges as a by-product.

The mechanism is visible in the derivative. The right derivative is
$\partial^+F_\beta/\partial\alpha=(\Psi(\mathbf w,\alpha)-\beta)/(1-\beta)$ and the left
derivative uses the left limit $\Psi(\mathbf w,\alpha^-)$; by convexity the minimum is
attained where $\Psi(\mathbf w,\alpha^-)\le\beta\le\Psi(\mathbf w,\alpha)$. Setting the
derivative to zero amounts to $\Psi(\mathbf w,\alpha)=\beta$ — the minimizing threshold
is exactly the $\beta$-quantile of the loss, the VaR.

## Joint minimization: VaR and CVaR at once

The second theorem extends this from minimizing over $\alpha$ alone to the joint choice
of $(\mathbf w,\alpha)$, making the auxiliary function a portfolio-optimization tool:

$$\min_{\mathbf w\in X}\phi_\beta(\mathbf w)=\min_{(\mathbf w,\alpha)\in X\times\mathbb R}F_\beta(\mathbf w,\alpha),$$

and a pair $(\mathbf w^*,\alpha^*)$ solves the right side iff $\mathbf w^*$ solves the
left and $\alpha^*\in A_\beta(\mathbf w^*)$. In the typical case where $A_\beta(\mathbf w^*)$
is a single point, one minimization yields both the CVaR-optimal $\mathbf w^*$ and its
$\beta$-VaR $\alpha^*$. The principle is the standard one of minimizing first over
$\alpha$ for each $\mathbf w$ (recovering $\phi_\beta$) then over $\mathbf w$; the
practical payoff is operating on the simple $F_\beta$ instead of the awkward
$\phi_\beta$. The presence of an expectation places this in stochastic programming.

## Convexity and the portfolio structure

The key tractability property is convexity: $F_\beta(\mathbf w,\alpha)$ is convex in
$(\mathbf w,\alpha)$, and $\phi_\beta(\mathbf w)$ convex in $\mathbf w$, whenever
$f(\mathbf w,\mathbf y)$ is convex in $\mathbf w$; then, with convex $X$, joint
minimization is a convex program, eliminating any gap between local and global minima —
the trait that sharply distinguishes CVaR from VaR. For portfolio selection with
$w_j\ge0$, $\sum_j w_j=1$, and loss $f(\mathbf w,\mathbf y)=-\mathbf w^\top\mathbf y$
linear in $\mathbf w$, $F_\beta$ is convex in $(\mathbf w,\alpha)$; the loss has mean
$\mu(\mathbf w)=-\mathbf w^\top\mathbf m$ and variance $\mathbf w^\top\mathbf V\mathbf w$,
and a minimum-return requirement gives the linear constraint $\mu(\mathbf w)\le-R$,
making $X$ a polyhedron and the minimization a convex program.

A precise link to the mean-variance framework of
[mean-variance selection](../foundations/mean-variance-selection.md) emerges in the
**normal case**: for $\beta\ge0.5$, both VaR and CVaR are $\mu(\mathbf w)+c(\beta)\sigma(\mathbf w)$
with positive coefficients $c_1(\beta),c_2(\beta)$, so on the set where the return
constraint is active minimizing CVaR, VaR or variance yields the **same portfolio** —
the elliptical case is the one context where minimum-CVaR, minimum-VaR and Markowitz
minimum-variance coincide.

## The scenario formulation and the linear program

The decisive advantage: the integral in $F_\beta$ needs no analytic density — only a
sampler. Sampling $\mathbf y_1,\dots,\mathbf y_q$ and approximating the expectation by
the sample mean gives

$$\tilde F_\beta(\mathbf w,\alpha)=\alpha+\frac{1}{q(1-\beta)}\sum_{k=1}^q[f(\mathbf w,\mathbf y_k)-\alpha]^+,$$

convex and piecewise-linear in $\alpha$ (non-differentiable but easily minimized as a
linear program). Introducing a per-scenario excess variable $u_k$, minimizing
$\tilde F_\beta$ over $X\times\mathbb R$ is equivalent to minimizing the linear
objective $\alpha+\frac{1}{q(1-\beta)}\sum_k u_k$ subject to the linear portfolio
constraints and $u_k\ge0$, $\mathbf w^\top\mathbf y_k+\alpha+u_k\ge0$. Jointly these
force $u_k\ge[f(\mathbf w,\mathbf y_k)-\alpha]^+$, and since $u_k$'s objective
coefficient is positive each is pushed to its minimum, reproducing the excess term.
The unknowns are the weights $\mathbf w$, the threshold $\alpha$, and the $q$ excesses
$u_k$; objective and constraints are all linear. Crucially, **this LP reduction does
not depend on $\mathbf y$ being normal** — it works identically for non-normal
distributions. So minimizing a portfolio's CVaR over scenarios is handed to an LP
solver that returns the optimal weights, the threshold (VaR) and the objective (CVaR)
simultaneously; the same LP structure serves MAD and minimax approaches.

## CVaR for general distributions

The continuity assumption is convenient but restrictive — scenario models and finite
sampling produce discontinuous loss distributions. The general treatment assumes
$\mathbf y$ governed by a measure $P$ **independent of $\mathbf w$** (essential for the
convexity results), with $f$ continuous in $\mathbf w$, measurable in $\mathbf y$, and
integrable. A positive jump $\Psi(\mathbf w,\alpha)-\Psi(\mathbf w,\alpha^-)=P\{f=\alpha\}$
signals a probability **atom** at $\alpha$. VaR keeps its definition (the min is
attained since $\Psi$ is non-decreasing, right-continuous); when $\Psi$ is a flat step
at level $\beta$ the equation $\Psi=\beta$ has an interval of solutions, motivating the
**upper VaR** $\alpha_\beta^+(\mathbf w)=\inf\{\alpha:\Psi(\mathbf w,\alpha)>\beta\}$,
with $\alpha_\beta\le\alpha_\beta^+$ always. Multiple solutions make VaR **unstable** —
a jump is certain if a slightly higher confidence is required.

General CVaR is the mean of the **$\beta$-tail distribution**, whose CDF is 0 below
$\alpha_\beta$ and $[\Psi(\mathbf w,\alpha)-\beta]/[1-\beta]$ above — a legitimate CDF
obtained by rescaling the portion of $\Psi$ between levels $\beta$ and 1. The subtlety:
at an atom on the threshold, only the part of the atom exceeding $\beta$ is aggregated,
giving exactly probability $1-\beta$. Alongside it, the **upper CVaR**
$\phi_\beta^+=\mathbb E\{f\mid f>\alpha_\beta\}$ and **lower CVaR**
$\phi_\beta^-=\mathbb E\{f\mid f\ge\alpha_\beta\}$ satisfy
$\phi_\beta^-\le\phi_\beta\le\phi_\beta^+$, with equality only when $\Psi$ has no jump
at the threshold — for scenario models, where a jump always exists, the inequalities
can be strict.

## Discrete representation and Expected Shortfall

The unusual trait giving CVaR its power is how it **splits an atom** at the threshold.
With $\lambda_\beta(\mathbf w)=[\Psi(\mathbf w,\alpha_\beta)-\beta]/[1-\beta]\in[0,1]$,
CVaR is the decision-dependent weighted average
$\phi_\beta=\lambda_\beta\,\alpha_\beta+(1-\lambda_\beta)\,\phi_\beta^+$ (or
$\phi_\beta=\alpha_\beta$ if $\alpha_\beta$ is the maximum possible loss), from which
CVaR dominates VaR, $\phi_\beta\ge\alpha_\beta$, strictly unless no loss exceeds the
threshold. For a **scenario model** with ordered losses $z_1<\dots<z_N$, probabilities
$p_k$, and $k_\beta$ the index with $\sum_1^{k_\beta}p_k\ge\beta>\sum_1^{k_\beta-1}p_k$,
the VaR is $z_{k_\beta}$ and

$$\phi_\beta(\mathbf w)=\frac{1}{1-\beta}\Big[\big(\textstyle\sum_1^{k_\beta}p_k-\beta\big)z_{k_\beta}+\sum_{k_\beta+1}^N p_k z_k\Big]:$$

CVaR is the weighted average of the tail losses — the VaR value gets only the fraction
of its probability exceeding $\beta$, and worse losses enter with full probability. If
the top point has $p_N>1-\beta$ then CVaR, lower CVaR and VaR all equal the maximum
loss. In the naming of tail measures, the upper CVaR is "mean shortfall",
$\phi_\beta^+-\alpha_\beta$ the "mean excess loss", and the lower CVaR the "tail VaR";
because it is **CVaR — not its upper or lower variants — that is coherent under
discontinuity**, identifying Expected Shortfall with CVaR is what preserves the good
properties.

## Coherence and the persistence of the minimization formula

The core general-distribution result is that the minimization formula persists: as a
function of $\alpha$, $F_\beta(\mathbf w,\cdot)$ is finite, convex (hence continuous),
with $\phi_\beta(\mathbf w)=\min_\alpha F_\beta$ whose argmin is a non-empty closed
bounded interval with left endpoint $\alpha_\beta$ and right endpoint $\alpha_\beta^+$;
convexity follows immediately from that of $[f-\alpha]^+$ in $\alpha$, and no analogous
formula holds for $\phi_\beta^+$ or $\phi_\beta^-$. If $f$ is convex in $\mathbf w$ then
$\phi_\beta$ is convex and $F_\beta$ jointly convex; if $f$ is **sublinear** in
$\mathbf w$ (subadditive and positively homogeneous — linearity being a special case)
then $\phi_\beta$ is sublinear. In the Artzner et al. framework a risk measure is
**coherent** if it is sublinear, equals $c$ on the constant loss $c$, and is monotone
under first-order stochastic dominance; the corollary is that **$\beta$-CVaR is
coherent** for general loss distributions when $f$ is linear in $\mathbf w$ —
a formidable advantage not shared by any other broadly applicable risk measure (Pflug
had already shown $\rho(z)=\min_\alpha\{\alpha+(1-\beta)^{-1}\mathbb E[z-\alpha]^+\}$
would be coherent). CVaR is also **stable** in $\beta$: $\phi_\beta$ varies continuously
with right and left derivatives, unlike other proposed measures.

## Computational link and extensions

Joint minimization extends to the general case unchanged:
$\min_{\mathbf w\in X}\phi_\beta=\min_{(\mathbf w,\alpha)}F_\beta$, so VaR and CVaR come
as a by-product with $\alpha_\beta(\mathbf w^*)\le\alpha^*\le\alpha_\beta^+(\mathbf w^*)$.
For a discrete space the auxiliary function is
$F_\beta(\mathbf w,\alpha)=\alpha+(1-\beta)^{-1}\sum_k p_k[f(\mathbf w,\mathbf y_k)-\alpha]^+$,
handled with variables $\eta_k\ge0$, $f(\mathbf w,\mathbf y_k)-\alpha-\eta_k\le0$ and
$\alpha+(1-\beta)^{-1}\sum_k p_k\eta_k\le\omega$ — all linear when $f$ is linear in
$\mathbf w$. CVaR can therefore enter a model **as a constraint** $\phi_\beta\le\omega$,
at one or several confidence levels $\beta_i$ with tolerances $\omega_i$, not just as
an objective. Applied to index tracking with a CVaR constraint, adding the constraint
improved risk characteristics both in- and out-of-sample. The closely related
conditional drawdown-at-risk (CDaR) is treated in
[drawdown risk measures](../risk-measures/drawdown-risk-measures.md), and this
framework — convex objective, joint minimization with the threshold, scenario LP
reduction, general-distribution coherence — underpins the
[risk budgeting and risk parity](../optimization/risk-budgeting-and-risk-parity.md)
approaches and the quantitative construction in
[quantitative selection and construction](../signals/quantitative-selection-and-construction.md).
