---
type: concept
title: "Choice Under Uncertainty"
description: The expected-utility foundation of portfolio choice — von Neumann-Morgenstern utility and its existence axioms, risk aversion as concavity measured by the Arrow-Pratt indices, and the exact conditions under which the mean-variance criterion is consistent with expected utility, leading to tangency selection of the optimal portfolio.
tags: [expected-utility, risk-aversion, von-neumann-morgenstern, arrow-pratt, certainty-equivalent, mean-variance, indifference-curves, elliptical-distributions]
sources:
  - id: openwiki-source-0ef71385a779c7501bb2117e
    resource: repo://docs/01_choice_under_uncertainty.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Choice Under Uncertainty

Portfolio selection is a decision made *before* investment outcomes are known:
future returns are random. This page builds the decision-theoretic basis for
preferring one portfolio distribution over another — the expected-utility paradigm —
and derives the conditions under which it collapses to the mean-variance criterion
used in [mean-variance selection](./mean-variance-selection.md). It also motivates
the shift, taken up in [coherent risk measures](../risk-measures/coherent-risk-measures.md),
away from variance when distributions are non-elliptical.

## Risk, uncertainty, and the expected-utility paradigm

Following Knight, one distinguishes **risk** (outcomes uncertain but assignable
measurable probabilities) from **uncertainty** proper (no assignable probabilities);
portfolio theory lives in the domain of risk, attaching a probability to each of $S$
future states. A decision under uncertainty has three ingredients: the *uncertainty*
(realization of one of $S$ states), the *actions* (the portfolio weights
$w_1,\dots,w_n$), and the *consequences* (final wealth $W_k=\sum_i x_{ik}w_i$ in the
realized state $k$). The investor maximizes von Neumann-Morgenstern (vNM) utility,
which the paradigm assumes exists, depends only on consequences, and is **additive
across states**:

$$U(W)=\sum_{s}\pi_s\,u(W_s)=E[u(W)].$$

The theory is *normative* (how one should choose) but is often used *positively* in
finance to describe actual behavior.

## Utility of certain amounts and expected utility

A **utility of money** function $u(x):M\to\mathbb R$ is strictly increasing, so it
preserves the natural ordering of certain amounts. Extending to random amounts, the
**expected utility** of a random variable $X$ is $E(u(X))=\sum_i u(x_i)p_i$ in the
finite discrete case and $\int u(t)f_X(t)\,dt$ in the continuous case; e.g. with
$u(x)=\ln x$ and $X=\{(1,\tfrac14),(3,\tfrac14),(7,\tfrac12)\}$, $E(u(X))\approx
1.247608$.

## Properties of the wealth-utility function

The utility of money satisfies two properties: it is **increasing** (non-satiation,
$u'(x)>0$, the agent maximizes wealth) and increases **less than proportionally** —
i.e. $u(x_1+\Delta x)-u(x_1)>u(x_2+\Delta x)-u(x_2)$ for $x_1<x_2$ — which is exactly
**concavity** ($u''(x)<0$) and encodes risk aversion. Graphically the curve rises
steeply for small amounts and flattens for large ones.

## Existence: preference axioms and the expected-utility theorem

The construction first needs an operation combining random prospects: the **mixture**
$X_1\alpha X_2$ realizes $X_1$ if event $A$ (probability $\alpha$) occurs and $X_2$
otherwise, scaling the two probability vectors by $\alpha$ and $1-\alpha$ so they
still sum to one. A preference relation with **reflexivity**, **transitivity**, and
**completeness** is a total weak preference $\succeq$; adding **Archimedean** and
**substitution** properties gives the five axioms of the vNM theorem: there exists
$u(x):M\to\mathbb R$ such that $X_1\succeq X_2$ under expected utility **iff**
$E(u(X_1))\ge E(u(X_2))$, and $u$ is **unique up to a positive affine transformation**
$a+b\,u(x)$ with $b>0$.

## Risk aversion: concavity, certainty equivalent, risk premium

The **certainty equivalent** is the certain amount giving the lottery's utility,
$CE\equiv u^{-1}(E[u(W)])$. For a concave $u$, **Jensen's inequality** gives
$u(CE)=E[u(W)]<u(E[W])$, hence $CE<E[W]$: a risk-averse investor prefers the certain
$E[W]$ to a lottery of equal expected value, so a risk-averse vNM utility must be
concave. The **risk premium** is $E(W)-CE$, the expected value the investor will give
up to remove randomness.

## Risk-aversion measures and utility families

Standardizing the second derivative by the first removes the scale dependence,
yielding the Arrow-Pratt **absolute risk aversion** $\mathrm{ARA}=-u''(W)/u'(W)$ and
**relative risk aversion** $\mathrm{RRA}=-u''(W)W/u'(W)$. The **CARA** family
$u(W)=-e^{-\rho W}$ has $\mathrm{ARA}=\rho$; the **CRRA** family
$u(W)=W^{1-\gamma}/(1-\gamma)$ (or $\ln W$ at $\gamma=1$) has $\mathrm{RRA}=\gamma$.
Log and exponential utilities satisfy non-satiation and concavity across their whole
domain, but **quadratic utility** $u(x)=x-\tfrac{a}{2}x^2$ has $u'(x)=1-ax>0$ only
for $x<1/a$ — it is not monotone increasing over the whole positive domain, a defect
central to the mean-variance consistency discussion.

## From expected utility to mean-variance

A Taylor expansion of $u(W)$ around $E(W)$ gives

$$E[u(W)]\approx u(E(W))+\tfrac{u''}{2}\mathrm{var}[W]+\tfrac{u'''}{6}\mathrm{skew}[W]+\tfrac{u^{(IV)}}{24}\mathrm{kurt}[W],$$

so expected utility depends only on mean and variance in three circumstances:
**approximately** for small risks (higher-order terms negligible), **exactly** if
returns are normal, or **exactly** if utility is quadratic. Under joint normality any
linear combination stays normal and a normal is fully described by its first two
moments. For quadratic utility $u(W)=W-bW^2$ the higher terms vanish and
$E[u(W)]=E(W)-b(E(W)^2+\mathrm{var}(W))$, reducing (up to monotone transforms) to
$E(W)-b\,\mathrm{var}(W)$; but quadratic utility has a satiation maximum and
*increasing* absolute risk aversion $\mathrm{ARA}=a/(1-aW)$, so it is a tractable
approximation rather than a complete specification.

## Consistency of mean-variance with expected utility

Mean-variance dominance is consistent with expected utility only under two mutually
exclusive circumstances: when the agent's utility over portfolio return $R_P$ is
**quadratic** $U(R_P)=R_P-\tfrac{a}{2}R_P^2$, or when the joint return distribution
is **multivariate elliptical** (equi-density surfaces are ellipsoids; the
multivariate normal and Student-t are special cases). The practical stakes are real:
"if one uses a variance-covariance model for non-elliptic distributions one can
severely underestimate events that cause the most severe losses." In the quadratic
case $E[U(R_P)]=r_P-\tfrac{a}{2}(r_P^2+\sigma_P^2)$ depends only on $r_P$ and
$\sigma_P^2$, so expected-utility dominance coincides with mean-variance dominance —
though the $X<1/a$ defect means every realization of $R_P$ must lie below $1/a$. The
resulting indifference curve $r_P-\tfrac{a}{2}(r_P^2+\sigma_P^2)=k$ is a **circle** in
the variance-mean plane, centered at $(0,\tfrac1a)$.

## The mean-variance utility function and indifference curves

Assuming consistency, one adopts the mean-variance utility $U=E(r)-\tfrac12 A\sigma^2$,
where $A$ is the risk-aversion coefficient: $A>0$ risk-averse (more so as $A$ grows),
$A=0$ neutral, $A<0$ risk-loving. The **mean-variance dominance criterion** says
portfolio $i$ dominates $j$ if $E(r_i)\ge E(r_j)$ and $\sigma_i\le\sigma_j$ with at
least one strict. **Indifference curves** (constant utility) obey
$E(r)=c+\tfrac12 A\sigma^2$, with slope $A\sigma>0$ and second derivative $A>0$: they
are increasing and convex, and steeper for higher $A$ (a more risk-averse investor
demands more compensation for the same added risk). The **certainty-equivalent
return** is $r_{CE}=E[\tilde r]-\tfrac12 A\sigma^2(\tilde r)$; if $r_f\ge r_{CE}$ it is
optimal to hold the risk-free asset.

## Tangency selection of the optimal portfolio

The efficient frontier (max $E(r_p)$ per level of $\sigma_p$) depends only on the
mean-variance criterion, not on preferences; choosing the **optimal** portfolio among
efficient ones does depend on risk aversion. The expected-utility-maximizing
efficient portfolio is where the frontier is **tangent to the highest-level
indifference curve** — the optimum $X^*$. With a risk-free asset available, the
optimal choice reduces to splitting wealth by a fraction $y$ between the risk-free
asset and a risky portfolio; maximizing $U(r_C)=r_f+y(E(r_T)-r_f)-\tfrac12 A y^2
\sigma_T^2$ gives the first-order condition $(E(r_T)-r_f)-Ay\sigma_T^2=0$ and the
optimal share

$$y^*=\frac{E(r_T)-r_f}{A\,\sigma_T^2},$$

inversely proportional to risk aversion $A$ and variance $\sigma_T^2$ and proportional
to the excess return $E(r_T)-r_f$. The construction of the reference risky portfolio
and its frontier is developed in [mean-variance selection](./mean-variance-selection.md).
