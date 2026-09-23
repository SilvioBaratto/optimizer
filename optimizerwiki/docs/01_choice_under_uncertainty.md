---
title: "Choice Under Uncertainty"
chapter: 1
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-10
---

> [!abstract] Summary
> The chapter grounds the evaluation of random financial choices in the expected utility principle: it defines the utility of certain amounts and its extension to random returns, states the properties of the wealth utility function and the assumptions on the preference relation that guarantee its existence, and shows how risk aversion corresponds to concavity, measured by the Arrow–Pratt indices. Finally, it establishes the consistency between the mean-variance criterion and expected utility — exact for quadratic utility — and derives from it the representation via indifference curves and the selection of the optimal portfolio by tangency.

## Risk, Uncertainty, and the Expected Utility Paradigm

Portfolio selection is a decision made before knowing the outcome of the investments: future returns are random quantities. It is therefore necessary to distinguish situations in which future outcomes are uncertain but measurable probabilities can be assigned to them — *risk* — from those in which such probabilities cannot be assigned for lack of a statistical basis or a definable distribution — *uncertainty* in the strict sense [Knight1921]. Portfolio theory is situated within the domain of risk: a probability is associated with each of the possible future states of the world.

A decision under uncertainty is described by three ingredients. The first is *uncertainty*, modeled as the future realization of one of $S$ possible states of the world. The second is *actions*: in finance, the choice of portfolio weights $w_1,\dots,w_n$. The third is *consequences*: the final outcome of the portfolio, that is, the wealth $W_k$ in the realized state of the world $k$,
$$W_k = x_{1k}w_1 + x_{2k}w_2 + \dots + x_{nk}w_n,$$
where $x_{ik}$ is the realized payoff of asset $i$ in state $k$.

The investor's problem consists in choosing the weights that maximize their *von Neumann–Morgenstern* (vNM) utility, $\max_{w} U(W_1,\dots,W_S) \equiv U(W)$ [vNM1944]. The expected utility paradigm rests on two assumptions: that it is possible to assign a probability $\pi_s$ to each state of the world, and that there exists a vNM utility $U(\cdot)$ that depends only on the consequences and is additive across states,
$$U(W) = U(W_1,\dots,W_S) = \pi_1 u(W_1) + \dots + \pi_S u(W_S) = E[u(W)].$$
Investors thus maximize $E[u(W)]$. This is a *normative* theory — of how an investor should behave — often used in finance also as a *positive* theory, to describe how investors actually behave.

## Utility of Certain Amounts and Expected Utility

The underlying idea is that the importance of a certain amount or return — hereafter referred to, sometimes loosely, by the term *wealth* — does not depend directly on its nominal value, but on the utility that value has for the person who receives it. Two individuals who receive the same monetary amount can derive a different benefit from it; choices must therefore be evaluated not on nominal monetary values, but on their utility.

A **money utility function** is defined as a function
$$u(x): M \to \mathbb{R},$$
where $x$ denotes the certain amount and $M$ the set of certain amounts, such that for every pair $x_1, x_2 \in M$
$$u(x_1) > u(x_2) \iff x_1 > x_2, \qquad u(x_1) = u(x_2) \iff x_1 = x_2.$$
That is, the function $u(\cdot)$ must preserve — be strictly increasing with respect to — the natural ordering of certain amounts.

Amounts, however, are frequently not certain: a tool is needed to compare random amounts or returns, namely **expected utility** (or mean utility). Given a random variable $X$ representing a random amount/return and a money utility function $u(\cdot)$, the expected utility of $X$ is defined, in the discrete finite case with $N$ outcomes $x_i$ of probability $p_i$, as
$$E(u(X)) = \sum_{i=1}^{N} u(x_i)\, p_i,$$
and, in the continuous case with density $f_X(\cdot)$, as
$$E(u(X)) = \int_{-\infty}^{+\infty} u(t)\, f_X(t)\, dt.$$
As an example, with logarithmic utility $u(x) = \ln(x)$ (for $x>0$) and wealth position $X = \{(1, \tfrac14), (3, \tfrac14), (7, \tfrac12)\}$, we obtain
$$E(u(X)) = \ln(1)\cdot\tfrac14 + \ln(3)\cdot\tfrac14 + \ln(7)\cdot\tfrac12 = 1{,}247608\ldots$$

## Properties of the Wealth Utility Function

A money utility function $u(x)$ satisfies (at least) two properties. The first is that **$u(x)$ is increasing**: given $x_1 < x_2$,
$$u(x_1) < u(x_2),$$
because the economic agent is a maximizer of the amount/return (non-satiation). The second is that **$u(x)$ increases less than proportionally**: given $x_1 < x_2$ and a change in amount $\Delta x$,
$$u(x_1 + \Delta x) - u(x_1) > u(x_2 + \Delta x) - u(x_2),$$
because the economic agent is risk-averse. This latter property is equivalent to requiring **concavity** of the utility function: an increase in wealth $\Delta x$ generates an increase in utility that is smaller the higher the starting level of wealth. Graphically, the wealth utility function is increasing but with decreasing slope: it rises rapidly for small amounts and progressively flattens out for large amounts.

In differential terms, non-satiation is expressed as $u'(x) > 0$ and risk aversion as $u''(x) < 0$. The crucial question is then whether there exists a wealth utility function capable of inducing on $M$ an ordering consistent with the decision-maker's weak preference relation. The answer, under suitable assumptions, is affirmative; the tools needed to establish this are introduced in the following section.

## Existence: Preference Axioms and the Expected Utility Theorem

The construction first requires an operation that combines random prospects. Given a random event $A$ of probability $\alpha$, and given two discrete random variables
$$X_1 = \{(x_{11}, p_{11}), \ldots, (x_{1N}, p_{1N})\}, \qquad X_2 = \{(x_{21}, p_{21}), \ldots, (x_{2M}, p_{2M})\},$$
the first realized if $A$ occurs and the second if it does not, the **mixture of $X_1$ and $X_2$ according to $A$**, denoted $X_1 \alpha X_2$, is defined as the random variable
$$X_1 \alpha X_2 = \{(x_{11}, p_{11}\alpha), \ldots, (x_{1N}, p_{1N}\alpha), (x_{21}, p_{21}(1-\alpha)), \ldots, (x_{2M}, p_{2M}(1-\alpha))\}.$$
The probabilities thus obtained lie between $0$ and $1$ and correctly sum to one, since
$$(p_{11} + \ldots + p_{1N})\alpha + (p_{21} + \ldots + p_{2M})(1-\alpha) = 1 \cdot \alpha + 1 \cdot (1-\alpha) = 1.$$

Let $R$ be a relation defined on the set $X$. It may satisfy the following properties: **reflexivity** ($X_i \succeq X_i$ for every $X_i$); **transitivity** (if $X_h \succeq X_i$ and $X_i \succeq X_j$, then $X_h \succeq X_j$); **completeness** (for every pair, $X_j \succeq X_i$ or $X_i \succeq X_j$). A relation endowed with these three properties is a total weak preference relation, denoted $\succeq$. To these are added the **Archimedean property** — for every triple with $X_h \succeq X_i \succeq X_j$ there exist probabilities $\alpha$ and $\beta$ such that
$$X_h \alpha X_j \succeq X_i \succeq X_j \beta X_h$$
— and the **substitution property** — if $X_i \succeq X_j$, then $X_i \alpha X_h \succeq X_j \alpha X_h$ for every $X_h$ and every probability $\alpha$.

These five properties are the assumptions that guarantee the existence of a utility function consistent with the expected utility principle, according to the following theorem [vNM1944].

> **Theorem.** Given a relation $R$ defined on $M$ that satisfies the properties of reflexivity, transitivity, completeness, Archimedeanness, and substitution, and given two discrete random variables $X_1$ and $X_2$, there exists a function $u(x): M \to \mathbb{R}$ such that $X_1$ dominates $X_2$ under the expected utility principle ($X_1 \overset{UA}{\succeq} X_2$) if and only if
> $$E(u(X_1)) \ge E(u(X_2));$$
> moreover $u(x)$ is unique up to a transformation of the type
> $$a + b \cdot u(x), \qquad a \in \mathbb{R},\ b > 0.$$

In other words, if the preference relation satisfies the five assumptions, there always exists a utility function that represents it through the comparison of expected utilities, and it is uniquely determined up to increasing affine transformations (a change of origin $a$ and of positive scale $b$).

## Risk Aversion: Concavity, Certainty Equivalent, and Risk Premium

The shape of the vNM utility function reflects risk preferences. Consider a lottery that yields final wealth $W_1$ or $W_2$. If $u(\cdot)$ is concave, the expected value of the lottery's utility, $E[u(W)]$, lies below the curve $u(W)$ evaluated at the expected value $E[W]$.

The **certainty equivalent** (CE) is defined as the certain amount that gives the same utility as the lottery, that is,
$$CE \equiv u^{-1}\big(E[u(W)]\big).$$
For a risk-averse investor, the utility function is concave and **Jensen's inequality** holds [Jensen1906]:
$$u(CE) = E[u(W)] < u(E[W]) \;\Longrightarrow\; CE < E[W].$$
The certainty equivalent is therefore less than the expected payoff: a risk-averse investor prefers the certain amount $E[W]$ to the lottery of equal expected value. It follows that the vNM utility function of a risk-averse investor must be concave.

The difference between the expected value and the certainty equivalent defines the **risk premium**:
$$\text{risk premium} = E(W) - CE,$$
that is, the amount the investor is willing to give up, in terms of expected value, in order to eliminate the randomness.

## Risk Aversion Measures and Families of Utility Functions

In finance it is assumed that investors are risk-averse, that is, willing to accept risk only if adequately compensated. Risk aversion depends on the second derivative of the utility function, which in turn depends on the scale parameter; to remove this dependence, it is standardized with respect to the first derivative [Pratt1964; Arrow1965]. Two standard measures are obtained: **absolute risk aversion** (Arrow–Pratt),
$$\mathrm{ARA} = -\frac{u''(W)}{u'(W)},$$
and **relative risk aversion**,
$$\mathrm{RRA} = -\frac{u''(W)\,W}{u'(W)}.$$

These measures correspond to notable families of utility functions. The **constant absolute risk aversion** (CARA) family is
$$u(W) = -e^{-\rho W}, \qquad \mathrm{ARA} = \rho,$$
while the **constant relative risk aversion** (CRRA) family is
$$u(W) = \frac{W^{1-\gamma}}{1-\gamma}\ (\gamma \ne 1), \qquad u(W) = \ln(W)\ (\gamma = 1), \qquad \mathrm{RRA} = \gamma.$$
Both are commonly used in asset pricing.

Other classical functions can be checked against the two required properties — non-satiation ($u'>0$) and risk aversion ($u''<0$). **Logarithmic utility** $u(x) = \ln(x)$, for $x>0$, has $u'(x) = 1/x > 0$ and $u''(x) = -1/x^2 < 0$: it satisfies both over the entire positive domain. **Exponential utility** $u(x) = 1 - e^{-ax}$, with $a>0$, has $u'(x) = a e^{-ax} > 0$ and $u''(x) = -a^2 e^{-ax} < 0$: it too satisfies non-satiation over the entire domain. **Quadratic utility**
$$u(x) = x - \frac{a}{2}x^2, \qquad a>0,$$
has $u'(x) = 1 - ax$ and $u''(x) = -a < 0$: it is concave by construction, but non-satiation $u'(x)>0$ holds only for $x < 1/a$. Unlike the logarithmic and exponential cases, the quadratic is therefore not monotonically increasing over the entire positive domain, but only over an interval bounded by the parameter $a$. This property will play a role in the discussion of consistency with the mean-variance criterion.

## From Expected Utility to Mean-Variance

Expected utility depends both on the probability distribution and on the utility function. To link it to only the first two moments, $u(W)$ is approximated around $E(W)$ with a Taylor expansion:
$$u(W) \approx u(E(W)) + u'(E(W))(W-E(W)) + \frac{u''(E(W))}{2}(W-E(W))^2 + \frac{u'''(E(W))}{6}(W-E(W))^3 + \frac{u^{(IV)}(E(W))}{24}(W-E(W))^4.$$
Taking expected values yields the approximation of vNM utility:
$$E[u(W)] \approx u(E(W)) + \frac{u''(E(W))}{2}\,\mathrm{var}[W] + \frac{u'''(E(W))}{6}\,\mathrm{skew}[W] + \frac{u^{(IV)}(E(W))}{24}\,\mathrm{kurt}[W].$$

Expected utility depends only on mean and variance — that is, Markowitz's mean-variance approach is compatible with the vNM paradigm [Markowitz1952] — under three circumstances: **approximately**, for small risks, where terms of order higher than the second are negligible; **exactly**, if returns are normally distributed; **exactly**, if utility is quadratic. In the case of **joint normality**, any linear combination of jointly normal variables is itself normal, and since the normal distribution is completely described by its first two moments, $r \sim N(\mu, \sigma^2)$, expected utility is also a function of only two parameters.

In the case of **quadratic utility** $u(W) = W - bW^2$, the terms of order higher than the second are zero and expected utility equals
$$E[u(W)] = E(W) - b\,E[W^2] = E(W) - b\big(E(W)^2 + \mathrm{var}(W)\big).$$
Ignoring the monotonic transformations that do not alter preferences,
$$E[u(W)] = E(W) - b\,\mathrm{var}(W),$$
a function of the mean alone and the variance alone. Quadratic utility does, however, have drawbacks: it has a maximum beyond which investors prefer "less to more," and its absolute risk aversion is increasing, $\mathrm{ARA} = a/(1-aW)$. It is thus more a tractable approximation than a complete specification of the utility function.

## Consistency Between the Mean-Variance Criterion and Expected Utility

The mean-variance dominance criterion is consistent with the expected utility criterion only under two mutually exclusive circumstances: when the agent's utility function, expressed in terms of portfolio return $R_P$, is **quadratic**,
$$U(R_P) = R_P - \frac{a}{2}R_P^2, \qquad a>0,$$
or when the joint distribution function of $R_1,\dots,R_N$ is a **multivariate elliptical** one — a distribution whose equi-density surfaces are ellipsoids. The multivariate normal and the multivariate Student's $t$ are particular cases of elliptical distributions. The importance of this condition is practical: «if one uses a variance–covariance model for non–elliptic distributions one can severely underestimate events that cause the most severe losses» [Szego2005].

Consistency in the quadratic case is shown directly. The expected value of quadratic utility in terms of $R_P$ is
$$\mathbb E[U(R_P)] = \mathbb E\!\left(R_P - \frac{a}{2}R_P^2\right) = \mathbb E(R_P) - \frac{a}{2}\,\mathbb E\!\left(R_P^2\right) = r_P - \frac{a}{2}\big(r_P^2+\sigma_P^2\big),$$
having used $\mathbb E(R_P^2) = r_P^2 + \sigma_P^2$. The expected value depends **only** on $r_P$ and on $\sigma_P^2$: for quadratic utility, therefore, the expected utility dominance criterion coincides with the mean-variance one. $\blacksquare$

Note that the drawback of the quadratic function emerges here as well: $U(X)$ is increasing only for $X < 1/a$, so that for correct use all realizations of $R_P$ must be less than $1/a$.

The generic indifference curve of expected quadratic utility,
$$r_P - \frac{a}{2}\big(r_P^2 + \sigma_P^2\big) = k,$$
is a circle in the variance–mean plane. Rearranging,
$$\sigma_P^2 + r_P^2 - \frac{2}{a}r_P + \frac{2}{a}k = 0,$$
which is the canonical representation of a circle centered at $\left(0, \tfrac1a\right)$ with radius $\sqrt{\tfrac{1-2ak}{a^2}}$.

## The Mean-Variance Utility Function and Indifference Curves

Assuming consistency, a mean-variance utility function expressed in terms of returns is adopted,
$$U = E(r) - \frac{1}{2}A\,\sigma^2,$$
where $E(r)$ is the expected return of the asset or portfolio, $\sigma^2$ the variance of returns, $\tfrac12$ a scale factor, and $A$ the risk aversion coefficient. The partial derivatives
$$\frac{\partial U}{\partial E(r)} > 0, \qquad \frac{\partial U}{\partial \sigma} = -A\,\sigma$$
show that utility increases with expected return and, for $A>0$, decreases with risk. The parameter $A$ measures the attitude toward risk: $A>0$ indicates a risk-averse investor (the more averse, the larger $A$); $A=0$ a risk-neutral investor; $A<0$ a risk-loving investor.

From this function follows the **mean-variance dominance criterion**: portfolio $i$ dominates portfolio $j$ if
$$E(r_i) \ge E(r_j) \quad\text{and}\quad \sigma_i \le \sigma_j,$$
with at least one strict inequality. In the $E(r)$–$\sigma$ plane, the preferred direction is that of higher expected return and lower risk.

An **indifference curve** is the locus of combinations of $E(r)$ and $\sigma$ that leave utility unchanged: equally desirable portfolios lie, in that plane, on the same curve. For a given $A$, indifference curves do not intersect. Imposing $U(r) = c$ constant yields their equation,
$$E(r) = c + \frac{1}{2}A\,\sigma^2,$$
with slope
$$\frac{dE(r)}{d\sigma} = A\,\sigma \ > 0 \qquad\text{and}\qquad \frac{d^2E(r)}{d\sigma^2} = A > 0.$$
The curves are therefore **increasing** — as $\sigma$ increases, $E(r)$ must increase to keep utility unchanged — and **convex** — the increase in $E(r)$ required to compensate for increases in $\sigma$ grows as the risk already assumed grows. For a given level of utility, curves with larger $A$ are steeper: a more risk-averse investor requires greater compensation for the same increase in risk.

Useful in this framework is the **certainty-equivalent return** $r_{CE}$: the return of a risk-free asset that leaves the investor indifferent between the risky asset and the certainty equivalent. Imposing $U(r_{CE}) = U(\tilde r)$ and noting that the certainty equivalent has zero variance ($\sigma^2_{r_{CE}} = 0$),
$$r_{CE} = E[\tilde r] - \frac{1}{2}A\,\sigma^2(\tilde r).$$
If $r_f \ge r_{CE}$, it is optimal to invest in the risk-free asset. As $A$ increases, the certainty equivalent decreases, consistent with greater aversion.

## Selecting the Optimal Portfolio by Tangency

The construction of the efficient frontier — the locus of portfolios with maximum $E(r_p)$ for each level of $\sigma_p$ — depends only on the mean-variance criterion and not on individual preferences; it is the subject of the chapter [[02 Selezione media-varianza]]. The choice of the **optimal portfolio** among the efficient ones instead depends on the investor's degree of risk aversion.

Assuming consistency between the mean-variance criterion and expected utility, the portfolio (among the efficient ones) that maximizes expected utility is the one at which the efficient frontier is **tangent to the indifference curve with the highest attainable level**. In the mean-variance plane, the family of indifference curves increases in the direction of higher utility levels; the optimal portfolio $X^{*}$ is identified by the point of tangency between the efficient frontier and the highest attainable indifference curve.

When the opportunity set includes a risk-free asset, the optimal choice reduces to the allocation of wealth between the risk-free asset and a risky portfolio, according to a share $y$. Maximizing the utility of the complete portfolio,
$$\max_{y}\; U(r_C) = E(r_C) - \frac{1}{2}A\,\sigma_C^2, \qquad E(r_C) = r_f + y\,\big(E(r_T)-r_f\big),\ \ \sigma_C^2 = y^2\sigma_T^2,$$
and substituting the constraints into the objective function yields
$$\max_{y}\; r_f + y\,\big(E(r_T)-r_f\big) - \frac{1}{2}A\,y^2\sigma_T^2.$$
The first-order condition
$$\frac{\partial U(r_C)}{\partial y} = \big(E(r_T)-r_f\big) - A\,y\,\sigma_T^2 = 0$$
gives the optimal share
$$y^{*} = \frac{E(r_T)-r_f}{A\,\sigma_T^2}.$$
The share invested in the risky portfolio is thus inversely proportional to risk aversion $A$ and to variance $\sigma_T^2$, and proportional to the excess expected return $E(r_T)-r_f$. Geometrically, $y^{*}$ identifies the point of tangency between the indifference curve and the efficient frontier with a risk-free asset, which determines the optimal complete portfolio. The construction of the reference risky portfolio and its frontier is developed in the chapter [[02 Selezione media-varianza]].

## References

- **[Arrow1965]** Arrow, K. J. (1965). The Theory of Risk Aversion. In: Aspects of the Theory of Risk-Bearing. Helsinki: Yrjö Jahnssonin Säätiö.
- **[Jensen1906]** Jensen, J. L. W. V. (1906). Sur les fonctions convexes et les inégalités entre les valeurs moyennes. Acta Mathematica, 30(1), 175–193.
- **[Knight1921]** Knight, F. H. (1921). Risk, Uncertainty and Profit. Boston: Houghton Mifflin.
- **[Markowitz1952]** Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77–91.
- **[Pratt1964]** Pratt, J. W. (1964). Risk Aversion in the Small and in the Large. Econometrica, 32(1–2), 122–136.
- **[Szego2005]** Szegö, G. (2005). Measures of Risk. European Journal of Operational Research, 163(1), 5–19.
- **[vNM1944]** von Neumann, J., Morgenstern, O. (1944). Theory of Games and Economic Behavior. Princeton: Princeton University Press.
