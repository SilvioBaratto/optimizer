---
title: "Risk Attribution, Budgets, and Limits"
chapter: 23
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-15
---

> [!abstract] Summary
> The chapter establishes how the risk of an already-allocated portfolio decomposes into additive contributions attributable to the individual components — securities, factors, sub-portfolios — and how these contributions become the tool for monitoring and controlling limits. It is shown that, for a risk measure homogeneous of degree one, the Euler contributions defined as partial derivatives are the only choice compatible with RORAC and sum to the overall risk; the formulas for standard deviation, Value-at-Risk, and Expected Shortfall are derived from this, their financial interpretation as expected loss contributions is given, and the axiomatic justification as a coherent allocation (Aumann-Shapley value) is provided. It concludes with the risk budgeting framework and the existence and uniqueness conditions for the portfolio that realizes it.

## Attribution Versus Construction: The Problem

Let an already-allocated portfolio be given. Denote by $X_1, \dots, X_n$ the profit/loss random variables of the individual securities, factors, or sub-portfolios, and by

$$X = \sum_{i=1}^{n} X_i$$

the profit/loss of the entire book [Tasche2008]. The economic capital required to cover large losses is determined by a risk measure $\rho$, that is, $\mathrm{EC} = \rho(X)$; in practice $\rho$ is typically linked to the variance or to a quantile of the portfolio loss distribution [Tasche2008].

Measuring the overall risk $\rho(X)$, however, is only the first step of portfolio-oriented management. For identifying concentrations, for risk-sensitive pricing, and for book diagnostics, one must decompose the economic capital into a sum of contributions attributable to the sub-portfolios or individual exposures [Tasche2008]. This is the attribution question: *how much does component $i$ contribute to the risk $\mathrm{EC} = \rho(X)$?* This is an operation distinct from constructing a portfolio with assigned contributions — the construction of equally-weighted risk contribution and risk parity portfolios is treated in [[17 Risk budgeting e risk parity]]; here the focus is ex-ante attribution on an already given book and the limit control that follows from it.

The problem is non-trivial because of the diversification effect: the sum of the "risks" of the individual components is normally greater than the "risk" of their sum [Denault2001]. The risk capital of the aggregate is lower than the sum of the stand-alone capitals, and this diversification benefit must be fairly allocated among the components [Denault2001]. The risk capital of a component, reduced by its share of the diversification benefit, is in fact a risk measure internal to the institution [Denault2001]. The attribution exercise serves essentially for comparison purposes: knowing the profit generated *and* the risk assumed by each component allows a far more sensible comparison than profit alone, and underlies risk-adjusted performance measures and return on risk-adjusted capital (RORAC) [Denault2001].

## The Euler Principle and RORAC Compatibility

**Dynamic setup and homogeneity.** To define the contributions it is convenient to introduce weights $u = (u_1, \dots, u_n)$ and consider

$$X(u) = \sum_{i=1}^{n} u_i X_i,$$

so that $X = X(1, \dots, 1)$; the variable $u_i$ is interpreted as the amount invested in the underlying position $X_i$ [Tasche2008]. Fixing the distribution of $(X_1, \dots, X_n)$, set $f_\rho(u) = \rho(X(u))$. The Euler allocation principle applies to any risk measure homogeneous of degree one, that is, such that $\rho(hX) = h\,\rho(X)$ for $h \geq 0$ [Tasche2008].

**Defining contributions via RORAC.** Setting $\mu_i = \mathrm{E}[X_i]$, the portfolio RORAC and that of the individual component are defined as:

$$\mathrm{RORAC}(X) = \frac{\mathrm{E}[X]}{\rho(X)}, \qquad \mathrm{RORAC}(X_i \mid X) = \frac{\mathrm{E}[X_i]}{\rho(X_i \mid X)} = \frac{\mu_i}{\rho(X_i \mid X)},$$

where $\rho(X_i \mid X)$ is the risk contribution (yet to be defined) of $X_i$ to $\rho(X)$ [Tasche2008]. Two properties are economically desirable. The contributions satisfy the *full allocation property* if

$$\sum_{i=1}^{n} \rho(X_i \mid X) = \rho(X),$$

and are *RORAC compatible* if there exist $\epsilon_i > 0$ such that

$$\mathrm{RORAC}(X_i \mid X) > \mathrm{RORAC}(X) \;\Rightarrow\; \mathrm{RORAC}(X + hX_i) > \mathrm{RORAC}(X) \quad \text{for } 0 < h < \epsilon_i,$$

that is: if component $i$ has a higher RORAC than the portfolio, (marginally) increasing its weight improves the overall RORAC [Tasche2008]. RORAC compatibility is the formal translation of the fact that the contributions correctly guide reallocation decisions.

**Uniqueness.** For a "smooth" risk measure, RORAC compatibility completely determines the contributions. If $f_\rho$ is continuously differentiable and the contributions $\rho(X_i\mid X)$ are RORAC compatible for arbitrary expected values $\mu_1, \dots, \mu_n$, then they are uniquely given by

$$\rho_{\text{Euler}}(X_i \mid X) = \left. \frac{d\rho}{dh}(X + hX_i) \right|_{h=0} = \frac{\partial f_\rho}{\partial u_i}(1, \dots, 1),$$

that is, by the directional derivative of the risk in the direction of component $i$ [Tasche2008]. This result — proven in [Tasche1999] — is the deep reason why Euler allocation is not an artifice but the economically mandatory choice.

**Euler's theorem and additivity.** It remains to verify the full allocation property. By Euler's theorem on homogeneous functions, if $f_\rho$ is continuously differentiable it satisfies

$$f_\rho(u) = \sum_{i=1}^{n} u_i \, \frac{\partial f_\rho(u)}{\partial u_i}$$

for every $u$ in its domain *if and only if* $f_\rho$ is homogeneous of degree one [Tasche2008]. Consequently, for a risk measure with continuously differentiable $f_\rho$, the two properties are obtained simultaneously if and only if $\rho$ is homogeneous of degree one; the contributions are then uniquely determined by the partial derivative. The contributions so defined are called *Euler contributions*, and the procedure of attributing capital by computing such contributions is called *Euler allocation* [Tasche2008]. The principle has been justified by several authors: from a practical perspective, emphasizing that Euler contributions naturally sum to the overall economic capital [PatrikEtAl1999]; observing that they are fully compatible with portfolio diagnostics and optimization [Litterman1996]; and via game-theoretic reasoning [Denault2001].

**Marginal contributions.** The *marginal contribution* (also called the with-without principle) must be distinguished from the Euler contribution; it is defined by the difference between the risk with and without component $i$:

$$\rho_{\text{marg}}(X_i \mid X) = \rho(X) - \rho(X - X_i).$$

For sub-additive, continuously differentiable, and homogeneous-of-degree-one measures, marginal contributions are always smaller than Euler contributions,

$$\rho_{\text{marg}}(X_i \mid X) \leq \rho_{\text{Euler}}(X_i \mid X),$$

and consequently their sum *underestimates* the total risk, $\sum_i \rho_{\text{marg}}(X_i \mid X) \leq \rho(X)$ [Tasche2008]. The marginal contribution therefore does not satisfy the full allocation property; forcing it by rescaling the contributions, $\rho^{*}_{\text{marg}}(X_i \mid X) = \rho_{\text{marg}}(X_i \mid X)\,\rho(X) / \sum_j \rho_{\text{marg}}(X_j \mid X)$, generally produces contributions that are no longer RORAC compatible [Tasche2008]. This is the technical reason why ex-ante attribution is based on Euler contributions and not on marginal ones.

## The Contribution Formulas: Standard Deviation, VaR, and Expected Shortfall

In principle the partial derivative defining the Euler contribution can be computed directly; for the most commonly used families of risk measures, however, closed-form expressions exist [Tasche2008].

**Standard-deviation-based measures.** Consider the family $\sigma_c(X) = c\sqrt{\mathrm{var}[X]}$, with $c > 0$; the constant $c$ is typically chosen so that $\mathrm{P}[X \leq \mathrm{E}[X] - \sigma_c(X)] \leq 1 - \alpha$, under normality assumptions or — as a robust alternative — via the one-sided Chebyshev inequality $\mathrm{P}[X \leq \mathrm{E}[X] - \sigma_c(X)] \leq 1/(1+c^2)$ [Tasche2008]. These measures are homogeneous of degree one and sub-additive; the Euler contributions are obtained by differentiation:

$$\sigma_c(X_i \mid X) = c\,\frac{\mathrm{cov}[X_i, X]}{\sqrt{\mathrm{var}[X]}}.$$

The contribution is thus proportional to the covariance between the component and the portfolio [Tasche2008].

**Value-at-Risk.** Given the $\gamma$-quantile $q_\gamma(Y) = \min\{y : \mathrm{P}[Y \leq y] \geq \gamma\}$, the Value-at-Risk of $X$ at confidence level $\alpha$ is $\mathrm{VaR}_\alpha(X) = q_\alpha(-X)$ [Tasche2008]. VaR is homogeneous of degree one and co-monotonic additive, but not in general sub-additive. Under regularity conditions that in particular imply the existence of a density for $X$, the Euler contributions to VaR can be expressed as conditional expectations [GourierouxEtAl2000]:

$$\mathrm{VaR}_\alpha(X_i \mid X) = -\mathrm{E}[X_i \mid X = -\mathrm{VaR}_\alpha(X)],$$

where $\mathrm{E}[X_i \mid X]$ is the conditional expectation of $X_i$ given $X$ [Tasche2008]. Often it is not the VaR but the unexpected loss that matters, $\mathrm{UL}_{\mathrm{VaR},\alpha}(X) = \mathrm{VaR}_\alpha(X - \mathrm{E}[X]) = \mathrm{VaR}_\alpha(X) + \mathrm{E}[X]$, and since the Euler contribution of $X_i$ to $\mathrm{E}[X]$ is obviously $\mathrm{E}[X_i]$, the formulas for the contributions to UL follow immediately [Tasche2008].

**Expected Shortfall.** Expected Shortfall at level $\alpha$ is the average of the VaRs at level $\alpha$ and above, $\mathrm{ES}_\alpha(X) = (1-\alpha)^{-1}\int_\alpha^1 \mathrm{VaR}_u(X)\,du$; an alternative name is Conditional Value-at-Risk (CVaR) [AcerbiTasche2002][RockafellarUryasev2002]. ES is homogeneous of degree one, co-monotonic additive, and sub-additive. Under regularity conditions ES can equivalently be written $\mathrm{ES}_\alpha(X) = -\mathrm{E}[X \mid X \leq -\mathrm{VaR}_\alpha(X)]$, and the formula for the Euler contributions is

$$\mathrm{ES}_\alpha(X_i \mid X) = -\mathrm{E}[X_i \mid X \leq -\mathrm{VaR}_\alpha(X)] = -(1-\alpha)^{-1}\,\mathrm{E}\!\left[X_i\,\mathbf{1}_{\{X \leq -\mathrm{VaR}_\alpha(X)\}}\right].$$

Unlike the VaR case, here the conditional expectation is *elementary*, because the conditioning event has positive probability of occurring [Tasche2008]. Optimizing the corresponding measure is treated in [[16 Ottimizzazione del CVaR - la formulazione di Rockafellar-Uryasev]]; computing the covariance needed for the standard-deviation formulas refers back to the techniques of [[18 Stima dinamica della covarianza - EWMA e GARCH multivariato]].

## Estimating Contributions from Sample Data

In most cases the loss distribution cannot be computed analytically but must be estimated from a simulated or historical sample. Given a sample of $N$ observations $(x_{1,k}, \dots, x_{n,k})$, the empirical measure $\widehat{\mathrm{P}}_N$ approximates the joint distribution and, by the law of large numbers, consistent estimators are obtained [Tasche2008].

**Standard deviation and Expected Shortfall.** For standard-deviation-based measures and for ES, statistically consistent estimators of the contributions are obtained simply by substituting the empirical variables $\widehat{X}, \widehat{X}_i$ for $X, X_i$ in the contribution formulas; thanks to the elementary conditional-expectation representation, estimating the contributions to ES is direct [Tasche2008].

**Value-at-Risk and kernel estimation.** The naïve approach does not work for VaR contributions when $X$ has a continuous distribution, since the conditioning event $\{X = -\mathrm{VaR}_\alpha(X)\}$ has zero probability. One then resorts to smoothing the empirical measure (kernel estimation). Introducing a variable $\xi$ independent of the data with density $\varphi$ (for example standard normal) and a bandwidth $b > 0$, one arrives at the approximation

$$\mathrm{VaR}_\alpha(X_i \mid X) \approx -\frac{\sum_{k=1}^{N} x_{i,k}\,\varphi\!\left(\frac{-\mathrm{VaR}_\alpha(\widehat{X}+b\xi) - x_k}{b}\right)}{\sum_{k=1}^{N} \varphi\!\left(\frac{-\mathrm{VaR}_\alpha(\widehat{X}+b\xi) - x_k}{b}\right)},$$

whose right-hand side is precisely the Nadaraya-Watson kernel estimator of the conditional expectation $\mathrm{E}[X_i \mid X = -\mathrm{VaR}_\alpha(X)]$ [Tasche2008]. A natural relationship thus emerges between Euler contributions to VaR and nonparametric estimation of conditional expectations. The choice of bandwidth is crucial; a practical rule is Silverman's, $b = 0.9\,\min(\sigma, R/1.34)\,N^{-1/5}$, with $\sigma$ and $R$ the standard deviation and interquartile range of the sample [Tasche2008]. The sum of these approximate contributions differs from natural estimators of portfolio VaR, but the difference tends to be small. It should be noted that estimates of contributions to VaR and ES are highly volatile, and importance sampling methods have been proposed to mitigate the problem [Tasche2008].

**Cornish-Fisher approximation.** An analytical alternative exploits the Cornish-Fisher expansion, which retains the "mean plus z-score times standard deviation" form but corrects the z-score for the higher moments [MinaUlmer1999]:

$$\mathrm{VaR} = \mu + \tilde{z}_\alpha\,\sigma, \qquad \tilde{z}_\alpha \approx z_\alpha + \tfrac{1}{6}(z_\alpha^2 - 1)\,s + \tfrac{1}{24}(z_\alpha^3 - 3z_\alpha)\,k - \tfrac{1}{36}(2z_\alpha^3 - 5z_\alpha)\,s^2,$$

where $s$ and $k$ are skewness and excess kurtosis, and $z_\alpha$ the normal z-score [Qian2006]. Since the mean is a linear function of the weights, the standard deviation the root of a quadratic function, and skewness and kurtosis third- and fourth-degree polynomial functions with coefficients given by the higher co-moments, the Cornish-Fisher VaR becomes an explicit function of the weights whose partial derivatives — and hence the contributions — can be derived analytically [Qian2006]. The method is particularly relevant for portfolios that include hedge funds, whose returns can have significant skewness and kurtosis: a risk budget that ignores these moments would seriously underestimate risk [Qian2006].

## Marginal Contributions, Concentration, and Diversification

Euler allocation is particularly well suited to identifying risk concentrations. In supervisory language, a risk concentration is any single exposure or group of exposures with the potential to produce losses large enough to threaten the soundness of a bank, and is probably the single most important cause of serious problems at intermediaries; the internal framework must therefore document how such concentrations and the corresponding limits are calculated [Tasche2008].

**Sub-additivity and the upper bound on contributions.** If $\rho$ is homogeneous of degree one *and* sub-additive, the Euler contributions never exceed the stand-alone risks of the components:

$$\rho_{\text{Euler}}(X_i \mid X) \leq \rho(X_i), \qquad i = 1, \dots, n.$$

In particular, for credit exposures, the contribution cannot exceed the face value of the exposure [Tasche2008]. For measures homogeneous of degree one and continuously differentiable, this property of the Euler contributions and the sub-additivity of the risk measure are in fact equivalent [Tasche2008].

**Diversification indices.** One defines the portfolio's *diversification index* and the *marginal diversification index* of sub-portfolio $X_i$:

$$\mathrm{DI}_\rho(X) = \frac{\rho(X)}{\sum_{i=1}^{n} \rho(X_i)}, \qquad \mathrm{DI}_\rho(X_i \mid X) = \frac{\rho_{\text{Euler}}(X_i \mid X)}{\rho(X_i)}.$$

If $\rho$ is homogeneous of degree one, sub-additive, and co-monotonic additive, then $\mathrm{DI}_\rho(X) \leq 1$, and a value close to 100% signals that the components are "nearly" co-monotonic, that is, strongly dependent: a portfolio with an index close to 100% can be considered highly concentrated in risk, one with a low index well diversified [Tasche2008]. The marginal indices instead indicate the *potential* for diversification: a portfolio with high unrealized diversification potential may be considered concentrated [Tasche2008].

**Factor risk impact.** The non-linear analysis of the impact of one or more systematic factors on portfolio risk is obtained by applying Euler allocation to the decomposition of the loss into the conditional expectation on the factors and its orthogonal complement. Given a set of factors $S = (S_1, \dots, S_k)$, the *risk impact* of factor $S$ on $L$ is defined as

$$\mathrm{RI}_\rho(L \mid S) = \frac{\rho(\mathrm{E}[L \mid S] \mid L)}{\rho(L)},$$

where $\rho(\mathrm{E}[L \mid S] \mid L)$ is the Euler contribution of the conditional expectation $\mathrm{E}[L\mid S]$ to $\rho(L)$ [Tasche2008]. Since the conditional expectation is uncorrelated with its own residual, $\mathrm{corr}[\mathrm{E}[L\mid S], L - \mathrm{E}[L\mid S]] = 0$, the first step amounts to decomposing the portfolio loss into a deterministic function of the factors and an uncorrelated residual, exhaustively and inclusive of non-linear effects [Tasche2008]. For the standard-deviation-based measure, which is translation invariant, the risk impact coincides with

$$\mathrm{RI}_\sigma(L \mid S) = \frac{\mathrm{var}[\mathrm{E}[L \mid S]]}{\mathrm{var}[L]} \in [0, 1],$$

that is, with a generalization of the $R^2$ coefficient of determination of regression analysis: the factors can be ranked by their RI, with a higher value signaling a larger impact that may warrant more attention [Tasche2008]. The construction of factor models is developed in [[05 Modelli fattoriali]].

## The Financial Interpretation of Contributions (Qian)

A recurring objection is that the risk contribution, defined as the partial derivative of risk with respect to the weights, is a mere mathematical decomposition without economic justification; since risk — standard deviation or VaR — is not additive, one might wonder whether risk budgets make real sense and whether they sum to 100% [Qian2006]. The answer is affirmative, and it goes through the interpretation of contributions as expected contributions *to the loss*.

**Percentage contribution to risk.** For a two-security portfolio with weights $w_1, w_2$, volatilities $\sigma_1, \sigma_2$, and correlation $\rho$, the standard deviation is $\sigma = \sqrt{w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2\rho w_1 w_2 \sigma_1 \sigma_2}$. The percentage contribution to risk of security $i$, weight times marginal contribution divided by risk, equals

$$p_1 = \left(w_1\frac{\partial \sigma}{\partial w_1}\right)\!\Big/\sigma = \frac{w_1^2\sigma_1^2 + \rho w_1 w_2 \sigma_1 \sigma_2}{\sigma^2},$$

and analogously $p_2$, with $p_1 + p_2 = 1$ [Qian2006]. $p_i$ is the ratio between the covariance of component $i$'s return with the portfolio and the total variance: it is thus the *beta* of the component relative to the portfolio, and the sum of the betas is unity [Qian2006]. This interpretation, however, does not yet provide an economic reason.

**Expected contribution to loss.** Suppose the portfolio suffers a loss of size $L$. The expected percentage contribution to loss of security $i$ is $c_i = \mathrm{E}(w_i r_i \mid w_1 r_1 + w_2 r_2 = L)/L$, that is, the expected contribution of the security conditioned on the loss, divided by $L$ [Qian2006]. Assuming jointly normal returns, the theory of the conditional distribution of normal variables gives

$$c_1 = p_1 + \frac{p_2 w_1 \mu_1 - p_1 w_2 \mu_2}{L} \triangleq p_1 + \frac{D_1}{L}, \qquad c_2 = p_2 + \frac{D_2}{L},$$

where $\mu_1, \mu_2$ are the expected returns and $D_2 = -D_1$ [Qian2006]. It follows that $c_1 + c_2 = p_1 + p_2 = 1$: the expected contributions to loss sum to 100%, exactly like the percentage contributions to risk. In the general case with $N$ assets and weight vector $w$, setting $\mu_R = w'\mu$ as the portfolio's expected return,

$$c_i = \frac{w_i \mu_i}{L} + \frac{\mathrm{cov}(w_i r_i, R)}{\mathrm{var}(R)}\left(1 - \frac{\mu_R}{L}\right) = p_i + \frac{w_i \mu_i - p_i \mu_R}{L} \triangleq p_i + \frac{D_i}{L}.$$

This is the central relation: the risk contribution is an expected contribution to loss, and this is why risk budgets sum up [Qian2006].

**Three special cases.** The two contributions coincide, $c_i = p_i$, in three situations. First, if the expected returns are zero ($\mu_i = 0$), as is reasonable to assume over short horizons (daily or weekly), then $D_i = 0$ and the percentage contribution to risk perfectly explains the expected contribution to loss, regardless of $L$ [Qian2006]. Second, if a security has zero weight its contribution is zero (trivial case). Third, and more interesting, when

$$\frac{w_1 \mu_1}{p_1} = \frac{w_2 \mu_2}{p_2},$$

which is the first-order condition for a mean-variance-optimal portfolio: for optimal portfolios the percentage contribution to risk equals the expected percentage contribution to total return, and risk budgets become expected-return budgets [Qian2006][Sharpe2002]. This equivalence, however, holds only for mean-variance-optimal portfolios (treated in [[02 Selezione media-varianza]]); for an actual portfolio, not necessarily optimal, the interpretation of the contribution as an estimate of the likely contribution to a given loss remains valid regardless [Qian2006].

**Relevance during crises.** The terms $D_i$ measure the sub-optimality of the portfolio. When the loss $L$ greatly exceeds the $D_i$'s, $c_i \approx p_i$: during financial crises, when losses greatly exceed expected returns, the contribution to loss is well captured by the risk contribution; conversely, in calm periods with small losses, the contribution to loss — a simple ex-post attribution — may bear no relation to the risk contribution, but these small events should not lead one to dismiss the concept's usefulness [Qian2006].

**The VaR contribution as a loss contribution.** The interpretation extends to VaR. Since VaR is a homogeneous function of degree one of the weights, the identity $\mathrm{VaR} = \sum_i w_i\,\partial \mathrm{VaR}/\partial w_i$ holds, and the percentage contribution to VaR is $c_i = w_i(\partial \mathrm{VaR}/\partial w_i)/\mathrm{VaR}$ [Hallerbach2002][Litterman1996]. It can be shown that this contribution is exactly the expected contribution to a loss of size equal to the VaR, $c_i = \mathrm{E}(w_i r_i \mid R = \mathrm{VaR})/\mathrm{VaR}$ [Hallerbach2002][Qian2006]. Compared with the standard-deviation case there are subtle differences: under normality the $p_i$ are independent of the loss, while the interpretation of the VaR contribution is more restrictive, applying only to the loss that exactly equals the VaR — changing the VaR changes the contribution, so for losses of different sizes it must be recomputed [Qian2006]. In summary, the risk contribution, or risk budget, can be viewed as a contribution to loss, or loss budget: risk budgeting as loss budgeting [Qian2006].

## The Coherence of Risk Capital Allocation (Denault)

Denault's approach is axiomatic: first the necessary properties of an allocation principle are argued, then the principles satisfying them are sought, in a manner parallel to how Artzner, Delbaen, Eber, and Heath defined the coherence of risk *measures* [Denault2001][ArtznerEtAl1999]. It is assumed that all risk measures are coherent, that is, satisfy sub-additivity, monotonicity, positive homogeneity, and translation invariance — the properties discussed in [[04 Misure di rischio coerenti]] [Denault2001][ArtznerEtAl1999].

**Allocation principle and axioms.** An allocation principle is a function $\Pi$ that assigns to every problem $(N, \rho)$ an allocation $(K_1, \dots, K_n)$ such that $\sum_{i\in N} K_i = \rho(X)$: the condition guarantees that the risk capital is *entirely* allocated [Denault2001]. A principle is *coherent* if, for every problem, the allocation satisfies three properties:

1. **No undercut.** $\displaystyle \sum_{i\in M} K_i \leq \rho\!\left(\sum_{i\in M} X_i\right)$ for every $M \subseteq N$: no portfolio, nor any coalition of portfolios, can be allocated more capital than it would face as an entity separate from the institution [Denault2001].
2. **Symmetry.** If, upon joining any subset $M \subseteq N\setminus\{i,j\}$, portfolios $i$ and $j$ give the same contribution to risk capital, then $K_i = K_j$: the allocation of a portfolio depends only on its contribution to the institution's internal risk, and on nothing else [Denault2001].
3. **Riskless allocation.** $K_n = \rho(\alpha r_f) = -\alpha$: a riskless portfolio must be allocated exactly its (negative) risk, which also means that, all else being equal, a portfolio that increases its cash position sees its allocated capital decrease by the same amount [Denault2001].

**Modeling as a cooperative game.** The allocation is modeled as a cooperative game associating the portfolios with the players and the risk measure with the cost function $c(S) = \rho\!\left(\sum_{i\in S} X_i\right)$ for $S \subseteq N$; since $\rho$ is coherent and thus sub-additive, $c$ is sub-additive [Denault2001]. The allocations satisfying no-undercut are exactly those lying in the *core* of the game, the set of allocations $\sum_{i\in S} K_i \leq c(S)$ for every coalition $S$ [Denault2001]. Non-emptiness of the core is thus crucial for the existence of coherent allocation principles, and it is shown that:

> If a risk capital allocation problem is modeled as a cooperative game whose cost function $c$ is defined via a coherent risk measure $\rho$, then its core is non-empty [Denault2001].

The proof rests on the Bondareva-Shapley theorem (the core is non-empty if and only if the game is balanced) and uses both the sub-additivity and the homogeneity of $\rho$ [Denault2001].

**Shapley value and fractional games.** The Shapley value $K_i^{Sh} = \sum_{S \ni i} \frac{(s-1)!(n-s)!}{n!}\big(c(S) - c(S\setminus\{i\})\big)$ provides a coherent allocation principle, but its evaluation requires computing $c$ on each of the $2^n$ coalitions [Denault2001][Shapley1953]. Denault extends the analysis to *fractional players*, which is more natural since it is possible to consider fractions of portfolios. A game with fractional players $(N, \Lambda, r)$ is given by a vector $\Lambda$ of full-presence levels and a cost function $r(\lambda) = \rho\!\left(\sum_i \frac{\lambda_i}{\Lambda_i} X_i\right)$, with $r(\Lambda) = \rho(N)$ [Denault2001][AumannShapley1974].

**The Aumann-Shapley value as gradient.** The Aumann-Shapley value, an extension of the Shapley value to non-atomic games, is defined as cost per unit

$$\phi_i^{AS}(N, \Lambda, r) = k_i^{AS} = \int_0^1 \frac{\partial r}{\partial \lambda_i}(\gamma \Lambda)\,d\gamma,$$

that is, an average of portfolio $i$'s marginal costs as the activity level increases uniformly from $0$ to $\Lambda$ [Denault2001][AumannShapley1974]. Since the partial derivative of a function homogeneous of degree $k$ is homogeneous of degree $k-1$, and $r$ is homogeneous of degree one, the integral simplifies and

$$k_i^{AS} = \frac{\partial r(\Lambda)}{\partial \lambda_i}, \qquad \phi(N,\Lambda,r)^{AS} = k^{AS} = \nabla r(\Lambda).$$

The per-unit allocation vector is thus the *gradient* of $r$ evaluated at the full-presence level; the allocated capital is $K^{AS} = k^{AS} * \Lambda$ (component-wise product) [Denault2001]. This is the *gradient principle*.

**Coherence and uniqueness.** Aubin shows that the fuzzy core of a game with positively homogeneous $r$ coincides with the sub-differential $\partial r(\Lambda)$; if $r$ is convex (as well as homogeneous) the fuzzy core is non-empty, convex, and compact, and if $r$ is differentiable at $\Lambda$ the core reduces to the single vector $\nabla r(\Lambda)$ [Denault2001][Aubin1981]. The main practical result follows:

> If $(N, r, \Lambda)$ is a game with fractional players, with $r$ a coherent cost function differentiable at $\Lambda$, then the Aumann-Shapley value is a coherent fuzzy value [Denault2001].

The proof uses the fact that, under positive homogeneity, $r$ is sub-additive if and only if it is convex [Denault2001]. The Aumann-Shapley value is in fact the only *linear* coherent allocation principle when the cost function is suitably differentiable; linearity, however, is not required by coherence, since the additive part conflicts with the riskless property — a coherent risk measure cannot be the sum of two coherent measures [Denault2001].

**The Euler principle.** The feasibility of the allocation vector follows directly from Euler's theorem, which is why the principle $\nabla r(\Lambda)$ has been called the *Euler principle* [Denault2001][PatrikEtAl1999]. Denault emphasizes the role of the measure's properties: sub-additivity is necessary for the existence of an allocation without undercut; homogeneity ensures the simple form $\nabla r(\Lambda)$; both serve to prove the non-emptiness of the core; the riskless property is central to the definition of the riskless allocation [Denault2001]. There is full convergence with Tasche's characterization via RORAC: given differentiability conditions on the measure, the correct way to allocate capital is via Aumann-Shapley prices, which turn out to be the only satisfactory allocation principle [Denault2001][Tasche1999]. When the measure is Expected Shortfall, the coherent contribution turns out itself to be of shortfall type, $K_i = \mathrm{E}[-X_i \mid \sum_i X_i \leq q_\alpha]$, in agreement with the formulas of the previous section [Denault2001]. Another axiomatic approach to allocation, with a diversification axiom, also concludes that the Euler principle is the only one compatible [Kalkbrener2005].

**Non-negativity.** The problem of non-negativity remains unresolved: a portfolio may well have negative risk measure, and nothing in itself justifies requiring the allocated capital to be non-negative; allocating a negative amount, however, becomes problematic when used in a RAPM ratio of the type return/allocated capital [Denault2001]. If no portfolio in the institution ever reduces the risk measure by joining a subset, then the Shapley value (and, under the equivalent condition, the Aumann-Shapley prices) is necessarily non-negative [Denault2001].

## Risk Budgets and Limits: Existence and Uniqueness (Bruder-Roncalli)

Assigning risk budgets to the components and realizing them is the inverse operation of attribution: one goes from given contributions to the portfolio that produces them. Let $x = (x_1, \dots, x_n)$ be the vector of exposures (or weights) and $\mathcal{R}(x_1, \dots, x_n)$ a risk measure. If the measure is coherent and convex, it satisfies the Euler decomposition, and it is natural to define the risk contribution of component $i$ [BruderRoncalli2012][ArtznerEtAl1999]:

$$\mathcal{R}(x) = \sum_{i=1}^{n} x_i \frac{\partial \mathcal{R}(x)}{\partial x_i}, \qquad \mathrm{RC}_i(x) = x_i \frac{\partial \mathcal{R}(x)}{\partial x_i}.$$

**Definition of the risk-budgeting portfolio.** Given a set of budgets $\{b_1, \dots, b_n\}$, the risk budgeting (RB) portfolio is defined by the constraints

$$\mathrm{RC}_i(x_1, \dots, x_n) = b_i, \qquad i = 1, \dots, n.$$

Unlike the weight-budgeting portfolio ($x_i = b_i$), the RB portfolio requires solving a *system of non-linear equations* [BruderRoncalli2012]. With volatility as the measure, $\mathcal{R}(x) = \sqrt{x^\top \Sigma x}$, the risk contribution is $\mathrm{RC}_i(x) = x_i (\Sigma x)_i / \sqrt{x^\top \Sigma x}$, which satisfies $\sum_i \mathrm{RC}_i(x) = \sigma(x)$ [BruderRoncalli2012]. In a Gaussian world, managing exposures with VaR or Expected Shortfall is equivalent to managing them with volatility, since $\mathrm{VaR}(x;\alpha) = \Phi^{-1}(\alpha)\sqrt{x^\top \Sigma x}$ and ES is a multiple of it, so the measures differ only by a multiplicative constant and the *relative* contributions coincide [BruderRoncalli2012].

**Specifying limits as constraints.** The system of equalities is too general to define a portfolio of managerial interest. It is preferable to express weights and budgets in relative terms and impose a long-only book, so that the "proper" specification of the RB portfolio becomes the system of constraints on the contributions

$$\begin{cases} x_i \cdot (\Sigma x)_i = b_i \cdot (x^\top \Sigma x) \\ b_i \geq 0, \quad x_i \geq 0 \\ \sum_{i=1}^{n} b_i = 1, \quad \sum_{i=1}^{n} x_i = 1 \end{cases}$$

[BruderRoncalli2012]. Specifying a negative risk contribution would imply that risk is strongly concentrated in the other components; imposing non-negative budgets and weights and their normalization thus translates exposure and concentration limits into direct constraints on the contributions [BruderRoncalli2012].

**Zero budgets.** A difficulty arises when some budget is set to zero: $\mathrm{RC}_k = 0$ admits the solution $x_k = 0$ but also a solution with $x_k > 0$, which however requires negative correlations $\sum_{j\neq k} x_j \rho_{k,j}\sigma_j < 0$ [BruderRoncalli2012]. Since an investor who zeroes out a budget expects not to hold the asset, one must impose the strict constraint $b_i > 0$ and, to handle zero budgets, preliminarily reduce the universe by excluding the corresponding assets; the specification becomes

$$x^\star = \left\{ x \in [0,1]^n : \sum_{i=1}^n x_i = 1,\; x_i \cdot (\Sigma x)_i = b_i \cdot (x^\top \Sigma x) \right\}, \quad b \in\, ]0,1]^n,\; \sum_{i=1}^n b_i = 1.$$

[BruderRoncalli2012]. With zero budgets one generally generates up to $2^m$ solutions, where $m$ is the number of assets for which both marginal risk and budget are zero, and the only acceptable solution is the one obtained as the limit of perturbing the budgets with $b_i = \varepsilon_i \to 0$ [BruderRoncalli2012].

**Existence and uniqueness.** The RB portfolio can be found with an SQP algorithm, but existence and uniqueness are better demonstrated by reformulating the problem as

$$y^\star = \arg\min \sqrt{y^\top \Sigma y} \quad \text{s.t.} \quad \sum_{i=1}^{n} b_i \ln y_i \geq c, \;\; y \geq 0,$$

with $c$ an arbitrary constant; the RB portfolio is then obtained by normalizing, $x_i^\star = y_i^\star / \sum_i y_i^\star$ [BruderRoncalli2012]. The logarithmic constraint is convex and the objective function is convex: the problem is the minimization of a convex function with a convex constraint, so the RB portfolio *exists and is unique* provided the covariance matrix $\Sigma$ is positive definite and $b_i > 0$ [BruderRoncalli2012]. The Kuhn-Tucker conditions of the Lagrangian give $y_i\,\partial\sigma(y)/\partial y_i = \lambda_c b_i$, that is, contributions proportional to the budgets, and the constraint's multiplier is strictly positive because the constraint is necessarily active [BruderRoncalli2012]. The tools of convex optimization, duality, and KKT conditions are treated in [[14 Ottimizzazione convessa - coni, dualità e KKT]].

**Positioning between minimum variance and weight budgeting.** Considering the parametric version

$$x^\star(c) = \arg\min \sqrt{x^\top \Sigma x} \quad \text{s.t.} \quad \sum_{i=1}^n b_i \ln x_i \geq c, \;\; \mathbf{1}^\top x = 1, \;\; x \geq 0,$$

it is shown that the volatility $\sigma(x^\star(c))$ is an increasing function of $c$, with two polar cases: $c = -\infty$ gives the minimum-variance portfolio, $c = \sum_i b_i \ln b_i$ gives the weight-budgeting portfolio. The volatility of the RB portfolio is thus between the two [BruderRoncalli2012]:

$$\sigma_{\mathrm{mv}} \leq \sigma_{\mathrm{rb}} \leq \sigma_{\mathrm{wb}}.$$

The volatility of the risk-budgeting portfolio lies between that of the minimum-variance and that of the weight-budgeting portfolio [BruderRoncalli2012].

**Optimality and performance budgets.** Like the ERC portfolio — whose construction is in [[17 Risk budgeting e risk parity]] — risk budgeting is a heuristic method, but it connects to optimality. For the quadratic utility function $x^\top \mu - \phi\, x^\top \Sigma x$ associated with Markowitz's criterion, portfolio $x$ is optimal if $\tilde\mu = (2/\phi)\Sigma x$; the performance contribution of asset $i$ is then

$$\mathrm{PC}_i = x_i \tilde\mu_i = \frac{2}{\phi} x_i (\Sigma x)_i \propto b_i.$$

The specification of risk budgets thus determines not only how much risk to invest in an asset, but also how much expected performance to attribute to it — the performance contributions turn out to be proportional to the risk budgets [BruderRoncalli2012]. This links Bruder-Roncalli's analysis to Qian's optimal case, in which risk budgets and return budgets coincide [Qian2006][Sharpe2002].

**Beta interpretation.** The covariance between the returns of asset $i$ and the portfolio is $(\Sigma x)_i$; the beta $\beta_i = (\Sigma x)_i / (x^\top \Sigma x)$ measures the asset's sensitivity to the systematic risk represented by the portfolio. Since $\mathrm{RC}_i = x_i \beta_i \sigma(x)$, from the condition $\mathrm{RC}_i = b_i$ it follows that

$$x_i = \frac{b_i \beta_i^{-1}}{\sum_{j=1}^n b_j \beta_j^{-1}},$$

so the weight is inversely proportional to the beta, although the relation is endogenous since $\beta_i$ in turn depends on the portfolio [BruderRoncalli2012]. In ex-ante monitoring of a book, controlling the contributions $\mathrm{RC}_i = x_i \beta_i \sigma(x)$ thus means jointly controlling the weight, beta, and volatility of each component relative to the overall risk.

## References

- **[AcerbiTasche2002]** Acerbi, C. & Tasche, D. (2002). On the coherence of expected shortfall. Journal of Banking & Finance, 26(7), 1487-1503.
- **[ArtznerEtAl1999]** Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). Coherent measures of risk. Mathematical Finance, 9(3), 203-228.
- **[Aubin1981]** Aubin, J.-P. (1981). Cooperative fuzzy games. Mathematics of Operations Research, 6(1), 1-13.
- **[AumannShapley1974]** Aumann, R. J. & Shapley, L. S. (1974). Values of Non-Atomic Games. Princeton University Press.
- **[BruderRoncalli2012]** Bruder, B. & Roncalli, T. (2012). Managing Risk Exposures using the Risk Budgeting Approach. Working Paper, Lyxor Asset Management / Université d'Évry val d'Essonne. MPRA Paper No. 37246.
- **[Denault2001]** Denault, M. (2001). Coherent allocation of risk capital. Journal of Risk, 4(1), 1-34.
- **[GourierouxEtAl2000]** Gouriéroux, C., Laurent, J. P. & Scaillet, O. (2000). Sensitivity analysis of Values at Risk. Journal of Empirical Finance, 7(3-4), 225-245.
- **[Hallerbach2002]** Hallerbach, W. G. (2002). Decomposing Portfolio Value-at-Risk: A General Analysis. Journal of Risk, 5(2), 1-18.
- **[Kalkbrener2005]** Kalkbrener, M. (2005). An axiomatic approach to capital allocation. Mathematical Finance, 15(3), 425-437.
- **[Litterman1996]** Litterman, R. (1996). Hot Spots and Hedges. The Journal of Portfolio Management, 22(5), 52-75.
- **[MaillardEtAl2010]** Maillard, S., Roncalli, T. & Teiletche, J. (2010). The Properties of Equally Weighted Risk Contributions Portfolios. The Journal of Portfolio Management, 36(4), 60-70.
- **[MinaUlmer1999]** Mina, J. & Ulmer, A. (1999). Delta-Gamma Four Ways. RiskMetrics Group, New York.
- **[PatrikEtAl1999]** Patrik, G., Bernegger, S. & Rüegg, M. (1999). The use of risk adjusted capital to support business decision making. Casualty Actuarial Society Forum, Spring, 243-334.
- **[Qian2006]** Qian, E. (2006). On the Financial Interpretation of Risk Contribution: Risk Budgets Do Add Up. Journal of Investment Management, 4(4), Fourth Quarter.
- **[RockafellarUryasev2002]** Rockafellar, R. T. & Uryasev, S. (2002). Conditional Value-at-Risk for general loss distributions. Journal of Banking & Finance, 26(7), 1443-1471.
- **[Shapley1953]** Shapley, L. S. (1953). A value for n-person games. In Kuhn, H. W. & Tucker, A. W. (eds.), Contributions to the Theory of Games, Volume II, Annals of Mathematics Studies 28, Princeton University Press, 307-317.
- **[Sharpe2002]** Sharpe, W. F. (2002). Budgeting and Monitoring Pension Fund Risk. Financial Analysts Journal, 58(5), 74-86.
- **[Tasche1999]** Tasche, D. (1999). Risk contributions and performance measurement. Working paper, Technische Universität München.
- **[Tasche2008]** Tasche, D. (2008). Capital Allocation to Business Units and Sub-Portfolios: the Euler Principle. arXiv preprint q-fin/0708.2542v3. (Also in: Resti, A. (ed.), Pillar II in the New Basel Accord, Risk Books, 2008, pp. 423-453.)
