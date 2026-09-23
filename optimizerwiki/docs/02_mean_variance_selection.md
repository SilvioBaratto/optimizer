---
title: "Mean-Variance Selection"
chapter: 2
tags:
  - optimizer
  - tipo/capitolo
grounded: false
generated: 2026-07-10
---

> [!abstract] Summary
> The chapter formulates the portfolio selection problem in the mean-variance plane following Markowitz and derives its solution, from the two-asset case — with its three correlation sub-cases — to the general case with N assets, showing that the frontier is a parabola (or hyperbola) and that the problem admits a unique solution. It then introduces the risk-free asset, the capital allocation line, the Sharpe ratio and the tangency portfolio, the two mutual fund theorem and its extensions. It closes with the decomposition of portfolio volatility into the contributions of individual assets and the marginal contribution to risk, the quantities on which risk budgeting is based.

## The Mean-Variance Selection Problem

Modern portfolio theory originates with the formulation of the selection problem due to Markowitz [Markowitz1952], who introduced the mean-variance choice logic that later became one of the leading optimization methods used by professional investors [BerkDeMarzo2019]. Many of the same ideas were developed independently, and in the same year, by Roy [Roy1952], and some were anticipated by de Finetti [deFinetti1940].

The investment choice problem consists in determining the optimal allocation of wealth among the available investment opportunities; its solution, in the static single-period formulation, is known as portfolio selection theory [Merton1990]. The analysis rests on a set of assumptions: a frictionless market (no transaction costs or taxes), price-taking behavior on the part of the investor, and no institutional constraints [Merton1990].

The choice is built on two quantities. Since the return on an investment is a random variable, its expected value is taken as a measure of profitability and its variance as a measure of riskiness. For a risk-averse investor, von Neumann–Morgenstern expected utility $U(X)$ can be expressed as a function of the mean and variance of the return alone, $\mathbb{E}[U(X)] =: V(r_X,\sigma_X^2)$, with
$$\frac{\partial V(r_X,\sigma_X^2)}{\partial r_X} > 0, \qquad \frac{\partial V(r_X,\sigma_X^2)}{\partial \sigma_X^2} < 0,$$
that is, utility increases with expected return and decreases with variance [Ingersoll1987]. The conditions under which expected utility depends exactly on only two moments — jointly normal returns or quadratic utility — and the foundations of choice under uncertainty are discussed in [[01 Scelta in condizioni di incertezza]].

From this setup follows the definition of an efficient portfolio. As Ingersoll points out, the class of portfolios potentially optimal for such investors is that of portfolios with maximum expected return for a given level of variance and, simultaneously, minimum variance for a given expected return: portfolios said to be mean-variance efficient [Ingersoll1987]. Equivalently, portfolio $i$ dominates portfolio $j$ if
$$\mathbb{E}(r_i) \ge \mathbb{E}(r_j) \quad \text{and} \quad \sigma_i \le \sigma_j,$$
with at least one strict inequality; a portfolio is inefficient whenever there exists another feasible allocation that is better both in expected return and in volatility [BerkDeMarzo2019]. In the mean-standard deviation plane the preferred direction is "northwest" (higher return, lower risk).

The full path unfolds in three stages: identify the efficient frontier of risky assets alone; identify the efficient frontier in the presence of a risk-free asset, together with the associated tangency portfolio; finally, choose the optimal portfolio along that frontier based on the investor's degree of risk aversion. The first two stages do not depend on individual preferences but only on the mean-variance criterion; the third does [BerkDeMarzo2019].

## Returns, Covariance, and Correlation

A portfolio is described by its weights $x_i$, the fraction of total value invested in each asset,
$$x_i = \frac{\text{Value of investment } i}{\text{Total portfolio value}}, \qquad \sum_{i=1}^N x_i = 1,$$
where a negative weight represents a short position [BerkDeMarzo2019]. The arithmetic return is additive across assets: the portfolio return is the weighted average of the individual asset returns with weights equal to the portfolio weights,
$$R_P = \sum_{i=1}^N x_i R_i.$$
Since the expected value is a linear operator, the expected return of the portfolio is the weighted average of the expected returns,
$$\mathbb{E}(R_P) = \sum_{i=1}^N x_i r_i =: r_P = \mathbf{x}'\mathbf{r}.$$

Variance, on the other hand, is not a linear operator. Expanding $\mathbb{V}\mathrm{ar}(R_P) = \mathbb{C}\mathrm{ovar}(R_P,R_P)$ yields
$$\sigma_P^2 = \sum_{i=1}^N x_i^2\sigma_i^2 + \sum_{i=1}^N\sum_{j=1,\,j\neq i}^N x_ix_j\sigma_{i,j} = \sum_{i=1}^N x_i^2\sigma_i^2 + 2\sum_{i=1}^N\sum_{j=i+1}^N x_ix_j\sigma_{i,j},$$
which in matrix notation is written $\sigma_P^2 = \mathbf{x}'\mathbf{V}\mathbf{x}$, with $\mathbf{V}$ the variance-covariance matrix [Merton1990]. The covariance
$$\sigma_{i,j} = \mathbb{C}\mathrm{ovar}(R_i,R_j) = \mathbb{E}\big[(R_i-\mathbb{E}(R_i))(R_j-\mathbb{E}(R_j))\big] = \mathbb{E}(R_iR_j)-\mathbb{E}(R_i)\mathbb{E}(R_j)$$
measures the co-movement of returns: it is positive if the two assets tend to deviate from their means in the same direction, negative in the opposite case, and zero in the absence of a systematic tendency [BerkDeMarzo2019]. The covariance of a variable with itself is its variance, $\mathbb{C}\mathrm{ovar}(R_i,R_i)=\sigma_i^2$.

The sign of the covariance is easy to interpret, its magnitude is not: it grows both with the volatility of the assets and with the strength of the relationship. To isolate the latter, it is normalized using the Bravais–Pearson linear correlation coefficient
$$\rho_{i,j} = \frac{\sigma_{i,j}}{\sigma_i\sigma_j} \in [-1,1],$$
which equals $+1$ when returns always move in the same direction, $-1$ when they always move in opposite directions, and $0$ when they are uncorrelated [BerkDeMarzo2019]. Recalling that $\sigma_{i,j}=\rho_{i,j}\sigma_i\sigma_j$ and that $\rho_{i,i}=1$, portfolio variance admits the compact double-summation form
$$\sigma_P^2 = \sum_{i=1}^N\sum_{j=1}^N x_ix_j\rho_{i,j}\sigma_i\sigma_j = \sum_{i=1}^N\sum_{j=1}^N x_ix_j\sigma_{i,j}.$$
Returns tend to be more correlated the more similarly they are affected by economic events: assets in the same sector typically show higher correlations than assets in different sectors [BerkDeMarzo2019].

## Estimating Parameters from Historical Returns

The true means and (co)variances are not observable, since the states of the world and their probabilities are unknown: means and (co)variances must be estimated from realized returns. Treating each observation in the historical series as an equiprobable scenario, $\pi_s = 1/T$, yields the sample estimators
$$\hat\mu_i = \frac{1}{T}\sum_{t=1}^T r_{it}, \qquad \hat\sigma_i^2 = \frac{1}{T}\sum_{t=1}^T (r_{it}-\hat\mu_i)^2, \qquad \widehat{\mathrm{cov}}(r_i,r_j) = \frac{1}{T}\sum_{t=1}^T (r_{it}-\hat\mu_i)(r_{jt}-\hat\mu_j).$$
Since the mean is itself typically estimated, a degrees-of-freedom correction is applied (negligible for large $T$) that makes the estimators unbiased, replacing $T$ with $T-1$ in the denominator of variance and covariance [BerkDeMarzo2019]:
$$\hat\sigma_i^2 = \frac{1}{T-1}\sum_{t=1}^T (r_{it}-\bar r_i)^2, \qquad \widehat{\mathrm{cov}}(r_i,r_j) = \frac{1}{T-1}\sum_{t=1}^T (r_{it}-\bar r_i)(r_{jt}-\bar r_j).$$

Alongside the arithmetic mean, an unbiased estimator of expected return per period, sits the geometric mean $\bar r_{i,\text{geom}} = TV_i^{1/T}-1$, which accounts for compounding and accurately represents past performance. The frequency of the data (daily versus monthly) matters only for the estimation of (co)variances — whose accuracy increases with frequency — and not for mean returns: in the absence of serial correlation, per-period variances add up, so that $\sigma^2_{\text{annual}} = 12\,\sigma^2_{\text{monthly}}$ and the standard deviation grows at rate $\sqrt{T}$ [BerkDeMarzo2019]. Reliable input estimates are decisive for implementing mean-variance optimization: if returns are stationary and the sample is sufficiently long the estimates are accurate, but very old returns may no longer be representative. The treatment of estimation error is developed further in [[08 Errore di stima, outlier e shrinkage]].

Regarding the choice between arithmetic and logarithmic returns, note that the logarithmic return $r^*_t = \ln[(P_t+D)/P_{t-1}]$ approximates the arithmetic one to first order for small returns, since $\ln(1+r)\approx r$; logarithmic returns are additive over time (suitable for analyzing historical series), but the logarithmic return of a portfolio is not linear in the returns of its assets, whereas arithmetic returns are additive across assets and therefore suitable for portfolio analysis [BerkDeMarzo2019].

## The Case N = 2 and the Correlation Sub-Cases

The role of correlation emerges clearly in the simplest possible portfolio, the one with only two risky investment choices $X_1$ and $X_2$, with returns $R_1,R_2$, means $r_1,r_2$, variances $\sigma_1^2,\sigma_2^2$, and covariance $\sigma_{1,2}=\rho_{1,2}\sigma_1\sigma_2$. Assume $r_1<r_2$ and $\sigma_1^2<\sigma_2^2$. With $x_1+x_2=1$ and $x_2=1-x_1$,
$$r_P = x_1r_1+(1-x_1)r_2, \qquad \sigma_P^2 = x_1^2\sigma_1^2+(1-x_1)^2\sigma_2^2+2x_1(1-x_1)\rho_{1,2}\sigma_1\sigma_2.$$
With weights in $[0,1]$ the first two terms do not exceed $\sigma_1^2$ and $\sigma_2^2$ respectively, while the third is negative if $\rho_{1,2}\in[-1,0)$, zero if $\rho_{1,2}=0$, and positive if $\rho_{1,2}\in(0,1]$: the sign and magnitude of the covariance thus determine whether, and by how much, diversification reduces variance relative to that of the individual assets. Portfolio volatility is lower than the weighted average of the individual volatilities whenever $\rho_{1,2}<1$, and this is the benefit of diversification [BerkDeMarzo2019]. The numerical examples that follow all use $r_1=0{,}0125$, $\sigma_1^2=0{,}0400$, $r_2=0{,}0305$, $\sigma_2^2=0{,}0900$.

**Sub-case $\rho_{1,2}=+1$ (perfect positive correlation).** In this case $\sigma_P^2 = [x_1\sigma_1+(1-x_1)\sigma_2]^2$, with both terms non-negative for $x_1\in[0,1]$, from which
$$\sigma_P = x_1\sigma_1+(1-x_1)\sigma_2.$$
Solving for $x_1=(r_P-r_2)/(r_1-r_2)$ from the mean expression and substituting yields the frontier
$$r_P = \frac{r_2\sigma_1-r_1\sigma_2}{\sigma_1-\sigma_2} + \frac{r_1-r_2}{\sigma_1-\sigma_2}\,\sigma_P,$$
in which $r_P$ is a positive affine transformation of $\sigma_P$: in the $(\sigma,r)$ plane the two assets are joined by a straight-line segment, with no contraction of volatility and no diversification benefit. Allowing short sales extends the line in both directions; imposing $\sigma_P=0$ yields the risk-free portfolio $x_1=\sigma_2/(\sigma_2-\sigma_1)>1$, $x_2=-\sigma_1/(\sigma_2-\sigma_1)<0$, attainable only by short-selling the more volatile asset [BerkDeMarzo2019]. In the example the frontier is the segment joining $X_1$ ($\mathrm{SD}\approx0{,}20$) to $X_2$ ($\mathrm{SD}\approx0{,}30$).

**Sub-case $\rho_{1,2}=-1$ (perfect negative correlation).** Now $\sigma_P^2 = [x_1\sigma_1-(1-x_1)\sigma_2]^2$, that is, $\sigma_P = |x_1\sigma_1-(1-x_1)\sigma_2|$. Volatility vanishes at
$$\mathbf{x} = \Big(\tfrac{\sigma_2}{\sigma_1+\sigma_2},\ \tfrac{\sigma_1}{\sigma_1+\sigma_2}\Big),$$
regardless of $\sigma_1^2$ and $\sigma_2^2$: it is possible to construct a zero-risk portfolio (perfect hedge) using long positions only [BerkDeMarzo2019]. The frontier splits into two half-lines that meet on the $\sigma=0$ axis:
$$r_P = \frac{r_2\sigma_1+r_1\sigma_2}{\sigma_1+\sigma_2} \pm \frac{r_1-r_2}{\sigma_1+\sigma_2}\,\sigma_P,$$
whose branch with the higher return for a given level of risk is efficient and the other — toward the lower-return asset — is inefficient, being dominated. In the example the intercept at near-zero risk is located at $r_P\approx0{,}0197$.

**Sub-case $\rho_{1,2}\in(-1,1)$ (intermediate correlations).** For suitable values of $\rho_{1,2}$ there exist portfolios with
$$\sigma_P^2 < \min\{\sigma_1^2,\sigma_2^2\},$$
that is, with variance below the smaller of the two individual variances. As $\rho_{1,2}$ decreases, the curve joining the two assets in the mean-risk plane bows increasingly to the left, increasing the diversification benefit; as $\rho_{1,2}$ increases toward $1$ it flattens toward the straight-line segment, the limiting case with no diversification. Correlation, however, has no effect on the expected return, which remains linear in the weights [BerkDeMarzo2019].

## The General Case N ≥ 2: Convexity and the Solution Theorem

In its basic version, the selection problem minimizes variance for a given target expected return $\pi$, subject to the budget constraint:
$$\min_{x_1,\dots,x_N} \mathbf{x}'\mathbf{V}\mathbf{x} \quad \text{s.t.} \quad \begin{cases} \mathbf{x}'\mathbf{r}=\pi\\ \mathbf{x}'\mathbf{e}=1,\end{cases}$$
where $\mathbf{e}$ is the unit vector. Equivalently, one can maximize expected return for a given level of variance; repeating the procedure for a sufficient number of target values traces out the entire frontier [BerkDeMarzo2019]. Technically, a convex function is being minimized subject to linear constraints: $\mathbf{x}'\mathbf{V}\mathbf{x}$ is convex because $\mathbf{V}$ is positive definite, and the two linear constraints define a convex set; the problem therefore has a unique solution and it suffices to derive the first-order conditions [ConstantinidesMalliaris1995]. Indeed, if returns are non-degenerate ($\sigma_i^2>0$ for every $i$), then $\partial^2(\mathbf{x}'\mathbf{V}\mathbf{x})/\partial x_i^2 = 2\sigma_i^2>0$ and the quadratic form, being a variance, is positive for every $\mathbf{x}\neq\mathbf{0}_N$.

**Theorem.** Let $\mathbf{V}$ be the $N\times N$ variance-covariance matrix and $\mathbf{r}$ the vector of means. If $\mathbf{V}$ is nonsingular and positive definite and $r_i\neq r_j$ for some pair, then the problem has the unique solution
$$\mathbf{x}^* = \frac{(\gamma\mathbf{V}^{-1}\mathbf{r}-\beta\mathbf{V}^{-1}\mathbf{e})\,\pi + (\alpha\mathbf{V}^{-1}\mathbf{e}-\beta\mathbf{V}^{-1}\mathbf{r})}{\alpha\gamma-\beta^2},$$
where $\alpha = \mathbf{r}'\mathbf{V}^{-1}\mathbf{r}$, $\beta = \mathbf{r}'\mathbf{V}^{-1}\mathbf{e}$, $\gamma = \mathbf{e}'\mathbf{V}^{-1}\mathbf{e}$ [ConstantinidesMalliaris1995].

*Proof.* Form the Lagrangian $\mathfrak{L} = \mathbf{x}'\mathbf{V}\mathbf{x} - \lambda_1(\mathbf{x}'\mathbf{r}-\pi) - \lambda_2(\mathbf{x}'\mathbf{e}-1)$, whose first-order conditions are
$$2\mathbf{x}'\mathbf{V} - \lambda_1\mathbf{r}' - \lambda_2\mathbf{e}' = \mathbf{0}_N, \qquad \mathbf{x}'\mathbf{r}=\pi, \qquad \mathbf{x}'\mathbf{e}=1.$$
From the first, since $\mathbf{V}$ is nonsingular, $\mathbf{x}' = \tfrac{1}{2}\lambda_1\mathbf{r}'\mathbf{V}^{-1} + \tfrac{1}{2}\lambda_2\mathbf{e}'\mathbf{V}^{-1}$; substituting into the other two gives $\tfrac{1}{2}\lambda_1 = (\pi\gamma-\beta)/(\alpha\gamma-\beta^2)$ and $\tfrac{1}{2}\lambda_2 = (\alpha-\pi\beta)/(\alpha\gamma-\beta^2)$, from which, substituting back, the expression for $\mathbf{x}^*$ follows. $\square$

The variance of the optimal portfolio is a function of the target return:
$$\sigma_{P^*}^2 = \mathbf{x}^{*\prime}\mathbf{V}\mathbf{x}^* = \frac{\gamma\pi^2-2\beta\pi+\alpha}{\alpha\gamma-\beta^2},$$
which describes a **parabola** in the variance-mean plane and, taking its square root, a **hyperbola** in the standard-deviation-mean plane [ConstantinidesMalliaris1995]. The vertex of the parabola has coordinates $(\sigma_{P,v}^2 = 1/\gamma,\ r_{P,v}=\beta/\gamma)$; that of the hyperbola $(\sigma_{P,v}=\sqrt{1/\gamma},\ r_{P,v}=\beta/\gamma)$. This vertex is the **global minimum-variance portfolio (GMVP)**, the portfolio with the lowest variance overall, which separates the inefficient (lower) branch from the efficient (upper) one; its weights do not depend on the expected returns [BerkDeMarzo2019]. Only portfolios with expected return above that of the GMVP are efficient.

With three or more assets the set of feasible portfolios forms a region: the portfolios on its boundary are the minimum-variance portfolios, and the part above the GMVP constitutes the efficient frontier. Adding new investment opportunities expands the diversification possibilities and improves the frontier: the old frontier is contained within the new one [BerkDeMarzo2019]. A notable fact is that individually dominated assets can still receive positive weight in frontier portfolios thanks to their low or negative correlation with the others, which reduces overall variance. Markowitz showed precisely that it is an asset's covariance with the investor's portfolio that determines its incremental risk, so that the risk of an investment cannot be assessed in isolation [BerkDeMarzo2019]. In the presence of constraints, such as non-negativity, the problem loses its closed form and must be solved numerically; this aspect is taken up in [[06 Vincoli e metaeuristiche]].

## Diversification: Market Risk and Specific Risk

The number $N$ of assets plays a significant role in determining portfolio variance. This can be seen by considering the equal-weighted portfolio, with $x_i=1/N$. Substituting into the variance expression and introducing the average variance $\bar\sigma^2 = \tfrac{1}{N}\sum_{i=1}^N\sigma_i^2$ and the average covariance $\overline{\mathrm{Cov}} = \tfrac{1}{N(N-1)}\sum_{i}\sum_{j\neq i}\sigma_{i,j}$, one obtains
$$\sigma_P^2 = \frac{1}{N}\bar\sigma^2 + \frac{N-1}{N}\overline{\mathrm{Cov}}.$$
As $N$ grows, the weight of the average variance vanishes and that of the average covariance tends to one, so that — as long as variances remain bounded — portfolio variance converges to the average covariance,
$$\lim_{N\to+\infty}\sigma_P^2 = \overline{\mathrm{Cov}} [Merton1990][BerkDeMarzo2019].$$
This result holds approximately for any well-diversified portfolio in which each weight is sufficiently small.

This yields a distinction between two components of risk [BerkDeMarzo2019]:

- **firm-specific risk** (diversifiable, idiosyncratic, unsystematic), which can be eliminated through diversification;
- **market risk** (systematic, non-diversifiable), which persists even in a very large portfolio.

If risks were independent ($\overline{\mathrm{Cov}}=0$), variance could be reduced to zero: the volatility of the equal-weighted portfolio of $n$ independent, identical risks with standard deviation $\sigma$ is $\sigma/\sqrt{n}$, which vanishes as $n\to\infty$. In general, however, equity returns show positive correlations, and the irreducible risk of a diversified portfolio is determined by the average covariance of returns [BerkDeMarzo2019]. With a typical volatility of $40\%$ and a typical correlation of $25\%$ between assets, the volatility of the equal-weighted portfolio
$$SD(R_P) = \sqrt{\tfrac{1}{n}(0{,}40^2)+\big(1-\tfrac{1}{n}\big)(0{,}25\times0{,}40\times0{,}40)}$$
decreases rapidly as $n$ increases — nearly half of the volatility of the individual assets is eliminated, and most of the benefit is achieved with as few as thirty assets — but it converges to $\sqrt{0{,}25\times0{,}40\times0{,}40}=20\%$, not to zero. The benefit is more pronounced at the start: the reduction from one to two assets is much larger than that from one hundred to one hundred and one [BerkDeMarzo2019]. Combining more highly correlated assets, for example within the same sector, offers less diversification; international assets, being less correlated, offer more. It is in this sense that diversification constitutes a "free lunch": it reduces risk without sacrificing expected return [BerkDeMarzo2019][Markowitz1952].

## The Risk-Free Asset: Capital Allocation Line, Sharpe Ratio, and Tangency Portfolio

Beyond diversification there is a second way to control risk: keeping part of one's wealth in a risk-free investment. The risk-free asset has zero return variance and is uncorrelated with all other assets; in practice, short-term government securities with high credit quality or money market funds are used [BerkDeMarzo2019].

Consider investing a fraction $x$ in a risky portfolio $P$ and the remaining $1-x$ at the risk-free rate $r_f$. Since the variance and covariance of $r_f$ with $P$ are zero,
$$\mathbb{E}(R_{xP}) = r_f + x\big(\mathbb{E}(R_P)-r_f\big), \qquad SD(R_{xP}) = x\,SD(R_P).$$
As $x$ increases, risk and risk premium rise proportionally, so that in the mean-standard deviation plane the combinations lie on a line starting from the risk-free investment and passing through $P$: eliminating $x$ yields
$$\mathbb{E}(R_C) = r_f + \frac{\mathbb{E}(R_P)-r_f}{\sigma_P}\,\sigma_C.$$
Any line connecting $r_f$ to a risky portfolio $p$ is called a **capital allocation line (CAL)**: it represents the set of combinations obtainable between $r_f$ and $p$ [BerkDeMarzo2019]. For $x=1$ one is fully invested in $P$; for $x>1$ the risk-free asset is sold short — that is, one borrows at rate $r_f$ — giving rise to a leveraged portfolio, riskier than $P$ [BerkDeMarzo2019].

The slope of the CAL through $P$ is the portfolio's **Sharpe ratio**, the ratio of excess return to volatility,
$$SR_P = \frac{\mathbb{E}(R_P)-r_f}{\sigma_P},$$
originally introduced as the reward-to-variability ratio [Sharpe1966]. It measures compensation per unit of volatility. To obtain the highest expected return at every level of volatility, one must identify the risky portfolio that generates the steepest line when combined with the risk-free asset: geometrically, the CAL tangent to the efficient frontier of risky assets. The portfolio that generates this tangent line is the **tangency portfolio**, and it is the one with the maximum Sharpe ratio in the entire economy [BerkDeMarzo2019]. It is found by solving
$$\max_{\mathbf{w}}\ SR_P = \frac{\mathbb{E}(R_P)-r_f}{\sigma_P} \quad \text{s.t.}\quad \sum_{i=1}^N w_i = 1.$$
Every other risky portfolio lies below this line; once the risk-free asset is introduced, the new efficient frontier is no longer the upper branch of the Markowitz hyperbola but the half-line joining $r_f$ to the tangency portfolio, since it dominates the hyperbola at every level of risk [BerkDeMarzo2019]. Its intercept corresponds to full investment in the risk-free asset, the point of tangency to full investment in the risky portfolio; intermediate portfolios contain positive amounts of both, while those beyond the tangency point are leveraged.

## The Two Mutual Fund Theorem

The linear structure of the efficient frontier in the presence of the risk-free asset is formalized in the **two mutual fund theorem**: any portfolio located on the efficient frontier can be constructed as a linear combination of any two efficient portfolios belonging to that same frontier, the "mutual funds" the theorem refers to [BerkDeMarzo2019]. More generally, given two minimum-variance portfolios $\mathbf{w}_a$ and $\mathbf{w}_b$ with $\mathbb{E}(r_a)\neq\mathbb{E}(r_b)$, every minimum-variance portfolio is obtained as a linear combination of them, and conversely every linear combination of $\mathbf{w}_a$ and $\mathbf{w}_b$ is a minimum-variance portfolio; if both are efficient, then $\alpha\mathbf{w}_a+(1-\alpha)\mathbf{w}_b$ is efficient for $0\le\alpha\le1$ [BerkDeMarzo2019]. Setting $\mathbf{w}_c=\alpha\mathbf{w}_a+(1-\alpha)\mathbf{w}_b$, its expected return is the weighted average $\mathbb{E}(r_c)=\alpha\mathbb{E}(r_a)+(1-\alpha)\mathbb{E}(r_b)$, and the weights of a frontier portfolio are linear functions of its expected return, with
$$\alpha = \frac{\mathbb{E}(r_c)-\mathbb{E}(r_b)}{\mathbb{E}(r_a)-\mathbb{E}(r_b)}.$$
If the target portfolio lies between the two funds, both are held in positive amounts; if it lies outside, one of the two is sold short and the position in the other exceeds the available capital, with the excess financed by the short position [BerkDeMarzo2019].

The most striking consequence concerns the choice in the presence of the risk-free asset. If the two funds being combined are the risk-free asset and the tangency portfolio, then — since the efficient frontier coincides with the tangent CAL — all investors hold the same risky tangency portfolio, regardless of their degree of risk aversion. The only subjective choice is the allocation of wealth between the risk-free asset and the tangency portfolio: more risk-averse investors allocate a smaller share to it, less risk-averse ones a larger share, possibly resorting to leverage. The selection problem thus splits into two independent stages: determining the tangency portfolio is purely technical, while the allocation of the complete portfolio depends on preferences [BerkDeMarzo2019]. This separation result, which extends Markowitz's construction to the case with a risk-free asset, is due to Tobin [Tobin1958]. Constructing the complete portfolio $C$ with weight $y$ in the tangency portfolio $T$ and $1-y$ in the risk-free asset,
$$\mathbb{E}(R_C) = y\,\mathbb{E}(R_T)+(1-y)r_f, \qquad \sigma_C = y\,\sigma_T,$$
and substituting $y=\sigma_C/\sigma_T$ recovers the equation of the efficient frontier $\mathbb{E}(R_C) = r_f + SR_T\,\sigma_C$, the best attainable CAL. The choice of the optimal share $y$ based on the investor's utility is discussed in [[01 Scelta in condizioni di incertezza]]; the equilibrium implications, when the tangency portfolio coincides with the market portfolio, in [[05 Modelli fattoriali]].

## Different Lending and Borrowing Rates

The construction of the capital allocation line implicitly assumes that one can lend and borrow at the same risk-free rate. In practice the rate at which one lends is typically lower than the rate at which one borrows, and this changes the shape of the efficient frontier [BerkDeMarzo2019].

Suppose one can lend at $r_f=7\%$ but can only borrow at a higher rate $r_f^B=9\%$. Given a risky portfolio $p$ with $\mathbb{E}(R_p)=15\%$ and $\sigma_p=22\%$, the CAL then has two segments with different slopes:
$$\text{lending segment:}\quad SR = \frac{15-7}{22} = \frac{8}{22} \approx 0{,}36, \qquad \text{borrowing segment:}\quad SR = \frac{15-9}{22} = \frac{6}{22} \approx 0{,}27.$$
The CAL changes slope at the point of full investment in the risky portfolio ($y=1$): up to that point one lends at rate $r_f$, beyond it one borrows at the higher rate $r_f^B$, and the line becomes less steep [BerkDeMarzo2019].

As a result, there is no longer a single tangency portfolio valid for all investors, but rather two distinct tangency portfolios: one for the lending segment, constructed with respect to $r_f$, and one for the borrowing segment, constructed with respect to $r_f^B$. The efficient frontier then consists of three parts: the lending half-line up to the first tangency portfolio, a segment of the risky-assets-only frontier connecting the two tangency portfolios, and the borrowing half-line beyond the second [BerkDeMarzo2019].

## Non-Negativity Constraints: The Frontier as a Bounded Arc

The preceding derivations allow negative weights, that is, short sales: short positions extend the set of possible portfolios, and with them the risky-assets-only frontier extends beyond the extreme assets [BerkDeMarzo2019]. A negative weight corresponds to selling today an asset one does not own with the obligation to repurchase it in the future; the weights still sum to one, and short selling can increase expected return but also, substantially, portfolio volatility [BerkDeMarzo2019].

When, instead, non-negativity constraints are imposed — no short sales, $\mathbf{w}\ge0$ — the frontier problem becomes
$$\min_{\mathbf{w}}\ \sigma_p^2 = \sum_{i=1}^N\sum_{j=1}^N w_iw_j\sigma_{ij} \quad \text{s.t.}\quad \sum_{i=1}^N w_i = 1,\ \ \mathbf{w}\ge0,\ \ \sum_{i=1}^N w_i\mathbb{E}(r_i) = \bar\mu_p,$$
which in general does not admit a closed-form solution and must be solved numerically [BerkDeMarzo2019].

The effects of the constraints are already visible in the two-asset case. With $\rho_{1,2}=+1$, in the absence of short sales the frontier is exactly the segment between the two assets; the risk-free portfolio, which would require $x_1>1$ and $x_2<0$, is no longer attainable, and the less risky asset then coincides with the GMVP [BerkDeMarzo2019]. With $\rho_{1,2}=-1$, on the other hand, the zero-risk portfolio is already obtained with weights in $[0,1]$ and remains feasible even under the constraint. In general, imposing non-negativity means the efficient frontier no longer extends indefinitely but becomes a bounded arc: it reaches, as its extremes, the individual assets, unable to exceed, on the upside, the asset with maximum expected return, nor, on the downside, to fall below the GMVP of the feasible set [BerkDeMarzo2019]. The systematic treatment of constraints and the corresponding solution methods are taken up in [[06 Vincoli e metaeuristiche]].

## Volatility Decomposition and Marginal Contribution to Risk

The variance of a portfolio can be written as the weighted average of the covariances of each asset with the portfolio itself:
$$\mathbb{V}\mathrm{ar}(R_P) = \mathbb{C}\mathrm{ovar}(R_P,R_P) = \sum_{i=1}^N x_i\,\mathbb{C}\mathrm{ovar}(R_i,R_P).$$
This form reveals that portfolio risk depends on how each asset's return moves relative to the portfolio [BerkDeMarzo2019]. The quantity $\mathbb{C}\mathrm{ovar}(R_i,R_P)$ measures the marginal increase in variance for a small increase in asset $i$ within the portfolio, financed via a short position in the risk-free asset [BerkDeMarzo2019]. This is the **marginal contribution to risk** of asset $i$: for a well-diversified portfolio, in which the weights are small and similar, overall variance is determined almost entirely by average covariances, not by individual variances.

Expressing covariance through correlation and dividing by the portfolio's standard deviation yields the decomposition of volatility into the contributions of individual assets:
$$SD(R_P) = \sum_{i=1}^N \underbrace{x_i \times SD(R_i) \times \mathrm{Corr}(R_i,R_P)}_{\text{contribution of asset } i}.$$
Each asset contributes to portfolio volatility according to its own total risk $SD(R_i)$, rescaled by its correlation with the portfolio, which selects the fraction of risk common to the portfolio [BerkDeMarzo2019]. In a portfolio with non-negative weights, unless all assets have correlation $+1$ with the portfolio,
$$SD(R_P) = \sum_{i=1}^N x_i\,SD(R_i)\,\mathrm{Corr}(R_i,R_P) < \sum_{i=1}^N x_i\,SD(R_i):$$
the expected return is the weighted average of expected returns, but volatility is strictly lower than the weighted average of volatilities — part of the volatility is eliminated through diversification [BerkDeMarzo2019].

This decomposition also provides the criterion for deciding whether it is worth increasing the share of an asset. Adding asset $i$ financed by the risk-free asset increases expected return by its excess $\mathbb{E}(R_i)-r_f$ and risk by its incremental contribution $SD(R_i)\times\mathrm{Corr}(R_i,R_P)$; the operation improves the Sharpe ratio if
$$\frac{\mathbb{E}(R_i)-r_f}{SD(R_i)} > \mathrm{Corr}(R_i,R_P)\times\frac{\mathbb{E}(R_P)-r_f}{SD(R_P)}.$$
Defining the asset's beta relative to the portfolio
$$\beta_i^P \equiv \frac{SD(R_i)\times\mathrm{Corr}(R_i,R_P)}{SD(R_P)},$$
the condition becomes $\mathbb{E}(R_i) > r_f + \beta_i^P\big(\mathbb{E}(R_P)-r_f\big)$, where the right-hand side is the return required to compensate for the risk that $i$ contributes to the portfolio [BerkDeMarzo2019]. At the optimum, trading continues until the expected return of every asset equals its required return: a portfolio is efficient if and only if
$$\mathbb{E}(R_i) = r_f + \beta_i^{\text{eff}}\big(\mathbb{E}(R_{\text{eff}})-r_f\big)$$
holds for every asset, with $R_{\text{eff}}$ the return of the efficient portfolio [BerkDeMarzo2019]. Since $x_i\,SD(R_i)\,\mathrm{Corr}(R_i,R_P) = x_i\,\beta_i^P\,SD(R_P)$, the risk contributions are apportioned according to the products $x_i\beta_i^P$: it is on this decomposition of overall risk into the contributions of individual assets that risk budgeting rests, that is, the practice of allocating a given risk budget among assets based on their marginal contributions. The role of covariance with the portfolio as a measure of systematic risk, and its connection to the market portfolio, are developed in [[05 Modelli fattoriali]]; risk measures alternative to variance in [[04 Misure di rischio coerenti]].

## References

- **[BerkDeMarzo2019]** Berk, J. & DeMarzo, P. (2019). Corporate Finance. Pearson.
- **[ConstantinidesMalliaris1995]** Constantinides, G. M. & Malliaris, A. G. (1995). Portfolio Theory. In R. A. Jarrow, V. Maksimovic & W. T. Ziemba (Eds.), Finance (Handbooks in Operations Research and Management Science, vol. 9). Amsterdam: Elsevier.
- **[Ingersoll1987]** Ingersoll, J. E. Jr. (1987). Theory of Financial Decision Making. Totowa (NJ): Rowman & Littlefield.
- **[Markowitz1952]** Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77–91.
- **[Merton1990]** Merton, R. C. (1990). Continuous-Time Finance. Oxford: Blackwell.
- **[Roy1952]** Roy, A. D. (1952). Safety First and the Holding of Assets. Econometrica, 20(3), 431–449.
- **[Sharpe1966]** Sharpe, W. F. (1966). Mutual Fund Performance. The Journal of Business, 39(1), 119–138.
- **[Tobin1958]** Tobin, J. (1958). Liquidity Preference as Behavior Towards Risk. The Review of Economic Studies, 25(2), 65–86.
- **[deFinetti1940]** de Finetti, B. (1940). Il problema dei «pieni». Giornale dell'Istituto Italiano degli Attuari, 11, 1–88 (English translation in Journal of Investment Management, 4, 2006).
