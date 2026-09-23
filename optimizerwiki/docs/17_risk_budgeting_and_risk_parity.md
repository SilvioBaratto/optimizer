---
title: "Risk Budgeting and Risk Parity"
chapter: 17
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-11
---

> [!abstract] Summary
> The chapter shifts the object of allocation from capital to risk: starting from the decomposition of volatility into marginal and total contributions, justified as Euler's identity for a homogeneous function of degree one, it defines the equal risk contribution (ERC) portfolio and establishes its existence and uniqueness as the solution of a convex problem with a logarithmic constraint. It shows that the ERC sits — in terms of both volatility and risk concentration — between the minimum-variance portfolio and the equally weighted one, and reports Qian's perspective on risk parity across asset classes, in which equalizing risk contributions corrects the hidden concentration of a classic 60/40.

## From Capital to Risk

Optimal portfolio construction has a long history in the academic literature; more than fifty years ago [Markowitz1952] formalized the problem within a mean-variance framework, in which the rational investor seeks to maximize expected return for a given level of volatility (cf. [[02 Selezione media-varianza]]). As powerful and elegant as it is, this solution suffers from two flaws well known in practice: optimal portfolios tend to concentrate in a limited subset of the available securities, and the mean-variance solution is excessively sensitive to input parameters, particularly to expected returns, where small variations can substantially alter the portfolio's composition [Merton1980]. Alternative methods — the resampling of [Michaud1989], robust asset allocation — introduce an additional computational burden, and a significant share of investors prefer heuristic solutions, simple to implement and considered robust because they do not depend on expected returns [MaillardRoncalliTeiletche2010].

Two well-known examples of such techniques are the minimum-variance (MV) portfolio and the equally weighted one. The former is the only portfolio on the efficient frontier that does not incorporate information about expected returns, and for this reason it is recognized as robust; however, it suffers from the concentration flaw. A simple and natural way to address this problem is to assign the same weight to all securities: equally weighted, or $1/n$, portfolios are widely used in practice and have proven efficient out of sample [DeMiguelEtAl2009]. Their flaw is that they can lead to very limited risk diversification if individual risks differ significantly [MaillardRoncalliTeiletche2010].

The idea running through this chapter is a heuristic intermediate between these two extremes: equalizing the risk contributions coming from the different components of the portfolio [MaillardRoncalliTeiletche2010]. This means allocating risk rather than capital. The risk contribution of a component $i$ is the share of the portfolio's total risk attributable to that component, and is calculated as the product of the share allocated to component $i$ and its marginal contribution to risk, the latter given by the change in the portfolio's total risk induced by an infinitesimal increase in position $i$. Managing portfolios in terms of risk contributions has become standard practice for institutional investors under the label of \emph{risk budgeting}: the analysis of the portfolio in terms of risk contributions rather than capital weights [MaillardRoncalliTeiletche2010]. [Qian2006] showed that risk contributions are not a mere mathematical (ex-ante) decomposition of risk, but have financial significance, being good predictors of the contribution to losses (ex-post), especially for losses of large magnitude.

## The Risk Decomposition: Marginal and Total Contributions

Consider a portfolio $x=(x_1,x_2,\dots,x_n)$ of $n$ risky securities. Let $\sigma_i^2$ be the variance of security $i$, $\sigma_{ij}$ the covariance between securities $i$ and $j$, and $\Sigma$ the covariance matrix. The portfolio's risk, measured by volatility, is
$$\sigma(x)=\sqrt{x^\top\Sigma x}=\sqrt{\textstyle\sum_i x_i^2\sigma_i^2+\sum_i\sum_{j\neq i}x_ix_j\sigma_{ij}}\,.$$
The marginal risk contributions $\partial_{x_i}\sigma(x)$ are defined as the partial derivatives
$$\partial_{x_i}\sigma(x)=\frac{\partial\sigma(x)}{\partial x_i}=\frac{x_i\sigma_i^2+\sum_{j\neq i}x_j\sigma_{ij}}{\sigma(x)}\,. $$
The adjective "marginal" qualifies the fact that these quantities give the change in portfolio volatility induced by a small increase in the weight of a component. In vector form, noting that $\Sigma$ is the covariance matrix of security returns, the $n$ marginal contributions are written as the vector $\dfrac{\Sigma x}{\sqrt{x^\top\Sigma x}}$ [MaillardRoncalliTeiletche2010].

Let $\sigma_i(x)=x_i\times\partial_{x_i}\sigma(x)$ denote the (total) risk contribution of security $i$. One then obtains the decomposition
$$\sigma(x)=\sum_{i=1}^n\sigma_i(x)\,,$$
so that the portfolio's risk can be seen as the sum of the total risk contributions of its components (cf. [[02 Selezione media-varianza]]). The justification for this additivity lies in a property of volatility: $\sigma$ is a homogeneous function of degree $1$, that is, $\sigma(\lambda x)=\lambda\,\sigma(x)$ for $\lambda>0$. It therefore satisfies Euler's theorem and can be reduced to the sum of its arguments multiplied by their respective first partial derivatives, which gives exactly $\sigma(x)=\sum_i x_i\,\partial_{x_i}\sigma(x)=\sum_i\sigma_i(x)$ [MaillardRoncalliTeiletche2010]. In vector form the verification is immediate:
$$x^\top\frac{\Sigma x}{\sqrt{x^\top\Sigma x}}=\frac{x^\top\Sigma x}{\sqrt{x^\top\Sigma x}}=\sqrt{x^\top\Sigma x}=\sigma(x)\,.$$

We have restricted ourselves here to volatility as the risk measure. The principle of risk contributions can also be applied to other measures: in theory it is only necessary for the risk measure to be linearly homogeneous in the weights, so that the portfolio's total risk is entirely decomposed into its components. Under suitable assumptions this holds, for example, for Value-at-Risk [Hallerbach2003] (cf. [[04 Misure di rischio coerenti]]).

## The Equal Risk Contribution (ERC) Portfolio

Starting from the definition of the risk contribution $\sigma_i(x)$, the idea of the ERC (\emph{Equally-weighted Risk Contributions}) strategy is to find a risk-balanced portfolio such that the risk contribution is the same for all securities. We deliberately restrict ourselves to the case with no short sales, $0\le x\le 1$: most investors cannot take short positions, and since the goal is to compare the ERC with other heuristics it is appropriate to keep similar constraints for fairness of comparison. The problem is thus written as
$$x^\star=\Big\{x\in[0,1]^n:\ \textstyle\sum_i x_i=1,\ \ x_i\times\partial_{x_i}\sigma(x)=x_j\times\partial_{x_j}\sigma(x)\ \text{for all } i,j\Big\}\,.$$
Since $\partial_{x_i}\sigma(x)\propto(\Sigma x)_i$, the problem becomes equivalently
$$x^\star=\Big\{x\in[0,1]^n:\ \textstyle\sum_i x_i=1,\ \ x_i\times(\Sigma x)_i=x_j\times(\Sigma x)_j\ \text{for all } i,j\Big\}\,,$$
where $(\Sigma x)_i$ denotes the $i$-th component of the vector $\Sigma x$ [MaillardRoncalliTeiletche2010]. The budget constraint $\sum_i x_i=1$ acts only as normalization: if a portfolio $y$ satisfies $y_i\,\partial_{y_i}\sigma(y)=y_j\,\partial_{y_j}\sigma(y)$ with $y_i\ge 0$ but $\sum_i y_i\neq 1$, then the portfolio $x$ defined by $x_i=y_i/\sum_{i=1}^n y_i$ is the ERC portfolio.

\textbf{The two-asset case.} Let $\rho$ be the correlation and $x=(w,1-w)$ the vector of weights. The vector of total risk contributions is
$$\frac{1}{\sigma(x)}\begin{pmatrix} w^2\sigma_1^2+w(1-w)\rho\sigma_1\sigma_2 \\ (1-w)^2\sigma_2^2+w(1-w)\rho\sigma_1\sigma_2\end{pmatrix}\,.$$
Finding the ERC portfolio means seeking $w$ such that the two rows are equal, that is, $w^2\sigma_1^2=(1-w)^2\sigma_2^2$. The unique solution satisfying $0\le w\le 1$ is
$$x^\star=\left(\frac{\sigma_1^{-1}}{\sigma_1^{-1}+\sigma_2^{-1}},\ \frac{\sigma_2^{-1}}{\sigma_1^{-1}+\sigma_2^{-1}}\right)\,.$$
Note that this solution does not depend on the correlation $\rho$ [MaillardRoncalliTeiletche2010].

\textbf{The general case.} For $n>2$ the number of parameters grows rapidly, with $n$ individual volatilities and $n(n-1)/2$ pairwise correlations. Some special cases admit an analytical solution. If the correlations are constant, $\rho_{ij}=\rho$ for all $i,j$, the total contribution of component $i$ becomes $\sigma_i(x)=x_i\sigma_i\big((1-\rho)x_i\sigma_i+\rho\sum_j x_j\sigma_j\big)/\sigma(x)$, and the ERC condition $\sigma_i(x)=\sigma_j(x)$ turns out to be equivalent to $x_i\sigma_i=x_j\sigma_j$. Together with the normalization constraint, one deduces
$$x_i=\frac{\sigma_i^{-1}}{\sum_{j=1}^n\sigma_j^{-1}}\,.$$
The weight allocated to each component is given by the ratio of the inverse of its volatility to the harmonic mean of the volatilities: the higher (lower) a component's volatility, the lower (higher) its weight in the ERC portfolio [MaillardRoncalliTeiletche2010].

If instead all volatilities are equal, $\sigma_i=\sigma$, but correlations differ, the same reasoning gives
$$x_i=\frac{\big(\sum_{k=1}^n x_k\rho_{ik}\big)^{-1}}{\sum_{j=1}^n\big(\sum_{k=1}^n x_k\rho_{jk}\big)^{-1}}\,,$$
where the weight assigned to component $i$ equals the ratio between the inverse of the weighted average of $i$'s correlations with the other components and the same average taken over all components. Unlike the bivariate case and the constant-correlation case, this solution is endogenous, since $x_i$ is a function of itself, both directly and through the constraint $\sum_i x_i=1$.

The same endogeneity naturally arises in the general case, with both volatilities and correlations different. Starting from the covariance of component $i$'s returns with those of the aggregate portfolio, $\sigma_{ix}=\mathrm{cov}\big(r_i,\sum_j x_j r_j\big)=\sum_j x_j\sigma_{ij}$, one has $\sigma_i(x)=x_i\,\sigma_{ix}/\sigma(x)$. Introducing the beta of component $i$ relative to the portfolio, $\beta_i=\sigma_{ix}/\sigma^2(x)$, one writes $\sigma_i(x)=x_i\,\beta_i\,\sigma(x)$. Since the ERC is defined by $\sigma_i(x)=\sigma_j(x)=\sigma(x)/n$ for all $i,j$, it follows that
$$x_i=\frac{\beta_i^{-1}}{\sum_{j=1}^n\beta_j^{-1}}=\frac{\beta_i^{-1}}{n}\,.$$
The weight assigned to component $i$ is inversely proportional to its beta: the higher (lower) the beta, the lower (higher) the weight, meaning that components with high volatility or high correlation with the other securities are penalized. This solution too is endogenous, since $x_i$ depends on the beta $\beta_i$, which by definition depends on the portfolio $x$ [MaillardRoncalliTeiletche2010].

## Existence, Uniqueness, and Numerical Solution

The preceding expressions allow the ERC solution to be interpreted in terms of a security's risk relative to the rest of the portfolio, but, because of the endogeneity, they do not in general offer a closed-form solution: finding the portfolio requires a numerical algorithm [MaillardRoncalliTeiletche2010].

A first approach consists in solving, with a sequential quadratic programming (SQP) algorithm, the problem
$$x^\star=\arg\min f(x)\quad\text{with}\quad \mathbf{1}^\top x=1\ \text{and}\ 0\le x\le 1\,,$$
where
$$f(x)=\sum_{i=1}^n\sum_{j=1}^n\big(x_i(\Sigma x)_i-x_j(\Sigma x)_j\big)^2\,.$$
The existence of the ERC portfolio is guaranteed only when the condition $f(x^\star)=0$ holds, that is, $x_i(\Sigma x)_i=x_j(\Sigma x)_j$ for all $i,j$: in essence the program minimizes the variance of the (rescaled) risk contributions.

An alternative consists in considering the optimization problem
$$y^\star=\arg\min\sqrt{y^\top\Sigma y}\quad\text{with}\quad \begin{cases}\sum_{i=1}^n\ln y_i\ge c\\ y\ge 0\end{cases}$$
with $c$ an arbitrary constant. In this case the program resembles a variance minimization problem subject to a constraint of sufficient diversification of the weights, implied by the first restriction; the ERC portfolio is expressed as $x_i^\star=y_i^\star/\sum_{i=1}^n y_i^\star$. This formulation has the advantage of showing that the ERC solution is unique as long as the covariance matrix $\Sigma$ is positive definite: it indeed defines the minimization of a quadratic function (a convex function) with a lower-bound constraint that is itself a convex function (cf. [[14 Ottimizzazione convessa - coni, dualità e KKT]]). Relaxing the long-only constraint, one can instead obtain various solutions satisfying the ERC condition [MaillardRoncalliTeiletche2010].

## Positioning Between Minimum Variance and Equally Weighted

The $1/n$ and minimum-variance (MV) portfolios are widely used in practice; the ERC naturally sits between the two and thus appears as a good potential substitute for both. In the two-asset case, the $1/n$ portfolio has $w^*_{1/n}=1/2$: it coincides with the ERC only when the two securities' volatilities are equal, $\sigma_1=\sigma_2$. For the minimum-variance portfolio, the unconstrained solution is
$$w^*_{\mathrm{mv}}=\frac{\sigma_2^2-\rho\sigma_1\sigma_2}{\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2}\,,$$
and it is easily verified that the MV portfolio coincides with the ERC only in the equally weighted case with $\sigma_1=\sigma_2$ [MaillardRoncalliTeiletche2010].

In the general case, the three strategies are distinguished by their respective mathematical definitions (using the fact that MV portfolios equalize marginal risk contributions):
$$\begin{aligned} x_i&=x_j &&(1/n)\\ \partial_{x_i}\sigma(x)&=\partial_{x_j}\sigma(x) &&(\text{mv})\\ x_i\,\partial_{x_i}\sigma(x)&=x_j\,\partial_{x_j}\sigma(x) &&(\text{erc})\end{aligned}$$
The $1/n$ portfolio equalizes the weights; the MV portfolio equalizes the marginal risk contributions (for a minimum-variance portfolio a small increase in any security leads, ex-ante, to the same increase in total risk, but the total contributions are, except in special cases, far from equal, so that the investor concentrates risk in a limited number of positions); the ERC equalizes the total risk contributions [MaillardRoncalliTeiletche2010]. The ERC can thus be seen as a portfolio located between the $1/n$ and the MV.

This positioning becomes explicit when considering the modified version of the problem with a logarithmic constraint:
$$x^\star(c)=\arg\min\sqrt{x^\top\Sigma x}\quad\text{with}\quad\begin{cases}\sum_{i=1}^n\ln x_i\ge c\\ \mathbf{1}^\top x=1\\ x\ge 0\end{cases}$$
The portfolio's volatility is minimized subject to the additional constraint $\sum_{i=1}^n\ln x_i\ge c$, where $c$ is a constant determined by the ERC portfolio and interpretable as the minimum level of diversification among the components necessary to obtain it. Two polar cases are defined by $c=-\infty$, which returns the MV portfolio, and $c=-n\ln n$, which returns the $1/n$ portfolio: indeed the quantity $\sum_i\ln x_i$, subject to $\sum_i x_i=1$, is maximized for $x_i=1/n$. In statistical terms, the quantity $-\sum_i x_i\ln x_i$ is known as entropy. This reinforces the interpretation of the ERC as an intermediate portfolio between the MV and the $1/n$, that is, a form of minimum-variance portfolio subject to a constraint of sufficient diversification of the weights [MaillardRoncalliTeiletche2010].

From this program follows a natural ordering of the volatilities of the three portfolios:
$$\sigma_{\mathrm{mv}}\le\sigma_{\mathrm{erc}}\le\sigma_{1/n}\,,$$
with the MV, as expected, the least volatile, the $1/n$ the most volatile, and the ERC positioned between the two [MaillardRoncalliTeiletche2010].

A numerical example makes the difference concrete. With four assets of volatility $10\%$, $20\%$, $30\%$, and $40\%$ respectively and a constant correlation matrix, the $1/n$ assigns $25\%$ to each security, while the ERC solution is $48\%$, $24\%$, $16\%$, $12\%$ (consistent with proportionality to the inverse of volatility); the MV solution instead depends on the correlation, and for zero correlation becomes $x^{\mathrm{mv}}=(70{,}2\%,\,17{,}6\%,\,7{,}8\%,\,4{,}4\%)$, a portfolio far more concentrated than the ERC. With a correlation matrix in which the third security has zero correlation with the first two and correlation $-0{,}5$ with the fourth, the MV portfolio concentrates weight and risk in the first security (weight and risk contribution equal to $74{,}5\%$) and excludes the second, while the ERC invests in all securities with a risk contribution of $25\%$ each; the ERC has volatility higher than that of the MV but lower than that of the $1/n$, and is markedly more balanced in terms of risk contributions [MaillardRoncalliTeiletche2010].

Out-of-sample empirical tests confirm the picture: equally weighted portfolios appear inferior in terms of performance and by any risk measure; minimum-variance portfolios can achieve higher Sharpe ratios thanks to lower volatility, but are exposed to larger drawdowns in the short run, are consistently much more concentrated, and appear substantially less efficient in terms of portfolio turnover [MaillardRoncalliTeiletche2010].

## When the ERC Is Optimal

It is useful to identify when the ERC portfolio coincides with the maximum Sharpe ratio (MSR) portfolio, also known as the tangency portfolio, whose composition is $\dfrac{\Sigma^{-1}(\mu-r)}{\mathbf{1}^\top\Sigma^{-1}(\mu-r)}$, where $\mu$ is the vector of expected returns and $r$ the risk-free rate [Martellini2008]. The MSR portfolio is defined as the one in which the ratio of marginal excess return to marginal risk is the same for all securities and equal to the portfolio's Sharpe ratio [Scherer2007]:
$$\frac{\mu(x)-r}{\sigma(x)}=\frac{\partial_x\mu(x)-r}{\partial_x\sigma(x)}\,.$$
Since $\mu(x)=x^\top\mu$ and $\sigma(x)=\sqrt{x^\top\Sigma x}$, one has $\partial_x\mu(x)=\mu$ and $\partial_x\sigma(x)=\Sigma x/\sigma(x)$; it follows that $x$ is MSR if it satisfies
$$\mu-r=\left(\frac{\mu(x)-r}{\sigma(x)}\right)\frac{\Sigma x}{\sigma(x)}\,.$$
It can be shown that the ERC portfolio is optimal if a constant correlation matrix is assumed and all securities are assumed to have the same Sharpe ratio. Indeed, under constant correlation the total risk contribution of component $i$ equals $(\Sigma x)_i/\sigma(x)$, and by definition this contribution will be equal for all securities; to verify the previous condition it then suffices that each security exhibit the same individual Sharpe ratio $s_i=(\mu_i-r)/\sigma_i$. Conversely, when correlations differ or securities have different Sharpe ratios, the ERC portfolio differs from the MSR [MaillardRoncalliTeiletche2010].

## Risk Budgeting: Assigning Risk Budgets

The ERC is the simplest manifestation of a more general principle. Risk budgeting is the management of the portfolio in terms of risk contributions rather than capital weights: the allocator assigns each component a risk budget, and seeks the portfolio that realizes it. The ERC constitutes a particular form of risk budgeting in which the allocator distributes the same risk budget to every component, so that no component contributes more to total risk than the others (at least on an ex-ante basis) [MaillardRoncalliTeiletche2010]. In this sense the ERC applies a kind of "$1/n$ filter" in terms of risk, maximizing the dispersion of risks rather than that of weights.

The fact that the budget constraint $\sum_i x_i=1$ acts only as normalization is what makes the risk budgeting framework operational: the allocation condition is formulated on the products $x_i\,\partial_{x_i}\sigma(x)$, that is, on the total risk contributions, and the constraint on the weights intervenes only downstream to rescale the portfolio [MaillardRoncalliTeiletche2010]. The financial relevance of this framework is that risk contributions are not a pure algebraic decomposition: [Qian2006] shows that they are good predictors of the contribution to losses, particularly for losses of large magnitude, so that allocating risk budgets is, to a first approximation, equivalent to allocating expected loss budgets.

The ERC strategy shares a diversification-based philosophy with the \emph{Most Diversified Portfolio} (MDP) of [ChoueifatyCoignard2008] (cf. [[19 Massima diversificazione]]); the two portfolios are, however, generally distinct, and coincide only when the correlation coefficient among the components is unique [MaillardRoncalliTeiletche2010].

## Qian's Perspective: Risk Parity Across Asset Classes

The formulation of risk budgeting as the allocation of risk across asset classes is at the center of the \emph{Risk Parity Portfolios} of [Qian2005]: a family of beta-efficient portfolios that allocate market risk equally among asset classes — equities, bonds, and commodities. The approach differs from traditional asset allocation: it aims to provide true diversification, limiting the impact of losses from individual components on the overall portfolio, and is expected to generate a higher return for a given target risk level.

\textbf{Eggs in one basket.} A well-known investment axiom recommends not putting all one's eggs in one basket. Yet a balanced $60/40$ portfolio — $60\%$ equities and $40\%$ bonds — places over $90\%$ of the eggs in one basket. The reason is that size matters: "equity eggs" are about nine times larger than "bond eggs." Assuming an annual standard deviation of $15\%$ for equities and $5\%$ for bonds, in variance terms equities are nine times riskier than bonds. With the egg analogy: six equity eggs of size $9$ and four bond eggs of size $1$ give an equivalent of $6\times 9+4=58$ eggs, of which $54$ are equity eggs, i.e., about $93\%$ [Qian2005].

The analogy is not far from reality. Between $1983$ and $2004$, the excess return of the Russell 1000 index had an annualized volatility of $15{,}1\%$ and the Lehman Aggregate Bond Index of $4{,}6\%$, with a correlation between the two of $0{,}2$. On this data, in a $60/40$ portfolio equities contributed $93\%$ of the risk and bonds the remaining $7\%$. The message is clear: although a $60/40$ may appear balanced in terms of capital allocation, it is heavily concentrated from the standpoint of risk allocation [Qian2005]. A further sign of equity dominance is the correlation between the return of the $60/40$ and that of the Russell 1000, above $0{,}98$ over the period considered.

\textbf{From risk contribution to loss contribution.} Why should investors care about risk contribution? Research shows that it is a very accurate indicator of the contribution to losses. For the $60/40$, for losses exceeding $2\%$, equities contributed on average $95{,}6\%$ and bonds $4{,}4\%$ (over $44$ monthly observations); for losses exceeding $3\%$, equities contributed $100{,}1\%$ and bonds $-0{,}1\%$; for losses exceeding $4\%$, equities contributed $101{,}9\%$ and bonds $-1{,}9\%$. For losses beyond $2\%$, therefore, equities contributed on average $96\%$ of the losses, a value very close to the $93\%$ risk contribution calculated earlier. This provides empirical evidence for the economic interpretation of risk contribution: it approximates the expected contribution to losses of the underlying components, and this holds when variances and covariances are used to compute it [Qian2005].

\textbf{Risk parity portfolios.} From these observations follows why the $60/40$ is not well diversified: when a loss of appreciable size occurs, over $90\%$ is attributable to equities, so that the diversification effect of bonds is insignificant. One can construct a portfolio that limits the impact of large losses from individual components by making the expected contribution to losses the same for all: in the equity/bond example, an allocation of $23\%$ to the Russell 1000 and $77\%$ to the Lehman Aggregate produces equal risk contribution from equities and bonds [Qian2005]. For the parity portfolio so constructed, the contribution to losses exceeding $2\%$ is $48{,}4\%$ for equities and $51{,}6\%$ for bonds, close to parity [Qian2005].

In terms of results, measured as excess return over three-month Treasury bills over the period $1983$–$2004$: the Russell 1000 had average return $8{,}3\%$, standard deviation $15{,}1\%$, and Sharpe ratio $0{,}55$; the Lehman Aggregate $3{,}7\%$, $4{,}6\%$, and $0{,}80$; the $60/40$ $6{,}4\%$, $9{,}6\%$, and $0{,}67$; the parity portfolio $4{,}7\%$, $5{,}4\%$, and $0{,}87$. The $60/40$'s Sharpe ratio of $0{,}67$ is lower than that of bonds: an indication of poor diversification, since the overall Sharpe ratio is lower than that of one of its components. By contrast, the parity portfolio's Sharpe ratio, $0{,}87$, is higher than that of both equities and bonds, representing the benefits of true diversification [Qian2005].

\textbf{The "optimality" of risk parity.} Unlike the traditional approach, which involves estimating long-term expected returns and mean-variance optimization, risk parity portfolios are based purely on risk diversification. The question remains: why should parity of risk contributions lead to efficient portfolios? The reason is that risk parity portfolios are mean-variance optimal if the underlying components have equal Sharpe ratios and their returns are uncorrelated. Equal Sharpe ratios imply that expected return is proportional to the risk of each asset class, which is theoretically attractive because it means that assets are priced according to their risk; in practice one can derive an asset class's implied return from its Sharpe ratio and assess its plausibility. Moreover, the actual correlation between equities and bonds, though not zero, is rather low. These elements support regarding risk parity portfolios as efficient, not only in terms of risk allocation, but also in the classical mean-variance sense [Qian2005].

Risk parity portfolios present further benefits. First, every asset is guaranteed a nonzero weight. Second, the weights are influenced by the correlations among returns in a desirable way: assets that have exhibited higher correlations with the other classes receive lower weight, those with lower correlations receive higher weight; commodities, for example, would receive significant weight thanks to their low correlations with equities and bonds [Qian2005].

\textbf{Risk level and leverage.} The unlevered version of the parity portfolio has a lower return than the $60/40$ because of its much lower risk; an investor might not achieve their return target simply by creating a high-Sharpe-ratio portfolio. One solution is the use of leverage to reach higher return levels. Since bonds have much lower risk than equities, risk parity portfolios apply leverage to bonds so that they have the same risk contribution as equities; the levered version retains the high Sharpe ratio but with higher returns. The levered version of the parity portfolio with $1{,}8:1$ leverage has risk equal to that of the $60/40$ and average return $8{,}4\%$ against $6{,}4\%$, outperforming the $60/40$ by about $2\%$ annually; a levered parity portfolio with risk equal to that of equities outperformed the Russell 1000 by nearly $5\%$ annually with $2{,}8:1$ leverage. In backtests risk parity portfolios had a Sharpe ratio of $1{,}1$ over the period $1983$–$2004$ [Qian2005].

Risk parity portfolios can be used as standalone beta products or combined with alpha strategies to further enhance returns; [Qian2005] indicates three uses: an unlevered version with risk of $4$–$5\%$, similar to the Lehman Aggregate; a levered version with a ratio of about $2:1$ and target risk of $8$–$10\%$, similar to domestic or global balanced portfolios; and a global macro strategy with risk of $16$–$20\%$ and $4:1$ leverage, similar to that of a typical hedge fund.

## References

- **[ChoueifatyCoignard2008]** Choueifaty, Y. & Coignard, Y. (2008). Towards maximum diversification. Journal of Portfolio Management, 34(4), pp. 40–51.
- **[DeMiguelEtAl2009]** DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? Review of Financial Studies, 22, pp. 1915–1953.
- **[Hallerbach2003]** Hallerbach, W. (2003). Decomposing portfolio Value-at-Risk: A general analysis. Journal of Risk, 5(2), pp. 1–18.
- **[MaillardRoncalliTeiletche2010]** Maillard, S., Roncalli, T. & Teiletche, J. (2010). On the properties of equally-weighted risk contributions portfolios. Journal of Portfolio Management, 36(4), pp. 60–70.
- **[Markowitz1952]** Markowitz, H.M. (1952). Portfolio selection. Journal of Finance, 7, pp. 77–91.
- **[Martellini2008]** Martellini, L. (2008). Toward the design of better equity benchmarks. Journal of Portfolio Management, 34(4), pp. 1–8.
- **[Merton1980]** Merton, R.C. (1980). On estimating the expected return on the market: An exploratory investigation. Journal of Financial Economics, 8, pp. 323–361.
- **[Michaud1989]** Michaud, R. (1989). The Markowitz optimization enigma: Is optimized optimal? Financial Analysts Journal, 45, pp. 31–42.
- **[Qian2005]** Qian, E. (2005). Risk parity portfolios: Efficient portfolios through true diversification. PanAgora Asset Management, September.
- **[Qian2006]** Qian, E. (2006). On the financial interpretation of risk contributions: Risk budgets do add up. Journal of Investment Management, Fourth Quarter.
- **[Scherer2007]** Scherer, B. (2007). Portfolio Construction & Risk Budgeting. Risk Books, Third Edition.
