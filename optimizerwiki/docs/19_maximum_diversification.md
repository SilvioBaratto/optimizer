---
title: "Maximum Diversification"
chapter: 19
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-11
---

> [!abstract] Summary
> The chapter introduces diversification as an autonomous allocation criterion, alternative to utility maximization: the diversification ratio is defined as the ratio between the weighted average of securities' volatilities and the portfolio's volatility, and the Most-Diversified Portfolio as the one that maximizes it. Its characterization as the minimum-variance portfolio on the correlation matrix is established, along with the core properties — every held security has the same correlation with the portfolio, correlation equal to the ratio of diversification ratios — and the three invariances under duplication, leverage, and linear combination. Finally, the relationship with the global minimum variance and the tangency portfolio is shown, identifying in the proportionality between expected returns and volatility the condition under which the most diversified portfolio is mean-variance optimal.

## Diversification as an Allocation Criterion

Mean-variance selection, described in [[02 Selezione media-varianza]], requires two ingredients: the covariance matrix of returns and the vector of expected returns. The former can be estimated with a reasonable degree of reliability, while expected returns remain much harder to estimate, to the point that the most widespread models — the CAPM of [Sharpe1964] or the Black-Litterman model discussed in [[03 Limiti della MPT e il modello di Black-Litterman]] — end up, one way or another, partially or entirely setting them aside [ChoueifatyCoignard2008]. In parallel, the conviction has spread that market-capitalization indices are not efficient, and alternative solutions have been proposed such as fundamental indexation [Arnott2005] or equal weighting [ChoueifatyCoignard2008]. In a seminal contribution, [HaugenBaker1991] had already shown, for the period 1972-1989, that a minimum-variance portfolio on U.S. equities achieved returns equal to or higher than those of a broad capitalization-weighted index, with systematically lower volatility, thereby highlighting the *ex post* inefficiency of the cap-weighted index [ChoueifatyFroidureReynier2013].

Against this background sits the *maximum diversification* criterion. Echoing Markowitz, diversification is "the only free lunch" in finance [ChoueifatyCoignard2008]; the idea is to treat it not as a side effect of utility optimization, but as the explicit objective of portfolio construction. The concept of maximum diversification, introduced in [Choueifaty2006] through a formal measure of the degree of diversification — the *diversification ratio* — and the portfolio that maximizes it were made known to a wide audience by [ChoueifatyCoignard2008] and subsequently explored further by [ChoueifatyFroidureReynier2013]. The result is an investment style that favors diversification and avoids bets based on return forecasting or on trust in the implicit bets of capitalization-weighted benchmarks [ChoueifatyCoignard2008].

In what follows, consider a universe of $N$ risky securities $\{S_1,\dots,S_N\}$ with volatility vector $\sigma=(\sigma_i)$, correlation matrix $C=(\rho_{i,j})$, and covariance matrix $V=\Sigma=(\rho_{i,j}\sigma_i\sigma_j)$. A portfolio is identified by the weight vector $w=(w_i)$ with $\sum_i w_i=1$; unless stated otherwise, all portfolios are constrained to be long-only, that is, with non-negative weights [ChoueifatyFroidureReynier2013].

## The Diversification Ratio

Let $\langle w\mid\sigma\rangle=\sum_i w_i\sigma_i$ be the weighted average of securities' volatilities and $\sigma(w)=\sqrt{w'Vw}$ the portfolio's volatility. The *diversification ratio* of a portfolio $P=(w_1,\dots,w_N)$ is defined as

$$D(w)=\frac{\langle w\mid\sigma\rangle}{\sigma(w)}=\frac{w'\sigma}{\sqrt{w'Vw}},$$

that is, as the ratio between the weighted average of securities' volatilities and the portfolio's volatility [ChoueifatyCoignard2008][ChoueifatyFroidureReynier2013]. This measure embodies the very nature of diversification: for a long-only portfolio, the overall volatility is less than or equal to the weighted sum of the individual securities' volatilities, so the diversification ratio is always greater than or equal to $1$, and equals exactly $1$ for a mono-asset portfolio, in which there is no diversification at all [ChoueifatyFroidureReynier2013]. For a long-only portfolio the ratio is strictly greater than $1$ unless the portfolio is equivalent to a mono-asset portfolio [ChoueifatyCoignard2008].

Two elementary cases fix the intuition. An equally weighted portfolio of two independent assets with the same volatility has diversification ratio $\sqrt{2}$; more generally, for $N$ independent assets with the same volatility the ratio equals $\sqrt{N}$, since the average volatility of the assets equals their common volatility, while the volatility of the equally weighted portfolio is that volatility divided by the square root of the number of assets [ChoueifatyFroidureReynier2013]. In essence, the diversification ratio measures the diversification gained by holding assets that are not perfectly correlated.

A two-security example clarifies the operational meaning of the criterion. With two securities $A$ and $B$ of volatility $15\%$ and $30\%$ and correlation strictly less than $1$, diversifying means wanting both to contribute equally to the portfolio's volatility: the weights that maximize diversification are then inversely proportional to the volatilities, that is, $66{,}6\%$ on $A$ and $33{,}3\%$ on $B$ [ChoueifatyCoignard2008]. With three securities — two strongly correlated bank stocks ($\rho=0{,}9$) and a third, pharmaceutical, weakly correlated ($\rho=0{,}1$) with each of the two, all of equal volatility — the weights that maximize diversification are $25{,}7\%$ for each bank stock and $48{,}6\%$ for the pharmaceutical stock: diversification rewards the security that brings more independent risk [ChoueifatyCoignard2008].

## Decomposition and Interpretation of the Diversification Ratio

The diversification ratio admits a decomposition that isolates its two levers. Setting $\bar w=w\odot\sigma$ (element-wise product of weights and volatilities), the portfolio variance is written as

$$\sigma^2(w)=\sum_i \bar w_i^2+\rho(w)\sum_{i\neq j}\bar w_i\bar w_j =(1-\rho(w))\sum_i \bar w_i^2+\rho(w)\Big(\sum_i \bar w_i\Big)^2,$$

where the identity $\sum_{i\neq j}\bar w_i\bar w_j=(\sum_i \bar w_i)^2-\sum_i \bar w_i^2$ has been used. Dividing by $(\sum_i \bar w_i)^2$ gives the decomposition [ChoueifatyFroidureReynier2013]

$$D(w)=\big[\rho(w)\,(1-\mathrm{CR}(w))+\mathrm{CR}(w)\big]^{-1/2},\qquad \frac{1}{D(w)^2}=(1-\rho(w))\,\mathrm{CR}(w)+\rho(w),$$

where $\rho(w)$ is the average security correlation, weighted by volatility,

$$\rho(w)=\frac{\sum_{i\neq j}(w_i\sigma_i\,w_j\sigma_j)\,\rho_{i,j}}{\sum_{i\neq j}(w_i\sigma_i\,w_j\sigma_j)},$$

and $\mathrm{CR}(w)$ is the *volatility-weighted concentration ratio*

$$\mathrm{CR}(w)=\frac{\sum_i (w_i\sigma_i)^2}{\big(\sum_i w_i\sigma_i\big)^2}.$$

A fully concentrated long-only portfolio has unit $\mathrm{CR}$ (mono-asset portfolio), while a volatility-equally-weighted portfolio has the minimum $\mathrm{CR}$, equal to the inverse of the number of assets it contains [ChoueifatyFroidureReynier2013]. The concentration ratio generalizes the Herfindahl-Hirschman index by measuring not only the concentration of weights but also the concentration of risks, since assets are weighted proportionally to their volatilities [ChoueifatyFroidureReynier2013].

The decomposition explicitly shows that the diversification ratio increases when the average correlation and/or the concentration ratio decrease. At the extreme, if correlations tend to $1$, the ratio equals $1$ regardless of the value of $\mathrm{CR}$: a portfolio of perfectly correlated assets is no more diversified than a single asset. When pairwise correlations are all equal, the diversification ratio varies only through the $\mathrm{CR}$, and maximizing it is equivalent to minimizing the $\mathrm{CR}$ [ChoueifatyFroidureReynier2013].

The square of the diversification ratio admits an interpretation in terms of degrees of freedom. Consider a universe of $F$ independent risk factors and a portfolio whose exposure to each factor is inversely proportional to the factor's volatility: such a portfolio allocates the risk budget uniformly across all factors and has $D^2=F$. Hence $D^2$ can be read as the number of independent risk factors — or degrees of freedom — represented in the portfolio, and $F$ as the effective number of independent factors [ChoueifatyFroidureReynier2013]. As an example, the diversification ratio of an index such as the MSCI World was $1{,}7$ at the end of 2010, which implies that a passive investor in that index was effectively exposed to $1{,}7^2\approx 3$ independent risk factors; maximizing the ratio would have yielded $D=2{,}6$, that is, $2{,}6^2\approx 7$ effective degrees of freedom [ChoueifatyFroidureReynier2013].

## The Most Diversified Portfolio: Definition, Existence, First-Order Condition

Let $\Gamma$ be a set of linear constraints on the weights; a usual constraint is the long-only one, which imposes non-negative weights. The portfolio that, under the constraints $\Gamma$, maximizes the diversification ratio in the universe $U$ is the *Most-Diversified Portfolio* (MDP), denoted $M(\Gamma,U)$ [ChoueifatyCoignard2008]. In the long-only case

$$w^{\mathrm{MDP}}=\arg\max_{w\in\Pi^+} D(w),$$

where $\Pi^+$ is the set of long-only portfolios with weights summing to $1$ [ChoueifatyFroidureReynier2013].

**Existence and uniqueness.** Since the diversification ratio is invariant under scalar multiplication of the weights, the maximization problem is equivalent to the quadratic problem

$$\min_w \tfrac12\,w'Vw \quad\text{s.t.}\quad w_i\ge 0,\ \sum_i w_i\sigma_i=1,$$

with the weights rescaled to sum to $1$ afterward. This is a quadratic programming problem over a convex set: existence follows, and uniqueness also follows if the covariance matrix is definite [Berkovitz2001][ChoueifatyFroidureReynier2013]. For the objective to be finite, there must not exist in $\Pi^+$ a zero-volatility portfolio; in the equity context this is equivalent to assuming that there is no long-only, zero-risk portfolio that carries a positive premium, a condition that holds when $V$ is definite [ChoueifatyFroidureReynier2013].

**First-order condition.** The logarithm of the (positive) objective is $f(w)=\ln D(w)=\ln\langle\sigma\mid w\rangle-\tfrac12\ln\langle Vw\mid w\rangle$, with gradient

$$\nabla f_w=\frac{1}{\langle\sigma\mid w\rangle}\sigma-\frac{1}{\langle Vw\mid w\rangle}Vw.$$

Applying the Karush-Kuhn-Tucker theorem (see [[14 Ottimizzazione convessa - coni, dualità e KKT]]), at an optimal point $w^*$ there exist a vector $\nu\in\mathbb{R}^N$ and a scalar $\mu$ such that

$$\frac{1}{\langle\sigma\mid w^*\rangle}\sigma-\frac{1}{\sigma(w^*)^2}Vw^*+\mu\mathbf{1}+\nu=0,\qquad \min(\nu,w^*)=0,\qquad \langle w^*\mid\mathbf{1}\rangle=1.$$

Left-multiplying the first condition by $w^{*\prime}$ gives $\mu=0$, and one sees that the first condition is independent of the constraint that the weights sum to $1$ — consistent with the scalar-multiplication invariance of the diversification ratio. Setting $\lambda=\sigma^2(w^*)\nu$, the optimal point satisfies [ChoueifatyFroidureReynier2013]

$$V\,w^{\mathrm{MDP}}=\frac{\sigma(w^{\mathrm{MDP}})}{D(w^{\mathrm{MDP}})}\,\sigma+\lambda,\qquad \min(\lambda,w^{\mathrm{MDP}})=0,$$

where the dual variables $\lambda$ are non-negative and satisfy the complementarity $\min(\lambda,w^{\mathrm{MDP}})=0$.

## Geometric Characterization: Minimum Variance on the Correlation Matrix

The most diversified portfolio admits a transparent characterization: it is the minimum-variance portfolio computed on the *correlation matrix* rather than on the covariance matrix. This is obtained by transferring the problem to a synthetic universe in which all securities have the same expected volatility [ChoueifatyCoignard2008].

Assuming that investors can lend and borrow cash at the same rate, one defines the synthetic assets

$$Y_i=\frac{X_i}{\sigma_i}+\Big(1-\frac{1}{\sigma_i}\Big)\$,$$

where $\$$ is the risk-free asset. In the new universe $U_S=\{Y_1,\dots,Y_N\}$ the volatility of each $Y_i$ equals $1$, so the vector of synthetic volatilities is $\Sigma_S=\mathbf{1}$ and the synthetic covariance matrix $V_S$ coincides with the correlation matrix $C$ of the original assets, since correlation does not change with leverage. The synthetic diversification ratio is $D(S)=S'\Sigma_S/\sqrt{S'V_S S}$; imposing $S'\Sigma_S=1$, maximizing $D(S)$ is equivalent to maximizing $1/\sqrt{S'CS}$, that is, to **minimizing**

$$S'CS.$$

In a universe where all securities have the same volatility, one therefore minimizes the variance, which is exactly the benefit expected from diversification [ChoueifatyCoignard2008].

If $C$ is invertible (and $\Gamma=\varnothing$), the synthetic solution is unique and equals

$$S\propto C^{-1}\mathbf{1},$$

that is, the vector of synthetic weights is proportional to the inverse of the correlation matrix multiplied by a vector of ones. Reconstructing the original assets — each synthetic weight must be divided by the respective asset's volatility, and the portfolio rescaled to be $100\%$ invested — the weights of the real portfolio $M$ are

$$M=\Big(\frac{w_{S1}}{\sigma_1},\dots,\frac{w_{SN}}{\sigma_N},\ \Big(1-\sum_{i=1}^{N}\frac{w_{Si}}{\sigma_i}\Big)\$\Big),\qquad M\propto \sigma^{-1}C^{-1}\mathbf{1},$$

where $\sigma$ here is the diagonal matrix of volatilities [ChoueifatyCoignard2008]. The most diversified portfolio is therefore the minimum-variance portfolio on the correlation matrix, rescaled by the inverse volatilities.

## The Core Properties: Same Correlation with the Portfolio

From the form $M=\kappa\,\sigma^{-1}C^{-1}\mathbf{1}$ (with $\kappa$ a constant) follows a set of properties concerning the correlation between any portfolio and the most diversified portfolio. Since $V\,M=\sigma C\sigma\,\kappa\sigma^{-1}C^{-1}\mathbf{1}=\kappa\,\sigma$, the correlation of a portfolio $P$ with $M$ is

$$\rho_{P,M}=\frac{P'\sigma C\sigma\,M}{\sigma_P\sigma_M}=\frac{\sum_i w_i\sigma_i}{\sigma_P}\cdot\frac{\kappa}{\sigma_M}=D(P)\,\frac{\kappa}{\sigma_M}.$$

The correlation of $P$ with the most diversified portfolio is thus proportional to the diversification ratio of $P$ [ChoueifatyCoignard2008].

Applying the formula to a single security $i$, whose diversification ratio is $1$ (no diversification), gives

$$\rho_{i,M}=\frac{\kappa}{\sigma_M},$$

a value identical for every security in the universe. The most diversified portfolio is therefore the one in which all assets have the same positive correlation with it [ChoueifatyCoignard2008]. Setting $P=M$ in the previous formula identifies the constant, $\kappa=\sigma_M/D(M)$, and the correlation between a generic portfolio $P$ and $M$ can be rewritten as the ratio of the respective diversification ratios:

$$\rho_{P,M}=\frac{D(P)}{D(M)},$$

while for a single security $\rho_{i,M}=1/D(M)$ [ChoueifatyCoignard2008]. With this information one builds a single-factor (diversification) model reminiscent in form of the CAPM, but identifying the correlation with the ratio of diversification levels:

$$R_P=\alpha_P+\frac{\sigma_P}{\sigma_M}\frac{D(P)}{D(M)}R_M+\varepsilon_P,$$

where $R$ is an excess return over cash and $\alpha_P,\varepsilon_P$ are the constant and error terms usually associated with a regression [ChoueifatyCoignard2008].

In the constrained long-only case, [ChoueifatyFroidureReynier2013] formulate an equivalent definition, the *core property*, which clarifies the nature of the portfolio:

> (1) Every security not held by the most diversified portfolio is more correlated with it than any security that belongs to it; moreover, all securities belonging to the portfolio have the same correlation with it.

This property shows that all assets in the universe are in fact represented in the portfolio, even if it does not physically hold them. A most diversified portfolio built on an index of $500$ securities may hold about $50$ of them; this does not mean it is not diversified, since the $450$ securities it does not hold are more correlated with the portfolio than the $50$ it actually holds — consistent with the idea that the most diversified portfolio is the *non-diversifiable* portfolio [ChoueifatyFroidureReynier2013]. The core property admits an equivalent formulation, which constitutes its proof basis: the long-only most diversified portfolio is that long-only portfolio such that the correlation between any other long-only portfolio and it is greater than or equal to the ratio of the respective diversification ratios,

$$\rho_{w,w^{\mathrm{MDP}}}\ge \frac{D(w)}{D(w^{\mathrm{MDP}})}.$$

The more diversified a long-only portfolio is, the higher its correlation with the most diversified portfolio [ChoueifatyFroidureReynier2013]. In the long-only case, all assets with non-zero weight have the same correlation with the portfolio; assets with zero weight, excluded from the optimization, have correlations with the portfolio higher than those of the non-zero-weight assets [ChoueifatyCoignard2008].

## Portfolio Invariances

An *unbiased* and agnostic portfolio construction process should respect certain elementary rules, motivated by the fact that portfolios produced by such processes depend heavily on the structure of the universe considered: it is reasonable to require that a universe equivalent to the original one produce exactly the same portfolio [ChoueifatyFroidureReynier2013]. [ChoueifatyFroidureReynier2013] formalize three *invariance properties*:

1. **Duplication invariance.** If an asset is duplicated (for example due to multiple listings of the same security), the process should produce the same portfolio, regardless of the duplication.
2. **Leverage invariance.** If a company changes its financial leverage, all else being equal, the weights allocated to its underlying business should not change, since the cash exposure is treated separately.
3. **Polico invariance.** Adding a positive linear combination ("polico") of assets already present in the universe — for example a long-only leveraged ETF on a subset of the universe — should not alter the portfolio's weights on the original assets, given that these were already available in the starting universe.

The most diversified portfolio satisfies all three invariances. Duplication generally leaves the weights on the original assets unchanged, since the introduction of a redundant asset generates a redundant equation in the first-order conditions [ChoueifatyFroidureReynier2013]. Leverage invariance is proven by rewriting the first-order condition as $\sigma\odot C(\sigma\odot w)=\delta\sigma+\lambda$: applying a positive leverage vector $L$, the leveraged assets have the same correlation matrix $C$ and volatility $\sigma^L=(L_i\sigma_i)$, and the portfolio $w^L=k\,w\oslash L$ satisfies the first-order condition in the leveraged universe, so that $\sigma^L\odot w^L=k\,\sigma\odot w$ [ChoueifatyFroidureReynier2013]. Polico invariance follows from the core property: since every non-selected asset has correlation greater than $1/D(M)$ and the diversification ratio of a polico is greater than $1$, one has $\rho(\Lambda,M)\ge D(\Lambda)/D(M)>1/D(M)$, so the polico is never selected and the portfolio remains unchanged [ChoueifatyFroidureReynier2013].

With a simple two-asset universe $A$ and $B$ ($\sigma_A=20\%$, $\sigma_B=10\%$, $\rho_{AB}=50\%$), the equally weighted portfolio (EW), the minimum-variance portfolio (MV), the equal risk contribution portfolio (ERC — see [[17 Risk budgeting e risk parity]]), and the most diversified portfolio (MDP) are compared. Only the MDP and the ERC provide a genuinely diversified risk allocation: the EW concentrates the risk contribution in the more volatile asset, while the MV invests $100\%$ in the low-risk asset [ChoueifatyFroidureReynier2013]. As for the invariances, the MDP respects all three; the MV is invariant under duplication but not under leverage or polico; the ERC is invariant under leverage but not under duplication or polico; the EW respects none of the invariances. The EW and ERC reflect the belief that representativeness is achieved only by investing in all the securities present; the EW and MV embed implicit bets on companies' leverage [ChoueifatyFroidureReynier2013].

## Relationship with the Global Minimum Variance and the Tangency Portfolio

The most diversified portfolio sits in a precise relationship with two notable portfolios on the mean-variance frontier.

**Global minimum variance.** If all securities in the universe have the same volatility, the most diversified portfolio coincides with the global minimum-variance portfolio [ChoueifatyCoignard2008]. The reason is immediate from the preceding characterization: when the volatilities are equal, minimizing $S'CS$ on the correlation matrix is equivalent to minimizing $w'Vw$ on the covariance matrix, since the two coincide up to a scale factor.

**Tangency portfolio and mean-variance optimality.** The condition under which the most diversified portfolio is mean-variance optimal is that the securities' expected excess returns be proportional to their volatilities — that is, that "risk is compensated" homogeneously across securities [ChoueifatyFroidureReynier2013]. Consider a homogeneous universe in which there is no reason to believe *ex ante* that one security compensates risk more than another: the *ex ante* Sharpe ratios of the individual securities are identical, and the expected excess return (EER) of each security is proportional to its volatility. Denoting by $k>0$ a constant and by $r_f$ the risk-free rate,

$$E(r_i)-r_f=k\,\sigma_i.$$

Then, for any portfolio of weights $w$,

$$E(r_w)-r_f=\sum_{i=1}^{N}w_i\big(E(r_i)-r_f\big)=k\,\langle w\mid\sigma\rangle=k\,\sigma(w)\,D(w),$$

where the last equality uses the definition of the diversification ratio. Dividing both sides by $\sigma(w)$ shows that, in this homogeneous universe, maximizing the diversification ratio is equivalent to maximizing the Sharpe ratio: the most diversified portfolio is then the tangency portfolio, mean-variance optimal [ChoueifatyCoignard2008][ChoueifatyFroidureReynier2013]. Assuming the CAPM assumptions hold in the form of [Sharpe1991] and that all investors agree that the EERs of individual securities are proportional to their volatilities, one recovers the security market line relation, in which the role of the market portfolio is played by the most diversified portfolio:

$$E(r_i)-r_f=\rho_{\mathrm{MDP}}\,\frac{\sigma_i}{\sigma_{\mathrm{MDP}}}\,\big(E(r_{\mathrm{MDP}})-r_f\big),$$

where $\rho_{\mathrm{MDP}}$ is the — constant — correlation of all assets with the unconstrained most diversified portfolio. In this setting assets are compensated in proportion to their exposure to systematic risk, which here corresponds to exposure to the most diversified portfolio [ChoueifatyFroidureReynier2013]. The same assumptions allow one to read the expected returns implied by the optimality of alternative benchmarks: a capitalization-weighted benchmark is optimal when the EERs are proportional to the securities' total risk and to their correlation with the benchmark, while the minimum-variance portfolio is optimal when the EERs are equal across all assets, $E(r_i)=K$ [ChoueifatyCoignard2008].

## Empirical Evidence and Link to the Implemented Criterion

The most diversified portfolio belongs to the family of risk-based allocation criteria, alongside risk budgeting and risk parity treated in [[17 Risk budgeting e risk parity]] and minimum variance discussed in [[02 Selezione media-varianza]]: like these, it requires no estimates of expected returns and is built from the covariance matrix alone. Its operational implementation consists in solving the long-only quadratic problem $\min_w \tfrac12 w'Vw$ subject to $w_i\ge 0$ and $\sum_i w_i\sigma_i=1$, with the weights rescaled to sum to $1$ afterward, or — equivalently — in directly maximizing the diversification ratio $D(w)=\langle w\mid\sigma\rangle/\sigma(w)$ under the desired constraints $\Gamma$ [ChoueifatyCoignard2008][ChoueifatyFroidureReynier2013]. In practice, maximum per-security weight constraints, sector or regional limits, and possible turnover penalties are added to the program, and the covariance matrix is estimated with the methods discussed in [[08 Errore di stima, outlier e shrinkage]] and [[18 Stima dinamica della covarianza - EWMA e GARCH multivariato]]; the long-only constraint itself has an effect similar to that of a robust estimation technique [Jagannathan2003][ChoueifatyFroidureReynier2013].

Empirically, [ChoueifatyCoignard2008] construct the long-only most diversified portfolio at the end of each month on U.S. and Eurozone equity universes (1992-2008), estimating covariance over $250$ days of daily returns and capping the risk contribution at $4\%$ per asset: the portfolio achieves risk-adjusted returns higher than those of the capitalization-weighted benchmark, the minimum-variance portfolio, and the equally weighted portfolio, with volatility lower than that of the capitalization index ($13{,}9\%$ versus $17{,}9\%$ for the Eurozone; $12{,}7\%$ versus $13{,}4\%$ for the United States). [ChoueifatyFroidureReynier2013] extend the analysis to the MSCI World universe (1999-2010): the most diversified portfolio has the highest diversification ratio — its primary objective — and also the highest Sharpe ratio among the portfolios compared, and is thus the closest candidate to the tangency portfolio; the minimum-variance portfolio, on the other hand, achieves the lowest *ex post* volatility. In this sense the MV and the MDP each keep their own promise: minimizing *ex post* volatility for the former, maximizing *ex post* Sharpe ratio for the latter [ChoueifatyFroidureReynier2013]. The three-factor regressions of [FamaFrench1993] confirm that the most diversified portfolio produces the highest alpha among the tested strategies, consistent with its maximum diversification objective and with a balanced exposure to the universe's actual risk factors [ChoueifatyFroidureReynier2013].

When the correlation matrix is not invertible, the solution may not be unique; but since all portfolios so obtained provide maximum diversification — and are perfectly correlated with one another — one is indifferent to the choice [ChoueifatyCoignard2008][ChoueifatyFroidureReynier2013]. Classical financial theory defines the equity risk premium as the return of the non-diversifiable portfolio; the objective of the maximum diversification criterion is precisely to articulate a coherent methodology that delivers to the investor the entire benefit of that premium, and the results recalled here indicate the most diversified portfolio as a solid candidate for the role of non-diversifiable portfolio [ChoueifatyFroidureReynier2013].

## References

- **[Arnott2005]** Arnott, R., Hsu, J., & Moore, P. (2005). Fundamental Indexation. Financial Analysts Journal, 61(2), 83–99.
- **[Berkovitz2001]** Berkovitz, L. D. (2001). Convexity and Optimization in R^n. Wiley.
- **[Choueifaty2006]** Choueifaty, Y. (2006). Methods and Systems for Providing an Anti-Benchmark Portfolio. USPTO 60/816,276.
- **[ChoueifatyCoignard2008]** Choueifaty, Y., & Coignard, Y. (2008). Toward Maximum Diversification. Journal of Portfolio Management, 35(1), 40–51.
- **[ChoueifatyFroidureReynier2013]** Choueifaty, Y., Froidure, T., & Reynier, J. (2013). Properties of the Most Diversified Portfolio. Journal of Investment Strategies, 2(2), 1–22.
- **[FamaFrench1993]** Fama, E. F., & French, K. R. (1993). Common Risk Factors in the Returns on Stocks and Bonds. Journal of Financial Economics, 33(1), 3–56.
- **[HaugenBaker1991]** Haugen, R. A., & Baker, N. (1991). The Efficient Market Inefficiency of Capitalization-Weighted Stock Portfolios. Journal of Portfolio Management, 17(3), 35–40.
- **[Jagannathan2003]** Jagannathan, R., & Ma, T. (2003). Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps. Journal of Finance, 58(4), 1651–1684.
- **[Maillard2010]** Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios. Journal of Portfolio Management, 36(4), 60–70.
- **[Markowitz1952]** Markowitz, H. M. (1952). Portfolio Selection. Journal of Finance, 7(1), 77–91.
- **[Sharpe1964]** Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk. Journal of Finance, 19(3), 425–442.
- **[Sharpe1991]** Sharpe, W. F. (1991). Capital Asset Prices With and Without Negative Holdings. Journal of Finance, 46(2), 489–509.
