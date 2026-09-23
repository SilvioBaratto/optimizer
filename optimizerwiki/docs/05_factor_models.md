---
title: "Factor Models"
chapter: 5
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-10
---

> [!abstract] Summary
> The chapter shows that mean-variance selection in Markowitz's form requires a number of inputs of order $N^2$, often ill-conditioned, and that factor models reduce this cost to $O(N)$ or $O(NK)$ by routing the co-movement of assets through a few common factors. It derives the covariance structure of Sharpe's diagonal model, of the single-index model, and of $K$-factor models, their OLS estimation with the associated assumptions, and establishes, via equilibrium (CAPM) and no-arbitrage (APT) arguments, the linear relationship between expected return and beta. The chapter closes with empirical tests of these relationships — from the small-firm effect to the Fama-French three-factor model, up to the "factor zoo" — and with notes on smart beta and regularization.

## The Cost of the Full Covariance Matrix and Ill-Conditioning

Mean-variance portfolio selection, described in the chapter [[02 Selezione media-varianza]], requires, in order to build the efficient frontier of $N$ assets, an expected return and a volatility for each asset and a correlation for each pair. Since the correlation matrix is symmetric with unit diagonal, the number of distinct correlation terms is $N(N-1)/2$, and the overall number of parameters is

$$\#\text{parameters} = \underbrace{N}_{\text{exp. returns}} + \underbrace{N}_{\text{variances}} + \underbrace{\frac{N(N-1)}{2}}_{\text{covariances}} = \frac{N(N+3)}{2},$$

of order $O(N^2)$ [Markowitz1952]. For a fund tracking $150$ companies one needs $150$ expected returns, $150$ volatilities, and $\tfrac{150\cdot 149}{2}=11\,175$ correlations, for a total of $11\,475$ inputs; with $N=350$ the parameter count exceeds $60\,000$. This growth, dominated by the covariance term, has three operational consequences: the estimation burden and the resulting estimation error, the difficulty of managing such a large number of parameters, and the risk that the variance-covariance matrix turns out to be **ill-conditioned** due to estimation error and rounding [Sharpe1963].

Ill-conditioning is not an abstraction. Consider three assets with volatilities $\sigma_1=20\%$, $\sigma_2=25\%$, $\sigma_3=30\%$ and pairwise correlations $\rho_{12}=0{,}9$, $\rho_{13}=0{,}9$, $\rho_{23}=-0{,}9$. The correlation matrix

$$R=\begin{pmatrix} 1 & 0{,}9 & 0{,}9 \\ 0{,}9 & 1 & -0{,}9 \\ 0{,}9 & -0{,}9 & 1 \end{pmatrix}$$

has determinant $\det(R)=-2{,}888<0$ and is therefore not positive semi-definite. Forming $\Sigma = DRD$ with $D=\mathrm{diag}(\sigma_1,\sigma_2,\sigma_3)$ and choosing the weights $w=(-1,1,1)$ (with $\sum_i w_i=1$, short sales allowed), one obtains

$$\sigma_p^2 = w'\Sigma w = 0{,}04+0{,}0625+0{,}09-0{,}09-0{,}108-0{,}135 = -0{,}1405<0,$$

a result with no economic meaning. A positive semi-definite matrix instead guarantees $w'\Sigma w\ge 0$ for every $w$: correlations estimated one at a time do not guarantee a jointly valid covariance matrix, optimization can become unstable, and "optimal" portfolios can imply negative variance. As discussed in the chapter [[03 Limiti della MPT e il modello di Black-Litterman]], estimation error is the Achilles' heel of mean-variance optimization. The **factor model** approach consists in not modeling correlations directly, but rather the sources of co-movement: returns are assumed to move jointly because they are driven by a small number of **common factors**, drastically reducing the number of parameters and typically producing a more stable covariance matrix.

## Sharpe's Diagonal Model

The first parsimonious formalization is **Sharpe's diagonal model** [Sharpe1963]. The return of asset $i$ is described by

$$R_i = A_i + B_i I + C_i, \qquad i=1,\dots,N,$$

where $A_i$ and $B_i$ are parameters to be estimated, $I$ is a **common risk factor** — the single, common source of risk for all $N$ assets — and $C_i$ is a random error term with

$$E(C_i)=0,\qquad \mathrm{Var}(C_i)=Q_i,\qquad \mathrm{Cov}(C_i,C_j)=0\ \ (i\neq j).$$

The common factor is in turn modeled as $I=A_{N+1}+C_{N+1}$, with $A_{N+1}$ a parameter, $E(C_{N+1})=0$, $\mathrm{Var}(C_{N+1})=Q_{N+1}$, and $\mathrm{Cov}(C_{N+1},C_i)=0$ for every $i$.

From these assumptions the moments follow. The **expected return** is

$$E(R_i)=E\big(A_i+B_i(A_{N+1}+C_{N+1})+C_i\big)=A_i+B_iA_{N+1},$$

exploiting the linearity of the expectation operator and $E(C_{N+1})=E(C_i)=0$. The **variance** is obtained by noting that the constant terms have zero variance and that $C_{N+1}$ and $C_i$ are uncorrelated:

$$\mathrm{Var}(R_i)=\mathrm{Var}(B_iC_{N+1})+\mathrm{Var}(C_i)=B_i^2 Q_{N+1}+Q_i.$$

The **covariance** between two assets is the model's key result:

$$\mathrm{Cov}(R_i,R_j)=\mathrm{Cov}(B_iC_{N+1}+C_i,\ B_jC_{N+1}+C_j)=B_iB_jQ_{N+1}\qquad(i\neq j),$$

since all other covariance terms vanish. The dependence between any two assets therefore passes **solely** through the sensitivities $B_i,B_j$ to the common factor and the variance $Q_{N+1}$: there is no need to estimate the $N(N-1)/2$ covariances directly.

The model requires $A_i,B_i,Q_i$ for each asset ($3N$ parameters) plus $A_{N+1}$ and $Q_{N+1}$ ($2$ parameters), that is, $3N+2$ against Markowitz's $(N^2+3N)/2$. For few assets ($N=2,3$) Sharpe's model requires more parameters ($160\%$ and $122{,}2\%$ of Markowitz's), but the ratio collapses as $N$ grows: already at $N=10$ it is below half ($49{,}2\%$), and at $N=350$ it is just $1{,}7\%$.

At the portfolio level, the risk factor is formally treated as an **additional asset** (the $(N+1)$-th). From $R_P=\sum_{i=1}^N X_i R_i$, substituting $I=A_{N+1}+C_{N+1}$ and setting $X_{N+1}=\sum_{i=1}^N X_iB_i$, one obtains $R_P=\sum_{i=1}^{N+1}X_i(A_i+C_i)$, from which

$$E(R_P)=\sum_{i=1}^N X_i(A_i+B_iA_{N+1}),$$
$$\mathrm{Var}(R_P)=\sum_{i=1}^N X_i^2\big(Q_i+B_i^2Q_{N+1}\big)+2\sum_{i=1}^N\sum_{j=i+1}^N X_iX_jB_iB_jQ_{N+1}.$$

Portfolio variance depends only on the variances of the assets and on the variance of the factor $Q_{N+1}$, not on the covariances between assets: this is the property that makes the model "diagonal." The selection problem retains the mean-variance form,

$$\max_{x_1,\dots,x_N}\ \lambda\,\mathbf{x}'\mathbf{r}-\mathbf{x}'\mathbf{V}\mathbf{x}\quad\text{s.t.}\quad \mathbf{x}'\mathbf{e}=1,\ x_i\ge 0,$$

with $\mathbf{r}$ the vector of expected returns $A_i+B_iA_{N+1}$ and $\mathbf{V}$ the covariance matrix with simplified structure, but over a drastically smaller number of parameters.

## The Single-Index Model and the Factor Structure of Covariance

The one-factor case is formulated more generally in terms of excess returns $r_i^e=r_i-r_f$:

$$r_i^e = a_i + \beta_i f,$$

where $f$ is a common factor, $\beta_i$ measures its exposure, and $a_i$ is an asset-specific component. If $f$ is **traded** — that is, it is the excess return of a portfolio — it is priced directly by $E[f]$ (examples: the market return $r_m-r_f$, the SMB or HML factors); if it is **non-traded**, the risk premium must be inferred from cross-sectional equilibrium constraints (examples: shocks to consumption growth, GDP innovations).

The most relevant case is the **single-index model**, in which the factor is the market: every asset has a systematic component tied to the market and a firm-specific one. Decomposing $a_i=\alpha_i+e_i$, with $\alpha_i=E(a_i)$ and $e_i$ a zero-mean stochastic error,

$$r_i^e = \alpha_i + \beta_i r_m^e + e_i.$$

This is a statistical representation, not an equilibrium model; its assumptions are: $E(e_i)=0$; correlation between assets passes only through the factor, $E(e_ie_j)=0$ for $i\neq j$; no correlation with the market, $E\!\big(e_i(r_m^e-E(r_m^e))\big)=0$; with $\sigma_{e_i}^2=E(e_i^2)$ and $\sigma_m^2=\mathrm{Var}(r_m^e)$.

From this the moments follow. The expected return is $E(r_i^e)=\alpha_i+\beta_iE(r_m^e)$, the sum of a specific component and a systematic component. The variance, since $\mathrm{Cov}(r_m^e,e_i)=0$, is

$$\sigma_i^2 = \beta_i^2\sigma_m^2+\sigma_{e_i}^2,$$

and the covariance between two assets, since the terms involving the errors vanish,

$$\sigma_{ij}=\beta_i\beta_j\sigma_m^2\qquad(i\neq j).$$

Combining diagonal and off-diagonal terms, the variance-covariance matrix takes the compact form

$$\Sigma = \sigma_m^2\,\beta\beta' + \mathrm{diag}(\sigma_{e_1}^2,\dots,\sigma_{e_N}^2),\qquad \beta=(\beta_1,\dots,\beta_N)',$$

where $\sigma_m^2\beta\beta'$ is the common covariance tied to the market and $\mathrm{diag}(\sigma_{e_i}^2)$ is the specific risk. The number of parameters is $N$ betas $+\ N$ idiosyncratic variances $+\ 1$ market variance $+\ N$ expected returns $=3N+1$, of order $O(N)$: dimensionality drops from $O(N^2)$ to $O(N)$, and $\Sigma$ is typically more stable and better conditioned.

The structure also clarifies the mechanism of diversification. For an equal-weighted portfolio ($w_i=1/N$),

$$r_p^e=\alpha_p+\beta_p r_m^e+e_p,\qquad \alpha_p=\tfrac1N\sum_i\alpha_i,\ \beta_p=\tfrac1N\sum_i\beta_i,\ e_p=\tfrac1N\sum_i e_i,$$

with variance $\sigma_p^2=\beta_p^2\sigma_m^2+\sigma_{e_p}^2$. Since the errors are uncorrelated,

$$\sigma^2(e_p)=\sum_{i=1}^N\Big(\tfrac1N\Big)^2\sigma^2(e_i)=\tfrac1N\,\bar\sigma^2(e)\ \xrightarrow[N\to\infty]{}\ 0.$$

As the number of assets grows, the idiosyncratic component vanishes, while the non-diversifiable **systematic risk** $\beta_p^2\sigma_m^2$ remains.

## Centered Factors

A factor $f$ is **centered** if $E[f]=0$. Given $\mu_f=E[f]$, set $\tilde f=f-\mu_f$, so that $E[\tilde f]=0$. Substituting $f=\tilde f+\mu_f$ into the one-factor model,

$$r_i^e=(\alpha_i+\beta_i\mu_f)+\beta_i\tilde f+e_i=\tilde\alpha_i+\beta_i\tilde f+e_i:$$

the transformation changes the intercept but does not alter $\beta_i$, because covariance with the factor is invariant under translation,

$$\beta_i=\frac{\mathrm{Cov}(r_i^e,f)}{\mathrm{Var}(f)}=\frac{\mathrm{Cov}(r_i^e,\tilde f)}{\mathrm{Var}(\tilde f)}.$$

Centering the factor has an interpretation in terms of shocks: $\tilde f$ isolates the unexpected component, and risk premia compensate for exposure to these unexpected movements. Conversely, when $f=r_m-r_f$ is traded, it is often not centered: leaving the factor uncentered, the relation on expected returns $E[r_i^e]=\alpha_i+\beta_iE[r_m-r_f]$ makes the market risk premium immediately apparent, whereas it would otherwise be absorbed into the intercept. The operational rule is thus: center factors to emphasize their interpretation as shocks; do not center them to highlight the risk premium of traded factors.

## K-Factor Models and the Decomposition of Covariance

A single factor can leave residual cross-sectional correlation in the errors $e_i$, because assets move together for several reasons: sectors, investment styles (size, value, momentum), macroeconomic shocks (rates, inflation, exchange rates, commodities). One thus moves to the multifactor model. In the case of two traded factors $r_m^e$ and $r_l^e$,

$$r_i^e=\alpha_i+\beta_{im}r_m^e+\beta_{il}r_l^e+e_i,$$

if the factors are uncorrelated ($\mathrm{Cov}(r_m^e,r_l^e)=0$) then $\mathrm{Var}(r_i^e)=\beta_{im}^2\sigma_m^2+\beta_{il}^2\sigma_l^2+\sigma_{e_i}^2$ and $\mathrm{Cov}(r_i^e,r_j^e)=\beta_{im}\beta_{jm}\sigma_m^2+\beta_{il}\beta_{jl}\sigma_l^2$; if instead $\mathrm{Cov}(r_m^e,r_l^e)=\sigma_{ml}\neq 0$ interaction terms appear, for example $\mathrm{Cov}(r_i^e,r_j^e)=\beta_{im}\beta_{jm}\sigma_m^2+\beta_{il}\beta_{jl}\sigma_l^2+(\beta_{im}\beta_{jl}+\beta_{il}\beta_{jm})\sigma_{ml}$.

In general, with $K$ factors,

$$r_i^e=\alpha_i+\beta_i'\mathbf{f}+e_i,\qquad \mathbf{r}^e=\alpha+B\mathbf{f}+\mathbf{e},\ \ B\in\mathbb{R}^{N\times K},$$

and covariance decomposes as

$$\Sigma=B\Sigma_f B'+\Sigma_e,\qquad \Sigma_f=\mathrm{Cov}(\mathbf{f}),\ \ \Sigma_e=\mathrm{diag}(\sigma_{e_1}^2,\dots,\sigma_{e_N}^2).$$

The term $B\Sigma_fB'$ has rank at most $K$: systematic risk lies in a $K$-dimensional subspace of $\mathbb{R}^N$, and assets co-vary through their exposures $\beta_i$; $\Sigma_e$ is diagonal, because idiosyncratic risk does not generate covariance between assets. If $K\ll N$, few factors explain most of the co-movement. The variance of a single asset is $\mathrm{Var}(r_i^e)=\beta_i'\Sigma_f\beta_i+\sigma_{e_i}^2$; with uncorrelated factors $\Sigma_f$ is diagonal and

$$\mathrm{Var}(r_i^e)=\sum_{k=1}^K\beta_{ik}^2\mathrm{Var}(f_k)+\sigma_{e_i}^2,\qquad \mathrm{Cov}(r_i^e,r_j^e)=\sum_{k=1}^K\beta_{ik}\beta_{jk}\mathrm{Var}(f_k),$$

while with correlated factors terms in $\mathrm{Cov}(f_m,f_n)$ are added: systematic risk behaves like a portfolio of factors with weights equal to the betas.

The structure is preserved at the portfolio level: from $r_p^e=\sum_i w_i r_i^e$ one obtains $r_p^e=\alpha_p+\beta_p'\mathbf{f}+e_p$ with **portfolio betas** that are weighted averages of the individual betas,

$$\beta_{pk}=\sum_{i=1}^N w_i\beta_{ik}.$$

As for the number of parameters, the $K$-uncorrelated-factor model requires $N(K+2)+K$ parameters (order $O(NK)$) and the correlated-factor one $N(K+2)+\tfrac{K(K+1)}{2}$; for $N=500$ and $K=5$, against the $125\,750$ of the full covariance matrix, only about $3\,500$ parameters are needed. If $K\ll N$, dimensionality drops from $O(N^2)$ to roughly $O(NK)$.

## Specifying and Extracting Factors

It remains to specify and estimate the factors. A first approach is **economic**. **Macroeconomic factors** [ChenRollRoss1986] include industrial production, expected and unexpected inflation, the term spread, and the default spread. **Firm-characteristic factors** lead to the three-factor model

$$r_{it}^e=\alpha_i+\beta_{im}r_{mt}^e+\beta_{iSMB}\,SMB_t+\beta_{iHML}\,HML_t+e_{it},$$

where $SMB$ (Small minus Big) captures size and $HML$ (High minus Low) the book-to-market ratio [FamaFrench1993]. The three methods have mirror-image advantages and disadvantages: factor analysis is purely statistical and requires no predetermined factors, but the factors are not unique and lack immediate economic interpretation; macroeconomic variables have clear interpretation but are hard to measure; firm characteristics are more intuitive but exploit features associated with past anomalies.

A second approach is **statistical**, by extraction. Principal Component Analysis (PCA) observes the historical series of excess returns $\mathbf{r}^e=(r_1^e,\dots,r_N^e)'$, computes their sample covariance, and **diagonalizes** it:

$$\Sigma=V\Lambda V',$$

where the columns of $V$ are orthogonal directions (eigenvectors) and $\Lambda$ contains the variance explained along each direction (eigenvalues). Returns form a cloud in an $N$-dimensional space; PCA identifies the orthogonal axes of maximum dispersion, ordered by explained variance, and retains the first $K$, with $K\ll N$. The first component $f_{1t}=v_1'r_t^e$ is the direction of maximum common variation, often has the same sign for most assets, and is economically similar to a broad market factor; the second, $f_{2t}=v_2'r_t^e$, orthogonal to the first, explains the largest remaining variation, often contrasts groups of assets, and can resemble a style or sector portfolio. In a sample of S&P 500 stocks, the first component alone explains about $30\%$ of total variance, and the first five explain about $50\%$ cumulatively.

The difference is one of principle: in economic models the structure is imposed first and then estimated, with clear interpretation; in PCA the structure emerges from the data by maximizing explained variance, with no prior interpretation, and the factors can change across samples. The trade-off is between statistical efficiency and economic interpretability; both approaches reduce $\Sigma$ to a more tractable form.

## OLS Estimation of Betas: Assumptions, Gauss-Markov, Frisch-Waugh-Lovell

The parameters of factor models are typically estimated with ordinary least squares (OLS). In matrix form, the time-series regression for asset $i$ is $\mathbf{r}_i^e=X\theta_i+\mathbf{e}_i$ with $X=[\mathbf{1},F]$, and the estimator is

$$\hat\theta_i=(X'X)^{-1}X'\mathbf{r}_i^e,$$

with first element $\hat\alpha_i$ and remaining elements $\hat\beta_i$. In the single-index model, $r_{it}^e=\alpha_i+\beta_i r_{mt}^e+e_{it}$, the estimators take the explicit form

$$\hat\beta_i=\frac{\mathrm{Cov}(r_i^e,r_m^e)}{\sigma_m^2}=\frac{\sum_t(r_{it}^e-\bar r_i^e)(r_{mt}^e-\bar r_m^e)}{\sum_t(r_{mt}^e-\bar r_m^e)^2},\qquad \hat\alpha_i=\bar r_i^e-\hat\beta_i\bar r_m^e.$$

The **OLS assumptions** are: linearity of the population model, $Y=\alpha+\beta_1X_1+\cdots+\beta_kX_k+\epsilon$; exogeneity, $E[\epsilon\mid X]=0$ (from which $\mathrm{Cov}(X_j,\epsilon)=0$, i.e. $X_j$ exogenous); i.i.d. sampling; finite fourth moments; homoskedasticity, $\mathrm{Var}(\epsilon\mid X)=\sigma^2$; no perfect collinearity. Exogeneity is the critical condition: if a relevant variable is omitted from the model, its effect is absorbed into the error and, when the regressors are correlated, $\mathrm{Cov}(X_1,\epsilon)=\beta_2\mathrm{Cov}(X_1,X_2)\neq 0$, making the estimator biased and inconsistent ($\hat\beta_1\xrightarrow{p}\beta_1+\rho_{X_1\epsilon}\sigma_\epsilon/\sigma_X$). Similarly, a regressor observed with error, $X^*=X+u$, generates endogeneity with attenuation bias toward zero: in the computable regression $\hat\beta=\beta\,\mathrm{Var}(X)/[\mathrm{Var}(X)+\mathrm{Var}(u)]$. In the case of perfect collinearity the normal equations have no unique solution: the remedy is to drop one of the collinear regressors.

The estimator $\hat\beta_j$ is well approximated by a normal $\hat\beta_j\sim N(\beta_j,\hat\sigma_{\hat\beta_j}^2)$ with

$$\hat\sigma_{\hat\beta_j}^2=\frac{\hat\sigma^2}{\sum_i(X_{ij}-\bar X_j)^2(1-R_j^2)},\qquad \hat\sigma^2=\frac{1}{n-k}\sum_i\hat\epsilon_i^2,$$

where $R_j^2$ is the $R^2$ of the regression of $X_j$ on the other regressors: high correlation between regressors (near-collinearity) inflates the standard error, and in the limit of perfect collinearity $R_j^2\to 1$ and $\hat\sigma_{\hat\beta}\to\infty$. Under the assumptions of linearity, exogeneity, homoskedasticity, and random sampling, the **Gauss-Markov theorem** establishes that OLS is the **best linear unbiased estimator (BLUE)**, that is, the minimum-variance estimator within the class of linear, unbiased estimators.

A result useful for interpreting multifactor betas is the **Frisch-Waugh-Lovell theorem**: in the model $\mathbf{Y}=\beta_1\mathbf{X}^1+\beta_2\mathbf{X}^2+\epsilon$, the estimator $\hat\beta_1$ is obtained by (a) regressing $\mathbf{Y}$ on $\mathbf{X}^2$ to get the residuals $\hat\epsilon_y$, (b) regressing each regressor in $\mathbf{X}^1$ on all the regressors in $\mathbf{X}^2$ to get the residuals $\hat u$, (c) regressing $\hat\epsilon_y$ on $\hat u$. This shows that $\hat\beta_1$ measures the relationship between $Y$ and $X_1$ **once the effects of** $X_2$ **have been removed**: in factor models, the beta on one factor measures exposure net of the other factors.

Empirically, estimated betas contain error, growing with the distance of the coefficient from $1$, and vary over time; on average they tend toward $1$ [Blume1975]. The **Blume correction** measures this adjustment by regressing one period's beta on the previous period's, typically obtaining

$$\beta_{i,2}=0{,}67\,\beta_{i,1}+0{,}33.$$

A raw beta of $0{,}743$ thus becomes $\beta^{adj}=0{,}67\cdot 0{,}743+0{,}33=0{,}828$. It is in fact a **shrinkage** technique toward the average beta, of the same kind as the regularization discussed in the appendix.

## The CAPM: Equilibrium, the Capital Market Line, the Security Market Line

The **Capital Asset Pricing Model** closes the portfolio problem by imposing general equilibrium: given prices, each investor optimizes (partial equilibrium), but when everyone optimizes simultaneously and markets clear (demand = supply), prices and expected returns are determined endogenously. Developed independently by [Sharpe1964], [Lintner1965], and [Mossin1966], it is a positive theory linking expected returns solely to systematic risk. Its **assumptions** concern markets (perfect competition with price-taking investors, single-period horizon, only publicly traded securities, lending and borrowing at rate $r_f$, no frictions or taxes, infinitely divisible securities, short sales allowed) and information and preferences (perfect, costless information, rational mean-variance-optimizing investors, homogeneous expectations).

Under homogeneous expectations all investors face the same mean-variance frontier and choose the same tangency portfolio; for markets to clear, this portfolio must coincide with the **market portfolio** $m$, with weights equal to market-capitalization shares,

$$w_i^*=\frac{P_i\times \#\,\text{shares}_i}{\sum_j P_j\times \#\,\text{shares}_j}.$$

Each investor holds the same risky portfolio in different amounts, combining it with $r_f$ according to their own risk aversion. This gives rise endogenously to the **capital market line (CML)**, the efficient capital allocation line in which the risky portfolio is $m$; since $m$ is efficient, the CML has the maximum Sharpe ratio and provides the theoretical foundation for passive index investing. The **market price of risk** depends on average risk aversion $\bar A$ and on aggregate risk:

$$E(r_m)-r_f=\bar A\,\sigma_m^2,\qquad \bar A=\frac{E(r_m)-r_f}{\sigma_m^2}.$$

The pricing relation is derived using a portfolio approach. The relevant risk of a single asset $s$ is its contribution to the risk of the market portfolio. Isolating from the variance $\mathrm{Var}(r_m)=\sum_i\sum_j w_i^*w_j^*\sigma_{ij}$ the terms relating to $s$,

$$\text{part of }\mathrm{Var}(r_m)\text{ due to }s=w_s^*\,\mathrm{Cov}(r_s,r_m),$$

while the contribution to the premium is $w_s^*[E(r_s)-r_f]$. In equilibrium the ratio of the contribution to the premium to the contribution to variance must be the same for every asset and for the market itself:

$$\frac{E(r_s)-r_f}{\sigma_{sm}}=\frac{E(r_m)-r_f}{\sigma_m^2}.$$

Rearranging yields the **fundamental equation of the CAPM**

$$E(r_s)-r_f=\beta_s\,[E(r_m)-r_f],\qquad \beta_s=\frac{\sigma_{sm}}{\sigma_m^2},$$

or, in excess terms, $E(r_s^e)=\beta_s E(r_m^e)$. The premium does not depend on the asset's total volatility: only systematic risk is priced. Multiplying by the weights and summing, the relation holds for every portfolio, $E(r_p)=r_f+\beta_p[E(r_m)-r_f]$, and for the market $\beta_m=1$. Graphically, in the $(\beta_i,E(r_i))$ space, this is the **security market line (SML)**

$$E(r_i)=r_f+\beta_i[E(r_m)-r_f],$$

with intercept $r_f$ and slope equal to the market risk premium. Unlike the CML — which applies only to efficient portfolios and links return to total risk $\sigma$ — the SML applies to every asset and portfolio and links return only to systematic risk $\beta$: total volatility is not rewarded per se. A correctly priced asset lies on the SML; assets above it are undervalued, those below overvalued.

The deviation from the SML opens the door to **active management**. Empirically one estimates the single-index regression $r_{it}^e=\alpha_i+\beta_i r_{mt}^e+e_{it}$, which allows $\alpha\neq 0$: since the CAPM predicts $\alpha=0$, a positive intercept signals an anomalous return not explained by beta. For the estimate $\hat r_{At}^e=0{,}01+0{,}9\,r_{mt}^e$, since $E(r_{At}^e)=0{,}01+0{,}9\,E(r_{mt}^e)>0{,}9\,E(r_{mt}^e)$, the asset is undervalued. In a sample of equity mutual funds over 1972-1991, however, the average alpha is slightly negative and not significant ($=-0{,}06$), consistent with the idea that, net of costs, active managers do not systematically beat the CAPM [Malkiel1995].

The model has been extended in many directions. In the absence of a risk-free asset one obtains the **zero-beta CAPM** [Black1972]: every efficient portfolio is associated with a zero-correlation portfolio, and for the market there exists $z$ with $\beta_z=0$ such that $E(r_i)-E(r_z)=\beta_i[E(r_m)-E(r_z)]$; since generally $E(r_z)>r_f$, the resulting SML is flatter. The **Intertemporal CAPM (ICAPM)** [Merton1973] introduces, beyond return uncertainty, the risk of changes in investment opportunities and consumption prices: investors form hedging portfolios and accept lower expected returns on them, so that $K$ extra-market factors require premia,

$$E(r_i)=\beta_{iM}E(r_m)+\sum_{k=1}^K\beta_{ik}E(r_k).$$

## The APT: Arbitrage, Well-Diversified Portfolios, the Multifactor SML

**Arbitrage Pricing Theory** [Ross1976] starts from a different basis: it requires neither the mean-variance paradigm nor equilibrium conditions, but only the **no-arbitrage principle**. Its assumptions are that returns follow a linear factor structure, that no arbitrage opportunities exist, that assets are numerous enough to allow the law of large numbers and the construction of portfolios that diversify away specific risk, and that the factors are observable or estimable [Wei1988]. **Arbitrage** occurs when a certain profit is possible with no net investment; the **law of one price** states that two assets equivalent in every relevant respect must have the same price, a guarantee ensured by arbitrageurs who buy the undervalued asset and sell the overvalued one until the opportunity is eliminated.

For a portfolio with a single centered macroeconomic factor $\tilde f$,

$$r_p^e=E(r_p^e)+\beta_p\tilde f+e_p,\qquad E(r_p^e)=\alpha_p+\beta_p\mu_f,$$

if $p$ is **well diversified** ($w_i\approx 0$ as $N\to\infty$) idiosyncratic risk vanishes, $\sigma_{e_p}^2\to 0$, and $r_p^e=E(r_p^e)+\beta_p\tilde f$. Two well-diversified portfolios with the same beta cannot have different premia: if $A$ returns $0{,}10+1{,}0\cdot\tilde f$ and $B$ returns $0{,}08+1{,}0\cdot\tilde f$, a long position in $A$ and a short one in $B$ for a million cancels the random component and leaves a certain profit of $\$20\,000$ with no net investment — an arbitrage, unsustainable in equilibrium. More generally, premia must be proportional to betas: a portfolio $C$ below the line joining $A$ and $r_f$ would be arbitraged against the combination $0{,}3\,r_f+0{,}7\,A$, which has the same beta and a higher return.

Specifying the factor as the excess market return, $\mu_f=E(r_m^e)$, and imposing $\alpha_p=0$ (no arbitrage) for well-diversified portfolios, one obtains the same SML as the CAPM

$$E(r_p^e)=\beta_p E(r_m^e)\qquad\Longleftrightarrow\qquad E(r_p)=r_f+\beta_p E(r_m-r_f),$$

but without the restrictive assumptions of the CAPM and without requiring the true market portfolio: any well-diversified portfolio on the SML can serve as a reference. The derivation rests on only three requirements — a factor model, a sufficient number of assets for diversification, and no arbitrage. The APT applies to well-diversified portfolios, not necessarily to individual assets: if a single asset violates the relation, the effect on a large portfolio is negligible, so individual mispricings are allowed; but if many assets violate it, the relation fails for portfolios too, and arbitrage arises.

The generalization to $K$ factors gives, for the excess return of a well-diversified portfolio, $r_p^e=E(r_p^e)+\sum_{k=1}^K\beta_{kp}\tilde f_k$. If the factors are traded, $\tilde f_k=r_k^e-E(r_k^e)$, and rearranging separates a fixed component from a random one:

$$r_p^e=\underbrace{\Big[E(r_p^e)-\sum_k\beta_{kp}E(r_k^e)\Big]}_{\text{fixed}}+\underbrace{\sum_k\beta_{kp}r_k^e}_{\text{replicating portfolio}}.$$

The second term is a **replicating portfolio** that invests $\beta_{kp}$ in each factor and $1-\sum_k\beta_{kp}$ in the risk-free asset. Since this portfolio exactly replicates the random component of $p$, if the fixed components differed one would buy the higher one and sell the lower one, obtaining an arbitrage. The absence of arbitrage thus requires equality of the fixed components, from which the **multifactor SML**

$$E(r_p^e)=\sum_{k=1}^K\beta_{kp}\,\underbrace{E(r_k^e)}_{\lambda_k},\qquad E(r_i)=r_f+\sum_{k=1}^K\beta_{ki}\lambda_k,$$

where the $\lambda_k$ are the **prices of risk** of the factors. A **factor portfolio** for factor $k$ has unit exposure to $k$ and zero exposure to the others, and for it $\lambda_k=E(r_k^e)$. In a $K$-factor model, $K+1$ assets suffice to replicate any given set of $K$ betas ($K$ for the betas, $1$ to impose weights summing to one); the factors need not be traded, since replicating portfolios can be built from existing assets. Formally, an **arbitrage strategy** satisfies zero net investment $\sum_i w_i=0$, zero risk $\sum_i w_i\beta_{ki}=0\ \forall k$, and positive expected payoff $E(\sum_i w_i r_i^e)>0$. The APT pricing equation can coexist with the CAPM in the same market without necessarily implying it.

## Empirical Tests: from CAPM Tests to the Factor Zoo

Linear factor models amount to a multivariate SML $E(r^e)=\beta'\lambda$ and are tested with two types of regressions: **time-series** and **cross-sectional**. It is convenient to work with excess returns, so that the model's restriction is simply $\alpha_i=0$: for a traded factor, $r_{it}^e=\alpha_i+\beta_i f_t+e_{it}$ must have a zero intercept for every asset, and the premium is estimated as the sample mean of the factor, $\hat\lambda=E(f)$. The procedure consists of estimating the regression for each test asset, estimating $\hat\lambda$, testing the significance of the individual pricing errors $\alpha_i$, and testing their joint nullity, using OLS or robust standard errors (White, Hansen-Hodrick for overlapping returns).

When the factor is not itself a return, the **cross-sectional approach** is needed: betas are first estimated via time-series regressions, then $E(r_i^e)=\alpha+\hat\beta_i'\lambda+u_i$ is regressed across assets to obtain the premia, with $\alpha$ the pricing error (expected to be zero). This approach is necessary because imposing the restriction on the intercept produces $a_i=\beta_i'(\lambda-E(f))$, which is testable only with an estimate of $\lambda$. The **Fama-MacBeth** procedure [FamaMacBeth1973] estimates a cross-sectional regression in each period,

$$r_{it}^e=\alpha_t+\hat\beta_i'\lambda_t+u_{it},$$

and averages over time, $\hat\lambda=\tfrac1T\sum_t\hat\lambda_t$, with standard errors built from the time-series variation of the estimates, $\sigma^2(\hat\lambda)=\tfrac{1}{T^2}\sum_t(\hat\lambda_t-\hat\lambda)^2$: partitioning the sample allows one to study the sampling error of the averages, also accounting for serial correlation in $\hat\lambda_t$.

**Portfolios** are commonly used as test assets rather than individual stocks: sorting stocks by a characteristic sharpens the cross-sectional variation and mitigates errors-in-variables. Early tests on individual stocks produced a SML that was too flat due to attenuation bias from measurement error in the betas; forming portfolios sorted by beta produces more accurate estimates. The typical procedure identifies a characteristic linked to average returns, sorts stocks by it, checks whether the portfolios differ in average return and whether these differences are explained by beta; if not, an **anomaly** arises that requires additional factors. One of the first failures is the **small-firm effect**: portfolios of small firms achieve returns higher than predicted by their market beta [Banz1981].

The **Roll critique** [Roll1977] observes that the central implication of the CAPM is the mean-variance efficiency of the market portfolio, which is however unobservable: any ex post efficient proxy mechanically satisfies the CAPM, so tests jointly verify the model and the proxy. Follow-up work extends the proxy [Stambaugh1982], shows that a proxy sufficiently correlated with the true market preserves the power of the test [Shanken1987], or finds that the CAPM can be rejected even with nearly efficient proxies [RollRoss1994].

The **Fama-French three-factor model** [FamaFrench1993; FamaFrench1996] resolves the CAPM's limitation in explaining the **value premium**: value stocks (high book-to-market) earn higher average returns than growth stocks, and the difference is not captured by market betas. Adding the factors $SMB$ (small-minus-big) and $HML$ (high-minus-low),

$$E(r_i)-r_f=\beta_{i,m}E(r_m-r_f)+\beta_{i,h}HML+\beta_{i,s}SMB,$$

the betas on $SMB$ and $HML$ explain the cross-sectional variation in average returns much better. The premium admits risk-based interpretations — $HML$ predicts GDP growth beyond the market [LiewVassalou2000], and the $HML$ beta is higher in recessions, when the market premium is higher [PetkovaZhang2005] — and behavioral interpretations tied to optimism and overreaction on glamour stocks. The model can be read as an implementation of the APT, with an $R^2$ of $90$-$95\%$. Further refinements adjust $HML$ for intangible assets, preserving its pricing ability [EisfeldtKimPapanikolaou2022].

Other factors have been added. **Past performance** generates momentum (short-term) and reversal (long-term): buying the decile of winners and selling that of losers, the momentum strategy on a $12$-$2$ formation period returns on average $+1{,}51\%$ monthly, while reversal on a $60$-$13$ formation period returns $-0{,}74\%$ [FamaFrench1996]; it is common to augment the three factors with a momentum factor. **Liquidity risk** is priced: constructing an aggregate liquidity measure from price reversals in illiquid stocks, stocks that covary with aggregate liquidity earn higher returns, with alphas unexplained by either the CAPM or Fama-French [PastorStambaugh2003]. Non-traded macroeconomic factors — industrial production, default spread, term structure, inflation — are priced in two-stage tests [ChenRollRoss1986]. The proliferation of predictors has produced the so-called **"factor zoo"** [HouXueZhang2015], with problems of data mining and multiple testing and anomalies that weaken after discovery: systematic variation in returns goes beyond the CAPM beta, but its correct specification remains an open question.

## The Efficient Frontier Under Factor Models and Smart Beta Strategies

Imposing a factor structure on the covariance matrix changes the efficient frontier. Comparing, on five assets and the three Fama-French factors, the full-covariance approach (Markowitz, $N(N+3)/2=20$ parameters), the single-index model ($3N+1=16$), and the correlated three-factor model ($N(K+2)+K(K+1)/2=31$), the three frontiers turn out very similar in the central region, with slight divergences at high volatility. The Sharpe ratios of the tangency portfolios are nearly identical — $35{,}30\%$ for Markowitz, $34{,}46\%$ for the single-index model, $34{,}73\%$ for the three-factor model — because the factor structure captures most of the systematic co-movement. Markowitz is the most flexible, but factor models reduce dimensionality with limited impact on performance, trading small efficiency losses for greater robustness; indeed, the accumulation of estimation errors in the full covariance matrix can produce a portfolio that is effectively inferior to the factor-based one.

**Smart beta strategies** are built on the same principle. Since some factors are associated with persistent risk premia, these strategies deliberately tilt the portfolio toward specific factors in a transparent, rules-based way: while traditional indexing offers exposure only to the market factor, smart beta seeks targeted exposure to value, size, momentum, and similar factors. Formally, for a portfolio with weights $w$ one has $r_p^e=\alpha_p+\beta_p'\mathbf{f}+e_p$ with $\beta_p=w'B$, and the expected return according to the model is $E[r_i^e]=\alpha_i+\beta_i'\lambda$, with $\lambda$ the vector of risk premia. Building a smart beta portfolio means choosing $w$ so that $\beta_p$ is tilted toward the selected factors, for example maximizing exposure to a factor $k$,

$$\max_w \beta_{p,k}\quad\text{s.t.}\quad \mathbf{1}'w=1,$$

or solving a mean-variance optimization with factor structure $\max_w w'B\lambda-\tfrac{\gamma}{2}w'\Sigma w$, with $\Sigma=B\Sigma_fB'+\Sigma_e$. A value-investing criterion sets weights as a function of book-to-market, $w_i\propto f(B/M_i)$. Performance is then evaluated with the regression $r_p^e=\alpha_p+\beta_p'\mathbf{f}+e_p$, where $\alpha_p$ measures anomalous performance and $\beta_p$ the realized factor exposures.

## Appendix: Ridge and LASSO Regularization

OLS minimizes training error, but a model that reproduces the data perfectly can perform poorly out of sample (**overfitting**). The expected prediction error decomposes as

$$\text{Error}=\underbrace{\text{Bias}^2}_{\text{too simple}}+\underbrace{\text{Variance}}_{\text{too complex}}+\underbrace{\sigma_e^2}_{\text{irreducible}}:$$

fewer variables or stronger shrinkage raise bias, more variables or correlated predictors raise variance. With many candidate factors, often correlated (the "factor zoo"), **regularization** deliberately introduces some bias by constraining $\beta$ in exchange for a strong reduction in variance. Since it compares the magnitude of coefficients, it requires **standardizing** the data.

**Ridge regression** imposes an $L_2$ constraint, $\min_\beta\mathbf{e}'\mathbf{e}$ subject to $\|\beta\|_2\le t$, equivalent to the Lagrangian form with penalty

$$\min_\beta\ \sum_t\Big(r_{it}^e-\alpha_i-\sum_k\beta_{ik}f_{kt}\Big)^2+\lambda\sum_{k=1}^K\beta_{ik}^2,\qquad \lambda\ge 0.$$

The penalty $\lambda\beta_k^2$ grows quadratically, hitting large coefficients much harder than small ones, and does not include the intercept so as not to distort the mean of $r_{it}^e$. The parameter $\lambda$ is a hyperparameter chosen by cross-validation, not estimated from the training data: for $\lambda=0$ one recovers OLS, for $\lambda\to\infty$ all coefficients tend to zero, and for intermediate values they are shrunk toward zero but not set exactly to zero.

**LASSO regression** (Least Absolute Shrinkage and Selection Operator) instead uses the $L_1$ penalty,

$$\min_\beta\ \sum_t\Big(r_{it}^e-\alpha_i-\sum_k\beta_{ik}f_{kt}\Big)^2+\lambda\sum_{k=1}^K|\beta_{ik}|,$$

with no closed-form solution (it is solved by coordinate descent). Unlike ridge, LASSO drives the coefficients of negligible variables exactly to zero, performing **variable selection** and producing a sparser, more interpretable model. In summary: ridge (circular constraint, closed form) shrinks all coefficients proportionally without zeroing them out, and is ideal with many correlated factors that are all potentially relevant; LASSO (diamond-shaped constraint, no closed form) can zero out the less relevant coefficients, and is ideal when, among many factors, only a few really matter. As noted earlier, the Blume correction of betas is a special case of this same shrinkage principle.

## References

- **[Banz1981]** Banz, R. W. (1981). The Relationship between Return and Market Value of Common Stocks. Journal of Financial Economics, 9(1), 3-18.
- **[Black1972]** Black, F. (1972). Capital Market Equilibrium with Restricted Borrowing. The Journal of Business, 45(3), 444-455.
- **[Blume1975]** Blume, M. E. (1975). Betas and Their Regression Tendencies. The Journal of Finance, 30(3), 785-795.
- **[ChenRollRoss1986]** Chen, N.-F., Roll, R., & Ross, S. A. (1986). Economic Forces and the Stock Market. The Journal of Business, 59(3), 383-403.
- **[EisfeldtKimPapanikolaou2022]** Eisfeldt, A. L., Kim, E., & Papanikolaou, D. (2022). Intangible Value. Review of Finance, 26(6), 1449-1483.
- **[FamaFrench1993]** Fama, E. F., & French, K. R. (1993). Common Risk Factors in the Returns on Stocks and Bonds. Journal of Financial Economics, 33(1), 3-56.
- **[FamaFrench1996]** Fama, E. F., & French, K. R. (1996). Multifactor Explanations of Asset Pricing Anomalies. The Journal of Finance, 51(1), 55-84.
- **[FamaMacBeth1973]** Fama, E. F., & MacBeth, J. D. (1973). Risk, Return, and Equilibrium: Empirical Tests. Journal of Political Economy, 81(3), 607-636.
- **[HouXueZhang2015]** Hou, K., Xue, C., & Zhang, L. (2015). Digesting Anomalies: An Investment Approach. The Review of Financial Studies, 28(3), 650-705.
- **[LiewVassalou2000]** Liew, J., & Vassalou, M. (2000). Can Book-to-Market, Size and Momentum Be Risk Factors That Predict Economic Growth? Journal of Financial Economics, 57(2), 221-245.
- **[Lintner1965]** Lintner, J. (1965). The Valuation of Risk Assets and the Selection of Risky Investments in Stock Portfolios and Capital Budgets. The Review of Economics and Statistics, 47(1), 13-37.
- **[Malkiel1995]** Malkiel, B. G. (1995). Returns from Investing in Equity Mutual Funds 1971 to 1991. The Journal of Finance, 50(2), 549-572.
- **[Markowitz1952]** Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
- **[Merton1973]** Merton, R. C. (1973). An Intertemporal Capital Asset Pricing Model. Econometrica, 41(5), 867-887.
- **[Mossin1966]** Mossin, J. (1966). Equilibrium in a Capital Asset Market. Econometrica, 34(4), 768-783.
- **[PastorStambaugh2003]** Pástor, L., & Stambaugh, R. F. (2003). Liquidity Risk and Expected Stock Returns. Journal of Political Economy, 111(3), 642-685.
- **[PetkovaZhang2005]** Petkova, R., & Zhang, L. (2005). Is Value Riskier than Growth? Journal of Financial Economics, 78(1), 187-202.
- **[Roll1977]** Roll, R. (1977). A Critique of the Asset Pricing Theory's Tests. Part I: On Past and Potential Testability of the Theory. Journal of Financial Economics, 4(2), 129-176.
- **[RollRoss1994]** Roll, R., & Ross, S. A. (1994). On the Cross-sectional Relation between Expected Returns and Betas. The Journal of Finance, 49(1), 101-121.
- **[Ross1976]** Ross, S. A. (1976). The Arbitrage Theory of Capital Asset Pricing. Journal of Economic Theory, 13(3), 341-360.
- **[Shanken1987]** Shanken, J. (1987). Multivariate Proxies and Asset Pricing Relations: Living with the Roll Critique. Journal of Financial Economics, 18(1), 91-110.
- **[Sharpe1963]** Sharpe, W. F. (1963). A Simplified Model for Portfolio Analysis. Management Science, 9(2), 277-293.
- **[Sharpe1964]** Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk. The Journal of Finance, 19(3), 425-442.
- **[Stambaugh1982]** Stambaugh, R. F. (1982). On the Exclusion of Assets from Tests of the Two-Parameter Model: A Sensitivity Analysis. Journal of Financial Economics, 10(3), 237-268.
- **[Wei1988]** Wei, K. C. J. (1988). An Asset-Pricing Theory Unifying the CAPM and APT. The Journal of Finance, 43(4), 881-892.
