---
title: "Limits of MPT and the Black-Litterman Model"
chapter: 3
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-10
---

> [!abstract] Summary
> The chapter examines the six assumptions underlying the base version of mean-variance theory and where they break down, then addresses the practical problem of choosing the target return through three procedures (tangency portfolio, direct maximization of expected utility, the rule of the maximum between the equal-weighted return and the global minimum-variance return). It then presents the Black-Litterman model as a Bayesian response, which updates the expected returns implied by market equilibrium in light of the investor's views, and illustrates its derivation with a complete numerical example.

## The Six Assumptions of the Base Model and Where They Break Down

The base version of the static portfolio selection model of *Modern Portfolio Theory* (MPT) — the one that allows short sales of financial assets [Markowitz1952] — rests on six assumptions. These are assumptions that are not conceptually difficult to understand, even though, in general, not all of them are particularly realistic.

- **Assumption 1**: there are no costs associated with buying and selling financial assets;
- **Assumption 2**: there is no taxation on gains arising from buying and selling financial assets;
- **Assumption 3**: every risky asset is perfectly divisible;
- **Assumption 4**: short sales of every risky asset are allowed;
- **Assumption 5**: economic agents know the first and second moments of the returns of every financial asset;
- **Assumption 6**: the buying and selling actions of economic agents do not affect the probability distributions of the returns of financial assets.

**Assumptions 1–3: the frictionless market.** Assumptions 1, 2, and 3 are jointly known as the *frictionless market* assumptions. It is almost superfluous to point out how unrealistic they are, particularly Assumptions 1 and 2. In recent decades, the ability to invest through online intermediaries has greatly reduced, at the very least, the costs associated with buying and selling financial assets, often bringing them close to zero. As for Assumption 3, it is in fact not always possible to buy or sell arbitrary quantities of financial assets, for example fractions of an asset. In general it is only possible to buy or sell whole numbers of lots of financial assets, each lot in turn made up of a fixed whole number of the same asset: in technical language this is called the *minimum lot size*. For example, the minimum lot for many stocks traded on the Italian stock market consists of a single share, but no less. Classifying as frictions — that is, as elements that can be disregarded if suitably handled — aspects that are instead generally structural in an economic context can lead to dangerously underestimating the impact of these very elements on the static portfolio selection process.

**Assumption 4: absence of institutional restrictions.** Assumption 4 is known as the assumption of *absence of institutional restrictions*. It can generally be realistic, since short sales of financial assets are allowed in many financial markets. It should be noted, however, that in some periods of particular economic or financial tension, the various national financial market supervisory authorities may suspend the possibility of short selling.

**Assumption 5: knowledge of the first and second moments of returns.** Assumptions 5 and 6 relate to the behavior of the economic agents who buy and sell financial assets, that is, investors. Assumption 5 is less innocuous than it may seem, for at least the following reasons. First, it implicitly presupposes that the probability distributions of the returns of financial assets are fully described only by their first and second moments, that is, that these distributions are symmetric and mesokurtic: a natural candidate for this type of distribution is the Gaussian. Furthermore, knowing the first and second moments of the returns of every financial asset means knowing, in addition to the mean and variance of the various returns, also all the covariances between those returns, since covariances too are second moments (cross moments, to be precise). Moreover, economic agents can never know the "true" first and second moments of financial assets: they must settle for some estimate of them, which implies that they possess sufficient resources to carry out such estimates. Finally, Assumption 5 implicitly presupposes that the first and second moments of every financial asset exist, an assumption that is not so obvious. In quantitative finance, the returns of financial assets are in fact frequently modeled using probability distributions whose second moment is not finite and therefore does not exist: this is the case, for example, with *stable Pareto-Lévy* probability distributions [Mandelbrot1963].

**Assumption 6: the price-taking investor.** This last assumption is known as the *price-taking investor* assumption. It essentially states that the quantities of financial assets bought and sold by investors are so small that they cannot influence the prices of those same assets — and hence the returns, which are a function of prices. In other words, all investors involved in buying and selling activity are small investors. This assumption too, like Assumptions 1, 2, and 3, is not very realistic, given the constant presence in financial markets of medium- and large-sized investors, who are rather *price-making investors*, that is, capable of influencing prices by buying and selling large quantities of financial assets.

A thorough discussion of this set of assumptions can be found in [EltonEtAl2007] and [Merton1992]. The underlying mean-variance construction is developed in the chapter [[02 Selezione media-varianza]], while the decision-making framework under uncertainty is taken up in the chapter [[01 Scelta in condizioni di incertezza]].

## The Practical Choice of the Target Return

The mean-variance model is not used intensively by finance professionals. The main practical problem is the **practical choice of the target return**. In classical mean-variance optimization the investor must specify a desired return $r_P = \pi$; however, it can be difficult to identify a "coherent" value of $\pi$, especially in unstable financial contexts. In particular, an investor risks selecting:

- a portfolio with a **high** $\pi$, characterized by excessive variance;
- or a portfolio with a **low** $\pi$, "leaving expected return on the table."

There are three possible practical procedures for addressing this problem.

**Procedure 1 — The Tangency Portfolio.** Following the *fund separation theorem* [Tobin1958], the investor chooses to hold a combination of the risk-free asset and the **tangency portfolio**. The analytical expression of the tangency portfolio is

$$\boldsymbol{x} = \frac{\boldsymbol{V}^{-1}(\boldsymbol{r} - r_c \boldsymbol{e})}{\boldsymbol{e}' \boldsymbol{V}^{-1}(\boldsymbol{r} - r_c \boldsymbol{e})},$$

where $\boldsymbol{V}$ is the variance-covariance matrix of returns, $\boldsymbol{r}$ the vector of expected returns, $r_c$ the risk-free return, and $\boldsymbol{e}$ the unit vector. This procedure avoids the direct specification of $\pi$, since the combination of the risk-free asset and the tangency portfolio implicitly determines the point on the efficient frontier. The **practical limitation** is that, in general, sample tangency portfolios tend to perform poorly out of sample.

**Procedure 2 — Direct Maximization of Expected Utility.** Directly maximizing the investor's expected utility likewise does not require specifying $\pi$. The mathematical programming problem to be solved is

$$\max_{\boldsymbol{x}} \; \boldsymbol{x}'\boldsymbol{r} - \frac{\lambda}{2}\,\boldsymbol{x}'\boldsymbol{V}\boldsymbol{x} \qquad \text{s.t.} \quad \boldsymbol{x}'\boldsymbol{e} = 1,$$

whose solution is

$$\boldsymbol{x} = \frac{1}{\lambda}\boldsymbol{V}^{-1}\left(\boldsymbol{r} - \frac{\boldsymbol{e}'\boldsymbol{V}^{-1}\boldsymbol{r} - \lambda}{\boldsymbol{e}'\boldsymbol{V}^{-1}\boldsymbol{e}}\,\boldsymbol{e}\right).$$

The **practical limitation** concerns the choice of the risk aversion parameter $\lambda$: various scholars have estimated that $\lambda$ lies in the interval $[0, 5]$.

**Procedure 3 — The Maximum Between $r_{1/N}$ and $r_{GMV}$.** A further way to select an optimal mean-variance portfolio without specifying $\pi$ consists in setting the target return equal to

$$\max(r_{1/N}, r_{GMV}),$$

where $r_{1/N}$ is the return of the equal-weighted ("1 over $N$") portfolio and $r_{GMV}$ is the return of the global minimum-variance (GMV) portfolio. The problem to be solved to select the global minimum-variance portfolio is

$$\min_{\boldsymbol{x}} \; \boldsymbol{x}'\boldsymbol{V}\boldsymbol{x} \qquad \text{s.t.} \quad \boldsymbol{x}'\boldsymbol{e} = 1,$$

with solution

$$\boldsymbol{x} = \frac{\boldsymbol{V}^{-1}\boldsymbol{e}}{\boldsymbol{e}'\boldsymbol{V}^{-1}\boldsymbol{e}}.$$

Both portfolios have interesting properties that justify their use.

## Comparison Between the 1/N Equal-Weighted Portfolio and the GMV Portfolio

In general, investors are interested in the portfolio based on $r_{GMV}$ only if it dominates, in the mean-variance sense, the portfolio based on $r_{1/N}$.

The **equal-weighted portfolio** $r_{1/N}$ has the following interesting characteristics:

- it is easy to implement;
- it has been shown to generate noteworthy performance;
- it avoids large concentrations in the same asset;
- it always invests in the best-performing assets;
- it never performs worse than the worst-performing asset;
- in the case of large estimation error in the mean-variance optimization process, it is expected to outperform mean-variance optimization itself [DeMiguelGarlappiUppal2009].

The **global minimum-variance portfolio** $r_{GMV}$ in turn has interesting characteristics:

- it is efficient;
- it is not affected by estimation errors relating to the expected returns of the assets. Considering i.i.d. normal distributions for asset returns, the confidence interval for the means is about 40% wider than the confidence interval for the standard deviation.

Following the 2007 financial crisis, investors shifted toward less risky portfolios. The systematic treatment of estimation error and the periodic review of allocation are taken up, respectively, in the chapters [[05 Modelli fattoriali]] and [[07 Revisione di portafoglio]].

## The Black-Litterman Model as a Bayesian Response

The **Black-Litterman model** [BlackLitterman1992] starts from the equilibrium framework described by the *Capital Asset Pricing Model* (CAPM) [Sharpe1964]. In that framework, the market portfolio $M$ coincides with the portfolio held by the representative investor: in the market portfolio the weights of individual assets are proportional to their share of the total value of the investable assets in the economy. Introducing the risk-free asset, all rational investors hold linear combinations of the risk-free return $R_F$ and the market portfolio $M$, the point of tangency between the frontier of risky assets alone and the line originating from $R_F$ (the *Capital Market Line*). In theory, therefore, all investors should choose the same portfolio of risky assets, namely the market portfolio $M$.

In practice, however, investors have different **opinions** and **expectations** about the assets present in the market. These opinions and expectations represent each investor's specific **views**, and are expressed in terms of:

- the **absolute expected return** of a security, or
- **changes in the expected return** of a security relative to that of another security.

In portfolio selection, integrating these views with market views plays an important role in investment decisions. The Black-Litterman model makes it possible to integrate the investor's views with those of the market, balancing them according to the "confidence" the investor places in their own views.

**Bayes' theorem as the foundation.** The model is built as an application of **Bayes' theorem** [Bayes1763]: the prior distribution of returns, implied by market equilibrium, is updated in light of the new information constituted by the investor's views, yielding a posterior distribution. In schematic terms,

$$\Pr(M\mid W) = \frac{\Pr(M)\,\Pr(W\mid M)}{\Pr(W)},$$

where $\Pr(M)$ denotes the probability of the market state (**prior distribution**), $\Pr(W\mid M)$ the probability of the views state conditional on the market state, $\Pr(W)$ the probability of the views state, and $\Pr(M\mid W)$ the probability of the market state conditional on the views state (**posterior distribution**). Market equilibrium $P(\mu)$ and the investor's views $P(Q\mid\mu)$ thus combine into an integrated distribution $P(\mu\mid Q)$.

## The Distributional Assumptions of the Prior and the Equilibrium Returns Π*

The starting point of the model is the construction of the **prior** distribution of market returns, based on two assumptions.

**Assumption 1.** Consider a market composed of $N$ risky-return securities, with normally distributed returns:

$$\mathbf{R} \sim \mathcal{N}(\mathbf{r}, \mathbf{V}),$$

where $\mathbf{R}$ is the vector of risky returns, $\mathbf{r}$ the vector of expected returns, and $\mathbf{V}$ the variance-covariance matrix of risky returns.

**Assumption 2.** Consider a vector of expected returns that is normally distributed:

$$\mathbf{r} \sim \mathcal{N}(\mathbf{\Pi}, \tau\mathbf{V}),$$

where $\mathbf{\Pi}$ is a suitable vector and $\tau \in [0, +\infty)$ a suitable uncertainty factor.

**Determining the equilibrium expected returns.** The model distinguishes two possible processes, linked by expected utility theory, that relate expected returns, risk, and portfolio weights. In the **forward process**, given the expected returns and the variance-covariance matrix, expected utility is maximized to obtain the (unknown) optimal portfolio weights:

$$\mathbf{r}, \mathbf{V} \;\rightarrow\; \max \mathbb{E}[U(\mathbf{r}, \mathbf{V})] \;\rightarrow\; \mathbf{x}^*.$$

In the **reverse process**, starting from the market weights $\mathbf{x}_M$, which are known (for example via the CAPM), one works back to the (unknown) equilibrium expected returns $\mathbf{\Pi}^*$ that justify them:

$$\mathbf{\Pi}^* \;\leftarrow\; \max \mathbb{E}[U(\mathbf{\Pi}^*, \mathbf{V})] \;\leftarrow\; \mathbf{x}_M, \mathbf{V}.$$

The Black-Litterman model uses the **reverse process**. The vector of equilibrium expected returns is obtained by maximizing the expected quadratic utility function

$$\max_{\mathbf{x}} \; \mathbf{x}_M' \mathbf{\Pi} - \frac{a}{2}\, \mathbf{x}_M' \mathbf{V} \mathbf{x}_M.$$

The first-order optimality conditions are obtained by setting $\partial U / \partial \mathbf{x}_M = 0$, from which

$$\mathbf{\Pi} - a\mathbf{V}\mathbf{x}_M = \mathbf{0}$$

and, finally,

$$\mathbf{\Pi}^* = a\mathbf{V}\mathbf{x}_M.$$

The vector $\mathbf{\Pi}^*$ thus obtained represents the expected returns implied by market equilibrium and constitutes the mean of the model's prior distribution (cf. Assumption 2).

## The Investor's Views and the Posterior Distribution

**The views.** The investor's views are specified through the relation

$$\mathbf{P}\mathbf{r} \sim (\mathbf{Q}, \mathbf{\Omega}),$$

where $\mathbf{P}$ is a $(K, N)$ matrix representing the investor's **view mapping**, $\mathbf{Q}$ is a $(K, 1)$ vector reporting the views in terms of **returns**, and $\mathbf{\Omega}$ is a **diagonal** $(K, K)$ matrix reporting the **variances** of the views; $K \le N$ equals the number of assets on which the investor has formulated views. Views can be formulated in **absolute terms**, that is, on the expected return of a single security, or in **relative terms**, that is, in terms of the return of one security relative to that of another.

**The posterior distribution.** Combining, according to Bayes' theorem, the prior distribution of market returns $\mathbf{r} \sim \mathcal{N}(\mathbf{\Pi}^*, \tau\mathbf{V})$, with $\mathbf{\Pi}^* = a\mathbf{V}\mathbf{x}_M$, with the distribution of the views $\mathbf{P}\mathbf{r} \sim (\mathbf{Q}, \mathbf{\Omega})$, yields the posterior distribution of returns according to the Black-Litterman model:

$$\mathbf{R}_{BL} \sim (\mathbf{r}_{BL}, \mathbf{V}_{BL}),$$

where

$$\mathbf{r}_{BL} = \left[(\tau\mathbf{V})^{-1} + \mathbf{P}'\mathbf{\Omega}^{-1}\mathbf{P}\right]^{-1} \cdot \left[(\tau\mathbf{V})^{-1}\mathbf{\Pi}^* + \mathbf{P}'\mathbf{\Omega}^{-1}\mathbf{Q}\right],$$

$$\mathbf{V}_{BL} = \left[(\tau\mathbf{V})^{-1} + \mathbf{P}'\mathbf{\Omega}^{-1}\mathbf{P}\right]^{-1}.$$

The vector $\mathbf{r}_{BL}$ is thus a weighted average of the implied equilibrium returns $\mathbf{\Pi}^*$ and the investor's views $\mathbf{Q}$: the relative weights depend on the uncertainty about market equilibrium (through $\tau\mathbf{V}$) and on the uncertainty about the views (through $\mathbf{\Omega}$). The smaller $\mathbf{\Omega}$ is — that is, the greater the investor's confidence in their own views — the closer $\mathbf{r}_{BL}$ moves to $\mathbf{Q}$; conversely, the larger $\mathbf{\Omega}$ is, the closer $\mathbf{r}_{BL}$ moves to $\mathbf{\Pi}^*$.

## Complete Numerical Example

The application of the model is illustrated for a market of $N = 3$ assets.

**Step 0 — Starting Data.** The market weights are $\mathbf{x}_M = (0.50,\, 0.30,\, 0.20)'$ and the risk aversion coefficient is $a = 2.5$. The variance-covariance matrix $\mathbf{V}$ is

$$\mathbf{V} = \begin{pmatrix} 0.0225 & 0.0060 & 0.0027 \\ 0.0060 & 0.0400 & 0.0090 \\ 0.0027 & 0.0090 & 0.0324 \end{pmatrix}.$$

**Step 1 — Equilibrium Expected Returns (Prior).** With $\mathbf{R}\sim \mathcal{N}(\mathbf{r}, \mathbf{V})$ and $\mathbf{r}\sim \mathcal{N}(\mathbf{\Pi}, \tau\mathbf{V})$, one computes

$$\mathbf{\Pi}^* = (0.0340,\, 0.0420,\, 0.0263)'.$$

The vector is obtained by applying the reverse optimization formula $\mathbf{\Pi}^* = a\mathbf{V}\mathbf{x}_M$ from the previous section to the market weights and the covariance from Step 0; the inverse map $\mathbf{x}_M = \tfrac{1}{a}\mathbf{V}^{-1}\mathbf{\Pi}^*$ returns exactly the starting weights.

**Step 2 — Formulating the Views.** The investor formulates $K = 2$ views:

- View 1 (absolute): $\mathbb{E}(r_1) = 0.06$;
- View 2 (relative): $\mathbb{E}(r_2 - r_3) = 0.02$.

**Step 3 — View Matrices.** These give

$$\mathbf{Q} = (0.06,\, 0.02), \qquad \mathbf{\Omega} = \begin{pmatrix} 0.0004 & 0 \\ 0 & 0.0009 \end{pmatrix}, \qquad \mathbf{P} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & -1 \end{pmatrix}.$$

**Step 4 — Posterior Distribution.** Applying the model's formulas yields

$$\mathbf{r}_{BL} = (0.0532,\, 0.0478,\, 0.0281),$$

with posterior variance-covariance matrix

$$\mathbf{V}_{BL} = \begin{pmatrix} 0.0002956 & 0.0000605 & 0.0000496 \\ 0.0000605 & 0.0013017 & 0.0009225 \\ 0.0000496 & 0.0009225 & 0.0012185 \end{pmatrix}.$$

**Step 5 — Optimal Portfolio Weights.** The optimal weights according to Black-Litterman are obtained from

$$\mathbf{x}_M = \frac{1}{a}\mathbf{V}^{-1}\mathbf{r}_{BL} = (0.627,\, 0.230,\, 0.143).$$

The comparison between the starting market weights $(0.50,\, 0.30,\, 0.20)$ and the normalized Black-Litterman weights $(0.627,\, 0.230,\, 0.143)$ shows how the investor's views alter the optimal allocation: the asset for which the higher absolute return view was formulated — Asset 1, with a 6% view — sees its weight increase noticeably relative to the starting 50%, at the expense of the other two assets.

## References

- **[Bayes1763]** Bayes T. (1763): «An Essay towards Solving a Problem in the Doctrine of Chances», Philosophical Transactions of the Royal Society of London, vol. 53, pp. 370-418.
- **[BlackLitterman1992]** Black F., Litterman R. (1992): «Global Portfolio Optimization», Financial Analysts Journal, vol. 48, n. 5, pp. 28-43.
- **[DeMiguelGarlappiUppal2009]** DeMiguel V., Garlappi L., Uppal R. (2009): «Optimal versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?», The Review of Financial Studies, vol. 22, n. 5, pp. 1915-1953.
- **[EltonEtAl2007]** Elton E. J., Gruber M. J., Brown S. J., Goetzmann W. N. (2007): Modern Portfolio Theory and Investment Analysis, Apogeo, Milan.
- **[Mandelbrot1963]** Mandelbrot B. (1963): «The Variation of Certain Speculative Prices», The Journal of Business, vol. 36, n. 4, pp. 394-419.
- **[Markowitz1952]** Markowitz H. (1952): «Portfolio Selection», The Journal of Finance, vol. 7, n. 1, pp. 77-91.
- **[Merton1992]** Merton R. C. (1992): Continuous-time Finance, Wiley-Blackwell, Oxford.
- **[Sharpe1964]** Sharpe W. F. (1964): «Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk», The Journal of Finance, vol. 19, n. 3, pp. 425-442.
- **[Tobin1958]** Tobin J. (1958): «Liquidity Preference as Behavior Towards Risk», The Review of Economic Studies, vol. 25, n. 2, pp. 65-86.
