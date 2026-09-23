---
title: "Introduction and Purpose"
chapter: 0
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-10
---

> [!abstract] Summary
> The chapter situates portfolio selection within the investment decision-making process and asset allocation, formally defines the financial portfolio as a transfer of wealth under uncertainty, and introduces the basic quantitative tools: measuring single-period performance via percentage and logarithmic returns, the mean–variance pair as measures of return and risk, the limits of variance, and the mean-variance dominance criterion. It also establishes the three-step logical structure — measuring uncertainty, defining an efficiency criterion, optimizing — around which the entire monograph is organized, closing with a map of the chapters.

## The Investment Decision-Making Process

**Asset allocation** (or portfolio management) refers to the set of procedures and decisions an investor implements in order to build and manage a portfolio with the desired characteristics, starting from assets with different characteristics. This set of activities unfolds as an ordered decision-making process, whose stages are:

- selection of asset classes;
- assumptions about asset characteristics;
- strategic asset allocation;
- tactical asset allocation;
- short-term portfolio review;
- long-term portfolio review;
- ex-post analysis.

Upstream of portfolio composition lies the assessment of asset characteristics. The available information falls into three categories — economic and financial fundamentals; intrinsic valuation elements; psychological and technical factors — and feeds a historical and forward-looking analysis aimed at estimating three key quantities: **return**, **risk**, and **dependence**. Assets themselves can be classified according to various dichotomies: equities/bonds, domestic/international, reference currency, conventional/alternative.

Each operational stage of the investment process corresponds to a class of decision problems, which can be tackled with specific quantitative tools. This parallel makes it possible to order financial theory according to the sequence of problems the manager encounters: the foundational topics of portfolio theory — the mean–variance model, the Capital Asset Pricing Model, and multifactor models — precede the more advanced topics of asset allocation, such as active management, tracking error volatility control, risk management, strategic, tactical, and dynamic allocation, top-down and bottom-up strategies, and performance attribution.

## Asset Allocation: Approaches and Determinants of Advantage

Asset allocation can be classified according to three distinct criteria. **With respect to the information set** used, a distinction is made between a *quantitative* approach, which treats the variables associated with each asset class as inputs to a constrained optimization problem; a *qualitative* approach, which considers only fundamental macroeconomic variables, company-level variables, and psychological and technical factors; and a *mixed* approach, which combines both categories of information at different levels and for different purposes. **With respect to the market**, allocation can be *conservative*, *moderate*, or *aggressive*, depending on whether the level of risk and return is below, in line with, or above the market; this positioning depends on the assets included in the portfolio and varies with market conditions. **With respect to the time horizon**, allocation is *strategic* (long-term view), *tactical* (short-term view), or *mixed*; the decomposition of overall performance across the different decision levels constitutes performance attribution.

The advantage of asset allocation over other approaches depends on three components: the **investor**, the **market**, and the **asset classes**. On the investor side, what matters are objectives, experience and mindset, financial flows and wealth profile, the time horizon, as well as awareness of the limits of the models and procedures used and plain common sense; investor types include banks, financial companies, banking foundations, pension funds, hedge funds, private individuals, and asset management companies (SGR). The reference market is analyzed through historical performance and scenario analysis, taking into account its location and microstructure. The choice of assets in the portfolio stems from the intersection of two groups of factors — the investor side (wealth, experience, and preferences) and the asset side (attention and expertise required, costs, liquidity, ethics) — from which the selection criteria for maturity, quality, and tax regime are derived.

The approach produces real operational advantages only if certain conditions are met, which can be summarized in six critical points. The first is **portfolio performance**, defined as

$$\text{Performance} = \text{Total Real Return adjusted for Risk},$$

where the return must be expressed in currency terms, *real* means adjusted for inflation, *total* takes into account the various sources of gain (capital account, interest, dividends) net of costs (taxes, transaction costs), and *risk* is viewed from multiple angles (market, liquidity, …); the approach is useful when there is no asset that dominates over time, so as to benefit from diversification, but it incurs a lag in recognizing profit opportunities or in avoiding significant losses during extreme market phases. The second is the **stability of relationships between assets**: the presence of structural breaks limits the use of past information to describe future behavior. The third is **dependence between assets**: high dependence — of which correlation is one possible measure — undermines the effects of diversification, and studies show that it increases over the long run due to globalization while tending to rise in the short run during periods of panic. The fourth is the **sensitivity of results**: if the optimal composition is highly sensitive to the initial inputs, the approach is operationally unusable, with remedies such as the Black-Litterman model, robust estimation techniques, and sensitivity analysis. The fifth is **rebalancing**, whose frequency and conditions affect performance and require models that account for the variability of estimates, transaction costs, and stop-loss rules. The sixth concerns **errors and fraud**, that is, the ability to avoid mistakes and the models used to monitor the investor's activity. These six factors guide the topics covered in the subsequent chapters, from portfolio review ([[07 Revisione di portafoglio]]) to the Black-Litterman model ([[03 Limiti della MPT e il modello di Black-Litterman]]).

## Investing as a Transfer of Wealth Under Uncertainty

The starting point of the theory is the consumer choice problem in its simplest form,

$$\max_{q_1,\dots,q_N} u(q_1,\dots,q_N) \quad \text{s.t.} \quad \begin{cases} \sum_{i=1}^{N} q_i p_i = M \\ q_i \geq 0,\ i=1,\dots,N, \end{cases}$$

where $q_1,\dots,q_N$ are the quantities of goods to be purchased, $u(\cdot)$ is the utility function, $M$ is income, and $p_1>0,\dots,p_N>0$ are the prices. The implicit assumption of this model is that, given the instant $t$ at which the economy is defined, that economy is born, lives, and dies at $t$, with no relation to future instants. This assumption is not realistic: many consumers do not use all of their current income $M$ for current consumption, but transfer the remaining part to future instants. A consumer in fact faces two related economic decisions — how to allocate current consumption among goods and services, and how to invest among various assets — known as the consumption–savings decision and the portfolio selection decision [ConstantinidesMalliaris1995]. The need to transfer current wealth to future instants is the reason a financial economy is needed.

The purpose of the monograph is the second decision: how to invest among various assets in order to transfer wealth from the current period to a future one. Since the future is more or less unknown — the riskiness faced by the investor is, roughly, of the type involved in a *random walk* — investing is a risky activity, and financial economics provides the theory, methods, and tools with which to transfer wealth while managing uncertainty. In general the two decisions cannot be made independently; however, many of the important results of portfolio theory are more easily derived in a single-period setting, where the consumption–savings allocation has a limited substantial impact on the results [ConstantinidesMalliaris1995]. We therefore place ourselves in a single-period economy, in which the portfolio selection problem is formalized directly, without paying attention to the consumption–savings problem. While deterministic calculus is adequate for maximizing utility subject to a budget constraint, portfolio selection instead involves a decision under uncertainty [ConstantinidesMalliaris1995].

Investment choices are represented as random variables. Let $\mathbb{X}=\{X_1,\dots,X_N\}$ be a set of $N$ stochastic investment choices, each characterized by negative/positive outcomes with given probabilities; in the discrete case

$$X_i = \{(x_{i,1},p_{i,1}),\dots,(x_{i,j},p_{i,j}),\dots,(x_{i,M_i},p_{i,M_i})\},$$

with $x_{i,j}$ the $j$-th realization of $X_i$ and $p_{i,j}$ its probability of occurrence, $0\le p_{i,j}\le 1$ and $\sum_{j=1}^{M_i} p_{i,j}=1$. How to identify the optimal choices depends on the investor's attitude toward randomness: in general, the "rational" investor prefers more to less and is risk-averse. The formalization of preferences under uncertainty is the subject of the chapter [[01 Scelta in condizioni di incertezza]].

On this basis the central object is defined. Let $W$ be a given wealth and $\mathbb{X}=\{X_1,\dots,X_N\}$ a set of $N$ investment choices (for example a stock market). A **(financial) portfolio** is defined as an $N$-vector

$$\mathbf{x}' = (x_1,\dots,x_N)$$

such that the generic element $x_i$, with $i=1,\dots,N$, denotes the percentage of $W$ invested in $X_i$, with

$$\sum_{i=1}^{N} x_i = 1.$$

In general terms, the portfolio is the technical tool that "transfers" wealth from one period to the next [Ingersoll1987].

## Measuring Single-Period Performance

How to measure the single-period performance of an investment choice depends on the law chosen to describe the price dynamics. The convention of financial mathematics adopts the **net percentage return**. Setting

$$P_t = P_{t-\Delta t}(1+R_{\%,\Delta t}) - D_{(t-\Delta t,t]},$$

where $P_t$ is the price at $t$, $D_{(t-\Delta t,t]}\ge 0$ is the dividend paid over $(t-\Delta t,t]$, and $R_{\%,\Delta t}$ is the net percentage return from $t-\Delta t$ to $t$, one obtains

$$R_{\%,\Delta t} = \frac{P_t + D_{(t-\Delta t,t]} - P_{t-\Delta t}}{P_{t-\Delta t}}.$$

With this dynamics, if $R_{\%,\Delta t} < D_{(t-\Delta t,t]}/P_{t-\Delta t} - 1$ (with $P_{t-\Delta t}>0$) then $P_t<0$, i.e. a negative price, an inadmissible outcome.

The convention of mathematical finance instead adopts the **net logarithmic return**. Setting

$$P_t = P_{t-\Delta t}\,e^{R_{\ln,\Delta t}} - D_{(t-\Delta t,t]},$$

with the symbols defined as above and $R_{\ln,\Delta t}$ the net logarithmic return, one obtains

$$R_{\ln,\Delta t} = \ln\!\left(\frac{P_t + D_{(t-\Delta t,t]}}{P_{t-\Delta t}}\right).$$

Here, if $D_{(t-\Delta t,t]}=0$ and $P_{t-\Delta t}>0$, then $P_t>0$ for any value of $R_{\ln,\Delta t}$; if instead $R_{\ln,\Delta t} < \ln(D_{(t-\Delta t,t]}/P_{t-\Delta t})$ with $D_{(t-\Delta t,t]}>0$ and $P_{t-\Delta t}>0$, then $P_t<0$.

**Asymptotic equivalence.** If $R_{\%,\Delta t}\in(-1,1)=(-100\%,100\%)$, then $R_{\%,\Delta t}\simeq R_{\ln,\Delta t}$. Indeed,

$$\begin{aligned}
R_{\ln,\Delta t} &= \ln\!\left(\frac{P_t+D_{(t-\Delta t,t]}}{P_{t-\Delta t}}\right) \\
&= \ln\!\left(1+\frac{P_t+D_{(t-\Delta t,t]}-P_{t-\Delta t}}{P_{t-\Delta t}}\right) \\
&= \ln(1+R_{\%,\Delta t}),
\end{aligned}$$

and, recalling the Taylor series expansion $\ln(1+x)=\sum_{i=1}^{+\infty}(-1)^{i-1}\frac{x^i}{i}$ for $x\in(-1,1)$,

$$R_{\ln,\Delta t} = \ln(1+R_{\%,\Delta t}) = R_{\%,\Delta t} - \frac{R_{\%,\Delta t}^2}{2} + \frac{R_{\%,\Delta t}^3}{3} - \frac{R_{\%,\Delta t}^4}{4} + \dots$$

for $R_{\%,\Delta t}\in(-100\%,100\%)$. The condition $R_{\%,\Delta t}\in(-100\%,100\%)$ generally holds when $\Delta t$ is sufficiently small ($\Delta t$ equal to a day, a week, a month, …), so that the two first-order terms coincide and the higher-order terms are negligible.

**Non-additivity.** Despite the asymptotic equivalence, $R_{\%,\Delta t}$ is not additive over time, whereas $R_{\ln,\Delta t}$ is. Considering two consecutive periods with prices $P_0,P_1,P_2$, the percentage return does not enjoy the additive property,

$$R_{\%,(0,1]}+R_{\%,(1,2]} \;=\; \frac{P_1^2-2P_0P_1+P_0P_2}{P_0P_1} \;\neq\; R_{\%,(0,2]} = \frac{P_2-P_0}{P_0},$$

while the logarithmic return does,

$$R_{\ln,(0,1]}+R_{\ln,(1,2]} \;=\; \ln\!\left(\frac{P_1}{P_0}\right)+\ln\!\left(\frac{P_2}{P_1}\right) \;=\; \ln\!\left(\frac{P_2}{P_0}\right) = R_{\ln,(0,2]}.$$

A numerical illustration confirms the result: with prices $100,125,100$ the one-period percentage returns are $25.00\%$ and $-20.00\%$, whose sum ($5.00\%$) does not coincide with the two-period percentage return ($0.00\%$); the corresponding logarithmic returns are $22.31\%$ and $-22.31\%$, whose sum ($0.00\%$) does coincide with the two-period logarithmic return ($0.00\%$).

## Mean and Variance as Measures of Return and Risk

Portfolio selection under uncertainty unfolds in three steps: first, identify a tool with which to "measure" the uncertainty associated with a given investment choice; second, define an efficiency criterion with which to divide all possible investment choices into two mutually exclusive sets — an efficient set and an inefficient one [Szego1980]; third, specify an appropriate optimization approach to identify, among the efficient choices, the optimal one. In particular, optimization can take the forms: given an upper bound on risk, maximize return; given a lower bound on return, minimize risk; optimize a suitable synthesis index based on return and risk (for example maximize $\text{return}-\lambda\cdot\text{risk}$, with $\lambda>0$ a measure of risk aversion); maximize a von Neumann–Morgenstern utility function [vonNeumannMorgenstern1944]. An investment choice is efficient with respect to a dominance criterion when it is not dominated by any other choice in the sense of that criterion.

For the first step, a pair of statistical indices of the random variable identified by the single-period return is adopted as the stochastic tool: the **mean** and the **variance** of that return. The rule is that the investor considers — or should consider — expected return a desirable thing and the variance of return an undesirable thing [Markowitz1952]. The innovation introduced by Markowitz was to measure the risk of a portfolio through the joint (multivariate) distribution of the returns of all assets, describing the marginal properties through the first two moments of the univariate distributions and the dependence structure through Pearson's linear correlation coefficient between each pair of returns [Szego2005].

**Mean, $\mathbb{E}(R)=r$.** In general terms, the mean of a random variable is a statistical index of location; from a financial point of view, the mean of the return is taken as a measure of the profitability of the investment choice, and more generally any odd moment of the return can be regarded as a measure of profitability. For discrete $X$, $X=\{(x_1,p_1),\dots,(x_M,p_M)\}$ with $0\le p_i\le 1$ and $\sum_{i=1}^M p_i=1$,

$$\mathbb{E}(X) = \sum_{i=1}^{M} x_i p_i;$$

for continuous $X$ with cumulative distribution function $F_X(\cdot)$ and/or density $f_X(\cdot)$,

$$\mathbb{E}(X) = \int_{-\infty}^{+\infty} t\,dF(t) \quad \text{and/or} \quad \mathbb{E}(X) = \int_{-\infty}^{+\infty} t\, f(t)\,dt.$$

In the continuous case $\mathbb{E}(X)$ might not exist.

**Variance, $\mathbb{Var}(R)=\sigma^2$.** In general terms, variance is a statistical index of variability; from a financial point of view, the variance of the return is taken as a measure of risk, and more generally any even moment of the return can be regarded as a measure of risk. In the discrete case

$$\mathbb{Var}(X) = \sum_{i=1}^{M} (x_i-\mathbb{E}(X))^2 p_i,$$

and in the continuous case

$$\mathbb{Var}(X) = \int_{-\infty}^{+\infty} (t-\mathbb{E}(X))^2\,dF(t) \quad \text{and/or} \quad \mathbb{Var}(X) = \int_{-\infty}^{+\infty} (t-\mathbb{E}(X))^2 f(t)\,dt,$$

with $\mathbb{Var}(X)$ possibly not existing in the continuous case. Mean and variance, in general, do not fully characterize a random variable.

## A Limit of Variance: Semi-Variance and Mean Absolute Deviation

Variance is not, in general, a "good" measure of risk, and its intrinsic limit can be illustrated with an example. Let

$$R = \left\{\left(1\%,\tfrac14\right),\left(3\%,\tfrac15\right),\left(9\%,\tfrac14\right),\left(10\%,\tfrac15\right),\left(12\%,\tfrac{1}{10}\right)\right\}.$$

The mean is

$$\mathbb{E}(R) = 1\%\cdot\tfrac14 + 3\%\cdot\tfrac15 + 9\%\cdot\tfrac14 + 10\%\cdot\tfrac15 + 12\%\cdot\tfrac{1}{10} = 6.3\%,$$

and the variance

$$\mathbb{Var}(R) = (4.124318\%)^2 = 17.01\%^2.$$

The deviations of the realizations from the mean are $-5.3\%$, $-3.3\%$, $2.7\%$, $3.7\%$, $5.7\%$: the last three are positive. These positive deviations are not risky — the investment yields more than the expected value — but variance, by squaring all the deviations, penalizes them exactly like the negative ones.

A first alternative measure is **semi-variance**, which considers only the negative deviations. In the discrete case

$$\text{semi-}\mathbb{Var}(R) = \sum_{i=1}^{M} (\min\{0,\, x_i-\mathbb{E}(R)\})^2 p_i,$$

and in the continuous case

$$\text{semi-}\mathbb{Var}(R) = \int_{-\infty}^{+\infty} (\min\{0,t-\mathbb{E}(R)\})^2\,dF_R(t) \quad \text{and/or} \quad \text{semi-}\mathbb{Var}(R) = \int_{-\infty}^{+\infty} (\min\{0,t-\mathbb{E}(R)\})^2 f_R(t)\,dt.$$

In the example,

$$\text{semi-}\mathbb{Var}(R) = (-5.3\%)^2\tfrac14 + (-3.3\%)^2\tfrac15 = (3.03323\%)^2 = 9.2005\%^2 < 17.01\%^2 = \mathbb{Var}(R).$$

Measuring risk through semi-variance was proposed by Markowitz on the grounds that only downside risk is relevant to the investor [Markowitz1959].

A second alternative measure is the **mean absolute deviation** (MAD). In the discrete case

$$\text{MAD}(R) = \sum_{i=1}^{M} |x_i-\mathbb{E}(R)|\,p_i,$$

and in the continuous case

$$\text{MAD}(R) = \int_{-\infty}^{+\infty} |t-\mathbb{E}(R)|\,dF_R(t) \quad \text{and/or} \quad \text{MAD}(R) = \int_{-\infty}^{+\infty} |t-\mathbb{E}(R)|\,f_R(t)\,dt.$$

In the example,

$$\text{MAD}(R) = |{-5.3\%}|\tfrac14 + |{-3.3\%}|\tfrac15 + |2.7\%|\tfrac14 + |3.7\%|\tfrac15 + |5.7\%|\tfrac{1}{10} = 3.97\%.$$

MAD is not directly comparable with variance, but its square is:

$$\text{MAD}^2(R) = (3.97\%)^2 = 15.76\%^2 < 17.01\%^2 = \mathbb{Var}(R).$$

The adoption of mean absolute deviation as a risk function in the portfolio optimization problem is due to Konno and Yamazaki [KonnoYamazaki1991]. In any case, neither semi-variance nor MAD are better risk measures than variance: the latter enjoys analytical advantages that other variability measures lack [Bortot1993]. The properties that a risk measure should satisfy are taken up systematically in the chapter [[04 Misure di rischio coerenti]].

## The Mean-Variance Dominance Criterion

For the second step — the efficiency criterion with which to divide investment choices into an efficient set and an inefficient one [Szego1980] — a criterion based on the concepts of mean and variance is adopted.

**Mean-variance dominance criterion.** Let $X_1$ and $X_2$ be two random variables (for example portfolio returns). $X_1$ is said to **dominate** $X_2$, i.e. $X_1$ is preferred to $X_2$, in the sense of mean-variance dominance if

$$\mathbb{E}(X_1) \geq \mathbb{E}(X_2) \quad \text{and} \quad \mathbb{Var}(X_1) \leq \mathbb{Var}(X_2)$$

and at least one of the two inequalities holds strictly. We write $X_1\succ_{MV}X_2$ for dominance, $X_1\sim_{MV}X_2$ (or $X_1=_{MV}X_2$) for indifference, and $X_1\succeq_{MV}X_2$ when $X_1$ dominates or is indifferent to $X_2$.

The criterion induces only a **partial ordering**. Let $\mathbb{X}=\{X_1,X_2,X_3\}$ with

$$\mathbb{E}(R_1)=4,\ \mathbb{Var}(X_1)=3;\qquad \mathbb{E}(R_2)=2,\ \mathbb{Var}(X_2)=7;\qquad \mathbb{E}(R_3)=6,\ \mathbb{Var}(X_3)=5.$$

We have $X_1\succ_{MV}X_2$ and $X_2\prec_{MV}X_3$, but $X_1$ and $X_3$ are not comparable: $X_3$ has a higher mean than $X_1$ but also a higher variance, so neither dominates the other. Plotting the choices in the $(\mathbb{Var}(X),\mathbb{E}(X))$ plane and fixing a reference point, dominance is determined in the quadrants with higher mean and lower variance (or lower mean and higher variance), while it remains undetermined in the other two quadrants.

The most natural way, from a financial point of view, to apply the criterion is to induce an ordering on a given set of investment choices,

$$\mathbb{X}=\{X_1,\dots,X_N\} \ \xrightarrow{\ MV\ }\ X_i\succ_{MV}X_j,\ X_j \mathbin{?}_{MV} X_k,\ X_k\sim_{MV}X_k,\ \dots$$

Fixing a level of expected return $\bar r$, among all choices with that mean the criterion identifies the minimum-variance one as dominant: for a given expected return, the rational, risk-averse investor prefers the lower variance. Repeating the procedure for two levels $\bar r_1 > \bar r_2$ yields the minimum-variance choices $\mathbf{x}_1$ and $\mathbf{x}_2$ respectively; iterating it over all possible levels of expected return builds, point by point, the set of mean-variance efficient choices. This construction anticipates the efficient frontier [Markowitz1952], the subject of the chapter [[02 Selezione media-varianza]].

## Map of the Corpus

The monograph follows the three-step structure introduced above — measuring uncertainty, defining an efficiency criterion, optimizing — and extends it to the subsequent developments of the theory and to its operational applications.

- [[01 Scelta in condizioni di incertezza]] — formalization of the investor's preferences under uncertainty and of von Neumann–Morgenstern utility functions, the foundation of the optimization criterion.
- [[02 Selezione media-varianza]] — from the mean-variance dominance criterion to Markowitz's efficient frontier and its construction.
- [[03 Limiti della MPT e il modello di Black-Litterman]] — the sensitivity of the optimal composition to the initial inputs and the remedies based on views and robust estimation.
- [[04 Misure di rischio coerenti]] — the properties required of a risk measure, beyond variance, semi-variance, and mean absolute deviation.
- [[05 Modelli fattoriali]] — the description of returns through common factors.
- [[06 Vincoli e metaeuristiche]] — portfolio selection in the presence of operational constraints and the corresponding optimization algorithms.
- [[07 Revisione di portafoglio]] — portfolio rebalancing over time, with estimate variability, transaction costs, and intervention rules.

## References

- **[Bortot1993]** Bortot, P., Magnani, U., Olivieri, G., Rossi, F. A., & Torrigiani, M. (1993). Matematica Finanziaria. Bologna: Monduzzi Editore.
- **[ConstantinidesMalliaris1995]** Constantinides, G. M., & Malliaris, A. G. (1995). Portfolio Theory. In R. A. Jarrow, V. Maksimovic & W. T. Ziemba (Eds.), Finance (Handbooks in Operations Research and Management Science, vol. 9). Amsterdam: North-Holland.
- **[Ingersoll1987]** Ingersoll, J. E. jr. (1987). Theory of Financial Decision Making. Totowa (NJ): Rowman & Littlefield.
- **[KonnoYamazaki1991]** Konno, H., & Yamazaki, H. (1991). Mean-Absolute Deviation Portfolio Optimization Model and Its Applications to Tokyo Stock Market. Management Science, 37(5), 519–531.
- **[Markowitz1952]** Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77–91.
- **[Markowitz1959]** Markowitz, H. (1959). Portfolio Selection: Efficient Diversification of Investments. New York: John Wiley & Sons.
- **[Szego1980]** Szegő, G. P. (1980). Portfolio Theory. With Application to Bank Asset Management. New York: Academic Press.
- **[Szego2005]** Szegő, G. (2005). Measures of risk. European Journal of Operational Research, 163(1), 5–19.
- **[vonNeumannMorgenstern1944]** von Neumann, J., & Morgenstern, O. (1944). Theory of Games and Economic Behavior. Princeton (NJ): Princeton University Press.
