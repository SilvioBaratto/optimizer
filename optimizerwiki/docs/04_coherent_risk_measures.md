---
title: "Coherent Risk Measures"
chapter: 4
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-10
---

> [!abstract] Summary
> The chapter defines Value-at-Risk, establishes its properties and limitations — notably the fact that it is not in general subadditive and therefore not coherent — and presents the RiskMetrics Group approach to its computation. It then introduces coherent measures built on the tail of the distribution (Average Value-at-Risk/Conditional VaR/Expected Shortfall, Tail Conditional Expectation, Worst Conditional Expectation) and the two-sided class, showing their definitions, equivalent formulas, and properties. It closes with stable Lévy-Pareto distributions, which provide the distributional framework within which variance ceases to be a reliable risk measure and tail measures gain significance.

## Value-at-Risk: Formal Definition and Graphical Interpretation

**Value-at-Risk (VaR)** is a risk measure that has been widely used since the 1990s. In the middle of that decade it was approved by the Basel Committee as a method for calculating the capital reserve needed to cover market risks [Basel1996]. Qualitatively, VaR is defined as the *minimum* level of loss at a given confidence level, over a predefined time horizon (known as the *holding period*); the recommended confidence levels are $95\%$ and $99\%$. If, for example, an equity portfolio has a daily VaR at $5\%$ equal to €$50{,}000$, then there is a $5\%$ probability that the value of the portfolio will lose at least €$50{,}000$ from one day to the next; informally, this implies that the portfolio loses that amount one day out of twenty (one being $5\%$ of twenty).

Let $X$ be the risky payoff and $(1-\epsilon)$ the confidence level. Formally, VaR is defined as

$$VaR_\epsilon(X) = -\inf_x \{x \mid P(X \le x) \ge \epsilon\}.$$

An equivalent formulation, which favors the tail probability $\alpha$, identifies VaR as the quantile of the return distribution that leaves that probability to its left:

$$VaR_\alpha = \inf\{x \mid Pr(X>x) = \alpha\}.$$

Graphically, VaR is thus the quantile of the profit/loss distribution corresponding to the chosen tail probability: on the profit-and-loss density function it identifies the point beyond which (to the left) the probability $\alpha$ of the worst losses is concentrated; equivalently, on the cumulative distribution function, the horizontal line at the height of the tail probability intersects the curve exactly at the VaR value.

VaR has two notable properties. If $C$ is a riskless payoff, then

$$VaR_\epsilon(X + C) = VaR_\epsilon(X) - C;$$

if $\lambda$ is a positive constant, then

$$VaR_\epsilon(\lambda X) = \lambda \, VaR_\epsilon(X).$$

Neither property holds for variance, which therefore responds in a qualitatively different way to the addition of a certain component and to rescaling of the payoff.

## The Probabilistic Definition of VaR and Its Critique

From a probabilistic point of view, VaR at confidence level $\alpha$ is the value that satisfies the equality

$$P(L > VaR_\alpha) = 1-\alpha,$$

where $L$ denotes a generic loss distribution.

It is precisely this framing that reveals its main limitation. VaR is a measure that is *indifferent to how large the losses beyond the threshold actually are*: it says nothing about the size of the loss that exceeds the indicated level, so the actual loss can turn out to be greater than the VaR. It does not take much imagination to identify portfolios with the same $VaR_\alpha$ but with dramatically different loss levels in the worst $(1-\alpha)\%$ of cases. VaR identifies a threshold, not the severity of what lies beyond it.

To this is added a structural flaw. Denoting by $X$ and $Y$ two risky payoffs, it can happen that

$$VaR_\epsilon(X + Y) > VaR_\epsilon(X) + VaR_\epsilon(Y),$$

that is, the risk of the combined portfolio can turn out to be greater than the sum of the individual risks, contradicting the reasonable effect of diversification. In general, therefore, VaR is not subadditive. Since subadditivity is one of the properties required of a *coherent* risk measure [Artzner1999], VaR is not a coherent risk measure; one of the few cases in which subadditivity is satisfied is when the joint distribution of returns is Gaussian. This observation motivates the introduction of risk measures based on the tail of the distribution, treated in the following sections.

## Computing VaR: the RiskMetrics Group Approach

The computation of VaR — for either a single asset or a portfolio — can proceed along two routes: the RiskMetrics Group approach and the historical method [RiskMetrics1996].

The RiskMetrics Group approach rests on the assumption that stock returns follow a multivariate normal distribution. For a single asset, the standard normal distribution of the return is considered; the notable quantiles used are

$$z_{0.01} = -2.3263, \qquad z_{0.05} = -1.6449,$$

and VaR is obtained as

$$VaR_\epsilon = z_\epsilon \cdot \sigma_{return} + \mu_{return},$$

where $\sigma_{return}$ and $\mu_{return}$ are, respectively, the standard deviation and the mean of the asset's return. In other words, the quantiles of the standard normal, multiplied by the standard deviation of the return and shifted by its mean, directly provide VaR at the $99\%$ and $95\%$ confidence levels. The VaR of a portfolio is obtained in the same way, applying the same formula to the mean and standard deviation of the portfolio's return.

Alongside this parametric approach, the historical method derives VaR directly from the empirical realizations of returns, without assuming a distributional form.

## Average Value-at-Risk, Conditional VaR, and Expected Shortfall

The **Average Value-at-Risk (AVaR)**, also called **Conditional Value-at-Risk (CVaR)** or **Expected Shortfall (ES)**, is a coherent risk measure, free of the shortcomings of VaR and endowed with an intuitive interpretation. Its formal definition is the average of the VaRs computed for all tail levels below the chosen one:

$$AVaR_\epsilon(X) := \frac{1}{\epsilon} \int_0^\epsilon VaR_p(X)\, dp.$$

The contrast with VaR can be interpreted as two distinct questions. VaR answers *"how often might my portfolio lose at least a certain amount?"*, that is, how frequently the threshold is exceeded; CVaR instead answers *"when my portfolio loses more than that amount, how much might it lose?"*, that is, how much is lost on average in the tail beyond the VaR threshold. On the loss distribution, CVaR sits further into the tail than VaR — between the VaR threshold and the maximum observed loss — and summarizes its severity in a more conservative and informative way.

An equivalent characterization is obtained starting from the cumulative distribution function. In probability calculus, the cumulative distribution function of a random variable $X$ is the function $F:\mathbb{R}\to[0,1]$ defined by $F(x)=P(X\le x)$, non-decreasing, right-continuous, with $\lim_{x\to-\infty}F(x)=0$ and $\lim_{x\to+\infty}F(x)=1$; it is not necessarily left-continuous, since at a point $z$ in the support of a discrete variable $F(z)=\sum_{i}p(x_i)+p(z)$ with $p(z)\neq 0$, and hence a jump. Introducing the inverse function

$$F^{-1}(\alpha) = \inf\{x \mid F(x) \ge \alpha\},$$

Expected Shortfall is defined, given a holding period and a significance level $\alpha\in[0;1]$, as

$$ES_\alpha(X) := -\frac{1}{\alpha}\Big(E\big[X\,\mathbb{1}_{\{X \le x^{(\alpha)}\}}\big] - x^{(\alpha)}\big(P[X \le x^{(\alpha)}] - \alpha\big)\Big),$$

where $x^{(\alpha)}$ coincides with $VaR_\alpha$. It can be shown [AcerbiTasche2002a] that ES admits the equivalent form

$$ES_\alpha(X) = -\frac{1}{\alpha}\int_0^\alpha F^{-1}(p)\, dp,$$

which exhibits the structure of an average of tail quantiles already encountered in the definition of AVaR. Consistent with its nature, ES sits graphically further to the left of VaR in the tail of the distribution, summarizing the average severity of losses rather than merely capturing the threshold.

For continuous random variables — as is generally assumed for financial returns — ES coincides with CVaR. Empirically, ES is obtained by ordering the $n$ possible realizations and, given a significance level, selecting the $(1-\alpha)\%$ of the largest losses:

$$ES_\alpha(X) = -\frac{\sum_{i=1}^{w} X_{i:n}}{w}, \qquad w = \max\{m \mid m \le n(1-\alpha),\ m \in \mathbb{N}\},$$

where $w$ is the integer part of $n\times(1-\alpha)\%$; the practical computation of AVaR is also performed this way, using the historical method.

ES is a *universal* risk measure, applicable to every financial instrument and every underlying source of risk. It also enjoys *simplicity* and *completeness*, since it produces a single number even for portfolios exposed to different sources of risk, and *robustness*: unlike other tail-based measures, its use ensures a certain convergence of results even when the confidence level is varied by a few basis points — a property not guaranteed by VaR, TCE, and WCE.

## Tail Conditional Expectation, Worst Conditional Expectation, and Their Relation to CVaR

Two further risk measures, defined on the left tail of the return distribution, are the Tail Conditional Expectation and the Worst Conditional Expectation. The **Tail Conditional Expectation** (also known as *TailVaR*) is defined as

$$TCE_\alpha(X) \stackrel{def}{=} -E[X \mid X \le -VaR_\alpha(X)],$$

where $X$ denotes portfolio performance. The **Worst Conditional Expectation** is defined as

$$WCE_\alpha(X) \stackrel{def}{=} -\inf\{E[X \mid A] \mid P[A] > \alpha\},$$

where $A$ denotes unfavorable events or scenarios. Financially, both take into account "how bad is bad": they focus on the shape of the left tail — where the losses reside — and take its average value conditional on losses exceeding a certain value.

For continuous random variables, as is generally assumed for financial returns, TCE coincides with the Conditional Value-at-Risk

$$\frac{1}{\alpha}\int_0^\alpha VaR_\gamma(X)\, d\gamma,$$

thereby recovering the same average of tail quantiles that defines AVaR and ES. Between the two measures the ordering

$$TCE_\alpha \le WCE_\alpha$$

holds. Their status with respect to coherence is, however, asymmetric [AcerbiTasche2002b]: WCE fully satisfies the coherence axioms, but it is used almost exclusively in theoretical settings, since it requires knowledge of the entire underlying probability space; TCE is more tractable even in applied settings, but does not fully satisfy the coherence axioms, since it is not always subadditive.

## Two-Sided Coherent Risk Measures

The measures considered so far act on only one side of the distribution, the loss tail. A different construction instead combines both sides [ChenWang2008]. The convex combination of the $1$-norm of the *upside* of $X$ and the $p$-norm of the *downside* of $X$ leads to a new coherent risk measure:

$$\rho_{a,p}(X) \stackrel{def}{=} a\,\sigma_1^+(X) + (1-a)\,\sigma_p^-(X) - E[X] = a\left\|(X-E[X])^+\right\|_1 + (1-a)\left\|(X-E[X])^-\right\|_p - E[X],$$

where the $p$-norm of a vector $\mathbf{x}$ is

$$\|\mathbf{x}\|_p := \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}.$$

Financially, the measure is a linear combination of both positive and negative moments of the return distribution, with weights equal to $a$ and $1-a$ respectively. The $1$-norm takes into account returns above the expected return; the $p$-norm refers to the "adjusted" returns below it, that is, a centered moment of the distribution influenced by the parameter $p$, which reflects the degree of penetration of the distributional analysis and is closely tied to the investor's degree of risk aversion.

The two parameters model the investor's risk aversion:

- $a \in [0;1]$ is a *global* risk factor, reflecting the desired balance between upside and downside volatility;
- $p \in [1;+\infty[$ is a *local* risk factor, which grows proportionally with the investor's risk aversion and also incorporates information related to the characteristics of the return distribution, such as skewness and kurtosis.

## Stable Lévy-Pareto Distributions: Motivation

The underlying question linking risk measures to distributional shape concerns which law the logarithmic return

$$r_{t,1} = \ln\!\left(\frac{P_t}{P_{t-1}}\right)$$

follows. The starting empirical evidence — a histogram of daily returns heavily concentrated around zero, with a very high central peak and thin tails extending to extreme values — motivates a comparison between two hypotheses: the normal distribution and the stable Lévy-Pareto distribution. The salient differences are summarized in the following table:

| Normal distribution | Stable Lévy-Pareto distribution |
|---|---|
| Finite variance | Infinite variance |
| "Normal" tails | Fat tails |
| Lower risk | Higher risk |

The chronology of the theoretical developments leading to this contrast begins with Pareto's income distribution law, $Pr\{Y>y\}=y^{-\alpha}$ with $\alpha=1.7$ and infinite variance of $Y$ [Pareto1897]; continues with the birth of mathematical finance and the hypothesis that $r_{t,1}$ is normally distributed [Bachelier1900]; with the introduction of stable random variables, for which $\alpha\in(0,2]$ and the variance is infinite [Levy1925]; and culminates in the use of the logarithmic return as a stable random variable, with infinite variance and fat tails [Mandelbrot1963a][Mandelbrot1963b][Fama1963].

## Stable Random Variables: Definitions via Convolution

The following definitions of a stable random variable are equivalent to one another: choosing one of them as the "starting point," the others follow as theorems. The first three adopt a convolution — that is, summation — approach.

**Definition 1.** A random variable $X$ is *stable* if for every pair of constants $a,b>0$ there exist $c>0$ and $d\in\mathbb{R}$ such that

$$aX_1+bX_2 =_d cX+d,$$

where $X_1, X_2, X$ are independent and identically distributed and $=_d$ denotes equality in distribution. Intuitively: the "sum" of two stable random variables is still a stable random variable.

**Definition 2.** A random variable $X$ is *stable* if for every integer $n\ge 2$ there exist sequences of constants $a_1,\dots,a_n>0$ and $b_1,\dots,b_n\in\mathbb{R}$ such that

$$X_1+X_2+\cdots+X_n =_d a_nX+b_n,$$

with $X_1,\dots,X_n,X$ independent and identically distributed: the "sum" of $n$ stable random variables is still stable.

**Definition 3.** A random variable $X$ is *stable* if it has a domain of attraction, that is, if there exist independent and identically distributed random variables $X_1,X_2,\dots$ and sequences of constants $c_n>0$, $d_n\in\mathbb{R}$ such that

$$\frac{X_1+X_2+\cdots+X_n}{c_n}+d_n \to_d X \quad \text{as } n\to+\infty,$$

where $\to_d$ denotes convergence in distribution: the "sum" of a sufficiently large number of normalized random variables — even non-stable ones — is a stable random variable.

Definitions 1 and 2 are "siblings"; Definition 3 is a "cousin" of the first two; and normal random variables satisfy all three definitions.

These definitions have an immediate financial reading. The weekly return decomposes, by telescoping the price ratios, into the sum of daily returns:

$$r_{t,5} = \ln(P_t/P_{t-5}) = r_{t,1}+r_{t-1,1}+r_{t-2,1}+r_{t-3,1}+r_{t-4,1};$$

if daily returns are stable random variables, then their sum — the weekly return — is a stable random variable (Definitions 1 and 2). Similarly, the annual return decomposes into the sum of daily returns over roughly 250 trading days,

$$\ln(P_t/P_{t-250}) = r_{t,1}+r_{t-1,1}+\cdots+r_{t-249,1};$$

if daily returns are random variables that are not even stable, the sum of a sufficiently large number of them — the annual return — is well approximated by a stable random variable (Definition 3).

## Stable Random Variables: The Characteristic Function and Parameters

The fourth definition, equivalent to the previous ones, characterizes the stable class through the characteristic function.

**Definition 4.** A random variable $X$ is *stable* if there exist four parameters $\alpha\in(0,2]$, $\beta\in[-1,1]$, $\mu\in\mathbb{R}$, and $\sigma\in[0,+\infty)$ such that the characteristic function of $X$ is

$$E\!\left(e^{i\vartheta X}\right) = \begin{cases} \exp\!\left\{-\sigma^{\alpha}|\vartheta|^{\alpha}\left(1-i\,\beta\,sgn(\vartheta)\,tg\!\left(\dfrac{\alpha\pi}{2}\right)\right)+i\,\mu\,\vartheta\right\} & \alpha\neq1,\\[2mm] \exp\!\left\{-\sigma|\vartheta|\left(1-i\,\beta\,\dfrac{2}{\pi}\,sgn(\vartheta)\,\log(|\vartheta|)\right)+i\,\mu\,\vartheta\right\} & \alpha=1, \end{cases}$$

with $i=\sqrt{-1}$. This is written $X\sim S(\alpha,\beta,\mu,\sigma)$. The characteristic function can be specified in closed form for every random variable, and it serves both to specify the density function in closed form and to compute the moments; however, for a generic stable distribution it is almost never possible to specify the probability density function in closed form, which is available only in a few cases. Two of these are notable: for $\alpha=1$ and $\beta=0$ the distribution $S(1,0,\mu,\sigma)$ specializes to the **Cauchy distribution**,

$$f(x) = \frac{\sigma}{\pi[\sigma^{2}+(x-\mu)^{2}]},$$

and for $\alpha=2$ the distribution $S(2,\ast,\mu,\sigma)$ specializes to the **normal distribution** $N(\mu, 2\sigma^{2})$.

Each of the four parameters has a precise role.

**$\alpha\in(0,2]$ — characteristic exponent (stability index).** It is linked to kurtosis and provides a measure of the area, that is, the probability, subtended by the tails. As $\alpha$ decreases, the probability of extreme events (fat tails) increases: the density becomes more peaked at the center and with thicker tails, whereas at $\alpha=2$ — the normal — the probability of extreme events is lowest. Two implications follow: if $\alpha<2$ the variance is infinite, so the only stable distribution with finite variance is the normal ($\alpha=2$); if $\alpha<1$ the mean is infinite as well. In financial terms this does not mean that risk is infinite, but simply that variance is not a "good" measure of risk.

**$\mu\in\mathbb{R}$ — location parameter.** It is linked to the position of the distribution; if $\alpha\in[1,2]$, $\mu$ coincides with the mean value. The following theorem holds: if $X\sim S(\alpha,\beta,\mu,\sigma)$ and $k\in\mathbb{R}$, then $X+k \sim S(\alpha,\beta,\mu+k,\sigma)$; adding a constant shifts the distribution, and $\mu$ behaves like a mean.

**$\beta\in[-1,1]$ — skewness parameter.** It governs symmetry/asymmetry with respect to $\mu$: $\beta=0$ gives symmetry, $\beta<0$ left skewness, and $\beta>0$ right skewness; in particular $\beta=-1$ indicates complete left skewness and $\beta=1$ complete right skewness.

**$\sigma\in[0,+\infty)$ — scale parameter.** It is linked to dispersion. The following theorem holds: if $X\sim S(\alpha,\beta,\mu,\sigma)$ and $k\in\mathbb{R}$, then

$$k\cdot X \sim \begin{cases} S(\alpha,\, sgn(k)\,\beta,\, k\mu,\, |k|\sigma) & \alpha\neq1,\\[1mm] S\!\left(1,\, sgn(k)\,\beta,\, k\mu-\dfrac{2}{\pi}k\log(k)\,\sigma\beta,\, |k|\sigma\right) & \alpha=1; \end{cases}$$

multiplying by a constant returns a stable random variable with the same characteristic exponent but, in general, a different "shape." The parameter $\sigma$ behaves like a standard deviation and can therefore be used as a measure of risk.

## The Normality Hypothesis for Returns: A Historical Digression

Where does the original idea that $r_{t,1}=\ln(P_t/P_{t-1})$ follows a normal distribution come from? A first answer — reasonable but incorrect — traces it back to empirical observation: the "bell-shaped" form of return histograms, concentrated around zero, would seem to suggest normality. The correct answer, though less intuitive, is that the normality hypothesis arises from a well-grounded theoretical-methodological framework on the behavior of financial markets [Bachelier1900]. In it, the speculator has zero conditional expectation — the market prices assets according to a martingale measure — and the price evolves as a continuous Markov process, homogeneous in time and space; Bachelier showed that the density of the one-dimensional distributions satisfies the equation now known as the Chapman-Kolmogorov equation, and observed that the Gaussian density, with linearly increasing variance, solves it, obtaining the same law also as the limit of random walks [Courtault2000]. Bachelier focused on the Gaussian density, and not on more general density functions such as the stable Lévy-Pareto ones, because the latter did not yet exist: one would have to wait until 1925 for the introduction of stable random variables [Levy1925] and until 1963 for their use in finance [Mandelbrot1963a][Mandelbrot1963b][Fama1963].

The non-Gaussian approach makes two assertions: that the variances of empirical distributions behave as if they were infinite, and that empirical distributions conform better to the non-Gaussian members of the stable family; if the population variance of the first differences is infinite, the sample variance is probably a meaningless measure of dispersion [Fama1963]. Hence the problem: if the variance of logarithmic returns were infinite, what would become of Markowitz-style portfolio selection models [Markowitz1952] (cf. [[02 Selezione media-varianza]]), of risk-adjusted performance measures of the Sharpe ratio type, of Black-Scholes-and-Merton-style option pricing models?

It is important to distinguish the Paretian stable *distribution* from the Paretian stable *hypothesis* [Fama1963]: under both hypotheses — Gaussian and Paretian stable — it is assumed that the underlying distribution is stable; the conflict concerns the value of the characteristic exponent $\alpha$, which the Gaussian hypothesis sets equal to $2$ and the Paretian stable hypothesis sets strictly less than $2$. The comparison between the two frameworks is summarized as follows:

| | Bachelier-style framework | Lévy-Mandelbrot-Fama variant |
|---|---|---|
| Distributions | Stable | Stable |
| $\alpha$ | $2$ | $(0,2]$ |
| Risk | Lower | Higher |
| Risk measure | Volatility | ? |

This leaves open the question — which motivates empirical investigation — of which risk measure to adopt in the non-Gaussian variant, given that volatility loses its meaning when $\alpha<2$.

## Empirical Evidence on Equity Returns

Estimating the four parameters $\alpha,\beta,\mu,\sigma$ is not straightforward: several methods exist, all numerical, developed mainly between 1971 and 2001. The results reported here are based on an iterative regression-type estimation method [Koutrouvelis1980][Koutrouvelis1981], applied to the daily returns of six COMIT sector indices (Banking, Financial, Insurance, Communications, Real Estate, Industrial; 1984-1992, approximately 2000 observations each) and of four Italian equities (Ansaldo, Benetton, FIAT, FIAT preferred shares; sample sizes ranging from about 1750 to about 3500).

The estimates for the COMIT sector indices are:

| Index | $\alpha$ | $\beta$ | $\mu$ | $\sigma$ |
|---|---|---|---|---|
| Banking | 1.729 | 0.122 | 0.001 | 0.007 |
| Financial | 1.656 | 0.065 | 0.001 | 0.007 |
| Insurance | 1.659 | 0.151 | 0.002 | 0.008 |
| Communications | 1.631 | 0.037 | 0.000 | 0.007 |
| Real Estate | 1.680 | 0.179 | 0.001 | 0.005 |
| Industrial | 1.672 | 0.117 | 0.001 | 0.007 |

and those for the individual stocks:

| Stock | $\alpha$ | $\beta$ | $\mu$ | $\sigma$ |
|---|---|---|---|---|
| Ansaldo | 1.504 | 0.059 | 0.001 | 0.007 |
| Benetton | 1.562 | 0.063 | 0.001 | 0.008 |
| FIAT | 1.719 | 0.368 | 0.003 | 0.011 |
| FIAT preferred | 1.675 | 0.095 | 0.001 | 0.011 |

These results are considered representative of the distributions of equity returns: in general, returns are stably distributed; they are rarely normally distributed ($\alpha\neq 2$); and typically $\alpha\in(1,2)$, which corresponds to a finite mean of returns and an infinite variance of returns. The same picture emerges from a graphical comparison between the empirical histogram of log-returns and estimated density curves, in which the non-Gaussian stable distribution reproduces both the higher central peak and the heavier tails better than the normal [Rachev2003].

In finance, stable Lévy-Pareto distributions are used, among other things, in asset liability management, in credit risk models, in risk management, in portfolio selection, in models for the term structure of interest rates, and in option pricing; there are also connections with the fractal structure of financial returns, and research on these topics remains active.

## References

- **[AcerbiTasche2002a]** Acerbi, C., Tasche, D. (2002). «On the coherence of Expected Shortfall». Journal of Banking & Finance, 26(7), 1487-1503.
- **[AcerbiTasche2002b]** Acerbi, C., Tasche, D. (2002). «Expected Shortfall: a natural coherent alternative to Value at Risk». Economic Notes, 31(2), 379-388.
- **[Artzner1999]** Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999). «Coherent Measures of Risk». Mathematical Finance, 9(3), 203-228.
- **[Bachelier1900]** Bachelier, L. J.-B. A. (1900). «Théorie de la spéculation». Annales Scientifiques de l'École Normale Supérieure, 3, 21-86.
- **[Basel1996]** Basle Committee on Banking Supervision (1996). Amendment to the Capital Accord to Incorporate Market Risks. Basel: Bank for International Settlements.
- **[ChenWang2008]** Chen, Z., Wang, Y. (2008). «Two-sided coherent risk measures and their application in realistic portfolio optimization». Journal of Banking & Finance, 32(12), 2667-2673.
- **[Courtault2000]** Courtault, J.-M., Kabanov, Y., Bru, B., Crépel, P., Lebon, I., Le Marchand, A. (2000). «Louis Bachelier. On the centenary of the Théorie de la spéculation». Mathematical Finance, 10, 341-353.
- **[Fama1963]** Fama, E. F. (1963). «Mandelbrot and the stable Paretian hypothesis». The Journal of Business, 36, 420-429.
- **[Koutrouvelis1980]** Koutrouvelis, I. A. (1980). «Regression-type estimation of the parameters of stable laws». Journal of the American Statistical Association, 75, 918-928.
- **[Koutrouvelis1981]** Koutrouvelis, I. A. (1981). «An iterative procedure for the estimation of the parameters of stable laws». Communications in Statistics. Simulation and Computation, 10, 17-28.
- **[Levy1925]** Lévy, P. (1925). Calcul des probabilités. Paris: Gauthier-Villars.
- **[Mandelbrot1963a]** Mandelbrot, B. (1963). «New methods in statistical economics». The Journal of Political Economy, 71, 421-440.
- **[Mandelbrot1963b]** Mandelbrot, B. (1963). «The variation of certain speculative prices». The Journal of Business, 36, 394-419.
- **[Markowitz1952]** Markowitz, H. (1952). «Portfolio Selection». The Journal of Finance, 7(1), 77-91.
- **[Pareto1897]** Pareto, V. (1897). Cours d'économie politique, vol. II. Lausanne: F. Rouge.
- **[Rachev2003]** Rachev, S. T. (Ed.) (2003). Handbook of Heavy Tailed Distributions in Finance. Amsterdam: Elsevier/North-Holland.
- **[RiskMetrics1996]** J.P. Morgan / Reuters (1996). RiskMetrics — Technical Document, 4th ed. New York: Morgan Guaranty Trust Company (RiskMetrics Group).
