---
title: "Dynamic Covariance Estimation - EWMA and Multivariate GARCH"
chapter: 18
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-11
---

> [!abstract] Summary
> The chapter establishes that the covariance matrix of returns is not constant over time and derives two classes of conditional estimators: RiskMetrics' exponentially weighted moving average (EWMA), with the choice of the decay factor and its link to the IGARCH model, and Engle's Dynamic Conditional Correlation (DCC), which decomposes covariance into univariate GARCH variances and a separate dynamic of conditional correlations estimable in two steps. It shows the correlation update rule, its reconditioning to a valid matrix, the consistency of the two-step estimation, and the empirical properties, and establishes that a time-varying covariance constitutes an alternative input to the static sample covariance for portfolio optimization.

## Heteroskedasticity and Clustering: Why Covariance Is Time-Varying

Correlations and volatilities are critical inputs for many of the fundamental tasks of financial management. A hedging operation requires estimating the correlation between the returns of the assets involved: if correlations and volatilities change, the hedge ratio must be adjusted to incorporate the most recent information. Asset allocation and risk measurement are also based on correlations, but in this case a large number of them is typically required: constructing an optimal portfolio subject to a set of constraints requires a forecast of the covariance matrix of returns, and computing today's portfolio standard deviation requires the covariance matrix of all the assets that compose it [Engle2002]. These tasks involve estimating and forecasting potentially very large covariance matrices, with thousands of assets.

The empirical evidence motivating a dynamic treatment is conditional heteroskedasticity, that is, the fact that the variance of returns depends on time, and volatility clustering. Research in finance and econometrics shows that realizations of return time series often exhibit time-dependent volatility; this idea was first formalized in the ARCH (Auto Regressive Conditional Heteroscedasticity) model, based on the specification of conditional densities in successive periods with a time-varying volatility process [Engle1982]. The fact that the forecast of one period's variance depends on the previous period's variance is consistent with the autocorrelation observed in squared returns [RiskMetrics1996]. Descriptively, volatility reacts to market shocks — after a large-magnitude return it rises — and subsequently declines gradually as the shock moves further back in the sample: episodes of high and low volatility tend to cluster together. Correlations between assets show the same non-stationary character, with sometimes marked structural changes in the correlation process, as in the breakdown of ties among European currencies in August 1992 [Engle2002].

It is these regularities that make a static sample covariance matrix inadequate and require *conditional* covariance estimators, updated with the most recent information. The chapter presents two classes of such estimators: RiskMetrics' EWMA and Engle's DCC model.

## RiskMetrics' Exponentially Weighted Moving Average (EWMA)

One way to capture the dynamic characteristics of volatility is to use an exponential moving average of historical observations, in which the most recent observations receive the greatest weight [RiskMetrics1996]. For a set of $T$ returns, the equally weighted estimator (simple moving average, SMA) is contrasted with the exponentially weighted one (exponentially weighted moving average, EWMA):

$$\sigma = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(r_t-\bar r)^2}, \qquad \sigma = \sqrt{(1-\lambda)\sum_{t=1}^{T}\lambda^{t-1}(r_t-\bar r)^2}.$$

The exponential scheme depends on the parameter $\lambda$, with $0<\lambda<1$, often called the *decay factor*: it determines the relative weights applied to the observations and the effective amount of data used in the estimate. The factor $(1-\lambda)$ normalizes the weights exploiting the approximation

$$\sum_{j=1}^{T}\lambda^{j-1}\approx\frac{1}{1-\lambda},$$

valid in the limit as $T\to\infty$. Compared with the SMA, the EWMA has two advantages: volatility reacts more quickly to shocks, because recent data weigh more; and after a shock, volatility declines exponentially as the weight of the anomalous observation shrinks, instead of changing abruptly when the shock exits the sample window — something that in the SMA can happen months after the event [RiskMetrics1996].

An attractive property of the exponential estimator is that, assuming an infinite amount of data and zero sample mean, it can be rewritten in recursive form, which becomes the basis for forecasting. Setting the mean to zero, the variance forecast for period $t+1$ given information through $t$ is

$$\sigma^2_{t+1|t}=\lambda\,\sigma^2_{t|t-1}+(1-\lambda)\,r^2_{t}, \tag{EWMA}$$

and RiskMetrics' one-day volatility forecast is $\sigma_{t+1|t}=\sqrt{\lambda\,\sigma^2_{t|t-1}+(1-\lambda)\,r^2_{t}}$. The subscript $t+1|t$ reads "forecast at time $t+1$ given information through $t$" and underscores the time-dependent character of the variance. The recursive form is obtained from the weighted sum of past squares:

$$\sigma^2_{t+1|t}=(1-\lambda)\sum_{i=0}^{\infty}\lambda^i r^2_{t-i}=(1-\lambda)\left(r^2_t+\lambda r^2_{t-1}+\lambda^2 r^2_{t-2}+\dots\right)=\lambda\,\sigma^2_{t|t-1}+(1-\lambda)\,r^2_t.$$

Conditional covariance and correlation are built in the same way, except that instead of the square of one series one works with the product of two series [RiskMetrics1996]. For two returns $r_{1,t}$ and $r_{2,t}$, the recursive covariance forecast is

$$\sigma^2_{12,t+1|t}=\lambda\,\sigma^2_{12,t|t-1}+(1-\lambda)\,r_{1,t}\,r_{2,t},$$

derivable with the same geometric expansion applied to the cross products. The conditional correlation is the covariance divided by the product of the standard deviations:

$$\rho_{12,t+1|t}=\frac{\sigma^2_{12,t+1|t}}{\sigma_{1,t+1|t}\,\sigma_{2,t+1|t}}.$$

Applied to the entire matrix, this scheme produces a conditional covariance as an exponentially weighted sum of the squares and products of past returns, updated at each observation with a single recursive formula.

## Choosing the Decay Factor

The decay factor governs both the weights and the effective number of observations used. The latter can be quantified: defining the metric $\Omega^{*}_K=(1-\lambda)\sum_{t=K}^{\infty}\lambda^{t}$ and setting it equal to a tolerance level $\Upsilon_L$, the effective number of days of data $K$ used by the EWMA solves

$$\lambda^{K}(1-\lambda)\left(1+\lambda+\lambda^2+\dots\right)=\Upsilon_L \;\Longrightarrow\; K=\frac{\ln\Upsilon_L}{\ln\lambda}.$$

For a given tolerance, the closer $\lambda$ is to one, the larger the number of effective observations; with $\lambda=0{,}97$ and a tolerance of $1\%$ the EWMA uses about $151$ days of data [RiskMetrics1996]. A higher $\lambda$ produces more stable — but not necessarily more accurate — forecasts, because it exploits more observations.

The selection of the optimal $\lambda$ rests on a statistical criterion. Since $E_t[r^2_{t+1}]=\sigma^2_{t+1|t}$, the variance forecast error $\varepsilon_{t+1|t}=r^2_{t+1}-\sigma^2_{t+1|t}$ has zero expected value; it is therefore natural to choose $\lambda$ by minimizing the mean squared forecast error, that is, the root mean squared error (RMSE):

$$\mathrm{RMSE}_v=\sqrt{\frac{1}{T}\sum_{t=1}^{T}\left(r^2_{t+1}-\hat\sigma^2_{t+1|t}(\lambda)\right)^2},$$

where the forecast is written explicitly as a function of $\lambda$. The optimal factor $\lambda^{*}$ is the one that minimizes the RMSE over a grid of values; an analogous expression holds for covariance [RiskMetrics1996].

A crucial constraint in the multivariate case is that the decay factors are not independent of one another. The covariance matrix to which they belong must possess certain properties: variances cannot be negative, covariances must be symmetric, and every correlation must lie in $[-1,1]$. Since, writing each element as a function of its own $\lambda$, a bivariate matrix would depend on three distinct factors, the $\lambda$'s must be chosen consistently with the matrix to which they belong; it is possible to construct a positive semi-definite matrix with different factors, but it is subject to substantial bias. For this reason RiskMetrics applies a *single* decay factor to the entire covariance matrix. The value adopted is obtained as a weighted average of the individual optimal factors estimated over more than $450$ series, with weights

$$\phi_i=\theta_i^{-1}\Big/\sum_{i=1}^{N}\theta_i^{-1},\qquad \theta_i=\tau_i\Big/\sum_{i=1}^{N}\tau_i,\qquad \bar\lambda=\sum_{i=1}^{N}\phi_i\hat\lambda_i,$$

where $\tau_i$ is the minimum RMSE of the $i$-th series: the weights are thus a measure of individual forecast accuracy. Applying this procedure yields a factor of $0{,}94$ for daily data and $0{,}97$ for monthly data [RiskMetrics1996]. In the same comparison, the RiskMetrics exponential smoother with $\lambda=0{,}94$ for all assets is used as the reference estimator in subsequent studies as well [Engle2002].

## Multi-Step Forecasting and the Link to the IGARCH Model

Risk managers are often interested in horizons longer than one day. Assuming that log-prices are generated by $p_t=p_{t-1}+\sigma_t\varepsilon_t$ with $\varepsilon_t\sim\mathrm{IID}\,N(0,1)$, and writing the cumulative return over $T$ periods as the sum of the increments, the variance forecast over $T$ steps is the sum of the expected one-step variances. In the EWMA structure the variance forecasts for two consecutive periods coincide, $E_t[\sigma^2_{t+s}]=E_t[\sigma^2_{t+s-1}]$, so that

$$\sigma^2_{t+T|t}=\sum_{s=1}^{T}E_t[\sigma^2_{t+s}]=T\cdot\sigma^2_{t+1|t},\qquad \sigma_{t+T|t}=\sqrt{T}\,\sigma_{t+1|t}.$$

This is the so-called "square-root-of-time" rule. Covariance forecasts also scale linearly with the horizon, $\sigma^2_{12,t+T|t}=T\,\sigma^2_{12,t+1|t}$, and consequently the forecast correlation remains unchanged as the horizon varies, $\rho_{t+T|t}=\rho_{t+1|t}$ [RiskMetrics1996].

This derivation has a precise theoretical implication. While the square-root-of-time rule usually follows from the assumption of constant variances, here variances and covariances vary over time; what is implicitly assumed by modeling variances and covariances as exponential moving averages is that the variance process is *non-stationary*. A model of this type has been extensively studied in the literature and is known as the IGARCH (Integrated GARCH) model [Nelson1990]. The EWMA is essentially equivalent to an IGARCH without an intercept, although its motivation is "bottom-up" — building a model consistent with observed returns and simple to implement — rather than "top-down" as in the formal statistical formulation estimated by maximum likelihood [RiskMetrics1996]. It should be noted that scaling volatility forecasts can lead to nonsensical results when prices are mean-reverting, when barriers limit their movements, or when estimates optimized for one horizon are used for another.

## From Univariate GARCH to the Multivariate Case

The EWMA is the simple but rigid special case of a broader family of volatility models. In univariate GARCH the conditional variance is a function of past residuals and past variances:

$$h_{it}=\omega_i+\sum_{p=1}^{P_i}\alpha_{ip}\,r^2_{i,t-p}+\sum_{q=1}^{Q_i}\beta_{iq}\,h_{i,t-q},$$

with the usual constraints of non-negativity of variances and stationarity $\sum_{p}\alpha_{ip}+\sum_{q}\beta_{iq}<1$ [EngleSheppard2001]. The GARCH generalization of the ARCH model and its variants (IGARCH, EGARCH) were introduced because return realizations exhibit time-dependent volatility [Bollerslev1986; RiskMetrics1996]. An example estimated on daily sterling returns is $\sigma^2_t=0{,}0147+0{,}881\,\sigma^2_{t-1}+0{,}0828\,r^2_{t-1}$; compared with the EWMA $\sigma^2_{t+1|t}=0{,}94\,\sigma^2_{t|t-1}+0{,}06\,r^2_t$, the dynamics of the exponential model closely track those of the GARCH(1,1), which is unsurprising given its formal closeness to the IGARCH [RiskMetrics1996].

The extension to the multivariate case is natural but runs up against the dimensionality problem of the matrices. The most general form is the *vec* model, which parameterizes the vector of all covariances and variances:

$$\mathrm{vec}(H_t)=\mathrm{vec}(\Omega)+A\,\mathrm{vec}(r_{t-1}r_{t-1}')+B\,\mathrm{vec}(H_{t-1}),$$

where $A$ and $B$ are $n^2\times n^2$ matrices; without further restrictions the model does not guarantee the positive definiteness of $H_t$ [EngleKroner1995]. Useful restrictions derive from the BEKK representation, $H_t=\Omega+A\,(r_{t-1}r_{t-1}')\,A'+B\,H_{t-1}\,B'$, which ensures positive definiteness but in the general case requires $O(k^4)$ parameters, reduced to $O(k^2)$ in the diagonal and scalar versions [Engle2002; EngleSheppard2001]. The problem is that the number of parameters in general multivariate models is too large for tractable optimization, and very few papers consider more than five assets despite the need for much larger correlation matrices [Engle2002]. A useful structure is *variance targeting*, whereby the long-run covariance matrix is set equal to the sample covariance: in the scalar case the intercept simply becomes $\Omega=(1-\alpha-\beta)\,S$ with $S=\frac1T\sum_t r_t r_t'$. Other generalizations, such as the Orthogonal GARCH method and the Kroner and Ng model, remain burdened by the same growth in the number of parameters or by interpretation difficulties [KronerNg1998]. It is in response to this scaling problem that the DCC was born.

## Engle's Dynamic Conditional Correlation Model

The DCC can be seen as a generalization of Bollerslev's constant conditional correlation estimator [Bollerslev1990]. The idea is to decompose the conditional covariance matrix into the product of standard deviations and correlations:

$$H_t=D_t\,R_t\,D_t,\qquad D_t=\mathrm{diag}\{\sqrt{h_{i,t}}\},$$

where $D_t$ is the diagonal matrix of conditional standard deviations coming from univariate GARCH models and $R_t$ is the conditional correlation matrix [Engle2002; EngleSheppard2001]. Defining standardized residuals $\varepsilon_t=D_t^{-1}r_t$, one sees that

$$E_{t-1}(\varepsilon_t\varepsilon_t')=D_t^{-1}H_t D_t^{-1}=R_t,$$

so that $R_t$ is simultaneously the correlation matrix of returns and the covariance matrix of standardized residuals. This reflects the very definition of conditional correlation as the conditional covariance between standardized innovations,

$$\rho_{12t}=\frac{E_{t-1}(r_{1t}r_{2t})}{\sqrt{E_{t-1}(r^2_{1t})\,E_{t-1}(r^2_{2t})}}=E_{t-1}(\varepsilon_{1t}\varepsilon_{2t}),$$

which by the laws of probability necessarily lies in $[-1,1]$ [Engle2002]. Bollerslev's constant correlation model sets $R_t=R$; the DCC differs in allowing $R_t$ to vary over time, with the only additional requirement being that the conditional variances of the standardized residuals be unity so that $R_t$ remains a correlation matrix.

The correlation dynamics are built on an auxiliary matrix $Q_t$. The simplest specification is the integrated exponential smoother,

$$q_{i,j,t}=(1-\lambda)\,\varepsilon_{i,t-1}\varepsilon_{j,t-1}+\lambda\,q_{i,j,t-1},\qquad \rho_{i,j,t}=\frac{q_{i,j,t}}{\sqrt{q_{ii,t}\,q_{jj,t}}},$$

in which the $q$'s turn out to be integrated. The natural alternative is the form suggested by the GARCH(1,1), which introduces mean reversion,

$$q_{i,j,t}=\bar\rho_{i,j}+\alpha\left(\varepsilon_{i,t-1}\varepsilon_{j,t-1}-\bar\rho_{i,j}\right)+\beta\left(q_{i,j,t-1}-\bar\rho_{i,j}\right),$$

where $\bar\rho_{i,j}$ is the unconditional expected value of the cross product and, for variances, $\bar\rho_{i,i}=1$. In matrix form the two update rules are written as

$$Q_t=(1-\lambda)\left(\varepsilon_{t-1}\varepsilon_{t-1}'\right)+\lambda\,Q_{t-1},\qquad Q_t=S\,(1-\alpha-\beta)+\alpha\left(\varepsilon_{t-1}\varepsilon_{t-1}'\right)+\beta\,Q_{t-1},$$

with $S$ the unconditional correlation matrix of the standardized residuals. The model is mean-reverting as long as $\alpha+\beta<1$; when the sum equals one it reduces exactly to the integrated exponential smoother [Engle2002].

The element $q_{i,j,t}$ is not a correlation: $Q_t$ must be reconditioned to a valid correlation matrix by dividing by the roots of the diagonal elements,

$$R_t=Q_t^{*-1}\,Q_t\,Q_t^{*-1},\qquad Q_t^{*}=\mathrm{diag}\{\sqrt{q_{11,t}},\dots,\sqrt{q_{kk,t}}\},$$

so that $R_t$ has elements in $[-1,1]$ and unity on the diagonal [Engle2002; EngleSheppard2001]. The general form of the DCC encompasses both rules:

$$Q_t=\Big(1-\textstyle\sum_m\alpha_m-\sum_n\beta_n\Big)\bar Q+\sum_m\alpha_m\left(\varepsilon_{t-m}\varepsilon_{t-m}'\right)+\sum_n\beta_n\,Q_{t-n},$$

where $\bar Q$ is the unconditional covariance of the first-stage standardized residuals [EngleSheppard2001].

Positive definiteness is easy to guarantee. The correlation $\rho_{i,j,t}=q_{i,j,t}/\sqrt{q_{i,i,t}q_{j,j,t}}$ is positive definite because $Q_t$ is a weighted average of a positive definite matrix $\bar Q$, of positive semi-definite matrices $\varepsilon_t\varepsilon_t'$, and of a positive definite matrix $Q_{t-1}$; a proposition of linear algebra then shows that the positive definiteness of $Q_t$ implies that of $R_t$. In essence, the conditions for positive definiteness of the conditional covariance in the DCC are the same as those of a univariate GARCH process — in particular $\omega_i>0$, non-negativity and stationarity of the GARCH parameters, $\alpha_m,\beta_n\ge 0$, $\sum_m\alpha_m+\sum_n\beta_n<1$, and a positive minimum eigenvalue of $\bar R$ — and are sufficient, not necessary [EngleSheppard2001]. More complex positive-definite multivariate parameterizations, such as the MARCH family of Ding and Engle, can be used for the correlations provided the unconditional moments are fixed at the sample correlation matrix [DingEngle2001].

## Two-Step Estimation and Its Theoretical Properties

The decisive merit of the DCC is that the number of parameters to be estimated in the correlation process is independent of the number of series to be correlated, so that potentially very large correlation matrices become estimable [Engle2002]. This follows from the structure of the likelihood. Assuming $r_t\,|\,\mathcal{F}_{t-1}\sim N(0,H_t)$, the log-likelihood decomposes by exploiting $H_t=D_tR_tD_t$:

$$L=-\frac12\sum_{t=1}^{T}\left(k\log(2\pi)+2\log|D_t|+\log|R_t|+\varepsilon_t'R_t^{-1}\varepsilon_t\right).$$

Writing the parameters as $\theta$ (variances) and $\phi$ (correlations), the likelihood splits into the sum of a volatility part and a correlation part, $L(\theta,\phi)=L_V(\theta)+L_C(\theta,\phi)$, with

$$L_V(\theta)=-\frac12\sum_{t}\left(k\log(2\pi)+\log|D_t|^2+r_t'D_t^{-2}r_t\right)=-\frac12\sum_t\sum_{i=1}^{k}\left(\log(2\pi)+\log h_{i,t}+\frac{r^2_{i,t}}{h_{i,t}}\right),$$

$$L_C(\theta,\phi)=-\frac12\sum_{t}\left(\log|R_t|+\varepsilon_t'R_t^{-1}\varepsilon_t-\varepsilon_t'\varepsilon_t\right).$$

The volatility part is manifestly the sum of the individual GARCH likelihoods, maximizable series by series. This yields the two-step scheme: first,

$$\hat\theta=\arg\max_{\theta}\{L_V(\theta)\};$$

second, taking $\hat\theta$ as given,

$$\max_{\phi}\{L_C(\hat\theta,\phi)\}.$$

One thus first estimates the univariate GARCH models for the variances, then the dynamics of the conditional correlations using the standardized residuals [Engle2002; EngleSheppard2001].

The consistency of this two-step estimator follows from results for two-stage GMM estimators. Under reasonable regularity conditions, consistency of the first step ensures consistency of the second: the maximum of the second step is a function of the first-step estimates, and if these are consistent so are those of the second, provided the function is continuous in a neighborhood of the true parameters [Engle2002]. Formally, the first-order condition of the first step is $\nabla_\theta L_V(\theta)=0$ and that of the second $\nabla_\phi L_C(\hat\theta,\phi)=0$; under standard regularity conditions the estimates are consistent and asymptotically normal, with a covariance matrix of the familiar form given by the product of two inverted Hessians around an outer product of scores [Engle2002; NeweyMcFadden1994].

The properties of the second stage, however, require a correction. In the absence of normality the estimator retains the quasi-maximum likelihood (QML) interpretation. The parameter estimates are consistent and asymptotically normal with $\sqrt{n}(\hat\theta_n-\theta_0)\overset{A}{\to}N(0,A_0^{-1}B_0A_0'^{-1})$; for the GARCH parameters of each asset the asymptotic variance coincides with the Bollerslev–Wooldridge robust estimator, so that the first-stage standard errors remain consistent and only the asymptotic covariance of the correlation parameters must be modified [EngleSheppard2001]. Since both sets of parameters are estimated separately by limited-information likelihood, the estimates are not fully efficient; however, being consistent at rate $\sqrt{n}$, a single Newton–Raphson iteration on the second-stage likelihood is sufficient to achieve full efficiency. Furthermore, a likelihood ratio test with $r$ restrictions does not in general have a $\chi^2_r$ distribution, since the information matrix equality does not hold: the limiting distribution is a weighted sum of independent $\chi^2_1$ variables, with weights equal to the eigenvalues of an explicitly derived matrix [EngleSheppard2001].

One of the primary motivations for the DCC is that correlations are not constant over time. To verify this, Engle and Sheppard propose a test of the null of constant correlation against the alternative of dynamic correlation that requires only the estimation of a restricted VAR: the univariate GARCH models are estimated, the residuals standardized, jointly standardized with the symmetric square root of $\bar R$, and the outer products of the residuals regressed on a constant and their own lags. Under the null, the constant and all lagged parameters must be zero, and the statistic $\hat\delta'X'X\hat\delta/\hat\sigma^2$ is asymptotically $\chi^2_{(s+1)}$; in every model considered the null of constant correlation is rejected in favor of a dynamic structure [EngleSheppard2001].

## Empirical Properties of the Estimator

The empirical properties of the DCC have been tested both in Monte Carlo experiments with known correlation structure and on real data. In the simulation experiment a bivariate GARCH is generated — with a highly persistent component, $h_{1,t}=0{,}01+0{,}05\,r^2_{1,t-1}+0{,}94\,h_{1,t-1}$, and a less persistent one, $h_{2,t}=0{,}5+0{,}2\,r^2_{2,t-1}+0{,}5\,h_{2,t-1}$ — letting the correlation follow different processes (constant, sinusoidal, rapid sinusoidal, step, ramp) to represent gradual changes, rapid changes, and periods of constancy. Eight estimators are compared by mean absolute error, by multivariate GARCH diagnostic tests on the squared standardized residuals, and by Value at Risk tests based on the dynamic quantile test [Engle2002; EngleManganelli1999]. The mean-reverting DCC estimated by likelihood has the smallest mean absolute error in four of the six cases and is the best when errors are summed over all cases; overall the DCC methods are the best or nearly the best, whatever the criterion [Engle2002].

On real data the estimates reveal time-varying features that would otherwise be difficult to quantify. The correlation between the Dow Jones and NASDAQ, over a decade, mostly oscillates between $0{,}6$ and $0{,}9$ but falls below $0{,}4$ in 1993 and again in March 2000; the GARCH volatilities show that the NASDAQ has always been more volatile than the Dow, with a gap that widens toward the end of the sample. Currency correlations provide the clearest evidence of non-stationarity: the breakdown between the German mark and the pound and the lira in August 1992 is clearly visible, and the convergence of European currencies toward a correlation close to unity in the period preceding the launch of the euro in January 1999 emerges distinctly [Engle2002].

Specification checks compare the DCC with the RiskMetrics exponential smoother, the standard industry benchmark, using portfolio standard deviation standardized by the conditional variance, Value at Risk performance (HIT test), and the autocorrelation of standardized residuals [EngleSheppard2001]. A particularly revealing test uses the minimum-variance portfolio, whose weights are determined by the estimated covariance matrix,

$$w_t=\frac{H_t^{-1}\iota}{C_t},\qquad C_t\equiv \iota'H_t^{-1}\iota,$$

with $H_t$ the one-step covariance forecast built at $t-1$ and $\iota$ a vector of ones: if the covariance is misspecified, the minimum-variance portfolio amplifies the flaw. The DCC estimator produces standardized residuals with variance within the confidence interval for all models with fewer than ten assets and performs well even in larger models, while the RiskMetrics estimator produces no standardized series within the $95\%$ confidence interval for any model. Overall the DCC is comparable to the benchmarks for large models and superior to RiskMetrics for small volatility models, and is competitive with multivariate GARCH specifications while being much simpler to estimate [EngleSheppard2001; Engle2002].

A practically desirable feature is consistency between multivariate and univariate forecasts: when new variables are added to the system, the volatility forecasts of the original assets remain unchanged and the correlations may even remain unchanged depending on how the model is revised [Engle2002]. Multi-step forecasting, however, does not admit a direct recursive solution as in the univariate GARCH, because the evolution process of $Q_t$ is non-linear; approximations are therefore used, and the one that solves $R_t$ forward directly turns out to be simpler to implement and subject to less bias [EngleSheppard2001].

## Time-Varying Covariance as an Input for Portfolio Optimization

The thread linking this chapter to portfolio selection is direct. Constructing an optimal portfolio subject to constraints and computing the portfolio's standard deviation both require a forecast of the covariance matrix of returns [Engle2002]. The dynamic estimators presented here — RiskMetrics' EWMA and Engle's DCC — provide such a conditional covariance matrix, updated at each observation with the most recent information, which constitutes an alternative input to the static sample covariance used in mean-variance selection (cf. [[02 Selezione media-varianza]]) and in matrix regularization methods (cf. [[08 Errore di stima, outlier e shrinkage]]).

This link becomes concrete in the minimum-variance portfolio, whose weights $w_t=H_t^{-1}\iota/(\iota'H_t^{-1}\iota)$ depend entirely on the estimated conditional covariance matrix $H_t$ [EngleSheppard2001]. Replacing the static matrix with the forecast $H_t$ produced by a dynamic model, the optimal weights themselves become time-varying and react to the changes in volatility and correlation documented in the first section. The quality of the input covariance is thus transmitted to the quality of the portfolio: this is why the minimum-variance portfolio is used as a specification test bench, and why a poorly estimated covariance matrix produces realized standard deviations far from the expected value [EngleSheppard2001]. Dynamic covariance estimation techniques thus sit upstream of the optimization problems treated elsewhere in the monograph (cf. [[14 Ottimizzazione convessa - coni, dualità e KKT]] and [[17 Risk budgeting e risk parity]]), as suppliers of the risk input on which those problems depend.

## References

- **[Bollerslev1986]** Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. Journal of Econometrics, 31(3), 307–327.
- **[Bollerslev1990]** Bollerslev, T. (1990). Modelling the Coherence in Short-run Nominal Exchange Rates: A Multivariate Generalized ARCH Model. Review of Economics and Statistics, 72(3), 498–505.
- **[DingEngle2001]** Ding, Z. & Engle, R. F. (2001). Large Scale Conditional Covariance Matrix Modeling, Estimation and Testing. Academia Economic Papers, 29(2), 157–184.
- **[Engle1982]** Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. Econometrica, 50(4), 987–1007.
- **[Engle2002]** Engle, R. F. (2002). Dynamic Conditional Correlation: A Simple Class of Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models. Journal of Business & Economic Statistics, 20(3), 339–350.
- **[EngleKroner1995]** Engle, R. F. & Kroner, K. F. (1995). Multivariate Simultaneous Generalized ARCH. Econometric Theory, 11(1), 122–150.
- **[EngleManganelli1999]** Engle, R. F. & Manganelli, S. (1999). CAViaR: Conditional Value At Risk By Regression Quantiles. NBER Working Paper No. 7341. National Bureau of Economic Research, Cambridge, MA.
- **[EngleSheppard2001]** Engle, R. F. & Sheppard, K. (2001). Theoretical and Empirical Properties of Dynamic Conditional Correlation Multivariate GARCH. NBER Working Paper No. 8554. National Bureau of Economic Research, Cambridge, MA.
- **[KronerNg1998]** Kroner, K. F. & Ng, V. K. (1998). Modeling Asymmetric Comovements of Asset Returns. Review of Financial Studies, 11(4), 817–844.
- **[Nelson1990]** Nelson, D. B. (1990). Stationarity and Persistence in the GARCH(1,1) Model. Econometric Theory, 6(3), 318–334.
- **[NeweyMcFadden1994]** Newey, W. K. & McFadden, D. (1994). Large Sample Estimation and Hypothesis Testing. In R. F. Engle & D. McFadden (Eds.), Handbook of Econometrics, Volume IV, Ch. 36, 2111–2245. Elsevier Science B.V.
- **[RiskMetrics1996]** J.P. Morgan / Morgan Guaranty Trust Company (Zangari, P.) (1996). RiskMetrics — Technical Document, Fourth Edition, Chapter 5: Estimation and Forecast. New York.
