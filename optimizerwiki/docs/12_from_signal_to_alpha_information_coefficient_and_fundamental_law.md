---
title: "From Signal to Alpha - Information Coefficient and the Fundamental Law"
chapter: 12
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-10
---

> [!abstract] Summary
> The chapter establishes that active management is forecasting and that the value added by a strategy is proportional to the square of its information ratio, which decomposes, by the fundamental law of active management, into the product of skill (the information coefficient, the correlation between forecasts and outcomes) and breadth (the number of independent bets per year). It further derives the practical forecasting rule of thumb, whereby a security's alpha is the product of volatility, information coefficient, and standardized score, shows how refinement compresses the raw signal toward consensus in proportion to skill, how to combine multiple signals, and why forecasting return leaves the risk estimate essentially unchanged.

## Active Management Is Forecasting: From Raw Forecast to Refined Forecast

A manager who makes no forecast holds the benchmark portfolio. Consensus forecasts of expected returns, efficiently implemented, indeed lead to the market or benchmark portfolio; active managers earn that name only by investing in portfolios that deviate from it, and to the extent that they claim to do so efficiently on the basis of their own information, they are at least implicitly forecasting exceptional returns [GrinoldKahn2000]. Active management, in this sense, coincides with the activity of forecasting.

It is useful to distinguish three types of forecast. The *naïve* forecast is the consensus expected return: it is the information-free forecast, and leads to benchmark positions. The *raw* forecast contains the manager's information in its original form — an earnings estimate, a buy or sell recommendation, a momentum measure — expressed in heterogeneous units and scales, and is not directly a forecast of exceptional return. The *basic forecasting formula* transforms raw forecasts into *refined* forecasts, whose outputs have the form and units of exceptional returns, corrected for the informational content of the raw signal [GrinoldKahn2000].

Given the raw information $\mathbf{g}$ ($K$ forecasts) and the vector of excess returns $\mathbf{r}$ ($N$ securities), the basic formula links deviations of forecasts from their expected levels to deviations of returns from their expected levels:
$$E\{\mathbf{r}\mid\mathbf{g}\} = E\{\mathbf{r}\} + \mathrm{Cov}\{\mathbf{r},\mathbf{g}\}\cdot\mathrm{Var}^{-1}\{\mathbf{g}\}\cdot(\mathbf{g}-E\{\mathbf{g}\}).$$
The refined forecast is defined as the change in expected return due to observing $\mathbf{g}$:
$$\boldsymbol{\phi} \equiv E\{\mathbf{r}\mid\mathbf{g}\} - E\{\mathbf{r}\} = \mathrm{Cov}\{\mathbf{r},\mathbf{g}\}\cdot\mathrm{Var}^{-1}\{\mathbf{g}\}\cdot(\mathbf{g}-E\{\mathbf{g}\}).$$
This is the exceptional return. Since the naïve forecast is $E\{\mathbf{r}\}=\boldsymbol{\beta}\cdot\mu_B$, with $\mu_B$ the consensus expected excess return of the benchmark, applying the formula directly to the residual returns $\boldsymbol{\theta}$ gives the equivalent $E\{\boldsymbol{\theta}\}=0$, and hence the alpha:
$$\boldsymbol{\alpha} = \mathrm{Cov}\{\boldsymbol{\theta},\mathbf{g}\}\cdot\mathrm{Var}^{-1}\{\mathbf{g}\}\cdot(\mathbf{g}-E\{\mathbf{g}\}).$$
Note that historical average returns are a poor alternative to consensus expected returns: they have very large sampling errors and are inadequate for new or changing securities (an issue taken up in [[08 Errore di stima, outlier e shrinkage]]). The basic formula is the best linear unbiased estimator (BLUE), with minimum mean-squared error, of the return conditional on the signal, and under joint normality of $\mathbf{r}$ and $\mathbf{g}$ it coincides with the conditional expectation and with the maximum-likelihood estimator [GrinoldKahn2000]. The factor structure that makes the residual returns $\boldsymbol{\theta}$ tractable is that of the factor models in [[05 Modelli fattoriali]].

## Information Ratio, Optimal Aggressiveness, and Value Added

The information ratio $IR$ measures an active manager's opportunities. If the manager exploits these opportunities in a mean-variance efficient way, the value added is proportional to the square of the information ratio; all investors seek out the strategies and managers with the highest information ratio [GrinoldKahn2000]. The value-added framework is the mean-variance one of [[02 Selezione media-varianza]].

Along the efficiently implemented residual frontier, the best attainable expected residual return at a given level of residual risk $\omega$ is $\alpha(\omega)=IR\cdot\omega$. The risk-adjusted value added, with residual risk aversion $\lambda_R$, is
$$\mathrm{VA}(\omega) = IR\cdot\omega - \lambda_R\,\omega^2.$$
The first-order condition $IR - 2\lambda_R\,\omega = 0$ gives the optimal level of aggressiveness
$$\omega^{*} = \frac{IR}{2\lambda_R},$$
and substituting back yields the optimal value added
$$\mathrm{VA}^{*} = IR\cdot\omega^{*} - \lambda_R\,(\omega^{*})^2 = \frac{IR^2}{2\lambda_R} - \frac{IR^2}{4\lambda_R} = \frac{IR^2}{4\lambda_R}.$$
Value added thus grows with the square of the information ratio. This relation is additive: if a sponsor allocates shares $y_n$ to managers with independent information ratios $IR_n$ and active risk $\omega_n$, maximizing $\sum_n y_n\,IR_n\,\omega_n - \lambda_{SA}\sum_n (y_n\,\omega_n)^2$ gives the optimal share $y_n^{*}=IR_n/(2\lambda_{SA}\,\omega_n)$, overall alpha $(1/2\lambda_{SA})\sum_n IR_n^2$, active variance $(1/2\lambda_{SA})^2\sum_n IR_n^2$, and hence
$$IR = \sqrt{\sum_n IR_n^2}.$$
Thus, three managers with information ratios $0{,}75$, $0{,}50$, and $0{,}30$ allow the sponsor to obtain $0{,}95=\sqrt{0{,}75^2+0{,}50^2+0{,}30^2}$ [GrinoldKahn2000].

## The Fundamental Law of Active Management

The fundamental law of active management gives a simple and surprisingly general formula that approximates the information ratio from two attributes of the strategy: breadth and skill [Grinold1989; GrinoldKahn2000].

*Breadth* $BR$ is defined as the number of independent forecasts of exceptional return made each year. The *information coefficient* $IC$ is the measure of the manager's skill: it is the correlation of each forecast with actual outcomes; for convenience it is assumed to be the same across all forecasts. The law links breadth and skill to the information ratio via the (approximate) relation
$$IR = IC\cdot\sqrt{BR}.$$
The approximation ignores the risk-reduction benefits offered by the forecasts; for relatively low values of $IC$ (below $0{,}1$) this reduction is extremely small. Combining the law with the relations of the previous section, aggressiveness and value added can be expressed as a function of skill and breadth:
$$\omega^{*} = \frac{IC\cdot\sqrt{BR}}{2\lambda_R}, \qquad \mathrm{VA}^{*} = \frac{IC^2\cdot BR}{4\lambda_R}.$$
The desired aggressiveness grows directly with skill and with the square root of breadth; value added grows with breadth and with the *square* of skill [Grinold1989].

To raise the information ratio from $0{,}5$ to $1{,}0$ one must double the skill, quadruple the breadth, or some combination of the two. A 50% increase in breadth (for the same skill) is equivalent to a 22% increase in skill, since $\sqrt{1{,}5}\approx1{,}22$. Strategies with the same information ratio can impose radically different requirements: an information ratio of $0{,}50$ is attainable by a market timer with $IC=0{,}25$ and $BR=4$ ($0{,}50=0{,}25\cdot\sqrt{4}$), by a stock picker following 100 companies with quarterly review, $IC=0{,}025$ and $BR=400$ ($0{,}50=0{,}025\cdot\sqrt{400}$), or by a specialist following two companies who revises their bets 200 times a year, with $BR=400$ and $IC=0{,}025$ [Grinold1989]. The message is clear: one must play often (high $BR$) and play well (high $IC$).

The power of breadth is captured in the roulette-wheel example. With 18 red slots, 18 black, and 1 green, each slot has probability $1/37$; the house's skill is $1/37$. On a single one-dollar bet the information ratio is $2{,}7027/99{,}9634 = 0{,}027$, close to the prediction $(1/37)\cdot\sqrt{1}$. Operating like a real casino with one million bets a year, the expected return remains $2{,}7027\%$ but the standard deviation falls to $0{,}09996\%$, with an information ratio of $27{,}0$, in line with $(1/37)\cdot\sqrt{1{,}000{,}000}\approx27{,}03$ [GrinoldKahn2000].

The law should not be interpreted as a version of the law of large numbers: it states that, holding skill fixed, more breadth is better, and it holds with $BR=10$ just as with $BR=1000$, because greater breadth for the same skill allows residual risk to be diversified. The law is a guideline, not an operational tool, and it is in particular difficult to accurately estimate $BR$, because of the independence requirement on the forecasts [Grinold1989].

## Skill as Correlation and the Link to the Hit Rate

Skill is the correlation between forecast and outcome. The direct link to the percentage of correct forecasts (hit rate) is obtained in the binary model of direction forecasting. Model market direction as a variable $x(t)=\pm1$ with mean 0 and standard deviation 1, and the forecast as $y(t)=\pm1$, also with mean 0 and standard deviation 1. The information coefficient, the correlation of $x(t)$ and $y(t)$, depends on the covariance:
$$IC = \mathrm{Cov}\{x(t),y(t)\} = \frac{1}{N}\sum_{t=1}^{N} x(t)\,y(t),$$
with $N$ observed bets on direction. If the direction is forecast correctly ($x=y$) $N_1$ times and incorrectly ($x=-y$) $N-N_1$ times,
$$IC = \frac{1}{N}\big[N_1 - (N-N_1)\big] = 2\cdot\frac{N_1}{N} - 1.$$
Setting $p=N_1/N$ as the hit rate, one obtains the relation
$$IC \approx 2\,p - 1.$$
The relation shows how little information is needed for success: an information coefficient of $0{,}0577$ corresponds to correctly forecasting direction only $52{,}885\%$ of the time, a minimal edge that nonetheless, over 200 securities every quarter, produces an information ratio above $1{,}0$; an $IC=0{,}02$ (implied accuracy of $51\%$) over 200 securities quarterly produces a respectable information ratio of $0{,}56$ [Grinold1989; GrinoldKahn2000]. In the absence of sufficient history to estimate $IC$, the following vague but time-tested guidelines apply: a good forecaster has $IC=0{,}05$, an excellent one $IC=0{,}10$, a world-class one $IC=0{,}15$; an $IC$ above $0{,}20$ usually signals a flawed backtest or an impending insider-trading investigation — a warning that ties back to the cautions of [[10 Validazione, data snooping e overfitting]] [GrinoldKahn2000].

## The Golden Rule of Forecasting: Alpha Equals Volatility Times IC Times Score

The basic formula takes a particularly transparent form in the case of a single security and a single forecast. Writing the refined forecast as
$$\phi = \mathrm{Std}\{r\}\cdot\mathrm{Corr}\{r,g\}\cdot\left(\frac{g-E\{g\}}{\mathrm{Std}\{g\}}\right),$$
three factors can be recognized: the volatility of the return to be forecast, the information coefficient $\mathrm{Corr}\{r,g\}$, and the standardized version of the raw forecast. The latter — the raw forecast with its mean subtracted and divided by its standard deviation — is called the *score* or *z-score*. This yields the golden rule (rule of thumb) of forecasting [Grinold1994; GrinoldKahn2000]:
$$\text{refined forecast} = \text{volatility}\cdot IC\cdot\text{score}.$$
The same rule emerges from regression analysis. Given a historical series of forecasts $g(t)$ and subsequent returns $r(t)$, the regression $r(t)=c_0+c_1\,g(t)+\epsilon(t)$ has least-squares estimate $c_1=\mathrm{Cov}\{r,g\}/\mathrm{Var}\{g\}=\mathrm{Std}\{r\}\cdot\mathrm{Corr}\{r,g\}/\mathrm{Std}\{g\}$; defining the score $z(t)=(g(t)-m_g)/\mathrm{Std}\{g\}$ one recovers
$$\phi = \mathrm{Std}\{r\}\cdot\mathrm{Corr}\{r,g\}\cdot z(T+1) = \text{volatility}\cdot IC\cdot\text{score}.$$
The binary model and the regression thus lead to the same rule: volatility and $IC$ are constant for a given signal, while the score distinguishes one forecast from another. The score, by construction, has mean and standard deviation close to 0 and 1 [Grinold1994].

The refinement process controls for three elements. It controls for *expectations*, by subtracting the expected raw forecast in computing the score: exceptional price movement is expected only when the raw information differs from the consensus — when earnings match expectations the price does not move. It controls for *skill* via $IC$: if $IC=0$ the raw forecast contains no useful information and the refined forecast of exceptional return is set to zero. Finally, it controls for *volatility*: $IC$ and score are dimensionless, and it is the volatility term that supplies the dimensions of return, so that, for the same score, the more volatile security receives the larger alpha [Grinold1994; GrinoldKahn2000].

The rule allows refining information even in unstructured situations. For a simple tip on a security with typical residual volatility of $20\%$: for $IC$ one looks at the track record of the source (excellent source $IC=0{,}1$, good $IC=0{,}05$, useless $IC=0$); for the score one assigns $1{,}0$ to a very positive tip and $2{,}0$ to a very, very positive one. A very positive tip from an excellent source is thus worth $0{,}20\cdot0{,}10\cdot1{,}0=2{,}0\%$ of alpha. Similarly, a buy and sell list (score $+1{,}0$ and $-1{,}0$) with $IC=0{,}09$ assigns alphas proportional to residual volatility, so that the optimizer favors, among the buy-list securities, those with lower residual risk [GrinoldKahn2000].

## Refinement as Compression Toward the Consensus

The golden rule makes explicit the sense in which refinement *compresses* the raw signal toward the consensus, in proportion to skill. Since the score has unit standard deviation, the dispersion of refined forecasts is
$$\mathrm{Std}\{\phi\} = \mathrm{Std}\{r\}\cdot IC.$$
The dispersion of the alphas thus equals $IC$ multiplied by the dispersion of realized returns: the alphas are compressed, relative to what a naïve reading of the signal would suggest, by a factor equal to skill. In the limit $IC=0$ all refined forecasts collapse to zero, that is, to the consensus residual return, and benchmark positions are held, as should be the case in the absence of information [Grinold1994].

This can be seen by explicitly composing the signal. Imagining the residual return $\theta_n$ as the sum of independent contributions and modeling the forecast as
$$\alpha = IC\cdot\big[IC\cdot\theta + \omega\cdot\sqrt{1-IC^2}\cdot z\big],$$
with $z$ a random variable with mean 0 and variance 1, one obtains $\mathrm{Var}\{\alpha\}=IC^2\cdot\mathrm{Var}\{\theta\}$ and $\mathrm{Cov}\{\alpha,\theta\}=IC^2\cdot\mathrm{Var}\{\theta\}$, from which $\mathrm{Corr}\{\alpha,\theta\}=IC$ [GrinoldKahn2000]. Alpha is thus a rescaled version of the residual return, with dispersion reduced by the factor $IC$: extreme raw forecasts are pulled back toward the consensus the more so the lower the forecaster's skill. If the scores are normally distributed, with quarterly volatility of $9\%$ and $IC=0{,}0833$, the refined forecast is $0{,}75$ times the score in percentage points, and falls between $-0{,}75$ and $+0{,}75$ two quarters out of three [Grinold1994].

## Combining Multiple Signals

With multiple sources of information, the fundamental law is additive in the squares of the information ratios. If one class has $BR_1$ securities with skill $IC_1$ and a second has $BR_2$ securities with skill $IC_2$, the aggregate information ratio satisfies, under optimal implementation,
$$IR^2 = BR_1\cdot IC_1^2 + BR_2\cdot IC_2^2.$$
Thus a manager following 200 securities with semiannual forecasts ($BR=400$) and $IC=0{,}04$ has $IR=0{,}8$; adding 100 securities with two annual forecasts and $IC=0{,}03$, the value added becomes proportional to $0{,}64+(0{,}03)^2\cdot200=0{,}82$, and the information ratio rises to $0{,}906=\sqrt{0{,}82}$ [Grinold1989]. Additivity also holds across different dimensions (stock selection plus market timing) and for the international portfolio, where active return comes from currency positions, country allocations, and selection within markets.

The additivity of squares, however, presupposes independence of the forecasts: the second forecast must not be based on a source correlated with the first. If $\gamma$ is the correlation between two sources with equal skill $IC$, the skill of the combined sources is
$$IC(\text{com}) = IC\cdot\sqrt{\frac{2}{1+\gamma}}.$$
For $\gamma=0$, $IC^2(\text{com})=2\cdot IC^2$: the two sources add up their capacity to add value. As $\gamma$ approaches 1, the value of the second source diminishes, since part of its information merely reinforces what was already known from the first [Grinold1989].

The same principle governs combination at the score level. For a security with two raw forecasts $g$ and $g'$ correlated with coefficient $\rho_{gg'}$, the combined refined forecast is
$$\phi = \mathrm{Std}\{r\}\cdot IC_g^{*}\cdot z_g + \mathrm{Std}\{r\}\cdot IC_{g'}^{*}\cdot z_{g'},$$
with revised skills that account for the correlation between forecasts:
$$IC_g^{*} = \frac{IC_g - \rho_{gg'}\cdot IC_{g'}}{1-\rho_{gg'}^2}, \qquad IC_{g'}^{*} = \frac{IC_{g'} - \rho_{gg'}\cdot IC_g}{1-\rho_{gg'}^2}.$$
If the forecasts are uncorrelated the combination reduces to the sum of the separate refined forecasts; if they are perfectly correlated ($\rho_{gg'}=1$) the formulas degenerate and the second forecast adds nothing. The information coefficient of the combined forecast is
$$IC_{\text{combined}} = \sqrt{\frac{IC_g^2 + IC_{g'}^2 - 2\,\rho_{gg'}\cdot IC_g\cdot IC_{g'}}{1-\rho_{gg'}^2}},$$
and for uncorrelated forecasts the square of the combined $IC$ is the sum of the squares of the two component $IC$s [GrinoldKahn2000]. With three signals of equal skill, of which the first two are strongly correlated with each other but uncorrelated with the third, refinement halves the $IC$s of the two correlated signals, effectively counting the independent signal on par with the sum of the two correlated ones.

## Cross-Sectional Scores and Factor Forecasts

In the case, typical for the institutional manager, of many securities, the basic formula continues to hold and the golden rule applies to each security $n$:
$$\phi_n = \omega_n\cdot IC\cdot z_{\mathrm{TS},n},$$
where $z_{\mathrm{TS},n}$ is the security's time-series score, with mean 0 and standard deviation 1 over time. In practice, however, one has only a single numerical forecast per security at each instant, from which a *cross-sectional* score $z_{\mathrm{CS},n}$ is derived (mean 0 and standard deviation 1 across securities at a given time), not the required time-series score. How to refine cross-sectional scores depends critically on how the signal's time-series volatilities vary from security to security [GrinoldKahn2000].

If the signal's time-series volatility is identical for every security, the time-series scores coincide with the cross-sectional ones and $\alpha_n=\omega_n\cdot IC\cdot z_{\mathrm{CS},n}$: one still multiplies by volatility. If instead the signal's time-series volatilities are proportional to the securities' volatilities, then
$$\phi_n = IC\cdot c_g\cdot z_{\mathrm{CS},n},$$
with $c_g$ a constant that does not depend on the security: the refined forecasts are proportional to the cross-sectional scores and *independent* of volatility. In that case multiplying the cross-sectional scores by volatility would be wrong. The golden rule remains "volatility $\cdot$ $IC$ $\cdot$ score," but sometimes this is simply proportional to "$IC$ $\cdot$ cross-sectional score" [GrinoldKahn2000]. Empirical evidence on six equity signals shows, for five of them (dividend discount model, estimate change, estimate revision, relative strength, residual reversal), a strong positive linear relationship between the signal's time-series volatility and the security's residual volatility, so scores should not be rescaled by volatility; only for sector momentum, where the signal's volatility does not vary with the security's, do volatility-scaled scores prove superior.

A further structure for the many-securities case is the factor model. If one has information and forecasts the return of some factors, the forecasts for the other factors should not be set to zero. If a signal $g_1$ forecasts factor $b_1$ and $\rho_{1j}$ is the correlation between $b_1$ and $b_j$, the basic formula gives
$$E\{b_j\mid g_1\} = IC\cdot\rho_{1j}\cdot\omega_j\cdot z_1,$$
so that, when forecasting $E\{b_1\mid g_1\}\neq0$, it is not consistent to set $E\{b_j\mid g_1\}=0$ [GrinoldKahn2000]. Empirical verification on a book-to-price factor confirms that using information about $b_1$ to also bet on $b_j$ improves performance, and that the square of the information ratio of the strategy betting on all factors approximately equals the sum of the squares of the information ratios of the component strategies. These ideas were developed by Black and Litterman in the context of international allocation [BlackLitterman1991] and are taken up in [[03 Limiti della MPT e il modello di Black-Litterman]]. The operational construction of portfolios from these alphas, and their scaling consistent with the information ratio, is the subject of [[13 Selezione e costruzione quantitativa]].

## Why Forecasting Return Does Not Affect the Risk Estimate

A surprising result justifies decoupling the estimation of alpha from the estimation of covariance: return forecasts have a negligible effect on risk forecasts. Consider the temptation to replace a historical correlation — say $0{,}95$ between two indices — with a negative correlation, on the grounds of a forecast that one will do well and the other poorly. This temptation is mistaken, because it confuses the notion of conditional mean (how research influences expected return) with that of conditional covariance (how research influences variances and covariances) [GrinoldKahn2000].

The small effect that does exist depends not on the forecast but only on the forecaster's skill. Let $\sigma_{\mathrm{PRIOR}}$ and $\sigma_{\mathrm{POST}}$ be the volatility estimates without and with the forecasting information. The basic variance formula, $\mathrm{Var}\{r\mid g\}=\mathrm{Var}\{r\}-\mathrm{Cov}\{r,g\}\cdot\mathrm{Var}^{-1}\{g\}\cdot\mathrm{Cov}\{g,r\}$, leads to
$$\sigma_{\mathrm{POST}} = \sigma_{\mathrm{PRIOR}}\cdot\sqrt{1-IC^2}.$$
With $\sigma_{\mathrm{PRIOR}}=18\%$ annual, the post-forecast volatility is $17{,}98\%$ for $IC=0{,}05$, $17{,}91\%$ for $IC=0{,}10$, $17{,}80\%$ for $IC=0{,}15$, $17{,}43\%$ for $IC=0{,}25$; only for unrealistic levels of skill does the effect become relevant ($5{,}62\%$ for $IC=0{,}95$, $0\%$ for $IC=1$). At reasonable levels of $IC$ (from 0 to $0{,}15$) the effect on volatility is minimal [GrinoldKahn2000]. The result arises from the fact that risk measures uncertainty about return: a skillful forecaster reduces the amount of uncertainty (a perfect forecaster eliminates it), but the magnitude of the residual uncertainty remains the same regardless of the particular forecast. Similarly, the revised correlation between two securities,
$$\rho_{ML}^{*} = \rho_{ML}\cdot\left[\frac{1-IC_M\cdot IC_L}{\sqrt{(1-IC_M^2)(1-IC_L^2)}}\right],$$
remains unchanged if $IC_M=IC_L$ and changes very little for $IC$ in the range 0 to $0{,}15$: it depends on skill, not on the forecast [GrinoldKahn2000]. The practical conclusion is that those forecasting short-horizon returns can ignore the impact of such forecasts on volatility and correlation estimates, focusing on the expected-return component rather than the risk one. Issues of signal horizon and persistence are treated in [[11 Decadimento del segnale, orizzonte e turnover]].

## Uncertainty in the Information Coefficient and Alpha Compression

The entire preceding derivation assumes $IC$ is known. A practical problem is uncertainty about the information coefficient itself, and how it should affect the refined signals: for the same estimated $IC$, one expects to weight more heavily the signal whose $IC$ is estimated with greater precision [GrinoldKahn2000].

Refining a signal via the regression $\theta(t)=a+b\cdot g(t)+\epsilon_\theta(t)$ and introducing a prior $\hat b=0$ on the coefficient, the Bayesian methodology leads, following Connor, to
$$b' = \left[\frac{1}{1+\dfrac{1}{T\cdot E\{R^2/(1-R^2)\}}}\right]\cdot b,$$
which involves the expected $R^2$ of the regression. Since this $R^2$ should equal $IC^2$, and hence be very small, it is approximated as
$$b' \approx \left(\frac{1}{1+\dfrac{1}{T\cdot IC^2}}\right)\cdot b,$$
with $T$ the number of months of observation [Connor1997; GrinoldKahn2000]. This is a shrinkage of the original estimate $b$ in the face of uncertainty: with a lot of data or a high information coefficient one stays close to the naïve estimate, while with few periods or a low $IC$ one shrinks toward zero. The shrinkage is significant even for good signals observed for a long time: for $IC=0{,}05$ the factor is $0{,}13$ after 60 months and $0{,}23$ after 120 months. Since with $IC\ll1$ the estimation error of $IC$ dominates the overall error in the regression coefficient, it is reasonable to assume that Bayesian shrinkage can be applied directly to $IC$: the greater the uncertainty about the estimated information coefficient, the more $IC$ is shrunk toward zero [GrinoldKahn2000]. With multiple signals, the same shrinkage applies by substituting the marginal $R^2$s for the total $R^2$, placing a premium on parsimony — a new signal with small marginal explanatory power undergoes substantial shrinkage. The connection to the broader topic of estimator shrinkage is developed in [[08 Errore di stima, outlier e shrinkage]].

## References

- **[BlackLitterman1991]** Black, F. & Litterman, R. (1991). "Global Asset Allocation with Equities, Bonds, and Currencies." Fixed Income Research, Goldman, Sachs & Co., New York.
- **[Connor1997]** Connor, G. (1997). "Sensible Return Forecasting for Portfolio Management." Financial Analysts Journal, 53(5), 44–51.
- **[Grinold1989]** Grinold, R. C. (1989). "The Fundamental Law of Active Management." Journal of Portfolio Management, 15(3), 30–37.
- **[Grinold1994]** Grinold, R. C. (1994). "Alpha Is Volatility Times IC Times Score, or Real Alphas Don't Get Eaten." Journal of Portfolio Management, 20(4), 9–16.
- **[GrinoldKahn2000]** Grinold, R. C. & Kahn, R. N. (2000). Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk (2nd ed.). New York: McGraw-Hill.
