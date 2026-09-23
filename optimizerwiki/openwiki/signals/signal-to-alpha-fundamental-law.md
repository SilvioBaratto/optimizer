---
type: concept
title: "From Signal to Alpha: Information Coefficient and the Fundamental Law"
description: How a raw research signal becomes a return forecast with the dimensions of an excess return — the forecasting rule alpha = volatility × IC × score — and how the value a strategy can add scales with the square of its information ratio, which the Fundamental Law of Active Management decomposes into skill (the information coefficient, the correlation of forecasts with outcomes) times the square root of breadth (the number of independent bets per year). Includes refinement as shrinkage toward consensus, the combination of correlated signals, and why a return forecast leaves the risk estimate essentially unchanged.
tags: [information-coefficient, information-ratio, fundamental-law-of-active-management, breadth, skill, alpha, forecasting-rule, z-score, shrinkage, signal-combination, hit-rate]
sources:
  - id: openwiki-source-991b8fcd429753c5cbd88ebd
    resource: repo://docs/12_from_signal_to_alpha_information_coefficient_and_fundamental_law.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# From Signal to Alpha: Information Coefficient and the Fundamental Law

This chapter establishes that active management *is* forecasting, and that the value a strategy adds is
proportional to the square of its information ratio, which the Fundamental Law of Active Management
decomposes into the product of skill (the information coefficient, the correlation between forecasts and
outcomes) and breadth (the number of independent bets per year). It derives the practical forecasting
rule — a stock's alpha is the product of volatility, information coefficient and standardized score —
and shows how refinement compresses the raw signal toward consensus in proportion to skill, how to
combine several signals, and why a return forecast leaves the risk estimate essentially unchanged. It
feeds the raw signals catalogued in [fundamental stock selection](./fundamental-stock-selection.md), is
used to build portfolios in [quantitative selection and construction](./quantitative-selection-and-construction.md),
and its skill measures decay over time as documented in [signal decay, horizon and turnover](./signal-decay-horizon-turnover.md).

## Active management is forecasting: from raw to refined forecasts

A manager who forms no forecast holds the benchmark portfolio: consensus expected-return forecasts,
efficiently implemented, lead to the market or benchmark portfolio, so active managers earn the title
only by holding portfolios that depart from it, and to the extent they claim to do so efficiently on
their information they are at least implicitly forecasting exceptional returns. Three forecast types are
useful to distinguish. The *naïve* forecast is the consensus expected return — the information-free
forecast that leads to benchmark positions. The *raw* forecast carries the manager's information in its
original form (an earnings estimate, a buy/sell recommendation, a momentum measure) in heterogeneous
units and scales, and is not directly a forecast of exceptional return. The basic forecasting formula
turns raw forecasts into *refined* forecasts whose outputs have the form and units of exceptional
returns, corrected for the raw signal's information content.

Given raw information $\mathbf{g}$ ($K$ forecasts) and the excess-return vector $\mathbf{r}$ ($N$
stocks), the basic formula links deviations of the forecasts from their expected levels to deviations of
returns from theirs, and the refined forecast is the change in expected return caused by observing
$\mathbf{g}$,
$$\boldsymbol{\phi} \equiv E\{\mathbf{r}\mid\mathbf{g}\} - E\{\mathbf{r}\} = \mathrm{Cov}\{\mathbf{r},\mathbf{g}\}\cdot\mathrm{Var}^{-1}\{\mathbf{g}\}\cdot(\mathbf{g}-E\{\mathbf{g}\}).$$
Applied to residual returns $\boldsymbol{\theta}$ (with $E\{\boldsymbol{\theta}\}=0$) this yields the
alpha, $\boldsymbol{\alpha} = \mathrm{Cov}\{\boldsymbol{\theta},\mathbf{g}\}\cdot\mathrm{Var}^{-1}
\{\mathbf{g}\}\cdot(\mathbf{g}-E\{\mathbf{g}\})$. The basic formula is the best linear unbiased estimator
(BLUE) of the return conditional on the signal, and under joint normality of $\mathbf{r}$ and
$\mathbf{g}$ it coincides with the conditional expectation and the maximum-likelihood estimator.
Historical average returns are a poor substitute for consensus expected returns — very large sampling
error, and inadequate for new or changing stocks. The factor structure that makes the residual returns
$\boldsymbol{\theta}$ tractable is that of the [factor models](../factor-models/factor-models.md).

## Information ratio, optimal aggressiveness and value added

The information ratio $IR$ measures an active manager's opportunities. If those opportunities are
exploited mean-variance efficiently, along the residual frontier the best expected residual return at a
given residual risk $\omega$ is $\alpha(\omega)=IR\cdot\omega$, so the risk-adjusted value added at
residual risk aversion $\lambda_R$ is $\mathrm{VA}(\omega)=IR\cdot\omega-\lambda_R\omega^2$. The
first-order condition gives the optimal aggressiveness
$$\omega^{*} = \frac{IR}{2\lambda_R},$$
and substituting yields the optimal value added
$$\mathrm{VA}^{*} = \frac{IR^2}{4\lambda_R},$$
so value added grows with the *square* of the information ratio and every investor seeks the highest-IR
strategies. The relation is additive: allocating to managers with independent information ratios
$IR_n$, the sponsor's aggregate information ratio is
$$IR = \sqrt{\sum_n IR_n^2},$$
so three managers with information ratios $0.75$, $0.50$ and $0.30$ let the sponsor achieve
$0.95=\sqrt{0.75^2+0.50^2+0.30^2}$.

## The Fundamental Law of Active Management

The Fundamental Law of Active Management gives a simple, surprisingly general formula that approximates
the information ratio from two attributes of the strategy: breadth and skill. *Breadth* $BR$ is the
number of independent forecasts of exceptional return made per year; the *information coefficient* $IC$
is the manager's skill, the correlation of each forecast with the actual outcome (assumed equal across
forecasts for convenience). The law links them by
$$IR = IC\cdot\sqrt{BR},$$
an approximation that ignores the risk-reduction benefit of the forecasts, negligible for low $IC$
(below $0.1$). Combined with the value-added relations, aggressiveness and value added become
$$\omega^{*} = \frac{IC\cdot\sqrt{BR}}{2\lambda_R}, \qquad \mathrm{VA}^{*} = \frac{IC^2\cdot BR}{4\lambda_R},$$
so desired aggressiveness rises with skill and with the square root of breadth, while value added rises
with breadth and with the *square* of skill.

To raise the information ratio from $0.5$ to $1.0$ one must double skill, quadruple breadth, or combine
the two; a $50\%$ increase in breadth equals a $22\%$ increase in skill, since $\sqrt{1.5}\approx1.22$.
Strategies with the same information ratio can impose radically different requirements: $IR=0.50$ is
reachable by a market timer with $IC=0.25$ and $BR=4$, by a selector following 100 firms with quarterly
revision at $IC=0.025$ and $BR=400$, or by a specialist following two firms but revising bets 200 times
a year with $BR=400$ and $IC=0.025$. The message is to play often (high $BR$) and to play well (high
$IC$). The power of breadth shows in the roulette bank: with 18 red, 18 black and 1 green each pocket
has probability $1/37$ (the bank's skill), a single \$1 bet has information ratio
$2.7027/99.9634=0.027\approx(1/37)\sqrt{1}$, while a casino making a million bets a year keeps the same
$2.7027\%$ expected return but cuts the standard deviation to $0.09996\%$ for an information ratio of
$27.0\approx(1/37)\sqrt{1{,}000{,}000}$. The law is not the law of large numbers — it holds that more
breadth at fixed skill is better, at $BR=10$ as at $BR=1000$ — and it is a guideline, not an operational
tool, since $BR$ is hard to estimate given the independence requirement.

## Skill as correlation and the link to the hit rate

Skill is the correlation between forecast and outcome, and the direct link to the fraction of correct
forecasts (hit rate) appears in the binary direction model. With market direction $x(t)=\pm1$ and
forecast $y(t)=\pm1$ (each mean 0, standard deviation 1), the information coefficient is their covariance
$IC=\tfrac1N\sum_t x(t)y(t)$; if the direction is called correctly $N_1$ times out of $N$,
$$IC = 2\cdot\frac{N_1}{N} - 1 \approx 2p - 1,$$
where $p=N_1/N$ is the hit rate. How little information success requires: $IC=0.0577$ corresponds to
calling direction correctly only $52.885\%$ of the time, yet over 200 stocks each quarter that produces
an information ratio above $1.0$; $IC=0.02$ (implied $51\%$ accuracy) over 200 quarterly stocks yields a
respectable $0.56$. Absent enough history to estimate $IC$, the vague but tested guides are: a good
forecaster has $IC=0.05$, an excellent one $0.10$, a world-class one $0.15$; an $IC$ above $0.20$
usually signals a flawed backtest or an imminent insider-trading investigation.

## The golden rule of forecasting: alpha equals volatility times IC times score

The basic formula is most transparent for one stock and one forecast: writing the refined forecast as
$$\phi = \mathrm{Std}\{r\}\cdot\mathrm{Corr}\{r,g\}\cdot\left(\frac{g-E\{g\}}{\mathrm{Std}\{g\}}\right),$$
three factors appear — the volatility of the return to be forecast, the information coefficient
$\mathrm{Corr}\{r,g\}$, and the standardized raw forecast (the raw forecast minus its mean, divided by
its standard deviation), called the *score* or *z-score*. Hence the golden rule of forecasting,
$$\text{refined forecast} = \text{volatility}\cdot IC\cdot\text{score}.$$
The same rule emerges from the regression $r(t)=c_0+c_1 g(t)+\epsilon(t)$, whose least-squares slope is
$c_1=\mathrm{Std}\{r\}\cdot\mathrm{Corr}\{r,g\}/\mathrm{Std}\{g\}$; defining the score
$z(t)=(g(t)-m_g)/\mathrm{Std}\{g\}$ recovers $\phi=\mathrm{Std}\{r\}\cdot\mathrm{Corr}\{r,g\}\cdot z(T+1)$.
Volatility and $IC$ are constants for a given signal, while the score distinguishes one forecast from
another.

Refinement controls for three things. It controls for *expectations*, subtracting the expected raw
forecast in the score — exceptional price movement is expected only when the raw information departs from
consensus (when earnings meet expectations the price does not move). It controls for *skill* through the
$IC$: if $IC=0$ the raw forecast carries no useful information and the refined exceptional-return
forecast is set to zero. And it controls for *volatility*: $IC$ and score are dimensionless, so it is the
volatility term that supplies return units, and at equal score the more volatile stock receives the
larger alpha. The rule refines information even in unstructured settings: for a tip on a stock with
$20\%$ residual volatility, an excellent source ($IC=0.1$) giving a very positive signal (score $1.0$) is
worth $0.20\cdot0.10\cdot1.0=2.0\%$ of alpha.

## Refinement as compression toward consensus

The golden rule makes explicit the sense in which refinement *compresses* the raw signal toward
consensus in proportion to skill. Since the score has unit standard deviation, the dispersion of refined
forecasts is
$$\mathrm{Std}\{\phi\} = \mathrm{Std}\{r\}\cdot IC,$$
i.e. the $IC$ times the dispersion of realized returns: alphas are compressed, relative to a naïve
reading of the signal, by a factor equal to skill. In the limit $IC=0$ all refined forecasts collapse to
zero — the consensus residual return — and benchmark positions are held. Composing the signal explicitly
as $\alpha=IC\cdot[IC\cdot\theta+\omega\sqrt{1-IC^2}\cdot z]$ gives $\mathrm{Var}\{\alpha\}=IC^2
\mathrm{Var}\{\theta\}$ and $\mathrm{Corr}\{\alpha,\theta\}=IC$: the alpha is a rescaled version of the
residual return, with dispersion cut by the factor $IC$, so extreme raw forecasts are pulled toward
consensus the more the weaker the forecaster's skill. With normal scores, quarterly volatility $9\%$ and
$IC=0.0833$, the refined forecast is $0.75$ times the score in percentage points.

## Combining several signals

With several information sources the law is additive in the squares of the information ratios: with
$BR_1$ stocks at skill $IC_1$ and $BR_2$ at skill $IC_2$,
$$IR^2 = BR_1\cdot IC_1^2 + BR_2\cdot IC_2^2.$$
Thus a manager following 200 stocks with semiannual forecasts ($BR=400$) at $IC=0.04$ has $IR=0.8$;
adding 100 stocks with two annual forecasts at $IC=0.03$ raises value added to $0.64+(0.03)^2\cdot200=0.82$
and the information ratio to $0.906=\sqrt{0.82}$. Additivity of the squares presumes independence: if
$\gamma$ is the correlation between two equally skilled sources, the combined skill is
$$IC(\text{com}) = IC\cdot\sqrt{\frac{2}{1+\gamma}},$$
so at $\gamma=0$ the sources add their value-adding capacity ($IC^2(\text{com})=2IC^2$) while as
$\gamma\to1$ the second source adds nothing. At the score level, for two raw forecasts $g,g'$ correlated
by $\rho_{gg'}$ the combined skill is
$$IC_{\text{combined}} = \sqrt{\frac{IC_g^2 + IC_{g'}^2 - 2\rho_{gg'}IC_g IC_{g'}}{1-\rho_{gg'}^2}},$$
which reduces to the sum of separate refined forecasts when uncorrelated and degenerates when perfectly
correlated. With three equally skilled signals — two strongly correlated with each other but
uncorrelated with the third — refinement halves the two correlated signals' $IC$s, effectively counting
the independent signal on a par with the sum of the two correlated ones.

## Cross-sectional scores and factor forecasts

For many stocks the basic formula still holds and the golden rule applies to each stock $n$,
$\phi_n=\omega_n\cdot IC\cdot z_{\mathrm{TS},n}$, with $z_{\mathrm{TS},n}$ the stock's time-series score.
In practice one has a single numeric forecast per stock at each instant, giving a *cross-sectional* score
$z_{\mathrm{CS},n}$ rather than the required time-series score, and how to refine it depends critically on
how the signal's time-series volatility varies across stocks. If the signal's time-series volatility is
identical across stocks, the two scores coincide and $\alpha_n=\omega_n\cdot IC\cdot z_{\mathrm{CS},n}$
(multiply by volatility); if instead the signal volatilities are proportional to the stock volatilities,
then
$$\phi_n = IC\cdot c_g\cdot z_{\mathrm{CS},n},$$
with $c_g$ a stock-independent constant, so refined forecasts are proportional to the cross-sectional
scores and *independent* of volatility — and multiplying by volatility would then be wrong. Empirically,
five of six equity signals (dividend discount model, estimate change, estimate revision, relative
strength, residual reversal) show a strong positive relation between signal time-series volatility and
stock residual volatility (do not rescale by volatility); only sector momentum, whose signal volatility
does not track stock volatility, is better with scores multiplied by volatility.

A further structure is the factor model: if a signal $g_1$ forecasts factor $b_1$ and $\rho_{1j}$ is the
correlation between $b_1$ and $b_j$, the basic formula gives $E\{b_j\mid g_1\}=IC\cdot\rho_{1j}\cdot
\omega_j\cdot z_1$, so when forecasting $E\{b_1\mid g_1\}\neq0$ it is inconsistent to set the other
factors' forecasts to zero. Empirically, using information on $b_1$ to also bet on $b_j$ improves
performance, and the squared information ratio of betting on all factors roughly equals the sum of the
component squared information ratios — ideas developed by Black and Litterman in the international
allocation context. The operational construction of portfolios from these alphas is the subject of
[quantitative selection and construction](./quantitative-selection-and-construction.md).

## Why the return forecast does not disturb the risk estimate

A striking result justifies decoupling alpha estimation from covariance estimation: return forecasts have
negligible effect on risk forecasts. The temptation to replace a historical correlation (say $0.95$
between two indices) with a negative one because one is forecast to do well and the other poorly is wrong
— it confuses conditional mean (how research affects expected return) with conditional covariance (how it
affects variances and covariances). What little effect exists depends only on skill, not on the forecast:
from the variance formula,
$$\sigma_{\mathrm{POST}} = \sigma_{\mathrm{PRIOR}}\cdot\sqrt{1-IC^2},$$
so with $\sigma_{\mathrm{PRIOR}}=18\%$ annual the post-forecast volatility is $17.98\%$ at $IC=0.05$,
$17.91\%$ at $0.10$, $17.80\%$ at $0.15$, $17.43\%$ at $0.25$ — minimal at reasonable $IC$ (0 to $0.15$),
material only at unrealistic skill ($5.62\%$ at $IC=0.95$, $0\%$ at $IC=1$). Risk measures the uncertainty
about return: a skilled forecaster reduces the *amount* of uncertainty, but its magnitude is the same
whatever the particular forecast. Likewise the revised correlation
$\rho_{ML}^{*}=\rho_{ML}\cdot(1-IC_M IC_L)/\sqrt{(1-IC_M^2)(1-IC_L^2)}$ is unchanged when $IC_M=IC_L$ and
barely moves for $IC$ in $[0,0.15]$. A short-horizon return forecaster can therefore ignore the impact of
forecasts on volatility and correlation estimates.

## Uncertainty in the information coefficient and alpha compression

The whole derivation assumes $IC$ is known. When the $IC$ is itself uncertain, at equal estimated $IC$
one should weight more heavily the signal whose $IC$ is estimated more precisely. Refining a signal by
regression $\theta(t)=a+b\cdot g(t)+\epsilon_\theta(t)$ with a prior $\hat b=0$, the Bayesian methodology
(following Connor) gives, since the regression $R^2$ should equal $IC^2$ and be very small,
$$b' \approx \left(\frac{1}{1+\dfrac{1}{T\cdot IC^2}}\right)\cdot b,$$
with $T$ the number of months observed — a shrinkage of the estimate toward zero that is stronger with
fewer periods or lower $IC$, and stays near the naïve estimate with more data or high skill. The
compression is significant even for good, long-observed signals: at $IC=0.05$ the factor is $0.13$ after
60 months and $0.23$ after 120. Since with $IC\ll1$ the $IC$ estimation error dominates, it is reasonable
to apply the Bayesian compression directly to the $IC$: the more uncertain the estimated information
coefficient, the more the $IC$ is compressed toward zero. With several signals, substituting marginal
$R^2$ for the total puts a premium on parsimony — a new signal with small marginal explanatory power is
compressed substantially.
