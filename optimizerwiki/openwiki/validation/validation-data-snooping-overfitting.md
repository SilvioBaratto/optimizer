---
type: "Reference"
title: "Validation, Data Snooping and Overfitting"
openwiki_generated: true
sources:
  - id: openwiki-source-d98dc5b53e0ab64ce91bde09
    resource: repo://docs/10_validation_data_snooping_and_overfitting.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---


# Validation, Data Snooping and Overfitting

This chapter establishes why a strategy's performance measured on historical data is a biased estimate
of its future performance, and quantifies the discount to apply. With enough trials one always finds a
signal correlated by pure chance; the chapter formalizes this bias through the multiple-testing problem
and derives Harvey-Liu's haircut Sharpe ratio, Bailey-López de Prado's deflated Sharpe ratio, and
Bailey-Borwein-López de Prado-Zhu's probability of backtest overfitting. Two validation criteria follow:
a cautiously conducted out-of-sample test, and an economic rationale independent of the statistical
evidence. It closes the loop on the signals that produce the alphas of
[from signal to alpha](../signals/signal-to-alpha-fundamental-law.md) and the portfolios built in
[quantitative selection and construction](../signals/quantitative-selection-and-construction.md).

## The backtest paradox: why the past deceives

Portfolio construction has so far been treated as optimization from estimated moments; the question the
whole exercise turns on remains: will a strategy that would have worked on past data also work in the
future? The answer is not obvious. William H. Miller's Legg Mason Value Trust beat the market every year
from 1991 to 2005, an unmatched streak, then fell nearly 65% in 2007–2008 — about twice the market —
handing back essentially all the accumulated advantage, raising the question of whether the pre-2007
record was skill or luck. It is not isolated: Forbes' Honor Roll of top funds, tracked over 19 years,
returned 11.2% annually against the market index's 13.1%, and the funds' superior past performance did
not predict their future ability to beat the market.

A backtest is a historical simulation of how a strategy would have behaved in the past — a powerful and
necessary research tool, but easily manipulated. The structural reason it deceives is that random
samples contain patterns: a systematic search over a large space of strategies will eventually identify
one that profits from the random configuration in which the data happened to fall, and optimizing
parameters to maximize backtest performance fits the strategy to those coincidences — backtest
overfitting. Ten tosses of a fair coin can by chance give $\{+,+,+,+,+,-,-,-,-,-\}$; concluding the
optimal rule is "expect heads on the first five, tails on the last five" is a rule designed to profit
from a past-only pattern, with zero predictive power on the future however well it appeared to work.

Competition among managers keeps the signal-to-noise ratio in financial series low, which raises the
probability of "discovering" a chance configuration rather than a real signal; hypothesis testing must
therefore use large samples, unlike sciences where the signal dominates the noise (Newton needed no
statistical tests for gravitation). The situation is worsened by memory effects: variables that build up
strong tension revert to equilibrium violently, "undoing" prior patterns rather than merely "diluting"
them. Because overfitting tends to identify rules that would have profited from the most extreme random
patterns, and such extreme patterns must be undone under memory, overfitting leads not only to zero
expected gain but to the maximization of out-of-sample loss — which may explain why so many systematic
funds underperform their advertising.

## Data snooping and selection bias

The size effect — that low-capitalization (or high book-to-market) stocks show positive alphas — was
documented by Banz in 1981; at the time the evidence did not convince, since financial economists had
combed the data for positive-alpha stocks and, because of estimation error, it is always possible to
find some, and to find something they have in common. Many attributed Banz's result to a data-snooping
bias: given enough characteristics, one will always be correlated by pure chance with the estimation
error of mean returns. The same holds for Jegadeesh and Titman's momentum. Data snooping is thus
intrinsic to empirical research on equity returns, and many well-established anomalies — certain
technical trading rules, calendar effects — are overturned once data-mining biases are accounted for.

The bias has a root more general than the single backtest. Researchers running multiple tests on the
same data tend to publish only those passing a significance test, hiding the rest, exposing the investor
to a biased sample of results. This selection bias takes many forms: analysts not reporting the full
extent of their experiments (the "file drawer effect"), journals publishing only "positive" outcomes
("publication bias"), indices tracking only surviving hedge funds ("survivorship bias"), managers
publishing only histories of so-far-profitable strategies ("self-selection bias" or "backfilling"). All
withhold critical information from the decision-maker, yielding a Type I error probability far larger
than expected.

Recall a statistical test's structure. A Type I error, probability $\alpha$ (the significance level), is
choosing a strategy that should have been rejected (a false positive); a Type II error, probability
$\beta$, is rejecting a strategy that should have been chosen (a false negative). Standard practice fixes
the Type I probability low (e.g. $\alpha=5\%$) while maximizing power $1-\beta$. But a 5% false-positive
probability holds only when the test is applied exactly once: applied to the same data many times —
potentially millions or billions in modern quantitative research — false positives become almost
certain. Selecting the best strategy from many exposes one to the "winner's curse" and an inflated
Sharpe ratio, so out-of-sample performance disappoints (regression toward the mean). The problem's
severity has led some to paraphrase Ioannidis that "most claimed research findings in financial
economics are likely false."

## The multiple-testing problem

Testing more than one hypothesis makes false rejections of the null more likely. Methods split into two
categories: those controlling the family-wise error rate and those controlling the false-discovery rate.
Testing $M$ hypotheses with p-values, of which $R$ are rejected (the discoveries, true and false, the
null being no skill) and $N_f$ the number of false discoveries, the family-wise error rate is the
probability of at least one false discovery,
$$\mathrm{FWER} = \mathrm{Pr}(N_f \geq 1),$$
while with the false discovery proportion $\mathrm{FDP}=N_f/R$ (and $0$ when $R=0$), the false discovery
rate is $\mathrm{FDR}=E[\mathrm{FDP}]$. Both generalize the single-test Type I probability; FDR
procedures let false discoveries grow with the number of tests and are thus more permissive than FWER
procedures.

Harvey and Liu, following Harvey, Liu and Zhu, give three p-value adjustments on the ordered p-values
$p_{(1)}\leq\dots\leq p_{(M)}$. Bonferroni multiplies each by the number of tests,
$$p_{(i)}^{\text{Bonferroni}} = \min[M\, p_{(i)}, 1],$$
Holm uses the ordered sequence,
$$p_{(i)}^{\text{Holm}} = \min\Big[\max_{j \leq i}\{(M-j+1)\, p_{(j)}\}, 1\Big],$$
both controlling FWER, with Bonferroni the stricter since $p_{(i)}^{\text{Holm}}\leq
p_{(i)}^{\text{Bonferroni}}$. The Benjamini-Hochberg-Yekutieli (BHY) method controls FDR sequentially
from the largest p-value,
$$p_{(i)}^{\text{BHY}} = \min\Big[p_{(i+1)}^{\text{BHY}}, \frac{M \cdot c(M)}{i} p_{(i)}\Big], \quad p_{(M)}^{\text{BHY}}=p_{(M)},$$
with $c(M)=\sum_{j=1}^{M}1/j$, valid under arbitrary dependence among the test statistics. Bonferroni and
Holm eliminate all false discoveries regardless of the number of tests — appropriate for a space mission
where one component's failure is catastrophic — but managers can accept false discoveries growing with
the number of tests, so for finance Harvey and Liu recommend controlling the *rate* rather than the
absolute number, i.e. BHY.

## The Sharpe ratio as a t-statistic and its haircut

The link between multiple testing and portfolio evaluation runs through the Sharpe ratio. For a
zero-cost strategy's return sample $(r_1,\dots,r_T)$ (e.g. long-short $r_t=R_t^L-R_t^S$) with sample mean
$\hat\mu$ and standard deviation $\hat\sigma$, the t-statistic for the null of zero mean return is
$$t\text{-statistic} = \frac{\hat{\mu}}{\hat{\sigma}/\sqrt{T}},$$
following a $t$ distribution with $T-1$ degrees of freedom under i.i.d. normal returns; since
$\widehat{SR}=\hat\mu/\hat\sigma$, one has $\widehat{SR}=t\text{-ratio}/\sqrt{T}$. At fixed $T$ a higher
Sharpe ratio means a higher t-statistic and lower p-value, justifying its use as an attractiveness
measure; for monthly inputs the annual Sharpe is $\sqrt{12}\,\hat\mu/\hat\sigma$.

Transforming the Sharpe ratio into a t-statistic and then a p-value, a single test gives
$p^S=\mathrm{Pr}(|r|>\widehat{SR}\sqrt{T})$ with $r$ a $t$ variable of $T-1$ degrees of freedom
(two-sided, since a long-short strategy profits from a positive or negative mean). If the researcher
tried $N$ strategies and presents the best, assuming $N$ independent tests the probability of a maximum
t-statistic at least as large is
$$p^M = 1 - (1 - p^S)^N.$$
For $N=1$, $p^M=p^S$; but with $N=10$ and $p^S=0.05$, $p^M=0.401$ — about a 40% chance of finding by
chance a strategy with an equally high t-statistic. Equating a single test's p-value to $p^M$ defines the
haircut Sharpe ratio $\widehat{HSR}$ via $p^M=\mathrm{Pr}(|r|>\widehat{HSR}\sqrt{T})$, so
$\widehat{HSR}<\widehat{SR}$, with proportional haircut $hc=(\widehat{SR}-\widehat{HSR})/\widehat{SR}$.
On twenty years of monthly returns ($T=240$) an annual Sharpe of $0.75$ gives $p^S=0.0008$; with $N=200$
tests $p^M=0.15$, implying an adjusted annual Sharpe of $0.32$ — about a 60% haircut.

The common practice of discounting the reported Sharpe by 50% is only a rule of thumb and a serious
error: the multiple-testing discount is nonlinear. High Sharpe ratios are penalized only moderately while
marginal ones are penalized heavily — which makes economic sense, since very high Sharpe ratios are
likely real discoveries while marginal strategies are the most likely false positives. Across three
strategies (E/P, momentum and Frazzini-Pedersen betting-against-beta), with $N=50$ the discount is nearly
50% for the least profitable E/P strategy (annual Sharpe $0.43$) but only $7.9\%$ for the most profitable
BAB (annual Sharpe $0.78$). In general, annual Sharpe ratios below $0.4$ are almost always discounted by
more than 50%, sometimes far more, while those above $1.0$ are discounted at most 25% — the 50% rule is
too lenient for small Sharpes and too harsh for large ones. Inverting the problem, with 240 observations
and 10% annual volatility the minimum monthly mean return for a single test is $0.365\%$ (about $4.4\%$
annual) but rises under BHY to $0.616\%$ (about $7.4\%$ annual).

## The deflated Sharpe ratio

A second approach corrects the Sharpe ratio directly for the whole context of trials. Given $N$
independent backtests of a strategy class, each with estimated Sharpe $\widehat{SR}_n$ having mean
$E[\{\widehat{SR}_n\}]$ and variance $V[\{\widehat{SR}_n\}]$, the expected maximum over $N$ independent
trials is approximated, via extreme value theory, as
$$E[\max\{\widehat{SR}_n\}] \approx E[\{\widehat{SR}_n\}] + \sqrt{V[\{\widehat{SR}_n\}]}\left((1-\gamma)\,Z^{-1}\!\left[1-\tfrac{1}{N}\right] + \gamma\,Z^{-1}\!\left[1-\tfrac{1}{N\,e}\right]\right),$$
with $\gamma\approx0.5772$ the Euler-Mascheroni constant, $Z$ the standard-normal CDF and $e$ Euler's
number. The consequence is stark: the expected maximum Sharpe grows with the number of independent
trials $N$ even in the total absence of skill ($E[\{\widehat{SR}_n\}]=0$, $V[\{\widehat{SR}_n\}]>0$), so
finding good backtests as more candidates are examined is a pure consequence of random behavior.

On this basis Bailey and López de Prado build the deflated Sharpe ratio (DSR), leaning on the
probabilistic Sharpe ratio (PSR) — the probability the true Sharpe exceeds a threshold, accounting for
sample length and the first four moments so as to correct inflation from short, non-normal samples. The
DSR is a PSR whose rejection threshold is the expected maximum under $H_0:SR=0$,
$$\widehat{DSR} = \widehat{PSR}(\widehat{SR}_0) = Z\!\left[\frac{(\widehat{SR} - \widehat{SR}_0)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3\,\widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\,\widehat{SR}^2}}\right],$$
with $\widehat{SR}_0$ the expected-maximum threshold, $T$ the series length, $\hat\gamma_3$ skewness and
$\hat\gamma_4$ kurtosis of the selected strategy's returns. Where the standard Sharpe uses only two
estimates, the DSR discounts using five extra variables: non-normality ($\hat\gamma_3,\hat\gamma_4$),
length $T$, the variance of tested Sharpe ratios $V[\{\widehat{SR}_n\}]$, and the number of independent
trials $N$.

A numerical example clarifies it. A strategist mining seasonal Treasury patterns finds many
configurations at annualized Sharpe $2$, one at $2.5$ over five years of daily data. Told to declare
$N=100$ independent trials, backtest variance $V[\{\widehat{SR}_n\}]=1/2$, length $T=1250$ and moments
$\hat\gamma_3=-3$, $\hat\gamma_4=10$, the threshold is $\widehat{SR}_0\approx0.1132$ (non-annualized) and
$$\widehat{DSR} \approx 0.9004 < 0.95,$$
so only a 90% probability the true Sharpe exceeds zero — under 95%, the investor declines. With normal
returns the investor would have accepted up to $N=88$ trials, showing it is critical to jointly account
for selection bias and non-normality. Establishing $N$ matters: the relevant count is *independent*
trials, not total trials $M$; with $\rho$ the average off-diagonal correlation of the $M\times M$
correlation matrix (bounded in $(-1/(M-1),1]$), the implied number of independent trials is
$$\bar{N} = \hat{\rho} + (1-\hat{\rho})M,$$
so $\bar N\to1$ as $\rho\to1$ and $\bar N\to M$ as $\rho\to0$; correlation reduces the effective number of
independent tests. The role $E[\max\{SR_n\}]$ plays here is played by Harvey and Liu's adjusted p-value
threshold — the methods are complementary and the DSR can be computed on their threshold too.

## The probability of backtest overfitting

A third approach, due to Bailey, Borwein, López de Prado and Zhu, evaluates the *selection process*
directly: what is the probability the in-sample best configuration is overfit? For overfitting to occur,
the maximum in-sample (IS) configuration must systematically underperform the others out-of-sample (OOS),
usually because the "optimal" IS strategy is so tied to training-set noise that further optimization is
useless or harmful. Over the rank space $\Omega$ of the $N!$ permutations of $(1,\dots,N)$, the process
overfits if the IS-optimal strategy has an expected OOS rank below the median. With $\bar r_n$ the OOS
rank and $\Omega_n^*=\{f\in\Omega\mid f_n=N\}$ the event that strategy $n$ is IS-best, the probability of
backtest overfitting is
$$PBO = \sum_{n=1}^{N} \mathrm{Prob}\big[\bar{r}_n < N/2 \mid r \in \Omega_n^*\big]\,\mathrm{Prob}[r \in \Omega_n^*],$$
the probability the IS-optimal strategy lands below the OOS median of all strategies. Here "in-sample"
means the subset used to select the optimum among the $N$ alternatives, not the period the underlying
model is estimated on — overfitting concerns the selection process, so it can be defined model-free and
non-parametrically.

To estimate the PBO the authors propose combinatorially symmetric cross-validation (CSCV). Collect the
$N$ trials' performance series into a $(T\times N)$ matrix $M$, partition its rows into an even number $S$
of disjoint equal submatrices, and form all combinations taken in groups of $S/2$, numbering
$\binom{S}{S/2}$. For each combination the training set $J$ unites its $S/2$ submatrices, the test set
$\bar J$ is the complement; the IS-best strategy $n^*$ on $J$ has OOS rank on $\bar J$, whose relative
rank $\bar\omega_c=\bar r_{n^*}^c/(N+1)\in(0,1)$ becomes the logit $\lambda_c=\ln(\bar\omega_c/(1-\bar
\omega_c))$. High logits indicate IS-OOS consistency, hence low overfitting; over the relative-frequency
distribution $f(\lambda)$ the PBO is the mass left of zero,
$$\phi = \int_{-\infty}^{0} f(\lambda)\,d\lambda,$$
the frequency with which the IS-optimal strategy underperforms the OOS median. Per Neyman-Pearson one
can reject models whose estimated PBO exceeds $0.05$. With $S=16$ one generates $12{,}870$ logits, and for
a four-year daily backtest $S=16$ equals quarterly partitions, preserving serial-correlation structure —
a reasonable value in most cases.

The framework also reads off performance degradation directly. Recording $(R_{n^*},\bar R_{n^*})$, the IS
and OOS Sharpe of the selected strategy per combination — $R_{n^*}$ the IS maximum but $\bar R_{n^*}$ not
necessarily the OOS maximum ($\bar R_{n^*}<\max\{\bar R\}$ in general) — the regression
$\bar R_{n^*}^c=\alpha+\beta R_{n^*}^c+\varepsilon^c$ usually gives a *negative* slope $\beta$: overfit
backtests minimize future performance. A useful statistic is the proportion of combinations with negative
performance $\mathrm{Prob}[\bar R_{n^*}^c<0]$, which can be high even when $\phi\approx0$ (poor OOS
performance for reasons other than overfitting). In one example, with all IS Sharpe ratios positive
between $1$ and $3$, about 78% of OOS Sharpes are negative and the PBO is 74% — a high IS Sharpe alone
says nothing about representativeness — while a contrasting real case shows ~3% loss probability and a PBO
of $0.04\%$. A further check is stochastic dominance: if the OOS Sharpe distribution of IS-optimal
configurations does not dominate, even to second order, that of all configurations, it is a clear sign of
overfitting.

## Implications for the model builder

These tools converge on one point: the most important and almost always missing piece of information in
published backtests is the number of trials attempted. Without it a backtest cannot be assessed; a
backtest for which the researcher has not controlled the search extent is worthless however excellent the
reported performance, and investors and referees should demand it. The usual disclaimer that "past
returns do not guarantee future results" is too lenient when, under joint selection bias and overfitting,
adverse outcomes are in fact very likely.

The first validation criterion is an out-of-sample test conducted with full awareness of its limits. The
holdout method — splitting into an in-sample set to estimate the model and an out-of-sample set to
validate, possibly repeated as k-fold cross-validation — does not prevent overfitting: it assesses
generality as if a single trial had occurred, ignoring the rise in false positives with the number of
trials. Applied enough times (twenty for 95% confidence) false positives are expected, not improbable.
Validation techniques guard against the hypothesis suggested by the data (Type III errors) but do not
control backtest overfitting, and the holdout is inadequate for small samples with high estimation
variance — the chosen holdout section might be exactly the one that refutes a valid strategy or supports
an invalid one, and different holdouts reach different conclusions.

Even a true OOS test has intrinsic limits. First, it may not really be out-of-sample: a researcher who
tries a strategy, sees it fail, revises and retries is not truly OOS, and the difference is hard for an
outsider to see. Second, like any statistical test it holds only probabilistically — success can be luck,
IS or OOS. Third, since the researcher has already lived through the data period, no true OOS on
historical data exists, especially with economic variables. There is also a Type I/Type II trade-off in
splitting: withholding OOS data raises the chance of missing true discoveries in the shorter in-sample.
A true OOS test on previously unpublished data or uncorrelated markets remains the cleanest way to assess
viability, but for most strategies it is unavailable.

The second criterion concerns how many trials to attempt and with what justification. Multiple testing
should not be abused: experiments must be carefully planned, and investment theory — not computing power —
should motivate which are worth running. Each extra trial irrevocably raises the false-positive
probability; the optimal-stopping theory (the "secretary problem" or $1/e$ law) samples a random $1/e$
(~37%) fraction of the theoretically justifiable configurations, then continues drawing one at a time
until one beats all the earlier ones. A single-test p-value is meaningful only when strongly motivated by
economic theory and directly proxied, not when hundreds or thousands of strategies were explored and only
the most profitable presented. A strategy's economic foundation figures among the evaluation criteria
alongside Sharpe ratio, significance level and drawdown, so empirical evidence of a signal gains
credibility only with an explanation of the risk it remunerates — the same logic by which size,
book-to-market and momentum alphas read as premia for risks uncaptured by the CAPM rather than
risk-free opportunities. The economic rationale must therefore be independent of the statistical evidence
the strategy generates, not deduced after the fact from the backtest it is meant to validate.
