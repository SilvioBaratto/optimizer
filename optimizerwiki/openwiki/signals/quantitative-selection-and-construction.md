---
type: concept
title: "Quantitative Selection and Construction"
description: The pipeline from a fundamental valuation signal to a standardized alpha and finally to a selected list of securities — how a dividend discount model yields an alpha equal to forecast growth minus implied growth, how the raw signal is cross-sectionally rebased and shrunk toward the prior by skill, how the information coefficient is measured via score-portfolios and long/short factor portfolios, and the Alpha Analysis (IC-consistent scaling, extreme-value trimming, and neutralization of unwanted exposures) together with the screen as direct portfolio construction by sorting on alpha.
tags: [alpha, dividend-discount-model, implied-growth, shrinkage, information-coefficient, factor-portfolios, alpha-analysis, neutralization, screen, portfolio-construction]
sources:
  - id: openwiki-source-98cceec8b5297308bc20b2f4
    resource: repo://docs/13_quantitative_selection_and_construction.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Quantitative Selection and Construction

This chapter reconstructs the pipeline from a fundamental valuation signal to a standardized alpha and,
finally, to a list of selected securities. It shows how a dividend discount model produces an alpha
equal to the difference between forecast growth and implied growth, how the raw signal is
cross-sectionally rebased and shrunk toward the prior by skill, and how the information coefficient is
measured by building score-portfolios and long/short factor portfolios. It closes with the Alpha
Analysis — IC-consistent scaling, trimming of extremes, and neutralization of unwanted exposures — and
the screen as direct portfolio construction by sorting on alphas. It connects signal research
downstream to the [signal-to-alpha fundamental law](./signal-to-alpha-fundamental-law.md), to the
optimizer's [constraints and metaheuristics](../optimization/constraints-and-metaheuristics.md), and to
the full [conditional-forecasts-to-weights](../workflows/from-conditional-forecasts-to-weights.md)
workflow.

## From fundamental valuation to alpha

Theoretically correct equity valuation, though useful for options and futures, has never been made
operational for equity: one resorts to ad hoc quantitative methods with only a vague link to theory.
Two principles must be kept in mind. The first is the *principle of humility*: the market may be right
and we may be wrong. The second is the fundamental law of active management, whereby one need not be
right much more than 50 percent of the time to add value; this law is taken as given and applied
throughout.

Corporate-finance theory sets the rules for an acceptable valuation model. Modigliani and Miller showed
that dividend policy affects only the timing of cash flows received by the shareholder — a "pay me now
or pay me later" arrangement — not the total value of payments, and that financing policy does not
alter the total firm value. Economic value arises from operations: equity value can thus be decomposed
into operating value and financial value, and any valuation scheme placing magic in dividends or debt
is dangerous.

The standard model is the *dividend discount model*. Under certainty, with $p(0)$ the price, $i_F$ the
risk-free rate, and $d(t)$ the dividend paid at time $t$,
$$p(0)=\frac{d(1)}{(1+i_F)}+\frac{d(2)}{(1+i_F)^2}+\cdots+\frac{d(t)}{(1+i_F)^t}+\cdots$$
The constant-growth (Gordon–Shapiro) model assumes $d(t)=d(1)\cdot(1+g)^{t-1}$ and gives
$$p(0)=\frac{d(1)}{y-g},$$
where $y$ is the expected discount rate of dividends. Decomposing return into dividend yield and
capital appreciation and taking expectations,
$$i_F+f=\frac{d}{p}+g=y,$$
with $f$ the expected excess return, and a simple growth model gives $g=(1-\kappa)\cdot\rho$, with
$\kappa$ the payout ratio and $\rho$ the return on reinvested capital.

For $N$ stocks the relation becomes $d_n/p_n+g_n=i_F+f_n=y_n$. Since the expected excess return
includes consensus and alpha, $f_n=\beta_n\cdot f_B+\alpha_n$, substituting yields a model of alpha in
terms of yield, risk (beta), and growth,
$$\alpha_n=\left(\frac{d_n}{p_n}-i_F\right)+\left(g_n-\beta_n\cdot f_B\right).$$
This is the *Golden Rule* of the dividend discount model — "$g$ in, $g$ out": each percentage point of
growth in adds one point to alpha, and alphas are worth only as much as the growth estimates that
generate them. Setting $\alpha_n=0$ gives the *implied growth rate*, the growth that fairly prices the
stock,
$$g^*_n=(i_F+\beta_n\cdot f_B)-\frac{d_n}{p_n}.$$
Combining the two expressions gives the central conversion relation,
$$\alpha_n=g_n-g^*_n,$$
the dividend-discount-model alpha is the difference between forecast growth and the growth implied by
the market price.

## Making growth rates realistic: rebasing and shrinkage

Implied growth rates are useful in several ways: they give a rational yardstick for what growth should
be, help detect systematic biases in analyst estimates (within sectors or over the whole universe), and
identify firms whose prices reflect unrealistic growth prospects. A recurring difficulty is that growth
estimates tend to be too high, since Wall Street research driving the consensus is interested in selling
stocks and bullish prospects help sell them.

A first, direct approach proceeds in three steps: (1) group stocks into sectors; (2) compute the
implied growth rate for each stock in the sector; (3) modify growth forecasts to have the same mean and
standard deviation as the implied rates in the sector. This is a linear transformation $g'_n=a+b\cdot
g_n$ with $a,b$ chosen so $g'$ reproduces the cross-sectional mean and standard deviation of the implied
rates $g^*$,
$$g'_n=\text{Mean}\{g^*\}+\left(\frac{\text{Std}\{g^*\}}{\text{Std}\{g\}}\right)\cdot\left(g_n-\text{Mean}\{g\}\right).$$
The revised estimate is the sector's average implied growth plus a term proportional to the initial
forecast's deviation. A difficulty is that this can erase *sector-timing* information: because implied
rates assume zero alphas, the procedure pushes sector alphas toward zero, harmless only if one has no
sector-timing skill.

A second approach explicitly incorporates the investor's skill at forecasting growth. Starting not from
the sector mean but from the *single stock's* implied rate, it applies the basic linear-forecasting
result,
$$g'_n=g^*_n+c\cdot(g_n-g^*_n).$$
The estimate starts from the stock's implied growth and departs from it according to the comparison
between initial forecast and implied. The constant $c$ depends on the investor's skill, measured by the
correlation between forecast and realized growth; with no skill $c=0$ and the revised growth coincides
with the implied. A third, more conventional approach is the three-stage dividend discount model, which
interpolates between a short-run growth $g_{IN}$ and a long-run growth $g_{EQ}$; in spirit it too
adjusts $g_{IN}$ toward $g_{EQ}$, and is no magic: it does not turn bad growth estimates into good
ones, and the Golden Rule still holds.

## Converting a valuation into alpha

To use a valuation model in active management, its information must be converted into forecasts of
exceptional return. Two standard modes differ in the assumption about the horizon over which the
mispricing dissolves.

In the *internal-rate-of-return* mode, the dividend stream and market price solve for the rate $y_n$
equating dividends to price. Alphas come in two steps: first aggregating rates to estimate the
benchmark excess return,
$$f_B=y_B-i_F=\sum_n y_n\cdot h_{B,n}-i_F,$$
then converting internal rates into expected residual returns,
$$\alpha_n=y_n-(i_F+\beta_n\cdot f_B).$$
The implicit assumption is that the mispricing *persists*: after a year the correct discount rate is
still $y_n$, not the "fair" $i_F+\beta_n f_B$, and the benefit keeps accruing.

In the *net-present-value* mode, the fair rate $y_n=i_F+\beta_n f_B$ is assumed and one solves for the
fair price, then compares it to the market price to measure over- or under-valuation. After adjusting
$f_B$ so aggregate fair value equals market value, and assuming the price error vanishes in a year, the
alpha is
$$\alpha^*_n=\left[\frac{p_n(\text{model})-p_n(\text{market})}{p_n(\text{market})}\right]\cdot(1+i_F+\beta_n\cdot f_B),$$
approximable by the percentage price error alone,
$\alpha^*_n\approx[p_n(\text{model})-p_n(\text{market})]/p_n(\text{market})$.

An alternative to projecting dividends is *comparative valuation*, pricing a firm by comparison with
similar firms via their current attributes. The accounting *clean-surplus* equation
$b(t)=b(t-1)+e(t)-d(t)$ decomposes earnings into a required and an exceptional part that gradually dies
out, and expresses price as a linear combination of expected earnings and book value. Applied to a
group of similar firms, one seeks common coefficients,
$$p_n(0)=c_1\cdot b_n(0)+c_2\cdot e_n(1)+\epsilon_n,$$
and the error $\epsilon_n$ identifies mispricings. In general comparative valuation estimates "market
price = theoretical price + error," giving the exceptional-return forecast
$$\alpha=-\frac{\text{error}}{\text{market price}}=\frac{\text{theoretical price}-\text{market price}}{\text{market price}},$$
presuming the theoretical price is more accurate and the stock converges to it over the alpha horizon.
The reading is one of arbitrage: firms split into over- and under-valued with identical attributes
(same sales, debt, earnings) are two identical "meta-firms" trading at different prices, an arbitrage
opportunity. If the model omits an important attribute — say brand value — price errors may measure that
missing factor rather than a true mispricing.

An extension is *returns-based analysis*, modeling residual returns as a function of attributes, or
excess returns with risk-control factors in an APT-like form,
$$r_n(t)=\sum_k X_{n,k}(t)\cdot b_k(t)+u_n(t),$$
where the exposures $X_{n,k}$ include both attributes and risk-control factors. In a GLS regression the
factor return $b_k(t)$ is the return of a *factor portfolio* with unit exposure to factor $k$, zero to
others, and minimum risk. Because least-squares estimation is sensitive to outliers, a practical rule
compresses the $X_{n,k}(t)$ outliers within $\pm 3$ standard deviations of the mean, so explanatory power
does not come from one or two suspect observations.

## The predictor as signal plus noise: the information coefficient

In active management, information is essentially an *alpha predictor*: any set of data for which one
asks whether it helps forecast alphas. Every predictor is *signal plus noise*: the signal is linked to
future returns, the noise masks it; random numbers contain no signal, only noise, and information
analysis is the effort to measure the signal-to-noise ratio.

A predictor spans multiple periods and stocks. The single-stock datum may be simple — $+1$ for buy-list
stocks and $-1$ for sell-list — or a precise alpha such as $2.15$ percent for one stock and $-3.72$
percent for another; other predictors are *scores*, such as category groupings or a ranking along some
dimension. One can start from alphas and produce a ranking, or start from a ranking and produce scores.
Information classifies along four dimensions: primary or processed; judgmental or unbiased; ordinal or
cardinal; historical, contemporaneous, or forecast. The ordinal/cardinal distinction is essential: with
*ordinal* data stocks are classified into ordered preference groups (buy/sell/hold), while with
*cardinal* data each stock gets a number whose magnitude carries meaning.

The *information coefficient* (IC) is the correlation between the data and realized alphas. If the datum
is all noise and no signal, the IC is $0$; all signal and no noise, $1$; a perverse relation gives a
negative IC; in all cases the IC lies between $+1$ and $-1$. The IC is a critical ingredient for the
information ratio under the fundamental law and a critical input for refining and combining signals.
The structural result used here is $\alpha=\text{volatility}\times IC\times\text{score}$, with the
score of zero mean and unit standard deviation.

## From information to portfolios: score-portfolios and factor portfolios

Information analysis is a two-step process: first, turn forecasts into portfolios; second, evaluate
those portfolios' performance. Since forecasts exist for every period, a portfolio is generated for
each period, and the chosen procedure depends on the forecast type. The illustration uses the US
book-to-price ratio, under the assumption it contains information about future returns — that high
book-to-price stocks outperform low — consistent with the evidence on value measures.

With *buy and sell recommendations* one equal-weights (or cap-weights) the buy group and the sell group.
With *scores* one builds a portfolio for each score, equal- or cap-weighting within each category:
sorting stocks by book-to-price and assigning 5 to the top fifth, 4 to the next, down to 1 to the bottom
fifth, splits them into *quintiles*. With true alphas one sorts by alpha and groups into quintiles (or
deciles, or halves), equal-weighting within each group: these are *score-portfolios*.

More refined procedures build *factor portfolios* that isolate the signal by controlling for factors.
With any numeric score one builds a factor portfolio that bets on the forecast without betting on the
market: a long and a short portfolio of equal value and equal beta such that the long has a unit bet on
the forecast relative to the short — for book-to-price, a ratio one standard deviation above the short's
— and is built to replicate the short as faithfully as possible. A more elaborate form matches long and
short on a set of prespecified control variables: sector, industry, small-cap exposure, and — by
controlling beta — market-risk exposure. Because they set up a controlled experiment isolating the
information in the data net of other market factors, these procedures are the recommended approach for
analyzing the information in any numeric score. Comparing quintile analysis and factor-portfolio
analysis shows that different construction approaches, from the same base information, lead to different
observed performance and different estimates of information content.

## Evaluating portfolios: t-statistic, information ratio, information coefficient

The simplest performance analysis computes and plots cumulative portfolio and benchmark returns,
supplemented with means and standard deviations. More sophisticated analyses probe statistical
significance, value added, and skill, measured by the *t*-statistic, information ratio, and information
coefficient — all related.

One starts from a regression of the portfolio's excess returns on the benchmark's, separating the
benchmark-linked from the non-benchmark-linked component,
$$r(t)=\alpha+\beta\cdot r_B(t)+\epsilon(t).$$
The regression estimates the portfolio's alpha and beta and assesses, via the *t*-statistic, whether the
alpha differs significantly from zero,
$$t\text{-stat}=\frac{\alpha}{\text{SE}(\alpha)},$$
the ratio of estimated alpha to its standard error; assuming normally distributed alphas, a
*t*-statistic above $2$ means the probability the returns are pure luck is below 5 percent.

The information ratio is the single best statistic capturing active management's value-added potential.
The *t*-statistic and information ratio are closely related: over $T$ years,
$$IR\approx\frac{t\text{-stat}}{\sqrt{T}},$$
increasingly exact with more observations. The close mathematical relation must not obscure the
distinction: the *t*-statistic measures statistical significance, the information ratio the
risk-return trade-off and the manager's value added.

The third statistic is the information coefficient, the correlation between forecast and realized
alphas (in information analysis, the correlation between the data and realized alphas), linked to the
information ratio by the fundamental law of active management,
$$IR\approx IC\cdot\sqrt{BR},$$
where $BR$ is the *breadth*, the number of independent bets per year the information allows. In practice
$BR$ is harder to measure than either the IR or the IC, since not all information items are independent.

## Alpha Analysis: scaling, trimming, and neutralization

Portfolio construction takes as inputs the current portfolio, the alphas, covariance estimates,
transaction costs, and an active-risk aversion; of these, the alphas are often unreasonable and subject
to hidden biases. Many implementation schemes — limits on active positions, turnover, sectors — are
partly a safeguard against poor-quality research: any construction procedure, however sophisticated, can
be replicated by first refining the alphas and then using a simple unconstrained mean-variance
optimization. A set of constraints leading to active positions $h^{*}_{PA}$, active risk $\psi^{*}_P$,
and information ratio $IR$ corresponds to the modified alphas
$$\alpha'=\left(\frac{IR}{\psi^{*}_P}\right)\cdot V\cdot h^{*}_{PA},\qquad \lambda'_A=\frac{IR}{2\cdot\psi^{*}_P}.$$
The *Alpha Analysis* refines alphas to make them consistent with the manager's beliefs and goals,
explicitly tying the refinement to the desired properties of the resulting portfolio.

**Scaling.** Alphas have a natural structure, $\alpha=\text{volatility}\cdot IC\cdot\text{score}$, with
the score of mean $0$ and standard deviation $1$, giving a natural scale
$\text{Std}\{\alpha\}\sim\text{volatility}\cdot IC$. An IC of $0.05$ and a typical residual risk of 30
percent give an alpha scale of $1.5$ percent; then the mean alpha is $0$, with about two-thirds of
stocks between $-1.5$ and $+1.5$ percent and about 5 percent beyond $\pm 3$ percent. The scale depends
on the manager's IC; alphas of the wrong scale must be rescaled. This is quantifiable: in an example the
original alphas have standard deviation $2.00$ percent and the constraint-modified ones $0.57$ percent,
implying the constraints effectively reduced the IC by 62 percent — a significant compression, better
made explicit than hidden under the optimizer's constraints.

**Trimming (winsorization).** The second refinement compresses extreme values. Very large alphas,
positive or negative, can have undue influence: stocks with alpha magnitude above, say, three times the
scale are examined closely. Some may rest on questionable data and are ignored (set to zero); others,
apparently genuine, are compressed to three times the scale. A more extreme approach forces alphas into
a normal distribution with zero benchmark alpha and the required scale; it is extreme because it
typically uses only ranking information and ignores alpha magnitude, and after the transformation
benchmark neutrality and scale must be re-verified.

**Neutralization.** Beyond scaling and trimming, one can remove unwanted biases or bets — *neutralization*,
with implications for both alphas and portfolios. Benchmark neutrality means the benchmark has zero
alpha; on the portfolio side, the optimum has beta $1$, with no benchmark bets. The simplest sets the
benchmark alpha to zero; in the same spirit alphas can be made *cash-neutral* so they induce no active
cash position. Benchmark neutrality is achieved by subtracting $\beta_n\cdot\alpha_B$ from each modified
alpha. For a group of forecast stocks $N_1$, defining the value-weighted fraction
$H\{N_1\}=\sum_{n\in N_1}h_{B,n}$ and mean alpha
$\alpha\{N_1\}=\sum_{n\in N_1}h_{B,n}\cdot\alpha_n/H\{N_1\}$, one sets
$$\alpha^*_n=\alpha_n-\alpha\{N_1\}\ \ (n\in N_1),\qquad \alpha^*_n=0\ \ (n\in N_0),$$
so uncovered stocks too get a neutral zero forecast. In a multifactor framework the manager identifies
each dimension as a source of risk or of value added: unable to forecast a factor, they should
neutralize alphas against it, leaving only forecastable-factor and stock-specific information. To make
alphas *industry-neutral* one computes each industry's cap-weighted mean alpha and subtracts it from
every alpha in that industry. Neutralization is not uniquely defined — unwanted active exposures can be
hedged in many ways — and it is better to decide *a priori* how to neutralize, since the a-priori
approach works better than trying all possibilities and choosing the best ex post.

## Screens: from an alpha list to selected securities

The *screen* is the most direct alternative to optimization. Construction techniques fall into four
generic classes — screens, stratification, linear programming, and quadratic programming — sharing the
same criterion, maximizing value added net of transaction costs, $\alpha_P-\lambda_A\cdot\psi_P^2-TC$.
The screen realizes this criterion with an essential recipe: (1) sort stocks by alpha; (2) pick the top
$N$ (say the top 50); (3) equal-weight (or cap-weight) them.

The screen also serves rebalancing. With alphas over a followed universe, stocks split into three
categories — say the top 40, the next 60, and the remaining 100 — forming a buy, hold, and sell list.
From the current portfolio one buys every stock on the buy list not held and sells every held stock on
the sell list; adjusting the three thresholds governs turnover.

The screen has several virtues. It is simple and transparent, with a clear link between cause (list
membership) and effect (portfolio membership); easy to automate; and robust, depending only on the
ranking — wild positive or negative alpha estimates do not alter its result. It exploits alphas by
concentrating the portfolio in high-alpha stocks, seeks risk control by including enough stocks and
weighting to avoid single-name concentration, and limits transaction costs through careful list sizing.

The screen also has limits. It ignores all information in the alphas beyond the ranking; it does not
protect against alpha biases — if all "buy-and-hold" stocks fell to the bottom of the ranking, the
portfolio would include none; and its risk control is piecemeal, to the point that screen-built
portfolios have proved considerably riskier than their managers imagined. The systematic comparison of
construction techniques — same input alphas, ignoring transaction costs — shows that quadratic
programming consistently attains the highest ex-post information ratios, while screening methods, which
do not methodically control risk, are more erratic and can even produce negative returns in some
periods. Despite these limits, the screen remains a very widely used construction technique.
