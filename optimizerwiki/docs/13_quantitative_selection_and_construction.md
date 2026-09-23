---
title: "Quantitative Selection and Construction"
chapter: 13
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-10
---

> [!abstract] Summary
> The chapter reconstructs the pipeline that leads from a fundamental valuation signal to a standardized alpha and, finally, to a list of selected securities. It shows how a dividend discount model produces an alpha equal to the difference between forecast growth and implied growth, how the raw signal should be rebased cross-sectionally and shrunk toward the prior as a function of skill, and how the information coefficient is measured by constructing score-portfolios and long/short factor portfolios. It finally establishes Alpha Analysis — scaling consistent with the IC, trimming of extremes, and neutralization of unwanted exposures — and the screen as direct portfolio construction by ranking on alphas.

## From Fundamental Valuation to Alpha

The theoretically correct valuation of stocks, though useful for options and futures, has never been made operational for equities: quantitative methods are therefore used on an ad hoc basis, with only a loose connection to theory [GrinoldKahn2000]. Two principles should be kept in mind. The first is the *principle of humility*: the market may be right and we may be wrong. The second is the fundamental law of active management, according to which one need not be right much more than 50 percent of the time to add value [Grinold1989]; this law is taken as given and applied in what follows (see [[12 Dal segnale all'alpha - information coefficient e la legge fondamentale]]).

Corporate finance theory sets the rules of the game for an acceptable valuation model. Modigliani and Miller showed that dividend policy affects only the timing of the cash flows received by the shareholder — an arrangement of the "pay me now or pay me later" type — and not the total value of the payments, and that financing policy does not alter the total value of the firm [ModiglianiMiller1961]. Economic value arises from operating activity: equity value can thus be decomposed into operating value and financial value. A corollary is that any valuation scheme that places some magic in dividends or debt is dangerous.

The standard model is the *dividend discount model*, which John Burr Williams anticipated by emphasizing the role of dividends as a determinant of value [Williams1938]. Under certainty, if $p(0)$ is the price, $i_F$ the risk-free rate, and $d(t)$ the dividend paid at time $t$,

$$p(0)=\frac{d(1)}{(1+i_F)}+\frac{d(2)}{(1+i_F)^2}+\cdots+\frac{d(t)}{(1+i_F)^t}+\cdots$$

The constant-growth, or Gordon–Shapiro, model assumes $d(t)=d(1)\cdot(1+g)^{t-1}$ and leads to the compact form [GordonShapiro1956]

$$p(0)=\frac{d(1)}{y-g},$$

where $y$ is the expected discount rate of the dividends. Decomposing the return into dividend yield and capital appreciation and taking expected values gives

$$i_F+f=\frac{d}{p}+g=y,$$

with $f$ the expected excess return. A simple growth model shows that the expected rate of appreciation equals the growth in earnings per share, with $g=(1-\kappa)\cdot\rho$, where $\kappa$ is the payout ratio and $\rho$ the return on reinvested capital.

For $N$ securities the relation becomes $d_n/p_n+g_n=i_F+f_n=y_n$. Since the expected excess return includes the consensus and the alpha, $f_n=\beta_n\cdot f_B+\alpha_n$, substituting yields a model of alpha in terms of yield, risk (beta), and growth:

$$\alpha_n=\left(\frac{d_n}{p_n}-i_F\right)+\left(g_n-\beta_n\cdot f_B\right).$$

This formula expresses the *Golden Rule* of the dividend discount model — "$g$ in, $g$ out": every percentage point of growth input adds a point to the alpha, and the alphas are only as good as the growth estimates that generate them. Setting $\alpha_n=0$ yields the *implied growth rate*, the growth that leaves the security fairly priced:

$$g^*_n=(i_F+\beta_n\cdot f_B)-\frac{d_n}{p_n}.$$

Combining the two expressions gives the central relation of the conversion:

$$\alpha_n=g_n-g^*_n,$$

the dividend discount model's alpha is the difference between forecast growth and the growth implied by the market price.

## Making Growth Rates Realistic: Rebasing and Shrinkage

Implied growth rates are useful in several ways: they provide a rational yardstick for what growth should be, help identify systematic biases in analysts' estimates — within sectors or across the entire universe — and identify companies whose prices reflect unrealistic growth prospects [GrinoldKahn2000]. A recurring difficulty is that growth estimates tend to be too high, since Wall Street research, which drives the consensus, is interested in selling securities, and bullish prospects help sell them.

A first, direct approach treats the problem in three steps: (1) group securities into sectors; (2) compute the implied growth rate for each security in the sector; (3) modify the growth forecasts so that they have the same mean and standard deviation as the sector's implied rates. This amounts to a linear transformation $g'_n=a+b\cdot g_n$, with $a$ and $b$ chosen so that $g'$ reproduces the cross-sectional mean and standard deviation of the implied rates $g^*$:

$$g'_n=\text{Mean}\{g^*\}+\left(\frac{\text{Std}\{g^*\}}{\text{Std}\{g\}}\right)\cdot\left(g_n-\text{Mean}\{g\}\right).$$

Mean and standard deviation are computed over the securities in the sector: the revised estimate is the sector's average implied growth plus a term proportional to the deviation of the initial forecast. A difficulty with this scheme is that it can wipe out *sector timing* information: since the implied rates assume zero alphas, the procedure pushes sector alphas toward zero, which is harmless only if there is no sector-timing skill.

A second approach explicitly incorporates the investor's skill in forecasting growth. Starting not from the sector average but from the *individual security's* implied rate, the basic result of linear forecasting is applied:

$$g'_n=g^*_n+c\cdot(g_n-g^*_n).$$

The estimate starts from the security's implied growth rate and departs from it as a function of the comparison between the initial forecast and the implied rate. The constant $c$ depends on the investor's skill, measured by the correlation between forecast and realized growth; in the absence of skill $c=0$ is set, and the revised growth coincides with the implied one (see [[12 Dal segnale all'alpha - information coefficient e la legge fondamentale]] for the forecasting rule). A third, more conventional and elaborate approach is the three-stage dividend discount model, which interpolates between a short-term growth $g_{IN}$ and a long-term growth $g_{EQ}$; in spirit it too is a way of adjusting $g_{IN}$ toward $g_{EQ}$, and it is not magic: it does not turn bad growth estimates into good ones, and the Golden Rule continues to hold.

## Converting a Valuation into Alpha

To use a valuation model in active management, the information it contains must be converted into forecasts of exceptional return. There are two standard modes, which differ in the assumption about the horizon over which the mispricing will dissolve [GrinoldKahn2000].

In the *internal rate of return* mode, the stream of dividends and the market price are used to solve for the rate $y_n$ that equates the dividends to the price. The alphas are obtained in two steps: the rates are first aggregated to estimate the benchmark's expected excess return,

$$f_B=y_B-i_F=\sum_n y_n\cdot h_{B,n}-i_F,$$

and the internal rates are then converted into expected residual returns,

$$\alpha_n=y_n-(i_F+\beta_n\cdot f_B).$$

The implicit assumption is that the mispricing *persists*: after a year the correct discount rate will still be $y_n$ and not the "fair" rate $i_F+\beta_n f_B$, and the benefit continues to be collected.

In the *net present value* mode, the fair rate is assumed to be $y_n=i_F+\beta_n f_B$ and the fair price is solved for, then compared with the market price to measure the degree of overvaluation or undervaluation. After adjusting $f_B$ so that the aggregate fair value equals the market value, and assuming the mispricing disappears within a year, the alpha is

$$\alpha^*_n=\left[\frac{p_n(\text{model})-p_n(\text{market})}{p_n(\text{market})}\right]\cdot(1+i_F+\beta_n\cdot f_B),$$

approximable by the percentage price error alone, $\alpha^*_n\approx[p_n(\text{model})-p_n(\text{market})]/p_n(\text{market})$.

An alternative to projecting dividends is *comparative valuation*, which prices a company by comparison with similar companies as a function of their current attributes. The *clean surplus* accounting identity, $b(t)=b(t-1)+e(t)-d(t)$, allows earnings to be decomposed into a required part and an exceptional part that gradually dies out, and leads to expressing price as a linear combination of expected earnings and book value [Ohlson1989]. Applying the idea to a group of similar companies, common coefficients are sought,

$$p_n(0)=c_1\cdot b_n(0)+c_2\cdot e_n(1)+\epsilon_n,$$

and the error $\epsilon_n$ identifies the mispricings. In general, comparative valuation estimates the relation "market price = theoretical price + error," from which the forecast of exceptional return

$$\alpha=-\frac{\text{error}}{\text{market price}}=\frac{\text{theoretical price}-\text{market price}}{\text{market price}},$$

presuming that the theoretical price is more accurate than the market price and that the security will converge to it over the alpha horizon. The reading is one of arbitrage: companies are split into overvalued and undervalued ones with identical attributes — same sales, same debt, same earnings — and two identical "meta-companies" quoting at different prices constitute an arbitrage opportunity. If the model omits an important attribute — for example brand value — the pricing errors may measure that missing factor rather than a genuine mispricing.

An extension is *returns-based analysis*, which attacks the problem directly by modeling residual returns as a function of attributes, or excess returns with risk-control factors in an APT-type form,

$$r_n(t)=\sum_k X_{n,k}(t)\cdot b_k(t)+u_n(t),$$

where the exposures $X_{n,k}$ include both the attributes and the risk-control factors. In a GLS regression the factor return $b_k(t)$ is the return of a *factor portfolio* with unit exposure to factor $k$, zero exposure to the others, and minimum risk. Since least-squares estimation is sensitive to outliers, a rule of thumb is to cap the outliers of the $X_{n,k}(t)$ within $\pm 3$ standard deviations from the mean, to avoid the explanatory power resting on one or two suspect observations.

## The Predictor as Signal Plus Noise: the Information Coefficient

In the context of active management, information is, in essence, an *alpha predictor*: any set of data one asks whether it helps predict alphas [GrinoldKahn2000]. In general every predictor is made of *signal plus noise*: the signal is tied to future returns, the noise masks it and makes the task difficult; random numbers contain no signal, only noise, and information analysis is the effort to measure the signal-to-noise ratio.

A predictor covers multiple periods and multiple securities. The datum for a single security can be simple — $+1$ for securities on the buy list and $-1$ for those on the sell list — or a precise alpha, such as $2{,}15$ percent for one security and $-3{,}72$ percent for another; other predictors are *scores*, for example groupings into categories or a ranking along some dimension. One can start from alphas and produce a ranking, or start from a ranking and produce scores. Information is classified along four dimensions: primary or processed; judgmental or unbiased; ordinal or cardinal; historical, contemporaneous, or forward-looking. The distinction between ordinal and cardinal is essential in what follows: with *ordinal* data securities are classified into groups with an indication of order of preference — the buy/sell/hold classification is an example; with *cardinal* data a number is associated with each security, with a meaning attached to the magnitude of the value.

The *information coefficient* is the correlation between the data and the realized alphas. If the datum is all noise and no signal, the IC is $0$; if it is all signal and no noise, it is $1$; if there is a perverse relationship between the datum and the subsequent alpha, the IC can be negative. In every case the IC lies between $+1$ and $-1$. The IC is a critical ingredient in determining the information ratio, according to the fundamental law, and a critical input for refining and combining signals. In what follows the known result that structures alpha as $\alpha=\text{volatility}\times IC\times\text{score}$ is assumed, with the score having zero mean and unit standard deviation (see [[12 Dal segnale all'alpha - information coefficient e la legge fondamentale]]).

## From Information to Portfolios: Score-Portfolios and Factor Portfolios

Information analysis is a two-step process: first, transform the forecasts into portfolios; second, evaluate the performance of those portfolios [GrinoldKahn2000]. Since forecasts exist for every period, a portfolio is generated for every period, and the chosen procedure depends on the type of forecast. As an illustration, the book-to-price ratio in the United States is used, under the assumption that this ratio contains information about future returns — that high book-to-price securities outperform low book-to-price ones — consistent with the evidence on "value" measures [RosenbergReidLanstein1985].

With *buy and sell recommendations*, the buy group and the sell group are equal-weighted (or capitalization-weighted). With *scores*, a portfolio is built for each score, equal-weighting or capitalization-weighting within each category: ranking securities by book-to-price and assigning 5 to the top fifth, 4 to the next, and so on down to 1 for the bottom fifth, securities are divided into *quintiles*. With true alphas, securities are ranked by alpha and grouped into quintiles (or deciles, or halves), equal-weighting within each group: this yields the *score-portfolios*.

More refined procedures build *factor portfolios* that isolate the signal by controlling for factors. With any numerical score, a factor portfolio is built that bets on the forecast without betting on the market: it consists of a long portfolio and a short one, of equal value and equal beta, such that the long has a unit bet on the forecast relative to the short — for book-to-price, a ratio one standard deviation higher than the short's — and is built to replicate the short as closely as possible otherwise. A more elaborate form matches long and short on a set of prespecified control variables: sector, industry, small-capitalization exposures, and — by controlling for beta — market risk exposure. Since they set up a controlled experiment that isolates the information contained in the data net of the other market factors, these procedures are the recommended approach for analyzing the information content of any numerical score. The comparison between quintile analysis and factor-portfolio analysis shows that different construction approaches, starting from the same underlying information, lead to different observed performance and different estimates of information content.

## Evaluating the Portfolios: t-statistic, Information Ratio, Information Coefficient

The simplest form of performance analysis is to compute the cumulative returns of the portfolios and of the benchmark and plot them, supplementing this with means and standard deviations [GrinoldKahn2000]. More sophisticated analyses investigate statistical significance, value added, and skill, measured by the *t*-statistic, the information ratio, and the information coefficient — quantities that are all interrelated.

One starts from a regression of the portfolio's excess returns against the benchmark's, to separate the component tied to the benchmark from the one not tied to it:

$$r(t)=\alpha+\beta\cdot r_B(t)+\epsilon(t).$$

The regression estimates the portfolio's alpha and beta and assesses, via the *t*-statistic, whether the alpha differs significantly from zero. The statistic is

$$t\text{-stat}=\frac{\alpha}{\text{SE}(\alpha)},$$

the ratio of the estimated alpha to the standard error of the estimate; assuming normally distributed alphas, if the *t*-statistic exceeds $2$ the probability that the returns are the result of pure luck is below 5 percent.

The information ratio is the single best statistic for capturing the potential value added by active management. The *t*-statistic and the information ratio are closely related: if returns are observed over $T$ years,

$$IR\approx\frac{t\text{-stat}}{\sqrt{T}},$$

a relation that becomes more exact the greater the number of observations. The close mathematical relation should not, however, obscure the fundamental distinction: the *t*-statistic measures the statistical significance of the return, while the information ratio captures its risk-return trade-off and the manager's value added.

The third statistic is the information coefficient, the correlation between forecast alphas and realized alphas; in the context of information analysis, the correlation between the data and the realized alphas. The IC is linked to the information ratio by the fundamental law of active management [Grinold1989]:

$$IR\approx IC\cdot\sqrt{BR},$$

where $BR$ is the *breadth*, the number of independent bets per year allowed by the information. In practice $BR$ is harder to measure than either the information ratio or the information coefficient: not all the information items generated are independent.

## Alpha Analysis: Scaling, Trimming, and Neutralization

Portfolio construction receives as input the current portfolio, the alphas, the covariance estimates, the transaction costs, and an active risk aversion; of these inputs, the alphas are often unreasonable and subject to hidden biases [GrinoldKahn2000]. Many implementation schemes — limits on active positions, on turnover, on sectors — are in part a safeguard against poor-quality research: any construction procedure, however sophisticated, can be replicated by first refining the alphas and then using a simple unconstrained mean-variance optimization. A set of constraints leading to active positions $h^{*}_{PA}$, active risk $\psi^{*}_P$, and information ratio $IR$ indeed corresponds to the modified alphas

$$\alpha'=\left(\frac{IR}{\psi^{*}_P}\right)\cdot V\cdot h^{*}_{PA},\qquad \lambda'_A=\frac{IR}{2\cdot\psi^{*}_P}.$$

*Alpha Analysis* refines the alphas to make them consistent with the manager's beliefs and objectives, explicitly linking the refinement to the desired properties of the resulting portfolio.

**Scaling.** Alphas have a natural structure, $\alpha=\text{volatility}\cdot IC\cdot\text{score}$, with the score having mean $0$ and standard deviation $1$: hence a natural scale $\text{Std}\{\alpha\}\sim\text{volatility}\cdot IC$. An information coefficient of $0{,}05$ and a typical residual risk of 30 percent lead to an alpha scale of $1{,}5$ percent; in that case the average alpha is $0$, with about two-thirds of securities between $-1{,}5$ and $+1{,}5$ percent and about 5 percent with alphas beyond $\pm 3$ percent. The scale depends on the manager's IC: if the input alphas do not have the correct scale, they must be rescaled. The phenomenon is quantifiable: in one example the original alphas have a standard deviation of $2{,}00$ percent and those modified by the constraints have $0{,}57$ percent, implying that the constraints have effectively reduced the IC by 62 percent — a significant compression, better made explicit than hidden beneath the optimizer's constraints.

**Trimming (winsorization).** The second refinement caps extreme values. Very large alphas, positive or negative, can have undue influence: securities with alphas exceeding, say, three times the scale in magnitude are examined closely. Some may depend on questionable data and should be ignored (set to zero); the others, apparently genuine, should be capped at three times the scale. A more extreme approach forces the alphas into a normal distribution with zero benchmark alpha and the required scale; it is extreme because it typically uses only the ranking information and ignores the magnitude of the alphas, and after the transformation benchmark neutrality and scale must be re-verified.

**Neutralization.** Beyond scaling and trimming, unwanted biases or bets can be removed: this is *neutralization*, with implications for both the alphas and the portfolios. Benchmark neutrality means the benchmark has zero alpha; on the portfolio side, that the optimum will have beta $1$, with no bet on the benchmark. The simplest approach sets the benchmark's alpha to zero; in the same spirit the alphas can be made *cash-neutral*, so that they do not induce an active cash position. Benchmark neutrality is achieved by subtracting from each modified alpha the term $\beta_n\cdot\alpha_B$. For a group of securities $N_1$ with a forecast, defining the value-weighted fraction $H\{N_1\}=\sum_{n\in N_1}h_{B,n}$ and the average alpha $\alpha\{N_1\}=\sum_{n\in N_1}h_{B,n}\cdot\alpha_n/H\{N_1\}$, one sets

$$\alpha^*_n=\alpha_n-\alpha\{N_1\}\ \ (n\in N_1),\qquad \alpha^*_n=0\ \ (n\in N_0),$$

so that even the uncovered securities have a zero, and hence neutral, forecast. In the multifactor framework the manager identifies each dimension as either a source of risk or of value added: if unable to forecast a factor, the alphas should be neutralized with respect to it, leaving only the information about the forecastable factors and the security-specific information. To make the alphas *industry-neutral*, the (capitalization-weighted) average alpha of each industry is computed and subtracted from every alpha in that industry. Neutralization is not uniquely defined — unwanted active exposures can be hedged in many ways — and it is best to decide *a priori* how to neutralize, since the a priori approach works better than trying every possibility and choosing the best one ex post.

## Screens: from the List of Alphas to the Selected Securities

The *screen* is the most direct alternative to optimization. Portfolio construction techniques fall into four generic classes — screens, stratification, linear programming, and quadratic programming — united by the same criterion, maximizing value added net of transaction costs, $\alpha_P-\lambda_A\cdot\psi_P^2-TC$ [GrinoldKahn2000]. The screen implements this criterion with an essential recipe: (1) rank securities by alpha; (2) choose the top $N$ securities — for example the top 50; (3) equal-weight them (or weight them by capitalization).

The screen also serves for rebalancing. With alphas over a followed universe of securities, securities are split into three categories — for example the top 40, the next 60, and the remaining 100 — building a buy list, a hold list, and a sell list. Starting from the current portfolio, every security that is on the buy list but not in the portfolio is bought, and every security that is in the portfolio and on the sell list is sold; adjusting the three thresholds governs turnover.

The screen has several merits. It is simple and transparent, with a clear link between cause (list membership) and effect (portfolio membership); it is easy to automate; it is robust, because it depends only on the ranking — wild estimates of positive or negative alphas do not alter the outcome. The screen exploits the alphas by concentrating the portfolio in high-alpha securities, seeks risk control by including a sufficient number of securities and weighting them so as to avoid concentration in a single name, and limits transaction costs through careful choice of list size.

The screen, however, also has limitations. It ignores all the information contained in the alphas beyond the ranking; it offers no protection against biases in the alphas — if all the "widow-and-orphan" stocks ended up at the bottom of the ranking, the portfolio would include none of them; risk control is fragmentary, to the point that screen-built portfolios have proven considerably riskier than their managers imagined. The systematic comparison of construction techniques — for the same input alphas and ignoring transaction costs — shows that quadratic programming consistently achieves the highest ex post information ratios, while screening methods, which do not methodically control risk, are more erratic and can even produce negative returns in some periods [Muller1993]. Despite these limitations, the screen remains a widely used portfolio construction technique.

## References

- **[GordonShapiro1956]** Gordon, M. J. & Shapiro, E. (1956). Capital Equipment Analysis: The Required Rate of Profit. Management Science, 3(1), 102–110.
- **[Grinold1989]** Grinold, R. C. (1989). The Fundamental Law of Active Management. Journal of Portfolio Management, 15(3), 30–37.
- **[GrinoldKahn2000]** Grinold, R. C. & Kahn, R. N. (2000). Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk (2nd ed.). New York: McGraw-Hill.
- **[ModiglianiMiller1961]** Modigliani, F. & Miller, M. H. (1961). Dividend Policy, Growth, and the Valuation of Shares. Journal of Business, 34(4), 411–433.
- **[Muller1993]** Muller, P. (1993). Empirical Tests of Biases in Equity Portfolio Optimization. In S. A. Zenios (Ed.), Financial Optimization (pp. 80–98). Cambridge: Cambridge University Press.
- **[Ohlson1989]** Ohlson, J. A. (1989). Accounting Earnings, Book Value, and Dividends: The Theory of the Clean Surplus Equation (Part I). Columbia University working paper, January 1989.
- **[RosenbergReidLanstein1985]** Rosenberg, B., Reid, K. & Lanstein, R. (1985). Persuasive Evidence of Market Inefficiency. Journal of Portfolio Management, 11(3), 9–17.
- **[Williams1938]** Williams, J. B. (1938). The Theory of Investment Value. Cambridge, MA: Harvard University Press (reprint: Amsterdam, North-Holland, 1964).
