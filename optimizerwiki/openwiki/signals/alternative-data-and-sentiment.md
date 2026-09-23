---
type: concept
title: "Signals from Alternative Data and Sentiment"
description: How textual, attention, and behavioral data — financial news, lexical dictionaries, aggregate sentiment indices, Google searches, social-media moods, and large-scale news analytics — are turned into quantitative signals for the cross-section and time-series of stock returns, organized by the empirically testable dichotomy between the sentiment hypothesis (transient price pressure followed by reversal) and the information hypothesis (permanent impact), and aggregated across many noisy, correlated signals with machine learning.
tags: [sentiment, alternative-data, textual-analysis, dictionaries, attention, social-media, news-analytics, machine-learning, risk-premium, cross-section]
sources:
  - id: openwiki-source-31f4781ab865f58fdc50ebf7
    resource: repo://docs/25_signals_from_alternative_data_and_sentiment.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Signals from Alternative Data and Sentiment

This chapter shows how textual, attention, and behavioral information — financial news, lexical
dictionaries, aggregate sentiment indices, Google searches, moods extracted from social media, and
large-scale news analytics — is transformed into quantitative signals for the cross-section and
time-series of stock returns. The organizing thread is the empirically testable distinction between
the **sentiment hypothesis**, which predicts transient price pressure followed by reversal, and the
**information hypothesis**, which predicts permanent impact. It closes with the aggregation of many
correlated, noisy signals via machine learning for risk-premium measurement — the same estimation
problem taken up for macro predictors in
[macro signals and risk premia](../regimes/macro-signals-and-risk-premia.md) and for accounting
characteristics in [fundamental stock selection](./fundamental-stock-selection.md).

## Sentiment, information, and the two testable hypotheses

In classical asset-pricing theory investor sentiment plays no role in the cross-section of prices,
realized returns, or expected returns. For sentiment to matter, two ingredients must hold jointly:
investors whose demand is driven by beliefs not justified by the facts, and limits to arbitrage that
prevent rational operators from promptly correcting mispricings. Consistently, sentiment can be
defined as a belief about future cash flows and investment risks not justified by the facts at hand.

This framing supplies a diagnostic criterion that organizes the whole empirical program: the
sentiment theory predicts that short-horizon returns induced by a signal are reversed in the long
run, whereas the information theory predicts they persist indefinitely. A signal's temporal
signature — pressure followed by reversal, versus lasting impact — becomes the test of whether a
textual, attention, or behavioral datum carries noise or news. Because sentiment acts where
arbitrage is costly and valuation is hard, its effects are expected to concentrate in speculative,
hard-to-value, hard-to-arbitrage stocks, and attention signals are expected to produce temporary
pressure on stocks held by individual investors. The operative question is not merely "does the text
predict returns?" but "through which channel, with which sign, and for how long?".

## From text to number: content analysis and dictionaries

Turning text into a number requires classifying words into categories and counting them. The
methodological principle is that content analysis stands or falls by its categories: the choice of
vocabulary determines the quality of the signal. One approach applies a predefined psycholinguistic
dictionary — the media-pessimism measure applies principal component analysis to the counts of the
77 categories of the General Inquirer (Harvard IV-4 dictionary) in a daily Wall Street Journal
column, and the first factor, interpreted as pessimism, loads on the Negative, Weak, Fail, and Fall
categories, summarizing an entire daily text into a single indicator projected onto the dimension of
maximum joint variance of the counts.

A dictionary built for a different domain, however, systematically misclassifies in a financial
setting: across a large set of 10-K annual reports, 73.8% of the occurrences of words classified as
negative by the Harvard H4N list are not actually negative in the financial context — terms such as
"tax," "cost," "capital," "board," "liability," "foreign," and "vice" express neutral or technical
content. Hence the construction of domain-specific dictionaries: a financial negative-tone word list
(Fin-Neg, 2,337 words) plus auxiliary lists for positive tone, uncertainty, litigious language, and
strong/weak modality. Category specificity precedes any statistical sophistication.

A second choice concerns term weighting in the bag-of-words model. Proportional weighting assigns
each word a weight equal to its relative frequency, but the frequency distribution is highly unequal
and a few common words dominate the counts (Zipf's law). The tf.idf weighting corrects this by
combining a term's document frequency with its rarity in the corpus,
$$w_{i,j} = \begin{cases} \dfrac{1+\log(tf_{i,j})}{1+\log(a_j)}\cdot \log\dfrac{N}{df_i} & \text{if } tf_{i,j}\ge 1\\[2mm] 0 & \text{otherwise,} \end{cases}$$
where $tf_{i,j}$ is the count of term $i$ in document $j$, $a_j$ the average term frequency in $j$,
$N$ the number of documents, and $df_i$ the number of documents containing term $i$. The first factor
normalizes term frequency for document length; the idf factor damps ubiquitous words and boosts rare,
specific ones. The measure is not neutral to the outcome.

## Media sentiment and short-horizon predictability

The media-pessimism measure is placed in a vector autoregression that, estimated by least squares,
is equivalent to a Granger-causality test,
$$Dow_t = \alpha_1 + \sum_{i=1}^{5}\beta_{1i}\,Dow_{t-i} + \sum_{i=1}^{5}\gamma_{1i}\,Pess_{t-i} + \sum_{i=1}^{5}\delta_{1i}\,Vlm_{t-i} + \lambda_1\,Exog_{t-1} + \varepsilon_{1t},$$
with $Dow_t$ the index return, $Pess_t$ pessimism, $Vlm_t$ volume, and $Exog$ calendar and
macroeconomic controls. A one-standard-deviation rise in pessimism is followed by downward pressure
of about $-8.1$ basis points on the next day's return; over days 2–5 a reversal of about $6.8$ basis
points occurs, so the cumulative effect (about $-1.3$ basis points) is not significantly different
from zero. The signal produces temporary price pressure that fully unwinds — the signature predicted
by the sentiment theory and by noise-trading models.

Two collateral results reinforce the interpretation. Unusually high or low pessimism (large absolute
deviations) precedes elevated trading volume, consistent with departures from the normal tone
activating trading; and pessimism predicts a negative return on the size factor SMB, indicating the
downward pressure concentrates on small-capitalization stocks typically held by individual
investors. A trading strategy on the signal earns roughly 7% gross annually, but transaction costs
erode profitability, making the signal more relevant as scientific evidence than as an operational
opportunity.

At the single-stock level, the proportion of financial negative-tone words in a 10-K predicts
negative abnormal returns around the filing date, with a $t$-statistic of $-2.64$, net of controls
for size, book-to-market, momentum, and other characteristics. Weighting is decisive here: the
Fin-Neg list is significant already under proportional weighting, whereas the Harvard H4N list is
significant only under tf.idf weighting. Tone measurement thus depends jointly on the dictionary and
the weighting scheme, and the signal's sign — bad news anticipating low returns — matches the
index-level result.

## The aggregate sentiment index and the cross-section

A complementary top-down approach builds a sentiment index by aggregating market indicators rather
than text. Six variables are combined: the closed-end-fund discount (CEFD), share turnover (TURN),
the number of IPOs (NIPO), the first-day IPO return (RIPO), the equity share in new issues (S), and
the dividend premium (PDND). Extracting the first principal component, with each indicator lagged
where it leads the others, yields
$$SENT_t = -0.241\,CEFD_t + 0.242\,TURN_{t-1} + 0.253\,NIPO_t + 0.257\,RIPO_{t-1} + 0.112\,S_t - 0.283\,PDND_{t-1},$$
which explains about 49% of the variance; orthogonalizing the proxies first against industrial
production, consumption, employment, and recessions gives the index $SENT^{\perp}$, purged of the
macroeconomic component, which explains about 53%.

The cross-sectional effects are formalized by a conditional-characteristics model,
$$E_{t-1}[R_{it}] = a + a_1\,T_{t-1} + b_1'\,x_{i,t-1} + b_2'\,T_{t-1}\,x_{i,t-1},$$
where $T_{t-1}$ is beginning-of-period sentiment and $x_{i,t-1}$ a vector of firm characteristics; the
interaction term $b_2'$ captures the dependence of the cross-sectional premia on sentiment. The
evidence — from both double sorts and predictive regressions of long-short portfolios — shows that
when sentiment is high, future returns are relatively low for young, small, volatile, unprofitable,
non-dividend-paying, high-growth, and distressed stocks (hard to value and to arbitrage), and that
these configurations attenuate or reverse when sentiment is low. For several growth and distress
variables the conditional relation is U-shaped: stocks with extreme, not intermediate, values react
to sentiment. Consistently, the size effect emerges only in low-sentiment periods, and sentiment's
predictive coefficient on SMB is about $-0.40$ per one-standard-deviation point of the index. Two
checks rule out compensation for classical systematic risk: $SENT^{\perp}$ is orthogonal to
macroeconomic conditions, and estimating time-varying market betas gives a composite coefficient that
would reconcile the results with a conditional CAPM but, when significant, typically has the wrong
sign. Sentiment betas rise with a stock's degree of speculativeness, and highly speculative stocks
earn low future returns when sentiment is high.

## Attention as a direct signal: Google searches

Sentiment measures tone; attention measures instead how much a stock is the object of investor
interest, and is a distinct channel. Google's Search Volume Index (SVI) provides a direct, revealed
measure of retail-investor attention, an alternative to indirect proxies such as turnover, news
coverage, or advertising. The measure is de-seasonalized and de-trended by constructing the abnormal
search volume,
$$ASVI_t = \log(SVI_t) - \log\!\big[\text{Med}(SVI_{t-1},\dots,SVI_{t-8})\big],$$
the difference between the log of current volume and the log of its median over the prior eight
weeks. That attention is conceptually different from sentiment is confirmed by the correlation
between SVI and news-extracted sentiment being only on the order of 1.4–2.3%.

The economic mechanism is attention-induced price pressure: individual investors, facing an enormous
number of stocks, tend to buy those that catch their attention, generating temporary net buying and
hence a transient overvaluation that unwinds over time. A one-standard-deviation rise in ASVI is
associated with a positive aggregate return of over 30 basis points in the following two weeks (about
18.7 in the first week and 14.9 in the second), followed by a reversal within the year; the effect is
stronger for small stocks and those more heavily traded by retail investors. For IPOs, high
pre-placement attention precedes a higher first-day return (on the order of 16.98% versus 10.90%) and
long-run underperformance, consistent with a temporary attention-fueled overvaluation. Retail
attention thus sits, like media pessimism, on the transient pole of the sentiment-information
dichotomy, but its measure isolates a specific, direct behavioral channel.

## Collective moods from social media

Social media allow measuring not only tonal polarity but more articulated collective emotional
states. From 9.85 million tweets collected during 2008, two public-mood measures are extracted: a
bipolar positive/negative indicator (OpinionFinder) and a multidimensional profile (GPOMS) that
decomposes mood into six dimensions — Calm, Alert, Sure, Vital, Kind, Happy. Each mood series is
standardized against a local window,
$$Z_{X_t} = \frac{X_t - \bar{x}(X_{t\pm k})}{\sigma(X_{t\pm k})},$$
with $\bar{x}$ and $\sigma$ the mean and standard deviation over a window of width $\pm k$ around
date $t$.

Predictive power is assessed with Granger-causality tests, comparing an index-return model on its own
lags,
$$D_t = \alpha + \sum_{i=1}^{n}\beta_i\,D_{t-i} + \varepsilon_t,$$
with one that adds the lags of a mood dimension,
$$D_t = \alpha + \sum_{i=1}^{n}\beta_i\,D_{t-i} + \sum_{i=1}^{n}\gamma_i\,X_{t-i} + \varepsilon_t.$$
The result is selective: the Calm dimension alone Granger-causes the Dow Jones return, at lags 2–6
days with significance below 5%, while neither the bipolar indicator nor the other GPOMS dimensions
show predictive power. Feeding Calm into a self-organizing fuzzy neural network (SOFNN) raises
directional-prediction accuracy for the index to about 87.6% with a reduction in mean percentage
error. Once again it is not generic polarity that matters but a specific emotional dimension — the
choice of what to measure is integral to the signal.

## Large-scale news analytics: news versus sentiment

At industrial scale the sentiment-information dichotomy can be tested directly. An archive of 900,754
Reuters articles linked to firm identifiers over 2003–2010 is scored by a sentiment engine — a
three-layer neural network trained on a sample of 3,000 triple-annotated articles — returning, per
story, the probabilities of being positive, negative, or neutral. The central result concerns the
aggregation horizon. Measured day by day, predictability lasts only one or two days (post-publication
returns of about 0.17% at day 1 and 0.04% at day 2), in line with earlier literature; aggregated
weekly, predictability extends to 13 weeks (a full quarter), with a decile spread in the announcement
week of 3.75%. The longer duration indicates the network extracts permanent information not yet
impounded in prices, not mere transient sentiment or a liquidity effect.

An essential methodological step is controlling for the mere existence of news. Firms with news have
returns different from firms without news: like the dog that does not bark, whether an article is
published carries information. Neutral news outperforms the absence of news, and this positive premium
contradicts the adage that no news is good news. To isolate the effect of tone, a Fama-MacBeth
cross-sectional regression is used,
$$r_{i,t} = \alpha_{k,t} + \gamma_{k,t}\,\mathbf{1}_{news,\,t-k} + \beta_{k,t}\,Positive_{i,t-k} + \delta_{k,t}\,Negative_{i,t-k} + \epsilon_{i,t},$$
where $\mathbf{1}_{news}$ isolates the publication effect, distinguishing it from the positive- and
negative-sentiment effects at each lag $k$ from 0 to 13.

A sharp asymmetry emerges. Positive sentiment has a strong contemporaneous effect but is impounded
within about a week, with lagged coefficients near zero; negative sentiment instead predicts low
returns for a whole quarter, with significant coefficients at lags 1–6 and lag 10. This slow reaction
to bad news is consistent with short-selling constraints and the idea that bad news travels slowly.
Much of the delayed reaction concentrates around subsequent earnings announcements: announcement-week
spreads reach 5.57% ($t=2.7$) and remain positive afterward (1.83%, $t=2.1$), consistent with
post-earnings-announcement drift. Earnings announcements thus act as a price-discovery channel for
information not immediately incorporated at publication.

## Aggregating the signals: machine learning in asset pricing

The signals described — textual tone, attention, mood, news — are numerous, noisy, and mutually
correlated: exactly the setting where machine-learning methods offer an advantage. Risk-premium
measurement is at bottom a prediction problem, since the premium is the conditional expectation of the
future excess return,
$$r_{i,t+1} = E_t(r_{i,t+1}) + \epsilon_{i,t+1}, \qquad E_t(r_{i,t+1}) = g^{\star}(z_{i,t}),$$
where $g^{\star}(\cdot)$ is a flexible function of the predictor vector $z_{i,t}$, held identical over
time and across stocks. The predictor set is large: 94 firm characteristics interacted with eight
macroeconomic variables plus a constant, and 74 industry indicators, for a total of
$94\times(8+1)+74 = 920$ baseline signals across nearly 30,000 stocks over 1957–2016.

On this set the canon of methods is compared. OLS with all predictors fails, producing an
out-of-sample monthly $R^2$ of $-3.46\%$ from overparameterization; regularization saves it. The
elastic net penalizes the objective with
$$\phi(\theta;\lambda,\rho) = \lambda(1-\rho)\sum_{j=1}^{P}|\theta_j| + \tfrac{1}{2}\lambda\rho\sum_{j=1}^{P}\theta_j^2,$$
combining selection (lasso, $\rho=0$) and shrinkage (ridge, $\rho=1$); dimension reduction via
principal components (PCR) and partial least squares (PLS) brings the out-of-sample $R^2$ to about
0.26–0.27%. Mean squared loss can be robustified against the heavy tails of returns with the Huber
function,
$$H(x;\xi) = \begin{cases} x^2 & |x|\le \xi\\ 2\xi|x| - \xi^2 & |x| > \xi, \end{cases}$$
with hyper-parameters chosen on a validation sample distinct from the estimation and test samples.
Performance is measured with an out-of-sample $R^2$ referred to a value of zero, not the historical
mean of returns,
$$R^2_{oos} = 1 - \frac{\sum_{(i,t)}(r_{i,t+1} - \hat{r}_{i,t+1})^2}{\sum_{(i,t)} r_{i,t+1}^2}.$$

The nonlinear methods dominate. Regression trees, random forests, and boosting achieve monthly $R^2$
on the order of 0.33–0.34%, and neural networks reach 0.33–0.40%, with a multi-layer architecture
defined recursively by $x_k^{(l)} = \text{ReLU}\big(x^{(l-1)\prime}\theta_k^{(l-1)}\big)$. Notably,
"shallow" learning prevails: the three-layer network (NN3) is best, while deeper networks do not
improve, given the data scarcity and low signal-to-noise ratio of the problem. That the gains are not
an artifact of micro-cap illiquidity is confirmed by nonlinear methods excelling also on large, liquid
stocks. The source of the advantage is the ability to capture nonlinearities and, above all,
interactions among predictors: short-term reversal is linear among small stocks but concave among
large ones, and the size effect is stronger when aggregate valuations are low (high book-to-market)
and equity issuance is low.

Economically the gains are large. Using the relation between predictability and the Sharpe ratio of an
active investor,
$$SR^{\ast} = \sqrt{\frac{SR^2 + R^2}{1-R^2}},$$
a market-timing strategy on the S&P 500 based on the network's forecasts raises the annualized Sharpe
ratio from 0.51 to 0.77. A value-weighted decile long-short spread sorted on the network's forecasts
achieves an annualized Sharpe ratio of 1.35 (2.45 equal-weighted), against 0.61 and 0.83 for the
linear three-characteristic benchmark, with significant alphas relative to a momentum-augmented
five-factor model. Finally, all methods agree on a narrow set of dominant signals: price trends
(short-term reversal, momentum, industry momentum), followed by liquidity measures (size, dollar
volume, Amihud illiquidity, bid-ask spread) and volatility measures (total and idiosyncratic
volatility, beta). Machine learning thus resolves the multiple-comparisons and collinearity problem
of the "factor zoo," but provides measurement, not mechanism.

## Measure, arbitrage, and permanence: an overview

The signals examined lie along the continuum drawn by the sentiment-information dichotomy. Media
pessimism and retail attention revealed by Google searches produce temporary price pressure followed
by reversal (the sentiment signature); large-scale news processed by a neural network and the
persistent underreaction to bad news show lasting impact (the information signature). The same
conceptual frame — developed from limits to arbitrage and noise-trading models — explains why
sentiment effects concentrate on hard-to-value, hard-to-arbitrage stocks and why the reaction to bad
news is slowed by short-selling constraints.

The recurring methodological theme is that the measure is not neutral: categories determine the
signal, dictionaries must be domain-specific, term weighting alters significance, controlling for
attention and for publication is necessary to avoid confusing the news effect with the tone effect,
and orthogonalization against macroeconomic variables isolates the properly sentiment component. Each
of these choices is as much a part of the signal as the model that uses it. Machine learning is the
aggregation frontier — absorbing hundreds of heterogeneous, correlated signals into a single
risk-premium forecasting model with substantial, horizon-robust statistical and economic gains — yet
even this synthesis confirms that the dominant signals remain price trends, liquidity, and volatility,
within which sentiment, attention, and text enter as incremental, conditioning information. Because
market efficiency makes return variation dominated by unforecastable news, alternative data and
aggregation improve the measurement of the premium but leave open the question of underlying economic
mechanisms.
