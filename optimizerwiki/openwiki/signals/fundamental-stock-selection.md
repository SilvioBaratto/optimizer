---
type: concept
title: "Fundamental Stock Selection"
description: Narrowing an investable universe from financial-statement information — a taxonomy of ratios (profitability, liquidity, leverage, operating efficiency, valuation), the role of multiples (P/E, price-to-book, EV/EBITDA) as value signals and the limits of comparing non-comparable firms — and the finding that size and value are cross-sectional characteristics that generate alpha against the CAPM, organized into self-financing, well-diversified factor portfolios culminating in the Fama-French-Carhart specification.
tags: [fundamentals, financial-statements, ratios, valuation-multiples, value, size, momentum, factor-portfolios, fama-french-carhart, cross-section]
sources:
  - id: openwiki-source-55c59f46dbd42abda7c6315e
    resource: repo://docs/09_security_selection_on_fundamentals.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Fundamental Stock Selection

This chapter shows how to narrow an investable universe to a set of securities starting from
financial-statement information: it builds a taxonomy of ratios — profitability, liquidity, leverage,
operating efficiency, valuation — and reads the role of multiples (P/E, price-to-book, EV/EBITDA) as
value signals, together with the limits of comparing non-comparable firms. It then establishes that
size and value are characteristics that order stocks in the cross-section, generating alpha relative
to the CAPM, and that such strategies organize into self-financing, well-diversified factor
portfolios, culminating in the Fama-French-Carhart specification — the empirical roots of the
[factor models](../factor-models/factor-models.md) framework, of the mechanical
[quantitative selection and construction](./quantitative-selection-and-construction.md) pipeline, and
of the [signal-to-alpha fundamental law](./signal-to-alpha-fundamental-law.md).

## From investable universe to a restricted set: the balance sheet as raw material

Security selection starts from the information firms periodically communicate to the investor
community. Financial statements are accounting reports describing a firm's past performance that
public companies publish quarterly and annually; they are the instrument through which investors,
analysts, and other outsiders obtain information about the corporation. Every public company must
produce four documents: the balance sheet, the income statement, the statement of cash flows, and the
statement of stockholders' equity.

The balance sheet is a snapshot of financial position at an instant, listing assets and liabilities;
the two sides equal by construction under the accounting identity
$$\text{Assets} = \text{Liabilities} + \text{Stockholders' Equity}.$$
The difference between assets and liabilities is the book value of equity, an accounting measure of
the firm's net worth. The income statement reports revenues and costs over an interval and arrives, as
its bottom line, at net income or earnings, whose per-share expression is
$$\text{EPS} = \frac{\text{Net Income}}{\text{Shares Outstanding}}.$$
The statement of cash flows reconciles income with the cash actually generated, split into operating,
investing, and financing activities. Accounting standards — US GAAP, IFRS elsewhere — provide a common
set of rules and a standard format, making it easier to compare different firms. Investors use the
statements in two ways: comparing a firm with itself over time, or comparing it with similar firms via
a common set of financial ratios. Narrowing the universe means applying either criterion
systematically, sorting and selecting on ratios measuring profitability, liquidity, working capital,
interest coverage, leverage, valuation, and operating return.

The raw material has limits. Many assets are carried at historical cost rather than current value, and
some valuable assets — employee expertise, reputation, customer and supplier relationships, the value
of future innovations, management quality — do not appear on the balance sheet at all. For these
reasons the book value of equity, however accurate accounting-wise, is an imprecise measure of the
true equity value. The market value of equity, or market capitalization, is instead
$$\text{Market Value of Equity} = \text{Shares outstanding} \times \text{Market price per share},$$
and depends not on the historical cost of assets but on what investors expect them to produce in the
future.

## The taxonomy of financial ratios

**Profitability.** The income statement gives the profitability measures. Gross margin is gross
profit over sales, reflecting the ability to sell a product for more than its production cost.
Operating margin (operating income over sales) measures earnings before interest and taxes per unit of
revenue; the EBIT margin is analogous. The net profit margin (net income over sales) is the fraction
of each revenue unit left to shareholders after interest and taxes — but comparisons require caution,
since differences may also arise from leverage and different accounting assumptions.

**Liquidity.** From the balance sheet, short-term solvency is judged. The current ratio compares
current assets to current liabilities,
$$\text{Current Ratio} = \frac{\text{Current Assets}}{\text{Current Liabilities}}.$$
A stricter test is the quick ratio, whose numerator counts only cash and near-cash assets (short-term
investments and receivables), excluding inventory because it may be illiquid; the cash ratio, with
cash alone in the numerator, is the most stringent. A higher ratio implies less risk of a cash
shortfall, but these ratios consider only current assets: a firm can be very liquid with modest ratios
if it generates cash quickly from operations.

**Operating efficiency and working capital.** Combining income statement and balance sheet measures
working-capital efficiency. Accounts receivable days express receivables in days of sales,
$$\text{Accounts Receivable Days} = \frac{\text{Accounts Receivable}}{\text{Average Daily Sales}},$$
with analogous ratios for payables and inventory. Alternatively, turnover ratios express annual
revenues or costs as a multiple of the working-capital item, such as inventory turnover = annual cost
of sales / inventory; higher turnover means fewer days and more efficient use of working capital.
Asset turnover (sales / total assets) measures how efficiently assets generate sales.

**Interest coverage.** Creditors judge the ability to meet financial charges with interest coverage
ratios, typically EBIT as a multiple of interest; a high ratio means the firm earns much more than
needed to pay interest. Because depreciation and amortization are deducted in EBIT though they are not
cash outlays, analysts often compute
$$\text{EBITDA} = \text{EBIT} + \text{Depreciation and Amortization},$$
as a measure of the cash the firm generates from operations to pay interest.

**Leverage.** The debt-equity ratio is total debt / total equity, computable with book or market
equity; because book equity is hard to interpret — and can even be negative, making the ratio
meaningless — comparing debt with the market value of equity is more informative. The debt-to-capital
ratio is total debt / (total equity + total debt), and net debt is debt in excess of cash reserves,
$$\text{Net Debt} = \text{Total Debt} - \text{Excess Cash \& Short-term Investments};$$
a firm with more cash than debt could immediately extinguish its debts and thus has no effective
leverage. The equity multiplier, in book terms total assets / book value of equity, captures the
amplification of accounting returns from leverage.

**Operating return.** Return on investment relates income to invested capital. Return on equity is
ROE = net income / book value of equity; return on assets, ROA = (net income + interest expense) /
book value of assets, includes interest in the numerator because assets are financed by both debt and
equity and is less sensitive to leverage than ROE. Return on invested capital,
$$\text{ROIC} = \frac{\text{EBIT}\,(1 - \text{tax rate})}{\text{Book Value of Equity} + \text{Net Debt}},$$
measures the after-tax profit generated by the business alone, and is the most useful of the three for
assessing underlying performance.

**The DuPont decomposition.** ROE can be decomposed via the DuPont Identity into profitability, asset
efficiency, and leverage,
$$\text{ROE} = \underbrace{\frac{\text{Net Income}}{\text{Sales}}}_{\text{Net Profit Margin}} \times \underbrace{\frac{\text{Sales}}{\text{Total Assets}}}_{\text{Asset Turnover}} \times \underbrace{\frac{\text{Total Assets}}{\text{Book Value of Equity}}}_{\text{Equity Multiplier}}.$$
The first factor is overall profitability, the second the efficiency with which assets generate sales;
their product is return on assets, and multiplying by the equity multiplier (a leverage measure) gives
ROE. Two firms with similar margins can have different ROE through different asset turnover and
leverage.

## Multiples as value signals

Valuation ratios relate market value to measures of firm scale and are used to judge whether a stock
is expensive or cheap, and for within-industry comparisons.

**Market-to-book (price-to-book).** The ratio of market capitalization to book equity is the
market-to-book (price-to-book) ratio,
$$\text{Market-to-Book Ratio} = \frac{\text{Market Value of Equity}}{\text{Book Value of Equity}}.$$
For most successful firms this ratio far exceeds one, indicating that the value of assets put to work
exceeds their historical cost; its variation reflects differences in fundamentals and the value added
by management. Analysts label low market-to-book firms **value stocks** and high market-to-book firms
**growth stocks**.

**Price-earnings.** The most common valuation multiple is the price-earnings (P/E) ratio, on a total
or per-share basis,
$$\text{P/E Ratio} = \frac{\text{Market Capitalization}}{\text{Net Income}} = \frac{\text{Share Price}}{\text{Earnings per Share}}.$$
Buying a share acquires rights to the firm's future earnings; because differences in earnings scale
tend to persist, one pays proportionally more for a stock with higher current earnings. The P/E is
grounded in the constant-growth dividend-discount model: starting from $P_0 = Div_1/(r_E - g)$ and
dividing by expected earnings $EPS_1$ gives the forward P/E,
$$\text{Forward P/E} = \frac{P_0}{EPS_1} = \frac{Div_1/EPS_1}{r_E - g} = \frac{\text{Dividend Payout Rate}}{r_E - g}.$$
The forward P/E (on the next twelve months' expected earnings) is preferred for valuation to the
trailing P/E (prior twelve months). The formula shows that two stocks with the same payout, EPS growth,
and equivalent risk (hence cost of capital) should have the same P/E, and that high-growth firms and
sectors — able to generate cash well beyond investment needs and thus sustain high payouts — should
have high P/E; riskier firms, all else equal, have lower P/E.

**Enterprise-value multiples.** Because the P/E uses the equity value, it is sensitive to the choice of
leverage and of limited use in comparing firms with markedly different leverage. This is overcome by
valuing the underlying business through the enterprise value,
$$\text{Enterprise Value} = \text{Market Value of Equity} + \text{Debt} - \text{Cash},$$
the value of the business net of cash and independent of debt. Since enterprise value is the whole-firm
value before debt repayment, an appropriate multiple divides it by a pre-interest earnings or flow
measure — typically EV/EBIT, EV/EBITDA, or EV/free-cash-flow — and because capital expenditures can
vary widely period to period, most practitioners favor EV/EBITDA. This too is grounded in the
constant-growth model: with constant expected free-cash-flow growth,
$$\frac{V_0}{EBITDA_1} = \frac{FCF_1/EBITDA_1}{r_{wacc} - g_{FCF}},$$
higher with higher growth and lower capital needs.

**Aggregate consistency.** A multiple's numerator and denominator must both refer to the whole firm or
both to equity holders only. Price and capitalization are equity quantities, matched to EPS or net
income; revenues, operating income, and EBITDA belong to the whole firm and are matched to enterprise
value. P/E and EBIT/EBITDA multiples are meaningless with negative earnings, in which case EV/sales is
often used, with the risk that earnings are negative because the business model is fundamentally
flawed.

## The limits of comparing non-comparable firms

Using multiples is an application of the comparables method: rather than discounting the firm's flows
directly, its value is estimated from that of comparable firms expected to generate very similar flows,
with the multiple adjusting for scale — as a building is valued from the price per square meter of
similar recently sold buildings, a stock is valued substituting an appropriate scale measure for the
square footage. Concretely, a stock's value is estimated by multiplying its current EPS by the average
P/E of comparables, assuming similar future risk, payout, and growth.

The method's weakness is that identical firms do not exist. If comparables were identical their
multiples would coincide exactly; since they are not, a multiple's usefulness depends on the nature of
inter-firm differences and the multiple's sensitivity to them. Comparing the multiples of a set of
footwear firms, each multiple shows significant dispersion around the mean: even EV/EBITDA, the least
variable, cannot deliver a precise value estimate. The differences reflect differences in expected
growth, profitability, risk (hence cost of capital), and sometimes accounting conventions across
countries. Investors understand these differences and price accordingly, but valuing with multiples
gives no clear guidance on how to adjust for them except by narrowing the comparable set.

Hence the fundamental limit: the comparables approach ignores important inter-firm differences —
exceptional management, an efficient production process, a patent on new technology are all ignored
when a multiple is applied. A second limit is that multiples give only relative information: they say
how a firm is valued relative to others in the comparison set, but do not help establish whether an
entire sector is overvalued — a problem acute during the late-1990s Internet boom, when new multiples
were invented to justify the values of firms with no positive flows or earnings. Multiples are thus a
shortcut relative to discounted-cash-flow methods: in exchange for simplicity and grounding in actual
prices, one forgoes DCF's ability to incorporate firm-specific information and run sensitivity
analyses. No single technique gives a definitive answer on true value; practitioners use a combination
and take comfort from their convergence.

## Prices, information, and the limits of active selection

Fundamental selection confronts a fact: the market price of a listed firm already incorporates very
accurate information, aggregated from a multitude of investors, about the true value of the shares.
When a buyer wants to buy, the willingness of others to sell signals that they value it differently;
the information that others are willing to trade induces buyers and sellers to revise their estimates
until consensus is reached, so markets aggregate the information and opinions of many investors. It
follows that if a valuation model disagrees with the price, this is more likely a sign of an error in
one's assumptions than of a mispriced stock.

The idea that competition among investors eliminates all positive-NPV trading opportunities is the
efficient markets hypothesis: securities are correctly priced on their future flows, given available
information. Competition is fiercest — and the hypothesis holds best — when information is public and
easy to interpret: the price reacts almost instantly, and few investors trade before adjustment is
complete. When information is private or hard to interpret, its holders can profit, and prices reflect
it only gradually. This frames the stakes of active selection: an investor can find positive-NPV
opportunities only given a barrier to free competition — expertise or access to information known to
few, or lower transaction costs — and the source of advantage must be hard to replicate, or the gains
are eroded. Testing efficiency, however, requires a theory of how risk determines expected returns: the
hypothesis in return terms says equivalent-risk securities have the same expected return, and is
incomplete until equivalent risk is defined. It is in this choice of risk measure that the question of
whether fundamental characteristics generate genuine excess return is decided.

## Style-based techniques: size and value in the cross-section

Managers often distinguish strategies by the stocks they hold: small versus large, value versus
growth. If the CAPM held, the market portfolio would be efficient and no strategy could beat the market
without extra risk; the difference between a stock's expected return and the return required along the
security market line is its alpha,
$$\alpha_s = E[R_s] - r_s, \qquad r_s = r_f + \beta_s\,(E[R_{Mkt}] - r_f),$$
zero for all stocks when the market portfolio is efficient. Some fundamental characteristics, however,
order stocks in the cross-section so as to produce systematically non-zero alpha.

**The size effect.** Low-capitalization stocks have historically earned higher average returns than
the market portfolio; despite high beta, their returns look high even accounting for the higher beta —
the size effect. Sorting stocks annually by capitalization into ten decile portfolios and recording
their monthly excess returns, almost all portfolios lie above the security market line, most markedly
in the smallest deciles, and a joint test that all alphas are zero is statistically rejected. The size
effect was first identified by Banz in 1981.

**The value effect.** Analogous results arise using the book-to-market ratio to form portfolios. Value
stocks (high book-to-market) tend to have positive alpha, while growth stocks (low book-to-market) have
low or negative alpha; again the joint test of zero alphas is rejected. The systematic evidence on
risk-adjusted return and market value is documented by Fama and French.

**A theoretical explanation.** After Banz, a theoretical reason linking capitalization and expected
returns emerged: as long as beta is not a perfect risk measure — through estimation error or because
the market portfolio is not efficient — one should expect to observe the size effect. A positive-alpha
stock has, all else equal, a higher expected return; the only way to offer a higher expected return is
to buy the same dividend stream at a lower price; a lower price means lower capitalization (and higher
book-to-market). So a portfolio of low-cap or high-book-to-market stocks collects high-expected-return
stocks and, if the market is inefficient, positive alpha. Consider two firms with the same perpetual
$1$ million dividend: with costs of capital of $14\%$ and $10\%$, their market values are
$1/0.14 = \$7.143$ million and $1/0.10 = \$10$ million. Assigning both, by error or inefficiency, the
same beta whose CAPM required return is $12\%$, the lower-value firm has $\alpha = 0.14 - 0.12 = 2\%$
and the higher-value firm $\alpha = 0.10 - 0.12 = -2\%$: the smaller-cap firm has the higher alpha.

**Momentum.** Past returns also order stocks. Sorting monthly by return over the prior $6$–$12$ months,
the best stocks earn positive alpha over the next $3$–$12$ months: a momentum strategy buying high
past-return stocks and shorting low past-return stocks would have produced, over $1965$–$1989$, an
alpha above $12\%$ per year.

**The data-snooping caveat.** At discovery, many researchers found the evidence unconvincing, because
searching over a large set of characteristics can always, by pure chance, turn up one correlated with
the estimation error of average returns — the data-snooping bias. The existence of positive-alpha
strategies leaves only two conclusions: either investors systematically ignore positive-NPV
opportunities, or such strategies carry a risk investors are unwilling to bear that the CAPM does not
capture. Since these strategies are long known, easily constructed, and widely followed, the first
conclusion is implausible; the second remains — the market portfolio is not efficient and beta does not
adequately measure systematic risk. Possible reasons include proxy error in the observed market
portfolio, systematic behavioral biases, and exposure to non-tradable risks outside the portfolio,
first among them human capital.

## Building factor portfolios

If the market portfolio is not efficient, computing expected returns requires an alternative way to
identify an efficient portfolio. Identifying efficient portfolios directly is hard, since expected
return and standard deviation cannot be measured precisely; but two properties of the efficient
portfolio are known: it is well diversified, and it can be built from other well-diversified
portfolios. This last point is crucial: it means one need not identify the efficient portfolio itself,
but only a collection of well-diversified factor portfolios from which an efficient portfolio can be
built, and use the collection to measure risk.

With $N$ factor portfolios of returns $R_{F1}, \dots, R_{FN}$, the expected return of stock $s$ is
$$E[R_s] = r_f + \sum_{n=1}^{N} \beta_s^{Fn}\,(E[R_{Fn}] - r_f),$$
where the factor betas measure the expected percentage change in the stock's excess return for a $1\%$
change in the factor portfolio's excess return, holding other exposures constant. A stock's risk
premium is the sum of each factor's premium times the stock's sensitivity. With one efficient
portfolio the model is single-factor; with several it is a multifactor model, also known as Arbitrage
Pricing Theory.

**Self-financing portfolios.** Each factor's expected premium $E[R_{Fn}] - r_f$ is the expected return
of a portfolio that borrows at $r_f$ to invest in the factor portfolio. Since such a portfolio costs
nothing to build, it is self-financing; one can also obtain it by going long some stocks and short
others of equal market value. In general, a self-financing portfolio has weights summing to zero rather
than one. Requiring all factor portfolios to be self-financing rewrites the model as
$$E[R_s] = r_f + \sum_{n=1}^{N} \beta_s^{Fn}\,E[R_{Fn}],$$
computing the cost of capital without identifying the efficient portfolio, relying on the weaker
condition that an efficient portfolio be constructible from well-diversified portfolios.

**Choosing the portfolios.** The first natural portfolio is the market itself: though not necessarily
efficient, it has historically commanded a large premium and captures much systematic risk, included as
a self-financing long-market/short-riskless position. The others come from style strategies with
apparent positive alpha:

- *Size strategy.* Each year firms split into two equal-weighted portfolios by market equity — below
  ($S$) and above ($B$) the NYSE median; buying $S$ financed by shorting $B$ is the small-minus-big
  (SMB) portfolio.
- *Book-to-market strategy.* Firms below the 30th percentile of book-to-market form $L$ and above the
  70th form $H$; long $H$ financed by short $L$ is the high-minus-low (HML) portfolio, long value and
  short growth.
- *Past-return strategy.* Sorting by the prior year's return and going long the best 30% and short the
  worst 30% gives the prior one-year momentum (PR1YR) portfolio.

**The Fama-French-Carhart specification.** The collection of four portfolios — market excess return
$(Mkt - r_f)$, SMB, HML, and PR1YR — is the most common multifactor choice, and stock $s$'s expected
return is the Fama-French-Carhart specification,
$$E[R_s] = r_f + \beta_s^{Mkt}(E[R_{Mkt}] - r_f) + \beta_s^{SMB} E[R_{SMB}] + \beta_s^{HML} E[R_{HML}] + \beta_s^{PR1YR} E[R_{PR1YR}],$$
where the four factor betas measure sensitivity to each portfolio. The portfolios were identified by
Eugene Fama, Kenneth French, and Mark Carhart. The multifactor model's advantage is that it is far
easier to find a collection of portfolios capturing systematic risk than a single one; its disadvantage
is that each portfolio's expected return must be estimated, and each added portfolio increases
implementation difficulty. Since it is unclear which economic risk each portfolio captures, historical
average returns are used; and because FFC portfolio returns are very volatile, over eighty years of
data are used to estimate the expected return, which remains imprecise. The one domain where FFC
clearly improves on the CAPM is measuring the risk of actively managed mutual funds: whereas under the
CAPM funds with high past returns show positive alpha, repeating the test with FFC finds no evidence
that such funds have positive future alpha.
