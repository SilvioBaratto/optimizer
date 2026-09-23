---
type: concept
title: "Macroeconomic Signals and Expected Risk Premia"
description: Which macroeconomic and financial variables predict time-varying expected risk premia for stocks and bonds, how they are validated out of sample, and the theory-driven constraints (sign restrictions, non-negativity, valuation anchoring) that make otherwise fragile predictors exploitable.
tags: [macro-signals, risk-premia, return-predictability, term-spread, credit-spreads, volatility-managed, out-of-sample, regime]
sources:
  - id: openwiki-source-3ea81f0a892662ff6fd9ec24
    resource: repo://docs/29_macroeconomic_signals_and_expected_risk_premia.md
generated: { by: "claude-code", at: "2026-09-23T08:34:43.263Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Macroeconomic Signals and Expected Risk Premia

The [mean-variance construction](../foundations/mean-variance-selection.md) takes the vector
of expected returns and the covariance matrix as given. The return-predictability literature
argues that the *expected* risk premium is not constant but varies systematically with the
business cycle, and that observable variables — the shape of the yield curve, credit
spreads, asset variance, broad macroeconomic aggregates — capture part of that variation. A
*conditional* expected return $E_t[r_{t+1}]$ is a function of the information available at
$t$; if the function is non-trivial, an investor can tilt allocation toward assets when
their expected premium is high and away when it is low. The conditional expected returns
produced here are the input the [from conditional forecasts to weights](../workflows/from-conditional-forecasts-to-weights.md)
chapter turns into portfolio weights, and the same indicators measure the latent regimes of
[market regimes and views](./market-regimes-and-views.md).

## The predictive regression and its out-of-sample test

The standard test bed is the predictive regression
$$ r_{t+1} = \alpha + \beta\, x_t + \varepsilon_{t+1}, $$
where $r_{t+1}$ is the asset's excess return (equity premium, bond excess return), $x_t$ is
the predictor known at $t$, and the null of no predictability is $\beta = 0$. In-sample
significance is not enough: the decisive criterion is out-of-sample performance, measured by
the out-of-sample $R^2$
$$ R^2_{OS} = 1 - \frac{\sum_{t}\big(r_{t+1}-\hat r_{t+1}\big)^2}{\sum_{t}\big(r_{t+1}-\bar r_{t+1}\big)^2}, $$
where $\hat r_{t+1}$ is the recursively estimated model forecast and $\bar r_{t+1}$ the
historical mean, a stringent benchmark. Because estimating an extra coefficient injects
noise, under the null the expected value of $R^2_{OS}$ is slightly negative, so even a zero
$R^2_{OS}$ is already weak evidence for predictability.

## The yield-curve slope and the probability of recession

The most robust macroeconomic predictor is the slope of the term structure, the spread
between a long and a short government rate,
$$ SPREAD_t = R^{L}_t - R^{S}_t, $$
defined as the ten-year Treasury yield minus the three-month yield. A flat or inverted curve
($SPREAD_t \le 0$) tends to anticipate a real slowdown. The link to recession probability is
formalized with a probit: with $R_{t+k}=1$ if the economy is in recession $k$ quarters after
$t$,
$$ \Pr\!\big(R_{t+k}=1 \mid X_t\big) = \Phi\!\big(\alpha_0 + \alpha_1\, X_t\big), \qquad X_t = SPREAD_t. $$
The benchmark four-quarter estimate on 1960–1995 quarterly data is
$\Pr(R_{t+4}=1) = \Phi(-0.66 - 0.81\, SPREAD_t)$, with pseudo-$R^2 = 0.296$ and a strongly
significant slope ($t \approx -4.99$); a spread of 0.76 points maps to a 10% probability,
$-0.82$ to 50%, and $-2.40$ to 90%. Across a battery of leading indicators no single one
dominates at every horizon, but the term spread is the clear choice beyond two quarters and
out to roughly four-to-six quarters, while equity indices are competitive at the very
shortest horizons. Out of sample the spread's superiority grows with the horizon, and adding
a third regressor generally hurts — hence a lesson of parsimony. The recession probability is
a direct input for the regime views of [market regimes and views](./market-regimes-and-views.md).

## The term structure as a predictor of real activity

The curve predicts not only the binary recession event but continuous future real growth.
Using the same ten-year-minus-three-month spread, the cumulative dependent variable is the
annualized percentage change in real output from $t$ to $t+k$,
$$ Y_{t,\,t+k} \equiv \frac{400}{k}\,\log(y_{t+k}/y_t), $$
regressed on the current spread, $Y_{t,\,t+k} = \alpha_0 + \alpha_1\, SPREAD_t + \sum_i \beta_i X_{it} + \epsilon_t$,
on 1955–1988 quarterly data with Newey–West standard errors for the overlapping-horizon
moving-average errors. The estimated $\alpha_1$ is positive and significant: at four quarters
$\alpha_0 = 1.70$ and $\alpha_1 = 1.30$, so a 100-basis-point spread predicts annual real
growth of about $1.70\% + 1.30\% = 3\%$. Cumulative predictive power extends about four
years, peaking at five-to-seven quarters where the spread alone explains more than a third of
future output variation ($\bar R^2 \approx 0.38$). The signal holds for all private
demand components — consumption, durables, investment — but **not** for government spending.
In incremental-content tests the spread keeps a significant coefficient out to about three
years even after controlling for the real federal-funds rate, leading indicators, lagged
growth, and lagged inflation, and it beats survey forecasts in and out of sample.

## Credit spreads, expected loss, and the excess bond premium

Credit spreads — corporate minus matched-maturity government yields — are classic cyclical
predictors but crude: they blend compensation for expected default loss with a residual
reflecting risk appetite. A precise spread is built by discounting the corporate bond's exact
cash flows on the zero-coupon Treasury curve to form a **synthetic** risk-free bond, giving
the single-bond spread $S_{it}[k] = y_{it}[k] - y^{f}_{t}[k]$ and the GZ index
$S^{GZ}_t = \frac{1}{N_t}\sum_i\sum_k S_{it}[k]$. Default risk is measured by a Merton-style
distance-to-default
$$ DD_{it} = \frac{\ln(V_{it}/D_{it}) + (\mu_{V,it} - \tfrac{1}{2}\sigma^{2}_{V,it})}{\sigma_{V,it}}, $$
with implied default probability $\Phi(-DD)$. Projecting the log spread on $DD$ and issue
characteristics gives a predicted (expected-loss) spread $\hat S^{GZ}_t$ (adjusted
$R^2 \approx 0.65$–$0.70$), and the residual defines the *excess bond premium*
$$ EBP_t = S^{GZ}_t - \hat S^{GZ}_t. $$
The EBP is strongly cyclical and predicts future consumption, investment, industrial
production, and employment with coefficients systematically larger in absolute value than
the expected-loss component; since the mid-1980s essentially **all** of the GZ spread's
predictive content for output growth is attributable to the EBP. In a VAR, a one-standard-
deviation EBP shock (about 20 basis points) triggers protracted declines in output and a
cumulative equity-market fall of roughly 7%. The interpretation is that the EBP measures
intermediaries' effective risk-bearing capacity — a risk-premium signal common to corporate
bonds and equities, a natural ingredient for the systemic-risk reading of
[turbulence and systemic risk](./turbulence-and-systemic-risk.md).

## Volatility as a timing signal

A different signal comes not from the level but from the variance of assets: scale exposure
to a risk factor in inverse proportion to its expected variance. The motivation is
mean-variance: the optimal weight is $w^{*}_t \propto E_t[f_{t+1}]/\hat\sigma^{2}_t(f)$, and
because volatility is highly variable and persistent but does not predict return, the
numerator is treated as roughly constant. The *volatility-managed* portfolio is
$$ f^{\sigma}_{t+1} = \frac{c}{\hat\sigma^{2}_t(f)}\, f_{t+1}, $$
with $c$ chosen so managed and original strategies share the same unconditional variance and
$\hat\sigma^{2}_t(f) = RV^{2}_t = \sum_{d} f^{2}_{t,d}$ estimated from the prior month's daily
realized variance — mechanical, real-time, parameter-free: raise exposure after calm months,
cut it after turbulent ones. Regressing the managed strategy on the original,
$f^{\sigma}_{t+1} = \alpha + \beta\, f_{t+1} + \varepsilon_{t+1}$, yields positive significant
$\alpha$ for the market (about 4.9% annual, $\beta \approx 0.6$) and for value, momentum,
profitability, investment, and currency carry (with size the exception). A positive $\alpha$
means the managed strategy expands the mean-variance frontier; the gain is measured by the
appraisal ratio $\alpha/\sigma(\varepsilon)$ (about 0.33 annual for the market), lifting the
Sharpe ratio to $S_{new} = \sqrt{S_{old}^{2} + (\alpha/\sigma_\varepsilon)^{2}}$ (roughly a
25% increase) and delivering a mean-variance utility gain
$\Delta U_{MV} = (S_{new}^{2} - S_{old}^{2})/S_{old}^{2}$ of about 65% for a market-only
investor. The result rests on an asymmetry — past volatility predicts current variance
strongly but future return weakly — so cutting exposure in high-variance months lowers risk
more than it sacrifices return. This timing signal complements the level predictors and
connects to the [dynamic covariance](../estimation/dynamic-covariance.md) estimation.

## Compressing many series into a few dynamic factors

A macro dashboard holds dozens or hundreds of series — too many for a predictive regression
without exhausting degrees of freedom. A panel of 132 standardized macro series is compressed
into a few common dynamic factors via the factor structure
$$ x_{it} = \lambda_i'\, f_t + e_{it}, $$
with $f_t$ a reduced vector ($r \ll N$) of latent common factors estimated by principal
components (diffusion indices); information criteria select $r = 8$ factors, the first
explaining 17.7% of panel variance and the eight together about 50%. The factors enter a
bond excess-return regression
$$ rx^{(n)}_{t+1} = \alpha'\, \hat F_t + \beta'\, Z_t + \epsilon_{t+1}, $$
where $Z_t$ holds the Cochrane–Piazzesi forward-rate factor $CP_t$. A BIC-selected subset
$(\hat F_{1t}, \hat F_{1t}^{3}, \hat F_{2t}, \hat F_{3t}, \hat F_{4t}, \hat F_{8t})$ is
preferred; the first "real" factor (loading on employment and production) is the single most
important. Macro factors add content beyond $CP_t$: alone they explain 21%–26% of one-year
excess-return variation, and with $CP_t$ they raise $\bar R^2$ to about 44% for the two-year
bond (versus 31% for $CP_t$ alone). This information is **not** in the yield curve, and the
resulting premia are strongly countercyclical — the real factor correlates over 90% with
industrial-production growth, so premia are high in recessions and low in expansions — and the
result holds out of sample (RMSE 79%–93% of the constant-premium benchmark). This applies the
[factor models](../factor-models/factor-models.md) framework to predictors, not returns.

## Combining many forecasts: macro and technical predictors

If many predictors are each weak, combining their forecasts beats selecting one. The equity
premium is forecast by uniting fourteen macro-finance variables (valuation ratios, volatility,
net equity issuance, short and long rates, term spread, default yield and return spreads,
inflation) with fourteen technical trend rules (moving averages, momentum, on-balance
volume). To combine without over-parameterizing, principal components of each predictor set
serve as diffusion-index regressors,
$$ r_{t+1} = \alpha + \sum_{k=1}^{K}\beta_k\, \hat F^{j}_{k,t} + \varepsilon_{t+1}, \qquad j \in \{\text{ECON}, \text{TECH}, \text{ALL}\}. $$
The key result is in-sample additivity: PC-ALL's $R^2$ (2.02%) equals the sum of PC-ECON
(1.18%) and PC-TECH (0.84%), so macro and technical predictors carry **complementary**
information. Out of sample PC-ALL reaches $R^2_{OS} = 1.79\%$, above either family alone, and
its predictability is strongly countercyclical (11.24% in recessions versus negative in
expansions). Translated into a mean-variance weight $w_t = \frac{1}{\gamma}\hat r_{t+1}/\hat\sigma^2_{t+1}$,
PC-ALL yields a certainty-equivalent gain of about 4.9% annual. These heterogeneous-signal
fusion techniques connect to the [alternative-data signals](../signals/alternative-data-and-sentiment.md).

## The reality check: out-of-sample fragility and theory constraints

The survey must be read against a severe caution. A systematic exam of equity-premium
predictors compares in- and out-of-sample performance via the cumulative difference in
squared forecast errors between the historical-mean null and the predictive model,
$$ \Delta_t = \sum_{s\le t}\big(r_{s}-\bar r_{s-1}\big)^2 - \sum_{s\le t}\big(r_{s}-\hat r_{s}\big)^2, $$
where a rising stretch means the predictor beats the mean. For most variables $\Delta_t$ does
not rise steadily: in-sample predictability does not survive out of sample, coefficients are
unstable, and much apparent success concentrates in a few episodes (notably the 1973–1975 oil
shock) before vanishing. In many cases out-of-sample $\bar R^2$ is negative — the historical
mean is unbeatable.

The response is that predictors become useful under theory-motivated restrictions rather than
free estimation. First, sign and non-negativity constraints: if $\hat\beta_t$ has the wrong
sign it is set to zero, and a negative premium forecast is truncated,
$$ \hat r_t^{\,restr} = \max\!\big(0,\ \hat\alpha_t + \hat\beta_t\, x_t\big). $$
Second, a steady-state valuation anchor: from the Gordon growth identity $D/P = R - G$ and
$G = (1 - D/E)\,ROE$, the expected return is built from the current valuation ratio anchored
to historical payout and $ROE$, e.g. $\hat R_{EP} = (D/E)(E/P) + (1 - D/E)\,ROE$. With these
restrictions many predictors beat the historical mean out of sample, with small but positive
$R^2_{OS}$. Small $R^2_{OS}$ is still economically relevant: the proportional increase in a
mean-variance investor's expected portfolio return is
$$ \Big(\frac{R^2}{1-R^2}\Big)\Big(\frac{1+S^2}{S^2}\Big), $$
so the right yardstick for $R^2$ is the **squared Sharpe** $S^2$ — with a monthly equity
Sharpe near 0.11 ($S^2 \approx 1.2\%$), a monthly $R^2_{OS}$ near 0.43% raises portfolio
return by about a third. The joint lesson is that macro signals must be used with constraints
— coefficient sign, non-negative premium, valuation anchors — that stabilize estimation and
make them exploitable. It reinforces [estimation error and shrinkage](../estimation/estimation-error-and-shrinkage.md),
and these constrained conditional expected returns are what
[from conditional forecasts to weights](../workflows/from-conditional-forecasts-to-weights.md)
turns into weights.
