---
type: concept
title: "Signal Decay, Horizon and Turnover"
description: How a predictive signal loses power over time, how the speed of that decay — measured by signal autocorrelation and the information horizon — determines the turnover a signal imposes, how averaging lagged factors trades signal for lower turnover, and how, under predictable returns with quadratic transaction costs, the optimal policy does not chase the static optimal portfolio but an aim portfolio that weights signals by their persistence, moving toward it only partially.
tags: [signal-decay, information-horizon, information-coefficient, autocorrelation, turnover, transaction-costs, aim-portfolio, dynamic-trading, mean-reversion, alpha-decay]
sources:
  - id: openwiki-source-66102062a538319cff35a35d
    resource: repo://docs/11_signal_decay_horizon_and_turnover.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Signal Decay, Horizon and Turnover

This chapter establishes that a predictive signal loses power over time, and that the speed of this
decay — measured by the signal's autocorrelation and its information horizon — determines the turnover
the signal imposes on the portfolio. It shows how averaging lagged factors reduces turnover at the cost
of part of the signal, and formalizes the trade-off with a model that maximizes the information ratio
subject to a persistence constraint. It finally establishes that, with predictable returns and
quadratic transaction costs, the optimal policy does not chase the static optimal portfolio but an
*aim portfolio* that weights signals by their persistence, moving toward it only partially. It connects
the [signal-to-alpha fundamental law](./signal-to-alpha-fundamental-law.md) to the trading side —
[optimal execution and market impact](../workflows/optimal-execution-and-market-impact.md) and
[portfolio revision](../workflows/portfolio-revision.md).

## Information decay and the signal's horizon

A quantitative factor produces, each period, a cross-sectional forecast of stock returns; but the
information it embeds is not confined to the single period after its observation. To measure the
persistence of information it helps to distinguish three information coefficients (IC), all defined as
cross-sectional correlations between factor values and stock returns.

The first is the conventional IC, matching the factor known at time $t$ with the return realized from
$t$ to $t+1$,
$$IC_{t,t} = \mathrm{corr}\!\left(F_t, R_t\right).$$
The second is the lagged IC, matching a factor already known at time $t-l$ with the later return from
$t$ to $t+1$,
$$IC_{t-l,t} = \mathrm{corr}\!\left(F_{t-l}, R_t\right).$$
The lagged IC directly measures information decay: a factor with a positive conventional IC at
observation may, one or more periods later, retain a positive IC (information persists) or see it
vanish rapidly toward zero. The third is the horizon IC, matching the factor known at time $t$ with the
return cumulated over $h$ periods forward,
$$IC_t^{h} = \mathrm{corr}\!\left(F_t, R_{t,t+h}\right), \qquad h = 0,1,\dots,H.$$
The information horizon is how far into the future the signal keeps predictive power: as $h$ grows, the
cumulative return accumulates the information the factor still captures.

The three coefficients are related. Because the cumulative return $R_{t,t+h}$ is the sum of the single
periods' returns from $t$ to $t+h$, the approximate relation is
$$IC_t^{h} \approx \frac{IC_{t,t} + IC_{t-1,t} + \cdots + IC_{t-h,t}}{\sqrt{h+1}} = \mathrm{avg}(IC)\,\sqrt{h+1},$$
where the last equality holds when the lagged coefficients are roughly constant at their average
$\mathrm{avg}(IC)$. The horizon IC thus grows as the square root of the number of periods covered — the
same form as the fundamental law of active management, where the information ratio scales with the
square root of breadth. Horizon and decay are two faces of one property: a factor whose lagged ICs stay
positive long has an extended horizon and a horizon IC that keeps rising; a factor whose lagged ICs
quickly become zero, or change sign, has a short horizon and a horizon IC that flattens or reverses.

An empirical comparison of two factors — price momentum (PM, the nine-month return lagged by one) and
earnings-to-price (E2P) — makes the distinction concrete. Momentum's average IC starts high at lag zero
and decays rapidly with the lag, near zero within a few lags; E2P's starts lower but stays nearly
stable at later lags. Correspondingly, momentum's information ratio collapses with the lag while E2P's
holds. The value signal keeps information over a long horizon; the momentum signal exhausts it quickly.

## Signal autocorrelation and induced turnover

A signal that decays fast changes rapidly period to period; and since the portfolio is built from the
signal, a volatile signal imposes heavy repositioning. Repositioning is measured by turnover,
$$T = \frac{1}{2}\sum_{i=1}^{N} \left| w_i^{t+1} - w_i^{t} \right|,$$
with $w_i^t$ the weight of stock $i$ at time $t$ summed over $N$ stocks. What links turnover to signal
decay is the forecast autocorrelation, the cross-sectional correlation between the signal's values in
two consecutive periods,
$$\rho_f = \mathrm{corr}\!\left(\tilde F^{\,t+1}, \tilde F^{\,t}\right).$$
For an optimized portfolio the turnover is
$$T = \sqrt{\frac{N}{\pi}}\;\sigma_{model}\,\sqrt{1-\rho_f}\;E\!\left(\frac{1}{\sigma}\right),$$
with $\sigma_{model}$ the target tracking error and $\sigma$ the stocks' specific risk. Turnover is
higher the higher the target tracking error, the larger the number of stocks (as $\sqrt{N}$), the lower
the forecast autocorrelation, and the lower the average specific risk. The third point ties turnover to
decay: the factor $\sqrt{1-\rho_f}$ grows as autocorrelation falls. A slow-decaying signal has
$\rho_f$ near one and generates little turnover; a fast-decaying signal has low $\rho_f$ and generates
high turnover.

The empirical classification of factors confirms the link. Ranking average $\rho_f$ by category gives
high values for value factors (E2PFY0 $0.96$; B2P $0.93$; CFO2EV $0.84$), low values for momentum
factors (EarnRev9 $0.64$; Ret9Monx1 $0.60$; LtgRev9 $0.37$), and intermediate values for quality
factors (RNOA $0.89$; XF $0.76$; NCOinc $0.80$). So momentum factors, with the lowest autocorrelation,
have the highest turnover; value factors, with the highest autocorrelation, the lowest; quality factors
sit in between.

## Responsiveness versus stability: the moving average of factors

The conflict is now explicit: closely tracking a fast-decaying signal means updating the portfolio
frequently and bearing high turnover. Among the ways to slow turnover are a direct turnover constraint,
a composition shift toward more value and less momentum, and the use of a moving average of factors.

The moving average combines the factor's current value with its lagged values. In the two-term form,
MA(2),
$$F_{ma}^{\,t} = v_0\,F^{\,t} + v_1\,F^{\,t-1}.$$
Averaging the factor with its lag increases its autocorrelation and hence reduces turnover. With a
factor of lag-one autocorrelation $\rho_f(1)=0.90$ (and $\rho_f(2)=0.81$), the moving average's
autocorrelation peaks at about $0.95$ at equal weights $v_1=0.5$. Since turnover scales with
$\sqrt{1-\rho_f}$, the moving average's turnover is
$$\frac{\sqrt{1-0.95}}{\sqrt{1-0.90}} \approx 71\%$$
of the raw factor's turnover.

The turnover reduction is not free. Lagged factors forecast future returns more weakly than current
ones — the information decay documented by lagged ICs — so averaging lowers alpha. The question is the
right trade-off between lower turnover and reduced alpha.

The answer casts the choice as an optimization. The combination weights $v=(v_1,\dots,v_M)$ of the
factors are chosen to maximize the model's information ratio,
$$IR = \frac{\mathrm{avg}(IC)}{\mathrm{std}(IC)} = \frac{v' \cdot \overline{IC}}{\sqrt{\,v' \cdot S_{IC} \cdot v\,}},$$
where $\overline{IC}$ is the vector of average ICs and $S_{IC}$ their covariance matrix; the inputs are
the average IC, the IC standard deviation, and their correlations, and the optimum exists in closed
form. For two factors — E2P and PM, with IC correlation $-0.4$ — the unconstrained optimal weights are
PM $36\%$ and E2P $64\%$.

To control turnover, the model extends the regressor set to include, alongside the current factors,
their lagged values, forming a moving-average combined forecast
$$F_{c,ma}^{\,t} = v_{01}F_1^{\,t} + v_{02}F_2^{\,t} + v_{11}F_1^{\,t-1} + v_{12}F_2^{\,t-1} + \cdots,$$
and maximizes the information ratio subject to the combined forecast's autocorrelation equaling a
target level,
$$\text{Maximize}\quad IR = \frac{v' \cdot \overline{IC}}{\sqrt{\,v' \cdot S_{IC} \cdot v\,}} \qquad \text{subject to}\quad \rho_{f_{c,ma}} = \rho_{target}.$$
The solution across $\rho_{target}$ traces the responsiveness–stability trade-off. As the target
autocorrelation rises from $0.85$ to $0.97$, turnover decreases monotonically while the information
ratio first rises, peaks at $2.39$ at $\rho_f = 0.89$, then falls. In aggregate weights, at the lowest
autocorrelation levels the combined forecast loads entirely on the current factors ($w_0 = 100\%$);
only when higher autocorrelation is required does weight shift progressively to the lagged factors
($w_1, w_2, w_3$), at the expense of alpha. The information-ratio maximum defines the turnover level the
signal economically justifies.

## The dynamic trading model with predictable returns

The same trade-off — signal adherence versus turnover cost — can be attacked at the level of portfolio
construction, not just factor combination. Gârleanu-Pedersen derive the optimal portfolio policy when
returns are predictable by signals of different mean-reversion speeds and trading is costly.

Consider an economy with $S$ securities. Price excess returns between $t$ and $t+1$, in the $S\times 1$
vector $r_{t+1}$, are generated by
$$r_{t+1} = B f_t + u_{t+1},$$
where $f_t$ is a $K\times 1$ vector of return-predicting factors known at time $t$, $B$ the $S\times K$
factor-loading matrix, and $u_{t+1}$ zero-mean unpredictable noise with $\mathrm{var}_t(u_{t+1}) =
\Sigma$. The factors evolve as
$$\Delta f_{t+1} = -\Phi f_t + \varepsilon_{t+1},$$
where $\Delta f_{t+1} = f_{t+1} - f_t$, the $K\times K$ matrix $\Phi$ collects the factors'
mean-reversion coefficients, and $\varepsilon_{t+1}$ is the predictor shock with $\mathrm{var}_t
(\varepsilon_{t+1}) = \Omega$. The matrix $\Phi$ is the continuous analog of signal decay: a factor with
small $\Phi$ mean-reverts slowly (slow alpha decay, persistent signal), one with large $\Phi$ reverts
quickly. Different predictors — say a momentum signal predicting high return next month and a value
signal predicting good return over the year — have different mean-reversion speeds.

Trading is costly: the transaction cost of trading $\Delta x_t = x_t - x_{t-1}$ shares is quadratic,
$$TC(\Delta x_t) = \frac{1}{2}\,\Delta x_t^{\top} \Lambda \, \Delta x_t,$$
with $\Lambda$ a symmetric positive-definite matrix measuring the cost level; $\Lambda$ is a
multidimensional version of Kyle's lambda, since trading $\Delta x_t$ moves the price by
$\tfrac{1}{2}\Lambda\Delta x_t$. In the special case where costs are proportional to the amount of risk
(Assumption A), $\Lambda = \lambda \Sigma$ for a scalar $\lambda > 0$.

The investor chooses the dynamic strategy $(x_0, x_1, \dots)$ to maximize the present value of all
future expected excess returns, penalized for risk and trading costs,
$$\max_{x_0,x_1,\dots} E_0\!\left[\sum_t (1-\rho)^{t+1}\!\left(x_t^{\top} r_{t+1} - \frac{\gamma}{2}\,x_t^{\top}\Sigma x_t\right) - \frac{(1-\rho)^{t}}{2}\,\Delta x_t^{\top}\Lambda\,\Delta x_t\right],$$
with $\rho \in (0,1)$ the discount factor and $\gamma$ the risk-aversion coefficient. Dynamic
programming with the value function $V(x_{t-1},f_t)$ satisfies the Bellman equation
$$V(x_{t-1},f_t) = \max_{x_t}\left\{ -\frac{1}{2}\Delta x_t^{\top}\Lambda\Delta x_t + (1-\rho)\!\left(x_t^{\top}E_t[r_{t+1}] - \frac{\gamma}{2}x_t^{\top}\Sigma x_t + E_t[V(x_t,f_{t+1})]\right)\right\}.$$
The solution is unique and the value function quadratic,
$$V(x_t,f_{t+1}) = -\frac{1}{2}x_t^{\top}A_{xx}x_t + x_t^{\top}A_{xf}f_{t+1} + \frac{1}{2}f_{t+1}^{\top}A_{ff}f_{t+1} + A_0,$$
with $A_{xx}$ positive definite. Without transaction costs the investor could re-optimize at no cost and
consider only the current opportunity, disregarding signal decay; it is the costs that make the future
relevant, forcing consideration of the optimal portfolio now and at later instants.

## The aim portfolio and the partial trade toward it

The optimal policy has a simple, intuitive form: the investor aims at a certain position — the aim
portfolio — but moves only partially toward it because of transaction costs. Proposition 2 establishes
the optimal portfolio as
$$x_t = x_{t-1} + \Lambda^{-1}A_{xx}\left(aim_t - x_{t-1}\right), \qquad aim_t = A_{xx}^{-1}A_{xf}f_t,$$
trading at a proportional rate $\Lambda^{-1}A_{xx}$ toward the aim. Under Assumption A the trading rate
is the scalar $a/\lambda < 1$, with
$$a = \frac{-(\gamma + \lambda\rho) + \sqrt{(\gamma + \lambda\rho)^2 + 4\gamma\lambda(1-\rho)}}{2(1-\rho)\lambda},$$
so the optimal portfolio is a weighted average of the existing position and the aim,
$$x_t = \left(1 - \frac{a}{\lambda}\right)x_{t-1} + \frac{a}{\lambda}\,aim_t.$$
The weight $a/\lambda$ on the aim (also called the trading rate) measures how far the investor
rebalances toward the target. Rebalancing is always for a fixed fraction of the distance from the aim:
the trading rate is independent of the current portfolio $x_{t-1}$ and its past history. The rate is
higher the lower the transaction costs $\lambda$ (high costs impose slower trading), and increasing in
risk aversion $\gamma$ (greater aversion makes deviating from the aim costlier).

Here the role of transaction costs is precisely to make repositioning gradual rather than instant.
Without them ($\Lambda = 0$) the investor would hold the static optimal portfolio at every instant, the
Markowitz tangency position,
$$Markowitz_t = (\gamma\Sigma)^{-1}B f_t,$$
for the best risk-return ratio every period. But since the return-predicting factors change over time,
the Markowitz portfolio is a moving target. With transaction costs it is not optimal to chase it fully
each step; it is optimal to slow the trading speed and move only partially toward an aim portfolio. The
trade-off is thus between signal adherence (pushing toward the current Markowitz portfolio) and the
turnover cost chasing it would generate — the concern of [portfolio revision](../workflows/portfolio-revision.md).

## Aim in front of the target: weighting signals by persistence

The aim portfolio does not coincide with the current Markowitz portfolio. Because transaction costs
prevent easy repositioning, the investor must look not only at the present opportunity but at those in
future periods, adjusting the target accordingly: *aim in front of the target*. Proposition 3 expresses
the aim as a weighted average of the current Markowitz portfolio and next period's expected aim, with
$z = \gamma/(\gamma + a)$,
$$aim_t = z\,Markowitz_t + (1-z)\,E_t(aim_{t+1}).$$
Iterating forward, the aim is an exponentially weighted average of current and expected future
Markowitz portfolios at all future dates,
$$aim_t = \sum_{\tau = t}^{\infty} z(1-z)^{\tau - t}\, E_t\!\left(Markowitz_\tau\right).$$
The weight $z$ on the current Markowitz portfolio decreases with transaction costs $\lambda$ and
increases with risk aversion $\gamma$: the lower the costs, the more the aim concentrates on the present
opportunity.

The factor dynamics make the weighting explicit. From $\Delta f_{\tau+1} = -\Phi f_\tau +
\varepsilon_{\tau+1}$ the expected factor is $E_t(f_\tau) = (I-\Phi)^{\tau - t} f_t$, and since
$Markowitz_\tau = (\gamma\Sigma)^{-1}B f_\tau$,
$$aim_t = (\gamma\Sigma)^{-1}B\,z\sum_{j=0}^{\infty}\big[(1-z)(I-\Phi)\big]^{j} f_t = (\gamma\Sigma)^{-1}B\,z\big[I - (1-z)(I-\Phi)\big]^{-1} f_t.$$
Using $z = \gamma/(\gamma+a)$ and $1-z = a/(\gamma+a)$ gives $I - (1-z)(I-\Phi) = z\big[I +
(a/\gamma)\Phi\big]$, hence Proposition 4,
$$aim_t = (\gamma\Sigma)^{-1}B\left(I + \frac{a}{\gamma}\Phi\right)^{-1} f_t.$$
The aim is thus the Markowitz portfolio built as if the signals $f$ were rescaled by their
mean-reversion $\Phi$. If $\Phi$ is diagonal, $\Phi = \mathrm{diag}(\phi^1,\dots,\phi^K)$ — the fairly
standard assumption that each factor's dynamics depend only on its own level — the aim simplifies to the
Markowitz portfolio with each factor $f_t^k$ rescaled by its own alpha decay $\phi^k$,
$$aim_t = (\gamma\Sigma)^{-1}B\left(\frac{f_t^{1}}{1 + \phi^{1} a/\gamma},\;\dots,\;\frac{f_t^{K}}{1 + \phi^{K} a/\gamma}\right)^{\!\top}.$$
The rescaling penalizes factors with high alpha decay $\phi^k$ more strongly. This not only shrinks the
position but changes the signals' relative importance: slower-mean-reverting predictors get more weight
in the aim. An investor facing transaction costs must trade more aggressively on persistent signals than
on fast-reverting ones, because the former's benefits accumulate over longer periods and are thus
greater — the same principle as the combined-signal model, obtained here at the position level rather
than the factor-combination level.

The optimal strategy, followed forever, drives the position toward an exponentially weighted average of
past aims. Proposition 5 (position homing in) establishes, under Assumption A,
$$x_t = \sum_{\tau = -\infty}^{t} \frac{a}{\lambda}\left(1 - \frac{a}{\lambda}\right)^{t-\tau} aim_\tau.$$
A special case shows the direct link with the chapter's information horizon. If an investor predicts a
single market's return using past returns — the first signal $f_t^1$ is yesterday's return, the second
$f_t^2$ the day before, and so on for $K$ periods — then today's "yesterday" is tomorrow's "day before,"
so $f_{t+1}^{k} = f_t^{k-1}$ for $k>1$. Then the optimal strategy is
$$x_t = \left(1 - \frac{a}{\lambda}\right)x_{t-1} + \frac{a}{\lambda}\,\frac{\beta}{\sigma^2(1-z)}\sum_{k=1}^{K}\left(1 - z^{K+1-k}\right) f_t^{k},$$
with $z = (\gamma + a)^{-1}a$: the portfolio gives the most weight to the first signal (yesterday's
return), the second to the second, and so on, because the first signal will keep counting for the
longest time, the second for the second longest. The weight on a signal is ordered by the length of its
residual horizon.

## Static versus dynamic and empirical evidence

Weighting by persistence distinguishes the dynamic solution from any static optimization. When the
investor fully discounts the future ($\rho = 1$), the problem becomes static,
$$\max_{x_t}\; x_t^{\top}E_t(r_t) - \frac{\gamma}{2}x_t^{\top}\Sigma x_t - \frac{\lambda}{2}\Delta x_t^{\top}\Sigma\Delta x_t,$$
with solution
$$x_t = \frac{\lambda}{\gamma + \lambda}x_{t-1} + \frac{\gamma}{\gamma + \lambda}(\gamma\Sigma)^{-1}E_t(r_{t+1}) = x_{t-1} + \frac{\gamma}{\gamma + \lambda}\left(Markowitz_t - x_{t-1}\right).$$
This static portfolio differs from the dynamic one in two ways: the weight on the current position
$x_{t-1}$ differs, and, above all, the aim differs, since in the static case the aim is the Markowitz
portfolio. The first flaw — a suboptimal trading rate ignoring the future benefits of the position — can
be corrected by tuning $\lambda$ or $\gamma$. The second cannot: with several predictors, no choice of
$\gamma$ and $\lambda$ recovers the dynamic solution, because static optimization treats all factors
alike while the dynamic one gives more weight to slower-decaying factors. Recovering the dynamic
solution would require altering not only $\gamma$ and $\lambda$ but the expected returns $Bf_t$ by
rescaling the signals per Proposition 4.

The empirical evidence uses fifteen liquid commodity futures over 01/01/1996 – 23/01/2009. Returns are
predicted with a characteristic-based specification where each commodity uses its own past returns over
three horizons — five days, one year, five years — built as rolling Sharpe ratios. The estimated panel
regression is
$$r_{t+1}^{s} = 0.001 + 10.32\,f_t^{5D,s} + 122.34\,f_t^{1Y,s} - 205.59\,f_t^{5Y,s} + u_{t+1}^{s},$$
so prices show continuation at short and medium frequencies and reversal over long horizons. The three
predictors have very different mean reversions,
$$\Delta f_{t+1}^{5D,s} = -0.2519\,f_t^{5D,s} + \varepsilon_{t+1}^{5D,s}, \quad \Delta f_{t+1}^{1Y,s} = -0.0034\,f_t^{1Y,s} + \varepsilon_{t+1}^{1Y,s}, \quad \Delta f_{t+1}^{5Y,s} = -0.0010\,f_t^{5Y,s} + \varepsilon_{t+1}^{5Y,s},$$
corresponding to a half-life (the expected time for half the signal to vanish) of $2.4$ days for the
five-day signal, $206$ days for the one-year signal, and $700$ days for the five-year signal.

Three strategies are compared: the optimal dynamic strategy, the Markowitz portfolio (optimal absent
costs), and a class of static one-period-optimization strategies for ten trading speeds. Gross of costs
the highest Sharpe ratio is the Markowitz portfolio's; net of costs the optimal dynamic portfolio is
best, with a net Sharpe about $20\%$ above the best static strategy's, while the Markowitz portfolio
incurs enormous trading costs. The engine of outperformance is that the dynamic strategy gives less
weight to the five-day signal due to its rapid alpha decay: the static strategy only controls overall
trading speed, which is not enough — it either incurs high costs chasing the fleeting target or trades
so slowly it fails to capture the return — while the dynamic strategy trades relatively fast but mainly
follows the more persistent signals.

Finally, the response to new information clarifies the mechanics. Following a shock to a predictor, the
Markowitz portfolio jumps immediately and then mean-reverts at the alpha-decay speed; the optimal
portfolio rises more slowly to minimize trading costs and — exiting the position equally slowly — may
end up holding a larger position than Markowitz. For the five-year-signal shock the effects are slower
and of opposite sign, since five-year returns predict future reversals. The optimal position is thus a
smoother version of the Markowitz portfolio, reducing its trading costs while capturing most of its
excess return.
