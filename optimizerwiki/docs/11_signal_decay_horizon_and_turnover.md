---
title: "Signal Decay, Horizon, and Turnover"
chapter: 11
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-10
---

> [!abstract] Summary
> The chapter establishes that a predictive signal loses power over time and that the speed of this decay — measured by the signal's autocorrelation and by the information horizon — determines the turnover the signal imposes on the portfolio. It shows how averaging lagged factors reduces turnover at the cost of some signal, and formalizes the trade-off with a model that maximizes the information ratio subject to a persistence constraint. It finally establishes that, with predictable returns and quadratic transaction costs, the optimal policy does not chase the static optimal portfolio but an aim portfolio that weights signals by their persistence, moving toward it only partially.

## Information Decay and the Signal Horizon

A quantitative factor produces, in each period, a cross-sectional forecast of security returns; but the information it embeds is not bound to manifest only in the period immediately following its observation, nor does it remain confined to it. To measure the persistence of information it is useful to distinguish, following [QianHuaSorensen2006], three information coefficients (IC), all defined as cross-sectional correlations between factor values and security returns.

The first is the conventional IC, which pairs the factor known at time $t$ with the return realized from time $t$ to time $t+1$:

$$IC_{t,t} = \mathrm{corr}\!\left(F_t, R_t\right).$$

The second is the lagged IC, which pairs a factor already known at time $t-l$ with the subsequent return from time $t$ to time $t+1$:

$$IC_{t-l,t} = \mathrm{corr}\!\left(F_{t-l}, R_t\right).$$

The lagged IC directly measures information decay: a factor that has a positive conventional IC at the time of its observation may retain, one or more periods later, a still-positive IC — a sign that the information persists — or see it fade rapidly toward zero.

The third is the horizon IC, which pairs the factor known at time $t$ with the cumulative return over a forward horizon of $h$ periods:

$$IC_t^{h} = \mathrm{corr}\!\left(F_t, R_{t,t+h}\right), \qquad h = 0,1,\dots,H.$$

The information horizon is thus the measure of how far into the future the signal retains predictive power: as $h$ grows, the information the factor still manages to capture accumulates in the cumulative return.

The three coefficients are not independent. Since the cumulative return $R_{t,t+h}$ is the sum of the returns of the individual periods from $t$ to $t+h$, [QianHuaSorensen2006] establishes the approximate relation

$$IC_t^{h} \approx \frac{IC_{t,t} + IC_{t-1,t} + \cdots + IC_{t-h,t}}{\sqrt{h+1}} = \mathrm{avg}(IC)\,\sqrt{h+1},$$

where the last equality holds when the lagged coefficients are roughly constant and equal to their average value $\mathrm{avg}(IC)$. The horizon IC thus grows as the square root of the number of periods covered, a form analogous to the fundamental law of active management [Grinold1989], in which the information ratio scales with the square root of breadth. The relation shows that horizon and decay are two faces of the same property: a factor whose lagged ICs remain positive for a long time possesses an extended information horizon and a horizon IC that keeps growing; a factor whose lagged ICs quickly become zero, or change sign, has a short horizon and a horizon IC that flattens out or reverses.

An empirical comparison between two factors — price momentum (PM, measured by the nine-month return lagged by one) and earnings-to-price (E2P) — makes the distinction concrete. The average IC of momentum starts from a high value at lag zero and decays rapidly as the lag increases, approaching zero after just a few lags; the average IC of E2P starts lower but remains nearly stable at subsequent lags [QianHuaSorensen2006]. Correspondingly, the information ratio of the momentum factor collapses with the lag, while that of E2P is maintained. The value signal retains information over a long horizon; the momentum signal exhausts it quickly.

## Signal Autocorrelation and Induced Turnover

A signal that decays quickly changes rapidly from one period to the next; and since the portfolio is built from the signal, a volatile signal imposes heavy repositioning. The measure of repositioning is turnover, defined as

$$T = \frac{1}{2}\sum_{i=1}^{N} \left| w_i^{t+1} - w_i^{t} \right|,$$

where $w_i^{t}$ is the weight of security $i$ in the portfolio at time $t$ and the sum runs over the $N$ securities. The quantity that links turnover to signal decay is the factor's forecast autocorrelation, that is, the cross-sectional correlation between signal values in two consecutive periods:

$$\rho_f = \mathrm{corr}\!\left(\tilde F^{\,t+1}, \tilde F^{\,t}\right).$$

For an optimized portfolio, [QianHuaSorensen2006], drawing on [QianEtAl2004], reports the expression

$$T = \sqrt{\frac{N}{\pi}}\;\sigma_{model}\,\sqrt{1-\rho_f}\;E\!\left(\frac{1}{\sigma}\right),$$

in which $\sigma_{model}$ is the target tracking error and $\sigma$ the specific risk of the securities. Turnover is higher the:

- higher the target tracking error;
- larger the number of securities, in proportion to the square root of $N$;
- lower the factor's autocorrelation, that is, the cross-sectional correlation between consecutive forecasts;
- lower the average specific risk of the securities.

It is the third point that links turnover to decay: the factor $\sqrt{1-\rho_f}$ grows as autocorrelation decreases. A slow-decaying signal has $\rho_f$ close to one and generates little turnover; a fast-decaying signal has low $\rho_f$ and generates high turnover.

The empirical classification of factors confirms this link. Ranking the average autocorrelation $\rho_f$ by category, [QianHuaSorensen2006] reports high values for value factors (E2PFY0 $0{,}96$; B2P $0{,}93$; CFO2EV $0{,}84$), low values for momentum factors (EarnRev9 $0{,}64$; Ret9Monx1 $0{,}60$; LtgRev9 $0{,}37$), and intermediate values for quality factors (RNOA $0{,}89$; XF $0{,}76$; NCOinc $0{,}80$). It follows that momentum factors, with the lowest autocorrelation, have the highest turnover; value factors, with the highest autocorrelation, have the lowest turnover; quality factors fall in between.

## Responsiveness Versus Stability: Moving Averages of Factors

The conflict is now explicit: closely following a fast-decaying signal means updating the portfolio frequently, and hence bearing high turnover. Among the ways to slow down turnover, [QianHuaSorensen2006] lists a direct turnover constraint, a shift in composition toward more value and less momentum, and the use of a moving average of the factors.

The moving average combines the current value of the factor with its lagged values. In the two-term form, MA(2),

$$F_{ma}^{\,t} = v_0\,F^{\,t} + v_1\,F^{\,t-1}.$$

Averaging the factor with its lag increases its autocorrelation, and therefore reduces its turnover. With a factor whose lag-one autocorrelation is $\rho_f(1)=0{,}90$ (and $\rho_f(2)=0{,}81$), the autocorrelation of the moving average reaches a maximum of about $0{,}95$ at equal weights $v_1=0{,}5$ [QianHuaSorensen2006]. Since turnover scales with $\sqrt{1-\rho_f}$, the turnover of the moving average equals

$$\frac{\sqrt{1-0{,}95}}{\sqrt{1-0{,}90}} \approx 71\%$$

of the turnover of the raw factor.

The reduction in turnover, however, is not free. Lagged factors predict future returns more weakly than current ones — this is the information decay documented by the lagged ICs — so averaging entails a reduction in alpha. The question is what the correct trade-off is between lower turnover and reduced alpha.

[QianHuaSorensen2006] answers by formulating the choice as an optimization problem. The combination weights $v=(v_1,\dots,v_M)$ of the factors are chosen to maximize the model's information ratio,

$$IR = \frac{\mathrm{avg}(IC)}{\mathrm{std}(IC)} = \frac{v' \cdot \overline{IC}}{\sqrt{\,v' \cdot S_{IC} \cdot v\,}},$$

where $\overline{IC}$ is the vector of average ICs and $S_{IC}$ the covariance matrix of the ICs; the model's inputs are the average IC, the standard deviation of the ICs, and their correlations, and the optimal solution exists in closed form. For two factors — E2P and PM, with an IC correlation of $-0{,}4$ — the unconstrained optimal weights turn out to be PM $36\%$ and E2P $64\%$.

To control turnover, the model extends the set of regressors to include, alongside the current factors, their lagged values, so as to form a combined moving-average forecast

$$F_{c,ma}^{\,t} = v_{01}F_1^{\,t} + v_{02}F_2^{\,t} + v_{11}F_1^{\,t-1} + v_{12}F_2^{\,t-1} + \cdots,$$

and maximizes the information ratio subject to the constraint that the autocorrelation of the combined forecast equal a target level:

$$\text{Maximize}\quad IR = \frac{v' \cdot \overline{IC}}{\sqrt{\,v' \cdot S_{IC} \cdot v\,}} \qquad \text{subject to}\quad \rho_{f_{c,ma}} = \rho_{target}.$$

The solution as $\rho_{target}$ varies traces out the trade-off between responsiveness and stability. As the target autocorrelation increases from $0{,}85$ to $0{,}97$, the model's turnover decreases monotonically, while the information ratio first increases, reaches a maximum of $2{,}39$ at $\rho_f = 0{,}89$, and then decreases [QianHuaSorensen2006]. In terms of aggregate weights, at lower autocorrelation levels the combined forecast loads entirely on the current factors ($w_0 = 100\%$); only when a higher autocorrelation is required does the weight progressively shift to the lagged factors ($w_1, w_2, w_3$), at the expense of alpha. The point of maximum information ratio defines the level of turnover economically justified by the signal.

## The Dynamic Trading Model with Predictable Returns

The same trade-off — adherence to the signal versus the cost of turnover — can be addressed at the level of portfolio construction itself, rather than merely factor combination. [GarleanuPedersen2013] derives the optimal portfolio policy when returns are predictable from signals with different mean-reversion speeds and trading is costly.

Consider an economy with $S$ securities. The price excess returns between $t$ and $t+1$, collected in the $S\times 1$ vector $r_{t+1}$, are generated by

$$r_{t+1} = B f_t + u_{t+1},$$

where $f_t$ is a $K\times 1$ vector of factors that predict returns, known to the investor already at time $t$; $B$ is the $S\times K$ matrix of factor loadings; and $u_{t+1}$ is unpredictable zero-mean noise, with $\mathrm{var}_t(u_{t+1}) = \Sigma$. The factors evolve according to

$$\Delta f_{t+1} = -\Phi f_t + \varepsilon_{t+1},$$

where $\Delta f_{t+1} = f_{t+1} - f_t$, the $K\times K$ matrix $\Phi$ collects the factors' mean-reversion coefficients, and $\varepsilon_{t+1}$ is the shock to the predictors, with $\mathrm{var}_t(\varepsilon_{t+1}) = \Omega$. The matrix $\Phi$ is the continuous analogue of signal decay: a factor with small $\Phi$ reverts slowly to the mean (slow alpha decay, persistent signal), one with large $\Phi$ reverts quickly (fast alpha decay, non-persistent signal). With different predictors — for example a momentum signal predicting a high return next month and a value signal predicting a good return over the year — the mean-reversion speeds differ [GarleanuPedersen2013].

Trading is costly: the transaction cost associated with trading $\Delta x_t = x_t - x_{t-1}$ shares is quadratic,

$$TC(\Delta x_t) = \frac{1}{2}\,\Delta x_t^{\top} \Lambda \, \Delta x_t,$$

with $\Lambda$ a symmetric positive-definite matrix measuring the level of costs; $\Lambda$ is a multidimensional version of Kyle's lambda, since trading $\Delta x_t$ moves the price by $\tfrac{1}{2}\Lambda\Delta x_t$. In the special case where costs are proportional to the amount of risk (Assumption A), $\Lambda = \lambda \Sigma$ for a scalar $\lambda > 0$.

The investor chooses the dynamic strategy $(x_0, x_1, \dots)$ to maximize the present value of all future expected excess returns, penalized for risk and trading costs:

$$\max_{x_0,x_1,\dots} E_0\!\left[\sum_t (1-\rho)^{t+1}\!\left(x_t^{\top} r_{t+1} - \frac{\gamma}{2}\,x_t^{\top}\Sigma x_t\right) - \frac{(1-\rho)^{t}}{2}\,\Delta x_t^{\top}\Lambda\,\Delta x_t\right],$$

where $\rho \in (0,1)$ is the discount factor and $\gamma$ the risk aversion coefficient. The problem is solved via dynamic programming by introducing the value function $V(x_{t-1},f_t)$, which satisfies the Bellman equation

$$V(x_{t-1},f_t) = \max_{x_t}\left\{ -\frac{1}{2}\Delta x_t^{\top}\Lambda\Delta x_t + (1-\rho)\!\left(x_t^{\top}E_t[r_{t+1}] - \frac{\gamma}{2}x_t^{\top}\Sigma x_t + E_t[V(x_t,f_{t+1})]\right)\right\}.$$

The solution is unique and the value function is quadratic:

$$V(x_t,f_{t+1}) = -\frac{1}{2}x_t^{\top}A_{xx}x_t + x_t^{\top}A_{xf}f_{t+1} + \frac{1}{2}f_{t+1}^{\top}A_{ff}f_{t+1} + A_0,$$

with $A_{xx}$ positive definite. In the absence of transaction costs the investor could re-optimize at zero cost and consider only the current opportunity, without regard for signal decay; it is the costs that make the future relevant, forcing the investor to consider the optimal portfolio both now and at subsequent instants [GarleanuPedersen2013].

## The Aim Portfolio and Partial Trading Toward It

The optimal policy has a simple, intuitive form: the investor aims at a certain position — the aim portfolio — but moves toward it only partially because of transaction costs. Proposition 2 of [GarleanuPedersen2013] establishes that the optimal portfolio is

$$x_t = x_{t-1} + \Lambda^{-1}A_{xx}\left(aim_t - x_{t-1}\right), \qquad aim_t = A_{xx}^{-1}A_{xf}f_t,$$

that is, one trades at a rate proportional to the matrix $\Lambda^{-1}A_{xx}$, in the direction of the aim portfolio. Under Assumption A the trading rate is the scalar $a/\lambda < 1$, with

$$a = \frac{-(\gamma + \lambda\rho) + \sqrt{(\gamma + \lambda\rho)^2 + 4\gamma\lambda(1-\rho)}}{2(1-\rho)\lambda}.$$

The optimal portfolio is then a weighted average of the existing position and the aim:

$$x_t = \left(1 - \frac{a}{\lambda}\right)x_{t-1} + \frac{a}{\lambda}\,aim_t.$$

The weight $a/\lambda$ on the aim — which [GarleanuPedersen2013] also calls the trading rate — measures how far the investor rebalances toward the target. Rebalancing always occurs by a fixed fraction of the distance from the aim: the trading rate is independent of the current portfolio $x_{t-1}$ and its past history. The rate is higher the lower the transaction costs $\lambda$: high costs require trading more slowly. The rate instead increases in risk aversion $\gamma$, since greater aversion makes deviating from the aim more costly.

In this model the role of transaction costs is precisely to make repositioning gradual rather than instantaneous. In their absence ($\Lambda = 0$) the investor would hold at every instant the static optimal portfolio, that is, the Markowitz portfolio — the tangency position, consistent with [Markowitz1952] —

$$Markowitz_t = (\gamma\Sigma)^{-1}B f_t,$$

so as to obtain the best risk-return trade-off every period. But since the factors that predict returns change over time, the Markowitz portfolio is a moving target. With transaction costs it is not optimal to chase it entirely at every step; it is optimal to slow down the trading speed and move only partially toward an aim portfolio. The trade-off is thus between adherence to the signal (which would push toward the current Markowitz portfolio) and the cost of turnover that chasing it would generate (cf. the treatment in [[07 Revisione di portafoglio]]).

## Aiming in Front of the Target: Weighting Signals by Persistence

The aim portfolio does not coincide with the current Markowitz portfolio. Since transaction costs prevent changing position easily, the investor must look not only at the present opportunity but also at the one that will present itself in future periods, and adjust the target accordingly: one must aim in front of the target. Proposition 3 of [GarleanuPedersen2013] expresses the aim as a weighted average of the current Markowitz portfolio and the expected aim of the following period, setting $z = \gamma/(\gamma + a)$:

$$aim_t = z\,Markowitz_t + (1-z)\,E_t(aim_{t+1}).$$

Iterating this recursion forward, the aim is written as an exponentially weighted average of current and expected Markowitz portfolios at all future dates:

$$aim_t = \sum_{\tau = t}^{\infty} z(1-z)^{\tau - t}\, E_t\!\left(Markowitz_\tau\right).$$

The weight $z$ on the current Markowitz portfolio decreases with transaction costs $\lambda$ and increases with risk aversion $\gamma$: the lower the costs, the more the aim concentrates on the present opportunity.

The dynamics of the factors make the weighting explicit. From the evolution law $\Delta f_{\tau+1} = -\Phi f_\tau + \varepsilon_{\tau+1}$ it follows that the expected factor is $E_t(f_\tau) = (I-\Phi)^{\tau - t} f_t$, and since $Markowitz_\tau = (\gamma\Sigma)^{-1}B f_\tau$,

$$aim_t = (\gamma\Sigma)^{-1}B\,z\sum_{j=0}^{\infty}\big[(1-z)(I-\Phi)\big]^{j} f_t = (\gamma\Sigma)^{-1}B\,z\big[I - (1-z)(I-\Phi)\big]^{-1} f_t.$$

Using $z = \gamma/(\gamma+a)$ and $1-z = a/(\gamma+a)$ one has $I - (1-z)(I-\Phi) = z\big[I + (a/\gamma)\Phi\big]$, from which Proposition 4:

$$aim_t = (\gamma\Sigma)^{-1}B\left(I + \frac{a}{\gamma}\Phi\right)^{-1} f_t.$$

The aim is thus the Markowitz portfolio constructed as if the signals $f$ were rescaled according to their mean reversion $\Phi$. If the matrix $\Phi$ is diagonal, $\Phi = \mathrm{diag}(\phi^1,\dots,\phi^K)$ — the fairly standard assumption that each factor's dynamics depends only on its own level — the aim simplifies to the Markowitz portfolio with each factor $f_t^k$ rescaled according to its own alpha decay $\phi^k$:

$$aim_t = (\gamma\Sigma)^{-1}B\left(\frac{f_t^{1}}{1 + \phi^{1} a/\gamma},\;\dots,\;\frac{f_t^{K}}{1 + \phi^{K} a/\gamma}\right)^{\!\top}.$$

The rescaling penalizes more heavily factors with high alpha decay $\phi^k$. This not only reduces the size of the position, but also changes the relative importance of the signals: predictors with slower mean reversion receive more weight in the aim [GarleanuPedersen2013]. An investor facing transaction costs must trade more aggressively on persistent signals than on fast-mean-reverting ones, because the benefits of the former accumulate over longer periods and are therefore greater. This is the same principle as the combined-signal model — giving more weight to slow-decaying factors — obtained here at the level of the position rather than of factor combination.

The optimal strategy, followed indefinitely, makes the position converge toward an exponentially weighted average of past aims. Proposition 5 (position homing in) establishes that, under Assumption A,

$$x_t = \sum_{\tau = -\infty}^{t} \frac{a}{\lambda}\left(1 - \frac{a}{\lambda}\right)^{t-\tau} aim_\tau.$$

A special case shows the direct link with the information horizon of the chapter. If an investor forecasts the return of a single market using past returns — the first signal $f_t^1$ is yesterday's return, the second $f_t^2$ the day before yesterday's, and so on for $K$ periods — then today's "yesterday" is tomorrow's "day before yesterday," so that $f_{t+1}^{k} = f_t^{k-1}$ for $k>1$. In this case the optimal strategy takes the form

$$x_t = \left(1 - \frac{a}{\lambda}\right)x_{t-1} + \frac{a}{\lambda}\,\frac{\beta}{\sigma^2(1-z)}\sum_{k=1}^{K}\left(1 - z^{K+1-k}\right) f_t^{k},$$

with $z = (\gamma + a)^{-1}a$: the portfolio gives the greatest weight to the first signal (yesterday's return), the second-greatest weight to the second signal, and so on, because the first signal will keep mattering for the longest time, the second for the second-longest time [GarleanuPedersen2013]. The weight attributed to a signal is ordered by the length of its remaining horizon.

## Static Versus Dynamic and Empirical Evidence

Weighting by persistence distinguishes the dynamic solution from any static optimization. When the investor fully discounts the future ($\rho = 1$), the problem becomes static and reduces to

$$\max_{x_t}\; x_t^{\top}E_t(r_t) - \frac{\gamma}{2}x_t^{\top}\Sigma x_t - \frac{\lambda}{2}\Delta x_t^{\top}\Sigma\Delta x_t,$$

with solution

$$x_t = \frac{\lambda}{\gamma + \lambda}x_{t-1} + \frac{\gamma}{\gamma + \lambda}(\gamma\Sigma)^{-1}E_t(r_{t+1}) = x_{t-1} + \frac{\gamma}{\gamma + \lambda}\left(Markowitz_t - x_{t-1}\right).$$

This static portfolio differs from the dynamic one in two ways: the weight on the current position $x_{t-1}$ is different, and above all the aim is different, since in the static case the aim is the Markowitz portfolio. The first flaw — a suboptimal trading rate, which does not account for the future benefits of the position — can be corrected by adjusting $\lambda$ or $\gamma$. The second flaw cannot: with multiple predictors, no choice of risk aversion $\gamma$ and cost $\lambda$ recovers the dynamic solution, because static optimization treats all factors the same way, while the dynamic one gives more weight to slower-decaying factors [GarleanuPedersen2013]. Recovering the dynamic solution would require modifying not only $\gamma$ and $\lambda$, but also the expected returns $Bf_t$ by rescaling the signals according to Proposition 4.

The empirical test is conducted on fifteen liquid commodity futures, over the period 01/01/1996 – 23/01/2009. Returns are forecast with a characteristics-based specification, in which each commodity uses its own past returns over three horizons — five days, one year, and five years — constructed as rolling Sharpe ratios. The estimated panel regression is

$$r_{t+1}^{s} = 0{,}001 + 10{,}32\,f_t^{5D,s} + 122{,}34\,f_t^{1Y,s} - 205{,}59\,f_t^{5Y,s} + u_{t+1}^{s},$$

from which prices show continuation at short and medium frequencies and reversal over long horizons. The three predictors have very different mean-reversion speeds:

$$\Delta f_{t+1}^{5D,s} = -0{,}2519\,f_t^{5D,s} + \varepsilon_{t+1}^{5D,s}, \quad \Delta f_{t+1}^{1Y,s} = -0{,}0034\,f_t^{1Y,s} + \varepsilon_{t+1}^{1Y,s}, \quad \Delta f_{t+1}^{5Y,s} = -0{,}0010\,f_t^{5Y,s} + \varepsilon_{t+1}^{5Y,s},$$

corresponding to a half-life (the expected time for half the signal to vanish) of $2{,}4$ days for the five-day signal, of $206$ days for the one-year signal, and of $700$ days for the five-year signal [GarleanuPedersen2013].

Three strategies are compared: the optimal dynamic strategy, the Markowitz portfolio (optimal in the absence of costs), and a class of strategies based on static single-period optimization, for ten different trading speeds. Before costs, the highest Sharpe ratio is that of the Markowitz portfolio; after costs, however, the optimal dynamic portfolio is the best, with a net Sharpe ratio about $20\%$ higher than that of the best static strategy, while the Markowitz portfolio incurs enormous trading costs. The engine of the outperformance is that the dynamic strategy gives less weight to the five-day signal because of its fast alpha decay: the static strategy tries only to control the overall trading speed, but this is not enough — it either incurs high costs chasing the fleeting target, or trades so slowly that it fails to capture the return — whereas the dynamic strategy trades relatively quickly but mainly follows the more persistent signals [GarleanuPedersen2013].

Finally, the response to new information clarifies the mechanics. Following a shock to a predictor, the Markowitz portfolio jumps immediately and then reverts to the mean at the speed of the alpha decay; the optimal portfolio increases more slowly, so as to minimize trading costs, and — since it also exits the position just as slowly — may eventually hold a larger position than the Markowitz one. For the shock to the five-year signal the effects are slower and of opposite sign, since five-year returns predict future reversals. The optimal position thus turns out to be a smoother version of the Markowitz portfolio, which reduces its trading costs while still capturing most of its excess return [GarleanuPedersen2013].

## References

- **[GarleanuPedersen2013]** Gârleanu, N., & Pedersen, L. H. (2013). Dynamic Trading with Predictable Returns and Transaction Costs. The Journal of Finance, 68(6), 2309–2340.
- **[Grinold1989]** Grinold, R. C. (1989). The Fundamental Law of Active Management. The Journal of Portfolio Management, 15(3), 30–37.
- **[Markowitz1952]** Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77–91.
- **[QianEtAl2004]** Qian, E., Hua, R., & Tilney, J. (2004). Portfolio Turnover of Quantitatively Managed Portfolios. Proceedings of the 2nd IASTED International Conference on Financial Engineering and Applications, Cambridge, MA.
- **[QianHuaSorensen2006]** Qian, E., Hua, R., & Sorensen, E. (2006). Information Horizon, Portfolio Turnover, and Optimal Alpha Model. Northfield Conference, October 2006. See also Qian, E. E., Hua, R. H., & Sorensen, E. H. (2007). Quantitative Equity Portfolio Management: Modern Techniques and Applications. Chapman and Hall/CRC.
