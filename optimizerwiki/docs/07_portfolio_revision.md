---
title: "Portfolio Review"
chapter: 7
tags:
  - optimizer
  - tipo/capitolo
grounded: true
generated: 2026-07-10
---

> [!abstract] Summary
> The chapter shows why an optimal portfolio selected at a given instant generally ceases to be optimal at subsequent instants, and how this forces the shift from static to dynamic management through portfolio review. It formalizes the condition of desirability for revision in the presence of capital gains tax and presents two classical models for identifying the potentially revised portfolio — that of Smith (1967) and that of Stone and Hill (1979) — closing with the characteristics a good revision model must possess.

## Between Static Management and Dynamic Management

The mean-variance model discussed in [[02 Selezione media-varianza]] indicates how to select a portfolio at a single point in time [Smith1967]. In an ideal single-period economy, developing from $t$ to $t+\Delta t$, this prescription is sufficient: the economic agent selects their optimal portfolio at $t$ and "releases" it, that is, liquidates it, at $t+\Delta t$.

In reality, however, the economic agent operates in an economy that is (at least) multi-period, articulated over the intervals from $t$ to $t+\Delta t$, from $t+\Delta t$ to $t+2\Delta t$, and in general from $t+i\Delta t$ to $t+(i+1)\Delta t$. In this context the agent:

- selects their optimal portfolio at $t$;
- can no longer "release" it at $t+\Delta t$;
- must manage it at $t+\Delta t$, $t+2\Delta t$, \ldots, $t+i\Delta t$, \ldots

The question then arises: in a multi-period economy, what does managing the portfolio at the instants $t+\Delta t$, $t+2\Delta t$, \ldots consist of?

In an *ideal* multi-period economy the means, variances, and linear correlation coefficients relating to the various investment choices remain the same at any point in time considered. Consequently, the optimal portfolio selected at $t$ remains optimal also at $t+\Delta t$, at $t+2\Delta t$, \ldots, at $t+i\Delta t$, \ldots, and dynamic management coincides with static management.

But, in general, in reality these means, variances, and linear correlation coefficients change as the point in time considered changes. Therefore the optimal portfolio selected at $t$ may no longer be either optimal or efficient at $t+\Delta t$, and/or at $t+2\Delta t$, \ldots, and/or at $t+i\Delta t$, \ldots. This is the boundary that separates static management from dynamic management and that makes portfolio review necessary.

## The Temporal Instability of the Optimal Portfolio

The sensitivity of the optimal portfolio to statistical parameters, and hence to the time window over which they are estimated, can be appreciated with a concrete example. Let $\mathbb{X}$ be the set of investment choices consisting of three equities from the Italian stock market:

- $X_1 =$ Alleanza Assicurazioni;
- $X_2 =$ Fondiaria–Sai;
- $X_3 =$ Unicredito Italiano.

Estimate the means, variances, and linear correlation coefficients of the rates of return over two daily windows offset by a single day: column A covers the period from August 22, 2006 to September 20, 2006, column B the period from August 23, 2006 to September 21, 2006. The values are as follows.

| Parameters | A | B |
|---|---|---|
| $r_1$ | 4.437\text{E}{-}04 | 4.037\text{E}{-}04 |
| $r_2$ | 6.648\text{E}{-}04 | 9.901\text{E}{-}04 |
| $r_3$ | 1.144\text{E}{-}03 | 1.213\text{E}{-}03 |
| $\sigma_1^2$ | 1.231\text{E}{-}04 | 1.229\text{E}{-}04 |
| $\sigma_2^2$ | 1.168\text{E}{-}04 | 1.514\text{E}{-}04 |
| $\sigma_3^2$ | 7.802\text{E}{-}05 | 7.794\text{E}{-}05 |
| $\rho_{1,2}$ | 1.591\text{E}{-}01 | 1.600\text{E}{-}01 |
| $\rho_{1,3}$ | $-7.665\text{E}{-}02$ | $-8.291\text{E}{-}02$ |
| $\rho_{2,3}$ | 4.074\text{E}{-}01 | 4.150\text{E}{-}01 |

Imposing a target expected rate of return $\pi = 0.00041$ in the base mean-variance portfolio problem [Markowitz1952], the optimal investment percentages turn out to be:

| Percentages | A | B |
|---|---|---|
| $x_1^*$ | 0.69193 | 0.96600 |
| $x_2^*$ | 0.52029 | 0.09500 |
| $x_3^*$ | $-0.21223$ | $-0.06100$ |

In the $\mathrm{Var}(X)$–$E(X)$ plane the efficient frontier computed on the column-A data is the upper branch of the mean-variance hyperbola: for a fixed level of variance, there exist portfolio combinations with lower expected return (inefficient frontier) than the optimal ones (efficient frontier).

Suppose now that the economic agent wants the portfolio to achieve an expected rate of return equal to $0.00041$ both from $t$ (September 20, 2006) to $t+\Delta t$ (September 21, 2006), and from $t+\Delta t$ to $t+2\Delta t$ (September 22, 2006). Moving from the column-A percentages to the column-B ones then requires, at $t+\Delta t$:

- **buying** $X_1$ (Alleanza Assicurazioni) for a percentage of capital equal to $0.27407 = 0.96600 - 0.69193$;
- **selling** $X_2$ (Fondiaria–Sai) for a percentage of capital equal to $0.42529 = 0.09500 - 0.52029$;
- **buying** $X_3$ (Unicredito Italiano) for a percentage of capital equal to $0.15123 = -0.06100 - (-0.21223)$.

The example concretely shows how, by changing the parameter estimation window by even a single day, the optimal portfolio changes significantly, requiring a revision of the portfolio's composition.

## Portfolio Review and Its Desirability

The set of portfolio adjustments made at $t+\Delta t$ is called **portfolio review** (in a frictionless market). In reality, however, these adjustments have a cost. In particular, revision costs may sometimes be greater than the increase in the expected rate of return of the revised portfolio: for this reason portfolio review is not always desirable.

The operational question then arises: how should a worthwhile portfolio review be carried out? The answer proceeds in three steps:

1. first, identify the main types of revision costs;
2. then, identify the potentially revised portfolio;
3. finally, verify whether the portfolio review is worthwhile.

These three steps organize the sections that follow.

## The Main Types of Revision Costs

There are three main types of costs that the economic agent may have to bear at the instants $t+\Delta t$, $t+2\Delta t$, \ldots, $t+i\Delta t$, \ldots

**1. Data collection and processing costs.** These are the costs of collecting new data, updating the corresponding data set, appropriately transforming this new data (for example from prices to percentage or logarithmic returns), and computing the quantities of interest (means, variances, linear correlation coefficients, and so on). This type of cost is **fixed**: the economic agent must bear it at every point in time, whether or not they revise their portfolio.

**2. Transaction costs.** These are the costs of trading the investment choices under consideration, that is, the costs to be paid to a professional intermediary (bank, broker, \ldots) to buy and/or (short-)sell the investment choices. This type of cost is **variable**: the economic agent must bear it only at the points in time when they revise their portfolio.

**3. Capital gains tax.** This is the tax levied on capital gains, that is, the tax to be paid on any realized increase in the value of the portfolio, expressed in monetary terms. Technically, capital gains taxation is not a cost; in this context, however, it can be regarded as having the same status as a cost. This status is clarified by the formalization in the following section.

## Condition of Desirability in the Presence of Capital Gains Tax

To fix ideas, consider a simple two-asset portfolio and introduce the following notation:

- $\mathbf{z}^*_\tau = (z^*_{\tau,1}, z^*_{\tau,2})$: the two-asset portfolio, expressed in terms of investment choices, selected or revised at $\tau$ (the subscript relating to the point in time makes the notation unambiguous);
- $P_{\tau,i}$: the price at $\tau$ of the $i$-th investment choice;
- $r_{\tau,i}$: the expected rate of return at $\tau$ of the $i$-th investment choice;
- $\Delta_{\tau,i}$: the number of units bought or (short-)sold at $\tau$ of the $i$-th investment choice;
- $t_{cg} \in [0,1)$: the percentage of the realized capital gain to be paid as tax.

At $t$ the economic agent selects their optimal portfolio $\mathbf{z}^*_t$. At $t+\Delta t$ they can adopt one of the following two strategies.

**Strategy 1 — No revision.** Do not carry out any revision, so that $\mathbf{z}^*_{t+\Delta t} = \mathbf{z}^*_t$. The expected return, expressed in monetary terms, of the unrevised portfolio is

$$r_{NR,\,t+\Delta t} = z^*_{t,1}\,P_{t+\Delta t,1}\,r_{t+\Delta t,1} + z^*_{t,2}\,P_{t+\Delta t,2}\,r_{t+\Delta t,2}.$$

**Strategy 2 — Revision.** Carry out a revision consisting of:

- divesting from the first investment choice the monetary amount $\Delta_{t+\Delta t,1}\,P_{t+\Delta t,1}$;
- paying the tax on any realized capital gain, that is, paying $t_{cg}\max\{\Delta_{t+\Delta t,1}(P_{t+\Delta t,1} - P_{t,1}),\,0\}$;
- investing the remaining monetary amount, $\Delta_{t+\Delta t,1}\,P_{t+\Delta t,1} - t_{cg}\max\{\Delta_{t+\Delta t,1}(P_{t+\Delta t,1} - P_{t,1}),\,0\}$, in the second investment choice.

The expected return, expressed in monetary terms, of the revised portfolio is therefore

$$r_{R,\,t+\Delta t} = \left(z^*_{t,1} - \Delta_{t+\Delta t,1}\right) P_{t+\Delta t,1}\, r_{t+\Delta t,1} + \Big(z^*_{t,2}\,P_{t+\Delta t,2} + \Delta_{t+\Delta t,1}\,P_{t+\Delta t,1} - t_{cg}\max\{\Delta_{t+\Delta t,1}(P_{t+\Delta t,1}-P_{t,1}),\,0\}\Big)\, r_{t+\Delta t,2}.$$

**Derivation of the desirability condition.** Revision is worthwhile if the expected return of the revised portfolio is greater than that of the unrevised portfolio, that is, if

$$r_{R,\,t+\Delta t} > r_{NR,\,t+\Delta t}.$$

Subtracting the two expressions term by term, the terms in $z^*_{t,1}\,P_{t+\Delta t,1}\,r_{t+\Delta t,1}$ and $z^*_{t,2}\,P_{t+\Delta t,2}\,r_{t+\Delta t,2}$ cancel out, and the inequality reduces to

$$\Delta_{t+\Delta t,1}\,P_{t+\Delta t,1}\,(r_{t+\Delta t,2} - r_{t+\Delta t,1}) > t_{cg}\max\{\Delta_{t+\Delta t,1}(P_{t+\Delta t,1}-P_{t,1}),\,0\}\, r_{t+\Delta t,2} \;\ge\; 0,$$

where the last inequality is non-negative since $t_{cg}\ge 0$, the $\max\{\cdot,0\}$ operator is non-negative, and $r_{t+\Delta t,2}\ge 0$. Dividing by $\Delta_{t+\Delta t,1}\,P_{t+\Delta t,1}$ gives the final form of the condition:

$$r_{t+\Delta t,2} > r_{t+\Delta t,1} + \frac{t_{cg}\max\{\Delta_{t+\Delta t,1}(P_{t+\Delta t,1}-P_{t,1}),\,0\}}{\Delta_{t+\Delta t,1}\,P_{t+\Delta t,1}}.$$

Revision is therefore worthwhile only if the expected rate of return of the asset into which capital is transferred exceeds that of the liquidated asset by an amount at least equal to the incidence, per unit of divested capital, of the tax on the realized capital gain.

Note that this type of "cost" depends on the occurrence of a realized capital gain. When this occurs, the type of "cost" under consideration is **variable**; it follows that the economic agent does not have to bear it at every instant when they revise their portfolio, but only when a capital gain is realized on the liquidated position. It is in this sense that the capital gains tax, while not technically a cost, takes on in this context the status of a variable cost.

## The Smith (1967) Model

There are not many approaches for identifying the potentially revised portfolio at $t+\Delta t$; two classical ones are presented here. The first is that of Smith [Smith1967], whose stated purpose is to extend an existing methodology for portfolio selection on an intertemporal basis, through an adaptive-type mechanism carried out at finite intervals.

The proposed revision strategy is structured as follows. At $t$ the economic agent selects their optimal portfolio expressed in terms of investment choices, $\mathbf{z}^*_t$. At $t+\Delta t$ one of the following three situations occurs:

1. $\mathbf{z}^*_t$ is still the optimal portfolio for the economic agent: no revision is necessary and $\mathbf{z}^*_{t+\Delta t} = \mathbf{z}^*_t$;
2. $\mathbf{z}^*_t$ is no longer optimal, but is again efficient: revision could be carried out;
3. $\mathbf{z}^*_t$ is no longer either optimal or efficient: revision could be carried out.

**Case 2.** In this case $\mathbf{z}^*_t$ lies on the new efficient frontier, so that a transition toward any other efficient portfolio would involve a trade-off between return and risk. Because of the difficulty of specifying indifference curves, it is postulated that if $\mathbf{z}^*_t$ is efficient the investor is satisfied to keep it [Smith1967]. Therefore no revision is carried out and $\mathbf{z}^*_{t+\Delta t} = \mathbf{z}^*_t$.

**Case 3.** When $\mathbf{z}^*_t$ is no longer either optimal or efficient, its revision can move toward one of three points on the new efficient frontier, identifiable in the $\mathrm{Var}(X)$–$E(X)$ plane starting from the point $\mathbf{z}^*_t$, now located below and to the right of the curve:

- **Point $A$** — on the frontier, at the same expected-return level $r_t$ as $\mathbf{z}^*_t$. The resulting portfolio $\mathbf{z}^*_{t+\Delta t}$ is efficient, has the same expected return, and a lower return variance than $\mathbf{z}^*_t$, but is not necessarily optimal. Since transitions are evaluated in terms of expected monetary return and costs, it is difficult to measure a monetary return comparable to the mere risk reduction achieved by moving toward $A$ [Smith1967]; in this sub-case no revision is carried out and $\mathbf{z}^*_{t+\Delta t} = \mathbf{z}^*_t$.
- **Point $B$** — the point of tangency between the efficient frontier and the preference function. The resulting portfolio $\mathbf{z}^*_{t+\Delta t}$ is optimal for the economic agent. If the investor could specify their own preference function, the tangency solution at $B$ would be a logical objective of the transition; but since specifying such a preference function is avoided, the portfolio at $B$ is not considered [Smith1967]. In this sub-case too no revision is carried out and $\mathbf{z}^*_{t+\Delta t} = \mathbf{z}^*_t$.
- **Point $C$** — on the frontier, at the same return variance as $\mathbf{z}^*_t$, at level $r_{t+\Delta t}$. The resulting portfolio $\mathbf{z}^*_{t+\Delta t}$ is efficient, has the same return variance and a higher expected return than $\mathbf{z}^*_t$, but is not necessarily optimal. In this situation the investor accepts the risk inherent in the existing portfolio (according to revised expectations) and seeks the maximum improvement in expected return; operationally this is feasible since return is measured in monetary terms, and the portfolio at $C$ is the target, or desired, portfolio in light of revised expectations [Smith1967]. In this sub-case the revision is actionable.

**Desirability condition.** Before carrying out the revision toward $C$, one checks whether it is worthwhile, that is, whether the increase in the expected return of the revised portfolio is greater than the associated revision costs:

$$\sum_{i=1}^{N} \left(z^*_{t+\Delta t,i} - z^*_{t,i}\right) P_{t+\Delta t,i}\, r_{t+\Delta t,i} > prc_{t+\Delta t},$$

where $prc_{t+\Delta t}$ denotes the portfolio revision costs to be borne at $t+\Delta t$.

## The Stone and Hill (1979) Model

The second approach is that of Stone and Hill [StoneHill1979], whose revision procedure is a generalization of the classical linear knapsack algorithm, retaining most of its characteristics: high computational efficiency and the ability to handle large-scale problems. The work proceeds in three stages: first, a linear portfolio selection problem (approximating the classical quadratic-linear problems) is formulated for use at $t$; then, a linear portfolio revision problem is formulated for use at $t+\Delta t$, $t+2\Delta t$, \ldots, $t+i\Delta t$, \ldots; finally, a solution procedure for the revised portfolio problem is provided and its computational efficiency discussed.

**The linear selection problem.** Introduce:

- $\mathbf{c}' = (r_1 - \beta_1\theta, \ldots, r_N - \beta_N\theta)$: the $N$-vector of risk-adjusted rates of return, where $\beta_i$ is the systematic risk of the $i$-th investment choice and $\theta$ is a constant that appropriately weights systematic risk;
- $f_i \in [0,1]$: an upper bound associated with $x_i$.

The linear portfolio selection problem is then

$$\max_{x_1,\ldots,x_N} \ \mathbf{x}'\mathbf{c} \qquad \text{s.t.} \quad \begin{cases} \mathbf{x}'\mathbf{e} = 1 \\ 0 \le x_i \le f_i, \quad i=1,\ldots,N. \end{cases}$$

**The linear revision problem.** Also introduce:

- $\mathbf{b}' = \mathbf{c}' - (bc_1,\ldots,bc_N)$: the $N$-vector of net risk-adjusted rates of return "earned" by buying the investment choices, where $bc_i$ is the cost of buying 1 monetary unit of the $i$-th investment choice;
- $\mathbf{s}' = \mathbf{c}' + (sc_1,\ldots,sc_N)$: the $N$-vector of net risk-adjusted rates of return "lost" by selling the investment choices, where $sc_i$ is the cost of selling 1 monetary unit of the $i$-th investment choice;
- $\mathbf{q}' = (q_1,\ldots,q_N)$: the $N$-vector of monetary amounts invested in the investment choices before revision;
- $\mathbf{z_b}' = (z_{b,1},\ldots,z_{b,N})$: the unknown $N$-vector of monetary amounts to be invested in each investment choice;
- $\mathbf{z_s}' = (z_{s,1},\ldots,z_{s,N})$: the unknown $N$-vector of monetary amounts to be divested from each investment choice.

The linear portfolio revision problem is then

$$\max_{z_{b,1},\ldots,z_{b,N},\,z_{s,1},\ldots,z_{s,N}} \ \mathbf{b}'\mathbf{z_b} - \mathbf{s}'\mathbf{z_s}$$

$$\text{s.t.} \quad \begin{cases} \mathbf{z_b}'[\mathbf{e} + (bc_1,\ldots,bc_N)] - \mathbf{z_s}'[\mathbf{e} - (bs_1,\ldots,bs_N)] = 0 \\ z_{b,i} \ge 0, \quad i=1,\ldots,N \\ z_{b,i} \le f_i\,\mathbf{q}'\mathbf{e} - q_i, \quad i=1,\ldots,N \\ z_{s,i} \ge 0, \quad i=1,\ldots,N \\ z_{s,i} \le q_i, \quad i=1,\ldots,N. \end{cases}$$

The first constraint ensures that the monetary amount to be invested (plus the associated transaction costs) equals the monetary amount to be divested (minus the associated transaction costs). The constraint $z_{b,i} \le f_i\,\mathbf{q}'\mathbf{e} - q_i$ ensures that the percentage of the pre-revision portfolio value, equal to $\mathbf{q}'\mathbf{e}$, to be invested in the $i$-th investment choice is less than or equal to $f_i$.

**The exchange rule.** The value of the objective function (to be maximized) increases by buying the generic $i$-th investment choice and selling the generic $j$-th investment choice such that

$$c_i - bc_i > c_j + sc_j, \qquad i,j \in \{1,\ldots,N\} \wedge i \ne j,$$

that is, this value increases if

$$c_i - c_j > bc_i + sc_j, \qquad i,j \in \{1,\ldots,N\} \wedge i \ne j.$$

For the entire exchange program, therefore, the objective function amounts to maximizing the increase in the risk-adjusted rate of return of purchases over sales, net of total transaction costs. This structure directly suggests a solution approach: rank the purchase candidates based on $c_i - bc_i$ and rank the current portfolio positions based on $c_j + sc_j$; buying the purchase candidate with the highest rank and selling the portfolio position with the lowest rank produces the maximum improvement of the objective function, with the size of the exchange determined so that no constraint is violated [StoneHill1979].

## Characteristics of a Good Revision Model

The two strategies described finally suggest what characteristics a good portfolio revision model must have. It must possess:

- **realism**;
- **low (computational) complexity**;
- **effectiveness**.

## References

- **[Markowitz1952]** Markowitz, H. (1952), «Portfolio Selection», The Journal of Finance, Vol. 7, No. 1, pp. 77–91.
- **[Smith1967]** Smith, K.V. (1967), «A Transition Model for Portfolio Revision», The Journal of Finance, Vol. 22, No. 3, pp. 425–439.
- **[StoneHill1979]** Stone, B.K. and Hill, N.C. (1979), «Portfolio Management and the Shrinking Knapsack Algorithm», Journal of Financial and Quantitative Analysis, Vol. 14, No. 5, pp. 1071–1083.
