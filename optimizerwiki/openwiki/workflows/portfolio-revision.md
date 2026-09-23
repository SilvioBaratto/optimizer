---
type: concept
title: "Portfolio Revision"
description: Why an optimal portfolio selected at one instant generally ceases to be optimal later, forcing the move from static to dynamic management via portfolio revision — the three categories of revision cost, the capital-gains-tax convenience condition, and two classic models for locating the potentially revised portfolio (Smith 1967's transition model and Stone-Hill 1979's shrinking-knapsack linear program), closing with the properties of a good revision model.
tags: [portfolio-revision, rebalancing, dynamic-management, transaction-costs, capital-gains-tax, no-trade-region, smith-1967, stone-hill-1979, knapsack, mean-variance, turnover]
sources:
  - id: openwiki-source-0726ab43121cb6823461e30c
    resource: repo://docs/07_portfolio_revision.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---

# Portfolio Revision

An optimal portfolio selected at a given instant generally ceases to be optimal at later instants, and
this forces the move from static to dynamic management through *portfolio revision*. This chapter
formalizes when revision is worthwhile in the presence of a capital-gains tax and presents two classic
models for locating the potentially revised portfolio — Smith (1967) and Stone-Hill (1979) — closing with
the properties a good revision model must possess. Revision is the step that generates the
[turnover a decaying signal implies](../signals/signal-decay-horizon-turnover.md); once the revised
target weights are set (see [from conditional forecasts to weights](./from-conditional-forecasts-to-weights.md)),
they are traded through [optimal execution](./optimal-execution-and-market-impact.md).

## Between static and dynamic management

The mean-variance model shows how to select a portfolio at a single instant. In an ideal single-period
economy running from $t$ to $t+\Delta t$ this is sufficient: the agent selects the optimal portfolio at
$t$ and "releases" it — liquidates it — at $t+\Delta t$. In reality the agent operates in an (at least)
multi-period economy, articulated over intervals from $t+i\Delta t$ to $t+(i+1)\Delta t$, and so:
selects the optimal portfolio at $t$; can no longer "release" it at $t+\Delta t$; and must instead
*manage* it at $t+\Delta t$, $t+2\Delta t$, and so on.

In an *ideal* multi-period economy the means, variances and linear correlation coefficients of the
investment choices stay the same at every instant, so the portfolio optimal at $t$ remains optimal at all
later instants and dynamic management coincides with static management. In reality, however, these
moments change with the instant considered, so the portfolio optimal at $t$ may be neither optimal nor
efficient at $t+\Delta t$ or later. This is the limit that separates static from dynamic management and
makes portfolio revision necessary.

## The temporal instability of the optimal portfolio

The optimal portfolio's sensitivity to the statistical parameters — and hence to the estimation window —
is seen in a concrete example over three Italian equities (Alleanza Assicurazioni $X_1$, Fondiaria-Sai
$X_2$, Unicredito Italiano $X_3$). Estimating means, variances and correlations over two daily windows
offset by a single day — window A from 22 August to 20 September 2006, window B from 23 August to 21
September 2006 — and imposing a target expected return $\pi=0.00041$ in the basic mean-variance problem,
the optimal weights shift dramatically: $x_1^\star$ from $0.69193$ to $0.96600$, $x_2^\star$ from
$0.52029$ to $0.09500$, and $x_3^\star$ from $-0.21223$ to $-0.06100$.

If the agent wants the portfolio to earn expected return $0.00041$ both from $t$ (20 September) to
$t+\Delta t$ (21 September) and onward, the passage from column A to column B requires, at $t+\Delta t$:
**buying** $X_1$ for $0.27407$ of capital, **selling** $X_2$ for $0.42529$, and **buying** $X_3$ for
$0.15123$. Shifting the estimation window by a single day thus changes the optimal portfolio
significantly, forcing a revision of its composition.

## Portfolio revision and its convenience

The set of adjustments made at $t+\Delta t$ is called *portfolio revision* (in a frictionless market). In
reality these adjustments have a cost, and the revision costs may sometimes exceed the increase in the
revised portfolio's expected return — so revision is not always worthwhile. The operational question —
how to carry out a worthwhile revision — is answered in three steps: identify the main categories of
revision cost; locate the potentially revised portfolio; verify whether revision is worthwhile.

## The main categories of revision cost

Three categories of cost may arise at $t+\Delta t$, $t+2\Delta t$, and so on.

1. **Data collection and processing costs** — collecting new data, updating the data set, transforming it
   (e.g. from prices to percentage or log returns) and computing the quantities of interest (means,
   variances, correlations). This cost is **fixed**: it must be borne at every instant, whether or not the
   portfolio is revised.
2. **Transaction costs** — the costs of trading the investment choices, i.e. what is paid to an
   intermediary (bank, broker) to buy and/or (short-)sell. This cost is **variable**: it is borne only in
   the instants when the portfolio is actually revised.
3. **Capital-gains taxes** — the tax on any realized increase in the portfolio's value. Technically these
   are not costs, but in this context they can be treated as having the status of a cost, clarified by the
   formalization below.

## Convenience condition under a capital-gains tax

For a simple two-asset portfolio, let $\mathbf z^\star_\tau=(z^\star_{\tau,1},z^\star_{\tau,2})$ be the
portfolio (in investment choices) selected or revised at $\tau$, $P_{\tau,i}$ the price, $r_{\tau,i}$ the
expected return, $\Delta_{\tau,i}$ the number of units bought or (short-)sold at $\tau$, and
$t_{cg}\in[0,1)$ the fraction of realized capital gain paid as tax. At $t$ the agent selects
$\mathbf z^\star_t$; at $t+\Delta t$ it can either **not revise** (so $\mathbf z^\star_{t+\Delta
t}=\mathbf z^\star_t$, with numéraire expected return $r_{NR,t+\Delta t}$) or **revise** by disinvesting
$\Delta_{t+\Delta t,1}P_{t+\Delta t,1}$ from asset 1, paying the tax $t_{cg}\max\{\Delta_{t+\Delta
t,1}(P_{t+\Delta t,1}-P_{t,1}),0\}$ on any realized gain, and investing the remainder in asset 2 (giving
$r_{R,t+\Delta t}$).

Revision is worthwhile when $r_{R,t+\Delta t}>r_{NR,t+\Delta t}$. Subtracting the two expressions, the
shared terms cancel and the inequality reduces to
$$\Delta_{t+\Delta t,1}P_{t+\Delta t,1}(r_{t+\Delta t,2}-r_{t+\Delta t,1})>t_{cg}\max\{\Delta_{t+\Delta t,1}(P_{t+\Delta t,1}-P_{t,1}),0\}\,r_{t+\Delta t,2}\ge0,$$
where the right side is non-negative because $t_{cg}\ge0$, the $\max\{\cdot,0\}$ operator is non-negative
and $r_{t+\Delta t,2}\ge0$. Dividing by $\Delta_{t+\Delta t,1}P_{t+\Delta t,1}$ gives the final form
$$r_{t+\Delta t,2}>r_{t+\Delta t,1}+\frac{t_{cg}\max\{\Delta_{t+\Delta t,1}(P_{t+\Delta t,1}-P_{t,1}),0\}}{\Delta_{t+\Delta t,1}P_{t+\Delta t,1}}.$$
Revision is thus worthwhile only if the expected return of the asset receiving the capital exceeds that of
the liquidated asset by at least the per-unit incidence of the tax on the realized capital gain.

This "cost" arises only when a realized capital gain occurs; when it does, it is a **variable** cost borne
not in every revision instant but only when a gain is realized on the liquidated position — the sense in
which the capital-gains tax, though not technically a cost, has the status of a variable cost here.

## Smith's model (1967)

Smith's model extends portfolio selection to an intertemporal basis via an adaptive mechanism run at
finite intervals. At $t$ the agent selects the optimal portfolio $\mathbf z^\star_t$; at $t+\Delta t$ one
of three situations occurs: (1) $\mathbf z^\star_t$ is still optimal — no revision, $\mathbf
z^\star_{t+\Delta t}=\mathbf z^\star_t$; (2) $\mathbf z^\star_t$ is no longer optimal but is again
efficient — revision could be made; (3) $\mathbf z^\star_t$ is neither optimal nor efficient — revision
could be made.

**Case 2.** Here $\mathbf z^\star_t$ lies on the new efficient frontier, so any move to another efficient
portfolio trades return against risk. Because the indifference curves are hard to specify, one postulates
that if $\mathbf z^\star_t$ is efficient the investor is content to keep it; no revision is made.

**Case 3.** When $\mathbf z^\star_t$ is neither optimal nor efficient — now below and to the right of the
new frontier — its revision could move to one of three frontier points:
- **Point $A$** — on the frontier at the same expected return $r_t$ as $\mathbf z^\star_t$: efficient,
  same expected return, lower variance, but not necessarily optimal. Since transitions are judged in
  expected numéraire and costs, it is hard to measure a comparable numéraire return against a mere risk
  reduction, so no revision is made.
- **Point $B$** — the tangency between the efficient frontier and the preference function: optimal for the
  agent. But since the preference function is deliberately not specified, point $B$ is not considered and
  no revision is made.
- **Point $C$** — on the frontier at the same return variance as $\mathbf z^\star_t$ but at higher
  expected return $r_{t+\Delta t}$: efficient, same variance, higher expected return, not necessarily
  optimal. The investor accepts the existing portfolio's risk (under revised expectations) and seeks the
  maximum improvement in expected return; because return is measured in numéraire this is operational, and
  $C$ is the target (desired) portfolio under the revised expectations. Here revision is feasible.

**Convenience condition.** Before revising toward $C$, one checks that the increase in expected return
exceeds the revision costs:
$$\sum_{i=1}^N (z^\star_{t+\Delta t,i}-z^\star_{t,i})P_{t+\Delta t,i}\,r_{t+\Delta t,i}>prc_{t+\Delta t},$$
where $prc_{t+\Delta t}$ denotes the revision costs to be borne at $t+\Delta t$.

## The Stone-Hill model (1979)

Stone-Hill's revision procedure generalizes the classic linear knapsack algorithm, retaining most of its
features — high computational efficiency and the ability to handle large problems. It proceeds in three
stages: a linear portfolio-selection problem (approximating the classic quadratic-linear problems) for use
at $t$; a linear revision problem for use at later instants; and a solution procedure with its efficiency
analysis.

**Linear selection problem.** With $\mathbf c'=(r_1-\beta_1\theta,\dots,r_N-\beta_N\theta)$ the vector of
risk-adjusted returns ($\beta_i$ the systematic risk, $\theta$ a constant weighting it) and $f_i\in[0,1]$
an upper bound on $x_i$, the linear selection problem is
$$\max_{x_1,\dots,x_N}\mathbf x'\mathbf c\quad\text{s.t.}\quad \mathbf x'\mathbf e=1,\;\;0\le x_i\le f_i.$$

**Linear revision problem.** Introduce $\mathbf b'=\mathbf c'-(bc_1,\dots,bc_N)$ (risk-adjusted returns
net of the buying cost $bc_i$ per unit of numéraire), $\mathbf s'=\mathbf c'+(sc_1,\dots,sc_N)$ (net of
the selling cost $sc_i$), $\mathbf q'$ the numéraire amounts invested before revision, and unknown vectors
$\mathbf z_b'$ (amounts to invest) and $\mathbf z_s'$ (amounts to disinvest). The problem is
$$\max_{\mathbf z_b,\mathbf z_s}\;\mathbf b'\mathbf z_b-\mathbf s'\mathbf z_s$$
subject to a budget-balance constraint (amount invested plus transaction costs equals amount disinvested
minus its transaction costs), $z_{b,i}\ge0$, $z_{b,i}\le f_i\mathbf q'\mathbf e-q_i$ (holding the
$i$-th weight of the pre-revision value $\mathbf q'\mathbf e$ at or below $f_i$), $z_{s,i}\ge0$ and
$z_{s,i}\le q_i$.

**Exchange rule.** The objective increases by buying the $i$-th choice and selling the $j$-th choice such
that $c_i-bc_i>c_j+sc_j$, i.e. $c_i-c_j>bc_i+sc_j$ — so the whole exchange program maximizes the increase
of the buys' risk-adjusted return over the sells', net of total transaction costs. This structure yields a
direct solution: rank buy candidates by $c_i-bc_i$ and current holdings by $c_j+sc_j$; buying the
highest-ranked candidate and selling the lowest-ranked holding gives the maximum objective improvement,
with the exchange amount set so that no constraint is violated.

## Properties of a good revision model

The two strategies suggest what a good portfolio-revision model must possess: **realism**, **reduced
(computational) complexity**, and **effectiveness**.
