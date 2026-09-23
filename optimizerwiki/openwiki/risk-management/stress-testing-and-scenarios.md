---
type: "Reference"
title: "Stress Testing and Scenarios"
openwiki_generated: true
sources:
  - id: openwiki-source-c0967703b8285eabe93c7f1e
    resource: repo://docs/24_stress_testing_and_scenarios.md
generated: { by: "claude-code", at: "2026-09-22T11:04:21.953Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T08:34:43.263Z
---


# Stress Testing and Scenarios

This chapter positions stress testing as the ex-ante, forward-looking complement to the
history-based risk measures of the preceding chapters — the Value-at-Risk and coherent measures
of [coherent risk measures](../risk-measures/coherent-risk-measures.md), the Conditional
Value-at-Risk of [CVaR optimization](../optimization/cvar-optimization.md), the
[drawdown risk measures](../risk-measures/drawdown-risk-measures.md) — all of which describe a
portfolio's behavior under ordinary conditions by estimating risk from historically observed
statistical relations. Stress testing addresses the complementary question: what happens at the
extremes, when large shocks materialize. It shares the Mahalanobis machinery of
[turbulence and systemic risk](../regimes/turbulence-and-systemic-risk.md) and, through the
Maximum Loss Contribution, connects to the Euler decomposition of
[risk attribution, budget and limits](./risk-attribution-budget-limits.md).

## Beyond risk under normal conditions

A stress test is commonly described as the evaluation of a bank's financial position under a
severe but plausible scenario to support decisions; the term denotes not only the mechanics of
running individual tests but the whole context in which they are developed, evaluated and used.
Stress testing is a risk-management tool that integrates other approaches and measures, and is
especially important for forward-looking risk assessments, for overcoming the limits of models
and historical data, for capital and liquidity planning, for informing the bank's risk
tolerance, and for developing mitigation or contingency plans. Its importance is greatest after
long benign periods, when a faded memory of downturns breeds complacency, and during expansions,
when innovation spawns fast-growing new products with scarce loss history. The Basel II framework
anchors it in regulation: the Pillar 1 Internal Models Approach for market risk requires a
rigorous stress-testing program, and internal-ratings-based credit-risk approaches require
credit-risk stress tests to assess internal capital adequacy.

## The weaknesses revealed by the crisis

The depth and duration of the financial crisis led many banks and supervisors to question
whether pre-crisis stress-testing practices were sufficient: the crisis proved more severe than
banks' stress-test results indicated, and was possibly aggravated by weaknesses in the practices
themselves. The Basel Committee identifies four areas of weakness. On **use and governance**,
stress testing was often an isolated exercise of the risk function with little business-line
interaction — deemed not credible or merely mechanical — and many banks lacked an overarching
program, running separate tests per risk or portfolio with limited firm-wide integration,
reducing the ability to spot correlated tail exposures and concentrations. On **methodology**,
most risk models used historical statistical relations assuming a known, constant process; after
a long stable period the backward-looking information indicated benign conditions, missing both
severe shocks and the buildup of vulnerabilities, and historical relations such as correlations
proved unreliable once events unfolded. On **scenario selection**, most tests were not designed
to capture the extreme events actually experienced — mild shocks, shorter durations, understated
correlations — and pre-crisis "severe" scenarios typically produced losses no larger than a
quarter of earnings, with senior management often judging more extreme scenarios implausible. On
**specific risks and products**, complex structured products under stressed liquidity, pipeline
and securitization risk, basis risk, counterparty credit risk, contingent risks and funding
liquidity risk were insufficiently covered — structured-product tests failing to recognize that
their risk dynamics differ from similarly rated cash instruments like bonds.

## Principles of a stress-testing program

The Committee's principles, applied proportionately to size and complexity, cover governance,
use, methodology and scenario design. **Governance and integration**: stress testing must be an
integral part of the governance and risk-management culture and be actionable, influencing
decision-making at the appropriate level including board and senior-management strategic
decisions, with the board ultimately responsible and senior management responsible for
implementation, documented by written policies on a robust, flexible infrastructure.
**Complementary risk perspective**: a program must provide a perspective independent of and
complementary to tools such as VaR and economic capital, challenging projected risk
characteristics of new products and simulating scenarios where embedded statistical relations
break down; its outcomes can in particular indicate the validity of statistical models at the
high confidence intervals used to determine VaR, and it must be part of the ICAAP feeding capital
and liquidity planning. **Methodology and scenario selection**: tests must cover a range of risks
and business areas, integrated meaningfully firm-wide, accounting for interrelations among risk
factors, and impact is assessed against measures such as asset values, accounting or economic
P&L, regulatory capital or RWA, economic-capital requirements, and liquidity/funding gaps. Since
ex-ante estimation of stress-event probabilities is problematic — the statistical relations used
to derive them break down under stress — appropriate weight must be given to expert judgment, and
tests must present a range of severities including events causing the greatest damage by loss
size or reputation.

## The problem of hand-picked scenarios

The quality of a stress test depends crucially on scenario definition — a thought experiment, a
counterfactual in which the risk manager imagines adverse or catastrophic events that could hit
the portfolio, exposed to two pitfalls: considering implausible scenarios and neglecting
plausible ones. A bias toward historical experience can ignore plausible but damaging scenarios
never yet realized, a dangerous blind spot; at the opposite extreme, excessive weight on highly
implausible scenarios confronts management with the awkward question of whether to react to
alarming results from highly implausible scenarios. Breuer, Jandačka, Rheinberger and Summer give
operational precision to the three Basel requirements — **plausibility**, **severity** and
**suggestiveness of risk-reduction actions** (usefulness): define an appropriate plausibility
region in terms of the risk-factor distribution and systematically search for the maximum-loss
scenario over it. In their setting each position's value at a future horizon depends on $n$
systematic risk factors $\boldsymbol r=(r_1,\dots,r_n)$ and $m$ idiosyncratic factors; the
systematic distribution is restricted to the elliptical class (with covariance $\mathrm{Cov}$ and
mean $\boldsymbol\mu$), to which the standard distributions of classical risk management belong,
while the idiosyncratic distribution may be arbitrary.

## Plausibility: the Mahalanobis distance

How plausible is an imagined extreme realization? An intuitive approach compares it to the mean,
measuring distance in standard deviations; for multivariate moves plausibility must also depend
on correlations, since a move congruent with the correlations is more plausible than one against
them. The statistical concept formalizing this is the **Mahalanobis distance**
$$ \mathrm{Maha}(\boldsymbol r) := \sqrt{(\boldsymbol r-\boldsymbol\mu)^{T}\,\mathrm{Cov}^{-1}\,(\boldsymbol r-\boldsymbol\mu)}, $$
the distance of the test point from the center of mass divided by the ellipsoid's width in that
direction, interpreted as the number of standard deviations of the multivariate move and
accounting for both the correlation structure and the factors' standard deviations. Plausibility
is defined directly in terms of $\mathrm{Maha}(\boldsymbol r)$ — a high Maha means low
plausibility. Earlier work defined plausibility via the *probability mass* of the ellipsoid of
all scenarios of equal or smaller Maha, which creates the **dimensional dependence of maximum
loss**: with a fixed-mass ellipsoid the maximum loss would depend on the arbitrary number of risk
factors. In one example, a bond portfolio with two yield curves in ten currencies modeled with
seven maturity buckets uses 150 factors, another with fifteen buckets 310 factors; at a 95%-mass
ellipsoid the second computes a maximum loss 1.4 times the first for the same portfolio and
plausibility — a problem that does not arise when plausibility is defined via the Mahalanobis
radius rather than probability mass.

## Partial scenarios: treating the unstressed factors

Portfolios are typically modeled with hundreds or thousands of factors; full-factor scenarios are
numerically intractable and hard to interpret, so *partial scenarios* involving only a few
factors (say a given FX move or GDP drop) are used. The unfixed factors can be treated four ways:
(i) held at their last observed value; (ii) at their unconditional expected value; (iii) at their
expected value *conditional* on the fixed factors, giving $\boldsymbol r_C$; (iv) left distributed
according to the conditional distribution, giving $\boldsymbol r_D$. Assuming an elliptical
distribution with density strictly decreasing in Maha,
$$ \mathrm{Maha}(\boldsymbol r_C) = \mathrm{Maha}(\boldsymbol r_D), $$
and this is the maximum plausibility attainable among all macro scenarios agreeing on the fixed
factors. Practically, method (iv) maximizes plausibility, but equivalent plausibility is achieved
with the computationally cheaper method (iii) — setting unfixed factors to their conditional
expected value; any other assignment yields less plausible macro scenarios.

## Severe scenarios: the systematic maximum-loss search

The key drawback of hand-picked scenarios is the danger of ignoring damaging but plausible
scenarios, creating an illusion of safety. Searching systematically over a plausible admissible
domain for the most damaging macro scenarios guarantees no damaging-but-plausible scenario is
missed. The admissible domain contains all scenarios with Maha below a threshold $k$,
$$ \mathrm{Ell}_k := \{\boldsymbol r : \mathrm{Maha}(\boldsymbol r) \le k\}, $$
an ellipsoid shaped by the factors' covariance matrix. Since a partial scenario specifies a
*distribution* of conditional values rather than a single value, severity is measured by the
Conditional Expected Profit (CEP): a partial scenario is severe if it has low CEP, and the method
reduces to
$$ \min_{\boldsymbol r\in\mathrm{Ell}_k}\mathrm{CEP}(\boldsymbol r). $$
The difference between the lowest CEP in the domain and the CEP in the expected scenario is the
**Maximum Loss** over the admissible domain — a concept that overcomes dimensional dependence,
since the maximum expected loss on $\mathrm{Ell}_k$ is unaffected by including or excluding factors
irrelevant to portfolio value. Worst-case search has three advantages over standard stress
testing: a controlled plausibility/severity trade-off governed by $k$ (higher $k$ gives worse but
less plausible scenarios); it overcomes historical bias by considering all plausible scenarios,
including ones not yet realized; and worst-case scenarios reflect portfolio-specific dangers —
what is worst for one portfolio may be harmless for another.

## Useful scenarios: key risk factors and risk-reduction action

Risk-reduction action is suggested by identifying the **key risk factors** — those contributing
most to the expected loss in the worst-case scenario. The Loss Contribution of factor $i$ in
scenario $\boldsymbol r$ is
$$ LC(i,\boldsymbol r) := \frac{CEP(\boldsymbol\mu) - CEP(\mu_1,\dots,\mu_{i-1}, r_i, \mu_{i+1},\dots,\mu_n)}{CEP(\boldsymbol\mu) - CEP(\boldsymbol r)}, $$
the loss if factor $i$ alone took its scenario value while the others stayed at $\boldsymbol\mu$,
as a percentage of the scenario loss; evaluated at the worst-case scenario $\boldsymbol r^{WC}$ it
is the **Maximum Loss Contribution** $MLC(i):=LC(i,\boldsymbol r^{WC})$. The contributions do not
in general sum to 100%. Assuming continuous second derivatives, $\sum_i LC(i,\boldsymbol r)=1$ for
all scenarios *if and only if* the CEP is additive $CEP(\boldsymbol r)=\sum_i g_i(r_i)$, i.e. all
cross-derivatives $\partial^2 CEP/\partial r_i\partial r_j$ ($i\ne j$) vanish. The sum measures
the role of factor interaction: greater than one means positive interaction (total loss below the
sum of individual losses), and the most dangerous case is a sum *below one* — negative
interaction, where simultaneous factor moves cause damage beyond the individual moves. A
consequence outside stress testing is that analyzing market and credit risk separately and
aggregating the separately computed capital can understate true risk by ignoring simultaneous
market-credit moves. Knowing the key factors enables hedges that pay off exactly when the key
factors take their worst-case value, or the fuller (costlier) hedge neutralizing damage from all
key-factor moves.

## An illustration: foreign-currency loans

The concepts are illustrated on floating-rate loans in domestic (EUR) or foreign (CHF) currency,
each position's value depending on GDP, domestic rate $r_h$, foreign rate $r_f$ and FX change
$E$, with the distribution from a time-series model and a Monte Carlo simulation of 100,000
four-step trajectories used to estimate the annual covariance matrix. The hand-picked scenario
"GDP contracts 3%" is a $5.42\sigma$ event; against the *same-plausibility* worst-case scenario,
conditional expected profit is considerably lower in the worst case for every portfolio — the
hand-picked GDP scenario reduces profits only moderately (a false sense of safety), while
equally plausible far more damaging scenarios exist, with foreign-currency worst-case losses of
about 11% exceeding the 8% total regulatory capital. Key factors are portfolio-dependent: for
foreign-currency loans FX alone contributes between 65.3% and 100% of worst-case losses (others
under 1%); for the domestic portfolio GDP contributes between 46.8% and 70.0%, with negative
GDP–interest-rate interaction explaining about a third of the worst-case loss. For a domestic B+
loan restricted to Maha below $k=6$ the two MLCs sum to $62.0+9.9=71.9\%$, well below 100% —
signaling the joint-move loss considerably exceeds the sum of individual-move losses.

## Reverse stress testing

The range of severities must include events causing the greatest damage, and a program must also
determine which scenarios could challenge the bank's viability — the **reverse stress tests** —
uncovering hidden risks and interactions. A reverse stress test starts from a known stressed
outcome (breaching regulatory capital ratios, illiquidity, insolvency) and asks which events
could lead there, including some extreme scenarios that would render the bank insolvent, requiring
senior-management involvement across all material risk areas and inducing consideration of
scenarios with contagion and systemic implications. Pre-crisis such analysis was deemed of little
value given the remote probability; banks now express the need to examine tail events, especially
for business lines whose traditional models indicate an exceptionally good risk/reward trade-off,
new products and markets untested by stress, and exposures lacking liquid two-sided markets. The
systematic worst-case search is the quantitative counterpart: the direct test fixes $k$ and finds
maximum loss within $\mathrm{Ell}_k$, while the inverse exercise moves from a loss and plausibility
level and finds the scenario producing it — exhaustive search over the admissible domain ensuring
no damaging-but-plausible scenario is lost while avoiding scenarios too implausible to be
credible.

## Stress testing in the risk dashboard

Stress testing does not replace normal-conditions risk measures but completes them with an
independent, complementary perspective. The Basel Committee explicitly positions it as a
complement to backward-looking quantitative-model tools — in particular VaR and economic capital —
noting its outcomes can indicate model validity at high confidence intervals. The result is a
dashboard where the measures of
[coherent risk measures](../risk-measures/coherent-risk-measures.md) and
[CVaR optimization](../optimization/cvar-optimization.md) describe the loss distribution within
sample experience — VaR quantifying a quantile, Conditional Value-at-Risk the expected loss beyond
it — and [drawdown risk measures](../risk-measures/drawdown-risk-measures.md) describe cumulative
peak-to-trough losses, while the stress test assesses robustness to severe-but-plausible scenarios
that historical statistical relations fail to capture, precisely because those relations break
down under stress. In the systematic formulation, severity is the maximum expected loss within the
plausibility ellipsoid $\mathrm{Ell}_k$, and the severity/plausibility trade-off is governed by the
Mahalanobis radius $k$ — the role $k$ plays for the stress test being the role the confidence level
plays for VaR and CVaR, anchoring the scenario's extremity to a defined, non-arbitrary plausibility
region. This compatibility with the standard quantitative risk-management framework is what makes
the method directly graftable onto the existing dashboard.
