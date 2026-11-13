# DECISIONS — Phase 0, issue [0.2]

The five technical decisions that block code. Each is recorded as a **recommendation awaiting
the author's sign-off**, not as a final ruling binding on Phases 2 and 6 (clarification 1).
Every entry carries a one-line rationale, so it can be re-evaluated when its premise moves.

**Phase 0 introduces no dependency into any `requirements.txt` or `pyproject.toml`.** The
declarations below describe what the consuming phase must add, not what this phase installed.

---

## Summary

| # | Decision | New dependency | Status |
|---|---|---|---|
| (a) | `hmmlearn` for the EM fit + our own Hamilton forward recursion for filtered probabilities | `hmmlearn` — **opt-in extra `[regimes]`** | recommended |
| (b) | `fredapi` inside `ingestion/`; the daemon owns the fetch, `optimizer/` receives the panel injected | `fredapi` in `ingestion/requirements.txt` | recommended |
| (c) | Hybrid — estimate λ and η from OHLCV proxies, take ρ and cross-impact as inputs | none | recommended |
| (d) | `DGS3MO` from FRED, converted in `fund/`, passed as a per-fold scalar | none | recommended |
| (e) | Ollama cloud models; audit by recording, not by reproducibility | none | recommended |

---

## (a) Markov-switching implementation

**Recommendation:** `hmmlearn` 0.3.3 for the multivariate EM/Baum-Welch fit, plus a Hamilton
forward recursion written in-package for the filtered probabilities. Declared as an **opt-in
extra** `[regimes]`, following the §4.5 precedent that keeps the heavy stack opt-in.

**Rationale:** it is the only candidate that can deliver the per-state correlation signs task
2.1b requires, and it adds no transitive dependency portopt does not already carry.

### Why not statsmodels

Task 2.1b's acceptance criteria require states carrying distinct means, volatilities **and
correlation signs** (§6.4). `statsmodels.tsa.regime_switching.MarkovRegression` is univariate —
it models a single endogenous series — so it cannot produce a per-state covariance matrix at
all. It is the canonical econometric implementation and exposes `filtered_marginal_probabilities`
and `smoothed_marginal_probabilities` natively, but it cannot meet the stated AC.

### Dependency cost, measured

| Package | Licence | Size | Transitive deps not already in portopt |
|---|---|---|---|
| **`hmmlearn` 0.3.3** | BSD-3-Clause (`License :: OSI Approved :: BSD License`) | **1.0 MB** | **none** — requires `numpy`, `scipy`, `scikit-learn`, all already runtime deps |
| `statsmodels` 0.14.6 | BSD-3-Clause | 35.6 MB | `patsy` (1.2 MB), `packaging` — **2 new** |

portopt today declares six runtime dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`,
`skfolio`, `jinja2`. `hmmlearn` costs 1.0 MB and zero new transitive packages; `statsmodels`
costs ~37 MB and two.

### What hmmlearn covers, and what it does not

Covered natively by `GaussianHMM(covariance_type="full")`:

| AC | API |
|---|---|
| estimated transition matrix | `transmat_` |
| stationary distribution | `get_stationary_distribution()` |
| log-likelihood | `score()` |
| per-state means, volatilities, correlation signs | `means_`, `covars_` with `covariance_type="full"` |
| determinism knob | `random_state` |
| sklearn idiom | `get_params` / `set_params`, metadata routing |

**Not covered — must be written:** `predict_proba` returns the *smoothed* posterior, computed
over the full sequence. hmmlearn exposes no public forward-only pass. Filtered probabilities
therefore require an in-package Hamilton forward recursion (~30 lines over `startprob_`,
`transmat_` and the per-state log-likelihoods). This is a feature, not a workaround: 2.1b
explicitly requires filtered and smoothed to be available side by side and never conflated,
because a smoothed probability uses future information and must never reach a live allocation.

### Risks to carry into Phase 2

- `hmmlearn` 0.3.3 is classified `Development Status :: 3 - Alpha`.
- The forward recursion is ours to test; it is the numerically delicate part.

### Consistency with §11

§11 refutes nothing about `hmmlearn` or `statsmodels`. The one adjacent entry — tool-use
thresholds tied to parameter count — concerns LLM sizing, not this choice. A fully hand-rolled
Hamilton filter with Kim smoother was considered and not recommended: it would make the most
numerically delicate module of Phase 2 also the one with no reference implementation to test
against.

---

## (b) Macro nowcast panel source, and which context fetches it

**Recommendation:** adopt `fredapi` inside `ingestion/`. The **ingestion daemon owns the fetch**;
`optimizer/regimes/_nowcast.py` receives the panel injected and never fetches; `fund/` is the
bridge that reads the database and passes the DataFrame.

**Rationale:** §2 forbids `ingestion/` importing `optimizer` and forbids `optimizer/` any DB or
HTTP dependency, so the only admissible placement puts the fetch in the daemon and the
computation in the library, with the fund carrying data between them.

### Placement is forced, not chosen

| Constraint | Source | Consequence |
|---|---|---|
| `ingestion/` must not import `optimizer` | §2, enforced by the `ingestion-test` CI job | the nowcast computation cannot live in the daemon |
| `optimizer/` has infrastructure dependencies: **none** | §2, must stay true word for word | the fetch cannot live in the library |
| the two contexts share the database, not the code | §8 | PostgreSQL is the only legal channel |

So: fetch → `ingestion/`; store → PostgreSQL; read → `fund/adapters/persistence/`; compute →
`optimizer/regimes/_nowcast.py`, panel injected as a parameter.

### Current state, measured

`ingestion/app/services/macro/scrapers/fred_scraper.py` already calls
`https://api.stlouisfed.org/fred` directly with `requests`, wrapped in the shared
`CircuitBreaker` and `RateLimiter`. It covers **18 series** across seven groups (spreads,
volatility, CLI, recession, rates, GDP, FX). The nowcast needs roughly 200.

### Implementation constraints attached to this decision

1. **`fredapi` must be wrapped, not substituted for, the existing resilience layer.** `fredapi`
   is an HTTP client; `CircuitBreaker`, `RateLimiter` and `retry_with_backoff` are module-level
   and shared by all three scrapers. The production-tested resilience path must survive the
   change.
2. **ALFRED vintages need no additional dependency.** They are `realtime_start` /
   `realtime_end` parameters on the same `/fred/series/observations` endpoint already in use.
3. **New persistence is required** in `ingestion/`: a macro-series panel model and repository.
   Today `ingestion/app/models/macro/` holds only `macro_regime.py`, and the existing FRED data
   has no panel table behind it.
4. Record `fredapi`'s licence and install weight when it is added in Phase 1/2. It lands in
   `ingestion/requirements.txt`, not in portopt, so the §4.5 extras precedent does not apply.
5. Keep consistent with issue [1.3] (ingestion research angle 5: idempotent upserts,
   watermarking, schema drift, incremental vs backfill). A ~200-series monthly panel with
   vintages is exactly the write path that work governs.

---

## (c) Microstructure source for impact calibration

**Recommendation:** hybrid. `optimizer/execution/_kyle_impact.py` estimates λ and η from daily
OHLCV proxies; ρ and the cross-impact matrix are **inputs** with documented defaults; every
estimated parameter is overridable from `_config.py`, and every estimate is labelled as a daily
proxy rather than a tick calibration.

**Rationale:** §10.4 is right that we hold no tick data, but we do hold complete daily OHLCV, on
which λ and η have well-established proxy estimators — and pretending otherwise would leave
`_kyle_impact.py`, which the spec describes as *"fornendone la stima/calibrazione"*, an empty
shell.

### What the data actually supports

| Table | Columns | Usable? |
|---|---|---|
| `price_history` | `date`, `open`, `high`, `low`, `close`, `volume`, `dividends`, `stock_splits` | ✅ full daily history |
| `ticker_profiles` | `bid`, `ask`, `bid_size`, `ask_size` | ❌ **no `date` column** — a snapshot overwritten on every fetch, zero history |

This refines §10.4: bid/ask are present in the schema, but only as an unhistoried snapshot, so
they cannot calibrate anything. There is no tick data, no trade sign and no book depth.

### Parameter by parameter

| Parameter | Treatment | Basis |
|---|---|---|
| λ — permanent impact (Kyle) | **estimated** | Amihud illiquidity: `mean(|r_t|) / (P_t · V_t)` over `price_history` |
| η — temporary impact | **estimated** | Corwin–Schultz effective spread from consecutive-day high/low |
| ρ — resilience (Obizhaeva–Wang) | **input**, documented default | book replenishment is not observable at daily frequency |
| cross-impact matrix | **input**, diagonal fallback | requires order flow we do not have |

Pure numpy; no new dependency; no DB, HTTP or LLM dependency reaches `optimizer/`.

### Why not the alternatives

- **Inputs only:** cleanest against §10.4, but leaves `execution/` unusable out of the box and
  moves the "where do the numbers come from" question downstream without answering it.
- **Full estimation from tick data:** requires a paid vendor, a new ingestion domain and new
  tables; blocks task 2.3 on a procurement decision that does not exist today.
- **Proxies with no override:** would prevent a user holding better calibration from using it.

---

## (d) Risk-free-rate provenance

**Recommendation:** `DGS3MO` from FRED — already ingested — converted to the return series'
periodicity **in `fund/`**, and passed to skfolio as a scalar equal to the mean over each fold's
**training** window.

**Rationale:** the 3-month T-bill is the standard proxy, it is already in the database, and
converting in `fund/` keeps `optimizer/` free of any database dependency while the per-fold
training mean keeps the backtest causal.

### Premise correction

`requirements.md` restates §10.4's claim that skfolio exposes no `risk_free_rate` parameter.
**Measured against skfolio 0.20.1, that is false:**

```
MeanRisk       risk_free_rate  default = 0.0
RiskBudgeting  risk_free_rate  default = 0.0
Portfolio      risk_free_rate  default = 0
```

Sharpe behaves as plain Return/StdDev because the default is zero, not because the parameter is
missing. The decision is therefore about provenance and units, not about working around an
absent API. Recorded in `VALIDATION.md` and in `SPEC-RECONCILIATION.md`.

### Mechanics

`FRED_RATE_SERIES` in `fred_scraper.py` already carries `DGS2`, `DGS3MO` and `DGS10`. FRED
publishes these as annualised percentages.

```
r_annual = value / 100
r_period = (1 + r_annual) ** (1 / 252) - 1        # daily returns
```

The conversion happens in `fund/`, which reads the database. `optimizer/` receives a float
already in the return series' units — no DB dependency, no unit inference.

**Walk-forward:** skfolio takes a scalar per fit. Use the mean of `DGS3MO` over each fold's
training window, never the full-sample mean and never a value that includes the test window.
This is the same class of constraint already recorded for walk-forward CV: constraints and
scalars are fixed per fold, so a backtest is single-regime per run rather than per-rebalance.

---

## (e) The Ollama model, and audit reproducibility

**Recommendation:** use Ollama's cloud models — `gemma4:cloud` recommended, for continuity with
the `gemma4` family already named in `Modelfile` and `ingestion/baml_src/clients.baml`. Base
audit reproducibility (§12 point 3) on **recording**, via `fund/app/services/_shared/audit_log.py`,
not on the ability to reproduce a run bit-for-bit.

**Rationale:** `seed` plus `temperature=0` was measured not to guarantee determinism, so an audit
trail that depends on re-running the model would be unsound whatever model is chosen.

### Measured: `seed` does not deliver determinism

Three identical requests, `temperature=0`, `seed=42`, same model, same prompt:

```
run 1: 'The vast, shimmering expanse of the sea holds a world of mystery...'
run 2: 'The vast, turquoise expanse of the sea stretches endlessly...'
run 3: 'The vast, turquoise expanse of the sea stretches endlessly...'
```

Run 1 differs. This is the ordinary consequence of batched inference: batch composition changes
the order of floating-point reductions. §12 point 3 cannot rest on `seed`.

`temperature 0` is already set in `ingestion/baml_src/clients.baml`; `seed` is not. Declaring
`seed` is still worth doing — it reduces variance — but it must not be presented as a
determinism guarantee.

### Consequences the author should see

1. **§9's "LLM self-hosted via Ollama" premise no longer holds.** Fund prompts — positions,
   macro views, proposed orders — leave the machine for `ollama.com`. This is a change in the
   system's confidentiality profile and §9 needs amending to describe what the system does.
   That amendment is outside discrepancies (a)–(g) and has not been applied.
2. **`.env` points at a model that is not installed.** `OLLAMA_MODEL=gemma4:12b-mlx`, and all
   four `OLLAMA_MODEL_*` agent variables, name a model absent from this host: every model
   present reports 0.0 GB and a `-cloud` suffix, i.e. a proxy to `ollama.com`. This is a real
   bug independent of this decision and must be fixed regardless.
3. **Add a startup check.** `fund/` should query `/api/tags` at boot and refuse to start if
   `OLLAMA_MODEL` is absent. That check is what would have surfaced finding 2 immediately
   instead of at Phase 0 review. Worth adding whichever model is chosen.

### What the audit records instead

`audit_log.py` is already in the `fund/` specification: model and version, every inference
parameter, the prompt hash and the complete output, written **before** the step is checkpointed.
Reproducibility becomes verifiability of what happened, which is what an audit needs and what a
non-deterministic model can actually support.
