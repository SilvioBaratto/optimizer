# VALIDATION — Phase 0, issue [0.3]

Re-validation of the time-sensitive facts of §9 and §12 *Ancora aperte* point 4.
`todo/TASKS.md` heads the same task "§9/§12.4"; `ARCHITECTURE.md` has no §12.4 — the section
meant is §12 point 4, reinforced by §13's fifth bullet.

**Probe date: 2026-08-19.** Python symbols were imported in a throwaway virtualenv that is not
committed; only the results below are. The Trading 212 row was validated by live read-only
HTTP calls using the key from `.env` — no order-submitting request was issued.

---

## Results

| # | Symbol / fact | §9 claims | Version checked | Outcome | Correct import path |
|---|---|---|---|---|---|
| 1 | `EventSourceResponse`, `ServerSentEvent` | `fastapi.sse`, FastAPI ≥ 0.135.0 | fastapi **0.139.0** | ✅ **PASS** | **`fastapi.sse`** — carries both symbols |
| 2 | `sse-starlette` fallback | v3.4.5 | **3.4.8** | ✅ **PASS**, version drift | `sse_starlette` — both symbols present |
| 3 | `PostgresSaver` / `AsyncPostgresSaver` | LangGraph durable substrate | `langgraph-checkpoint-postgres` **3.1.2** | ✅ **PASS** | `langgraph.checkpoint.postgres` / `langgraph.checkpoint.postgres.aio` |
| 4 | `interrupt()` / `Command(resume=...)` | LangGraph interrupt surface | `langgraph` **1.2.11** | ✅ **PASS** | `langgraph.types` |
| 5 | Trading 212 REST order endpoints | `POST /api/v0/equity/orders/{market,limit,stop,stop_limit}`, demo/live base URLs | live API, 2026-08-19 | ⚠️ **PARTIAL** — see below | `https://{live,demo}.trading212.com/api/v0` |
| 6 | skfolio solver defaults | CLARABEL default, SCIP for mixed-integer | skfolio **0.20.1** | ✅ **PASS** | `MeanRisk(solver="CLARABEL")` |

---

## Row 1 — `fastapi.sse` vs `fastapi.responses`

§9 flags this as the specific ambiguity to settle. Settled:

| Module | `EventSourceResponse` | `ServerSentEvent` |
|---|---|---|
| `fastapi.sse` | ✅ | ✅ |
| `fastapi.responses` | ✅ | ❌ |

**Use `fastapi.sse`.** It is the only path exposing both symbols; `fastapi.responses` re-exports
only the response class. The §9 claim is correct as written.

Installed FastAPI is 0.139.0, above the 0.135.0 floor §9 states.

## Row 2 — `sse-starlette`

Installed 3.4.8 against the 3.4.5 §9 records. Both `EventSourceResponse` and `ServerSentEvent`
import cleanly. The fallback remains viable; §9's pinned version is stale by three patch
releases, which is drift, not breakage.

## Rows 3–4 — LangGraph durable substrate

```
langgraph                        1.2.11
langgraph-checkpoint-postgres    3.1.2
langgraph-checkpoint             4.2.0
```

- `PostgresSaver` — `langgraph.checkpoint.postgres`; `from_conn_string` and `setup` both present
- `AsyncPostgresSaver` — `langgraph.checkpoint.postgres.aio`
- `interrupt` — `langgraph.types`, signature `(value: Any) -> Any`
- `Command` — `langgraph.types`, carries a `resume` field

The surface §9 depends on is intact. Note that `langgraph-checkpoint` resolved to 4.2.0 as a
transitive dependency of `langgraph-checkpoint-postgres` 3.1.2 — the two version lines are
independent and should be pinned together.

## Row 5 — Trading 212, partially validated

Validated read-only. Three findings, one of which contradicts §9.

**5a — base URLs confirmed.**

| Request | Result |
|---|---|
| `GET https://live.trading212.com/api/v0/equity/account/info` | **HTTP 200** |
| `GET https://demo.trading212.com/api/v0/equity/account/info` | **HTTP 401** |
| `GET https://live.trading212.com/api/v0/equity/orders` | **HTTP 200** |

Both hosts are real and enforce separate credential domains. `.env` carries
`TRADING_212_MODE=live`, so the 401 against demo is the expected behaviour of a live-scoped
key, not a broken URL. This also means **clarification 2's "live demo API call" could not be
performed** — validating demo requires a separate demo key.

**5b — the authentication scheme in §9 is incomplete.** §9 describes demo/live as
"scambio di base-URL e chiave". Sending the API key as a bare `Authorization` header returns
**401**. The scheme that works is HTTP Basic:

```
Authorization: Basic base64(TRADING_212_API_KEY:TRADING_212_SECRET_KEY)
```

This matches what `ingestion/app/services/universe/trading212/client.py` already does in
production. §9 omits the secret entirely. A `fund/` broker adapter written from §9 alone would
fail to authenticate.

**5c — the four order endpoints remain UNVERIFIED.** This is an honest gap, not a pass.

| Path | `GET` result | Reading |
|---|---|---|
| `/api/v0/equity/orders` | 200 | list-orders endpoint exists |
| `/api/v0/equity/orders/market` | 400 | — |
| `/api/v0/equity/orders/limit` | 400 | — |
| `/api/v0/equity/orders/stop` | 400 | — |
| `/api/v0/equity/orders/stop_limit` | 400 | — |

**400, not 404 and not 405.** That is consistent with `GET /api/v0/equity/orders/{id}` parsing
`market` as a malformed order id — it proves the path prefix routes, **not** that
`POST /api/v0/equity/orders/market` exists. Confirming the four POST endpoints for real would
require submitting an order, which clarification 5 forbids.

§11 records "Trading 212 has no official order API, only Selenium" as a refuted idea. Nothing
here resurrects it — the REST API demonstrably exists and answers. What is unproven is the
shape of the four order sub-paths, and that should be closed at unlock time, against the demo
account, together with the idempotency and reconciliation work §12 already defers there.

## Row 6 — skfolio solver defaults

Against the installed release, not the floor:

```
skfolio 0.20.1
MeanRisk.solver        default = 'CLARABEL'
MeanRisk.solver_params default = None
```

Docstring, verbatim: *"open-source option, we recommend using SCIP by setting `solver="SCIP"`."*
Both halves of the §9 claim hold. Per §9's own note, this must be re-checked at every `skfolio`
bump — a solver default upstream is a choice, not a contract.

---

## Correction to §10.4, found during this pass

§10.4 states that skfolio exposes no `risk_free_rate` parameter on `MeanRisk`, `RiskBudgeting`,
`Portfolio` or `RatioMeasure`. **That is false in skfolio 0.20.1:**

```
MeanRisk       risk_free_rate  default = 0.0
RiskBudgeting  risk_free_rate  default = 0.0
Portfolio      risk_free_rate  default = 0
```

The parameter exists on all three. The *practical* consequence §10.4 draws is still right —
`SHARPE_RATIO` behaves as plain Return/StdDev — but the cause is that the default is zero, not
that the parameter is absent. This materially changes decision 0.2(d): the task is supplying a
value and its units, not working around a missing API. See `DECISIONS.md` (d).

This correction is outside discrepancies (a)–(g); it is recorded in `SPEC-RECONCILIATION.md`
under *Findings outside (a)–(g)* and has not been applied to the spec source.
