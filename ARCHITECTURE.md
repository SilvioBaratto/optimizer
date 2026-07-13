# Architecture

Target architecture for the end-to-end quantitative investment platform described in
[`TODO.md`](./TODO.md). This document is the reference for how the system should be
structured as it grows from the current two shipped pieces (the ingestion daemon and the
`portopt` optimization library) into a platform with a four-agent LLM investment fund, an
HTTP API, and a rich frontend.

It records **decisions and their rationale**, not implementation. It was synthesized from a
multi-source, adversarially-verified research pass; claims that were investigated and
**refuted** are listed explicitly so we do not accidentally build them later.

---

## 1. Guiding principles

1. **Clean / hexagonal architecture with a strict inward-pointing dependency rule.**
   High-level domain logic never depends on low-level infrastructure. The domain layer is
   unaware of ports, adapters, or anything above it. Source code dependencies point inward
   only.

2. **Strict separation of concerns via bounded contexts.** Ingestion, optimization compute,
   agent orchestration, and the API are separate bounded contexts. Each is internally
   layered (domain → ports → adapters) and talks to the others only through explicit ports.

3. **Deterministic control wraps non-deterministic LLMs.** Every LLM-initiated action —
   especially a trade — passes through a deterministic validation layer and a human approval
   gate before it can have real-world effect. Rules are evaluated *before* the LLM is even
   consulted where possible.

4. **Reproducibility and auditability are engineered, not assumed.** No LLM (local or
   frontier) delivers both high accuracy and high determinism. We build the audit trail,
   checkpointing, and structured-output boundary around the models rather than trusting the
   models to be reproducible.

5. **Modular monolith first; carve out services only on proven evidence.** Keep the module
   seams clean so extracting a service later is a mechanical operation, not a rewrite.

---

## 2. Bounded contexts

Four bounded contexts. Each is hexagonal: a pure domain core, ports (interfaces) describing
what it needs from the outside world, and adapters (concrete implementations) at the edge.

| Bounded context | What it owns | Current code | Infra dependencies |
|-----------------|--------------|--------------|--------------------|
| **Ingestion** | Scheduled market/fundamental/macro data fetch into PostgreSQL | `api/` daemon | yfinance, FRED, Trading 212, news scrapers, Postgres |
| **Optimization compute** | Portfolio optimization (mean-risk, HRP/HERC, risk budgeting, robust/DR-CVaR, factor models, Black-Litterman) | `optimizer/` (`portopt`) | **None** — pure compute, stays that way |
| **Agent orchestration** | The four-agent fund: macro, allocation, risk, execution | *new* | Ollama (via BAML), broker API, Postgres checkpointer |
| **API** | HTTP surface for the frontend | *new* (was stripped) | The service layer only |

The optimization compute context is already a pure library with no DB, HTTP, or LLM
dependencies. It **must stay that way** — it is the reference example of a clean domain core.
The ingestion daemon already uses the repository pattern; that convention extends across the
whole platform.

---

## 3. Layering

**Dependency direction is the whole point of this diagram: every arrow points *inward*,
toward the core.** The frontend is the outermost driving adapter — a *client* of the API.
Nothing in the core knows the frontend exists; the core would run unchanged if the frontend
were deleted and replaced by a CLI or a test harness. This is the ports-and-adapters layout:
**driving** adapters on the left initiate calls into the core, the core calls out through
ports to **driven** adapters on the right, and all source-code dependencies point at the
center.

```
        DRIVING SIDE                        CORE                       DRIVEN SIDE
   (actors that call in)          (domain — depends on nothing)   (things the core calls out to)
   ───────────────────►  ═══════════════════════════════════►  ───────────────────►
        dependencies point inward  ──►        ◄──  dependencies point inward

 ┌─────────────────┐            ┌──────────────────────────────┐
 │ FRONTEND        │            │  SERVICE / USE-CASE LAYER    │        ┌──────────────────┐
 │ optimizer UI    │            │  "run optimization"          │──port─►│ Postgres repos   │
 │ (a client;      │──REST────► │  "run fund deliberation"     │        └──────────────────┘
 │  core is blind  │  +SSE/WS   │  "approve trade"             │        ┌──────────────────┐
 │  to it)         │  via API   │  fetch → update → persist    │──port─►│ yfinance / FRED /│
 └─────────────────┘  adapter   └───┬───────────┬───────────┬──┘        │ T212 / news      │
                          │         │           │           │           └──────────────────┘
 ┌─────────────────┐      │      ┌──▼──────┐ ┌──▼──────┐ ┌──▼─────────┐  ┌──────────────────┐
 │ API ADAPTER     │      │      │ DOMAIN: │ │ DOMAIN: │ │ AGENT      │  │ Ollama (via BAML)│
 │ (thin, FastAPI) │──────┘      │ portopt │ │ingestion│ │ ORCHESTR.  │◄─┤ typed LLM I/O    │
 │ parse · authz · │             │ (pure)  │ │ (rules) │ │ (graph)    │  └──────────────────┘
 │ JSON · SSE ·    │             │         │ │         │ │            │  ┌──────────────────┐
 │ NO logic        │             │mean-risk│ │         │ │ macro →    │  │ broker API       │
 └─────────────────┘             │HRP/HERC │ │         │ │ alloc →    │──┤ (real orders)    │
                                 │BL/CVaR  │ │         │ │ risk       │  └──────────────────┘
 ┌─────────────────┐             │factors  │ │         │ │   ↓interrupt  ┌──────────────────┐
 │ SCHEDULER       │──invokes──► │         │ │         │ │ GUARDRAIL  │──┤ state checkpointer│
 │ (APScheduler;   │  use-cases  │ NO DB   │ │         │ │ (rules,    │  │ (PostgresSaver)  │
 │  drives         │             │ NO HTTP │ │         │ │  kill-sw)  │  └──────────────────┘
 │  ingestion)     │             │ NO LLM  │ │         │ │   ↓        │
 └─────────────────┘             └─────────┘ └─────────┘ │ HUMAN GATE │
                                                         │   ↓ exec   │
                                                         └────────────┘
```

The four-agent orchestration graph is *inside* the core, on the right. It reaches the outside
world (Ollama, the broker, the checkpointer) only through **ports** — the LLM, broker, and
checkpointer boxes on the driven side are adapters implementing those ports. So the agents
never import an HTTP client or the frontend; they call use-cases and ports, nothing more.

**Responsibilities per layer:**

- **Driving adapters (frontend, API adapter, scheduler)** — the actors that call *into* the
  core. The frontend is a client of the API; the API adapter does web concerns only (parse,
  authz, JSON, SSE, error→status) with no business logic; the scheduler drives ingestion.
  None of them contain domain logic, and the core depends on none of them.
- **Service / use-case layer** — orchestrates: fetch data from a repository, update the
  domain model, persist changes. Depends on repository *abstractions*, so it runs against
  fakes in tests and real adapters in production. This is the single entry point into the
  core — every driving adapter goes through it.
- **Domain** — business rules. Pure. No framework, DB, or network imports.
- **Ports** — interfaces the domain/services declare (a broker port, a repository port, an
  LLM port).
- **Driven adapters** — concrete implementations of ports at the system edge (Postgres,
  data-source clients, Ollama/BAML, broker API, checkpointer). The core calls these; they
  never call the core.

---

## 4. Folder structure

Monorepo with **four top-level deliverables**, each a bounded context (or its client):

- `optimizer/` — the `portopt` library: the pure optimization-compute domain. Carries the
  heavy sklearn/skfolio/scipy stack. Untouched by this design.
- `api/` → **target name `ingestion/`** — the ingestion daemon. The `api/` name is a leftover
  from when it had an HTTP API (stripped on `refactor/strip-to-ingestion-pipeline`); it is a
  schedule-driven **worker/daemon**, not an HTTP service, so the standard monorepo convention
  is a domain-descriptive worker name (`ingestion/`), not `api/`. Rename is deferred (touches
  Docker, CI, `scheduler/` wrappers, CLAUDE.md) — tracked as a dedicated refactor. Stays lean
  and isolated; **must not import `optimizer`** and carries no sklearn stack (existing hard
  rule, and TODO.md wants the scheduler lightweight and failure-proof). It only writes
  market/macro/fundamental data to PostgreSQL.
- `fund/` — **new.** The agent fund + service layer + HTTP API. This is the app that carries
  the heavy stack: it *reads* the ingestion DB, *calls* `portopt` for compute, runs the
  four-agent graph, and executes trades. Keeping it separate from `api/` is what lets the
  scheduler stay lean.
- `frontend/` — **new.** A client of the `fund` HTTP API. The core is blind to it.

`api/` and `fund/` **share the database, not code** — each owns its own read/write access
layer (repositories + models). That is the deliberate bounded-context boundary: a shared
Postgres with a per-context access layer, not a shared ORM.

```
optimizer/                                  repo root (monorepo)
│
├── optimizer/                          ┃ CORE · optimization-compute context (portopt)
│   ├── preprocessing/  moments/  views/     pure domain — no DB, no HTTP, no LLM
│   ├── optimization/  pre_selection/        (existing submodules, unchanged)
│   ├── factors/  synthetic/  scoring/
│   ├── distance/  cluster/  uncertainty_set/
│   ├── linear_model/  online/  fx/
│   ├── universe/  validation/  tuning/
│   ├── rebalancing/  pipeline/
│   └── __init__.py
│
├── api/  →  ingestion/                     INGESTION DAEMON (lean; no optimizer import)
│                                           (rename deferred; api/ = leftover HTTP-era name)
│   ├── app/
│   │   ├── models/          ┃ DRIVEN · SQLAlchemy ORM (jobs/ macro/ market_data/ universe/ _shared/)
│   │   ├── repositories/    ┃ DRIVEN · repository adapters (same domains + _shared/)
│   │   ├── services/        ┃ CORE   · ingestion use-cases (+ infrastructure/ _shared/)
│   │   ├── schemas/         ┃          typed arg objects / progress payloads
│   │   ├── worker.py        ┃ DRIVING · APScheduler daemon
│   │   ├── cli.py           ┃ DRIVING · manual runs
│   │   └── metrics.py
│   ├── baml_src/            ┃ BAML defs already here (news summarize, macro regime)
│   ├── baml_client/         ┃ generated (do not edit)
│   ├── alembic/  tests/  requirements.txt  pyproject.toml  Dockerfile
│
├── fund/                               NEW · AGENT FUND + SERVICE LAYER + HTTP API
│   ├── app/
│   │   ├── domain/          ┃ CORE   · pure domain models
│   │   │   ├── fund/             FundState (the contract) · MacroView ·
│   │   │   │                     Allocation · RiskVerdict · Approval
│   │   │   ├── trading/          Order · Position · GuardrailRule · KillSwitch
│   │   │   └── _shared/
│   │   ├── ports/           ┃ PORTS  · interfaces the core declares
│   │   │   ├── agent.py          Agent.run(FundState) -> FundState
│   │   │   ├── broker.py         place_order / cancel / positions
│   │   │   ├── llm.py            typed agent inference (BAML-backed)
│   │   │   ├── market_data.py    read prices / fundamentals / macro
│   │   │   └── checkpointer.py   graph-state persistence
│   │   ├── services/        ┃ CORE   · use-case / orchestration layer
│   │   │   ├── optimization/     run_optimization (wraps portopt)
│   │   │   ├── fund/             run_deliberation · approve_trade
│   │   │   └── _shared/
│   │   ├── agents/          ┃ CORE   · four-agent orchestration graph
│   │   │   ├── orchestrator.py   hand-rolled loop over FundState (the mediator)
│   │   │   ├── macro.py  allocation.py  risk.py  execution.py
│   │   │   │                     each implements ports/agent.py · no cross-imports
│   │   │   ├── guardrail.py      deterministic rule engine + kill-switch
│   │   │   └── prompts/          BAML function bindings
│   │   │                         (shared state = domain/fund/FundState)
│   │   ├── tools/           ┃ CORE   · small fixed tool registry (§5.3)
│   │   │   ├── registry.py       enum of tools; hand-registered at build time
│   │   │   ├── intent.py         BAML-typed ToolIntent (enum tool + typed args)
│   │   │   └── run.py            validate → dispatch → retry → fail-safe
│   │   │                         (read/compute only; in-process; no side effects)
│   │   ├── adapters/        ┃ DRIVEN · concrete port implementations
│   │   │   ├── persistence/      Postgres repos (reads ingestion DB) + models
│   │   │   ├── broker/           real broker API client
│   │   │   ├── llm/              Ollama + BAML client
│   │   │   └── checkpointer/     PostgresSaver
│   │   ├── http/            ┃ DRIVING · thin FastAPI adapter
│   │   │   ├── app.py            application factory
│   │   │   ├── routers/          optimize · fund · trades · stream
│   │   │   ├── deps.py           DI: wire services → routers
│   │   │   └── sse.py            SSE streaming (deliberation / progress)
│   │   └── main.py              entrypoint (uvicorn)
│   ├── baml_src/  baml_client/  tests/  requirements.txt  pyproject.toml  Dockerfile
│
├── frontend/                           NEW · DRIVING · client of the fund HTTP API
│   ├── src/
│   │   ├── api/                 generated/typed client for the HTTP API
│   │   ├── features/            optimizer dashboard · deliberation view · approvals
│   │   ├── components/  hooks/  lib/
│   │   └── main.*
│   ├── package.json  tsconfig.json  Dockerfile
│
├── tests/                                  library test suite (mirrors optimizer/)
├── scheduler/                              shell wrappers over api CLI
├── scripts/                                CI helpers
├── docker-compose.yml                      db · adminer · scheduler · fund · frontend
├── pyproject.toml                          portopt (library) config
├── ARCHITECTURE.md   TODO.md   CLAUDE.md   README.md
```

The `┃` column marks each module's hexagonal role: **DRIVING** adapters call into the core,
**CORE** is domain + use-cases + the agent graph (depends on nothing outward), **PORTS** are
the interfaces the core declares, **DRIVEN** adapters implement those ports at the edge.

---

## 5. The four-agent fund

The four agents are nodes driven by a **hand-rolled orchestrator** over the shared
`FundState` (§6), with LangGraph adopted only as the durable checkpoint/interrupt substrate.
They share the typed state and hand off through it — never directly. They do **not** call raw
HTTP endpoints; they call internal **use-cases** as tools, which keeps the service layer as
the single orchestration boundary.

```
Macro agent ──────┐
                  ├──►  shared graph state  ──►  proposed allocation
Allocation agent ─┤
Risk agent ───────┤
Execution agent ──┘

Every trade the execution agent proposes passes through, in order:

  1. Deterministic pre-check   — context conditions evaluated with NO LLM call
                                 (kill-switch, hard limits). Deterministic flags
                                 always override probabilistic LLM routing.
  2. Rule-based validation     — position / exposure / concentration limits.
                                 CLASSICAL rule engine, NOT formal proof.
  3. Human approval gate       — orchestrator interrupt(); a human signs off.
  4. Broker adapter            — only now does a real order leave the system.

  Every state transition is checkpointed → complete, replayable audit trail.
```

Each agent's LLM output crosses a **typed structured-output boundary (BAML)** so the rest of
the system receives validated, typed objects — never free-form text.

### 5.1 How agents communicate (the code-level contract)

The requirement is that each agent stays a **clearly independent, easily maintainable unit**
while remaining fully connected to the others. The answer is a **blackboard / mediator**
pattern, *not* a network protocol (see §5.2 for why A2A is deferred):

> **Agents never talk to each other directly. They only read from and write to one typed
> shared state object, and an orchestrator decides who runs when.**

Three rules enforce it:

1. **No agent imports another agent.** `risk.py` never imports `allocation.py`. Zero
   cross-imports is what makes each agent independent — changing one cannot break another.
2. **Agents communicate only through the typed state object.** An agent reads the fields it
   needs and writes the fields it produces. That typed state object *is* the protocol. An
   agent's internals can be rewritten freely as long as it still honors the state contract.
3. **Every agent implements one interface (a port).** The orchestrator treats all four
   identically and can substitute a fake in tests by swapping one line.

```python
# domain/fund/state.py — THE contract. the one shared thing.
@dataclass(frozen=True)
class FundState:
    macro_view:      MacroView        | None = None   # written by macro agent
    allocation:      Allocation       | None = None   # written by allocation agent
    risk_verdict:    RiskVerdict       | None = None   # written by risk agent
    proposed_orders: tuple[Order, ...]  = ()           # written by execution agent
    approvals:       tuple[Approval, ...] = ()          # written by the human gate

# ports/agent.py — every agent looks identical from outside.
class Agent(Protocol):
    def run(self, state: FundState) -> FundState: ...

# agents/risk.py — a sealed box: imports state + its own tools, NOT other agents.
def run(state: FundState) -> FundState:
    verdict = assess(state.allocation, state.macro_view)   # its own logic
    return replace(state, risk_verdict=verdict)            # leave a message on the board
```

`FundState` is frozen, typed, and serializable, so it doubles as the **audit record**: the
same object that decouples the agents is what gets checkpointed at every step. One object
delivers both decoupling and auditability.

**Why this shape:** agents calling each other directly welds them together (change one → break
all); a network protocol (A2A) buys independence at the cost of a distributed-systems tax we
don't need yet. The blackboard pattern gives independence *and* connection at zero
infrastructure cost — and it is exactly what LangGraph implements natively (nodes + typed
`State` + edges), with checkpointing and interrupt gates added on top.

**Maintenance payoff:** each agent is independently **testable** (feed a `FundState`, assert
the `FundState` out — no other agent needed), **swappable** (rewrite the file, honor the
contract), **readable** (one file = one agent's whole job), and **replaceable by a remote
implementation later** (§5.2) without the orchestrator or the other agents noticing.

### 5.2 A2A / MCP as a future adapter (deferred, not adopted)

The A2A (Agent2Agent) protocol was evaluated and **deferred**. A2A exists to let *opaque*
agents built by *different vendors, frameworks, or languages* discover and call each other
over JSON-RPC/HTTP. Our four agents are one team, one repo, one language, sharing types and a
database — none of A2A's premises hold, and the spec's own "overkill when" list (tightly
coupled, synchronous, same infrastructure, no multi-tenant isolation) matches our situation
exactly. Adopting it now would mean building a distributed-systems protocol so our own
functions can call each other.

Because every agent already sits behind the `Agent` port (§5.1), adopting A2A later is a
**single adapter**, not a rewrite: an adapter that satisfies `Agent.run()` by calling a
remote agent over A2A and returning a `FundState`. Real triggers to revisit: (a) a
third-party or externally-owned agent must join the graph; (b) one agent must be written in
another language; (c) separate teams own separate agents behind a stable contract; (d) agents
are published for external discovery. None are true today.

Related: **MCP** (Model Context Protocol) is the complementary standard for *agent → tools*
access (A2A is *agent → agent*). If the requirement "agents take actions via our APIs" ever
needs a standardized tool interface, MCP is the more relevant standard to evaluate before
A2A. For now, agents call internal use-cases directly through the service layer (§5.3).

### 5.3 How agents use tools (local-model-safe)

"Agents take actions" (TODO.md) means calling tools — e.g. the risk agent computing risk on a
proposed allocation. The design is shaped by one hard fact: **small self-hosted models (7-30B)
are unreliable at native tool-calling** — they drop calls, hallucinate tool names, and emit
malformed JSON arguments. So we do **not** hand the LLM an autonomous tool-calling loop
(ReAct). Instead:

**1. Tools are code-driven by default — the LLM does not decide when they run.** The
orchestrator/agent code invokes the tool deterministically and passes the *result* to the LLM
to interpret. Concretely for the risk agent: the risk computation (a `portopt` call) **always
runs** on the proposed allocation; the local model only receives the numbers and writes the
`RiskVerdict` judgment. Risk math never depends on the model choosing to call it, and a weak
model cannot silently skip it.

**2. When a genuine choice exists, the LLM emits a typed *intent*, not a native tool call.**
The model returns a **BAML-typed object** with an enum `tool` field plus typed arguments —
grammar-constrained, so a weak model *cannot* produce an invalid tool name or unparseable
args. Code validates and dispatches. We do **not** use Ollama's native function-calling API
(it duplicates BAML's role and is less reliable on small models). This is "pick from a menu,"
not open-ended tool-calling.

**3. A tool is an in-process Python call behind a port.** Each tool is a use-case / `portopt`
call / port method — no network, no shell, no subprocess. The registry is a **small fixed set,
hand-registered at build time** (calc risk, run optimization, fetch data, propose orders). No
runtime discovery, no MCP. Agents reach tools through the same service-layer use-cases the API
exposes, so the agent path and the HTTP path share one action surface.

**4. Failure is validate → retry → fail safe.** Pydantic validates the typed intent; on failure
the step retries N times with the validation error fed back into the prompt; if it still fails,
the agent step **aborts and surfaces the failure** to the audit log — never a silent guess.
(A tool-intent failure may optionally escalate to the human gate via the LangGraph interrupt,
reusing the HITL substrate of §6.)

**Trust boundary.** LLM-accessible tools are **read-only or pure-compute** (fetch data, compute
risk, run optimization). Side-effecting actions — placing an order, spending money — are
**never** LLM-callable tools: they go through the deterministic guardrail + human approval gate
(§5, §6). The LLM proposes; it never pulls a trigger with real-world effect.

---

## 6. Key technology decisions

A second research pass (fitted to the fixed constraints above) settled the concrete library
choices. The headline reversal from the first pass: **we do NOT adopt LangGraph as the primary
orchestrator.** LangGraph's core value is channel/reducer graph state persisted as
thread-scoped checkpoints — which *duplicates the `FundState` contract we already own* (§5.1).
For four agents over a state object we control, a full graph engine mostly fights us.

### Orchestrator — hand-rolled over `FundState` (medium confidence)
A plain Python orchestrator iterates the four `Agent.run(FundState) -> FundState` ports over
the single shared state. No graph-framework DSL. Maximizes readability and avoids a framework
fighting our own state object. This *is* the blackboard pattern from §5.1 — the orchestrator
is the mediator.

### Durable substrate — LangGraph for checkpoint + interrupt ONLY (medium confidence)
The one thing worth borrowing from LangGraph is its battle-tested **PostgreSQL checkpointing**
(`PostgresSaver` / `AsyncPostgresSaver`, `langgraph-checkpoint-postgres`) plus
**`interrupt()` / `Command(resume=...)`** for pause-for-human-approval. For a real-money-later
system this is more reliable than a hand-rolled pause/resume, so we adopt it *solely* as the
durable substrate: **mount `FundState` as the single graph channel** so there is no state
duplication — the hand-rolled control flow stays ours; LangGraph only persists, pauses, and
resumes. **Gotcha:** on resume, an interrupted node re-executes from its top, so node logic
**must be idempotent** — this matters directly for the execution agent (see broker
idempotency below).

### Deterministic guardrail — Pydantic v2 validators (high confidence)
No rules-engine dependency. `@field_validator` (single field) + `@model_validator(mode='after')`
(cross-field: position / exposure / concentration limits + kill-switch) are the pass/fail
boundary. A violation **raises `ValueError` / `PydanticCustomError`** (→ `ValidationError`) —
this same clean boundary serves both the **pre-LLM deterministic pre-check** and the
**interrupt gate**. Lead with `ValueError`, **never `AssertionError`** (skipped under
`python -O`, which would silently disable a guardrail). Declarative JSON rule libraries
(`business-rules`, `json-logic-py`) are both minimally maintained → use only as a *serializable
data format* for limit definitions if desired, never as a load-bearing dependency.

### Deterministic-before-probabilistic hand-offs (high confidence)
Business-critical conditions are checked **first, with no LLM call**, so a tripped limit or the
kill-switch deterministically takes priority over any LLM routing. The Pydantic guardrail above
is the mechanism; the AG2 context-condition scheme is the reference pattern.

### Self-hosted LLMs via Ollama + BAML — BAML only (medium confidence)
BAML stays the single structured-output boundary behind `ports/llm.py`. Its schema-aligned
parsing produces typed output from local models (Llama 3.1, Mistral, Gemma 2) through the
`openai-generic` provider — already the mechanism in `api/baml_src/`. **PydanticAI was
evaluated and rejected** as an alternative: it is capable (native Ollama, grammar-constrained
JSON), but adopting it would run a *second* structured-output mechanism alongside BAML for no
gain. One boundary, one mechanism. **Caveat:** Ollama's schema enforcement only guarantees
grammar-valid tokens *at generation time* — a mid-generation truncation can still yield invalid
JSON, so keep a validate/retry layer at the port.

### Broker — Trading 212 official REST API behind the broker port (high confidence)
Trading 212 **does** expose an official public REST API that places real orders —
`POST /api/v0/equity/orders/{market,limit,stop,stop_limit}` for stocks and ETFs (the
"no API, Selenium-only" claim was refuted). Paper-first is a **base-URL + separate-key swap**:
demo `https://demo.trading212.com/api/v0` vs live `https://live.trading212.com/api/v0`. Build
a thin **`httpx`** client behind the broker port taking environment + key as config; do **not**
use the Selenium libraries (`pytrading212` = fragile UI scraper). **Advisory-first is our own
policy gate, not a broker limitation** — the capability exists; we choose to keep it off.
**Material caveats for real money:** the order API is BETA and **non-idempotent** (duplicate-order
risk — see §9), Invest / Stocks-&-Shares-ISA accounts only (no SIPP/CFD), main-account-currency
only, no order-by-value.

### Transport — FastAPI native SSE (high confidence)
Stream per-node deliberation events and `portopt` progress with FastAPI's **built-in SSE**
(`fastapi.sse.EventSourceResponse` + `ServerSentEvent`, added in FastAPI 0.135.0) — zero extra
dependency, with 15s keep-alive ping, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`
(defeats Nginx buffering) out of the box. It is built on **`sse-starlette`** — drop to that
directly only if we need knobs beyond the native layer. `ServerSentEvent`'s `data/event/id/retry`
fields carry typed per-node events; `Last-Event-ID` gives reconnection. **Note:** the repo has
*no* HTTP layer today (the daemon is headless APScheduler) — this FastAPI + SSE backend is
net-new, and lives in `fund/app/http/` (§4).

### Modular monolith first (fetch-phase evidence, not independently verified)
For a small team, microservices cost roughly 30-50% more operational effort for equivalent
functionality. The recommended shape is an **agent orchestrator as a modular monolith**, with
tool runners / retrieval externalized only if a real bottleneck appears. Likely first
candidate to extract: the execution agent, if it needs failure isolation from the rest.

### Concrete package list

| Concern | Adopt | Notes |
|---------|-------|-------|
| Orchestrator | **hand-rolled Python** over `FundState` | no graph DSL; the mediator of §5.1 |
| Durable pause/resume | **LangGraph** `PostgresSaver` + `interrupt()` **only** | `FundState` = single graph channel; nodes idempotent |
| LLM structured output | **BAML** behind `ports/llm.py` | PydanticAI rejected; keep validate/retry |
| Guardrail | **Pydantic v2** validators | `ValueError`, not `AssertionError`; JSON rule libs = data-only |
| Broker | **`httpx`** → official Trading 212 REST | demo/live = base-URL + key swap; add idempotency keys |
| Transport | **`fastapi.sse`** (FastAPI ≥ 0.135.0) | `sse-starlette` as fallback |

*Time-sensitivity: FastAPI native SSE (0.135.0), `sse-starlette` v3.4.5, and the Trading 212
live-order rollout are all at/after the Jan-2026 knowledge cutoff and were established by the
research pass — verify current versions and import paths (`fastapi.sse` vs `fastapi.responses`)
at implementation time.*

---

## 7. Ingestion daemon — verified design decisions

A dedicated research pass evaluated whether the ingestion daemon (`api/` → `ingestion/`, §4)
should migrate to a distributed task queue (Celery/Temporal/Dramatiq), async I/O
(asyncio/httpx-async/asyncpg), or multi-replica cron. **Verdict: keep the current
architecture** — in-process APScheduler + synchronous SQLAlchemy 2.0 + thread workers +
heartbeat/reaper — and apply four targeted refinements. The migration-justifying claims were
adversarially **refuted** (see §8); the "modernizations" would have been mistakes for this
workload.

### 7.1 Scheduler — keep in-process APScheduler; one daemon per DB is correct
APScheduler 3.x forbids sharing a job store between schedulers (*"Job stores must never be
shared between schedulers"*), so the **"exactly one daemon per DB" rule is the correct standard
mitigation, not a bug to fix**. For non-idempotent cron jobs the right bias is to **skip a
launch rather than risk a double launch** (Google SRE) — true exactly-once needs synchronous
Paxos consensus, which single-process APScheduler neither provides nor needs here. APScheduler
4.x (sanctioned multi-scheduler) was still pre-release at the Jan-2026 cutoff — do not plan a
4.x scale-out without revalidation.

### 7.2 Scaling path — documented, not built (advisory-lock leader election)
Single daemon per DB stays the design **now**. If fetch wall-clock ever outgrows the daily
window, the sanctioned scale path is a **PostgreSQL advisory-lock leader election**
(`pg_try_advisory_xact_lock`): replicas compete, only the lock holder fires cron triggers,
others act as hot standbys or pure fetch workers. Reuses the Postgres already in place — no
Redis/Redlock. **Deployment caveat:** PgBouncer transaction-pooling breaks session-level
advisory locks — use the transaction-scoped variant. No code now; recorded as the future path.

### 7.3 Orphan reaper — lease-based, and RECLAIM instead of FAIL
Two decisions, with a dependency:
- **Lease-based reaper (adopt).** Workers renew `last_heartbeat_at` on in-flight jobs every
  heartbeat cadence; the reaper reclaims a claim only once its **lease TTL** (a multiple of the
  cadence, e.g. 120s TTL vs 30s cadence) has expired. This eliminates the false-reaping of long
  synchronous steps — the exact failure mode CLAUDE.md already documents (the yfinance fetch and
  reference-index seed exceed the flat 300s timeout). Lease *renewal*, not a fixed claim-timeout,
  is what prevents reaping slow-but-alive work. Replaces the current synchronous-heartbeat
  workaround.
- **RECLAIM/re-queue orphans instead of FAIL (adopt, GATED).** Move from failing dead-worker
  jobs to re-running them (at-least-once, self-healing) — the queue-visibility-timeout model.
  **This is unsafe until §7.4 is satisfied**: re-running a non-idempotent job corrupts data.
  Gate: enable reclaim only after the idempotent-upsert audit is complete.

### 7.4 Idempotent writes — hard requirement (prerequisite for §7.3 reclaim)
Because the reaper (and any future queue) re-runs orphaned work, delivery is **at-least-once**,
so **every fetch job MUST write via idempotent upserts** (`INSERT ... ON CONFLICT DO UPDATE`
keyed on natural keys) — a re-run must be safe and convergent. **Action:** audit the existing
repositories/services and convert any plain-insert writes to upserts **before** enabling §7.3
reclaim. Until the audit is done, keep the reaper in FAIL mode.

### 7.5 Graceful shutdown — harden
Trap SIGTERM → stop claiming new jobs → drain/commit (or checkpoint) in-flight work → exit,
with a bounded drain + force-exit fallback so a stuck worker cannot hang forever. **Size the
container grace period well above the 30s default** (`terminationGracePeriodSeconds` ≈ preStop +
max job time + cleanup + buffer) — multi-minute fetch/seed steps are otherwise SIGKILLed
mid-write. A cleanly drained shutdown also avoids leaving orphaned rows for the reaper in the
first place. Applies to Docker restart-policy / systemd supervision too, not only Kubernetes.

### 7.6 Sync vs async — stay synchronous + thread workers
No async rewrite. The oft-cited "async is ~15x faster" is a **concurrency** effect (sequential
vs concurrent), not async-beating-threads; the daemon's thread pool (`YFINANCE_FETCH_WORKERS
1-16`) already captures the concurrency win and composes cleanly with the per-scraper
CircuitBreaker/RateLimiter for rate-limit compliance. The claims that would justify migrating —
"asyncio ~3.5x faster than threads", "asyncpg ~3.4x lower latency / ~1.9x throughput vs
SQLAlchemy" — were all **refuted 0-3** (§8). *Build-time revalidation: Python 3.13+
free-threading/no-GIL could shift the threads-vs-async tradeoff.*

---

## 8. Refuted ideas — do NOT build these

These were investigated and killed during research. Recorded so they are not revived later.

| Refuted idea | Verdict | What to do instead |
|--------------|---------|--------------------|
| Lean theorem-prover formal-verification gate on every trade | refuted 0-3 | Use a **classical rule-based** validation layer |
| Determinism and task accuracy are statistically uncorrelated | refuted 0-3 | Do not treat them as independent |
| "Only LangGraph + CrewAI connect to Ollama with no workaround" | refuted 0-3 | Verify framework↔Ollama connectivity ourselves |
| Parameter-count tool-use thresholds (e.g. 7B≈71%, 32B≈82%) | refuted 0-3 | Do not anchor model-size choice on these numbers |
| Smolagents best for local models (code vs JSON tool calls) | refuted 0-3 | Unsubstantiated; not a basis for framework choice |
| Hexagonal = onion = clean = screaming, fully interchangeable | refuted 1-2 | Same *family*, not identical — do not treat as one |
| Trading 212 has no official order API — Selenium UI automation only | refuted 0-2 | Official REST order API exists; build on it (§6), not Selenium |
| Multi-replica Paxos cron / APScheduler 4.x shared job store is the SOTA scaling answer | refuted 0-3 | Keep one daemon per DB; advisory-lock leader election if ever needed (§7.1–7.2) |
| asyncio ~3.5× faster than threads for I/O fetching | refuted 0-3 | Stay sync + threads; the win is concurrency, already captured (§7.6) |
| asyncpg ~3.4× lower latency / ~1.9× throughput vs SQLAlchemy | refuted 0-3 | Keep synchronous SQLAlchemy 2.0 (§7.6) |

---

## 9. Decisions taken & open questions

### Resolved (were open in the first pass)

- **Agent topology** — in-process blackboard nodes, not services, not A2A (§5.1, §5.2).
- **Agent communication** — one typed `FundState`, orchestrator as mediator, no cross-imports
  (§5.1).
- **Orchestrator** — hand-rolled over `FundState`; LangGraph adopted *only* as the durable
  checkpoint/interrupt substrate with `FundState` as the single channel (§6).
- **Guardrail** — Pydantic v2 validators, `ValueError`-based; JSON rule libs are data-only (§6).
- **LLM boundary** — BAML only; PydanticAI rejected (§6).
- **Broker** — official Trading 212 REST via `httpx`; demo/live = base-URL + key swap (§6).
- **Transport** — REST commands + FastAPI native SSE for streaming (§6).
- **Compliance posture** — advisory / paper-first; real execution gated off behind the
  guardrail + human approval gate (enforced as policy, not a broker limit).
- **Ingestion scheduler** — keep in-process APScheduler, one daemon per DB; advisory-lock leader
  election is the documented future scale path, not built (§7.1–7.2).
- **Ingestion reaper** — upgrade to lease-based, and switch orphan handling from FAIL to
  RECLAIM/re-queue — the reclaim switch **gated** on the idempotent-upsert audit (§7.3–7.4).
- **Ingestion I/O** — stay synchronous SQLAlchemy 2.0 + thread workers; no async rewrite (§7.6).

### Still open — need explicit decisions

0. **Idempotent-upsert audit (BLOCKS ingestion reclaim).** Before the reaper can RECLAIM/re-queue
   orphans (§7.3), audit every fetch job's writes and convert plain inserts to
   `INSERT ... ON CONFLICT DO UPDATE` on natural keys (§7.4). Until done, the reaper stays in FAIL
   mode. Research angle 5 (watermarking, schema-drift, incremental vs backfill for yfinance/FRED)
   was under-covered and warrants its own pass.

1. **Concrete regulatory / compliance requirements** (SEC RIA rules, MiFID II
   algorithmic-trading obligations, record-retention mandates). Research surfaced patterns but
   no verified specifics. Real money ⇒ obtain legal advice, not just architecture. This gates
   *un-gating* execution, not the advisory build.
2. **Trading 212 order idempotency & reconciliation (DEFERRED to un-gate time).** The order API
   is BETA and **non-idempotent** — a retried POST can place a duplicate real order. Documented
   requirement for the broker adapter before execution is un-gated: client-generated idempotency
   key per order intent, a dedup check before POST, and post-submit polling of order state to
   reconcile. No real-order code is written during the advisory phase, so the concrete design is
   deferred — but the requirement is recorded now because it interacts with the resume-idempotency
   gotcha in §6 (an interrupted execution node re-runs from its top).
3. **Reproducibility / determinism for audit (DEFERRED until a model is chosen).** Whether we set
   `temperature=0` + a fixed `seed` and record model + params + prompt hash per decision depends
   on the chosen Ollama model's seed/temperature support, which varies by model. Deferred to
   model-selection time. Until then: log model + params + full output per decision so audit works
   by inspection even without bit-reproducibility, and keep the validate/retry layer at the LLM
   port (Ollama guarantees grammar-valid tokens only at generation time).
4. **Build-time revalidation.** Several §6 facts sit at/after the Jan-2026 cutoff (FastAPI native
   SSE, `sse-starlette` version, Trading 212 live-order rollout). Verify versions and import paths
   when implementation starts.

---

## 10. Evidence quality & caveats

- Architecture findings rest mainly on one authoritative secondary source (*Architecture
  Patterns with Python*, Percival & Gregory — "Cosmic Python") plus canonical corroboration
  (Robert C. Martin's *Clean Architecture*, Cockburn's hexagonal). Strong for stable,
  non-time-sensitive design principles.
- The **orchestrator decision** (hand-roll over `FundState`; adopt LangGraph only for
  checkpoint + interrupt) is a synthesis judgment built on verified facts about each library
  plus the fixed constraints — not a directly-voted claim. The genuine residual tension is
  whether LangGraph's battle-tested Postgres checkpointing is worth the coupling versus a
  hand-rolled append-only audit/checkpoint table; we chose LangGraph-as-substrate for
  reliability in a real-money-later system.
- The library facts (PydanticAI/Ollama, Pydantic validators, Trading 212 REST orders, FastAPI
  SSE) are each individually verified 3-0 against primary/official docs.
- The determinism / `pass^k` evidence comes from a single, non-peer-reviewed 2026 preprint
  with small benchmarks and one internally unstable correlation result. The `pass^k`
  sub-claim was a 2-1 split (reproducibility work is deferred anyway — §9).
- Several package facts sit at/after the Jan-2026 knowledge cutoff and were established by the
  research pass, not prior knowledge. **Revalidate versions, import paths, and the fast-moving
  agent-framework landscape before committing** (§9, item 4).

- The **ingestion-daemon decisions (§7)** rest on primary/authoritative sources (Google SRE
  book, ACM Queue, official APScheduler 3.x + PostgreSQL + Kubernetes docs) — high confidence.
  The lease-based-reaper detail rests on a single project changelog (haiku.rag) corroborating an
  already-standard pattern. The sync-vs-async recommendation is medium confidence (weaker
  scraping-blog sources) but is reinforced by the migration-justifying claims being refuted 0-3.
- Ingestion research angle 5 (idempotent upserts, watermarking, schema-drift, outbox) was
  under-covered by verified evidence — treat those principles as sound but unverified; warrants
  its own pass (§9, item 0).

_Synthesized from three fan-out research passes: pass 1 (architecture) — 5 angles, 25 sources,
118 claims, 25 verified (16 confirmed / 9 refuted); pass 2 (libraries) — 5 angles, 23 sources,
111 claims, 25 verified (24 confirmed / 1 refuted); pass 3 (ingestion daemon) — 5 angles,
sources across primary docs, 25 verified (18 confirmed / 7 refuted)._
