# portopt-fund

The deep-agent investment-fund bridge. Dist **`portopt-fund`**, import package
**`fund`** (src-layout). Fourth member of the `optimizer` uv workspace, alongside
`portopt-core`, `portopt`, and `portopt-db`.

**Load-bearing principle: the LLM chooses, the optimizer computes.** Agents emit
only structured inputs and enums (constraint sets, view sets, allocation
decisions) — never portfolio weights. Weights come from `optimizer` (skfolio),
run deterministically inside `@tool` functions.

Built on [`deepagents`](https://github.com/langchain-ai/deepagents)
(LangChain/LangGraph): a PM orchestrator delegates to four subagents (economist,
allocator, risk-controller, executor) after a MiFID II profiling step. Durable
state (checkpointer + store) lives in a dedicated `langgraph` Postgres schema;
Alembic stays the sole owner of `public`.

## Boundary (guarded, load-bearing)

- `fund` **may** import `optimizer` + `portopt_db` — it is the bridge.
- `fund` **must not** import `app` (the ingestion daemon) —
  `tests/unit/hygiene/test_no_ingestion_import.py`.
- `ingestion` / `portopt-db` **must not** import `fund` / `deepagents` —
  held by the existing optimizer-import guards on those packages.
- `deepagents` / `langgraph` live **only** here.

## Commands

```bash
# installs fund editable into the shared venv
uv sync --all-packages --all-extras

cd fund && uv run pytest tests/ -v

fund          # human CLI      -> fund.cli:app
fund-tui      # watch terminal -> fund.tui.app:main
fund-worker   # daemon         -> fund.worker:main
```

## TUI (`fund-tui`)

The supervisor's read-first oversight cockpit (Textual): a left nav rail over one
content pane. Launch fund-wide or focused on a portfolio:

```bash
fund-tui                 # fund-wide landing (Posture)
fund-tui <portfolio-id>  # focus one portfolio
```

**Eight views** (sidebar order): **Posture** (roster + fund strip + focused detail,
incl. the real NAV tile) · **Deliberation** (run replay + live stream) · **Decisions**
(cross-portfolio HITL inbox) · **Mandate** (run-policy + guardrails) · **Audit**
(filterable decision timeline, CSV export) · **Performance** (metrics strip + holdings
table + drill-down, chart-free) · **Ask** (NL Q&A) · **Daily brief** (per-day recap,
markdown export).

**Ask** classifies each turn `query | action`. A query is answered from `fund.observe`
reads summarized by a **lazily-built** model on a worker thread (no model ⇒ a status
line, never a crash). An action (decision verbs only — approve / reject / killswitch /
freeze / re-arm / run / steer) routes through the shared `apply_action` **confirm +
audit** path; an ambiguous target asks a disambiguation follow-up first — **never silent
execution, never on a guess**. `e` exports the conversation to markdown.

**Help** opens on `?` and `F1` (F1 fires even while the Ask input holds focus). Status is
always **dot + label**, never colour alone.

Every data surface reads through the agent-stack-free `fund.observe` seam; the model and
the `optimizer`/`skfolio` performance tracker are lazy-imported inside workers / method
bodies, so `import fund.tui.*` stays agent-stack-light (guarded in `test_tui.py`).

See [`../SPEC.md`](../SPEC.md) for the Phase 0 architecture contract (D1–D4) and
[`../todo/deep_agent.md`](../todo/deep_agent.md) for the full roadmap.
