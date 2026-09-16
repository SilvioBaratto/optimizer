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
# Workspace sync (installs fund editable into the shared venv)
uv sync --all-packages --all-extras

# Tests
cd fund && uv run pytest tests/ -v

# Entrypoints (modules filled in later phases)
fund          # human CLI      -> fund.cli:app
fund-tui      # watch terminal -> fund.tui.app:main
fund-worker   # D6 daemon      -> fund.worker:main
```

See [`../SPEC.md`](../SPEC.md) for the Phase 0 architecture contract (D1–D4) and
[`../todo/deep_agent.md`](../todo/deep_agent.md) for the full roadmap.
