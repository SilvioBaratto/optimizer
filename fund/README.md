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

## Eligible universe

`PortfolioMandate.eligible_tickers` is a hard per-portfolio allowlist; empty means
unrestricted. `fund.tools.eligibility` resolves it once per run (on run, resume and the
adviser edit, from the current mandate) into `Eligibility(allowed, barred)` on
`RunContext.eligible`, and the weighting tools enforce it: `list_universe` returns only
the allowed set, `universe_filter` / `estimate_moments` / `optimize_portfolio` refuse an
outside ticker, `risk_check` records a `not_eligible` violation and `place_orders` and
the adviser edit refuse it. `get_prices` and `get_regime_stats` stay unrestricted. The
benchmark is not automatically eligible.

Set it with `fund mandate set <portfolio_id> --eligible T [--eligible T ...]`
(`--clear-eligible` to lift it) or the Mandate view's "eligible tickers" field. Saving
goes through `apply_action`, which warns once about tickers without a priced instrument
and syncs the portfolio's `mandate:<id>` pins in `pinned_instruments`, so the weekly
universe build keeps the instruments. Add an instrument the Screener never returns with
`portopt add-instrument` (see the ingestion README).

A name is barred with one reason: `no_instrument`, `delisted`, `not_priced`,
`not_t212_tradeable`, `esg:<gics sector>` or `esg_unclassified`. Under ESG exclusions a stock is judged by its
stored sector and an ETF by look-through of its latest `etf_sector_weights` snapshot
against the client's `look_through_tolerance` (strict 0 by default): commodity and
government / inflation-linked bond ETFs pass, any other fund needs a snapshot no older
than `FUND_ESG_SECTOR_WEIGHTS_MAX_AGE_DAYS` (180) that sums to one within 0.05, or it is
barred `esg_unclassified`. Most corporate and aggregate bond ETFs stay barred, because
their sector data is sparse.

### Broker scope (`tradeable_only`)

`PortfolioMandate.tradeable_only` (default off, stored in the mandate JSON, so no
migration and existing mandates are unchanged) scopes a portfolio to what the Trading
212 catalogue carries, for runs whose orders must be executable at the broker. Set it
with `fund mandate set <portfolio_id> --tradeable-only` (`--no-tradeable-only` lifts it)
or the Mandate view's "Tradeable on Trading 212 only" box; `fund mandate show` prints it.

It composes with the allowlist, so both must hold, and the same `Eligibility` enforces
it in every tool above. The resolver reads `instruments.is_t212_tradeable` through
`UniverseRepository.get_t212_untradeable_tickers`, a tri-state:

- an explicit False (matched against the catalogue and absent) is barred
  `not_t212_tradeable`;
- NULL on a row (never evaluated) is not barred;
- when no instrument has been evaluated at all (Trading 212 unconfigured, or the
  annotation step never ran) nothing is restricted, and `Eligibility.notes` says so in
  the PM seed and the mandate-save warning instead of emptying the universe;
- when everything evaluated is False the universe is honestly empty, each name with its
  reason.

With no allowlist the scope still engages as a refusal list over the whole universe
(`Eligibility.has_allowlist` is False): `list_universe` keeps its market-cap paging but
drops the explicit False rows, the weighting tools refuse them, and ETF ESG screening
is unchanged. Research and benchmarking leave the flag off and keep the full
multi-region universe.

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
