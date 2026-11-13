# ADR-0002: One uv workspace of four packages with import boundaries

- Status: Accepted
- Date: 2026-08-27

## Context

With the HTTP layer gone (ADR-0001), the code fell into four parts with different
dependencies: the optimization library, the ingestion daemon, the database layer both
the daemon and the fund bridge need, and the LLM fund bridge.

The workspace was scaffolded in `9d043cde` (2026-08-27, `portopt-db`), the database
plumbing moved into `portopt_db` in `df8d48c5` (same day), the library got its
`src/` layout in `a7f16ce0` (2026-09-11) and the `fund` package was committed in
`2eee25e6` (2026-09-16). The scaffolding commit also raised `requires-python` to
`>=3.12` everywhere, because the pinned numpy and pandas already required it and "uv
resolves the whole range and surfaced it".

## Decision

One uv workspace with a single virtualenv and four members:

| Package | Import name | Role |
|---|---|---|
| `portopt-core` | `optimizer` | Pure optimization library, database-agnostic |
| `portopt-db` | `portopt_db` | `Base`, models, repositories, `DatabaseManager`, the one Alembic tree |
| `portopt` (`ingestion/`) | `app` | Headless ingestion daemon |
| `portopt-fund` (`fund/`) | `fund` | LLM bridge: the model chooses, the optimizer computes |

Boundaries: `ingestion` and `portopt-db` do not import `optimizer`; `optimizer` does
not import `portopt_db`; only `fund` may import both; `deepagents` and `langgraph` live
only in `fund`. Packages are installed through uv only, because siblings resolve
through `[tool.uv.sources]`, which pip ignores.

## Alternatives considered

- **Separate environments or images per package**, which would make the boundaries
  physical. The scaffolding commit does not say why one shared environment was chosen.
  What is documented is the cost: with a single venv there is no install isolation.
- **Let ingestion import the optimizer.** `1928d9e7` (ADR-0001) removed that coupling
  on purpose. The daemon still ships scikit-learn, but only because yfinance's price
  repair imports it, which makes it a data-layer dependency, not an optimization one.

## Consequences

- The static import-scan tests are the only enforcement of the boundaries:
  `test_no_optimizer_import.py` (ingestion and portopt-db), `test_no_portopt_db_import.py`
  (root), `test_no_ingestion_import.py` (fund) and the two `test_no_agent_stack_import.py`
  guards. A new member needs its own guard.
- Each member keeps its own coverage floor and its own test suite, run from the member
  directory.
- All four members are versioned together (`0.1.0` at the time of writing) while only
  `portopt-core` is published.
