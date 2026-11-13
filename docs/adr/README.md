# Architecture decision records

One file per decision that shaped the repository, named `NNNN-short-title.md` with
a number that never changes. `tests/test_adr_format.py` keeps the shape below and
checks that every record is listed here.

## Template

```markdown
# ADR-NNNN: Title

- Status: Proposed | Accepted | Superseded by NNNN
- Date: YYYY-MM-DD

## Context
What forced the decision: the situation, the constraint, the evidence.

## Decision
What is decided, in the present tense.

## Alternatives considered
What else was possible and why it was not chosen. Say so when the reason was not
recorded at the time.

## Consequences
What becomes easier, what becomes harder, what must now be true.
```

A decision is never edited to say something else. When it changes, write a new record
and mark the old one `Superseded by NNNN`.

## Records written after the fact

Records 0001 to 0006 were written on 2026-11-13 from commit messages, `CLAUDE.md`
and the code, because the decisions had only been recorded as gotchas. Each cites the
commit that carries the reasoning. Where a commit does not say why an alternative was
rejected, the record says that instead of inventing a reason.

## Index

| Record | Title | Status | Date |
|---|---|---|---|
| [0001](0001-strip-to-ingestion-pipeline.md) | Strip the repository to a headless ingestion pipeline | Accepted | 2026-07-13 |
| [0002](0002-four-package-uv-workspace.md) | One uv workspace of four packages with import boundaries | Accepted | 2026-08-27 |
| [0003](0003-delete-optimizer-pipeline.md) | Delete `optimizer.pipeline`; the library is composable primitives | Accepted | 2026-09-15 |
| [0004](0004-heartbeat-lease-reaper.md) | Reap orphaned jobs by heartbeat lease | Accepted | 2026-10-24 |
| [0005](0005-langgraph-postgres-schema.md) | Keep the LangGraph tables in a dedicated Postgres schema | Accepted | 2026-09-16 |
| [0006](0006-llm-provider-registry.md) | A lazy provider registry over LangChain chat models | Accepted | 2026-09-18 |
| [0007](0007-branching-review-and-secrets-policy.md) | Branching, review and secrets policy for a single maintainer | Accepted | 2026-11-13 |
