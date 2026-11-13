# ADR-0001: Strip the repository to a headless ingestion pipeline

- Status: Accepted
- Date: 2026-07-13

## Context

Until July 2026 the repository shipped the `optimizer` library, a FastAPI service that
both ingested market data and served the library over `/api/v1/`, an Angular frontend,
a docs site, examples, a research package and a CLI.

Commit `869b4532` (2026-07-11) removed the frontend, docs site, examples, research
package and CLI, "leaving two shipped artifacts: the `optimizer/` library and the
`api/` FastAPI backend". Commit `1928d9e7` (2026-07-13) then went further. Its message
gives the reason: "The api/ side served two masters: it ingested market data on a
schedule AND exposed the optimizer library over /api/v1/. Only the ingestion half is
wanted. Serving the library also forced the whole sklearn/skfolio/scipy stack into the
API image and coupled two things that have no reason to share a process."

The directory `api/` was renamed `ingestion/` on 2026-08-19 (`8f47b07e`).

## Decision

Ingestion is a headless daemon. `app/worker.py` runs APScheduler in-process and blocks
on SIGTERM; `app/cli.py` drives the same step functions by hand. There is no FastAPI,
no route and no uvicorn. The scheduler is composed of public per-step functions, so the
cron pipelines and the CLI share one job-slot, heartbeat and progress path instead of a
parallel code path that drifts.

Any remaining reference to a route, `TestClient`, `uvicorn`, `app.main` or `/api/v1/` is
dead and is deleted, not revived (`CLAUDE.md`).

## Alternatives considered

- **Keep serving the library over HTTP from the ingestion service.** Rejected in
  `1928d9e7`: it pulled the sklearn/skfolio/scipy stack into the ingestion image and
  tied two unrelated processes together.
- **Keep the Angular frontend.** Removed in `869b4532`. That commit records what was
  removed and why the removal had to reach the API's pipeline-builder code; it does not
  weigh the frontend against an alternative, so no further reason is recorded.

## Consequences

- There is no HTTP API, so the standard's endpoint and OpenAPI rules (2.5, 2.6) have
  nothing to apply to. The interfaces are the CLI, the `background_jobs` table and the
  fund TUI.
- Job state is read from the `background_jobs` table, not from an endpoint.
- Some HTTP-era leftovers (response schemas, admin repository code) can still be found
  in the packages and are removed as they are met.
