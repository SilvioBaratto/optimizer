# SPEC-RECONCILIATION — Phase 0, issue [0.1]

> **Path note (2026-11-13).** This record was written while the architecture
> document sat in the git-ignored `todo` folder. When the document moved to
> `docs/`, every path below, including the `Verify:` commands, was rewritten to its
> new location so the commands run again from the repository root.

Rulings on the seven discrepancies `docs/ARCHITECTURE.md` contains against itself and against
the working tree, plus the location of the generation source.

**All rulings have been applied and the document re-rendered.** Verification commands and
counts are given per section.

---

## Where the generation source lives — RECOVERED

`ARCHITECTURE.md` says its generated sections are produced by `scriptorium arch` from
`data/architecture-spec/`. Neither is in this repository — there is no `data/` at the repository
root and no `scriptorium` on `PATH`. **Both exist in the sibling repository
`/Volumes/External SSD/optimizer-theory`:**

| Artefact | Location |
|---|---|
| Spec source | `optimizer-theory/data/architecture-spec/` — 92 `items/*.json`, 5 context JSONs, `capability/` (25), `demoted.json`, `item-audit.json` |
| Generator | `optimizer-theory/src/scriptorium/`, console script `scriptorium = "scriptorium.cli.main:main"` |
| **Render target** | `optimizer-theory/docs/architecture/ARCHITECTURE.md` |

**Consequence: the 13 `<!-- GENERATO -->` markers are NOT demoted.** The fallback branch of the
acceptance criteria is conditional on the source being unrecoverable; it is recoverable, so the
markers stay and hand-edits below them remain genuinely at risk. Rulings were applied to the
spec source accordingly.

### Two facts that change how this document must be maintained

**1. `docs/ARCHITECTURE.md` is a copy, not the render target.** Before this pass, it was
byte-identical (same MD5, 3380 lines) to `optimizer-theory/docs/architecture/ARCHITECTURE.md`.
`discover_architecture_doc()` resolves the write target inside the vault and raises if it is
absent; `Roots.code_repo` — this repository — is documented as *"Read-only"*. **`scriptorium`
never writes into `optimizer/`.** The copy into `todo/` is a manual step. This applies to §2.1
too, even though §2.1 sits above every marker: hand-editing it here would be reverted by the
next copy-over.

The real pipeline:

```
optimizer-theory/data/architecture-spec/            JSON source
        │  scriptorium arch --assemble-only          (no LLM call)
        ▼
optimizer-theory/docs/architecture/ARCHITECTURE.md   render target
        │  manual copy
        ▼
optimizer/docs/ARCHITECTURE.md                       copy
```

**2. `todo/` is in this repository's `.gitignore`** (line 155). `ARCHITECTURE.md`, `PLAN.md`,
`TASKS.md` and the three Phase 0 deliverables are untracked here. The tracked home of the
architecture document is the vault.

### How this pass was executed

Work was isolated in a git worktree so the 14 uncommitted files on `optimizer-theory`'s `main`
were never touched:

```
git worktree add ../ot-spec-fix -b spec/phase0-reconciliation
SCRIPTORIUM_VAULT=../ot-spec-fix SCRIPTORIUM_CODE_REPO=/Volumes/External\ SSD/optimizer \
  scriptorium arch --assemble-only
```

`--assemble-only` issues no LLM query, so re-rendering is deterministic and cheap.

**Result: 54 files changed, 11 renamed.** The arch test suite goes from 12 failures to 6 — and
those 6 are the pre-existing baseline (a missing `pytest-asyncio` plugin), identical on an
untouched checkout. **These changes introduce no new test failure.**

---

## (a) `domain/fund/state.py` vs `fund_state.py`

**RULING: `fund_state.py` wins.**

`ports/agent.py` and `ports/llm.py` declared a connection to `fund/app/domain/fund/state.py`,
while the §6.1 tree names the file `fund_state.py`. The rendered document carried 15 references:
13 said `fund_state.py` — including the tree, the file's own §6.2 row, and
`frontend/src/app/services/fund.service.ts` — and 2 said `state.py`.

In the spec source the imbalance is the same shape but larger: **8 occurrences of `state.py`
across 3 item files**, against `fund_state.py` everywhere else.

| Item file | occurrences fixed |
|---|---|
| `fund-app-agents-prompts.json` | 4 |
| `fund-app-ports-agent-py.json` | 2 |
| `fund-app-ports-llm-py.json` | 2 |

Applied: 8 substitutions in 3 files.
**Verify:** `grep -c 'domain/fund/state\.py' docs/ARCHITECTURE.md` → `0`

## (b) `propose_orders.py` cites three non-existent execution modules

**RULING: the §4.1 tree wins; all three citations are remapped. No new file is created.**

`services/optimization/propose_orders.py` declared connections to
`optimizer/execution/schedule_trade.py`, `_multi_asset.py` and `_cost_estimate.py`. None appear
in the `execution/` tree, which lists `__init__.py`, `_aim_portfolio.py`, `_config.py`,
`_cost.py`, `_kyle_impact.py`, `_power_law_schedule.py`, `_resilience.py`, `_trajectory.py`,
`_turnover.py`.

| Cited | Remapped to | Justification |
|---|---|---|
| `schedule_trade.py` | `optimizer/execution/__init__.py` | No public non-underscore module exists anywhere under `optimizer/`; the barrel module is the package's public surface. House convention across all 19 submodules. |
| `_multi_asset.py` | `_trajectory.py` | The §4.2 row for `_trajectory.py` already claims it: *"entrambi estesi al caso multi-titolo con impatto incrociato — il motore di scheduling portfolio-level del pacchetto."* |
| `_cost_estimate.py` | `_cost.py` | `_cost.py` is the cost aggregator: *"un'unica stima di costo atteso."* |

Every cited capability already has a home; nothing needed inventing. No duplicate connection was
created — `propose_orders.py` listed no real `execution/` module beforehand.

Applied: 5 substitutions across `fund-app-services-optimization.json` (4) and `fund-tests.json` (1).
**Verify:** `grep -Ec 'schedule_trade|_multi_asset|_cost_estimate' docs/ARCHITECTURE.md` → `0`

## (c) `fund/` file descriptions cite §5.3 for content that lives in §6.3

**RULING: all `§5.3` citations inside `fund` specs become `§6.3`. The three legitimate §5.3
references elsewhere are untouched.**

§6.3 has exactly four numbered rules. Rule 1 is *"gli strumenti sono guidati dal codice per
default"*; rule 4 is *"il fallimento è valida → riprova → fallisci in sicurezza."* The spec
citations read *"§5.3 regola 1"* and *"§5.3 punto 4"* — they match §6.3's rules exactly. §5.3 is
the ingestion orphan reaper and has no tool registry in it.

Every one of the 53 source occurrences was checked for context; all refer to the fixed tool
registry, the typed `ToolIntent`, or the validate → dispatch → retry → fail-safe loop.

Applied: 53 substitutions across 17 `fund-*.json` files.
**Verify:** §5.3 inside the `<!-- GENERATO: fund -->` block → `0`; document-wide `§5.3` → `3`,
all in §5's reaper discussion, deliberately kept.

## (d) The `api/` → `ingestion/` rename is landed in code but not in the spec

**RULING: path-shaped rename applied across every layer. The rename record is preserved; the
Trading 212 `/api/v0` URLs are preserved.**

The issue text describes this as "the §5.1 and §8.1 trees are stale". The measured footprint was
**893 occurrences plus 11 filenames** — the entire §5.2 file table, not two trees. The rename is
already landed in code (commit `8f47b07`), so the specification, not the working tree, was wrong.

### Six layers, three of which had to move atomically

| Layer | What | Count |
|---|---|---|
| 1 | `items/*.json` — `checklist_path` and every `files[].path` | 864 |
| 2 | `ingestion.json` (22), `top-level.json` (7) | 29 |
| 3 | **`src/scriptorium/pipelines/arch/checklist.py`** — 11 `FileChecklistItem` entries | 11 |
| 4 | `writer.py` `_CONTEXT_TITLES`, plus docstrings in `runner.py`, `schemas.py`, `sources.py` | 5 |
| 5 | `item-audit.json` | 11 |
| 6 | `tests/golden/arch_render.json`, `tests/golden/legacy_schemas.json`, `tests/pipelines/test_arch.py`, `tests/domain/test_markdown.py` | regenerated |

**Layer 3 is the authority.** `contexts_from_items()` joins `it.checklist_path` against
`FILE_CHECKLIST` and against the coarse per-context specs keyed by the same path. Renaming the
JSON without renaming `checklist.py` would silently break the join: items drop out of `covered`,
the fill loop substitutes directory-level coarse specs, and §5.2 collapses from 274 file rows to
about 11 folder rows. The reverse fails the same way. Layers 1–3 therefore moved in one pass,
with the goldens refreshed alongside.

The 11 `items/api-*.json` files were renamed to `items/ingestion-*.json`. Routing ignores
filenames — it reads `checklist_path` from the content — so this is legibility, not correctness.

### Deliberately preserved

- **Trading 212**: `POST /api/v0/equity/orders/{market,limit,stop,stop_limit}` and the
  `demo|live.trading212.com/api/v0` base URLs. A blind `s/api\//ingestion\//g` would have
  corrupted these. **Verify:** `grep -c '/api/v0' docs/ARCHITECTURE.md` → `5`
- **The rename record** in §2.1 (lines 83, 86), which explains why the daemon was called `api/`.
- **`frontend/src/api/`** in §7 — an Angular directory, unrelated to the daemon.

### Hand-edited prose (not generated, so not covered by the source rename)

| Location | Change |
|---|---|
| §2 context table | `demone \`api/\`` → `demone \`ingestion/\`` |
| §2.1 | `separato da \`api/\`` → `\`ingestion/\`` |
| §2.1 | `` `api/` e `fund/` condividono il database `` → `` `ingestion/` `` |
| §2.1 tree | `├── api/ → ingestion/` → `├── ingestion/` |
| §2.1 tree | `wrapper shell sulla CLI di api` → `di ingestion` |
| §8 | `` **`api/` e `fund/` condividono** `` → `` `ingestion/` `` |
| §9 | `in uso in \`api/baml_src/\`` → `\`ingestion/baml_src/\`` |

**One tense correction, in scope because 0.1 reconciles against the working tree.** §2.1 said
*"La rinomina è rimandata perché tocca Docker, CI, i wrapper di `scheduler/` e CLAUDE.md."* The
rename has happened. Rewritten to *"La rinomina è stata eseguita nel commit `8f47b07`, che ha
toccato Docker, CI, i wrapper di `scheduler/` e CLAUDE.md."* The record is preserved; only the
tense is corrected.

**Verify:** `grep -c 'api/app' docs/ARCHITECTURE.md` → `0`

## (e) Directories enumerated with no member files

**RULING: the directory is the unit of specification. `adapters/broker/trading212/` stays
enumerated as the deliberate reference implementation, and is labelled as such.**

The premise in the issue text is inverted by the evidence.

**Nine directories, not four**, carry a role annotation with no member files:
`adapters/broker/`, `adapters/checkpointer/`, `adapters/llm/`, `agents/prompts/`,
`domain/_shared/`, `http/routers/`, `baml_client/`, `baml_src/`, `tests/`.

**The document is already internally consistent about them.** The §6.1 tree has 75 annotated
entries; the §6.2 table has 75 rows; the role tally in the header —
`config 3 · CORE 40 · DRIVEN 23 · DRIVING 4 · PORTS 5` — matches exactly. All nine directories
have real §6.2 rows with descriptions and connections. The directory is therefore *already* a
first-class unit, nine times over.

**The anomaly is the opposite of the one described:** `adapters/broker/` is double-specified —
it holds its own §6.2 row *and* a full six-file enumeration under `trading212/`.

**`demoted.json` in the spec source settles this with an artefact rather than an inference.** It
records six entries with the reason they are folder-level:

> *"espansione file-per-file rifiutata dall'auditor: … nessuna fonte reale su disco o in
> ARCHITECTURE.md per giustificare i singoli file. Mantenuta la specifica a livello di cartella."*

`checkpointer/`, `http/routers/`, `baml_src/`, `tests/`, plus `http/app.py` and `http/sse.py`.
An auditor already rejected enumeration as ungrounded — the same constraint as *"prefer an
honest gap to an invented symbol"*, enforced upstream.

Two distinct classes should be distinguished when reading the tree:

| Class | Members |
|---|---|
| **Actively demoted** (enumeration attempted and rejected) | `checkpointer/`, `http/routers/`, `baml_src/`, `tests/` |
| **Never expanded** | `adapters/llm/`, `agents/prompts/`, `domain/_shared/`, `baml_client/`, `adapters/broker/` |

Enumerating the remaining eight would mean inventing roughly thirty unverified filenames and
would break the 75/75 tree-to-table invariant. No spec-source change was required for this
ruling.

## (f) Barrel-module file counts

**RULING: `cinque` → `sei` in `attribution/`. Plus (f-bis): `sette` → `otto` in `execution/`.**

`optimizer/attribution/__init__.py` said it re-exports *"cinque moduli sorgente"* and then listed
six: `_config`, `_euler`, `_sample`, `_diagnostics`, `_factor_impact`, `_expected_loss`. The list
is right; the number is wrong.

Running the same audit across every barrel module found one more instance and cleared two:

| Module | Claim | Actual | Verdict |
|---|---|---|---|
| `attribution/` | "cinque moduli" ×3 | 6 | ❌ corrected, 3 sites |
| `execution/` | "altri sette file" | 8 | ❌ **(f-bis)** corrected, 1 site |
| `regimes/` | "sei moduli + `_factory`" = 7 | 7 | ✅ correct, untouched |
| `fx/` | "cinque moduli" ×2 | 5 (`_config`, `_converter`, `_decomposition`, `_factory`, `_rates`) | ✅ correct, untouched |

(f-bis) is folded into this section rather than given one of its own, so the seven-section
structure holds. It is the identical bug, found by the check (f) itself demands.

Applied: 3 substitutions in `optimizer-attribution.json`, 1 in `optimizer-execution.json`.
**Verify:** `grep -c 'altri sette file' docs/ARCHITECTURE.md` → `0`;
`grep -c 'altri otto file'` → `1`; `grep -c 'sei moduli'` → `3`

## (g) §2.1 places `ARCHITECTURE.md` and `TODO.md` at the repository root

**RULING, two parts.**

### (g1) `TODO.md` is not restored; its three references are removed

`TODO.md` existed. It was created in `3a84f64 docs: translate TODO.md to English` and deleted
two commits later in `b36ca8f refactor: … drop planning docs`, which removed the root
`ARCHITECTURE.md` (656 lines) at the same time. `0727ea9` then added `docs/ARCHITECTURE.md` as a
3380-line rewrite; `TODO.md` never returned.

Its content was eleven lines: an end-to-end optimizer; a scheduler that is stable, failure-proof,
fast, scalable and lightweight; four AIs simulating an investment fund; a frontend; an API. **As
information it is fully redundant** — every point is covered in far more detail by §2.1, §5,
§5.1–§5.6, §6, §6.1, §6.4, §7 and §9. Its only unique value was as the request in the author's
own words, and it remains recoverable from git (`git show b36ca8f^:TODO.md`).

| Reference | Was | Now |
|---|---|---|
| line 4 | *"…piattaforma descritta in [`TODO.md`](./TODO.md)"* | *"…piattaforma di investimento quantitativo."* — no external pointer |
| §2.1 prose | *"(regola dura già in vigore, e TODO.md vuole lo scheduler leggero e a prova di guasto)"* | *"(regola dura già in vigore)"* — the constraint is carried by §2 and §5 on their own |
| §2.1 tree | listed among root files | removed |

**Verify:** `grep -c 'TODO\.md' docs/ARCHITECTURE.md` → `0`

### (g2) §2.1 gains a `todo/` entry

Per clarification 3, §2.1 is amended to place `ARCHITECTURE.md` under `todo/` rather than moving
files to the root. `todo/` becomes a tree entry alongside `tests/`, `scheduler/` and `scripts/`,
with its files named — they were absent from the tree entirely, despite being where the whole
document is read from. The root line now lists only what is genuinely at the root.

```
├── tests/                                  suite di test della libreria            §8
├── scheduler/                              wrapper shell sulla CLI di ingestion
├── scripts/                                helper di CI
├── todo/                                   documentazione di progetto
│       ARCHITECTURE.md   PLAN.md   TASKS.md
│       SPEC-RECONCILIATION.md   DECISIONS.md   VALIDATION.md
├── docker-compose.yml                      db · adminer · scheduler · fund · frontend
├── pyproject.toml                          configurazione di portopt (libreria)
├── CLAUDE.md   README.md
```

§2.1 sits above every `<!-- GENERATO -->` marker, so `--assemble-only` does not overwrite it —
confirmed by re-running the render after the edit. But per the finding above, it lives in the
**vault** copy, so these edits were made there and copied here.

---

## Findings outside (a)–(g)

Recorded, not acted on. None is a ruling; each is a discrepancy of the same class found while
executing the seven.

### 1. §10.4 is factually wrong about `risk_free_rate` — **should be corrected**

§10.4 states skfolio exposes no `risk_free_rate` parameter on `MeanRisk`, `RiskBudgeting`,
`Portfolio` or `RatioMeasure`. Measured against skfolio 0.20.1, all three classes accept it
(default `0.0`). The conclusion §10.4 draws about `SHARPE_RATIO` still holds, but the cause is
the zero default, not a missing API. This materially changes decision 0.2(d). See
`VALIDATION.md`.

### 2. §9's "LLM self-hosted via Ollama" premise no longer holds

Decision 0.2(e) selects Ollama cloud models. Fund prompts — positions, macro views, proposed
orders — will leave the machine for `ollama.com`. §9 needs amending to describe the system as
built. See `DECISIONS.md` (e).

### 3. §9's Trading 212 authentication description is incomplete

§9 describes demo/live as a swap of base URL and key. The working scheme is HTTP Basic over
`base64(api_key:api_secret)`; a bare key returns 401. See `VALIDATION.md` row 5b.

### 4. `/api/v1/optimize` and `/api/v1/backtest` are referenced in the ingestion test spec

`items/ingestion-tests.json`, in the description of `tests/fixtures/smoke_prices.sql`, refers to
endpoints deleted with the HTTP layer. `CLAUDE.md` names `/api/v1/` explicitly as a leftover to
delete rather than revive. Not the rename, so not covered by (d).

### 5. The §6.1 tree omits `__init__.py` from seven directories

`app/`, `adapters/`, `agents/`, `domain/`, `http/`, `ports/`, `tools/` have none, while
`domain/fund/`, `domain/portfolio/`, `domain/trading/`, `services/*/`, `adapters/persistence/`
and `adapters/portfolio/` all do — and the real `ingestion/` has one at every level
(`app/`, `services/`, `repositories/`, `models/`, `schemas/`). Not a blocker under PEP 420
namespace packages. **Assigned to Phase 6**, when `fund/` is scaffolded; adding seven rows now
would change the 75/75 invariant and the role tally, which is a spec change rather than a
reconciliation.

### 6. Line-number citations in the spec source have drifted

Several item specs cite `ARCHITECTURE.md` line ranges — e.g. *"§6.3 regola 1, righe 411-416"*.
Line 411 now falls inside the `optimizer/optimization/` tree. Line-number citations rot on every
re-render; they should be replaced by section anchors.

### 7. `sources.py` scans a path that no longer exists

`sources.py` enumerates real files under `api/app/services/` for grounding. That path was renamed
on disk in commit `8f47b07`, so **any full regeneration pass was already broken before this
work**. Fixed here as part of (d) layer 4, but it means the generated specs predate the rename
and no full pass has run since.

---

## Verification summary

Run against `docs/ARCHITECTURE.md` after the re-render:

| Check | Expected | Actual |
|---|---|---|
| `grep -c 'domain/fund/state\.py'` | 0 | 0 |
| `grep -Ec 'schedule_trade\|_multi_asset\|_cost_estimate'` | 0 | 0 |
| `§5.3` inside the `fund` GENERATO block | 0 | 0 |
| `grep -c 'api/app'` | 0 | 0 |
| `grep -c 'TODO\.md'` | 0 | 0 |
| `grep -c 'altri sette file'` | 0 | 0 |
| `grep -c '/api/v0'` (must survive) | 5 | 5 |
| `grep -c '§5.3'` (reaper, must survive) | 3 | 3 |
| arch test suite | baseline 6 failures | 6 failures, unchanged |
