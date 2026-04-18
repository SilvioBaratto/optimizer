# WCAG 2.1 AA Audit

This document is the living audit report for the Optimizer frontend. It captures
the tooling used, the baseline items already satisfied in code, and the
remediation backlog, split into scoped sub-tasks.

## Scope

WCAG 2.1 **AA** across every page under `src/app/pages/` (13 pages):

| Route                     | Page folder           |
|---------------------------|-----------------------|
| `/`                       | `dashboard`           |
| `/portfolio-builder`      | `portfolio-builder`   |
| `/portfolio-detail`       | `portfolio-detail`    |
| `/optimization-studio`    | `optimization-studio` |
| `/backtesting`            | `backtesting`         |
| `/risk-center`            | `risk-center`         |
| `/factor-research`        | `factor-research`     |
| `/rebalancing`            | `rebalancing`         |
| `/attribution`            | `attribution`         |
| `/macro-intelligence`     | `macro-intelligence`  |
| `/ai-control-room`        | `ai-control-room`     |
| `/instrument-detail`      | `instrument-detail`   |
| `/settings`               | `settings`            |

AAA features (e.g. 7:1 contrast, sign language on media) are explicitly **out of scope**.

## Tooling

Live audits are run via `@axe-core/cli` from `scripts/a11y.mjs`.

```sh
# One-time install
npm install --legacy-peer-deps

# Run the dev server (leave running)
npm start

# In another shell, run the audit
npm run a11y

# Optional: point at a different base URL (docker preview, staging, etc.)
A11Y_BASE_URL=http://localhost:8080 npm run a11y
```

The script writes `docs/a11y-audit-results.md` with one table per route listing
every **critical** or **serious** violation. It exits non-zero when at least one
blocking violation remains, which makes it suitable for CI.

Per-route results are regenerated with every run — that file is the current
state of the app. This document (`a11y-audit.md`) tracks the baseline, the
backlog, and the fixes that have landed centrally.

## Baseline already satisfied

The following requirements are already met in the shared layer and do not need
per-page remediation:

| Requirement                                            | Location                                                          |
|--------------------------------------------------------|-------------------------------------------------------------------|
| `role="dialog"` + `aria-modal="true"` on modals        | `shared/modal/modal-container.ts`                                 |
| `aria-labelledby` wiring on titled modals              | `shared/modal/modal-container.ts`                                 |
| Focus trap (shift-tab / tab wrapping)                  | `shared/modal/modal-container.ts` (`onKeydown`)                   |
| Focus restoration to trigger on modal close            | `shared/modal/modal-container.ts` (`restoreFocus`)                |
| Esc-key closes modals                                  | `shared/modal/modal-container.ts` (host binding)                  |
| Skip-to-content link (AA technique G1)                 | `shared/layout/layout.html` (top of `<div>`)                      |
| Global `:focus-visible` 2px outline, 4.5:1+ contrast   | `src/styles.css` (line 205)                                       |
| `role="dialog"` on side-flyout                         | `shared/instrument-detail-flyout/instrument-detail-flyout.html`   |
| Esc-key dismissal on side-flyout                       | `shared/instrument-detail-flyout/instrument-detail-flyout.ts`     |
| `role="dialog"` + `aria-modal` on global-search palette| `shared/global-search/global-search.html`                         |
| `aria-label` on icon-only buttons (close / menu)       | Layout header + flyout header + modal container + chart toolbar   |
| `aria-label` on search, date-range, global-search inputs| `search-input`, `date-range-picker`, `global-search`             |
| `role="status" aria-live="polite"` on toasts           | `shared/notification/toast-container.ts`                          |
| `aria-label` on toast / alert-banner dismiss           | `shared/notification/toast-container.ts`, `alert-banner.ts`       |
| `<label for=…>` + `<input id=…>` in export-report-modal| `shared/modal/export-report-modal.ts`                             |
| Global search keyboard shortcut                        | `shared/layout/layout.ts` (`Cmd/Ctrl+K`)                          |
| `Cmd/Ctrl+1..9` page-switcher                          | `shared/layout/layout.ts`                                         |

## Remediation backlog

The review on #415 identified that full remediation exceeds a single work
session and recommended splitting into the sub-tasks below. The audit tooling
and baseline above have already been delivered. The remaining tasks, in
priority order:

### #415b — Centralized a11y fixes (shared layer)

- [ ] **Tailwind v4 contrast tokens** — audit `text-text-tertiary`,
  `text-text-secondary`, and `border` against `bg-surface`/`bg-surface-raised`
  for AA contrast (4.5:1 normal, 3:1 large). If any pair fails, darken the
  token centrally in `src/styles.css` or the theme CSS module — **not**
  per-component utilities.
- [ ] **Focus-visible utility** — add a single global rule so every interactive
  element renders a 3:1-contrast ring on keyboard focus. Suggested implementation:
  ```css
  :focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; }
  ```
- [ ] **Shared dialog ARIA helpers** — extract `role=dialog`, `aria-modal`,
  and focus-trap into a reusable directive (`[appFocusTrap]` + `<app-dialog>`)
  so ad-hoc flyouts and drawers inherit the behaviour without re-implementing it.

### #415c — Create Portfolio modal polish

- [ ] Confirm `Create Portfolio` (`portfolio-builder/portfolio-create.ts`) uses
  the shared `ModalContainer`. If it mounts its own overlay, migrate it so it
  inherits focus trap + Esc + role wiring.
- [ ] Verify every form input inside `CreatePortfolio*` has a visible or
  `aria-label`-only label (no placeholder-only labels).

### #415d — Per-page remediation: dashboard, portfolio-builder, portfolio-detail

- [ ] Fix every `critical` and `serious` violation reported by the runner for
  these three routes.
- [ ] Verify logical Tab order (DOM order matches visual order); if ECharts
  widgets are focusable, give them `aria-label` summaries.

### #415e — Per-page remediation: optimization-studio, risk-center, factor-research, backtesting

- [ ] As above for these four analytics pages.
- [ ] Surface chart results in `aria-live=polite` regions where dynamic updates
  would otherwise be missed by screen readers.

### #415f — Per-page remediation: ai-control-room, attribution, rebalancing, settings, macro-intelligence, instrument-detail

- [ ] As above for these six pages.

### #415g — Regression verification

- [ ] Automated Karma spec or manual script confirming:
  - `/` focuses the global search input
  - `Cmd/Ctrl+K` toggles the global search palette
  - `Cmd/Ctrl+1..9` routes to each page index in `layout.ts:PAGE_ROUTES`
  - OnPush, signals, and native control flow are not regressed in any fixed
    component (`grep -r "ngClass\|ngStyle\|\\*ngIf\|\\*ngFor" src/app` returns
    zero matches).

## What this PR delivered

Shared-layer a11y improvements that propagate to every page using the
components:

- `scripts/a11y.mjs` — runner aggregating `@axe-core/cli` output into markdown.
- `package.json` — `"a11y"` script entry and `@axe-core/cli` dev dependency.
- `docs/a11y-audit.md` (this file) — baseline + backlog.
- `shared/instrument-detail-flyout/*` — Escape-key dismissal so the flyout
  matches the modal container's keyboard affordance (+ 2 specs).
- `shared/search-input/search-input.ts` — `type="search"` + `aria-label="Search"`
  for reusable search inputs across pages.
- `shared/date-range-picker/date-range-picker.ts` — `aria-label="Start date"`
  and `aria-label="End date"` on custom-range inputs; `aria-hidden="true"` on
  the decorative separator glyph.
- `shared/global-search/global-search.html` — palette now has `role="dialog"`
  + `aria-modal="true"` + `aria-label="Global search"`; input has
  `type="search"` + `aria-label="Search pages"` + decorative icon marked
  `aria-hidden="true"`.

Everything under _Remediation backlog_ is explicitly **not** in this PR. Each
bullet corresponds to a follow-up issue.
