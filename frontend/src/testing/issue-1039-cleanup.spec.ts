/**
 * Contract tests for Issue #1039:
 * chore(cleanup): delete ai-control-room page and legacy builder page shell.
 *
 * Source-blind — derived from acceptance criteria only. No implementation file was
 * read when authoring these tests.
 *
 * Criteria verified:
 *  C1 – 'ai-control-room/' directory removed: no app-route path equals 'ai-control-room'.
 *  C2 – pipeline-stepper shell removed: the 'portfolio-builder-legacy' route is absent.
 *  C3 – pipeline-stepper sub-modules fully deleted in issue #1053 (no longer tested here).
 *  C4 – removed in issue #1049: inputs-pane, builder.store, and stage-strip were
 *       subsequently deleted as builder-local orphans and are no longer tested here.
 *  C5 – no lazy-load expression in the route config contains 'ai-control-room' or
 *       'PipelineStepperComponent'.
 *
 * Not tested (per oracle report):
 *  – "All tests pass" — boilerplate suite gate; no per-criterion assertion possible.
 *  – "SOLID, clean code, TDD" — subjective prose; no concrete runtime assertion possible.
 */

import { Routes } from '@angular/router';

// ── Route config (C1, C2, C5) ──────────────────────────────────────────────────────────────
// Assumed export name: 'routes' (Angular CLI default for app.routes.ts).
import { routes } from '../app/app.routes';

// ── Helpers ────────────────────────────────────────────────────────────────────────────────

function collectPaths(routeList: Routes): string[] {
  const paths: string[] = [];
  for (const r of routeList) {
    if (r.path !== undefined) paths.push(r.path);
    if (r.children?.length) paths.push(...collectPaths(r.children));
  }
  return paths;
}

/** Serialise loadComponent/loadChildren thunk bodies so we can grep them. */
function collectLoadSources(routeList: Routes): string[] {
  const sources: string[] = [];
  for (const r of routeList) {
    if (r.loadComponent) sources.push(r.loadComponent.toString());
    if ((r as { loadChildren?: () => unknown }).loadChildren) {
      sources.push(((r as { loadChildren: () => unknown }).loadChildren).toString());
    }
    if (r.children?.length) sources.push(...collectLoadSources(r.children));
  }
  return sources;
}

// ── Tests ──────────────────────────────────────────────────────────────────────────────────

describe('Issue #1039 — ai-control-room and pipeline-stepper shell cleanup', () => {
  let allPaths: string[];
  let loadSources: string[];

  beforeEach(() => {
    allPaths = collectPaths(routes);
    loadSources = collectLoadSources(routes);
  });

  // C1 ── ai-control-room directory removed ─────────────────────────────────────────────────

  describe('C1: ai-control-room/ directory is removed', () => {
    it('when app routes are inspected, no route path equals "ai-control-room"', () => {
      expect(allPaths).not.toContain('ai-control-room');
    });
  });

  // C2 ── pipeline-stepper shell / legacy builder route removed ─────────────────────────────

  describe('C2: pipeline-stepper shell route is removed', () => {
    it('when app routes are inspected, no route path equals "portfolio-builder-legacy"', () => {
      expect(allPaths).not.toContain('portfolio-builder-legacy');
    });
  });

  // C5 ── no lazy-load expression references ai-control-room or PipelineStepperComponent ─────

  describe('C5: no route lazy-load expression references deleted identifiers', () => {
    it('when route lazy-load expressions are inspected, none references "ai-control-room"', () => {
      const found = loadSources.some(s => s.includes('ai-control-room'));
      expect(found).withContext('a loadComponent/loadChildren thunk still imports from ai-control-room').toBeFalse();
    });

    it('when route lazy-load expressions are inspected, none references "PipelineStepperComponent"', () => {
      const found = loadSources.some(s => s.includes('PipelineStepperComponent'));
      expect(found).withContext('a loadComponent/loadChildren thunk still imports PipelineStepperComponent').toBeFalse();
    });
  });
});
