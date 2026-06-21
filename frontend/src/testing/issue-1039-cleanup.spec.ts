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
 *  C3 – retained pipeline-stepper sub-modules (step-params.model, chip-summary, step-summary)
 *       still import correctly; a compile failure here means the file was accidentally deleted.
 *  C4 – retained portfolio-builder page files (inputs-pane, builder.store, stage-strip) import
 *       correctly, which proves their ../../pipeline-stepper/… import chains are intact.
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

// ── Retained pipeline-stepper sub-modules (C3) ─────────────────────────────────────────────
// pipeline-stepper lives at app/pipeline-stepper/ (not inside portfolio-builder/).
// If any of these files is accidentally deleted the TypeScript compilation fails,
// causing the entire test suite to fail — that is the intended contract.
import * as StepParamsModel from '../app/pipeline-stepper/models/step-params.model';
import * as ChipSummary from '../app/pipeline-stepper/chip-summary';
import * as StepSummary from '../app/pipeline-stepper/step-summary';

// ── Retained portfolio-builder page files (C4) ──────────────────────────────────────────────
// Each file transitively imports from ../../pipeline-stepper/…, so a compile error here
// proves a broken import chain, not just a missing file.
// inputs-pane and stage-strip live inside same-name subdirs; builder.store is in state/.
import * as InputsPane from '../app/portfolio-builder/inputs-pane/inputs-pane';
import * as BuilderStore from '../app/portfolio-builder/state/builder.store';
import * as StageStrip from '../app/portfolio-builder/stage-strip/stage-strip';

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
    if ((r as any).loadChildren) sources.push(((r as any).loadChildren).toString());
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

  // C3 ── retained pipeline-stepper sub-modules remain importable ───────────────────────────

  describe('C3: retained pipeline-stepper modules are importable', () => {
    it('when step-params.model is imported, the module namespace is defined', () => {
      expect(StepParamsModel).toBeDefined();
    });

    it('when chip-summary is imported, the module namespace is defined', () => {
      expect(ChipSummary).toBeDefined();
    });

    it('when step-summary is imported, the module namespace is defined', () => {
      expect(StepSummary).toBeDefined();
    });
  });

  // C4 ── retained portfolio-builder page files compile with pipeline-stepper imports intact ─

  describe('C4: retained portfolio-builder page files are importable with their import chains', () => {
    it('when inputs-pane is imported, the module namespace is defined', () => {
      expect(InputsPane).toBeDefined();
    });

    it('when builder.store is imported, the module namespace is defined', () => {
      expect(BuilderStore).toBeDefined();
    });

    it('when stage-strip is imported, the module namespace is defined', () => {
      expect(StageStrip).toBeDefined();
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
