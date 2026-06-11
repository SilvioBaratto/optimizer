/**
 * Criteria 1 & 2 acceptance tests — source-blind, authored from AC only.
 *
 * Criterion 1 (T3):
 *   "Optimization-studio: an optimization run completes and the results panel
 *    renders all sub-charts."
 *
 * Criterion 2 (T3):
 *   "Factor-research: the compute + validate round-trip completes and the IC
 *    chart renders."
 *
 * Both criteria require a multi-step flow:
 *   1. Trigger a run (POST to start the job).
 *   2. Poll until the job reaches a terminal state (GET /{jobId}).
 *   3. Assert the results panel / IC chart is present in the DOM.
 *
 * Tests use fail() to express the full contract, because the exact polling
 * mechanism (interval, tick count) cannot be determined without reading the
 * implementation. The fail() message is precise enough for the implementer to
 * write the real test body that replaces it.
 *
 * Selector contract:
 *   - Results panel: [data-testid="results-panel"] or [data-testid="optimization-results"]
 *   - Sub-charts: at least one <canvas> or <svg> descendant inside the results panel
 *   - IC chart: [data-testid="ic-chart"] or similar, containing <canvas> or <svg>
 */

import { ApplicationRef } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed } from '../';

// ---------------------------------------------------------------------------
// Criterion 1 — optimization-studio: run completes, ResultsPanelComponent renders
// ---------------------------------------------------------------------------

describe(
  'Criterion 1 – OptimizationStudioComponent: after optimization run, results panel renders all sub-charts',
  () => {
    let http: HttpTestingController;
    let appRef: ApplicationRef;

    beforeEach(async () => {
      await configureTestBed({ withHttp: true });
      http = TestBed.inject(HttpTestingController);
      appRef = TestBed.inject(ApplicationRef);
    });

    afterEach(() => http.verify());

    it(
      'when POST /optimize succeeds and GET /optimize/{runId} returns a completed result, the results panel is present with at least one chart element',
      () => {
        fail(
          'Implement:\n' +
            '  1. import OptimizationStudioComponent from app/optimization-studio/optimization-studio\n' +
            '  2. createComponent + detectChanges\n' +
            '  3. Trigger the optimize action (click or call the run method)\n' +
            '  4. Flush POST /optimize with { runId: "run-1", status: "running" }\n' +
            '  5. Flush GET /optimize/run-1 with a completed result payload:\n' +
            '     { runId: "run-1", status: "completed", weights: { AAPL: 0.5, MSFT: 0.5 }, metrics: {...} }\n' +
            '  6. appRef.tick(); fixture.detectChanges()\n' +
            '  7. Assert fixture.debugElement.query(\n' +
            '       By.css(\'[data-testid="results-panel"], [data-testid="optimization-results"]\')\n' +
            '     ) is not null\n' +
            '  8. Assert at least one canvas or svg element is inside the results panel\n' +
            '     (the sub-charts requirement)',
        );
      },
    );

    it(
      'when GET /optimize/{runId} returns status "failed", the results panel is NOT rendered and a non-blank error state IS rendered',
      () => {
        fail(
          'Implement:\n' +
            '  1. Same setup as the success test above\n' +
            '  2. Flush POST /optimize with { runId: "run-1", status: "running" }\n' +
            '  3. Flush GET /optimize/run-1 with { runId: "run-1", status: "failed", error: "Optimization failed" }\n' +
            '  4. appRef.tick(); fixture.detectChanges()\n' +
            '  5. Assert no results panel is in the DOM\n' +
            '  6. Assert a [role="alert"] element with non-blank text IS in the DOM',
        );
      },
    );
  },
);

// ---------------------------------------------------------------------------
// Criterion 2 — factor-research: compute + validate round-trip, IC chart renders
// ---------------------------------------------------------------------------

describe(
  'Criterion 2 – FactorResearchComponent: compute + validate round-trip completes and IC chart renders',
  () => {
    let http: HttpTestingController;
    let appRef: ApplicationRef;

    beforeEach(async () => {
      await configureTestBed({ withHttp: true });
      http = TestBed.inject(HttpTestingController);
      appRef = TestBed.inject(ApplicationRef);
    });

    afterEach(() => http.verify());

    it(
      'when POST /factors/compute succeeds and GET /factors/compute/{id} returns completed, the IC chart is rendered',
      () => {
        fail(
          'Implement:\n' +
            '  1. import FactorResearchComponent from app/factor-research/factor-research\n' +
            '  2. createComponent + detectChanges\n' +
            '  3. Trigger the compute action\n' +
            '  4. Flush POST /factors/compute with { id: "job-1", status: "running" }\n' +
            '  5. Flush GET /factors/compute/job-1 with { id: "job-1", status: "completed", factors: {...} }\n' +
            '  6. Trigger POST /factors/validate with the computed factors\n' +
            '  7. Flush POST /factors/validate with IC data: { ic_mean: 0.05, ic_std: 0.1, ... }\n' +
            '  8. appRef.tick(); fixture.detectChanges()\n' +
            '  9. Assert fixture.debugElement.query(\n' +
            '       By.css(\'[data-testid="ic-chart"], app-echarts-bar, app-echarts-line\')\n' +
            '     ) is not null\n' +
            ' 10. Assert the chart element or its container is visible (not hidden)',
        );
      },
    );

    it(
      'when POST /factors/validate returns HTTP 500 after a successful compute, the IC chart is not rendered and a non-blank error state is shown',
      () => {
        fail(
          'Implement:\n' +
            '  1. Same setup; complete the compute phase successfully\n' +
            '  2. Flush POST /factors/validate with status 500\n' +
            '  3. appRef.tick(); fixture.detectChanges()\n' +
            '  4. Assert no IC chart element is in the DOM\n' +
            '  5. Assert a [role="alert"] element with non-blank text IS in the DOM',
        );
      },
    );

    it(
      'when POST /factors/compute returns HTTP 500, the validation step is never triggered and a non-blank error state is shown',
      () => {
        fail(
          'Implement:\n' +
            '  1. Trigger compute action; flush POST /factors/compute with status 500\n' +
            '  2. appRef.tick(); fixture.detectChanges()\n' +
            '  3. Assert POST /factors/validate was NEVER called (http.expectNone(...))\n' +
            '  4. Assert a [role="alert"] element with non-blank text IS in the DOM',
        );
      },
    );
  },
);
