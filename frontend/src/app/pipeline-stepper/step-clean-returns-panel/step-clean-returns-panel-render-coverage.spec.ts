/**
 * Render-coverage spec for StepCleanReturnsPanelComponent (issue #901).
 *
 * Drives every template branch at the DOM:
 *   - Ready  → [data-testid="run-step"]
 *   - Running → [data-testid="spinner"]
 *   - Completed + cleanResult → [data-testid="preprocessing-badge"] rows
 *   - lastError → [data-testid="error-block"]
 *
 * Secondary branches:
 *   - @for (step of r.preprocessing_steps) — 2-element array yields 2 badges
 *   - Static preprocessor list with $last separator (always-rendered config card)
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { StepCleanReturnsPanelComponent } from './step-clean-returns-panel';
import { StepStatus, type CleanReturnsStepResult } from '../../core/models/pipeline-builder.model';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface Harness {
  fixture: ComponentFixture<StepCleanReturnsPanelComponent>;
  el: HTMLElement;
}

async function build(): Promise<Harness> {
  await configureTestBed({ imports: [StepCleanReturnsPanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(StepCleanReturnsPanelComponent);
  fixture.componentRef.setInput('stepStatus', StepStatus.Pending);
  fixture.detectChanges();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

// ---------------------------------------------------------------------------
// Minimal result fixture
// ---------------------------------------------------------------------------

const RESULT: CleanReturnsStepResult = {
  n_days: 252,
  n_tickers: 40,
  return_start: '2020-01-02',
  return_end: '2024-01-01',
  preprocessing_steps: ['DataValidator', 'OutlierTreater'],
};

// ---------------------------------------------------------------------------
// Data-driven base-arm loop
// ---------------------------------------------------------------------------

interface StatusCase {
  label: string;
  status: StepStatus;
  result?: CleanReturnsStepResult;
  lastError?: string;
  testid: string;
}

const baseCases: StatusCase[] = [
  { label: 'Ready shows run button',          status: StepStatus.Ready,     testid: 'run-step'           },
  { label: 'Running shows spinner',            status: StepStatus.Running,   testid: 'spinner'            },
  { label: 'Completed shows preprocessing row', status: StepStatus.Completed, result: RESULT, testid: 'preprocessing-badge' },
  { label: 'lastError shows error-block',      status: StepStatus.Error,     lastError: 'fail',           testid: 'error-block' },
];

describe('StepCleanReturnsPanelComponent (render coverage)', () => {
  baseCases.forEach((c) => {
    it(`when ${c.label}`, async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', c.status);
      if (c.result !== undefined) fixture.componentRef.setInput('result', c.result);
      if (c.lastError)            fixture.componentRef.setInput('lastError', c.lastError);
      fixture.detectChanges();

      expect(el.querySelector(`[data-testid="${c.testid}"]`)).not.toBeNull();
    });
  });

  // ---- Secondary: @for preprocessing_steps count ---------------------------

  describe('@for preprocessing_steps badges', () => {
    it('when preprocessing_steps has 2 entries, renders 2 badges', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT);
      fixture.detectChanges();

      const badges = el.querySelectorAll('[data-testid="preprocessing-badge"]');
      expect(badges.length).toBe(2);
    });
  });

  // ---- Secondary: static preprocessor config card always renders -----------

  describe('static preprocessor config card', () => {
    it('when any status, the fixed pipeline config-card is present', async () => {
      const { el } = await build();

      expect(el.querySelector('[data-testid="config-card"]')).not.toBeNull();
    });

    it('when any status, the static preprocessor list renders all 4 names', async () => {
      const { el } = await build();

      const items = el.querySelectorAll('ol li.font-mono');
      expect(items.length).toBe(4);
    });
  });
});
