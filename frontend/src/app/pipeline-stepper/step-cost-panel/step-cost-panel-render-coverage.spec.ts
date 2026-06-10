/**
 * Render-coverage spec for StepCostPanelComponent (issue #901).
 *
 * Drives every template branch at the DOM:
 *   - Ready  → [data-testid="run-step"]
 *   - Running → [data-testid="spinner"]
 *   - Completed + costResult → [data-testid="exceeds-badge"]
 *   - lastError → [data-testid="error-block"]
 *
 * Secondary branches:
 *   - @if (costRows().length > 0) — app-data-table present vs absent
 *   - [class] binding on exceeds-badge — loss class when exceeds_assumed=true,
 *     gain class when exceeds_assumed=false
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { StepCostPanelComponent } from './step-cost-panel';
import {
  StepStatus,
  type CostStepResult,
  type CostCountryItem,
} from '../../core/models/pipeline-builder.model';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface Harness {
  fixture: ComponentFixture<StepCostPanelComponent>;
  el: HTMLElement;
}

async function build(): Promise<Harness> {
  await configureTestBed({ imports: [StepCostPanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(StepCostPanelComponent);
  fixture.componentRef.setInput('stepStatus', StepStatus.Pending);
  fixture.detectChanges();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

// ---------------------------------------------------------------------------
// Minimal result fixtures
// ---------------------------------------------------------------------------

const COUNTRY_ROW: CostCountryItem = {
  country: 'US',
  stamp: 0,
  spread: 5,
  fx: 2,
  total: 7,
};

const RESULT_EXCEEDS: CostStepResult = {
  cost_bps_actual: 12,
  cost_bps_assumed: 10,
  exceeds_assumed: true,
  per_country: [COUNTRY_ROW],
};

const RESULT_OK: CostStepResult = {
  cost_bps_actual: 8,
  cost_bps_assumed: 10,
  exceeds_assumed: false,
  per_country: [],
};

// ---------------------------------------------------------------------------
// Data-driven base-arm loop
// ---------------------------------------------------------------------------

interface StatusCase {
  label: string;
  status: StepStatus;
  result?: CostStepResult;
  lastError?: string;
  testid: string;
}

const baseCases: StatusCase[] = [
  { label: 'Ready shows run button',       status: StepStatus.Ready,     testid: 'run-step'     },
  { label: 'Running shows spinner',         status: StepStatus.Running,   testid: 'spinner'      },
  { label: 'Completed shows exceeds-badge', status: StepStatus.Completed, result: RESULT_OK,     testid: 'exceeds-badge' },
  { label: 'lastError shows error-block',   status: StepStatus.Error,     lastError: 'err',      testid: 'error-block'  },
];

describe('StepCostPanelComponent (render coverage)', () => {
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

  // ---- Secondary: @if (costRows().length > 0) ------------------------------

  describe('per-country table @if', () => {
    it('when per_country has rows, app-data-table renders', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_EXCEEDS);
      fixture.detectChanges();

      expect(el.querySelector('app-data-table')).not.toBeNull();
    });

    it('when per_country is empty, app-data-table is absent', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_OK);
      fixture.detectChanges();

      expect(el.querySelector('app-data-table')).toBeNull();
    });
  });

  // ---- Secondary: [class] binding on exceeds-badge -------------------------

  describe('exceeds-badge [class] binding', () => {
    it('when exceeds_assumed is true, the badge carries the loss class', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_EXCEEDS);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="exceeds-badge"]')!;
      expect(badge.className).toContain('text-loss');
    });

    it('when exceeds_assumed is false, the badge carries the gain class', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_OK);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="exceeds-badge"]')!;
      expect(badge.className).toContain('text-gain');
    });
  });
});
