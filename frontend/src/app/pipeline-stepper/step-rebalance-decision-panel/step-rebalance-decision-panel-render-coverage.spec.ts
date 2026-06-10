/**
 * Render-coverage spec for StepRebalanceDecisionPanelComponent (issue #902).
 *
 * Drives every template branch at the DOM:
 *   - Ready      → [data-testid="run-step"]
 *   - Running    → [data-testid="spinner"]
 *   - Completed  → [data-testid="decision-badge"]
 *   - lastError  → [data-testid="error-block"]
 *
 * Secondary branches:
 *   - [class] on decision-badge: text-gain (decision=true) vs text-text-tertiary (false)
 *   - [class] on weight-count-badge: text-warning (warning=true) vs text-gain (false)
 *   - @if (r.missing_sectors.length > 0) → missing-sector-badge present vs absent
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { StepRebalanceDecisionPanelComponent } from './step-rebalance-decision-panel';
import {
  StepStatus,
  type RebalanceDecisionStepResult,
} from '../../core/models/pipeline-builder.model';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface Harness {
  fixture: ComponentFixture<StepRebalanceDecisionPanelComponent>;
  el: HTMLElement;
}

async function build(): Promise<Harness> {
  await configureTestBed({ imports: [StepRebalanceDecisionPanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(StepRebalanceDecisionPanelComponent);
  fixture.componentRef.setInput('stepStatus', StepStatus.Pending);
  fixture.detectChanges();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

// ---------------------------------------------------------------------------
// Minimal result fixtures
// ---------------------------------------------------------------------------

const RESULT_REBALANCE: RebalanceDecisionStepResult = {
  decision: true,
  reason: 'Threshold exceeded',
  n_weights: 20,
  weight_count_warning: false,
  missing_sectors: ['Energy', 'Utilities'],
  current_date: '2024-03-01',
  last_review_date: '2023-12-01',
};

const RESULT_HOLD: RebalanceDecisionStepResult = {
  decision: false,
  reason: 'Within threshold',
  n_weights: 18,
  weight_count_warning: true,
  missing_sectors: [],
  current_date: '2024-03-01',
  last_review_date: '2023-12-01',
};

// ---------------------------------------------------------------------------
// Data-driven base-arm loop
// ---------------------------------------------------------------------------

interface StatusCase {
  label: string;
  status: StepStatus;
  result?: RebalanceDecisionStepResult;
  lastError?: string;
  testid: string;
}

const baseCases: StatusCase[] = [
  { label: 'Ready shows run button',       status: StepStatus.Ready,                                    testid: 'run-step'      },
  { label: 'Running shows spinner',         status: StepStatus.Running,                                  testid: 'spinner'       },
  { label: 'Completed shows decision-badge',status: StepStatus.Completed, result: RESULT_HOLD,           testid: 'decision-badge'},
  { label: 'lastError shows error-block',   status: StepStatus.Error,     lastError: 'rebalance error',  testid: 'error-block'   },
];

describe('StepRebalanceDecisionPanelComponent (render coverage)', () => {

  // ---- Base status arms -------------------------------------------------------

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

  // ---- decision-badge [class] binding ----------------------------------------

  describe('decision-badge [class] binding', () => {
    it('when decision is true, the badge carries text-gain', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_REBALANCE);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="decision-badge"]')!;
      expect(badge.className).toContain('text-gain');
    });

    it('when decision is false, the badge carries text-text-tertiary', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_HOLD);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="decision-badge"]')!;
      expect(badge.className).toContain('text-text-tertiary');
    });
  });

  // ---- weight-count-badge [class] binding ------------------------------------

  describe('weight-count-badge [class] binding', () => {
    it('when weight_count_warning is true, the badge carries text-warning', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_HOLD);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="weight-count-badge"]')!;
      expect(badge.className).toContain('text-warning');
    });

    it('when weight_count_warning is false, the badge carries text-gain', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_REBALANCE);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="weight-count-badge"]')!;
      expect(badge.className).toContain('text-gain');
    });
  });

  // ---- missing-sectors @if branch --------------------------------------------

  describe('missing_sectors @if branch', () => {
    it('when missing_sectors is non-empty, missing-sector-badge elements render', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_REBALANCE);
      fixture.detectChanges();

      const badges = el.querySelectorAll('[data-testid="missing-sector-badge"]');
      expect(badges.length).toBe(RESULT_REBALANCE.missing_sectors.length);
    });

    it('when missing_sectors is empty, no missing-sector-badge renders', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_HOLD);
      fixture.detectChanges();

      expect(el.querySelector('[data-testid="missing-sector-badge"]')).toBeNull();
    });
  });
});
