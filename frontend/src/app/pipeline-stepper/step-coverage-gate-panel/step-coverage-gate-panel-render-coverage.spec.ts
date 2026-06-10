/**
 * Render-coverage spec for StepCoverageGatePanelComponent (issue #901).
 *
 * Drives every template branch at the DOM:
 *   - Ready  → [data-testid="run-step"]
 *   - Running → [data-testid="spinner"]
 *   - Completed + coverageResult → [data-testid="passing-badge"] rows
 *   - lastError → [data-testid="error-block"]
 *
 * Secondary branches:
 *   - @for over passing_factors — non-empty renders badges, empty renders none
 *   - @for over is_only_factors — non-empty renders is-only-badge rows
 *   - @for over oos_only_factors — non-empty renders oos-only-badge rows
 *   - abortGateError extra input — set renders its branch (lastError block)
 *
 * NOTE: The component comment states abortGateError is accepted but NOT rendered
 * inside the panel template (the stepper shows it above the outlet). Asserting
 * that the input is accepted without error is the correct coverage target.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { StepCoverageGatePanelComponent } from './step-coverage-gate-panel';
import { StepStatus, type CoverageGateStepResult } from '../../core/models/pipeline-builder.model';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface Harness {
  fixture: ComponentFixture<StepCoverageGatePanelComponent>;
  el: HTMLElement;
}

async function build(): Promise<Harness> {
  await configureTestBed({ imports: [StepCoverageGatePanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(StepCoverageGatePanelComponent);
  fixture.componentRef.setInput('stepStatus', StepStatus.Pending);
  fixture.detectChanges();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

// ---------------------------------------------------------------------------
// Minimal result fixtures
// ---------------------------------------------------------------------------

const RESULT_WITH_FACTORS: CoverageGateStepResult = {
  passing_factors: ['momentum', 'quality'],
  is_only_factors: ['value'],
  oos_only_factors: ['low_vol'],
  n_passing: 2,
  min_factors: 2,
};

const RESULT_EMPTY_FACTORS: CoverageGateStepResult = {
  passing_factors: [],
  is_only_factors: [],
  oos_only_factors: [],
  n_passing: 0,
  min_factors: 2,
};

// ---------------------------------------------------------------------------
// Data-driven base-arm loop
// ---------------------------------------------------------------------------

interface StatusCase {
  label: string;
  status: StepStatus;
  result?: CoverageGateStepResult;
  lastError?: string;
  testid: string;
}

const baseCases: StatusCase[] = [
  { label: 'Ready shows run button',       status: StepStatus.Ready,     testid: 'run-step'    },
  { label: 'Running shows spinner',         status: StepStatus.Running,   testid: 'spinner'     },
  { label: 'Completed shows passing-badge', status: StepStatus.Completed, result: RESULT_WITH_FACTORS, testid: 'passing-badge' },
  { label: 'lastError shows error-block',   status: StepStatus.Error,     lastError: 'gate',    testid: 'error-block' },
];

describe('StepCoverageGatePanelComponent (render coverage)', () => {
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

  // ---- Secondary: @for passing_factors rows --------------------------------

  describe('passing_factors @for', () => {
    it('when passing_factors has 2 entries, renders 2 passing-badge rows', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_WITH_FACTORS);
      fixture.detectChanges();

      expect(el.querySelectorAll('[data-testid="passing-badge"]').length).toBe(2);
    });

    it('when passing_factors is empty, no passing-badge renders', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_EMPTY_FACTORS);
      fixture.detectChanges();

      expect(el.querySelectorAll('[data-testid="passing-badge"]').length).toBe(0);
    });
  });

  // ---- Secondary: @for is_only_factors rows --------------------------------

  describe('is_only_factors @for', () => {
    it('when is_only_factors has 1 entry, renders 1 is-only-badge', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_WITH_FACTORS);
      fixture.detectChanges();

      expect(el.querySelectorAll('[data-testid="is-only-badge"]').length).toBe(1);
    });

    it('when is_only_factors is empty, no is-only-badge renders', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_EMPTY_FACTORS);
      fixture.detectChanges();

      expect(el.querySelectorAll('[data-testid="is-only-badge"]').length).toBe(0);
    });
  });

  // ---- Secondary: @for oos_only_factors rows --------------------------------

  describe('oos_only_factors @for', () => {
    it('when oos_only_factors has 1 entry, renders 1 oos-only-badge', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_WITH_FACTORS);
      fixture.detectChanges();

      expect(el.querySelectorAll('[data-testid="oos-only-badge"]').length).toBe(1);
    });

    it('when oos_only_factors is empty, no oos-only-badge renders', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_EMPTY_FACTORS);
      fixture.detectChanges();

      expect(el.querySelectorAll('[data-testid="oos-only-badge"]').length).toBe(0);
    });
  });

  // ---- Secondary: abortGateError extra input --------------------------------

  describe('abortGateError input', () => {
    it('when abortGateError is set, the component renders without error', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_WITH_FACTORS);
      fixture.componentRef.setInput('abortGateError', 'gate failure: not enough passing factors');
      fixture.detectChanges();

      // abortGateError is NOT rendered inside the panel (handled by the stepper).
      // Assert the component still renders the completed grid correctly.
      expect(el.querySelector('[data-testid="passing-badge"]')).not.toBeNull();
    });
  });
});
