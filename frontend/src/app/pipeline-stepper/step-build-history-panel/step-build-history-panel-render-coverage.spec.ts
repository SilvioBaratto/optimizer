/**
 * Render-coverage spec for StepBuildHistoryPanelComponent (issue #901).
 *
 * Drives every template branch at the DOM:
 *   - Ready  → [data-testid="run-step"]
 *   - Running → [data-testid="spinner"]
 *   - Completed + buildResult → [data-testid="proxy-badge"]
 *   - lastError → [data-testid="error-block"]
 *
 * Secondary branches:
 *   - @if (r.failed_dates > 0) — failed-warning badge present vs absent
 *   - [class] binding on proxy-badge — gain class when market_proxy_loaded=true,
 *     warning class when market_proxy_loaded=false
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { StepBuildHistoryPanelComponent } from './step-build-history-panel';
import { StepStatus, type BuildHistoryStepResult } from '../../core/models/pipeline-builder.model';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface Harness {
  fixture: ComponentFixture<StepBuildHistoryPanelComponent>;
  el: HTMLElement;
}

async function build(): Promise<Harness> {
  await configureTestBed({ imports: [StepBuildHistoryPanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(StepBuildHistoryPanelComponent);
  fixture.componentRef.setInput('stepStatus', StepStatus.Pending);
  fixture.detectChanges();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

// ---------------------------------------------------------------------------
// Minimal result fixtures
// ---------------------------------------------------------------------------

const RESULT_PROXY_LOADED_NO_FAILURES: BuildHistoryStepResult = {
  succeeded_dates: 12,
  total_dates: 12,
  failed_dates: 0,
  n_factors: 5,
  rebalance_freq: 21,
  market_proxy_loaded: true,
};

const RESULT_PROXY_MISSING_WITH_FAILURES: BuildHistoryStepResult = {
  succeeded_dates: 10,
  total_dates: 12,
  failed_dates: 2,
  n_factors: 5,
  rebalance_freq: 21,
  market_proxy_loaded: false,
};

// ---------------------------------------------------------------------------
// Data-driven base-arm loop
// ---------------------------------------------------------------------------

interface StatusCase {
  label: string;
  status: StepStatus;
  result?: BuildHistoryStepResult;
  lastError?: string;
  testid: string;
}

const baseCases: StatusCase[] = [
  { label: 'Ready shows run button',     status: StepStatus.Ready,     testid: 'run-step'   },
  { label: 'Running shows spinner',       status: StepStatus.Running,   testid: 'spinner'    },
  { label: 'Completed shows proxy-badge', status: StepStatus.Completed, result: RESULT_PROXY_LOADED_NO_FAILURES, testid: 'proxy-badge' },
  { label: 'lastError shows error-block', status: StepStatus.Error,     lastError: 'err',    testid: 'error-block' },
];

describe('StepBuildHistoryPanelComponent (render coverage)', () => {
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

  // ---- Secondary: @if (r.failed_dates > 0) ---------------------------------

  describe('failed_dates @if', () => {
    it('when failed_dates > 0, the failed-warning badge renders', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_PROXY_MISSING_WITH_FAILURES);
      fixture.detectChanges();

      expect(el.querySelector('[data-testid="failed-warning"]')).not.toBeNull();
    });

    it('when failed_dates is 0, the failed-warning badge is absent', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_PROXY_LOADED_NO_FAILURES);
      fixture.detectChanges();

      expect(el.querySelector('[data-testid="failed-warning"]')).toBeNull();
    });
  });

  // ---- Secondary: [class] binding on proxy-badge ---------------------------

  describe('proxy-badge [class] binding', () => {
    it('when market_proxy_loaded is true, the badge carries the gain class', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_PROXY_LOADED_NO_FAILURES);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="proxy-badge"]')!;
      expect(badge.className).toContain('text-gain');
    });

    it('when market_proxy_loaded is false, the badge carries the warning class', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_PROXY_MISSING_WITH_FAILURES);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="proxy-badge"]')!;
      expect(badge.className).toContain('text-warning');
    });
  });
});
