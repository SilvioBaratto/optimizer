/**
 * Render-coverage spec for StepValidateIsPanelComponent (issue #902).
 *
 * Drives every template branch at the DOM:
 *   - Ready      → [data-testid="run-step"]
 *   - Running    → [data-testid="spinner"]
 *   - Completed  → [data-testid="vif-badge"] / [data-testid="vif-empty"]
 *   - lastError  → [data-testid="error-block"]
 *
 * Secondary branches:
 *   - config-card @if (isResult(); as r) → <dl> renders when result is set,
 *     placeholder <p> renders when result is null (independent of stepStatus)
 *   - @if (r.high_vif_factors.length > 0) → vif-badge present vs vif-empty
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { StepValidateIsPanelComponent } from './step-validate-is-panel';
import {
  StepStatus,
  type ValidateIsStepResult,
} from '../../core/models/pipeline-builder.model';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface Harness {
  fixture: ComponentFixture<StepValidateIsPanelComponent>;
  el: HTMLElement;
}

async function build(): Promise<Harness> {
  await configureTestBed({ imports: [StepValidateIsPanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(StepValidateIsPanelComponent);
  fixture.componentRef.setInput('stepStatus', StepStatus.Pending);
  fixture.detectChanges();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

// ---------------------------------------------------------------------------
// Minimal result fixtures
// ---------------------------------------------------------------------------

const RESULT_WITH_VIF: ValidateIsStepResult = {
  config: { newey_west_lags: 4, fdr_alpha: 0.1, t_stat_threshold: 1.65 },
  n_significant: 3,
  significant_factors: ['MTUM', 'QUAL', 'VALUE'],
  high_vif_factors: ['MTUM', 'SIZE'],
  ic_results: [],
};

const RESULT_NO_VIF: ValidateIsStepResult = {
  config: { newey_west_lags: 4, fdr_alpha: 0.1, t_stat_threshold: 1.65 },
  n_significant: 2,
  significant_factors: ['QUAL', 'VALUE'],
  high_vif_factors: [],
  ic_results: [],
};

// ---------------------------------------------------------------------------
// Data-driven base-arm loop
// ---------------------------------------------------------------------------

interface StatusCase {
  label: string;
  status: StepStatus;
  result?: ValidateIsStepResult;
  lastError?: string;
  testid: string;
}

const baseCases: StatusCase[] = [
  { label: 'Ready shows run button',        status: StepStatus.Ready,     testid: 'run-step'    },
  { label: 'Running shows spinner',          status: StepStatus.Running,   testid: 'spinner'     },
  { label: 'Completed shows stat grid',      status: StepStatus.Completed, result: RESULT_NO_VIF, testid: 'vif-empty' },
  { label: 'lastError shows error-block',    status: StepStatus.Error,     lastError: 'boom',    testid: 'error-block' },
];

describe('StepValidateIsPanelComponent (render coverage)', () => {

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

  // ---- config-card @if / @else (independent of stepStatus) ------------------

  describe('config-card @if/@else branch', () => {
    it('when result is set, the config <dl> renders inside config-card', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('result', RESULT_NO_VIF);
      fixture.detectChanges();

      const card = el.querySelector('[data-testid="config-card"]')!;
      expect(card.querySelector('dl')).not.toBeNull();
    });

    it('when result is null, the placeholder paragraph renders inside config-card', async () => {
      const { el } = await build(); // result defaults to null

      const card = el.querySelector('[data-testid="config-card"]')!;
      expect(card.querySelector('dl')).toBeNull();
      expect(card.querySelector('p')).not.toBeNull();
    });
  });

  // ---- VIF badge @if (non-empty) vs @else (empty) ---------------------------

  describe('high_vif_factors @if/@else branch', () => {
    it('when high_vif_factors is non-empty, vif-badge elements render', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_WITH_VIF);
      fixture.detectChanges();

      const badges = el.querySelectorAll('[data-testid="vif-badge"]');
      expect(badges.length).toBe(RESULT_WITH_VIF.high_vif_factors.length);
    });

    it('when high_vif_factors is empty, vif-empty renders and no vif-badge exists', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_NO_VIF);
      fixture.detectChanges();

      expect(el.querySelector('[data-testid="vif-empty"]')).not.toBeNull();
      expect(el.querySelector('[data-testid="vif-badge"]')).toBeNull();
    });
  });
});
