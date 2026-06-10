/**
 * Render-coverage spec for StepValidateOosPanelComponent (issue #902).
 *
 * Drives every template branch at the DOM:
 *   - Ready      → [data-testid="run-step"]
 *   - Running    → [data-testid="spinner"]
 *   - Completed  → app-data-table present when result supplied
 *   - lastError  → [data-testid="error-block"]
 *
 * Secondary branches:
 *   - config-card @if (oosResult(); as r) → <dl> renders when result is set,
 *     placeholder <p> renders when result is null (independent of stepStatus)
 *   - Completed arm renders app-data-table for OOS rows
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { StepValidateOosPanelComponent } from './step-validate-oos-panel';
import {
  StepStatus,
  type ValidateOosStepResult,
} from '../../core/models/pipeline-builder.model';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface Harness {
  fixture: ComponentFixture<StepValidateOosPanelComponent>;
  el: HTMLElement;
}

async function build(): Promise<Harness> {
  await configureTestBed({ imports: [StepValidateOosPanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(StepValidateOosPanelComponent);
  fixture.componentRef.setInput('stepStatus', StepStatus.Pending);
  fixture.detectChanges();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

// ---------------------------------------------------------------------------
// Minimal result fixtures
// ---------------------------------------------------------------------------

const RESULT_WITH_ROWS: ValidateOosStepResult = {
  n_folds: 4,
  oos_results: [
    { factor_name: 'MTUM', oos_mean_ic: 0.05, oos_icir: 0.8 },
    { factor_name: 'QUAL', oos_mean_ic: 0.03, oos_icir: 0.6 },
  ],
  config: { train_periods: 252, val_periods: 63, step_periods: 63 },
};

const RESULT_NO_ROWS: ValidateOosStepResult = {
  n_folds: 0,
  oos_results: [],
  config: { train_periods: 252, val_periods: 63, step_periods: 63 },
};

// ---------------------------------------------------------------------------
// Data-driven base-arm loop
// ---------------------------------------------------------------------------

interface StatusCase {
  label: string;
  status: StepStatus;
  result?: ValidateOosStepResult;
  lastError?: string;
  testid: string;
}

const baseCases: StatusCase[] = [
  { label: 'Ready shows run button',     status: StepStatus.Ready,     testid: 'run-step'   },
  { label: 'Running shows spinner',       status: StepStatus.Running,   testid: 'spinner'    },
  { label: 'Completed shows data table',  status: StepStatus.Completed, result: RESULT_WITH_ROWS, testid: 'spinner' },
  { label: 'lastError shows error-block', status: StepStatus.Error,     lastError: 'oos-err', testid: 'error-block' },
];

describe('StepValidateOosPanelComponent (render coverage)', () => {

  // ---- Base status arms -------------------------------------------------------

  it('when Ready shows run button', async () => {
    const { fixture, el } = await build();
    fixture.componentRef.setInput('stepStatus', StepStatus.Ready);
    fixture.detectChanges();
    expect(el.querySelector('[data-testid="run-step"]')).not.toBeNull();
  });

  it('when Running shows spinner', async () => {
    const { fixture, el } = await build();
    fixture.componentRef.setInput('stepStatus', StepStatus.Running);
    fixture.detectChanges();
    expect(el.querySelector('[data-testid="spinner"]')).not.toBeNull();
  });

  it('when Completed with result, app-data-table renders', async () => {
    const { fixture, el } = await build();
    fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
    fixture.componentRef.setInput('result', RESULT_WITH_ROWS);
    fixture.detectChanges();
    expect(el.querySelector('app-data-table')).not.toBeNull();
  });

  it('when lastError is set, error-block renders', async () => {
    const { fixture, el } = await build();
    fixture.componentRef.setInput('stepStatus', StepStatus.Error);
    fixture.componentRef.setInput('lastError', 'oos validation failed');
    fixture.detectChanges();
    expect(el.querySelector('[data-testid="error-block"]')).not.toBeNull();
  });

  // ---- config-card @if/@else (independent of stepStatus) --------------------

  describe('config-card @if/@else branch', () => {
    it('when result is set, the config <dl> renders inside config-card', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('result', RESULT_NO_ROWS);
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

  // ---- Completed arm: app-data-table absent when oos_results is empty -------

  describe('OOS table presence', () => {
    it('when Completed with non-empty oos_results, app-data-table is present', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_WITH_ROWS);
      fixture.detectChanges();

      expect(el.querySelector('app-data-table')).not.toBeNull();
    });

    it('when Completed with empty oos_results, app-data-table is still rendered (0 rows)', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_NO_ROWS);
      fixture.detectChanges();

      // The Completed arm always renders the table regardless of row count
      expect(el.querySelector('app-data-table')).not.toBeNull();
    });
  });
});
