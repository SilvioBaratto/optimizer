/**
 * Render-coverage spec for StepLoadPanelComponent (issue #901).
 *
 * NOTE: Base status-arm logic is already exercised by step-load-panel.spec.ts
 * (instance-level). This file mounts the full template to ensure the four
 * status-arm branches are hit at the DOM level for Karma branch coverage:
 *   - Ready  → [data-testid="run-step"]
 *   - Running → [data-testid="spinner"]
 *   - Completed + loadResult → [data-testid="assembly-hash"]
 *   - lastError → [data-testid="error-block"]
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { StepLoadPanelComponent } from './step-load-panel';
import { StepStatus, type LoadStepResult } from '../../core/models/pipeline-builder.model';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface Harness {
  fixture: ComponentFixture<StepLoadPanelComponent>;
  el: HTMLElement;
}

async function build(): Promise<Harness> {
  await configureTestBed({ imports: [StepLoadPanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(StepLoadPanelComponent);
  fixture.componentRef.setInput('stepStatus', StepStatus.Pending);
  fixture.detectChanges();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

// ---------------------------------------------------------------------------
// Minimal result fixture
// ---------------------------------------------------------------------------

const RESULT: LoadStepResult = {
  n_tickers: 50,
  n_trading_days: 252,
  assembly_hash: 'abc123',
  base_currency: 'USD',
  price_start: '2020-01-01',
  price_end: '2024-01-01',
};

// ---------------------------------------------------------------------------
// Data-driven base-arm loop
// ---------------------------------------------------------------------------

interface StatusCase {
  label: string;
  status: StepStatus;
  result?: LoadStepResult;
  lastError?: string;
  testid: string;
}

const baseCases: StatusCase[] = [
  { label: 'Ready shows run button',       status: StepStatus.Ready,     testid: 'run-step'       },
  { label: 'Running shows spinner',         status: StepStatus.Running,   testid: 'spinner'        },
  { label: 'Completed shows assembly-hash', status: StepStatus.Completed, result: RESULT, testid: 'assembly-hash' },
  { label: 'lastError shows error-block',   status: StepStatus.Error,     lastError: 'boom',       testid: 'error-block' },
];

describe('StepLoadPanelComponent (render coverage)', () => {
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
});
