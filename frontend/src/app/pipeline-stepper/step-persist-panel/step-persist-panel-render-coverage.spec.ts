/**
 * Render-coverage spec for StepPersistPanelComponent (issue #902).
 *
 * Drives every template branch at the DOM:
 *   - Ready      → [data-testid="run-step"]
 *   - Running    → [data-testid="spinner"]
 *   - Completed  → [data-testid="persisted-badge"] and [data-testid="checklist-badge"]
 *   - lastError  → [data-testid="error-block"]
 *
 * Secondary branches:
 *   - [class] on persisted-badge: text-gain (persisted=true) vs text-text-tertiary (false)
 *   - [class] on checklist-badge: text-gain (passed=true) vs text-loss (passed=false)
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { StepPersistPanelComponent } from './step-persist-panel';
import {
  StepStatus,
  type PersistStepResult,
} from '../../core/models/pipeline-builder.model';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface Harness {
  fixture: ComponentFixture<StepPersistPanelComponent>;
  el: HTMLElement;
}

async function build(): Promise<Harness> {
  await configureTestBed({ imports: [StepPersistPanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(StepPersistPanelComponent);
  fixture.componentRef.setInput('stepStatus', StepStatus.Pending);
  fixture.detectChanges();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

// ---------------------------------------------------------------------------
// Minimal result fixtures
// ---------------------------------------------------------------------------

const RESULT_PERSISTED_PASS: PersistStepResult = {
  persisted: true,
  reason: 'All checks passed',
  pass_count: 17,
  checklist_passed: true,
};

const RESULT_NOT_PERSISTED_FAIL: PersistStepResult = {
  persisted: false,
  reason: 'Checklist failed',
  pass_count: 14,
  checklist_passed: false,
};

// ---------------------------------------------------------------------------
// Data-driven base-arm loop
// ---------------------------------------------------------------------------

interface StatusCase {
  label: string;
  status: StepStatus;
  result?: PersistStepResult;
  lastError?: string;
  testid: string;
}

const baseCases: StatusCase[] = [
  { label: 'Ready shows run button',        status: StepStatus.Ready,                                       testid: 'run-step'      },
  { label: 'Running shows spinner',          status: StepStatus.Running,                                     testid: 'spinner'       },
  { label: 'Completed shows persisted-badge',status: StepStatus.Completed, result: RESULT_PERSISTED_PASS,   testid: 'persisted-badge'},
  { label: 'lastError shows error-block',    status: StepStatus.Error,     lastError: 'persist failed',      testid: 'error-block'   },
];

describe('StepPersistPanelComponent (render coverage)', () => {

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

  // ---- persisted-badge [class] binding ----------------------------------------

  describe('persisted-badge [class] binding', () => {
    it('when persisted is true, the badge carries text-gain', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_PERSISTED_PASS);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="persisted-badge"]')!;
      expect(badge.className).toContain('text-gain');
    });

    it('when persisted is false, the badge carries text-text-tertiary', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_NOT_PERSISTED_FAIL);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="persisted-badge"]')!;
      expect(badge.className).toContain('text-text-tertiary');
    });
  });

  // ---- checklist-badge [class] binding ----------------------------------------

  describe('checklist-badge [class] binding', () => {
    it('when checklist_passed is true, the badge carries text-gain', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_PERSISTED_PASS);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="checklist-badge"]')!;
      expect(badge.className).toContain('text-gain');
    });

    it('when checklist_passed is false, the badge carries text-loss', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_NOT_PERSISTED_FAIL);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="checklist-badge"]')!;
      expect(badge.className).toContain('text-loss');
    });
  });
});
