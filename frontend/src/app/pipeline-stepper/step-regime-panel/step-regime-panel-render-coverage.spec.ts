/**
 * Render-coverage spec for StepRegimePanelComponent (issue #901).
 *
 * Drives every template branch at the DOM:
 *   - Ready  → [data-testid="run-step"]
 *   - Running → [data-testid="spinner"]
 *   - Completed + regimeResult → [data-testid="persisted-badge"]
 *   - lastError → [data-testid="error-block"]
 *
 * Secondary branches:
 *   - @if (tiltEntries().length > 0) — result with tilts vs result without
 *   - @if (r.note) — note set vs absent
 *   - [class] binding on persisted-badge — 'bg-gain/10' when true, absent when false
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { StepRegimePanelComponent } from './step-regime-panel';
import { StepStatus, type RegimeStepResult } from '../../core/models/pipeline-builder.model';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface Harness {
  fixture: ComponentFixture<StepRegimePanelComponent>;
  el: HTMLElement;
}

async function build(): Promise<Harness> {
  await configureTestBed({ imports: [StepRegimePanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(StepRegimePanelComponent);
  fixture.componentRef.setInput('stepStatus', StepStatus.Pending);
  fixture.detectChanges();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

// ---------------------------------------------------------------------------
// Minimal result fixtures
// ---------------------------------------------------------------------------

const RESULT_WITH_TILTS: RegimeStepResult = {
  regime: 'expansion',
  tilts: { momentum: 1.2, quality: 0.9 },
  persisted: true,
  note: 'GDP above trend',
};

const RESULT_NO_TILTS: RegimeStepResult = {
  regime: 'contraction',
  tilts: {},
  persisted: false,
  note: '',
};

// ---------------------------------------------------------------------------
// Data-driven base-arm loop
// ---------------------------------------------------------------------------

interface StatusCase {
  label: string;
  status: StepStatus;
  result?: RegimeStepResult;
  lastError?: string;
  testid: string;
}

const baseCases: StatusCase[] = [
  { label: 'Ready shows run button',       status: StepStatus.Ready,     testid: 'run-step'       },
  { label: 'Running shows spinner',         status: StepStatus.Running,   testid: 'spinner'        },
  { label: 'Completed shows persisted-badge', status: StepStatus.Completed, result: RESULT_WITH_TILTS, testid: 'persisted-badge' },
  { label: 'lastError shows error-block',   status: StepStatus.Error,     lastError: 'err',        testid: 'error-block' },
];

describe('StepRegimePanelComponent (render coverage)', () => {
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

  // ---- Secondary: @if (tiltEntries().length > 0) ---------------------------

  describe('tilt entries @if', () => {
    it('when tilts is non-empty, the tilt dl renders', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_WITH_TILTS);
      fixture.detectChanges();

      expect(el.querySelector('dl')).not.toBeNull();
    });

    it('when tilts is empty, the tilt dl is absent', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_NO_TILTS);
      fixture.detectChanges();

      expect(el.querySelector('dl')).toBeNull();
    });
  });

  // ---- Secondary: @if (r.note) ---------------------------------------------

  describe('note @if', () => {
    it('when note is non-empty, the note paragraph renders', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_WITH_TILTS);
      fixture.detectChanges();

      const note = el.querySelector('p.italic');
      expect(note).not.toBeNull();
      expect(note!.textContent).toContain('GDP above trend');
    });

    it('when note is empty, the note paragraph is absent', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_NO_TILTS);
      fixture.detectChanges();

      expect(el.querySelector('p.italic')).toBeNull();
    });
  });

  // ---- Secondary: [class] binding on persisted-badge -----------------------

  describe('persisted-badge [class] binding', () => {
    it('when persisted is true, the badge carries the gain class', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_WITH_TILTS);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="persisted-badge"]')!;
      expect(badge.className).toContain('text-gain');
    });

    it('when persisted is false, the badge carries the tertiary class', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_NO_TILTS);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="persisted-badge"]')!;
      expect(badge.className).toContain('text-text-tertiary');
    });
  });
});
