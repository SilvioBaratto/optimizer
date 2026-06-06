import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StepCoverageGatePanelComponent } from './step-coverage-gate-panel';
import {
  StepStatus,
  type CoverageGateStepResult,
} from '../../models/pipeline-builder.model';

const SAMPLE_RESULT: CoverageGateStepResult = {
  passing_factors: ['MOM_12M', 'BAB'],
  is_only_factors: ['QUAL'],
  oos_only_factors: ['LOWVOL', 'SIZE'],
  n_passing: 2,
  min_factors: 2,
};

function setup(): ComponentFixture<StepCoverageGatePanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepCoverageGatePanelComponent);
}

function host(fix: ComponentFixture<StepCoverageGatePanelComponent>): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepCoverageGatePanelComponent', () => {
  describe('form defaults', () => {
    it('min_factors defaults to 2 and form is valid', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      expect(fix.componentInstance.form.controls.min_factors.value).toBe(2);
      expect(fix.componentInstance.form.valid).toBe(true);
    });
  });

  describe('min_factors validation', () => {
    it('when min_factors = 0, form is invalid', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      fix.componentInstance.form.controls.min_factors.setValue(0);
      expect(fix.componentInstance.form.valid).toBe(false);
    });

    it('when min_factors = 1, form is valid', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      fix.componentInstance.form.controls.min_factors.setValue(1);
      expect(fix.componentInstance.form.valid).toBe(true);
    });
  });

  describe('submit', () => {
    it('when Ready and form valid, submit emits runStep with min_factors', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const emitted: Record<string, unknown>[] = [];
      fix.componentInstance.runStep.subscribe((p) => emitted.push(p));

      fix.componentInstance.form.controls.min_factors.setValue(3);
      const btn = host(fix).querySelector<HTMLButtonElement>(
        'button[data-testid="run-step"]',
      );
      btn!.click();

      expect(emitted).toEqual([{ min_factors: 3 }]);
    });

    it('when form invalid, submit does NOT emit', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const emitted: unknown[] = [];
      fix.componentInstance.runStep.subscribe((p) => emitted.push(p));

      fix.componentInstance.form.controls.min_factors.setValue(0);
      fix.componentInstance.onSubmit();

      expect(emitted).toEqual([]);
    });
  });

  describe('Running state', () => {
    it('renders spinner, hides submit', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Running);
      await fix.whenStable();

      const el = host(fix);
      expect(el.querySelector('[data-testid="spinner"]')).not.toBeNull();
      expect(el.querySelector('button[data-testid="run-step"]')).toBeNull();
    });
  });

  describe('Completed render', () => {
    it('renders n_passing / min_factors stat card', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('2 / 2');
    });

    it('renders three badge groups: passing, IS-only, OOS-only', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const passing = host(fix).querySelectorAll(
        '[data-testid="passing-badge"]',
      );
      const isOnly = host(fix).querySelectorAll('[data-testid="is-only-badge"]');
      const oosOnly = host(fix).querySelectorAll('[data-testid="oos-only-badge"]');

      expect(Array.from(passing).map((b) => b.textContent?.trim())).toEqual([
        'MOM_12M',
        'BAB',
      ]);
      expect(Array.from(isOnly).map((b) => b.textContent?.trim())).toEqual([
        'QUAL',
      ]);
      expect(Array.from(oosOnly).map((b) => b.textContent?.trim())).toEqual([
        'LOWVOL',
        'SIZE',
      ]);
    });
  });

  describe('abort-gate handling (no duplication of stepper banner)', () => {
    it('when abortGateError is set, panel does NOT render its own gate-warning block', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('abortGateError', 'Only 1 factor passed');
      await fix.whenStable();

      // Stepper renders the abort banner outside the panel outlet. The
      // panel itself must NOT render a second copy.
      expect(host(fix).querySelector('[data-testid="abort-gate"]')).toBeNull();
    });
  });

  describe('Error render', () => {
    it('when lastError non-null, error block renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'coverage compute failed');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('coverage compute failed');
    });
  });
});
