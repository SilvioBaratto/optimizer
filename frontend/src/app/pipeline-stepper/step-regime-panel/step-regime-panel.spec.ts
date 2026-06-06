import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StepRegimePanelComponent } from './step-regime-panel';
import {
  StepStatus,
  type RegimeStepResult,
} from '../../core/models/pipeline-builder.model';

const SAMPLE_RESULT: RegimeStepResult = {
  regime: 'bull',
  tilts: { Quality: 1.1, Momentum: 0.9 },
  persisted: false,
  note: 'Regime Preview (inspection only)',
};

function setup(): ComponentFixture<StepRegimePanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepRegimePanelComponent);
}

function host(fix: ComponentFixture<StepRegimePanelComponent>): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepRegimePanelComponent', () => {
  describe('when stepStatus is Ready', () => {
    it('renders form with two checkboxes and Run Step button', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const el = host(fix);
      expect(el.querySelector('button[data-testid="run-step"]')).not.toBeNull();
      expect(
        el.querySelector('input[data-testid="enable-tilts-checkbox"]'),
      ).not.toBeNull();
      expect(
        el.querySelector('input[data-testid="persist-regime-checkbox"]'),
      ).not.toBeNull();
    });

    it('enable_tilts defaults to true, persist_regime defaults to false', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const { form } = fix.componentInstance;
      expect(form.controls.enable_tilts.value).toBe(true);
      expect(form.controls.persist_regime.value).toBe(false);
    });

    it('emits {enable_tilts, persist_regime} with correct values on submit', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const emitted: Record<string, unknown>[] = [];
      fix.componentInstance.runStep.subscribe((p) => emitted.push(p));

      fix.componentInstance.form.controls.enable_tilts.setValue(false);
      fix.componentInstance.form.controls.persist_regime.setValue(true);
      fix.componentInstance.onSubmit();

      expect(emitted).toEqual([{ enable_tilts: false, persist_regime: true }]);
    });

    it('emits exact default payload when submitted without changes', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const emitted: Record<string, unknown>[] = [];
      fix.componentInstance.runStep.subscribe((p) => emitted.push(p));
      fix.componentInstance.onSubmit();

      expect(emitted).toEqual([{ enable_tilts: true, persist_regime: false }]);
    });
  });

  describe('Running state', () => {
    it('renders spinner, hides Run Step button', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Running);
      await fix.whenStable();

      const el = host(fix);
      expect(el.querySelector('[data-testid="spinner"]')).not.toBeNull();
      expect(el.querySelector('button[data-testid="run-step"]')).toBeNull();
    });
  });

  describe('Completed render', () => {
    it('renders regime stat card and persisted badge', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('bull');
      expect(host(fix).querySelector('[data-testid="persisted-badge"]')).not.toBeNull();
    });

    it('renders tilt entries', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('Quality');
      expect(text).toContain('Momentum');
    });

    it('persisted badge is "no" when persisted is false', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="persisted-badge"]');
      expect(badge!.textContent).toContain('no');
    });
  });

  describe('Error render', () => {
    it('when lastError non-null, error block renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'Regime classification failed');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('Regime classification failed');
    });
  });
});
