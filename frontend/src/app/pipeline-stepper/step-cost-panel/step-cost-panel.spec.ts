import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StepCostPanelComponent } from './step-cost-panel';
import {
  StepStatus,
  type CostStepResult,
} from '../../core/models/pipeline-builder.model';

const SAMPLE_RESULT: CostStepResult = {
  cost_bps_actual: 8.4,
  cost_bps_assumed: 10.0,
  exceeds_assumed: false,
  per_country: [
    { country: 'UK', stamp: 5.0, spread: 2.5, fx: 0.5, total: 8.0 },
    { country: 'US', stamp: 0.0, spread: 2.0, fx: 0.0, total: 2.0 },
  ],
};

const RESULT_EXCEEDS: CostStepResult = {
  ...SAMPLE_RESULT,
  cost_bps_actual: 15.2,
  exceeds_assumed: true,
};

function setup(): ComponentFixture<StepCostPanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepCostPanelComponent);
}

function host(fix: ComponentFixture<StepCostPanelComponent>): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepCostPanelComponent', () => {
  describe('when stepStatus is Ready', () => {
    it('renders Run Step button with no form inputs', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const el = host(fix);
      expect(el.querySelector('button[data-testid="run-step"]')).not.toBeNull();
      expect(el.querySelector('input,select,textarea,form')).toBeNull();
    });

    it('emits {} on click', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const emitted: Record<string, unknown>[] = [];
      fix.componentInstance.runStep.subscribe((p) => emitted.push(p));

      host(fix)
        .querySelector<HTMLButtonElement>('button[data-testid="run-step"]')!
        .click();

      expect(emitted).toEqual([{}]);
    });
  });

  describe('Running state', () => {
    it('shows spinner, hides Run Step', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Running);
      await fix.whenStable();

      const el = host(fix);
      expect(el.querySelector('[data-testid="spinner"]')).not.toBeNull();
      expect(el.querySelector('button[data-testid="run-step"]')).toBeNull();
    });
  });

  describe('Completed render', () => {
    it('renders actual and assumed cost stat cards', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('8.40');
      expect(text).toContain('10.00');
    });

    it('exceeds badge shows "no" when not exceeding', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="exceeds-badge"]');
      expect(badge).not.toBeNull();
      expect(badge!.textContent).toContain('no');
    });

    it('exceeds badge shows "yes" when exceeds_assumed is true', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_EXCEEDS);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="exceeds-badge"]');
      expect(badge!.textContent).toContain('yes');
    });

    it('exposes costRows() from result.per_country', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      expect(fix.componentInstance.costRows() as unknown).toEqual(
        SAMPLE_RESULT.per_country as unknown,
      );
    });
  });

  describe('Error render', () => {
    it('when lastError non-null, error block renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'Cost calculation failed');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('Cost calculation failed');
    });
  });
});
