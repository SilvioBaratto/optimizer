import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StepValidateOosPanelComponent } from './step-validate-oos-panel';
import {
  StepStatus,
  type ValidateOosStepResult,
} from '../../models/pipeline-builder.model';

const SAMPLE_RESULT: ValidateOosStepResult = {
  n_folds: 6,
  oos_results: [
    { factor_name: 'MOM_12M', oos_mean_ic: 0.0312, oos_icir: 0.412 },
    { factor_name: 'BAB', oos_mean_ic: -0.0102, oos_icir: null },
  ],
  config: {
    train_periods: 8,
    val_periods: 4,
    step_periods: 2,
  },
};

function setup(): ComponentFixture<StepValidateOosPanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepValidateOosPanelComponent);
}

function host(fix: ComponentFixture<StepValidateOosPanelComponent>): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepValidateOosPanelComponent', () => {
  describe('read-only config card', () => {
    it('renders placeholder before result is available', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Pending);
      await fix.whenStable();

      const card = host(fix).querySelector('[data-testid="config-card"]');
      expect(card).not.toBeNull();
      expect(card!.textContent).toMatch(/awaiting|pending|after/i);
    });

    it('after completion, renders the three OOS config rows from result.config', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).querySelector('[data-testid="config-card"]')!
        .textContent ?? '';
      expect(text).toContain('Train periods');
      expect(text).toContain('8');
      expect(text).toContain('Val periods');
      expect(text).toContain('4');
      expect(text).toContain('Step periods');
      expect(text).toContain('2');
    });
  });

  describe('when stepStatus is Ready', () => {
    it('renders Run Step (no form) and emits runStep({}) on click', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const emitted: Record<string, unknown>[] = [];
      fix.componentInstance.runStep.subscribe((p) => emitted.push(p));

      const btn = host(fix).querySelector<HTMLButtonElement>(
        'button[data-testid="run-step"]',
      );
      expect(btn).not.toBeNull();
      expect(host(fix).querySelector('input,select,textarea,form')).toBeNull();
      btn!.click();
      expect(emitted).toEqual([{}]);
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
    it('exposes DataTable column config matching the AC', () => {
      const fix = setup();
      const cols = fix.componentInstance.oosColumns;
      const keys = cols.map((c) => c.key);
      expect(keys).toEqual(['factor_name', 'oos_mean_ic', 'oos_icir']);

      const ic = cols.find((c) => c.key === 'oos_mean_ic')!;
      expect(ic.colorBySign).toBe(true);
      const icir = cols.find((c) => c.key === 'oos_icir')!;
      expect(icir.colorBySign).toBe(true);
    });

    it('formats numeric columns to 4 decimals and renders "—" for null', () => {
      const fix = setup();
      const cols = fix.componentInstance.oosColumns;
      const icir = cols.find((c) => c.key === 'oos_icir')!;
      expect(icir.format!(0.4123)).toBe('0.4123');
      expect(icir.format!(null)).toBe('—');
    });

    it('passes oos_results to the table as rows', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      expect(fix.componentInstance.oosRows() as unknown).toEqual(
        SAMPLE_RESULT.oos_results as unknown,
      );
    });

    it('renders n_folds stat card', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('6');
      expect(text).toMatch(/fold/i);
    });
  });

  describe('Error render', () => {
    it('when lastError non-null, error block renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'OOS fold compute failed');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('OOS fold compute failed');
    });
  });
});
