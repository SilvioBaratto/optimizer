import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import {
  PREPROCESSING_PIPELINE,
  StepCleanReturnsPanelComponent,
} from './step-clean-returns-panel';
import {
  StepStatus,
  type CleanReturnsStepResult,
} from '../../models/pipeline-builder.model';

const SAMPLE_RESULT: CleanReturnsStepResult = {
  n_days: 1260,
  n_tickers: 850,
  return_start: '2020-01-03',
  return_end: '2024-12-31',
  preprocessing_steps: [
    'DataValidator',
    'OutlierTreater',
    'SectorImputer',
    'RegressionImputer',
  ],
};

function setup(): ComponentFixture<StepCleanReturnsPanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepCleanReturnsPanelComponent);
}

function host(fix: ComponentFixture<StepCleanReturnsPanelComponent>): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepCleanReturnsPanelComponent', () => {
  describe('PREPROCESSING_PIPELINE constant', () => {
    it('exports the four preprocessors in fixed order', () => {
      expect(PREPROCESSING_PIPELINE).toEqual([
        'DataValidator',
        'OutlierTreater',
        'SectorImputer',
        'RegressionImputer',
      ]);
    });
  });

  describe('read-only config card', () => {
    it('renders the four preprocessor names regardless of status', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Pending);
      await fix.whenStable();

      const card = host(fix).querySelector('[data-testid="config-card"]');
      expect(card).not.toBeNull();
      const text = card!.textContent ?? '';
      for (const name of PREPROCESSING_PIPELINE) {
        expect(text).toContain(name);
      }
    });

    it('has NO editable controls (no inputs / selects / textareas)', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const card = host(fix).querySelector('[data-testid="config-card"]')!;
      expect(card.querySelector('input,select,textarea')).toBeNull();
    });
  });

  describe('when stepStatus is Ready', () => {
    it('renders Run Step button and emits runStep({}) on click', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const emitted: Record<string, unknown>[] = [];
      fix.componentInstance.runStep.subscribe((p) => emitted.push(p));

      const btn = host(fix).querySelector<HTMLButtonElement>(
        'button[data-testid="run-step"]',
      );
      expect(btn).not.toBeNull();
      btn!.click();
      expect(emitted).toEqual([{}]);
    });
  });

  describe('when stepStatus is Running', () => {
    it('renders the spinner and hides Run Step button', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Running);
      await fix.whenStable();

      const el = host(fix);
      expect(el.querySelector('[data-testid="spinner"]')).not.toBeNull();
      expect(el.querySelector('button[data-testid="run-step"]')).toBeNull();
    });
  });

  describe('when stepStatus is Completed and result is non-null', () => {
    it('renders stat cards for n_days, n_tickers, date range', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('1,260');
      expect(text).toContain('850');
      expect(text).toContain('2020');
      expect(text).toContain('2024');
    });

    it('renders preprocessing_steps as a badge list', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const badges = host(fix).querySelectorAll(
        '[data-testid="preprocessing-badge"]',
      );
      expect(badges.length).toBe(4);
      const labels = Array.from(badges).map((b) => b.textContent?.trim());
      expect(labels).toEqual([
        'DataValidator',
        'OutlierTreater',
        'SectorImputer',
        'RegressionImputer',
      ]);
    });
  });

  describe('when lastError is non-null', () => {
    it('renders the error block', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'NaN survived preprocessing');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('NaN survived preprocessing');
    });
  });
});
