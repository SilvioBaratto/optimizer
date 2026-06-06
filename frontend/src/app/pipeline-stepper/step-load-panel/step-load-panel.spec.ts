import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StepLoadPanelComponent } from './step-load-panel';
import {
  StepStatus,
  type LoadStepResult,
} from '../../core/models/pipeline-builder.model';

const SAMPLE_RESULT: LoadStepResult = {
  n_tickers: 1250,
  n_trading_days: 1260,
  assembly_hash: 'a1b2c3d4e5f60718',
  base_currency: 'EUR',
  price_start: '2020-01-02',
  price_end: '2024-12-31',
};

function setup(): ComponentFixture<StepLoadPanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepLoadPanelComponent);
}

function host(fix: ComponentFixture<StepLoadPanelComponent>): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepLoadPanelComponent', () => {
  describe('when stepStatus is Ready', () => {
    it('renders the Run Step button and emits runStep({}) on click', async () => {
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
    it('renders the spinner and hides the Run Step button', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Running);
      await fix.whenStable();

      const el = host(fix);
      expect(el.querySelector('[data-testid="spinner"]')).not.toBeNull();
      expect(el.querySelector('button[data-testid="run-step"]')).toBeNull();
    });
  });

  describe('when stepStatus is Completed and result is non-null', () => {
    it('renders a stat grid with all 4 stats and the assembly_hash subhead', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('1,250');
      expect(text).toContain('1,260');
      expect(text).toContain('EUR');
      // price range rendered via FormatDatePipe — both endpoint years appear
      expect(text).toContain('2020');
      expect(text).toContain('2024');
      // assembly hash rendered as monospace subhead
      expect(text).toContain('a1b2c3d4e5f60718');
      expect(
        host(fix).querySelector('[data-testid="assembly-hash"]'),
      ).not.toBeNull();
    });
  });

  describe('when lastError is non-null', () => {
    it('renders the error block', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'DB preflight failed');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('DB preflight failed');
    });
  });
});
