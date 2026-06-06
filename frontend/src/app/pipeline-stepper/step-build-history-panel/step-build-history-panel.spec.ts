import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StepBuildHistoryPanelComponent } from './step-build-history-panel';
import {
  StepStatus,
  type BuildHistoryStepResult,
} from '../../core/models/pipeline-builder.model';

const RESULT_CLEAN: BuildHistoryStepResult = {
  succeeded_dates: 20,
  total_dates: 20,
  failed_dates: 0,
  n_factors: 17,
  rebalance_freq: 63,
  market_proxy_loaded: true,
};

const RESULT_WITH_FAILURES: BuildHistoryStepResult = {
  ...RESULT_CLEAN,
  succeeded_dates: 17,
  failed_dates: 3,
  market_proxy_loaded: false,
};

function setup(): ComponentFixture<StepBuildHistoryPanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepBuildHistoryPanelComponent);
}

function host(fix: ComponentFixture<StepBuildHistoryPanelComponent>): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepBuildHistoryPanelComponent', () => {
  describe('read-only metadata card', () => {
    it('renders FactorConstructionConfig, StandardizationConfig, min_success_fraction labels regardless of status', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Pending);
      await fix.whenStable();

      const card = host(fix).querySelector('[data-testid="config-card"]');
      expect(card).not.toBeNull();
      const text = card!.textContent ?? '';
      expect(text).toContain('FactorConstructionConfig');
      expect(text).toContain('StandardizationConfig');
      expect(text).toContain('min_success_fraction');
      expect(text).toContain('0.5');
    });
  });

  describe('form defaults', () => {
    it('market_proxy defaults to URTH and form valid', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      expect(fix.componentInstance.form.controls.market_proxy_ticker.value).toBe('URTH');
      expect(fix.componentInstance.form.valid).toBe(true);
    });

    it('when market_proxy is empty, form is invalid', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      fix.componentInstance.form.controls.market_proxy_ticker.setValue('');
      expect(fix.componentInstance.form.valid).toBe(false);
    });
  });

  describe('submit', () => {
    it('when Ready and form valid, submit emits runStep with market_proxy override', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const emitted: Record<string, unknown>[] = [];
      fix.componentInstance.runStep.subscribe((p) => emitted.push(p));

      fix.componentInstance.form.controls.market_proxy_ticker.setValue('SPY');
      const btn = host(fix).querySelector<HTMLButtonElement>(
        'button[data-testid="run-step"]',
      );
      btn!.click();

      expect(emitted).toEqual([{ market_proxy_ticker: 'SPY' }]);
    });

    it('when form invalid, submit does NOT emit', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const emitted: unknown[] = [];
      fix.componentInstance.runStep.subscribe((p) => emitted.push(p));

      fix.componentInstance.form.controls.market_proxy_ticker.setValue('');
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
    it('renders succeeded/total fraction, n_factors, rebalance_freq, proxy badge', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_CLEAN);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('20 / 20');
      expect(text).toContain('17');
      expect(text).toContain('63');
      expect(host(fix).querySelector('[data-testid="proxy-badge"]')).not.toBeNull();
    });

    it('when failed_dates is 0, NO warning chip renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_CLEAN);
      await fix.whenStable();

      expect(host(fix).querySelector('[data-testid="failed-warning"]')).toBeNull();
    });

    it('when failed_dates > 0, warning chip with the count renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_WITH_FAILURES);
      await fix.whenStable();

      const chip = host(fix).querySelector('[data-testid="failed-warning"]');
      expect(chip).not.toBeNull();
      expect(chip!.textContent).toContain('3');
    });
  });

  describe('Error render', () => {
    it('when lastError non-null, error block renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'min_success_fraction not met');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('min_success_fraction not met');
    });
  });
});
