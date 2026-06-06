import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StepOptimizePanelComponent } from './step-optimize-panel';
import {
  StepStatus,
  type OptimizeStepResult,
} from '../../models/pipeline-builder.model';

const SAMPLE_RESULT: OptimizeStepResult = {
  weights: [
    { ticker: 'AAPL', weight: 0.05 },
    { ticker: 'MSFT', weight: 0.04 },
  ],
  n_selected: 2,
  is_sharpe: 1.42,
  net_sharpe: 1.21,
  hockey_stick_warning: false,
  sector_breakdown: { Technology: 0.65, Financials: 0.35 },
  country_breakdown: { US: 0.8, UK: 0.2 },
};

function setup(): ComponentFixture<StepOptimizePanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepOptimizePanelComponent);
}

function host(fix: ComponentFixture<StepOptimizePanelComponent>): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepOptimizePanelComponent', () => {
  describe('when stepStatus is Ready', () => {
    it('renders Run Step button (no form)', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const el = host(fix);
      expect(el.querySelector('button[data-testid="run-step"]')).not.toBeNull();
      expect(el.querySelector('input,select,textarea,form')).toBeNull();
    });

    it('emits {} on Run Step click', async () => {
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
    it('renders spinner, hides Run Step', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Running);
      await fix.whenStable();

      const el = host(fix);
      expect(el.querySelector('[data-testid="spinner"]')).not.toBeNull();
      expect(el.querySelector('button[data-testid="run-step"]')).toBeNull();
    });
  });

  describe('Completed render', () => {
    it('renders n_selected, is_sharpe, net_sharpe stat cards', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('2');
      expect(text).toContain('1.420');
      expect(text).toContain('1.210');
    });

    it('hockey-stick badge shows "ok" when false', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="hockey-stick-badge"]');
      expect(badge).not.toBeNull();
      expect(badge!.textContent).toContain('ok');
    });

    it('hockey-stick badge shows "warning" when true', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', {
        ...SAMPLE_RESULT,
        hockey_stick_warning: true,
      });
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="hockey-stick-badge"]');
      expect(badge!.textContent).toContain('warning');
    });

    it('exposes weightRows() from result.weights', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      expect(fix.componentInstance.weightRows() as unknown).toEqual(
        SAMPLE_RESULT.weights as unknown,
      );
    });

    it('renders sector breakdown entries', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('Technology');
      expect(text).toContain('Financials');
    });
  });

  describe('Error render', () => {
    it('when lastError non-null, error block renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'Optimization failed: singular matrix');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('Optimization failed: singular matrix');
    });
  });
});
