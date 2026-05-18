import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StepRebalanceDecisionPanelComponent } from './step-rebalance-decision-panel';
import {
  StepStatus,
  type RebalanceDecisionStepResult,
} from '../../models/pipeline-builder.model';

const SAMPLE_RESULT: RebalanceDecisionStepResult = {
  decision: true,
  reason: 'calendar',
  n_weights: 20,
  weight_count_warning: false,
  missing_sectors: [],
  current_date: '2025-05-18',
  last_review_date: '2025-02-15',
};

const RESULT_WITH_WARNINGS: RebalanceDecisionStepResult = {
  ...SAMPLE_RESULT,
  decision: false,
  reason: 'cold_start',
  n_weights: 8,
  weight_count_warning: true,
  missing_sectors: ['Utilities', 'Real Estate'],
};

function setup(): ComponentFixture<StepRebalanceDecisionPanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepRebalanceDecisionPanelComponent);
}

function host(
  fix: ComponentFixture<StepRebalanceDecisionPanelComponent>,
): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepRebalanceDecisionPanelComponent', () => {
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
    it('renders decision badge as "Rebalance" when decision is true', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="decision-badge"]');
      expect(badge).not.toBeNull();
      expect(badge!.textContent).toContain('Rebalance');
    });

    it('renders decision badge as "Hold" when decision is false', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_WITH_WARNINGS);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="decision-badge"]');
      expect(badge!.textContent).toContain('Hold');
    });

    it('renders n_weights, reason, current_date', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('20');
      expect(text).toContain('calendar');
      expect(text).toContain('2025-05-18');
    });

    it('weight-count badge shows "ok" when no warning', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="weight-count-badge"]');
      expect(badge!.textContent).toContain('ok');
    });

    it('weight-count badge shows "outside [15, 30]" when warning', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_WITH_WARNINGS);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="weight-count-badge"]');
      expect(badge!.textContent).toContain('outside');
    });

    it('renders missing sector badges when present', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_WITH_WARNINGS);
      await fix.whenStable();

      const badges = host(fix).querySelectorAll('[data-testid="missing-sector-badge"]');
      expect(badges.length).toBe(2);
      const text = Array.from(badges)
        .map((b) => b.textContent)
        .join(' ');
      expect(text).toContain('Utilities');
      expect(text).toContain('Real Estate');
    });

    it('no missing sector badges when missing_sectors is empty', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      expect(
        host(fix).querySelectorAll('[data-testid="missing-sector-badge"]').length,
      ).toBe(0);
    });
  });

  describe('Error render', () => {
    it('when lastError non-null, error block renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'Rebalance decision failed');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('Rebalance decision failed');
    });
  });
});
