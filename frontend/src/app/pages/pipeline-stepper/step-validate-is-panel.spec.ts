import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StepValidateIsPanelComponent } from './step-validate-is-panel';
import {
  StepStatus,
  type ValidateIsStepResult,
} from '../../models/pipeline-builder.model';

const SAMPLE_RESULT: ValidateIsStepResult = {
  ic_results: [
    {
      factor_name: 'MOM_12M',
      mean_ic: 0.0423,
      ic_std: 0.1521,
      t_stat: 2.842,
      p_value: 0.0048,
      significant: true,
    },
    {
      factor_name: 'BAB',
      mean_ic: -0.0188,
      ic_std: 0.0921,
      t_stat: -1.412,
      p_value: 0.1592,
      significant: false,
    },
  ],
  n_significant: 1,
  significant_factors: ['MOM_12M'],
  high_vif_factors: ['QUAL', 'SIZE'],
  config: {
    newey_west_lags: 4,
    fdr_alpha: 0.1,
    t_stat_threshold: 1.645,
  },
};

const RESULT_NO_VIF: ValidateIsStepResult = {
  ...SAMPLE_RESULT,
  high_vif_factors: [],
};

function setup(): ComponentFixture<StepValidateIsPanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepValidateIsPanelComponent);
}

function host(fix: ComponentFixture<StepValidateIsPanelComponent>): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepValidateIsPanelComponent', () => {
  describe('read-only config card', () => {
    it('renders placeholder text before result is available', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Pending);
      await fix.whenStable();

      const card = host(fix).querySelector('[data-testid="config-card"]');
      expect(card).not.toBeNull();
      expect(card!.textContent).toMatch(/awaiting|pending|after/i);
    });

    it('after completion, renders the three IS config rows from result.config', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const card = host(fix).querySelector('[data-testid="config-card"]')!;
      const text = card.textContent ?? '';
      expect(text).toContain('Newey-West');
      expect(text).toContain('4');
      expect(text).toContain('FDR');
      expect(text).toContain('0.1');
      expect(text).toContain('t-stat');
      expect(text).toContain('1.645');
    });
  });

  describe('when stepStatus is Ready', () => {
    it('renders Run Step button (no form) and emits runStep({}) on click', async () => {
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
    it('exposes DataTable column config matching the AC', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const cols = fix.componentInstance.icColumns;
      const keys = cols.map((c) => c.key);
      expect(keys).toEqual([
        'factor_name',
        'mean_ic',
        'ic_std',
        't_stat',
        'p_value',
        'significant',
      ]);

      const tStat = cols.find((c) => c.key === 't_stat')!;
      expect(tStat.colorBySign).toBe(true);

      const sig = cols.find((c) => c.key === 'significant')!;
      expect(sig.type).toBe('badge');
      expect(sig.badgeMap).toBeDefined();
      expect(sig.badgeMap!['true'].colorClass).toContain('gain');
    });

    it('formats numeric columns to 4 decimals', () => {
      const fix = setup();
      const cols = fix.componentInstance.icColumns;
      const mean = cols.find((c) => c.key === 'mean_ic')!;
      expect(mean.format!(0.04234)).toBe('0.0423');
      expect(mean.format!(null)).toBe('—');
    });

    it('passes ic_results to the table as rows', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      expect(fix.componentInstance.icRows() as unknown).toEqual(
        SAMPLE_RESULT.ic_results as unknown,
      );
    });

    it('renders n_significant stat card', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('1');
      expect(text).toMatch(/significant/i);
    });

    it('renders high-VIF badge list when non-empty', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const badges = host(fix).querySelectorAll(
        '[data-testid="vif-badge"]',
      );
      expect(Array.from(badges).map((b) => b.textContent?.trim())).toEqual([
        'QUAL',
        'SIZE',
      ]);
      expect(host(fix).querySelector('[data-testid="vif-empty"]')).toBeNull();
    });

    it('renders "None" empty state when high_vif_factors is empty', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_NO_VIF);
      await fix.whenStable();

      const empty = host(fix).querySelector('[data-testid="vif-empty"]');
      expect(empty).not.toBeNull();
      expect(empty!.textContent).toContain('None');
    });
  });

  describe('Error render', () => {
    it('when lastError non-null, error block renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'IC compute failed');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('IC compute failed');
    });
  });
});
