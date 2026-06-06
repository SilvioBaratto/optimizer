import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StepReportPanelComponent } from './step-report-panel';
import {
  StepStatus,
  type ReportStepResult,
} from '../../models/pipeline-builder.model';

const SAMPLE_RESULT: ReportStepResult = {
  pass_count: 17,
  checklist_total: 17,
  checklist_passed: true,
  checklist_rules: [
    { rule: 'Sharpe > 0.5', pass: true, measured: 1.42, target: '> 0.5' },
    { rule: 'Max DD < 30%', pass: false, measured: 0.35, target: '< 0.30' },
  ],
  metrics: { performance: { sharpe: 1.42, sortino: 2.1 } },
  artifact_paths: {
    report_md: '/tmp/pipeline_abc123/report.md',
    weights_csv: '/tmp/pipeline_abc123/weights.csv',
    metrics_json: '/tmp/pipeline_abc123/metrics.json',
    checklist_json: '/tmp/pipeline_abc123/checklist.json',
    weights_diagnostic: '/tmp/pipeline_abc123/weights_diagnostic.csv',
  },
  chart_paths: ['/tmp/pipeline_abc123/equity_curve.png'],
  output_dir: '/tmp/pipeline_abc123',
};

const RESULT_FAILED: ReportStepResult = {
  ...SAMPLE_RESULT,
  pass_count: 14,
  checklist_passed: false,
};

function setup(): ComponentFixture<StepReportPanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepReportPanelComponent);
}

function host(fix: ComponentFixture<StepReportPanelComponent>): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepReportPanelComponent', () => {
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
    it('renders pass_count / checklist_total stat card', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('17 / 17');
    });

    it('checklist badge shows "PASS" when checklist_passed is true', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="checklist-badge"]');
      expect(badge).not.toBeNull();
      expect(badge!.textContent).toContain('PASS');
    });

    it('checklist badge shows "FAIL" when checklist_passed is false', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_FAILED);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="checklist-badge"]');
      expect(badge!.textContent).toContain('FAIL');
    });

    it('exposes checklistRows() from result.checklist_rules', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      expect(fix.componentInstance.checklistRows() as unknown).toEqual(
        SAMPLE_RESULT.checklist_rules as unknown,
      );
    });

    it('renders artifact paths', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', SAMPLE_RESULT);
      await fix.whenStable();

      const paths = host(fix).querySelectorAll('[data-testid="artifact-path"]');
      expect(paths.length).toBe(5);
      const texts = Array.from(paths).map((el) => el.textContent ?? '');
      expect(texts.some((t) => t.includes('report.md'))).toBe(true);
      expect(texts.some((t) => t.includes('weights.csv'))).toBe(true);
    });
  });

  describe('Error render', () => {
    it('when lastError non-null, error block renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'Report generation failed');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('Report generation failed');
    });
  });
});
