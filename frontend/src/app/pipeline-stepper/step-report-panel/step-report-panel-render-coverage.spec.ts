/**
 * Render-coverage spec for StepReportPanelComponent (issue #902).
 *
 * Drives every template branch at the DOM:
 *   - Ready      → [data-testid="run-step"]
 *   - Running    → [data-testid="spinner"]
 *   - Completed  → [data-testid="checklist-badge"]
 *   - lastError  → [data-testid="error-block"]
 *
 * Secondary branches:
 *   - [class] on checklist-badge: text-gain (passed=true) vs text-loss (passed=false)
 *   - @if (checklistRows().length > 0) → app-data-table present vs absent
 *   - @if (artifactEntries().length > 0) → artifact-path elements present vs absent
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { StepReportPanelComponent } from './step-report-panel';
import {
  StepStatus,
  type ReportStepResult,
} from '../../core/models/pipeline-builder.model';

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

interface Harness {
  fixture: ComponentFixture<StepReportPanelComponent>;
  el: HTMLElement;
}

async function build(): Promise<Harness> {
  await configureTestBed({ imports: [StepReportPanelComponent], withHttp: false });
  const fixture = TestBed.createComponent(StepReportPanelComponent);
  fixture.componentRef.setInput('stepStatus', StepStatus.Pending);
  fixture.detectChanges();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

// ---------------------------------------------------------------------------
// Minimal result fixtures
// ---------------------------------------------------------------------------

const ARTIFACT_PATHS = {
  report_md: '/out/report.md',
  weights_csv: '/out/weights.csv',
  metrics_json: '/out/metrics.json',
  checklist_json: '/out/checklist.json',
  weights_diagnostic: '/out/weights_diagnostic.csv',
};

const RESULT_FULL_PASS: ReportStepResult = {
  pass_count: 17,
  checklist_total: 17,
  checklist_passed: true,
  checklist_rules: [
    { rule: 'Sharpe > 0.5', pass: true, measured: 0.82, target: '> 0.5' },
    { rule: 'Max drawdown < 30%', pass: true, measured: '18%', target: '< 30%' },
  ],
  metrics: {},
  artifact_paths: ARTIFACT_PATHS,
  chart_paths: [],
  output_dir: '/out',
};

const RESULT_FAIL_NO_ROWS: ReportStepResult = {
  pass_count: 14,
  checklist_total: 17,
  checklist_passed: false,
  checklist_rules: [],
  metrics: {},
  // Empty object so artifactEntries().length === 0, covering the @else arm.
  artifact_paths: {} as ReportStepResult['artifact_paths'],
  chart_paths: [],
  output_dir: '/out',
};

// ---------------------------------------------------------------------------
// Data-driven base-arm loop
// ---------------------------------------------------------------------------

interface StatusCase {
  label: string;
  status: StepStatus;
  result?: ReportStepResult;
  lastError?: string;
  testid: string;
}

const baseCases: StatusCase[] = [
  { label: 'Ready shows run button',         status: StepStatus.Ready,                                  testid: 'run-step'       },
  { label: 'Running shows spinner',           status: StepStatus.Running,                                testid: 'spinner'        },
  { label: 'Completed shows checklist-badge', status: StepStatus.Completed, result: RESULT_FAIL_NO_ROWS, testid: 'checklist-badge'},
  { label: 'lastError shows error-block',     status: StepStatus.Error,     lastError: 'report error',   testid: 'error-block'    },
];

describe('StepReportPanelComponent (render coverage)', () => {

  // ---- Base status arms -------------------------------------------------------

  baseCases.forEach((c) => {
    it(`when ${c.label}`, async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', c.status);
      if (c.result !== undefined) fixture.componentRef.setInput('result', c.result);
      if (c.lastError)            fixture.componentRef.setInput('lastError', c.lastError);
      fixture.detectChanges();

      expect(el.querySelector(`[data-testid="${c.testid}"]`)).not.toBeNull();
    });
  });

  // ---- checklist-badge [class] binding ----------------------------------------

  describe('checklist-badge [class] binding', () => {
    it('when checklist_passed is true, the badge carries text-gain', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_FULL_PASS);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="checklist-badge"]')!;
      expect(badge.className).toContain('text-gain');
    });

    it('when checklist_passed is false, the badge carries text-loss', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_FAIL_NO_ROWS);
      fixture.detectChanges();

      const badge = el.querySelector<HTMLElement>('[data-testid="checklist-badge"]')!;
      expect(badge.className).toContain('text-loss');
    });
  });

  // ---- @if (checklistRows().length > 0) ----------------------------------------

  describe('checklist rules table @if branch', () => {
    it('when checklist_rules is non-empty, app-data-table renders', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_FULL_PASS);
      fixture.detectChanges();

      expect(el.querySelector('app-data-table')).not.toBeNull();
    });

    it('when checklist_rules is empty, app-data-table is absent', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_FAIL_NO_ROWS);
      fixture.detectChanges();

      expect(el.querySelector('app-data-table')).toBeNull();
    });
  });

  // ---- @if (artifactEntries().length > 0) -------------------------------------

  describe('artifact paths @if branch', () => {
    it('when artifact_paths has entries, artifact-path elements render', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_FULL_PASS);
      fixture.detectChanges();

      const entries = el.querySelectorAll('[data-testid="artifact-path"]');
      expect(entries.length).toBe(Object.keys(ARTIFACT_PATHS).length);
    });

    it('when artifact_paths is empty, no artifact-path elements render', async () => {
      const { fixture, el } = await build();

      fixture.componentRef.setInput('stepStatus', StepStatus.Completed);
      fixture.componentRef.setInput('result', RESULT_FAIL_NO_ROWS);
      fixture.detectChanges();

      // artifact_paths is {} so artifactEntries().length === 0 — @if is false.
      const entries = el.querySelectorAll('[data-testid="artifact-path"]');
      expect(entries.length).toBe(0);
    });
  });
});
