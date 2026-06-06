import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';

import {
  type ReportStepResult,
  StepStatus,
} from '../../core/models/pipeline-builder.model';
import {
  DataTableComponent,
  type TableColumn,
} from '../../shared/data-table/data-table';
import { StatCardComponent } from '../../shared/stat-card/stat-card';

const PASS_BADGE_MAP = {
  true: { value: 'PASS', colorClass: 'bg-gain/10 text-gain border border-gain/40' },
  false: { value: 'FAIL', colorClass: 'bg-loss/10 text-loss border border-loss/40' },
} as const;

@Component({
  selector: 'app-step-report-panel',
  imports: [DataTableComponent, StatCardComponent],
  templateUrl: './step-report-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepReportPanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  runStep = output<Record<string, unknown>>();

  readonly StepStatus = StepStatus;

  readonly checklistColumns: TableColumn[] = [
    { key: 'rule', label: 'Rule', type: 'text' },
    {
      key: 'pass',
      label: 'Result',
      type: 'badge',
      badgeMap: { ...PASS_BADGE_MAP },
    },
    { key: 'measured', label: 'Measured', align: 'right' },
    { key: 'target', label: 'Target', align: 'right' },
  ];

  reportResult = computed<ReportStepResult | null>(
    () => (this.result() as ReportStepResult | null) ?? null,
  );

  checklistRows = computed<Record<string, unknown>[]>(() => {
    const r = this.reportResult();
    return r ? (r.checklist_rules as unknown as Record<string, unknown>[]) : [];
  });

  artifactEntries = computed<Array<{ key: string; value: string }>>(() => {
    const r = this.reportResult();
    if (!r) return [];
    return Object.entries(r.artifact_paths).map(([key, value]) => ({ key, value }));
  });

  onRun(): void {
    this.runStep.emit({});
  }
}
