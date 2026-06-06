import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';

import {
  type CostStepResult,
  StepStatus,
} from '../../core/models/pipeline-builder.model';
import {
  DataTableComponent,
  type TableColumn,
} from '../../shared/data-table/data-table';
import { StatCardComponent } from '../../shared/stat-card/stat-card';

@Component({
  selector: 'app-step-cost-panel',
  imports: [DataTableComponent, StatCardComponent],
  templateUrl: './step-cost-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepCostPanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  runStep = output<Record<string, unknown>>();

  readonly StepStatus = StepStatus;

  readonly costColumns: TableColumn[] = [
    { key: 'country', label: 'Country', type: 'text', sortable: true },
    {
      key: 'stamp',
      label: 'Stamp (bps)',
      align: 'right',
      format: (v) => (typeof v === 'number' ? v.toFixed(2) : '—'),
    },
    {
      key: 'spread',
      label: 'Spread (bps)',
      align: 'right',
      format: (v) => (typeof v === 'number' ? v.toFixed(2) : '—'),
    },
    {
      key: 'fx',
      label: 'FX (bps)',
      align: 'right',
      format: (v) => (typeof v === 'number' ? v.toFixed(2) : '—'),
    },
    {
      key: 'total',
      label: 'Total (bps)',
      align: 'right',
      format: (v) => (typeof v === 'number' ? v.toFixed(2) : '—'),
    },
  ];

  costResult = computed<CostStepResult | null>(
    () => (this.result() as CostStepResult | null) ?? null,
  );

  costRows = computed<Record<string, unknown>[]>(() => {
    const r = this.costResult();
    return r ? (r.per_country as unknown as Record<string, unknown>[]) : [];
  });

  onRun(): void {
    this.runStep.emit({});
  }
}
