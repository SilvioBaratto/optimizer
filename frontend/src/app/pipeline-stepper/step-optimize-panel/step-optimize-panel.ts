import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';
import { DecimalPipe } from '@angular/common';

import {
  type OptimizeStepResult,
  StepStatus,
} from '../../core/models/pipeline-builder.model';
import {
  DataTableComponent,
  type TableColumn,
} from '../../shared/data-table/data-table';
import { StatCardComponent } from '../../shared/stat-card/stat-card';

function formatDecimals(decimals: number): (v: unknown) => string {
  return (v) => (typeof v === 'number' ? v.toFixed(decimals) : '—');
}

@Component({
  selector: 'app-step-optimize-panel',
  imports: [DecimalPipe, DataTableComponent, StatCardComponent],
  templateUrl: './step-optimize-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepOptimizePanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  runStep = output<Record<string, unknown>>();

  readonly StepStatus = StepStatus;

  readonly weightColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', type: 'text', sortable: true },
    {
      key: 'weight',
      label: 'Weight',
      align: 'right',
      format: (v) => (typeof v === 'number' ? (v * 100).toFixed(2) + '%' : '—'),
    },
  ];

  optimizeResult = computed<OptimizeStepResult | null>(
    () => (this.result() as OptimizeStepResult | null) ?? null,
  );

  weightRows = computed<Record<string, unknown>[]>(() => {
    const r = this.optimizeResult();
    return r ? (r.weights as unknown as Record<string, unknown>[]) : [];
  });

  sectorEntries = computed<Array<{ key: string; value: number }>>(() => {
    const r = this.optimizeResult();
    if (!r) return [];
    return Object.entries(r.sector_breakdown).map(([key, value]) => ({ key, value }));
  });

  countryEntries = computed<Array<{ key: string; value: number }>>(() => {
    const r = this.optimizeResult();
    if (!r) return [];
    return Object.entries(r.country_breakdown).map(([key, value]) => ({ key, value }));
  });

  onRun(): void {
    this.runStep.emit({});
  }
}
