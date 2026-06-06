import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';

import {
  type ValidateIsStepResult,
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

const SIGNIFICANT_BADGE_MAP = {
  true: { value: 'significant', colorClass: 'bg-gain/10 text-gain border border-gain/40' },
  false: { value: '—', colorClass: 'bg-surface-inset text-text-tertiary border border-border' },
} as const;

@Component({
  selector: 'app-step-validate-is-panel',
  imports: [DataTableComponent, StatCardComponent],
  templateUrl: './step-validate-is-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepValidateIsPanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  runStep = output<Record<string, unknown>>();

  readonly StepStatus = StepStatus;

  readonly icColumns: TableColumn[] = [
    { key: 'factor_name', label: 'Factor', type: 'text', sortable: true },
    { key: 'mean_ic', label: 'Mean IC', align: 'right', format: formatDecimals(4) },
    { key: 'ic_std', label: 'IC stdev', align: 'right', format: formatDecimals(4) },
    {
      key: 't_stat',
      label: 'NW t-stat',
      align: 'right',
      colorBySign: true,
      format: formatDecimals(4),
    },
    { key: 'p_value', label: 'p-value', align: 'right', format: formatDecimals(4) },
    {
      key: 'significant',
      label: 'Significant',
      type: 'badge',
      badgeMap: { ...SIGNIFICANT_BADGE_MAP },
    },
  ];

  isResult = computed<ValidateIsStepResult | null>(
    () => (this.result() as ValidateIsStepResult | null) ?? null,
  );

  icRows = computed<Record<string, unknown>[]>(() => {
    const r = this.isResult();
    return r ? (r.ic_results as unknown as Record<string, unknown>[]) : [];
  });

  onRun(): void {
    this.runStep.emit({});
  }
}
