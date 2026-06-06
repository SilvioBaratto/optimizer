import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';

import {
  type OosResultItem,
  type ValidateOosStepResult,
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
  selector: 'app-step-validate-oos-panel',
  imports: [DataTableComponent, StatCardComponent],
  templateUrl: './step-validate-oos-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepValidateOosPanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  runStep = output<Record<string, unknown>>();

  readonly StepStatus = StepStatus;

  readonly oosColumns: TableColumn[] = [
    { key: 'factor_name', label: 'Factor', type: 'text', sortable: true },
    {
      key: 'oos_mean_ic',
      label: 'OOS Mean IC',
      align: 'right',
      colorBySign: true,
      format: formatDecimals(4),
    },
    {
      key: 'oos_icir',
      label: 'OOS ICIR',
      align: 'right',
      colorBySign: true,
      format: formatDecimals(4),
    },
  ];

  oosResult = computed<ValidateOosStepResult | null>(
    () => (this.result() as ValidateOosStepResult | null) ?? null,
  );

  oosRows = computed<Record<string, unknown>[]>(() => {
    const r = this.oosResult();
    return r ? (r.oos_results as unknown as Record<string, unknown>[]) : [];
  });

  onRun(): void {
    this.runStep.emit({});
  }
}
