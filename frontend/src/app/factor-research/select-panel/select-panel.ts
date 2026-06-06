import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
  output,
} from '@angular/core';
import {
  FormBuilder,
  FormControl,
  FormGroup,
  ReactiveFormsModule,
} from '@angular/forms';
import { toSignal } from '@angular/core/rxjs-interop';

import { FormatService } from '../../core/services/format.service';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import {
  deriveSelectRows,
  type FactorSelectApiResponse,
  type FactorSelectRequest,
  type SelectRow,
  type SelectionMethod,
} from '../factor.model';

interface SelectFormControls {
  method: FormControl<SelectionMethod>;
  targetCount: FormControl<number>;
  targetQuantile: FormControl<number>;
  bufferEnabled: FormControl<boolean>;
  bufferFraction: FormControl<number>;
  sectorBalance: FormControl<boolean>;
}

const WINDOW_DAYS = 365;

@Component({
  selector: 'app-select-panel',
  imports: [ReactiveFormsModule, DataTableComponent],
  templateUrl: './select-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SelectPanelComponent {
  private readonly fmt = inject(FormatService);

  readonly tickers = input<string[]>([]);
  readonly selectResult = input<FactorSelectApiResponse | null>(null);
  readonly loading = input<boolean>(false);
  readonly errorMessage = input<string | null>(null);

  readonly runSelect = output<FactorSelectRequest>();

  readonly form: FormGroup<SelectFormControls> = this.buildForm();
  readonly method = toSignal(this.form.controls.method.valueChanges, {
    initialValue: this.form.controls.method.value,
  });
  readonly bufferEnabled = toSignal(this.form.controls.bufferEnabled.valueChanges, {
    initialValue: this.form.controls.bufferEnabled.value,
  });

  readonly columns: TableColumn[] = [
    { key: 'status', label: 'Status', sortable: true },
    { key: 'ticker', label: 'Ticker', sortable: true },
  ];

  readonly selectRows = computed<SelectRow[]>(() => deriveSelectRows(this.selectResult()));

  readonly tableRows = computed<Record<string, unknown>[]>(() =>
    this.selectRows().map((row) => ({ ...row })),
  );

  readonly summary = computed(() => {
    const res = this.selectResult();
    if (res === null) return null;
    return {
      count: res.count,
      turnover: formatTurnover(res.turnover),
    };
  });

  run(): void {
    this.runSelect.emit(this.buildPayload());
  }

  private buildPayload(): FactorSelectRequest {
    const v = this.form.getRawValue();
    const end = this.fmt.todayIso();
    const payload: FactorSelectRequest = {
      tickers: this.tickers(),
      start_date: daysBeforeIso(end, WINDOW_DAYS),
      end_date: end,
      method: v.method,
      sector_balance: v.sectorBalance,
    };
    if (v.method === 'fixed_count') {
      payload.target_count = v.targetCount;
    } else {
      payload.target_quantile = v.targetQuantile;
    }
    if (v.bufferEnabled) {
      payload.buffer_fraction = v.bufferFraction;
    }
    return payload;
  }

  private buildForm(): FormGroup<SelectFormControls> {
    const fb = inject(FormBuilder).nonNullable;
    return fb.group<SelectFormControls>({
      method: fb.control<SelectionMethod>('fixed_count'),
      targetCount: fb.control(30),
      targetQuantile: fb.control(0.2),
      bufferEnabled: fb.control(false),
      bufferFraction: fb.control(0.1),
      sectorBalance: fb.control(false),
    });
  }
}

function formatTurnover(turnover: number | null): string {
  if (turnover === null) return '—';
  return `${(turnover * 100).toFixed(2)}%`;
}

function daysBeforeIso(iso: string, days: number): string {
  const d = new Date(iso);
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}
