import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import {
  AbstractControl,
  FormArray,
  FormBuilder,
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  ValidationErrors,
  ValidatorFn,
  Validators,
} from '@angular/forms';
import { toSignal } from '@angular/core/rxjs-interop';

import { FormatService } from '../../core/services/format.service';
import {
  FACTOR_TYPES,
  type FactorExposureConstraintsApiResponse,
  type FactorExposureConstraintsRequest,
  type FactorType,
} from '../factor.model';

interface RowControls {
  factor: FormControl<FactorType>;
  min: FormControl<number>;
  max: FormControl<number>;
}

const WINDOW_DAYS = 365;
const COPIED_RESET_MS = 2000;

@Component({
  selector: 'app-exposure-constraints-panel',
  imports: [ReactiveFormsModule],
  templateUrl: './exposure-constraints-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ExposureConstraintsPanelComponent {
  readonly tickers = input<string[]>([]);
  readonly result = input<FactorExposureConstraintsApiResponse | null>(null);
  readonly loading = input<boolean>(false);
  readonly errorMessage = input<string | null>(null);

  readonly runConstraints = output<FactorExposureConstraintsRequest>();

  readonly factors = FACTOR_TYPES;
  private readonly fmt = inject(FormatService);
  private readonly fb = inject(FormBuilder).nonNullable;
  readonly form = this.buildForm();
  readonly copied = signal<boolean>(false);

  private readonly statusChanges = toSignal(this.form.statusChanges, {
    initialValue: this.form.status,
  });

  readonly canSubmit = computed(() => this.statusChanges() === 'VALID' && !this.loading());

  get rows(): FormArray<FormGroup<RowControls>> {
    return this.form.controls.rows;
  }

  addRow(): void {
    const nextFactor = this.firstUnusedFactor() ?? FACTOR_TYPES[0];
    this.rows.push(this.buildRow(nextFactor));
  }

  removeRow(index: number): void {
    if (this.rows.length <= 1) return;
    this.rows.removeAt(index);
  }

  submit(): void {
    if (this.form.invalid) return;
    this.runConstraints.emit(this.buildPayload());
  }

  copyJson(): void {
    const result = this.result();
    if (!result) return;
    const payload = JSON.stringify(
      {
        left_inequality: result.left_inequality,
        right_inequality: result.right_inequality,
      },
      null,
      2,
    );
    // Fire-and-forget: the clipboard API returns a promise but we give the user
    // immediate visual feedback via `copied` rather than waiting for the
    // microtask to resolve.
    void navigator.clipboard.writeText(payload);
    this.copied.set(true);
    setTimeout(() => this.copied.set(false), COPIED_RESET_MS);
  }

  private buildForm(): FormGroup<{ rows: FormArray<FormGroup<RowControls>> }> {
    return this.fb.group(
      {
        rows: this.fb.array<FormGroup<RowControls>>([this.buildRow(FACTOR_TYPES[0])], {
          validators: [Validators.required],
        }),
      },
      { validators: [uniqueFactorValidator] },
    );
  }

  private buildRow(factor: FactorType): FormGroup<RowControls> {
    return this.fb.group<RowControls>(
      {
        factor: this.fb.control<FactorType>(factor),
        min: this.fb.control(-0.5),
        max: this.fb.control(0.5),
      },
      { validators: [maxGtMinValidator] },
    );
  }

  private firstUnusedFactor(): FactorType | null {
    const used = new Set(this.rows.controls.map((r) => r.controls.factor.value));
    for (const f of FACTOR_TYPES) {
      if (!used.has(f)) return f;
    }
    return null;
  }

  private buildPayload(): FactorExposureConstraintsRequest {
    const end = this.fmt.todayIso();
    const bounds: Record<string, [number, number]> = {};
    for (const row of this.rows.controls) {
      const v = row.getRawValue();
      bounds[v.factor] = [v.min, v.max];
    }
    return {
      tickers: this.tickers(),
      start_date: daysBeforeIso(end, WINDOW_DAYS),
      end_date: end,
      bounds,
    };
  }
}

function maxGtMinValidator(control: AbstractControl): ValidationErrors | null {
  const group = control as FormGroup<RowControls>;
  const min = group.controls.min?.value;
  const max = group.controls.max?.value;
  if (typeof min !== 'number' || typeof max !== 'number') return null;
  return max > min ? null : { maxGtMin: true };
}

const uniqueFactorValidator: ValidatorFn = (control: AbstractControl) => {
  const rows = (control as FormGroup).controls['rows'] as FormArray<FormGroup<RowControls>>;
  const values = rows.controls.map((r) => r.controls.factor.value);
  const unique = new Set(values);
  return unique.size === values.length ? null : { uniqueFactor: true };
};

function daysBeforeIso(iso: string, days: number): string {
  const d = new Date(iso);
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}
