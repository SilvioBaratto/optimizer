import { ChangeDetectionStrategy, Component, output } from '@angular/core';
import {
  AbstractControl,
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  ValidationErrors,
  Validators,
} from '@angular/forms';

import type {
  BaseCurrency,
  RunLevelConfig,
} from '../../models/pipeline-builder.model';

function dateOrderValidator(group: AbstractControl): ValidationErrors | null {
  const start = group.get('start_date')?.value as string | null;
  const end = group.get('end_date')?.value as string | null;
  if (start && end && end < start) return { dateOrder: true };
  return null;
}

@Component({
  selector: 'app-run-config-panel',
  imports: [ReactiveFormsModule],
  templateUrl: './run-config-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class RunConfigPanelComponent {
  readonly currencies: readonly BaseCurrency[] = ['EUR', 'GBP', 'USD'];

  configSubmit = output<RunLevelConfig>();

  form = new FormGroup(
    {
      rebalance_freq: new FormControl(63, {
        nonNullable: true,
        validators: [Validators.required, Validators.min(1)],
      }),
      n_selected: new FormControl(20, {
        nonNullable: true,
        validators: [Validators.required, Validators.min(15), Validators.max(30)],
      }),
      cost_bps: new FormControl(10, {
        nonNullable: true,
        validators: [Validators.required, Validators.min(0)],
      }),
      tax_rate: new FormControl(0.26, {
        nonNullable: true,
        validators: [Validators.required, Validators.min(0), Validators.max(1)],
      }),
      base_currency: new FormControl<BaseCurrency>('EUR', { nonNullable: true }),
      robust: new FormControl(false, { nonNullable: true }),
      persist: new FormControl(false, { nonNullable: true }),
      start_date: new FormControl<string | null>(null),
      end_date: new FormControl<string | null>(null),
      seed: new FormControl(42, {
        nonNullable: true,
        validators: [Validators.required, Validators.min(0)],
      }),
    },
    { validators: dateOrderValidator },
  );

  onSubmit(): void {
    if (!this.form.valid) return;
    this.configSubmit.emit(this.form.getRawValue());
  }
}
