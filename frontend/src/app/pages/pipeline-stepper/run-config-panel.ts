import { ChangeDetectionStrategy, Component, output } from '@angular/core';
import {
  AbstractControl,
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  ValidationErrors,
  Validators,
} from '@angular/forms';

import type { BaseCurrency } from '../../models/pipeline-builder.model';
import type { PipelineConfigSubmit } from './step-params.model';
import { SCREEN_PRESETS, type ScreenPreset } from './step-screen-panel';

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
  readonly presets = SCREEN_PRESETS;

  configSubmit = output<PipelineConfigSubmit>();

  form = new FormGroup(
    {
      // ---- run-level config (RunLevelConfig) ----
      // Monthly (21) default: quarterly (63) yields too few OOS folds on
      // ~3y data → factor coverage gate fails. See research RUNBOOK.
      rebalance_freq: new FormControl(21, {
        nonNullable: true,
        validators: [Validators.required, Validators.min(1)],
      }),
      // Backend _validate_n_selected enforces the Cycle-2 §6.1 band [25, 50];
      // values outside it raise ValueError before the pipeline runs.
      n_selected: new FormControl(25, {
        nonNullable: true,
        validators: [Validators.required, Validators.min(25), Validators.max(50)],
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

      // ---- per-step params (StepParamsConfig) ----
      include_delisted: new FormControl(true, { nonNullable: true }),
      macro_country: new FormControl('USA', {
        nonNullable: true,
        validators: [Validators.required],
      }),
      preset: new FormControl<ScreenPreset>('developed_markets', {
        nonNullable: true,
        validators: [Validators.required],
      }),
      market_proxy_ticker: new FormControl('URTH', {
        nonNullable: true,
        validators: [Validators.required],
      }),
      min_factors: new FormControl(2, {
        nonNullable: true,
        validators: [Validators.required, Validators.min(1)],
      }),
      enable_tilts: new FormControl(true, { nonNullable: true }),
      persist_regime: new FormControl(false, { nonNullable: true }),
    },
    { validators: dateOrderValidator },
  );

  onSubmit(): void {
    if (!this.form.valid) return;
    const {
      include_delisted,
      macro_country,
      preset,
      market_proxy_ticker,
      min_factors,
      enable_tilts,
      persist_regime,
      ...config
    } = this.form.getRawValue();

    this.configSubmit.emit({
      config,
      steps: {
        load: { include_delisted, macro_country },
        screen: { preset },
        build_history: { market_proxy_ticker },
        coverage_gate: { min_factors },
        regime: { enable_tilts, persist_regime },
      },
    });
  }
}
