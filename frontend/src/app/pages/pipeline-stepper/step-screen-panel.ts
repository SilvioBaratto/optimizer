import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';
import { DecimalPipe } from '@angular/common';
import {
  AbstractControl,
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  ValidationErrors,
  Validators,
} from '@angular/forms';

import {
  type ScreenStepResult,
  StepStatus,
} from '../../models/pipeline-builder.model';
import { StatCardComponent } from '../../shared/stat-card/stat-card';

export const SCREEN_PRESETS = [
  'developed_markets',
  'broad_universe',
  'small_cap',
  'large_cap',
] as const;

export type ScreenPreset = (typeof SCREEN_PRESETS)[number];

function presetValidator(c: AbstractControl): ValidationErrors | null {
  return (SCREEN_PRESETS as readonly string[]).includes(c.value)
    ? null
    : { invalidPreset: true };
}

@Component({
  selector: 'app-step-screen-panel',
  imports: [DecimalPipe, ReactiveFormsModule, StatCardComponent],
  templateUrl: './step-screen-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepScreenPanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  runStep = output<Record<string, unknown>>();

  readonly StepStatus = StepStatus;
  readonly presets = SCREEN_PRESETS;

  form = new FormGroup({
    preset: new FormControl<ScreenPreset>('developed_markets', {
      nonNullable: true,
      validators: [Validators.required, presetValidator],
    }),
  });

  screenResult = computed<ScreenStepResult | null>(
    () => (this.result() as ScreenStepResult | null) ?? null,
  );

  onSubmit(): void {
    if (!this.form.valid) return;
    this.runStep.emit({ preset: this.form.controls.preset.value });
  }
}
