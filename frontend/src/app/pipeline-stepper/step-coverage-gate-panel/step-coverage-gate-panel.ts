import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';
import {
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';

import {
  type CoverageGateStepResult,
  StepStatus,
} from '../../core/models/pipeline-builder.model';
import { StatCardComponent } from '../../shared/stat-card/stat-card';

// `abortGateError` input is accepted from the outlet (#723 wiring) but
// not rendered here. The stepper already shows a top-level banner above
// the outlet for gate failures (`pipeline-stepper.html` `abortGateError`
// block); duplicating it inside the panel would show the same warning
// twice (#728 comment override).

@Component({
  selector: 'app-step-coverage-gate-panel',
  imports: [ReactiveFormsModule, StatCardComponent],
  templateUrl: './step-coverage-gate-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepCoverageGatePanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  abortGateError = input<string | null>(null);
  runStep = output<{ min_factors: number }>();

  readonly StepStatus = StepStatus;

  form = new FormGroup({
    min_factors: new FormControl<number>(2, {
      nonNullable: true,
      validators: [Validators.required, Validators.min(1)],
    }),
  });

  coverageResult = computed<CoverageGateStepResult | null>(
    () => (this.result() as CoverageGateStepResult | null) ?? null,
  );

  onSubmit(): void {
    if (!this.form.valid) return;
    this.runStep.emit({ min_factors: this.form.controls.min_factors.value });
  }
}
