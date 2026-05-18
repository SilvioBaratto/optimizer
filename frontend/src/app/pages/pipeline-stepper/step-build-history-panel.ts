import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';
import { DecimalPipe } from '@angular/common';
import {
  FormControl,
  FormGroup,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';

import {
  type BuildHistoryStepResult,
  StepStatus,
} from '../../models/pipeline-builder.model';
import { StatCardComponent } from '../../shared/stat-card/stat-card';

// Emits `market_proxy` per Cycle 5 spec. Backend currently reads
// `market_proxy_ticker` (api/app/services/pipeline_builder/steps.py:486);
// param-key alignment is deferred to a backend schema cycle. Unknown
// params are ignored, so this is forward-safe.

@Component({
  selector: 'app-step-build-history-panel',
  imports: [DecimalPipe, ReactiveFormsModule, StatCardComponent],
  templateUrl: './step-build-history-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepBuildHistoryPanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  runStep = output<Record<string, unknown>>();

  readonly StepStatus = StepStatus;

  form = new FormGroup({
    market_proxy: new FormControl<string>('URTH', {
      nonNullable: true,
      validators: [Validators.required],
    }),
  });

  buildResult = computed<BuildHistoryStepResult | null>(
    () => (this.result() as BuildHistoryStepResult | null) ?? null,
  );

  onSubmit(): void {
    if (!this.form.valid) return;
    this.runStep.emit({ market_proxy: this.form.controls.market_proxy.value });
  }
}
