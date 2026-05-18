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

// Emits `market_proxy_ticker` — matches BuildHistoryStepRequest and the
// `params["market_proxy_ticker"]` key read by step_build_history
// (api/app/services/pipeline_builder/steps.py:486).

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
    market_proxy_ticker: new FormControl<string>('URTH', {
      nonNullable: true,
      validators: [Validators.required],
    }),
  });

  buildResult = computed<BuildHistoryStepResult | null>(
    () => (this.result() as BuildHistoryStepResult | null) ?? null,
  );

  onSubmit(): void {
    if (!this.form.valid) return;
    this.runStep.emit({
      market_proxy_ticker: this.form.controls.market_proxy_ticker.value,
    });
  }
}
