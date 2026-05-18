import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';
import { DecimalPipe } from '@angular/common';

import {
  type LoadStepResult,
  StepStatus,
} from '../../models/pipeline-builder.model';
import { FormatDatePipe } from '../../shared/pipes/format-date.pipe';
import { StatCardComponent } from '../../shared/stat-card/stat-card';

// `result` input is widened to `Record<string, unknown> | null` so the
// outlet's generic `activeStepResult` computed in pipeline-stepper.ts
// type-checks under strictTemplates. The internal `loadResult` computed
// narrows to `LoadStepResult` at the consumption point (per #722's
// cast-at-boundary contract).

@Component({
  selector: 'app-step-load-panel',
  imports: [DecimalPipe, FormatDatePipe, StatCardComponent],
  templateUrl: './step-load-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepLoadPanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  runStep = output<Record<string, unknown>>();

  readonly StepStatus = StepStatus;

  loadResult = computed<LoadStepResult | null>(
    () => (this.result() as LoadStepResult | null) ?? null,
  );

  onRun(): void {
    this.runStep.emit({});
  }
}
