import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';
import { DecimalPipe } from '@angular/common';

import {
  type CleanReturnsStepResult,
  StepStatus,
} from '../../core/models/pipeline-builder.model';
import { FormatDatePipe } from '../../shared/pipes/format-date.pipe';
import { StatCardComponent } from '../../shared/stat-card/stat-card';

// Fixed preprocessing chain wired in optimizer/preprocessing/. Not
// user-configurable; rendered as read-only info card per Cycle 5 spec.
export const PREPROCESSING_PIPELINE = [
  'DataValidator',
  'OutlierTreater',
  'SectorImputer',
  'RegressionImputer',
] as const;

@Component({
  selector: 'app-step-clean-returns-panel',
  imports: [DecimalPipe, FormatDatePipe, StatCardComponent],
  templateUrl: './step-clean-returns-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepCleanReturnsPanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  runStep = output<Record<string, unknown>>();

  readonly StepStatus = StepStatus;
  readonly preprocessors = PREPROCESSING_PIPELINE;

  cleanResult = computed<CleanReturnsStepResult | null>(
    () => (this.result() as CleanReturnsStepResult | null) ?? null,
  );

  onRun(): void {
    this.runStep.emit({});
  }
}
