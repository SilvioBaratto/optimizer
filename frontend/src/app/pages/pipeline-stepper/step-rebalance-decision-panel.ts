import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';

import {
  type RebalanceDecisionStepResult,
  StepStatus,
} from '../../models/pipeline-builder.model';
import { StatCardComponent } from '../../shared/stat-card/stat-card';

@Component({
  selector: 'app-step-rebalance-decision-panel',
  imports: [StatCardComponent],
  templateUrl: './step-rebalance-decision-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepRebalanceDecisionPanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  runStep = output<Record<string, unknown>>();

  readonly StepStatus = StepStatus;

  rebalanceResult = computed<RebalanceDecisionStepResult | null>(
    () => (this.result() as RebalanceDecisionStepResult | null) ?? null,
  );

  onRun(): void {
    this.runStep.emit({});
  }
}
