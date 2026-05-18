import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';

import {
  type PersistStepResult,
  StepStatus,
} from '../../models/pipeline-builder.model';
import { StatCardComponent } from '../../shared/stat-card/stat-card';

@Component({
  selector: 'app-step-persist-panel',
  imports: [StatCardComponent],
  templateUrl: './step-persist-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepPersistPanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  sessionId = input<string | null>(null);
  runStep = output<Record<string, unknown>>();

  readonly StepStatus = StepStatus;

  persistResult = computed<PersistStepResult | null>(
    () => (this.result() as PersistStepResult | null) ?? null,
  );

  onRun(): void {
    this.runStep.emit({});
  }
}
