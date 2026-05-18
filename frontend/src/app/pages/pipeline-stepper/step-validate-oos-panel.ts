import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

import { StepStatus } from '../../models/pipeline-builder.model';

@Component({
  selector: 'app-step-validate-oos-panel',
  template: `<div class="text-data-sm text-text-tertiary">step:validate_oos (stub)</div>`,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepValidateOosPanelComponent {
  stepStatus = input<StepStatus>(StepStatus.Pending);
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  runStep = output<Record<string, unknown>>();
}
