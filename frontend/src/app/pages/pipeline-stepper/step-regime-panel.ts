import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule } from '@angular/forms';

import {
  type RegimeStepResult,
  StepStatus,
} from '../../models/pipeline-builder.model';
import { StatCardComponent } from '../../shared/stat-card/stat-card';

@Component({
  selector: 'app-step-regime-panel',
  imports: [ReactiveFormsModule, StatCardComponent],
  templateUrl: './step-regime-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StepRegimePanelComponent {
  stepStatus = input.required<StepStatus>();
  result = input<Record<string, unknown> | null>(null);
  lastError = input<string | null>(null);
  runStep = output<Record<string, unknown>>();

  readonly StepStatus = StepStatus;

  form = new FormGroup({
    enable_tilts: new FormControl<boolean>(true, { nonNullable: true }),
    persist_regime: new FormControl<boolean>(false, { nonNullable: true }),
  });

  regimeResult = computed<RegimeStepResult | null>(
    () => (this.result() as RegimeStepResult | null) ?? null,
  );

  tiltEntries = computed<Array<{ key: string; value: number }>>(() => {
    const r = this.regimeResult();
    if (!r) return [];
    return Object.entries(r.tilts).map(([key, value]) => ({ key, value }));
  });

  onSubmit(): void {
    this.runStep.emit({
      enable_tilts: this.form.controls.enable_tilts.value,
      persist_regime: this.form.controls.persist_regime.value,
    });
  }
}
