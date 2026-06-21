import { Component, signal, output, ChangeDetectionStrategy } from '@angular/core';
import { ReactiveFormsModule, FormControl, FormGroup } from '@angular/forms';

import { OPTIMIZE_METHODS, type MethodId } from '../../optimize/optimize-methods';
import type { OptimizerType } from '../../core/models/optimization.model';

export interface OptimizerRunRequest {
  optimizerType: OptimizerType;
  config: Record<string, unknown>;
}

@Component({
  selector: 'app-optimizer-panel',
  imports: [ReactiveFormsModule],
  templateUrl: './optimizer-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class OptimizerPanelComponent {
  readonly methods = OPTIMIZE_METHODS;
  readonly selectedMethodId = signal<MethodId>('max_sharpe');
  readonly runOptimization = output<OptimizerRunRequest>();

  readonly constraintsForm = new FormGroup({
    longOnly:        new FormControl(true,  { nonNullable: true }),
    stockCap:        new FormControl(0.10,  { nonNullable: true }),
    minPositionSize: new FormControl(0.02,  { nonNullable: true }),
    budget:          new FormControl(false, { nonNullable: true }),
  });

  selectMethod(id: string): void {
    this.selectedMethodId.set(id as MethodId);
  }

  emitRun(): void {
    const { stockCap, budget } = this.constraintsForm.getRawValue();
    this.runOptimization.emit({
      optimizerType: 'mean_risk',
      config: this.buildConfig(this.selectedMethodId(), stockCap, budget),
    });
  }

  private buildConfig(id: MethodId, stockCap: number, budget: boolean): Record<string, unknown> {
    const method = this.methods.find(m => m.id === id)!;
    const cfg: Record<string, unknown> = {
      objective: method.config.objective,
      risk_measure: method.config.risk_measure,
      max_weights: stockCap,
    };
    if (method.config.l2_coef !== undefined) cfg['l2_coef'] = method.config.l2_coef;
    if (budget) cfg['budget'] = 1.0;
    return cfg;
  }
}
