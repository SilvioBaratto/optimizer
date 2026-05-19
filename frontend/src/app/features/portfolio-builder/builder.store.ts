import { Injectable, type Signal, computed, signal } from '@angular/core';

import {
  type PipelineStepId,
  type RunLevelConfig,
  StepStatus,
} from '../../models/pipeline-builder.model';
import type { StepParamsConfig } from '../../pages/pipeline-stepper/step-params.model';

export interface BuilderSummary {
  readonly stage: PipelineStepId | null;
  readonly status: StepStatus;
  readonly hasConfig: boolean;
}

@Injectable()
export class BuilderStore {
  private readonly _config = signal<RunLevelConfig | null>(null);
  private readonly _stepParams = signal<StepParamsConfig | null>(null);
  private readonly _currentStage = signal<PipelineStepId | null>(null);
  private readonly _status = signal<StepStatus>(StepStatus.Pending);

  readonly config: Signal<RunLevelConfig | null> = this._config.asReadonly();
  readonly stepParams: Signal<StepParamsConfig | null> =
    this._stepParams.asReadonly();
  readonly currentStage: Signal<PipelineStepId | null> =
    this._currentStage.asReadonly();
  readonly status: Signal<StepStatus> = this._status.asReadonly();

  readonly summary: Signal<BuilderSummary> = computed(() => ({
    stage: this._currentStage(),
    status: this._status(),
    hasConfig: this._config() !== null,
  }));

  setConfig(value: RunLevelConfig | null): void {
    this._config.set(value);
  }

  setStepParams(value: StepParamsConfig | null): void {
    this._stepParams.set(value);
  }

  setStage(value: PipelineStepId | null): void {
    this._currentStage.set(value);
  }

  setStatus(value: StepStatus): void {
    this._status.set(value);
  }

  reset(): void {
    this._config.set(null);
    this._stepParams.set(null);
    this._currentStage.set(null);
    this._status.set(StepStatus.Pending);
  }
}
