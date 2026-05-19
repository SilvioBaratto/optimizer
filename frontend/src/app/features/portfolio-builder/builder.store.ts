import { Injectable, type Signal, computed, signal } from '@angular/core';

import {
  type PipelineStepId,
  type RunLevelConfig,
  StepStatus,
} from '../../models/pipeline-builder.model';
import type { StepParamsConfig } from '../../pages/pipeline-stepper/step-params.model';
import type { BuilderStage } from './builder-stage';

export interface BuilderSummary {
  readonly stage: BuilderStage | null;
  readonly status: StepStatus;
  readonly hasConfig: boolean;
}

type StepResultEntry = Record<string, unknown> | null;

@Injectable()
export class BuilderStore {
  private readonly _config = signal<RunLevelConfig | null>(null);
  private readonly _stepParams = signal<StepParamsConfig | null>(null);
  private readonly _currentStage = signal<BuilderStage | null>(null);
  private readonly _status = signal<StepStatus>(StepStatus.Pending);
  private readonly _stepResults = signal<Map<PipelineStepId, StepResultEntry>>(
    new Map(),
  );
  private readonly _stepStatuses = signal<Map<PipelineStepId, StepStatus>>(
    new Map(),
  );

  readonly config: Signal<RunLevelConfig | null> = this._config.asReadonly();
  readonly stepParams: Signal<StepParamsConfig | null> =
    this._stepParams.asReadonly();
  readonly currentStage: Signal<BuilderStage | null> =
    this._currentStage.asReadonly();
  readonly status: Signal<StepStatus> = this._status.asReadonly();
  readonly stepResults: Signal<Map<PipelineStepId, StepResultEntry>> =
    this._stepResults.asReadonly();
  readonly stepStatuses: Signal<Map<PipelineStepId, StepStatus>> =
    this._stepStatuses.asReadonly();

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

  setStage(value: BuilderStage | null): void {
    this._currentStage.set(value);
  }

  setStatus(value: StepStatus): void {
    this._status.set(value);
  }

  setStepResult(stepId: PipelineStepId, result: StepResultEntry): void {
    this._stepResults.update((m) => new Map(m).set(stepId, result));
  }

  setStepStatus(stepId: PipelineStepId, status: StepStatus): void {
    this._stepStatuses.update((m) => new Map(m).set(stepId, status));
  }

  reset(): void {
    this._config.set(null);
    this._stepParams.set(null);
    this._currentStage.set(null);
    this._status.set(StepStatus.Pending);
    this._stepResults.set(new Map());
    this._stepStatuses.set(new Map());
  }
}
