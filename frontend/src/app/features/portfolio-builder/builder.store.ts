import {
  DestroyRef,
  Injectable,
  InjectionToken,
  Injector,
  type Signal,
  computed,
  inject,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import type { BaseToggle } from '../../models/drift.model';
import {
  type PipelineStepId,
  type RunLevelConfig,
  StepStatus,
} from '../../models/pipeline-builder.model';
import type { StepParamsConfig } from '../../pages/pipeline-stepper/step-params.model';
import { PipelineBuilderApiService } from '../../services/pipeline-builder-api.service';
import type {
  BuilderBacktest,
  BuilderDrift,
  BuilderDriftDiagnostics,
  BuilderDriftStatus,
  BuilderFrontier,
  BuilderResult,
  BuilderResultStatus,
  BuilderTrades,
} from './builder-result.model';
import { BuilderResultService } from './builder-result.service';
import type { BuilderStage } from './builder-stage';

export type CanvasView = 'donut' | 'bars' | 'frontier' | 'backtest' | 'drift';

export interface DriftRunner {
  runExplicit(): void;
}

export const BUILDER_DRIFT_SERVICE = new InjectionToken<DriftRunner>(
  'BUILDER_DRIFT_SERVICE',
);

export interface BuilderSummary {
  readonly stage: BuilderStage | null;
  readonly status: StepStatus;
  readonly hasConfig: boolean;
}

type StepResultEntry = Record<string, unknown> | null;

@Injectable()
export class BuilderStore {
  private readonly api = inject(PipelineBuilderApiService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly injector = inject(Injector);

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
  private readonly _sessionId = signal<string | null>(null);
  private readonly _resultStatus = signal<BuilderResultStatus>('idle');
  private readonly _result = signal<BuilderResult | null>(null);
  private readonly _frontier = signal<BuilderFrontier | null>(null);
  private readonly _backtest = signal<BuilderBacktest | null>(null);
  private readonly _currentView = signal<CanvasView>('donut');
  private readonly _drift = signal<BuilderDrift | null>(null);
  private readonly _trades = signal<BuilderTrades | null>(null);
  private readonly _driftDiagnostics = signal<BuilderDriftDiagnostics | null>(
    null,
  );
  private readonly _deployableBase = signal<BaseToggle>('invested');
  private readonly _driftStatus = signal<BuilderDriftStatus>('idle');
  private readonly _portfolioName = signal<string | null>(null);

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
  readonly sessionId: Signal<string | null> = this._sessionId.asReadonly();
  readonly resultStatus: Signal<BuilderResultStatus> =
    this._resultStatus.asReadonly();
  readonly result: Signal<BuilderResult | null> = this._result.asReadonly();
  readonly frontier: Signal<BuilderFrontier | null> =
    this._frontier.asReadonly();
  readonly backtest: Signal<BuilderBacktest | null> =
    this._backtest.asReadonly();
  readonly currentView: Signal<CanvasView> = this._currentView.asReadonly();
  readonly drift: Signal<BuilderDrift | null> = this._drift.asReadonly();
  readonly trades: Signal<BuilderTrades | null> = this._trades.asReadonly();
  readonly driftDiagnostics: Signal<BuilderDriftDiagnostics | null> =
    this._driftDiagnostics.asReadonly();
  readonly deployableBase: Signal<BaseToggle> = this._deployableBase.asReadonly();
  readonly driftStatus: Signal<BuilderDriftStatus> =
    this._driftStatus.asReadonly();
  readonly portfolioName: Signal<string | null> =
    this._portfolioName.asReadonly();

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

  setSessionId(value: string | null): void {
    this._sessionId.set(value);
  }

  setResultStatus(value: BuilderResultStatus): void {
    this._resultStatus.set(value);
  }

  setResult(value: BuilderResult | null): void {
    this._result.set(value);
  }

  setFrontier(value: BuilderFrontier | null): void {
    this._frontier.set(value);
  }

  setBacktest(value: BuilderBacktest | null): void {
    this._backtest.set(value);
  }

  setCurrentView(value: CanvasView): void {
    this._currentView.set(value);
  }

  setDrift(value: BuilderDrift | null): void {
    this._drift.set(value);
  }

  setTrades(value: BuilderTrades | null): void {
    this._trades.set(value);
  }

  setDriftDiagnostics(value: BuilderDriftDiagnostics | null): void {
    this._driftDiagnostics.set(value);
  }

  setDeployableBase(value: BaseToggle): void {
    this._deployableBase.set(value);
  }

  setDriftStatus(value: BuilderDriftStatus): void {
    this._driftStatus.set(value);
  }

  setPortfolioName(value: string | null): void {
    this._portfolioName.set(value);
  }

  triggerResultRun(): void {
    this.injector.get(BuilderResultService).runExplicit();
  }

  triggerDriftRun(): void {
    this.injector.get(BUILDER_DRIFT_SERVICE).runExplicit();
  }

  optimize(): void {
    this.dispatchAction('optimize');
  }

  rebalance(): void {
    this.dispatchAction('rebalance_decision');
  }

  // Phase 2 placeholder — Cycle 3 wires the artifact download.
  export(): void {
    return;
  }

  reset(): void {
    this._config.set(null);
    this._stepParams.set(null);
    this._currentStage.set(null);
    this._status.set(StepStatus.Pending);
    this._stepResults.set(new Map());
    this._stepStatuses.set(new Map());
    this._sessionId.set(null);
    this._resultStatus.set('idle');
    this._result.set(null);
    this._frontier.set(null);
    this._backtest.set(null);
    this._currentView.set('donut');
    this._drift.set(null);
    this._trades.set(null);
    this._driftDiagnostics.set(null);
    this._deployableBase.set('invested');
    this._driftStatus.set('idle');
    this._portfolioName.set(null);
  }

  private dispatchAction(stepId: PipelineStepId): void {
    const sid = this._sessionId();
    if (!sid) return;
    if (this._status() === StepStatus.Running) return;
    this._status.set(StepStatus.Running);
    this.api
      .runStep(sid, stepId, {})
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({ error: () => this._status.set(StepStatus.Error) });
  }
}
