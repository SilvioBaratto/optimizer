import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  inject,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { HttpErrorResponse } from '@angular/common/http';
import {
  Observable,
  Subscription,
  catchError,
  of,
  switchMap,
  takeWhile,
  tap,
} from 'rxjs';

import {
  PIPELINE_STEPS,
  type PipelineStepId,
  StepStatus,
  type StepPollResponse,
} from '../../models/pipeline-builder.model';
import {
  type PipelineConfigSubmit,
  type StepParamsConfig,
  buildStepParams,
} from './step-params.model';
import { stepSummary } from './step-summary';
import { PipelineBuilderApiService } from '../../services/pipeline-builder-api.service';
import { JOB_POLL_TICK } from '../../shared/job-progress-tracker/job-progress-tracker';
import { RunConfigPanelComponent } from './run-config-panel';
import { StepSectionComponent } from './step-section/step-section';
import { StepBuildHistoryPanelComponent } from './step-build-history-panel';
import { StepCleanReturnsPanelComponent } from './step-clean-returns-panel';
import { StepCostPanelComponent } from './step-cost-panel';
import { StepCoverageGatePanelComponent } from './step-coverage-gate-panel';
import { StepLoadPanelComponent } from './step-load-panel';
import { StepOptimizePanelComponent } from './step-optimize-panel';
import { StepPersistPanelComponent } from './step-persist-panel';
import { StepRebalanceDecisionPanelComponent } from './step-rebalance-decision-panel';
import { StepRegimePanelComponent } from './step-regime-panel';
import { StepReportPanelComponent } from './step-report-panel';
import { StepScreenPanelComponent } from './step-screen-panel';
import { StepValidateIsPanelComponent } from './step-validate-is-panel';
import { StepValidateOosPanelComponent } from './step-validate-oos-panel';

type WizardPhase = 'config' | 'running';

@Component({
  selector: 'app-pipeline-stepper',
  imports: [
    RunConfigPanelComponent,
    StepSectionComponent,
    StepLoadPanelComponent,
    StepScreenPanelComponent,
    StepCleanReturnsPanelComponent,
    StepBuildHistoryPanelComponent,
    StepValidateIsPanelComponent,
    StepValidateOosPanelComponent,
    StepCoverageGatePanelComponent,
    StepRegimePanelComponent,
    StepOptimizePanelComponent,
    StepRebalanceDecisionPanelComponent,
    StepCostPanelComponent,
    StepReportPanelComponent,
    StepPersistPanelComponent,
  ],
  templateUrl: './pipeline-stepper.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PipelineStepperComponent {
  readonly steps = PIPELINE_STEPS;

  sessionId = signal<string | null>(null);
  stepStatuses = signal<Map<PipelineStepId, StepStatus>>(new Map());
  // Pipeline cursor: the step currently dispatched/polling or awaiting Continue.
  activeStepId = signal<PipelineStepId | null>(null);
  // UI disclosure only — never issues HTTP, never mutates stepStatuses.
  expandedStepId = signal<PipelineStepId | null>(null);
  stepParams = signal<StepParamsConfig | null>(null);
  phase = signal<WizardPhase>('config');
  sessionError = signal<string | null>(null);
  lastError = signal<string | null>(null);
  abortGateError = signal<string | null>(null);
  pollProgress = signal<StepPollResponse | null>(null);
  stepResults = signal<Map<PipelineStepId, StepPollResponse>>(new Map());

  completedCount = computed(
    () =>
      [...this.stepStatuses().values()].filter(
        (s) => s === StepStatus.Completed,
      ).length,
  );

  // Continue is offered only when the active step is Completed and a next
  // step exists. advance() has already marked that next step Ready.
  canContinue = computed(() => {
    const id = this.activeStepId();
    if (!id) return false;
    if (this.stepStatuses().get(id) !== StepStatus.Completed) return false;
    return nextStepId(id) !== null;
  });

  // Per-section caches keyed by step id (memoised so OnPush template reads
  // don't recompute per change-detection pass).
  private readonly summaries = computed(() => {
    const results = this.stepResults();
    const m = new Map<PipelineStepId, string>();
    for (const step of this.steps) {
      m.set(step.id, stepSummary(step.id, results.get(step.id)?.result ?? null));
    }
    return m;
  });

  private readonly api = inject(PipelineBuilderApiService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly tick$ = inject(JOB_POLL_TICK) as Observable<unknown>;
  private pollingSubscription?: Subscription;

  constructor() {
    this.destroyRef.onDestroy(() => this.disposeSession());
  }

  sectionStatus(id: PipelineStepId): StepStatus {
    return this.stepStatuses().get(id) ?? StepStatus.Pending;
  }

  sectionResult(id: PipelineStepId): Record<string, unknown> | null {
    return this.stepResults().get(id)?.result ?? null;
  }

  summaryFor(id: PipelineStepId): string {
    return this.summaries().get(id) ?? '';
  }

  // Stale errors must not bleed into other sections.
  errorFor(id: PipelineStepId): string | null {
    return id === this.activeStepId() ? this.lastError() : null;
  }

  isExpanded(id: PipelineStepId): boolean {
    return this.expandedStepId() === id;
  }

  showContinueFor(id: PipelineStepId): boolean {
    return id === this.activeStepId() && this.canContinue();
  }

  showRetryFor(id: PipelineStepId): boolean {
    return (
      id === this.activeStepId() &&
      this.sectionStatus(id) === StepStatus.Error
    );
  }

  onConfigSubmit(payload: PipelineConfigSubmit): void {
    this.sessionError.set(null);
    this.stepParams.set(payload.steps);
    this.api
      .createSession(payload.config)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (r) => this.startSession(r.sessionId),
        error: (e: HttpErrorResponse) =>
          this.sessionError.set(formatCreateError(e)),
      });
  }

  onAbort(): void {
    this.stopPolling();
    this.disposeSession();
    this.resetState();
  }

  // Chevron click — pure UI, no HTTP, pipeline cursor untouched. A Locked /
  // Pending step has nothing to show, so it stays collapsed.
  toggleExpand(id: PipelineStepId): void {
    const status = this.sectionStatus(id);
    if (status === StepStatus.Locked || status === StepStatus.Pending) return;
    this.expandedStepId.update((cur) => (cur === id ? null : id));
  }

  // User confirms moving on. advance() already unlocked N+1 to Ready.
  onContinue(): void {
    const cur = this.activeStepId();
    if (!cur || this.stepStatuses().get(cur) !== StepStatus.Completed) return;
    const nid = nextStepId(cur);
    if (nid === null) return;
    this.autoDispatch(nid);
  }

  onRetry(): void {
    const cur = this.activeStepId();
    if (!cur) return;
    this.lastError.set(null);
    this.abortGateError.set(null);
    this.updateStatus(cur, StepStatus.Ready);
    this.autoDispatch(cur);
  }

  // Panels emit their own params, but the single upfront form is the source
  // of truth — re-dispatch the active step from the registry, ignoring the
  // emitted payload. Used only for an explicit single-step re-run.
  onPanelRunStep(_params: Record<string, unknown>): void {
    const cur = this.activeStepId();
    if (!cur) return;
    if (this.stepStatuses().get(cur) === StepStatus.Running) return;
    this.autoDispatch(cur);
  }

  private autoDispatch(stepId: PipelineStepId): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.lastError.set(null);
    this.activeStepId.set(stepId);
    this.expandedStepId.set(stepId);
    this.dispatchStep(sid, stepId, buildStepParams(this.stepParams(), stepId));
  }

  private dispatchStep(
    sid: string,
    stepId: PipelineStepId,
    params: Record<string, unknown>,
  ): void {
    this.sessionError.set(null);
    this.api
      .runStep(sid, stepId, params)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => this.onDispatched(stepId),
        error: (e: HttpErrorResponse) =>
          this.sessionError.set(formatDispatchError(e)),
      });
  }

  private onDispatched(stepId: PipelineStepId): void {
    this.updateStatus(stepId, StepStatus.Running);
    this.startPolling(stepId);
  }

  private startPolling(stepId: PipelineStepId): void {
    this.stopPolling();
    const sid = this.sessionId();
    if (!sid) return;
    this.pollingSubscription = this.tick$
      .pipe(
        switchMap(() =>
          this.api.pollStep(sid, stepId).pipe(catchError(() => of(null))),
        ),
        tap((r) => r && this.applyPoll(stepId, r)),
        takeWhile((r) => !r || !isTerminal(r.status), true),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe();
  }

  private applyPoll(stepId: PipelineStepId, r: StepPollResponse): void {
    this.pollProgress.set(r);
    this.stepResults.update((m) => new Map(m).set(stepId, r));
    this.updateStatus(stepId, mapPollStatus(r.status));
    // No auto-dispatch: completion only unlocks the next step. The user
    // confirms via Continue.
    if (r.status === 'completed') this.advance(stepId);
    if (r.status === 'failed') this.handleFailure(stepId, r);
    if (isTerminal(r.status)) this.stopPolling();
  }

  private handleFailure(stepId: PipelineStepId, r: StepPollResponse): void {
    if (r.gateReason) {
      this.handleGateFailure(stepId, r.gateReason);
      return;
    }
    this.lastError.set(r.error ?? 'Step failed');
  }

  private handleGateFailure(_stepId: PipelineStepId, reason: string): void {
    this.abortGateError.set(reason);
  }

  private advance(completedStepId: PipelineStepId): void {
    const nextId = nextStepId(completedStepId);
    if (nextId === null) return;
    this.updateStatus(nextId, StepStatus.Ready);
  }

  private stopPolling(): void {
    this.pollingSubscription?.unsubscribe();
    this.pollingSubscription = undefined;
  }

  private updateStatus(stepId: PipelineStepId, status: StepStatus): void {
    this.stepStatuses.update((m) => new Map(m).set(stepId, status));
  }

  private startSession(id: string): void {
    this.sessionId.set(id);
    this.stepStatuses.set(initialStatuses());
    this.phase.set('running');
    // Auto-run step 1; the user drives the rest with Continue.
    this.autoDispatch('load');
  }

  private resetState(): void {
    this.sessionId.set(null);
    this.stepStatuses.set(new Map());
    this.activeStepId.set(null);
    this.expandedStepId.set(null);
    this.stepParams.set(null);
    this.sessionError.set(null);
    this.lastError.set(null);
    this.abortGateError.set(null);
    this.pollProgress.set(null);
    this.stepResults.set(new Map());
    this.phase.set('config');
  }

  private disposeSession(): void {
    const id = this.sessionId();
    if (!id) return;
    this.api.deleteSession(id).subscribe({ error: () => undefined });
  }
}

const TERMINAL_POLL: readonly string[] = ['completed', 'failed'];

function isTerminal(status: string): boolean {
  return TERMINAL_POLL.includes(status);
}

function mapPollStatus(s: string): StepStatus {
  if (s === 'running' || s === 'pending') return StepStatus.Running;
  if (s === 'completed') return StepStatus.Completed;
  if (s === 'failed') return StepStatus.Error;
  return StepStatus.Ready;
}

function nextStepId(id: PipelineStepId): PipelineStepId | null {
  const i = PIPELINE_STEPS.findIndex((s) => s.id === id);
  const next = PIPELINE_STEPS[i + 1];
  return next ? next.id : null;
}

function initialStatuses(): Map<PipelineStepId, StepStatus> {
  const map = new Map<PipelineStepId, StepStatus>();
  PIPELINE_STEPS.forEach((step, i) => {
    map.set(step.id, i === 0 ? StepStatus.Ready : StepStatus.Locked);
  });
  return map;
}

function formatCreateError(err: HttpErrorResponse): string {
  if (err.status === 429) {
    const detail = typeof err.error === 'object' && err.error?.detail;
    return detail || 'Pipeline session capacity reached. Try again shortly.';
  }
  return `Session creation failed (HTTP ${err.status})`;
}

function formatDispatchError(err: HttpErrorResponse): string {
  if (err.status === 409) return `Step is already running (HTTP 409)`;
  return `Step dispatch failed (HTTP ${err.status})`;
}
