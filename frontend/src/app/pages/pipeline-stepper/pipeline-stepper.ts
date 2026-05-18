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
  type RunLevelConfig,
  StepStatus,
  type StepPollResponse,
} from '../../models/pipeline-builder.model';
import { PipelineBuilderApiService } from '../../services/pipeline-builder-api.service';
import { JOB_POLL_TICK } from '../../shared/job-progress-tracker/job-progress-tracker';
import { RunConfigPanelComponent } from './run-config-panel';

type WizardPhase = 'config' | 'running';

@Component({
  selector: 'app-pipeline-stepper',
  imports: [RunConfigPanelComponent],
  templateUrl: './pipeline-stepper.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PipelineStepperComponent {
  readonly steps = PIPELINE_STEPS;

  sessionId = signal<string | null>(null);
  stepStatuses = signal<Map<PipelineStepId, StepStatus>>(new Map());
  activeStepId = signal<PipelineStepId | null>(null);
  phase = signal<WizardPhase>('config');
  sessionError = signal<string | null>(null);
  lastError = signal<string | null>(null);
  abortGateError = signal<string | null>(null);
  pollProgress = signal<StepPollResponse | null>(null);

  activeStep = computed(() =>
    this.steps.find((s) => s.id === this.activeStepId()) ?? null,
  );
  activeStepStatus = computed(() => {
    const id = this.activeStepId();
    return id ? (this.stepStatuses().get(id) ?? StepStatus.Pending) : StepStatus.Pending;
  });
  completedCount = computed(
    () => [...this.stepStatuses().values()].filter((s) => s === StepStatus.Completed).length,
  );
  nextEnabled = computed(() => {
    const id = this.activeStepId();
    if (!id) return false;
    if (this.stepStatuses().get(id) !== StepStatus.Completed) return false;
    return nextStepId(id) !== null;
  });

  private readonly api = inject(PipelineBuilderApiService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly tick$ = inject(JOB_POLL_TICK) as Observable<unknown>;
  private pollingSubscription?: Subscription;

  constructor() {
    this.destroyRef.onDestroy(() => this.disposeSession());
  }

  onConfigSubmit(config: RunLevelConfig): void {
    this.sessionError.set(null);
    this.api
      .createSession(config)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (r) => this.startSession(r.sessionId),
        error: (e: HttpErrorResponse) => this.sessionError.set(formatCreateError(e)),
      });
  }

  onAbort(): void {
    this.stopPolling();
    this.disposeSession();
    this.resetState();
  }

  selectStep(stepId: PipelineStepId): void {
    const status = this.stepStatuses().get(stepId);
    if (status === StepStatus.Ready || status === StepStatus.Completed) {
      this.activeStepId.set(stepId);
    }
  }

  runActiveStep(params: Record<string, unknown> = {}): void {
    if (this.activeStepStatus() !== StepStatus.Ready) return;
    const sid = this.sessionId();
    const stepId = this.activeStepId();
    if (!sid || !stepId) return;
    this.dispatchStep(sid, stepId, params);
  }

  statusDotClass(status: StepStatus | undefined | null): string {
    return STATUS_DOT[status ?? StepStatus.Pending];
  }

  rowClass(stepId: PipelineStepId): string {
    const base = 'w-full text-left px-3 py-2 rounded-md border transition-colors';
    return stepId === this.activeStepId()
      ? `${base} border-accent bg-accent/5`
      : `${base} border-border bg-surface-raised hover:border-border-muted`;
  }

  isClickable(status: StepStatus | undefined | null): boolean {
    return status === StepStatus.Ready || status === StepStatus.Completed;
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
        error: (e: HttpErrorResponse) => this.sessionError.set(formatDispatchError(e)),
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
        switchMap(() => this.api.pollStep(sid, stepId).pipe(catchError(() => of(null)))),
        tap((r) => r && this.applyPoll(stepId, r)),
        takeWhile((r) => !r || !isTerminal(r.status), true),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe();
  }

  private applyPoll(stepId: PipelineStepId, r: StepPollResponse): void {
    this.pollProgress.set(r);
    this.updateStatus(stepId, mapPollStatus(r.status));
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
    this.activeStepId.set('load');
    this.phase.set('running');
  }

  private resetState(): void {
    this.sessionId.set(null);
    this.stepStatuses.set(new Map());
    this.activeStepId.set(null);
    this.sessionError.set(null);
    this.lastError.set(null);
    this.abortGateError.set(null);
    this.pollProgress.set(null);
    this.phase.set('config');
  }

  private disposeSession(): void {
    const id = this.sessionId();
    if (!id) return;
    this.api.deleteSession(id).subscribe({ error: () => undefined });
  }
}

const STATUS_DOT: Record<StepStatus, string> = {
  [StepStatus.Pending]: 'bg-text-tertiary',
  [StepStatus.Locked]: 'bg-text-tertiary opacity-40',
  [StepStatus.Ready]: 'bg-accent ring-2 ring-accent/30',
  [StepStatus.Running]: 'bg-accent animate-pulse',
  [StepStatus.Completed]: 'bg-gain',
  [StepStatus.Error]: 'bg-loss',
};

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
