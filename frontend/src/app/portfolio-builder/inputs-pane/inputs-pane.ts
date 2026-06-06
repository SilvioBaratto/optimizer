import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  inject,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { HttpErrorResponse } from '@angular/common/http';
import {
  type Observable,
  type Subscription,
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
} from '../../core/models/pipeline-builder.model';
import {
  type PipelineConfigSubmit,
  buildStepParams,
} from '../../pipeline-stepper/models/step-params.model';
import { PipelineBuilderApiService } from '../../core/services/pipeline-builder-api.service';
import { JOB_POLL_TICK } from '../../shared/job-progress-tracker/job-progress-tracker';
import { RunConfigPanelComponent } from '../../pipeline-stepper/run-config-panel/run-config-panel';
import { StepLoadPanelComponent } from '../../pipeline-stepper/step-load-panel/step-load-panel';
import { StepScreenPanelComponent } from '../../pipeline-stepper/step-screen-panel/step-screen-panel';
import { StepCleanReturnsPanelComponent } from '../../pipeline-stepper/step-clean-returns-panel/step-clean-returns-panel';
import { StepBuildHistoryPanelComponent } from '../../pipeline-stepper/step-build-history-panel/step-build-history-panel';
import { StepValidateIsPanelComponent } from '../../pipeline-stepper/step-validate-is-panel/step-validate-is-panel';
import { StepValidateOosPanelComponent } from '../../pipeline-stepper/step-validate-oos-panel/step-validate-oos-panel';
import { StepCoverageGatePanelComponent } from '../../pipeline-stepper/step-coverage-gate-panel/step-coverage-gate-panel';
import { StepRegimePanelComponent } from '../../pipeline-stepper/step-regime-panel/step-regime-panel';
import { StepOptimizePanelComponent } from '../../pipeline-stepper/step-optimize-panel/step-optimize-panel';
import { StepRebalanceDecisionPanelComponent } from '../../pipeline-stepper/step-rebalance-decision-panel/step-rebalance-decision-panel';
import { StepCostPanelComponent } from '../../pipeline-stepper/step-cost-panel/step-cost-panel';
import { StepReportPanelComponent } from '../../pipeline-stepper/step-report-panel/step-report-panel';
import { StepPersistPanelComponent } from '../../pipeline-stepper/step-persist-panel/step-persist-panel';
import { StepSectionComponent } from '../../pipeline-stepper/step-section/step-section';
import {
  ACCORDION_LABELS,
  ACCORDION_SECTIONS,
  type AccordionSectionId,
  stepsForSection,
} from '../models/builder-stage';
import { sectionSummary } from '../../pipeline-stepper/chip-summary';
import { BuilderStore } from '../state/builder.store';

@Component({
  selector: 'app-inputs-pane',
  imports: [
    StepSectionComponent,
    RunConfigPanelComponent,
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
  templateUrl: './inputs-pane.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class InputsPaneComponent {
  readonly sections = ACCORDION_SECTIONS;
  readonly labels = ACCORDION_LABELS;
  readonly StepStatus = StepStatus;

  readonly activeStepId = signal<PipelineStepId | null>(null);
  readonly expandedSectionId = signal<AccordionSectionId | null>(null);
  readonly lastError = signal<string | null>(null);
  readonly abortGateError = signal<string | null>(null);

  private readonly store = inject(BuilderStore);
  private readonly api = inject(PipelineBuilderApiService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly tick$ = inject(JOB_POLL_TICK) as Observable<unknown>;
  private pollingSubscription?: Subscription;

  readonly stepStatuses = this.store.stepStatuses;
  readonly stepResults = this.store.stepResults;
  readonly sessionId = this.store.sessionId;

  sectionAggStatus(id: AccordionSectionId): StepStatus {
    return aggregateStatus(stepsForSection(id), this.stepStatuses());
  }

  sectionAggSummary(id: AccordionSectionId): string {
    return sectionSummary(id, this.stepResults());
  }

  sectionStatus(id: PipelineStepId): StepStatus {
    return this.stepStatuses().get(id) ?? StepStatus.Pending;
  }

  sectionResult(id: PipelineStepId): Record<string, unknown> | null {
    return this.stepResults().get(id) ?? null;
  }

  errorFor(id: PipelineStepId): string | null {
    return id === this.activeStepId() ? this.lastError() : null;
  }

  isExpanded(id: AccordionSectionId): boolean {
    return this.expandedSectionId() === id;
  }

  toggleSection(id: AccordionSectionId): void {
    this.expandedSectionId.update((cur) => (cur === id ? null : id));
  }

  showContinueForSection(id: AccordionSectionId): boolean {
    const active = this.activeStepId();
    if (!active || !stepsForSection(id).includes(active)) return false;
    if (this.stepStatuses().get(active) !== StepStatus.Completed) return false;
    return nextStepId(active) !== null;
  }

  showRetryForSection(id: AccordionSectionId): boolean {
    const active = this.activeStepId();
    if (!active || !stepsForSection(id).includes(active)) return false;
    return this.stepStatuses().get(active) === StepStatus.Error;
  }

  onConfigSubmit(payload: PipelineConfigSubmit): void {
    if (this.store.sessionId() !== null) {
      // Resubmit: wipe stale per-step state from the previous session
      // before issuing a new createSession. Caller-controlled per the
      // store's no-side-effect setSessionId contract.
      this.store.reset();
    }
    this.store.setConfig(payload.config);
    this.store.setStepParams(payload.steps);
    this.api
      .createSession(payload.config)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (r) => this.onSessionCreated(r.sessionId),
        error: (e: HttpErrorResponse) =>
          this.lastError.set(`Session creation failed (HTTP ${e.status})`),
      });
  }

  // Unlock the first step so its run-step button appears; later steps stay
  // Pending (gated) until each predecessor reports completed.
  private onSessionCreated(sessionId: string): void {
    this.store.setSessionId(sessionId);
    this.store.setStepStatus(PIPELINE_STEPS[0].id, StepStatus.Ready);
  }

  onPanelRunStep(stepId: PipelineStepId, _payload: Record<string, unknown>): void {
    if (this.stepStatuses().get(stepId) === StepStatus.Running) return;
    this.dispatchStep(stepId);
  }

  onContinue(stepId: PipelineStepId): void {
    if (this.stepStatuses().get(stepId) !== StepStatus.Completed) return;
    const nid = nextStepId(stepId);
    if (nid === null) return;
    this.dispatchStep(nid);
  }

  onRetry(stepId: PipelineStepId): void {
    this.lastError.set(null);
    this.abortGateError.set(null);
    this.store.setStepStatus(stepId, StepStatus.Ready);
    this.dispatchStep(stepId);
  }

  private dispatchStep(stepId: PipelineStepId): void {
    const sid = this.sessionId();
    if (!sid) return;
    this.lastError.set(null);
    this.activeStepId.set(stepId);
    const params = buildStepParams(this.store.stepParams(), stepId);
    this.api
      .runStep(sid, stepId, params)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => this.onDispatched(stepId),
        error: (e: HttpErrorResponse) =>
          this.lastError.set(`Step dispatch failed (HTTP ${e.status})`),
      });
  }

  private onDispatched(stepId: PipelineStepId): void {
    this.store.setStepStatus(stepId, StepStatus.Running);
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
    this.store.setStepResult(stepId, r.result ?? null);
    this.store.setStepStatus(stepId, mapPollStatus(r.status));
    if (r.status === 'failed') this.handleFailure(r);
    if (isTerminal(r.status)) this.stopPolling();
  }

  private handleFailure(r: StepPollResponse): void {
    if (r.gateReason) {
      this.abortGateError.set(r.gateReason);
      return;
    }
    this.lastError.set(r.error ?? 'Step failed');
  }

  private stopPolling(): void {
    this.pollingSubscription?.unsubscribe();
    this.pollingSubscription = undefined;
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

// Aggregate rule for section badge: any Error wins; else any Running; else
// all Completed; else any Ready; else any Pending; else Locked. Drives a
// single deterministic badge across the section's child steps.
function aggregateStatus(
  steps: readonly PipelineStepId[],
  statuses: Map<PipelineStepId, StepStatus>,
): StepStatus {
  const present = steps.map((id) => statuses.get(id) ?? StepStatus.Pending);
  if (present.some((s) => s === StepStatus.Error)) return StepStatus.Error;
  if (present.some((s) => s === StepStatus.Running)) return StepStatus.Running;
  if (present.every((s) => s === StepStatus.Completed))
    return StepStatus.Completed;
  if (present.some((s) => s === StepStatus.Ready)) return StepStatus.Ready;
  if (present.some((s) => s === StepStatus.Pending)) return StepStatus.Pending;
  return StepStatus.Locked;
}
