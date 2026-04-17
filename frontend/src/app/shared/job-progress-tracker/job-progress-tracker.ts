import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  InjectionToken,
  Signal,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Observable, Subscription, timer } from 'rxjs';
import { catchError, of, switchMap, takeWhile, tap } from 'rxjs';

import { JobsService } from '../../services/jobs.service';
import { JobStatus, JobSummary } from '../../models/jobs.model';
import { ProgressBarComponent } from '../progress-bar/progress-bar';

export const JOB_POLL_TICK = new InjectionToken<Observable<unknown>>(
  'JOB_POLL_TICK',
  {
    providedIn: 'root',
    factory: () => timer(0, 2000),
  },
);

const TERMINAL: readonly JobStatus[] = ['completed', 'failed'] as const;

function isTerminal(status: JobStatus): boolean {
  return TERMINAL.includes(status);
}

@Component({
  selector: 'app-job-progress-tracker',
  imports: [ProgressBarComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (job(); as j) {
      <div class="flex flex-col gap-2">
        <div class="flex items-center justify-between">
          <span
            class="text-xs font-medium uppercase tracking-wider"
            [class]="statusClass()"
          >
            {{ j.status }}
          </span>
          @if (elapsed(); as e) {
            <span class="text-xs text-text-tertiary">{{ e }}</span>
          }
        </div>
        <app-progress-bar
          [label]="j.domain"
          [current]="j.current"
          [total]="j.total"
        />
        @if (j.error) {
          <p class="text-xs text-loss">{{ j.error }}</p>
        }
      </div>
    } @else if (error()) {
      <p class="text-xs text-loss">{{ error() }}</p>
    }
  `,
})
export class JobProgressTrackerComponent {
  private readonly jobs = inject(JobsService);
  private readonly tick$ = inject(JOB_POLL_TICK);
  private readonly destroyRef = inject(DestroyRef);

  readonly jobId = input.required<string | null>();
  readonly completed = output<JobSummary>();
  readonly failed = output<JobSummary>();

  readonly job = signal<JobSummary | null>(null);
  readonly error = signal<string | null>(null);

  private subscription?: Subscription;

  readonly statusClass = computed(() => {
    const status = this.job()?.status;
    if (status === 'completed') return 'text-gain';
    if (status === 'failed') return 'text-loss';
    return 'text-text-secondary';
  });

  readonly elapsed: Signal<string | null> = computed(() => {
    const j = this.job();
    if (!j?.started_at) return null;
    const endMs = j.finished_at ? Date.parse(j.finished_at) : Date.now();
    const deltaSec = Math.max(0, Math.round((endMs - Date.parse(j.started_at)) / 1000));
    return `${deltaSec}s`;
  });

  constructor() {
    effect(() => {
      const id = this.jobId();
      this.resetState();
      if (id) this.startPolling(id);
    });
  }

  private startPolling(id: string): void {
    this.subscription = this.tick$
      .pipe(
        switchMap(() =>
          this.jobs.getJob(id).pipe(
            catchError((err: Error) => {
              this.error.set(err.message);
              return of(null);
            }),
          ),
        ),
        tap((j) => j && this.job.set(j)),
        takeWhile((j) => !j || !isTerminal(j.status), true),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((j) => j && this.emitIfTerminal(j));
  }

  private emitIfTerminal(j: JobSummary): void {
    if (j.status === 'completed') this.completed.emit(j);
    else if (j.status === 'failed') this.failed.emit(j);
  }

  private resetState(): void {
    this.subscription?.unsubscribe();
    this.subscription = undefined;
    this.job.set(null);
    this.error.set(null);
  }
}
