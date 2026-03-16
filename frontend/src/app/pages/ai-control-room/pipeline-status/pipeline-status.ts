import {
  Component,
  signal,
  computed,
  inject,
  DestroyRef,
  ChangeDetectionStrategy,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DatePipe } from '@angular/common';
import { interval } from 'rxjs';
import { switchMap, catchError } from 'rxjs/operators';
import { of } from 'rxjs';
import { LucideAngularModule } from 'lucide-angular';
import { JobsService } from '../../../services/jobs.service';
import { FreshnessIndicatorComponent } from '../../../shared/freshness-indicator/freshness-indicator';
import { ProgressBarComponent } from '../../../shared/progress-bar/progress-bar';
import type { DomainStatus } from '../../../models/jobs.model';

@Component({
  selector: 'app-pipeline-status',
  imports: [
    DatePipe,
    LucideAngularModule,
    FreshnessIndicatorComponent,
    ProgressBarComponent,
  ],
  templateUrl: './pipeline-status.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PipelineStatusComponent {
  private readonly jobsService = inject(JobsService);
  private readonly destroyRef = inject(DestroyRef);

  readonly domainStatuses = signal<DomainStatus[]>([]);
  readonly isLoading = signal(true);
  readonly pollError = signal<string | null>(null);
  readonly lastPolled = signal<Date | null>(null);

  readonly runningJobs = computed(() =>
    this.domainStatuses()
      .filter(d => d.running !== null)
      .map(d => d.running!),
  );

  readonly failedJobs = computed(() =>
    this.domainStatuses()
      .flatMap(d => d.recentFailures)
      .sort((a, b) => (b.started_at ?? '').localeCompare(a.started_at ?? ''))
      .slice(0, 5),
  );

  readonly hasRunning = computed(() => this.runningJobs().length > 0);
  readonly hasFailures = computed(() => this.failedJobs().length > 0);

  constructor() {
    this.loadOnce();

    interval(30_000)
      .pipe(
        switchMap(() => this.jobsService.getDomainStatuses().pipe(
          catchError(() => {
            this.pollError.set('Connection error');
            return of(null);
          }),
        )),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe(result => {
        if (result) {
          this.domainStatuses.set(result);
          this.pollError.set(null);
          this.lastPolled.set(new Date());
        }
      });
  }

  private loadOnce(): void {
    this.jobsService.getDomainStatuses().subscribe({
      next: result => {
        this.domainStatuses.set(result);
        this.isLoading.set(false);
        this.lastPolled.set(new Date());
      },
      error: () => {
        this.isLoading.set(false);
        this.pollError.set('Failed to load pipeline status');
      },
    });
  }

  ageString(d: DomainStatus): string {
    if (d.lastSuccessAgeHours == null) return '';
    const hours = d.lastSuccessAgeHours;
    if (hours < 1) return `${Math.round(hours * 60)}m ago`;
    if (hours < 24) return `${Math.round(hours)}h ago`;
    const days = Math.round(hours / 24);
    return `${days}d ago`;
  }

  durationString(seconds: number | null): string {
    if (seconds == null) return '-';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
  }

  errorExcerpt(error: string | null): string {
    if (!error) return '';
    return error.length > 120 ? error.substring(0, 120) + '...' : error;
  }

  relativeTime(date: Date): string {
    const seconds = Math.round((Date.now() - date.getTime()) / 1000);
    if (seconds < 60) return 'just now';
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ago`;
  }
}
