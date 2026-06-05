import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  effect,
  inject,
  signal,
} from '@angular/core';
import { DatePipe } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { JobProgressTrackerComponent } from '../../shared/job-progress-tracker/job-progress-tracker';
import { JobsService } from '../../services/jobs.service';
import {
  DOMAIN_META,
  type JobListResponse,
  type JobStatus,
  type JobSummary,
} from '../../models/jobs.model';

const DASH = '—';
const PAGE_SIZE_OPTIONS: readonly number[] = [10, 25, 50, 100];
const STATUS_OPTIONS: readonly JobStatus[] = [
  'pending',
  'running',
  'completed',
  'failed',
];
const DOMAIN_OPTIONS = DOMAIN_META.map((m) => ({ value: m.domain, label: m.label }));
const DOMAIN_LABEL_BY_VALUE = new Map<string, string>(
  DOMAIN_META.map((m) => [m.domain, m.label]),
);

interface JobRow extends Record<string, unknown> {
  id: string;
  idShort: string;
  domain: string;
  domainLabel: string;
  status: JobStatus;
  startedAt: string;
  duration: string;
  progress: string;
  error: string;
  raw: JobSummary;
}

@Component({
  selector: 'app-jobs-panel',
  imports: [DataTableComponent, JobProgressTrackerComponent],
  providers: [DatePipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './jobs-panel.html',
})
export class JobsPanelComponent {
  private readonly jobsSvc = inject(JobsService);
  private readonly datePipe = inject(DatePipe);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  readonly domainOptions = DOMAIN_OPTIONS;
  readonly statusOptions = STATUS_OPTIONS;
  readonly pageSizeOptions = PAGE_SIZE_OPTIONS;

  readonly domainFilter = signal<string[]>([]);
  readonly statusFilter = signal<JobStatus[]>([]);
  readonly pageSize = signal<number>(25);
  readonly page = signal<number>(0);

  readonly jobs = signal<JobSummary[]>([]);
  readonly total = signal<number>(0);
  readonly loading = signal<boolean>(false);
  readonly error = signal<string | null>(null);
  readonly selected = signal<JobSummary | null>(null);
  private readonly hasFetched = signal<boolean>(false);

  readonly tableColumns: TableColumn[] = [
    { key: 'idShort', label: 'Job' },
    { key: 'domainLabel', label: 'Domain' },
    { key: 'status', label: 'Status' },
    { key: 'startedAt', label: 'Started' },
    { key: 'duration', label: 'Duration', align: 'right' },
    { key: 'progress', label: 'Progress', align: 'right' },
    { key: 'error', label: 'Error' },
  ];

  readonly rows = computed<JobRow[]>(() => this.jobs().map((j) => this.toRow(j)));
  readonly hasRows = computed(() => this.rows().length > 0);
  readonly isEmpty = computed(
    () => this.hasFetched() && !this.loading() && this.total() === 0,
  );
  readonly prevDisabled = computed(() => this.page() === 0);
  readonly nextDisabled = computed(
    () => (this.page() + 1) * this.pageSize() >= this.total(),
  );

  constructor() {
    this.hydrateFromQueryParams();
    effect(() => this.refresh());
  }

  openJob(summary: JobSummary): void {
    this.selected.set(summary);
    this.jobsSvc
      .getJob(summary.id)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (full) => this.selected.set(full),
        error: () => { /* keep the summary we already have */ },
      });
  }

  onRowClick(row: Record<string, unknown>): void {
    const raw = (row as JobRow).raw;
    this.openJob(raw);
  }

  closeDrawer(): void {
    this.selected.set(null);
  }

  setDomainFilter(values: string[]): void {
    this.page.set(0);
    this.domainFilter.set(values);
  }

  setStatusFilter(values: JobStatus[]): void {
    this.page.set(0);
    this.statusFilter.set(values);
  }

  setPageSize(next: number): void {
    this.page.set(0);
    this.pageSize.set(next);
  }

  selectedOptions(event: Event): string[] {
    const select = event.target as HTMLSelectElement;
    return Array.from(select.selectedOptions).map((o) => o.value);
  }

  selectedStatuses(event: Event): JobStatus[] {
    return this.selectedOptions(event).filter(isJobStatus);
  }

  prevPage(): void {
    if (this.page() > 0) this.page.update((p) => p - 1);
  }

  nextPage(): void {
    if (!this.nextDisabled()) this.page.update((p) => p + 1);
  }

  private refresh(): void {
    const domain = this.domainFilter();
    const status = this.statusFilter();
    const limit = this.pageSize();
    const offset = this.page() * limit;
    this.syncQueryParams(domain, status, limit);
    this.loading.set(true);
    this.error.set(null);
    this.jobsSvc
      .listJobs({ domain, status, limit, offset })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.applyResponse(res),
        error: (err: { error?: { detail?: string }; message?: string }) =>
          this.applyError(err),
      });
  }

  private applyResponse(res: JobListResponse): void {
    this.jobs.set(res.jobs);
    this.total.set(res.total);
    this.loading.set(false);
    this.hasFetched.set(true);
  }

  private applyError(err: { error?: { detail?: string }; message?: string }): void {
    this.loading.set(false);
    this.hasFetched.set(true);
    this.error.set(err?.error?.detail ?? err?.message ?? 'Failed to load jobs');
  }

  private hydrateFromQueryParams(): void {
    const params = (this.route.snapshot?.queryParams ?? {}) as Record<string, unknown>;
    this.applyQueryParams(params);
    this.route.queryParams
      .pipe(takeUntilDestroyed(this.destroyRef))
      // subscribe-no-error: router queryParams never errors; lifecycle bound
      // via takeUntilDestroyed.
      .subscribe((qp) => this.applyQueryParams(qp));
  }

  private applyQueryParams(qp: Record<string, unknown>): void {
    this.domainFilter.set(asStringArray(qp['domain']));
    this.statusFilter.set(asStringArray(qp['status']).filter(isJobStatus));
    this.page.set(asInt(qp['page'], 0));
    this.pageSize.set(asInt(qp['pageSize'], 25));
  }

  private syncQueryParams(domain: string[], status: JobStatus[], limit: number): void {
    this.router.navigate([], {
      relativeTo: this.route,
      queryParams: {
        domain: domain.length > 0 ? domain : null,
        status: status.length > 0 ? status : null,
        page: this.page() > 0 ? this.page() : null,
        pageSize: limit !== 25 ? limit : null,
      },
      queryParamsHandling: 'merge',
      replaceUrl: true,
    });
  }

  private toRow(summary: JobSummary): JobRow {
    return {
      id: summary.id,
      idShort: summary.id.slice(0, 8),
      domain: summary.domain,
      domainLabel: DOMAIN_LABEL_BY_VALUE.get(summary.domain) ?? summary.domain,
      status: summary.status,
      startedAt: formatTimestamp(summary.started_at, this.datePipe),
      duration: formatDuration(summary.duration_seconds),
      progress: formatProgress(summary),
      error: formatError(summary),
      raw: summary,
    };
  }
}

function formatTimestamp(value: string | null, pipe: DatePipe): string {
  if (!value) return DASH;
  return pipe.transform(value, 'short') ?? DASH;
}

function formatDuration(seconds: number | null): string {
  if (seconds === null) return DASH;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const rem = Math.round(seconds - mins * 60);
  return `${mins}m ${rem}s`;
}

function formatProgress(job: JobSummary): string {
  if (job.status !== 'running') return DASH;
  return `${job.current}/${job.total || 0}`;
}

function formatError(job: JobSummary): string {
  if (job.status !== 'failed' || !job.error) return DASH;
  return job.error.length > 80 ? `${job.error.slice(0, 77)}…` : job.error;
}

function asStringArray(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(String);
  if (typeof value === 'string' && value.length > 0) return [value];
  return [];
}

function asInt(value: unknown, fallback: number): number {
  const n = typeof value === 'string' ? parseInt(value, 10) : Number(value);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}

function isJobStatus(value: string): value is JobStatus {
  return STATUS_OPTIONS.includes(value as JobStatus);
}
