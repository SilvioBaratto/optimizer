import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  inject,
  signal,
} from '@angular/core';
import { DatePipe } from '@angular/common';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { catchError, EMPTY } from 'rxjs';

import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { PageErrorBannerComponent } from '../../shared/components/page-error-banner/page-error-banner';
import { SchedulerService } from '../scheduler.service';
import type {
  SchedulerJobStatusDto,
  SchedulerStatusResponse,
} from '../scheduler.model';

const DASH = '—';

function formatTimestamp(value: unknown, datePipe: DatePipe): string {
  if (value == null || value === '') return DASH;
  return datePipe.transform(value as string, 'short') ?? DASH;
}

@Component({
  selector: 'app-scheduler-status-panel',
  imports: [DataTableComponent, PageErrorBannerComponent],
  providers: [DatePipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './scheduler-status-panel.html',
})
export class SchedulerStatusPanelComponent {
  private readonly scheduler = inject(SchedulerService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly datePipe = inject(DatePipe);

  readonly status = signal<SchedulerStatusResponse | null>(null);
  readonly loading = signal<boolean>(false);
  readonly loadError = signal<string | null>(null);

  readonly jobs = computed<SchedulerJobStatusDto[]>(() => this.status()?.jobs ?? []);
  readonly schedulerRunning = computed<boolean>(() => this.status()?.schedulerRunning ?? false);
  readonly isEmpty = computed<boolean>(
    () => !this.schedulerRunning() || this.jobs().length === 0,
  );

  readonly tableColumns: TableColumn[] = [
    { key: 'name', label: 'Job', sortable: true },
    { key: 'trigger', label: 'Trigger', sortable: true },
    { key: 'last_status', label: 'Last status', sortable: true },
    {
      key: 'next_run_time',
      label: 'Next run',
      sortable: true,
      format: (v) => formatTimestamp(v, this.datePipe),
    },
    {
      key: 'last_run_time',
      label: 'Last run',
      sortable: true,
      format: (v) => formatTimestamp(v, this.datePipe),
    },
  ];

  readonly tableRows = computed(() =>
    this.jobs().map((j) => ({
      name: j.name,
      trigger: j.trigger,
      last_status: j.lastStatus ?? DASH,
      next_run_time: j.nextRunTime,
      last_run_time: j.lastRunTime,
    }) as Record<string, unknown>),
  );

  constructor() {
    this.refresh();
  }

  refresh(): void {
    this.loading.set(true);
    this.loadError.set(null);
    this.scheduler
      .getStatus()
      .pipe(
        takeUntilDestroyed(this.destroyRef),
        catchError((err: Error) => {
          this.loadError.set(err.message ?? 'Failed to load scheduler status');
          this.loading.set(false);
          return EMPTY;
        }),
      )
      .subscribe({
        next: (s) => {
          this.status.set(s);
          this.loading.set(false);
        },
      });
  }
}
