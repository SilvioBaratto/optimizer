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

import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { SchedulerService } from '../../services/scheduler.service';
import type {
  SchedulerJobStatusDto,
  SchedulerStatusResponse,
} from '../../models/scheduler.model';

@Component({
  selector: 'app-scheduler-status-panel',
  imports: [DataTableComponent, DatePipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './scheduler-status-panel.html',
})
export class SchedulerStatusPanelComponent {
  private readonly scheduler = inject(SchedulerService);
  private readonly destroyRef = inject(DestroyRef);

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
  ];

  readonly tableRows = computed(() =>
    this.jobs().map((j) => ({
      name: j.name,
      trigger: j.trigger,
      last_status: j.lastStatus ?? '—',
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
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (s) => {
          this.status.set(s);
          this.loading.set(false);
        },
        error: (err: Error) => {
          this.loadError.set(err.message ?? 'Failed to load scheduler status');
          this.loading.set(false);
        },
      });
  }
}
