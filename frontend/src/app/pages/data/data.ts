import { Component, signal, computed, inject, DestroyRef, OnInit, ChangeDetectionStrategy } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DatabaseService } from '../../services/database.service';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { ProgressBarComponent } from '../../shared/progress-bar/progress-bar';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { LoadingSkeletonComponent } from '../../shared/loading-skeleton/loading-skeleton';
import { TableInfo } from '../../models/database.model';

interface FetchJob {
  label: string;
  description: string;
  active: boolean;
  progress: number;
  detail: string;
}

@Component({
  selector: 'app-data',
  imports: [StatCardComponent, ProgressBarComponent, DataTableComponent, LoadingSkeletonComponent],
  templateUrl: './data.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DataComponent implements OnInit {
  private readonly dbService = inject(DatabaseService);
  private readonly destroyRef = inject(DestroyRef);

  tables = signal<TableInfo[]>([]);
  tablesLoading = signal(true);

  universeJob = signal<FetchJob>({ label: 'Universe', description: 'Build instrument universe from Trading 212', active: false, progress: 0, detail: '' });
  yfinanceJob = signal<FetchJob>({ label: 'YFinance', description: 'Fetch price history, profiles, and fundamentals', active: false, progress: 0, detail: '' });
  macroJob = signal<FetchJob>({ label: 'Macro', description: 'Fetch economic indicators and bond yields', active: false, progress: 0, detail: '' });

  totalRows = computed(() => this.tables().reduce((sum, t) => sum + t.row_count, 0).toLocaleString());

  tableColumns: TableColumn[] = [
    { key: 'name', label: 'Table', sortable: true },
    { key: 'row_count', label: 'Rows', sortable: true, align: 'right', format: (v) => Number(v).toLocaleString() },
    { key: 'size_pretty', label: 'Size', align: 'right' },
  ];

  ngOnInit() {
    this.dbService.getTables()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(tables => {
        this.tables.set(tables);
        this.tablesLoading.set(false);
      });
  }

  startFetch(jobSignal: typeof this.universeJob) {
    const job = jobSignal();
    if (job.active) return;

    jobSignal.set({ ...job, active: true, progress: 0, detail: 'Starting...' });

    const steps = [
      { pct: 15, detail: 'Connecting to data source...' },
      { pct: 35, detail: 'Fetching records...' },
      { pct: 60, detail: 'Processing data...' },
      { pct: 85, detail: 'Writing to database...' },
      { pct: 100, detail: 'Completed' },
    ];

    let i = 0;
    const interval = setInterval(() => {
      if (i < steps.length) {
        jobSignal.set({ ...jobSignal(), active: i < steps.length - 1, progress: steps[i].pct, detail: steps[i].detail });
        i++;
      } else {
        clearInterval(interval);
      }
    }, 600);
  }
}
