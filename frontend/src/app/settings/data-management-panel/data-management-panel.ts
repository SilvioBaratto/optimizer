import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  inject,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ConfirmationDialogComponent } from '../../shared/modal/confirmation-dialog';
import { DatabaseService } from '../database.service';
import type {
  DatabaseStatus,
  HealthCheck,
  TableInfo,
} from '../database.model';

@Component({
  selector: 'app-data-management-panel',
  imports: [DataTableComponent, ConfirmationDialogComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './data-management-panel.html',
})
export class DataManagementPanelComponent {
  private readonly db = inject(DatabaseService);
  private readonly destroyRef = inject(DestroyRef);

  readonly health = signal<HealthCheck | null>(null);
  readonly status = signal<DatabaseStatus | null>(null);
  readonly tables = signal<TableInfo[]>([]);
  readonly loadError = signal<string | null>(null);
  readonly loading = signal<boolean>(false);

  readonly pendingDelete = signal<TableInfo | null>(null);
  readonly deleting = signal<boolean>(false);
  readonly deleteError = signal<string | null>(null);
  readonly deleteSuccess = signal<string | null>(null);

  readonly tableColumns: TableColumn[] = [
    { key: 'name', label: 'Table', sortable: true },
    { key: 'schema', label: 'Schema', sortable: true },
    { key: 'row_count', label: 'Rows', type: 'number', align: 'right', sortable: true },
    { key: 'size_pretty', label: 'Size', align: 'right', sortable: true },
  ];

  readonly tableRows = computed(() =>
    this.tables().map((t) => ({
      name: t.name,
      schema: t.schema,
      row_count: t.row_count,
      size_pretty: t.size_pretty,
    }) as Record<string, unknown>),
  );

  readonly healthLabel = computed(() => {
    const h = this.health();
    if (h == null) return 'unknown';
    return h.healthy ? 'healthy' : 'unhealthy';
  });
  readonly healthIsHealthy = computed(() => this.health()?.healthy === true);

  readonly deleteMessage = computed(() => {
    const t = this.pendingDelete();
    return t
      ? `All ${t.row_count.toLocaleString()} rows in \`${t.name}\` will be removed. This cannot be undone.`
      : '';
  });

  constructor() {
    this.refresh();
  }

  refresh(): void {
    this.loading.set(true);
    this.loadError.set(null);
    this.db
      .getHealth()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (h) => this.health.set(h),
        error: (err: Error) => this.loadError.set(err.message ?? 'health failed'),
      });
    this.db
      .getStatus()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({ next: (s) => this.status.set(s), error: () => this.status.set(null) });
    this.db
      .getTables()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (rows) => {
          this.tables.set(rows);
          this.loading.set(false);
        },
        error: (err: Error) => {
          this.loadError.set(err.message ?? 'tables failed');
          this.loading.set(false);
        },
      });
  }

  requestDelete(table: TableInfo): void {
    this.pendingDelete.set(table);
    this.deleteError.set(null);
  }

  cancelDelete(): void {
    this.pendingDelete.set(null);
  }

  confirmDelete(): void {
    const target = this.pendingDelete();
    if (!target) return;
    this.deleting.set(true);
    this.deleteError.set(null);
    this.db
      .truncateTable(target.name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => this.onDeleteSuccess(target.name),
        error: (err: Error) => this.onDeleteError(err.message ?? 'Truncate failed'),
      });
  }

  private onDeleteSuccess(tableName: string): void {
    this.deleting.set(false);
    this.pendingDelete.set(null);
    this.deleteSuccess.set(`Truncated ${tableName}`);
    this.refresh();
  }

  private onDeleteError(message: string): void {
    this.deleting.set(false);
    this.deleteError.set(message);
  }

  onRowClick(row: Record<string, unknown>): void {
    const match = this.tables().find((t) => t.name === row['name']);
    if (match) this.requestDelete(match);
  }
}
