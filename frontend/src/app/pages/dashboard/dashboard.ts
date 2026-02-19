import { Component, signal, inject, DestroyRef, OnInit, ChangeDetectionStrategy } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { RouterLink } from '@angular/router';
import { forkJoin } from 'rxjs';
import { DatabaseService } from '../../services/database.service';
import { UniverseService } from '../../services/universe.service';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { LoadingSkeletonComponent } from '../../shared/loading-skeleton/loading-skeleton';
import { HealthCheck, TableInfo } from '../../models/database.model';
import { UniverseStats } from '../../models/universe.model';

@Component({
  selector: 'app-dashboard',
  imports: [RouterLink, StatCardComponent, DataTableComponent, LoadingSkeletonComponent],
  templateUrl: './dashboard.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DashboardComponent implements OnInit {
  private readonly dbService = inject(DatabaseService);
  private readonly universeService = inject(UniverseService);
  private readonly destroyRef = inject(DestroyRef);

  health = signal<HealthCheck | null>(null);
  stats = signal<UniverseStats | null>(null);
  tables = signal<TableInfo[]>([]);
  loading = signal(true);

  tableColumns: TableColumn[] = [
    { key: 'name', label: 'Table', sortable: true },
    { key: 'row_count', label: 'Rows', sortable: true, align: 'right', format: (v) => Number(v).toLocaleString() },
    { key: 'size_pretty', label: 'Size', align: 'right' },
  ];

  ngOnInit() {
    forkJoin({
      health: this.dbService.getHealth(),
      stats: this.universeService.getStats(),
      tables: this.dbService.getTables(),
    }).pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(({ health, stats, tables }) => {
        this.health.set(health);
        this.stats.set(stats);
        this.tables.set(tables);
        this.loading.set(false);
      });
  }
}
