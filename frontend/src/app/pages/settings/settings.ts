import {
  Component,
  signal,
  inject,
  ChangeDetectionStrategy,
} from '@angular/core';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { LucideAngularModule } from 'lucide-angular';
import { MockFetchService } from '../../services/mock-fetch.service';
import { NotificationService } from '../../shared/notification/notification.service';
import { MOCK_DATA_SOURCES } from '../../mocks/settings-mocks';

@Component({
  selector: 'app-settings',
  imports: [
    PageHeaderComponent,
    TabGroupComponent,
    DataTableComponent,
    LucideAngularModule,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './settings.html',
})
export class SettingsComponent {
  private readonly notifications = inject(NotificationService);
  private readonly mockFetch = inject(MockFetchService);

  // ── Loading state ──
  isLoading = signal(true);
  hasError = signal(false);
  errorMessage = signal('');

  // ── State ──
  readonly activeTab = signal<'data'>('data');
  readonly cacheCleared = signal(false);

  // ── Static data ──
  readonly dataSources = MOCK_DATA_SOURCES;

  // ── Tabs ──
  readonly tabs: Tab[] = [
    { id: 'data', label: 'Data Management' },
  ];

  // ── Data Management: table columns ──
  readonly dataSourceColumns: TableColumn[] = [
    { key: 'name', label: 'Source', sortable: true },
    { key: 'type', label: 'Type', sortable: true },
    { key: 'status', label: 'Status', sortable: true, type: 'badge', badgeMap: {
      connected: { value: 'Connected', colorClass: 'bg-gain/15 text-gain' },
      disconnected: { value: 'Disconnected', colorClass: 'bg-surface-inset text-text-tertiary' },
      error: { value: 'Error', colorClass: 'bg-loss/15 text-loss' },
    }},
    { key: 'lastSync', label: 'Last Sync', sortable: true, type: 'date', dateFormat: 'medium' },
    { key: 'recordCount', label: 'Records', sortable: true, type: 'number', align: 'right' },
  ];

  readonly dataSourceRows = this.dataSources.map(ds => ({
    name: ds.name,
    type: ds.type,
    status: ds.status,
    lastSync: ds.lastSync,
    recordCount: ds.recordCount,
  }));

  constructor() {
    this.loadData();
  }

  loadData(): void {
    this.isLoading.set(true);
    this.hasError.set(false);
    this.mockFetch
      .fetch({ dataSources: MOCK_DATA_SOURCES })
      .then(() => {
        this.isLoading.set(false);
      })
      .catch((err: Error) => {
        this.hasError.set(true);
        this.errorMessage.set(err.message);
        this.isLoading.set(false);
      });
  }

  retry(): void {
    this.loadData();
  }

  onTabChange(tab: string): void {
    this.activeTab.set(tab as 'data');
  }

  clearCache(): void {
    this.cacheCleared.set(true);
    this.notifications.success('Cache cleared successfully');
    setTimeout(() => this.cacheCleared.set(false), 3000);
  }
}
