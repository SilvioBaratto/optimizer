import {
  Component,
  signal,
  computed,
  inject,
  ChangeDetectionStrategy,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatePipe } from '@angular/common';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { EchartsGaugeComponent, GaugeThreshold } from '../../shared/echarts-gauge/echarts-gauge';
import { FormatService } from '../../services/format.service';
import { MockFetchService } from '../../services/mock-fetch.service';
import { NotificationService } from '../../shared/notification/notification.service';
import type { UserPreferences, LogLevel, ChartColorScheme } from '../../models/settings.model';
import {
  MOCK_DATA_SOURCES,
  MOCK_USER_PREFERENCES,
  MOCK_SYSTEM_INFO,
  MOCK_API_KEYS,
  MOCK_LOG_ENTRIES,
  MOCK_EXPORT_SETTINGS,
  MOCK_MCP_SERVERS,
  CHART_COLOR_SCHEMES,
} from '../../mocks/settings-mocks';

type SettingsTab = 'data' | 'preferences' | 'api' | 'system';

@Component({
  selector: 'app-settings',
  imports: [
    FormsModule,
    DatePipe,
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
    DataTableComponent,
    EchartsGaugeComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './settings.html',
})
export class SettingsComponent {
  private readonly fmt = inject(FormatService);
  private readonly notifications = inject(NotificationService);
  private readonly mockFetch = inject(MockFetchService);

  // ── Loading state ──
  isLoading = signal(true);
  hasError = signal(false);
  errorMessage = signal('');

  // ── State ──
  readonly activeTab = signal<SettingsTab>('data');
  readonly preferences = signal<UserPreferences>({ ...MOCK_USER_PREFERENCES });
  readonly revealedKeys = signal<Set<string>>(new Set());
  readonly logLevelFilter = signal<LogLevel | 'all'>('all');
  readonly cacheCleared = signal(false);
  readonly selectedColorScheme = signal<ChartColorScheme>(MOCK_USER_PREFERENCES.theme === 'dark' ? 'default' : 'default');
  readonly exportSettings = signal({ ...MOCK_EXPORT_SETTINGS });

  // ── Static data ──
  readonly dataSources = MOCK_DATA_SOURCES;
  readonly systemInfo = MOCK_SYSTEM_INFO;
  readonly apiKeys = MOCK_API_KEYS;
  readonly logEntries = MOCK_LOG_ENTRIES;
  readonly mcpServers = MOCK_MCP_SERVERS;
  readonly colorSchemes = CHART_COLOR_SCHEMES;

  // ── Tabs ──
  readonly tabs: Tab[] = [
    { id: 'data', label: 'Data Management' },
    { id: 'preferences', label: 'Preferences' },
    { id: 'api', label: 'API & Integrations' },
    { id: 'system', label: 'System' },
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

  // ── Gauge thresholds ──
  readonly cpuThresholds: GaugeThreshold[] = [
    { value: 60, color: '#059669' },
    { value: 85, color: '#d97706' },
    { value: 100, color: '#dc2626' },
  ];

  readonly memThresholds: GaugeThreshold[] = [
    { value: 70, color: '#059669' },
    { value: 90, color: '#d97706' },
    { value: 100, color: '#dc2626' },
  ];

  // ── Filtered logs ──
  readonly filteredLogs = computed(() => {
    const filter = this.logLevelFilter();
    if (filter === 'all') return this.logEntries;
    const levels: Record<LogLevel, number> = { error: 0, warn: 1, info: 2, debug: 3 };
    const threshold = levels[filter];
    return this.logEntries.filter(l => levels[l.level] <= threshold);
  });

  readonly logFilterOptions: { value: LogLevel | 'all'; label: string }[] = [
    { value: 'all', label: 'All' },
    { value: 'error', label: 'Error' },
    { value: 'warn', label: 'Warn' },
    { value: 'info', label: 'Info' },
    { value: 'debug', label: 'Debug' },
  ];

  constructor() {
    this.loadData();
  }

  loadData(): void {
    this.isLoading.set(true);
    this.hasError.set(false);
    this.mockFetch
      .fetch({
        dataSources: MOCK_DATA_SOURCES,
        preferences: MOCK_USER_PREFERENCES,
        systemInfo: MOCK_SYSTEM_INFO,
        apiKeys: MOCK_API_KEYS,
        logEntries: MOCK_LOG_ENTRIES,
        exportSettings: MOCK_EXPORT_SETTINGS,
        mcpServers: MOCK_MCP_SERVERS,
      })
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

  // ── Methods ──
  onTabChange(tab: string): void {
    this.activeTab.set(tab as SettingsTab);
  }

  updatePreference<K extends keyof UserPreferences>(key: K, value: UserPreferences[K]): void {
    this.preferences.update(p => ({ ...p, [key]: value }));
    this.notifications.success(`Preference "${key}" updated`);
  }

  toggleKeyReveal(id: string): void {
    this.revealedKeys.update(set => {
      const next = new Set(set);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  isKeyRevealed(id: string): boolean {
    return this.revealedKeys().has(id);
  }

  clearCache(): void {
    this.cacheCleared.set(true);
    this.notifications.success('Cache cleared successfully');
    setTimeout(() => this.cacheCleared.set(false), 3000);
  }

  getMcpStatusClass(status: string): string {
    switch (status) {
      case 'online': return 'bg-gain';
      case 'degraded': return 'bg-warning';
      case 'offline': return 'bg-loss';
      default: return 'bg-text-tertiary';
    }
  }

  getLogLevelClass(level: string): string {
    switch (level) {
      case 'error': return 'text-loss';
      case 'warn': return 'text-warning';
      case 'info': return 'text-accent';
      case 'debug': return 'text-text-tertiary';
      default: return 'text-text-secondary';
    }
  }

  toggleExcelCharts(): void {
    this.exportSettings.update(s => ({ ...s, excelIncludeCharts: !s.excelIncludeCharts }));
  }

  updateExport<K extends keyof typeof MOCK_EXPORT_SETTINGS>(key: K, value: (typeof MOCK_EXPORT_SETTINGS)[K]): void {
    this.exportSettings.update(s => ({ ...s, [key]: value }));
  }

  getApiStatusClass(status: string): string {
    switch (status) {
      case 'valid': return 'bg-gain/15 text-gain';
      case 'expired': return 'bg-loss/15 text-loss';
      case 'missing': return 'bg-surface-inset text-text-tertiary';
      default: return '';
    }
  }
}
