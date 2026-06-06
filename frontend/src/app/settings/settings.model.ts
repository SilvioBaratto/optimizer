export type DataSourceStatus = 'connected' | 'disconnected' | 'error';

export interface DataSource {
  id: string;
  name: string;
  type: string;
  status: DataSourceStatus;
  lastSync: string;
  recordCount: number;
  errorMessage?: string;
}

export type ThemeMode = 'dark' | 'light' | 'system';

export interface UserPreferences {
  theme: ThemeMode;
  defaultBenchmark: string;
  currency: string;
  dateFormat: string;
  decimalPlaces: number;
  notificationsEnabled: boolean;
  autoRefreshInterval: number;
}

export interface SystemInfo {
  version: string;
  pythonVersion: string;
  databaseVersion: string;
  databaseSize: string;
  uptime: string;
  lastMigration: string;
  cpuUsage: number;
  memoryUsage: number;
}

export type ApiKeyStatus = 'valid' | 'expired' | 'missing';

export interface ApiKeyConfig {
  id: string;
  name: string;
  service: string;
  status: ApiKeyStatus;
  maskedKey: string;
  lastValidated: string;
  expiresAt?: string;
}

export type LogLevel = 'info' | 'warn' | 'error' | 'debug';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  source: string;
  message: string;
}

export interface ExportSettings {
  pdfPageSize: 'A4' | 'Letter';
  pdfOrientation: 'portrait' | 'landscape';
  excelIncludeCharts: boolean;
  csvDelimiter: ',' | ';' | '\t';
}

export type McpServerStatusType = 'online' | 'offline' | 'degraded';

export interface McpServerStatus {
  name: string;
  endpoint: string;
  status: McpServerStatusType;
  latencyMs: number;
}

export type ChartColorScheme = 'default' | 'monochrome' | 'warm' | 'cool' | 'colorblind-safe';

export interface ColorSchemeOption {
  id: ChartColorScheme;
  label: string;
  colors: string[];
}
