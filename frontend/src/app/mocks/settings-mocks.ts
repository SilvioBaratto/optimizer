import {
  DataSource,
  UserPreferences,
  SystemInfo,
  ApiKeyConfig,
  LogEntry,
  ExportSettings,
  McpServerStatus,
  ColorSchemeOption,
} from '../models/settings.model';

// ── Data Sources ──

export const MOCK_DATA_SOURCES: DataSource[] = [
  { id: 'ds1', name: 'Yahoo Finance', type: 'Market Data', status: 'connected', lastSync: '2026-02-25T06:00:00Z', recordCount: 1_847_293 },
  { id: 'ds2', name: 'Trading Economics', type: 'Macro Data', status: 'connected', lastSync: '2026-02-25T06:05:00Z', recordCount: 15_600 },
  { id: 'ds3', name: 'FRED', type: 'Economic Data', status: 'connected', lastSync: '2026-02-24T22:00:00Z', recordCount: 3_240 },
  { id: 'ds4', name: 'Trading 212', type: 'Brokerage', status: 'connected', lastSync: '2026-02-25T08:00:00Z', recordCount: 2_402 },
  { id: 'ds5', name: 'PostgreSQL', type: 'Database', status: 'connected', lastSync: '2026-02-25T09:00:00Z', recordCount: 4_688_995 },
];

// ── User Preferences ──

export const MOCK_USER_PREFERENCES: UserPreferences = {
  theme: 'dark',
  defaultBenchmark: 'SPY',
  currency: 'USD',
  dateFormat: 'YYYY-MM-DD',
  decimalPlaces: 2,
  notificationsEnabled: true,
  autoRefreshInterval: 60,
};

// ── System Info ──

export const MOCK_SYSTEM_INFO: SystemInfo = {
  version: '1.0.0',
  pythonVersion: '3.12.2',
  databaseVersion: 'PostgreSQL 16.2',
  databaseSize: '856 MB',
  uptime: '14d 8h 32m',
  lastMigration: '2026-02-20T10:00:00Z',
  cpuUsage: 0.23,
  memoryUsage: 0.58,
};

// ── API Key Configs ──

export const MOCK_API_KEYS: ApiKeyConfig[] = [
  { id: 'ak1', name: 'Yahoo Finance', service: 'yfinance', status: 'valid', maskedKey: 'yf_****_8x2q', lastValidated: '2026-02-25T06:00:00Z' },
  { id: 'ak2', name: 'Trading Economics', service: 'trading_economics', status: 'valid', maskedKey: 'te_****_m4kp', lastValidated: '2026-02-25T06:05:00Z', expiresAt: '2027-02-25T00:00:00Z' },
  { id: 'ak3', name: 'FRED', service: 'fred', status: 'valid', maskedKey: 'fr_****_9j3n', lastValidated: '2026-02-24T22:00:00Z' },
  { id: 'ak4', name: 'Trading 212', service: 'trading_212', status: 'valid', maskedKey: 't2_****_h7w1', lastValidated: '2026-02-25T08:00:00Z', expiresAt: '2026-08-25T00:00:00Z' },
];

// ── Log Entries ──

export const MOCK_LOG_ENTRIES: LogEntry[] = [
  { timestamp: '2026-02-25T09:45:12Z', level: 'info', source: 'scheduler', message: 'Daily pipeline started' },
  { timestamp: '2026-02-25T09:45:15Z', level: 'info', source: 'data_ingestion', message: 'Fetching price data for 847 instruments' },
  { timestamp: '2026-02-25T09:46:02Z', level: 'info', source: 'data_ingestion', message: 'Price data loaded: 847 instruments, 1.2M rows' },
  { timestamp: '2026-02-25T09:46:05Z', level: 'debug', source: 'data_ingestion', message: 'Cache hit ratio: 94.2%' },
  { timestamp: '2026-02-25T09:46:10Z', level: 'info', source: 'factor_engine', message: 'Computing 17 factors across universe' },
  { timestamp: '2026-02-25T09:47:30Z', level: 'warn', source: 'factor_engine', message: 'Factor IC below threshold for sentiment_change (0.032)' },
  { timestamp: '2026-02-25T09:47:45Z', level: 'info', source: 'factor_engine', message: '8/17 factors significant at 5% level' },
  { timestamp: '2026-02-25T09:48:00Z', level: 'info', source: 'risk_engine', message: 'Running VaR and stress test calculations' },
  { timestamp: '2026-02-25T09:48:22Z', level: 'warn', source: 'risk_engine', message: '1-Day VaR at 92% of limit ($245K / $265K)' },
  { timestamp: '2026-02-25T09:48:30Z', level: 'info', source: 'risk_engine', message: 'All 8 stress scenarios computed' },
  { timestamp: '2026-02-25T09:49:00Z', level: 'info', source: 'optimizer', message: 'Starting Max Sharpe optimization with BL views' },
  { timestamp: '2026-02-25T09:49:15Z', level: 'debug', source: 'optimizer', message: 'Prior: EmpiricalPrior with LedoitWolf covariance' },
  { timestamp: '2026-02-25T09:49:45Z', level: 'info', source: 'optimizer', message: 'Optimization converged in 23 iterations' },
  { timestamp: '2026-02-25T09:50:00Z', level: 'info', source: 'rebalancer', message: 'Drift check: 4.2% exceeds 2.5% threshold' },
  { timestamp: '2026-02-25T09:50:05Z', level: 'info', source: 'execution', message: 'Generating 10 rebalance orders' },
  { timestamp: '2026-02-25T09:50:10Z', level: 'error', source: 'execution', message: 'Rate limit hit on Trading 212 API, retrying in 5s' },
  { timestamp: '2026-02-25T09:50:16Z', level: 'info', source: 'execution', message: 'Retry successful, orders submitted' },
  { timestamp: '2026-02-25T09:51:00Z', level: 'info', source: 'monitoring', message: 'Post-trade compliance check passed' },
  { timestamp: '2026-02-25T09:51:05Z', level: 'debug', source: 'monitoring', message: 'Portfolio beta: 1.08, within 1.20 limit' },
  { timestamp: '2026-02-25T09:51:10Z', level: 'info', source: 'scheduler', message: 'Daily pipeline completed in 5m 58s' },
];

// ── Export Settings ──

export const MOCK_EXPORT_SETTINGS: ExportSettings = {
  pdfPageSize: 'A4',
  pdfOrientation: 'landscape',
  excelIncludeCharts: true,
  csvDelimiter: ',',
};

// ── MCP Server Statuses ──

export const MOCK_MCP_SERVERS: McpServerStatus[] = [
  { name: 'Portfolio Optimizer', endpoint: 'http://localhost:8000/mcp', status: 'online', latencyMs: 12 },
  { name: 'Factor Research', endpoint: 'http://localhost:8001/mcp', status: 'online', latencyMs: 28 },
  { name: 'Market Data Stream', endpoint: 'ws://localhost:8002/mcp', status: 'degraded', latencyMs: 145 },
];

// ── Chart Color Schemes ──

export const CHART_COLOR_SCHEMES: ColorSchemeOption[] = [
  { id: 'default', label: 'Default', colors: ['#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed', '#0891b2'] },
  { id: 'monochrome', label: 'Monochrome', colors: ['#18181b', '#3f3f46', '#52525b', '#71717a', '#a1a1aa', '#d4d4d8'] },
  { id: 'warm', label: 'Warm', colors: ['#dc2626', '#ea580c', '#d97706', '#ca8a04', '#65a30d', '#0d9488'] },
  { id: 'cool', label: 'Cool', colors: ['#1d4ed8', '#2563eb', '#0891b2', '#059669', '#4f46e5', '#7c3aed'] },
  { id: 'colorblind-safe', label: 'Colorblind Safe', colors: ['#0077bb', '#33bbee', '#009988', '#ee7733', '#cc3311', '#ee3377'] },
];
