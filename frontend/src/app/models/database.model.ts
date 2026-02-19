export interface HealthCheck {
  status: 'healthy' | 'unhealthy';
  latency_ms: number;
  database: string;
  version: string;
}

export interface TableInfo {
  name: string;
  schema: string;
  row_count: number;
  size_bytes: number;
  size_pretty: string;
}

export interface DatabaseStatus {
  health: HealthCheck;
  tables: TableInfo[];
  total_size_pretty: string;
}
