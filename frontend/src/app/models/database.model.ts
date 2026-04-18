/**
 * Mirrors the response from ``GET /api/v1/database/health``.
 *
 * Backend contract (api/app/api/v1/database.py:42-51):
 *   { healthy: bool, latency_ms: float, database_url: str }
 *
 * The ``database_url`` is password-masked server-side.
 */
export interface HealthCheck {
  healthy: boolean;
  latency_ms: number;
  database_url: string;
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

export interface TruncateResponse {
  table: string;
  status: string;
}
