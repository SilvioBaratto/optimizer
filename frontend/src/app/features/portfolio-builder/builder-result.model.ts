import type { EfficientFrontierPoint } from '../../models/optimization.model';

export type BuilderResultStatus =
  | 'idle'
  | 'running'
  | 'ok'
  | 'error'
  | 'stale';

export interface BuilderResult {
  readonly runId: string;
  readonly weights: Record<string, number>;
  readonly metrics: Record<string, unknown>;
  readonly riskContributions?: Record<string, number>;
  readonly optimizerType: string;
  readonly createdAt: string;
}

export interface BuilderFrontier {
  readonly points: EfficientFrontierPoint[];
}

export interface BuilderBacktest {
  readonly runId: string;
  readonly summaryStats: Record<string, number | null>;
  readonly equityCurve: Record<string, number | null>;
  readonly createdAt: string;
}
